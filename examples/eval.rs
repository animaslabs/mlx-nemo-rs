//! Example: Transcribe the shared eval set and write hypotheses for scoring.
//!
//! Runs every clip in `samples/eval/manifest.jsonl` through the streaming
//! pipeline (or, with `--offline`, a whole-file RNN-T model) and writes one JSON
//! line per clip with its text and timing. Scoring happens in mlx-models
//! (`samples/score.py`) so WER is normalized the same way for every port; the
//! cross-library benchmark there (`bench/bench.py`) drives this example too.
//!
//! Usage:
//!   cargo xtask download-eval
//!   cargo run --release --example eval -- --model path/to/model.nemo
//!   uv run samples/score.py /path/to/mlx-voice-rs/hypotheses.jsonl   # in mlx-models
//!
//! Options:
//!   --model <path>       RNN-T model (.nemo file or MLX directory)
//!   --offline            Transcribe whole files with EncDecRnntModel instead of streaming
//!   --dtype <dtype>      Offline encoder dtype: float32 (default), bfloat16 or float16
//!   --mode <idx>         Streaming mode index (default: 0)
//!   --ctc                Decode with a hybrid RNN-T/CTC model's CTC decoder
//!   --eval-dir <dir>     Eval set directory (default: samples/eval)
//!   --out <path>         Output JSONL (default: hypotheses.jsonl)
//!   --language <codes>   Only clips in these languages, comma-separated (e.g. en)
//!   --target-lang <lang> Target language of a multilingual model: a key such as en-US,
//!                        `auto` (default), or `clip` for each clip's manifest language
//!   --clip <name>        Only this clip (repeatable)
//!   --warmup <n>         Untimed runs on the first clip before timing (default: 1)
//!   --repeats <n>        Timed runs per clip; the median is reported (default: 1)

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_voice::asr::{DecoderType, EncDecRnntModel, StreamingOptions, TranscribeOptions};
use serde_json::{Value, json};
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut mode_idx: usize = 0;
    let mut eval_dir = PathBuf::from("samples/eval");
    let mut out_path = PathBuf::from("hypotheses.jsonl");
    let mut languages: Vec<String> = Vec::new();
    let mut clips: Vec<String> = Vec::new();
    let mut offline = false;
    let mut ctc = false;
    let mut warmup: usize = 1;
    let mut repeats: usize = 1;
    let mut dtype = String::from("float32");
    let mut target_lang: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        let value = || args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--offline" => {
                offline = true;
                i += 1;
                continue;
            }
            "--ctc" => {
                ctc = true;
                i += 1;
                continue;
            }
            "--warmup" => warmup = value().parse()?,
            "--dtype" => dtype = value(),
            "--repeats" => repeats = value().parse::<usize>()?.max(1),
            "--model" => model_path = Some(value()),
            "--mode" => mode_idx = value().parse()?,
            "--eval-dir" => eval_dir = value().into(),
            "--out" => out_path = value().into(),
            "--language" => languages.extend(value().split(',').map(str::to_string)),
            "--clip" => clips.push(value()),
            "--target-lang" => target_lang = Some(value()),
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 2;
    }
    let Some(model_path) = model_path else {
        eprintln!(
            "Usage: {} --model <model.nemo|dir> [--offline] [--ctc] [--dtype DTYPE] [--mode N] [--eval-dir DIR] \
             [--out FILE] [--language en,de] [--target-lang LANG|clip] [--clip NAME]... \
             [--warmup N] [--repeats N]",
            args[0]
        );
        std::process::exit(1);
    };

    let manifest_path = eval_dir.join("manifest.jsonl");
    let manifest: Vec<Value> = fs::read_to_string(&manifest_path)
        .map_err(|e| {
            format!(
                "{}: {e} (run `cargo xtask download-eval`)",
                manifest_path.display()
            )
        })?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let selected: Vec<&Value> = manifest
        .iter()
        .filter(|e| languages.is_empty() || languages.iter().any(|l| e["language"] == l.as_str()))
        .filter(|e| clips.is_empty() || clips.iter().any(|c| e["name"] == c.as_str()))
        .collect();

    println!(
        "Loading {} model from: {model_path}",
        if offline { "offline" } else { "streaming" }
    );
    let load_start = Instant::now();
    let decoder_type = if ctc {
        DecoderType::Ctc
    } else {
        DecoderType::Rnnt
    };
    let mut model = if offline {
        Model::Offline(
            EncDecRnntModel::from_pretrained(&model_path)?,
            TranscribeOptions::default().decoder_type(decoder_type),
        )
    } else {
        Model::Streaming(
            EncDecRnntModel::from_pretrained(&model_path)?,
            StreamingOptions::default()
                .mode(mode_idx)
                .decoder_type(decoder_type),
        )
    };
    let encoder_dtype = match dtype.as_str() {
        "float32" => None,
        "bfloat16" => Some(mlx_voice::mlx_rs::Dtype::Bfloat16),
        "float16" => Some(mlx_voice::mlx_rs::Dtype::Float16),
        other => return Err(format!("unsupported --dtype: {other}").into()),
    };
    if let Some(encoder_dtype) = encoder_dtype {
        match &mut model {
            Model::Offline(model, _) => model.set_encoder_dtype(encoder_dtype)?,
            Model::Streaming(..) => return Err("--dtype applies to --offline only".into()),
        }
    }
    let per_clip_lang = target_lang.as_deref() == Some("clip");
    if let Some(lang) = target_lang.as_deref().filter(|_| !per_clip_lang) {
        model.set_language(lang)?;
    }
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Loaded in {load_time:.2}s");

    if let Some(first) = selected.first() {
        let audio = eval_dir.join(first["audio"].as_str().unwrap_or_default());
        for _ in 0..warmup {
            model.transcribe(&audio)?;
        }
    }

    let model_name = Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());
    let mut out = BufWriter::new(File::create(&out_path)?);
    let (mut total_audio, mut total_elapsed) = (0.0, 0.0);

    for entry in &selected {
        let name = entry["name"].as_str().unwrap_or_default();
        let duration = entry["duration"].as_f64().unwrap_or_default();
        let audio = eval_dir.join(entry["audio"].as_str().unwrap_or_default());
        if per_clip_lang {
            model.set_language(entry["language"].as_str().unwrap_or_default())?;
        }

        let mut runs = Vec::with_capacity(repeats);
        let mut failure = None;
        for _ in 0..repeats {
            let start = Instant::now();
            match model.transcribe(&audio) {
                Ok((text, steps)) => runs.push((start.elapsed().as_secs_f64(), text, steps)),
                Err(e) => {
                    failure = Some(e.to_string());
                    break;
                }
            }
        }
        // Record the failure and move on, so one clip the model can't handle
        // doesn't throw away the rest of the run.
        if let Some(error) = failure {
            eprintln!("{name:30} failed: {error}");
            let row = json!({
                "name": name,
                "library": "mlx-voice-rs",
                "model": model_name,
                "streaming": !offline,
                "mode": mode_idx,
                "duration": duration,
                "error": error,
            });
            writeln!(out, "{row}")?;
            out.flush()?;
            continue;
        }
        runs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (elapsed, text, steps) = runs.swap_remove(runs.len() / 2);
        total_audio += duration;
        total_elapsed += elapsed;

        println!(
            "{name:30} {:>7.1}s audio  {elapsed:>6.1}s  RTF {:.3}",
            duration,
            elapsed / duration
        );
        let round = |x: f64, places: i32| (x * 10f64.powi(places)).round() / 10f64.powi(places);
        let mut row = json!({
            "name": name,
            "text": text.trim(),
            "library": "mlx-voice-rs",
            "model": model_name,
            "streaming": !offline,
            "mode": mode_idx,
            "device": "metal",
            "dtype": dtype,
            "duration": duration,
            "elapsed": round(elapsed, 3),
            "load_time": round(load_time, 2),
        });
        if ctc {
            row["decoder"] = json!("ctc");
        }
        if !steps.is_empty() {
            let mut ms: Vec<f64> = steps.iter().map(|s| s * 1000.0).collect();
            ms.sort_by(f64::total_cmp);
            let p95 = ms[((0.95 * ms.len() as f64).ceil() as usize).clamp(1, ms.len()) - 1];
            row["chunk_ms_mean"] = json!(round(ms.iter().sum::<f64>() / ms.len() as f64, 2));
            row["chunk_ms_p95"] = json!(round(p95, 2));
        }
        writeln!(out, "{row}")?;
        // Flush per clip so a long run that dies partway still leaves results.
        out.flush()?;
    }

    println!(
        "\n{} clips, {:.0}s audio in {:.0}s (RTF {:.3}). Wrote {}",
        selected.len(),
        total_audio,
        total_elapsed,
        total_elapsed / total_audio.max(1e-9),
        out_path.display()
    );
    Ok(())
}

enum Model {
    Offline(EncDecRnntModel, TranscribeOptions),
    Streaming(EncDecRnntModel, StreamingOptions),
}

impl Model {
    fn set_language(&mut self, language: &str) -> mlx_voice::Result<()> {
        match self {
            Model::Offline(model, _) => model.set_language(language),
            Model::Streaming(model, _) => model.set_language(language),
        }
    }

    /// Transcribe one file, returning the text and, when streaming, the wall
    /// time of each `transcribe_step` (seconds).
    fn transcribe(
        &mut self,
        audio: &Path,
    ) -> Result<(String, Vec<f64>), Box<dyn std::error::Error>> {
        match self {
            Model::Offline(model, options) => {
                let text = model.transcribe_with([audio], options)?.remove(0).text;
                Ok((text, Vec::new()))
            }
            Model::Streaming(model, options) => {
                let model_rate = model.sample_rate();
                let (samples, rate) = mlx_voice::audio::load_wav(audio)?;
                let samples = mlx_voice::audio::resample(&samples, rate, model_rate)?;
                let mut stream = model.stream(options.clone())?;
                stream.push_audio(&samples, model_rate)?;
                stream.end_audio()?;
                // Like `stream.finish()`, timing each step
                let mut steps = Vec::new();
                let mut start = Instant::now();
                while stream.next_update()?.is_some() {
                    let now = Instant::now();
                    steps.push((now - start).as_secs_f64());
                    start = now;
                }
                Ok((stream.text().to_string(), steps))
            }
        }
    }
}
