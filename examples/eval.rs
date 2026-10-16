//! Example: Transcribe the shared eval set and write hypotheses for scoring.
//!
//! Runs every clip in `samples/eval/manifest.jsonl` through the streaming
//! pipeline and writes one JSON line per clip. Scoring happens in mlx-models
//! (`samples/score.py`) so WER is normalized the same way for every port.
//!
//! Usage:
//!   cargo xtask download-eval
//!   cargo run --release --example eval -- --model path/to/model.nemo
//!   uv run samples/score.py /path/to/mlx-voice-rs/hypotheses.jsonl   # in mlx-models
//!
//! Options:
//!   --model <path>       Streaming RNN-T model (.nemo file or MLX directory)
//!   --mode <idx>         Streaming mode index (default: 0)
//!   --eval-dir <dir>     Eval set directory (default: samples/eval)
//!   --out <path>         Output JSONL (default: hypotheses.jsonl)
//!   --language <codes>   Only clips in these languages, comma-separated (e.g. en)
//!   --clip <name>        Only this clip (repeatable)

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_voice::asr::CacheAwareRNNTPipeline;
use serde_json::{json, Value};
use tracing_subscriber::{fmt, EnvFilter};

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

    let mut i = 1;
    while i < args.len() {
        let value = || args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--model" => model_path = Some(value()),
            "--mode" => mode_idx = value().parse()?,
            "--eval-dir" => eval_dir = value().into(),
            "--out" => out_path = value().into(),
            "--language" => languages.extend(value().split(',').map(str::to_string)),
            "--clip" => clips.push(value()),
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 2;
    }
    let Some(model_path) = model_path else {
        eprintln!(
            "Usage: {} --model <model.nemo|dir> [--mode N] [--eval-dir DIR] [--out FILE] \
             [--language en,de] [--clip NAME]...",
            args[0]
        );
        std::process::exit(1);
    };

    let manifest_path = eval_dir.join("manifest.jsonl");
    let manifest: Vec<Value> = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e} (run `cargo xtask download-eval`)", manifest_path.display()))?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let selected: Vec<&Value> = manifest
        .iter()
        .filter(|e| languages.is_empty() || languages.iter().any(|l| e["language"] == l.as_str()))
        .filter(|e| clips.is_empty() || clips.iter().any(|c| e["name"] == c.as_str()))
        .collect();

    println!("Loading pipeline from: {model_path}");
    let path = Path::new(&model_path);
    let mut pipeline = if path.is_dir() {
        CacheAwareRNNTPipeline::restore_from_directory(path, mode_idx)?
    } else {
        CacheAwareRNNTPipeline::restore(path, mode_idx)?
    };

    let model_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.clone());
    let mut out = BufWriter::new(File::create(&out_path)?);
    let (mut total_audio, mut total_elapsed) = (0.0, 0.0);

    for entry in &selected {
        let name = entry["name"].as_str().unwrap_or_default();
        let duration = entry["duration"].as_f64().unwrap_or_default();
        let audio = eval_dir.join(entry["audio"].as_str().unwrap_or_default());

        let start = Instant::now();
        pipeline.set_audio_file(&audio)?;
        let text = pipeline.run()?;
        let elapsed = start.elapsed().as_secs_f64();
        total_audio += duration;
        total_elapsed += elapsed;

        println!(
            "{name:30} {:>7.1}s audio  {elapsed:>6.1}s  RTF {:.3}",
            duration,
            elapsed / duration
        );
        let row = json!({
            "name": name,
            "text": text.trim(),
            "model": model_name,
            "mode": mode_idx,
            "elapsed": (elapsed * 100.0).round() / 100.0,
        });
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
