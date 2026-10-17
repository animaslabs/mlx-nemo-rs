//! Example: Multi-talker streaming transcription with speaker diarization.
//!
//! This example combines streaming ASR with speaker diarization to produce
//! speaker-tagged transcription output.
//!
//! Usage:
//!   cargo run --example multi_talker -- audio.wav
//!
//! Options:
//!   --asr-model <path>    Path to streaming ASR .nemo model (default: auto-detect in models/,
//!                         preferring multitalker-parakeet-streaming)
//!   --diar-model <path>   Path to SortFormer diarization .nemo model (default: auto-detect in models/)
//!   --mode <idx>          Streaming mode index (default: 0)
//!   --single-speaker      Transcribe chunks with at most one active speaker as speaker 0
//!   --output <path>       Write the segments as a SegLST JSON file

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use mlx_voice::asr::pipeline::{MultiTalkerConfig, MultiTalkerPipeline};
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> mlx_voice::Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut asr_model_path: Option<&str> = None;
    let mut diar_model_path: Option<&str> = None;
    let mut audio_path: Option<&str> = None;
    let mut output_path: Option<&str> = None;
    let mut mode_idx: usize = 0;
    let mut config = MultiTalkerConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--asr-model" => {
                i += 1;
                asr_model_path = Some(&args[i]);
            }
            "--diar-model" => {
                i += 1;
                diar_model_path = Some(&args[i]);
            }
            "--mode" => {
                i += 1;
                mode_idx = args[i].parse().expect("Invalid mode index");
            }
            "--single-speaker" => {
                config.single_speaker_mode = true;
            }
            "--output" => {
                i += 1;
                output_path = Some(&args[i]);
            }
            arg if !arg.starts_with('-') => {
                audio_path = Some(arg);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let asr_model_path = match asr_model_path {
        Some(p) => p.to_string(),
        None => find_model("models", "multitalker-parakeet-streaming")
            .or_else(|| find_model("models", "nemotron-speech-streaming"))
            .unwrap_or_else(|| {
                eprintln!("Error: could not find ASR model in models/. Use --asr-model <path>");
                print_usage();
                std::process::exit(1);
            }),
    };

    let diar_model_path = match diar_model_path {
        Some(p) => p.to_string(),
        None => find_model("models", "diar_streaming_sortformer").unwrap_or_else(|| {
            eprintln!(
                "Error: could not find diarization model in models/. Use --diar-model <path>"
            );
            print_usage();
            std::process::exit(1);
        }),
    };

    let audio_path = audio_path.unwrap_or_else(|| {
        eprintln!("Error: audio file path is required");
        print_usage();
        std::process::exit(1);
    });

    // Load pipeline
    tracing::info!("Loading ASR model from: {}", &asr_model_path);
    tracing::info!("Loading diarization model from: {}", &diar_model_path);

    let start = Instant::now();
    let mut pipeline = MultiTalkerPipeline::restore_with_config(
        &asr_model_path,
        &diar_model_path,
        mode_idx,
        config,
    )?;
    tracing::info!(
        "Models loaded in {:.2}s ({:?} mode)",
        start.elapsed().as_secs_f32(),
        pipeline.mode()
    );

    // Load audio
    tracing::info!("Loading audio from: {}", audio_path);
    pipeline.set_audio_file(audio_path)?;

    // Stream transcription step-by-step, printing words as they complete
    println!("\n--- Streaming words ---\n");

    let start = Instant::now();
    while let Some(output) = pipeline.transcribe_step()? {
        for word in &output.new_words {
            println!(
                "[{:7.2}s - {:7.2}s] speaker_{}: {}",
                word.start_time, word.end_time, word.speaker, word.text
            );
        }
        io::stdout().flush().ok();
        if output.is_final {
            break;
        }
    }
    let elapsed = start.elapsed();

    println!("\n--- Segments ---\n");
    let segments = pipeline.segments();
    for segment in &segments {
        println!(
            "[{:7.2}s - {:7.2}s] speaker_{}: {}",
            segment.start_time, segment.end_time, segment.speaker, segment.text
        );
    }

    println!("\n--- Done in {:.2}s ---", elapsed.as_secs_f32());

    if let Some(path) = output_path {
        let session_id = Path::new(audio_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session");
        let seglst: Vec<serde_json::Value> = segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "session_id": session_id,
                    "speaker": format!("speaker_{}", s.speaker),
                    "start_time": round_ms(s.start_time),
                    "end_time": round_ms(s.end_time),
                    "words": s.text,
                })
            })
            .collect();
        let json = serde_json::to_string_pretty(&seglst).expect("SegLST serializes");
        std::fs::write(path, json)?;
        println!("Wrote {} segments to {}", segments.len(), path);
    }

    io::stdout().flush().ok();

    Ok(())
}

/// Round seconds to milliseconds, avoiding f32 noise in the JSON output.
fn round_ms(seconds: f32) -> f64 {
    (seconds as f64 * 1000.0).round() / 1000.0
}

fn print_usage() {
    eprintln!(
        "\nUsage: multi_talker [OPTIONS] <audio.wav>\n\
         \n\
         Options:\n\
         \x20 --asr-model <path>    Path to streaming ASR .nemo model\n\
         \x20 --diar-model <path>   Path to SortFormer diarization .nemo model\n\
         \x20 --mode <idx>          Streaming mode index (default: 0)\n\
         \x20 --single-speaker      Treat chunks with one active speaker as speaker 0\n\
         \x20 --output <path>       Write segments as SegLST JSON\n\
         \n\
         Models are auto-detected from models/ if not specified."
    );
}

/// Search for a .nemo file under `base_dir` whose name contains `pattern`.
fn find_model(base_dir: &str, pattern: &str) -> Option<String> {
    find_nemo_file(Path::new(base_dir), pattern)
}

fn find_nemo_file(dir: &Path, pattern: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_nemo_file(&path, pattern) {
                return Some(found);
            }
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.contains(pattern) && name.ends_with(".nemo") {
                return Some(path.to_string_lossy().into_owned());
            }
        }
    }
    None
}
