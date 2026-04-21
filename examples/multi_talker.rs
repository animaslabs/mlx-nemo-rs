//! Example: Multi-talker streaming transcription with speaker diarization.
//!
//! This example combines streaming ASR with speaker diarization to produce
//! speaker-tagged transcription output.
//!
//! Usage:
//!   cargo run --example multi_talker -- audio.wav
//!
//! Options:
//!   --asr-model <path>    Path to streaming ASR .nemo model (default: auto-detect in models/)
//!   --diar-model <path>   Path to SortFormer diarization .nemo model (default: auto-detect in models/)
//!   --mode <idx>          Streaming mode index (default: 0)

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use mlx_nemo::asr::pipeline::MultiTalkerPipeline;
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> mlx_nemo::Result<()> {
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
    let mut mode_idx: usize = 0;

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
        None => find_model("models", "nemotron-speech-streaming")
            .unwrap_or_else(|| {
                eprintln!("Error: could not find ASR model in models/. Use --asr-model <path>");
                print_usage();
                std::process::exit(1);
            }),
    };

    let diar_model_path = match diar_model_path {
        Some(p) => p.to_string(),
        None => find_model("models", "diar_streaming_sortformer")
            .unwrap_or_else(|| {
                eprintln!("Error: could not find diarization model in models/. Use --diar-model <path>");
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
    let mut pipeline = MultiTalkerPipeline::restore(&asr_model_path, &diar_model_path, mode_idx)?;
    tracing::info!("Models loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Load audio
    tracing::info!("Loading audio from: {}", audio_path);
    pipeline.set_audio_file(audio_path)?;

    // Run transcription
    println!("\n--- Multi-Talker Transcription ---\n");

    let start = Instant::now();
    let segments = pipeline.run()?;
    let elapsed = start.elapsed();

    for segment in &segments {
        println!(
            "[Speaker {}] ({:.1}s - {:.1}s) {}",
            segment.speaker, segment.start_time, segment.end_time, segment.text
        );
    }

    println!("\n--- Done in {:.2}s ---", elapsed.as_secs_f32());

    io::stdout().flush().ok();

    Ok(())
}

fn print_usage() {
    eprintln!(
        "\nUsage: multi_talker [OPTIONS] <audio.wav>\n\
         \n\
         Options:\n\
         \x20 --asr-model <path>    Path to streaming ASR .nemo model\n\
         \x20 --diar-model <path>   Path to SortFormer diarization .nemo model\n\
         \x20 --mode <idx>          Streaming mode index (default: 0)\n\
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
