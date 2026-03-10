//! Example: Transcribe audio files using a NeMo ASR model.
//!
//! Usage:
//!   cargo run --example transcribe -- --model path/to/model.nemo audio1.wav audio2.wav

use std::path::PathBuf;
use std::time::Instant;

use mlx_nemo::asr::EncDecCTCModel;

fn main() -> mlx_nemo::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[1] != "--model" {
        eprintln!(
            "Usage: {} --model <model.nemo> <audio1.wav> [audio2.wav ...]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = &args[2];
    let audio_paths: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();

    println!("Loading model from: {}", model_path);
    let start = Instant::now();

    let model = EncDecCTCModel::restore(model_path)?;

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("Sample rate: {} Hz", model.sample_rate());
    println!();

    // Transcribe each file
    println!("Transcribing {} file(s)...", audio_paths.len());
    let start = Instant::now();

    let transcriptions = model.transcribe(&audio_paths)?;

    let elapsed = start.elapsed().as_secs_f32();
    println!("Transcription completed in {:.2}s", elapsed);
    println!();

    // Print results
    for (path, text) in audio_paths.iter().zip(transcriptions.iter()) {
        println!("File: {}", path);
        println!("Text: {}", text);
        println!();
    }

    Ok(())
}
