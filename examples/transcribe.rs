//! Example: Transcribe audio files using a NeMo CTC ASR model.
//!
//! Usage:
//!   cargo run --example transcribe -- --model path/to/model.nemo audio1.wav audio2.wav
//!   cargo run --example transcribe -- --model path/to/model_dir/ audio1.wav
//!
//! Supported model formats:
//!   - .nemo files (NeMo checkpoint archives)
//!   - Directory containing model.safetensors, config.yaml, and tokenizer.model
//!   - Hugging Face Hub repository id (with the `hub` feature)
//!
//! Environment:
//!   RUST_LOG=mlx_voice=debug   Enable debug logging

use std::path::Path;
use std::time::Instant;

use mlx_voice::asr::EncDecCtcModel;

fn main() -> mlx_voice::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[1] != "--model" {
        eprintln!(
            "Usage: {} --model <model.nemo|model.safetensors|model_dir/> <audio1.wav> [audio2.wav ...]",
            args[0]
        );
        eprintln!();
        eprintln!("Supported formats:");
        eprintln!("  .nemo          - NeMo checkpoint archive");
        eprintln!("  .safetensors   - Loads its directory");
        eprintln!("  directory/     - Directory with model.safetensors and config.yaml");
        std::process::exit(1);
    }

    let model_path = &args[2];

    let audio_paths: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();

    if audio_paths.is_empty() {
        eprintln!("Error: No audio files provided");
        std::process::exit(1);
    }

    println!("Loading model from: {}", model_path);
    let start = Instant::now();

    // Load model based on file type
    let mut model = load_model(model_path)?;

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

/// Load a model from anything `from_pretrained` accepts (a `.nemo` file, a
/// model directory or a Hugging Face Hub repository id). A `.safetensors` file
/// loads from its directory.
fn load_model(model_path: &str) -> mlx_voice::Result<EncDecCtcModel> {
    let path = Path::new(model_path);
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    {
        EncDecCtcModel::from_pretrained(path.parent().unwrap_or(Path::new(".")))
    } else {
        EncDecCtcModel::from_pretrained(model_path)
    }
}
