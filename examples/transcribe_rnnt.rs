//! Example: Transcribe audio files using a NeMo RNNT ASR model.
//!
//! Usage:
//!   cargo run --example transcribe_rnnt -- --model path/to/model.nemo audio1.wav audio2.wav
//!
//! Options:
//!   --beam <size>   Use beam search decoding with given beam size (default: greedy)

use std::time::Instant;

use mlx_nemo::asr::{EncDecRNNTModel, RnntDecodingStrategy};

fn main() -> mlx_nemo::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[1] != "--model" {
        eprintln!(
            "Usage: {} --model <model.nemo> [--beam <size>] <audio1.wav> [audio2.wav ...]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_path = &args[2];

    // Check for --beam option
    let (beam_size, audio_start_idx) = if args.len() > 4 && args[3] == "--beam" {
        let beam = args[4].parse::<usize>().unwrap_or(4);
        (Some(beam), 5)
    } else {
        (None, 3)
    };

    let audio_paths: Vec<&str> = args[audio_start_idx..].iter().map(|s| s.as_str()).collect();

    if audio_paths.is_empty() {
        eprintln!("Error: No audio files provided");
        std::process::exit(1);
    }

    println!("Loading RNNT model from: {}", model_path);
    let start = Instant::now();

    let mut model = EncDecRNNTModel::restore(model_path)?;

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("Sample rate: {} Hz", model.sample_rate());
    println!("Blank ID: {}", model.blank_id());
    println!(
        "Decoder type: {}",
        if model.is_lstm_decoder() {
            "LSTM-based"
        } else {
            "Stateless"
        }
    );

    // Set decoding strategy
    if let Some(beam) = beam_size {
        println!("Decoding strategy: Beam search (beam_size={})", beam);
        model.set_decoding_strategy(RnntDecodingStrategy::BeamSearch { beam_size: beam });
    } else {
        println!("Decoding strategy: Greedy");
    }
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
