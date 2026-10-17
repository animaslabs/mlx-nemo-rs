//! Example: Transcribe audio files using a NeMo RNNT ASR model.
//!
//! Usage:
//!   cargo run --example transcribe_rnnt -- --model path/to/model.nemo audio1.wav audio2.wav
//!   cargo run --example transcribe_rnnt -- --model path/to/model.safetensors audio1.wav
//!   cargo run --example transcribe_rnnt -- --model path/to/model_dir/ audio1.wav
//!   cargo run --example transcribe_rnnt -- --model path/to/model.safetensors --format mlx audio1.wav
//!
//! Supported model formats:
//!   - .nemo files (NeMo checkpoint archives)
//!   - .safetensors files (will look for config.yaml and tokenizer in same directory)
//!   - Directory containing model.safetensors, config.yaml, and tokenizer.model
//!
//! Options:
//!   --beam <size>     Use beam search decoding with given beam size (default: greedy)
//!   --format <fmt>    Weight format: "mlx" or "pytorch" (default: auto-detect)
//!   --language <lang> Target language of a multilingual model, e.g. en-US, de, auto (default)
//!   --keep-lang-tags  Keep the detected-language tags (e.g. <en-US>) in the text
//!   --ctc             Decode with a hybrid RNN-T/CTC model's CTC decoder
//!   --words           Print each word with its start and end time
//!
//! Environment:
//!   RUST_LOG=mlx_voice=debug   Enable debug logging

use std::path::Path;
use std::time::Instant;

use mlx_voice::WeightFormat;
use mlx_voice::asr::{DecoderType, EncDecRnntModel, TranscribeOptions};
use tracing_subscriber::EnvFilter;

fn main() -> mlx_voice::Result<()> {
    // Initialize tracing subscriber for debug output
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[1] != "--model" {
        eprintln!(
            "Usage: {} --model <model.nemo|model.safetensors|model_dir/> [options] <audio1.wav> [audio2.wav ...]",
            args[0]
        );
        eprintln!();
        eprintln!("Supported formats:");
        eprintln!("  .nemo          - NeMo checkpoint archive");
        eprintln!("  .safetensors   - Safetensors weights (config.yaml in same dir)");
        eprintln!("  directory/     - Directory with model.safetensors and config.yaml");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --beam <size>     Use beam search decoding (default: greedy)");
        eprintln!("  --format <fmt>    Weight format: mlx or pytorch (default: auto)");
        eprintln!("  --language <lang> Target language of a multilingual model (default: auto)");
        eprintln!("  --keep-lang-tags  Keep detected-language tags (e.g. <en-US>) in the text");
        eprintln!("  --ctc             Decode with a hybrid model's CTC decoder");
        std::process::exit(1);
    }

    let model_path = &args[2];

    // Parse optional flags
    let mut beam_size: Option<usize> = None;
    let mut weight_format: Option<WeightFormat> = None;
    let mut language: Option<String> = None;
    let mut keep_lang_tags = false;
    let mut ctc = false;
    let mut print_words = false;
    let mut audio_start_idx = 3;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--beam" => {
                if i + 1 < args.len() {
                    beam_size = Some(args[i + 1].parse().unwrap_or(4));
                    i += 2;
                    audio_start_idx = i;
                } else {
                    i += 1;
                }
            }
            "--format" => {
                if i + 1 < args.len() {
                    weight_format = match args[i + 1].to_lowercase().as_str() {
                        "mlx" => Some(WeightFormat::Mlx),
                        "pytorch" | "pt" => Some(WeightFormat::PyTorch),
                        _ => None,
                    };
                    i += 2;
                    audio_start_idx = i;
                } else {
                    i += 1;
                }
            }
            "--language" => {
                language = args.get(i + 1).cloned();
                i += 2;
                audio_start_idx = i;
            }
            "--keep-lang-tags" => {
                keep_lang_tags = true;
                i += 1;
                audio_start_idx = i;
            }
            "--words" => {
                print_words = true;
                i += 1;
                audio_start_idx = i;
            }
            "--ctc" => {
                ctc = true;
                i += 1;
                audio_start_idx = i;
            }
            _ => break,
        }
    }

    let audio_paths: Vec<&str> = args[audio_start_idx..].iter().map(|s| s.as_str()).collect();

    if audio_paths.is_empty() {
        eprintln!("Error: No audio files provided");
        std::process::exit(1);
    }

    println!("Loading RNNT model from: {}", model_path);
    let start = Instant::now();

    // Load model based on file type
    let mut model = load_model(model_path, weight_format)?;

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

    let mut options = TranscribeOptions::default().strip_lang_tags(!keep_lang_tags);
    if let Some(language) = language {
        options = options.language(language);
    }
    if let Some(language) = options.language.as_deref().or(model.language()) {
        println!("Target language: {language}");
    }
    if ctc {
        println!("Decoder: CTC");
        options = options.decoder_type(DecoderType::Ctc);
    }
    if let Some(beam) = beam_size {
        println!("Decoding strategy: Beam search (beam_size={})", beam);
        options = options.beam_search(beam);
    } else {
        println!("Decoding strategy: Greedy");
    }
    println!();

    // Transcribe each file
    println!("Transcribing {} file(s)...", audio_paths.len());
    let start = Instant::now();

    let transcriptions = model.transcribe_with(&audio_paths, &options)?;

    let elapsed = start.elapsed().as_secs_f32();
    println!("Transcription completed in {:.2}s", elapsed);
    println!();

    // Print results
    for (path, transcript) in audio_paths.iter().zip(transcriptions.iter()) {
        println!("File: {}", path);
        println!("Text: {}", transcript);
        if let Some(language) = &transcript.language {
            println!("Detected language: {language}");
        }
        if print_words {
            for word in &transcript.words {
                println!("  {:7.2} {:7.2}  {}", word.start, word.end, word.text);
            }
        }
        println!();
    }

    Ok(())
}

/// Load a model: a lone `.safetensors` file (with its config and tokenizer
/// found next to it), or anything `from_pretrained` accepts (a `.nemo` file, a
/// model directory or a Hugging Face Hub repository id).
fn load_model(
    model_path: &str,
    weight_format: Option<WeightFormat>,
) -> mlx_voice::Result<EncDecRnntModel> {
    let path = Path::new(model_path);
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    {
        // .safetensors file: look for config and tokenizer in same directory
        println!("  Format: Safetensors");
        let parent = path.parent().unwrap_or(Path::new("."));

        // Find config file
        let config_path = find_config_file(parent)?;
        println!("  Config: {}", config_path.display());

        // Find tokenizer (optional)
        let tokenizer_path = find_tokenizer_file(parent);
        if let Some(ref tok) = tokenizer_path {
            println!("  Tokenizer: {}", tok.display());
        }

        // Show weight format being used
        match weight_format {
            Some(WeightFormat::Mlx) => println!("  Weights: MLX format (no transposition)"),
            Some(WeightFormat::PyTorch) => {
                println!("  Weights: PyTorch format (will transpose Conv1d)")
            }
            Some(format) => println!("  Weights: {:?} format", format),
            None => println!("  Weights: Auto-detect from metadata"),
        }

        EncDecRnntModel::from_safetensors(
            path,
            &config_path,
            tokenizer_path.as_ref(),
            weight_format,
        )
    } else {
        if weight_format.is_some() {
            println!("  Warning: --format only applies to .safetensors files");
        }
        EncDecRnntModel::from_pretrained(model_path)
    }
}

/// Find config file in directory
fn find_config_file(dir: &Path) -> mlx_voice::Result<std::path::PathBuf> {
    let candidates = [
        "config.yaml",
        "model_config.yaml",
        "config.json",
        "model_config.json",
    ];

    for name in candidates {
        let path = dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(mlx_voice::Error::Checkpoint {
        message: format!("No config file found in {}", dir.display()),
        path: Some(dir.to_path_buf()),
    })
}

/// Find tokenizer file in directory (optional)
fn find_tokenizer_file(dir: &Path) -> Option<std::path::PathBuf> {
    let candidates = [
        "tokenizer.model",
        "tokenizer.spm",
        "sentencepiece.bpe.model",
    ];

    for name in candidates {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }

    // Search for any .model file
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".model") || name.ends_with(".spm") {
                    return Some(path);
                }
            }
        }
    }

    None
}
