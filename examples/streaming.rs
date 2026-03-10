//! Example: Streaming transcription of audio files using a NeMo ASR model.
//!
//! This example demonstrates real-time streaming inference, processing audio
//! in chunks and displaying partial transcriptions as they become available.
//!
//! Usage:
//!   cargo run --example streaming -- --model path/to/model.nemo audio.wav
//!
//! Options:
//!   --model <path>     Path to NeMo model (.nemo file)
//!   --mode <idx>       Streaming mode index (0 = lowest latency, default: 0)
//!   --chunk-ms <ms>    Audio chunk size in milliseconds (default: 100)
//!   --list-modes       List available streaming modes and exit

use std::io::{self, Write};
use std::time::Instant;

use mlx_nemo::asr::{EncDecRNNTModel, StreamingModeInfo};

fn main() -> mlx_nemo::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_path: Option<&str> = None;
    let mut audio_path: Option<&str> = None;
    let mut mode_idx: usize = 0;
    let mut chunk_ms: u32 = 100;
    let mut list_modes = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(&args[i]);
            }
            "--mode" => {
                i += 1;
                mode_idx = args[i].parse().unwrap_or(0);
            }
            "--chunk-ms" => {
                i += 1;
                chunk_ms = args[i].parse().unwrap_or(100);
            }
            "--list-modes" => {
                list_modes = true;
            }
            arg if !arg.starts_with('-') => {
                audio_path = Some(arg);
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!(
            "Usage: {} --model <model.nemo> [options] <audio.wav>",
            args[0]
        );
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>     Path to NeMo model (.nemo file)");
        eprintln!("  --mode <idx>       Streaming mode index (default: 0)");
        eprintln!("  --chunk-ms <ms>    Audio chunk size in milliseconds (default: 100)");
        eprintln!("  --list-modes       List available streaming modes and exit");
        std::process::exit(1);
    });

    println!("Loading model from: {}", model_path);
    let start = Instant::now();
    let mut model = EncDecRNNTModel::restore(model_path)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("Sample rate: {} Hz", model.sample_rate());

    // Check streaming support
    if !model.supports_streaming() {
        eprintln!("Error: Model does not support streaming inference.");
        eprintln!("Use a streaming-capable model like Nemotron.");
        std::process::exit(1);
    }

    // List available modes
    let modes = model.streaming_modes();
    if list_modes || modes.is_empty() {
        println!("\nAvailable streaming modes:");
        print_modes(&modes);
        if list_modes {
            return Ok(());
        }
    }

    let audio_path = audio_path.unwrap_or_else(|| {
        eprintln!("Error: No audio file specified.");
        std::process::exit(1);
    });

    // Load audio file
    println!("\nLoading audio: {}", audio_path);
    let (samples, file_sample_rate) = load_audio(audio_path)?;
    let audio_duration = samples.len() as f32 / file_sample_rate as f32;
    println!(
        "Audio duration: {:.2}s ({} samples at {} Hz)",
        audio_duration,
        samples.len(),
        file_sample_rate
    );

    // Resample if needed
    let target_rate = model.sample_rate();
    let samples = if file_sample_rate != target_rate {
        println!(
            "Resampling from {} Hz to {} Hz...",
            file_sample_rate, target_rate
        );
        resample_audio(&samples, file_sample_rate, target_rate)?
    } else {
        samples
    };

    // Initialize streaming session
    println!("\nInitializing streaming (mode {})", mode_idx);
    let mut session = model.init_streaming(mode_idx, 1)?;
    println!(
        "Chunk size: {} frames",
        session.config.chunk_size_for_step(true)
    );
    println!("Lookahead: {} frames", session.config.lookahead_frames);

    // Calculate chunk size in samples
    let chunk_samples = (target_rate as f32 * chunk_ms as f32 / 1000.0) as usize;
    println!(
        "Processing in {} ms chunks ({} samples)",
        chunk_ms, chunk_samples
    );

    // Process audio in chunks
    println!("\n--- Streaming Transcription ---");
    let start = Instant::now();
    let mut last_text = String::new();
    let mut chunks_processed = 0;

    let mut empty_outputs = 0;
    let mut total_samples_fed = 0usize;
    for chunk in samples.chunks(chunk_samples) {
        total_samples_fed += chunk.len();
        let output = model.process_chunk(&mut session, chunk)?;
        chunks_processed += 1;

        // Display partial result if text changed
        if !output.text.is_empty() && output.text[0] != last_text {
            // Clear line and print new text
            print!("\r\x1b[K"); // Clear line
            print!("{}", output.text[0]);
            io::stdout().flush().unwrap();
            last_text = output.text[0].clone();
        } else if output.text.is_empty() {
            empty_outputs += 1;
        }
    }

    eprintln!(
        "\nDebug: {} empty outputs, {} total chunks, {} samples fed (expected {})",
        empty_outputs,
        chunks_processed,
        total_samples_fed,
        samples.len()
    );

    // Finalize
    let final_output = model.finalize_streaming(session)?;
    println!(); // New line after streaming output

    let elapsed = start.elapsed().as_secs_f32();
    let rtf = elapsed / audio_duration;

    println!("\n--- Final Result ---");
    if !final_output.text.is_empty() {
        println!("Text: {}", final_output.text[0]);
    }

    println!("\n--- Statistics ---");
    println!("Audio duration: {:.2}s", audio_duration);
    println!("Processing time: {:.2}s", elapsed);
    println!("Real-time factor: {:.2}x", rtf);
    println!("Chunks processed: {}", chunks_processed);

    Ok(())
}

fn print_modes(modes: &[StreamingModeInfo]) {
    println!(
        "{:>5}  {:>12}  {:>12}  {:>12}",
        "Mode", "Left Ctx", "Right Ctx", "Latency"
    );
    println!("{}", "-".repeat(50));
    for mode in modes {
        let left = if mode.left_context < 0 {
            "unlimited".to_string()
        } else {
            mode.left_context.to_string()
        };
        println!(
            "{:>5}  {:>12}  {:>12}  {:>9.1} ms",
            mode.mode_idx, left, mode.right_context, mode.latency_ms
        );
    }
}

/// Load audio from a WAV file.
fn load_audio(path: &str) -> mlx_nemo::Result<(Vec<f32>, u32)> {
    use hound::WavReader;

    let reader = WavReader::open(path)
        .map_err(|e| mlx_nemo::Error::audio(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| mlx_nemo::Error::audio(format!("Failed to read samples: {}", e)))?;

            if channels > 1 {
                samples
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect()
            } else {
                samples
            }
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1 << (bits - 1)) as f32;

            let samples: Vec<i32> = reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| mlx_nemo::Error::audio(format!("Failed to read samples: {}", e)))?;

            if channels > 1 {
                samples
                    .chunks(channels)
                    .map(|chunk| {
                        chunk.iter().map(|&s| s as f32).sum::<f32>() / (channels as f32 * max_val)
                    })
                    .collect()
            } else {
                samples.iter().map(|&s| s as f32 / max_val).collect()
            }
        }
    };

    Ok((samples, sample_rate))
}

/// Resample audio to target sample rate.
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> mlx_nemo::Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| mlx_nemo::Error::audio(format!("Failed to create resampler: {}", e)))?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| mlx_nemo::Error::audio(format!("Resampling failed: {}", e)))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}
