//! Example: Streaming transcription using CacheAwareRNNTPipeline.
//!
//! This example demonstrates real-time streaming inference by:
//! 1. Loading audio from a file
//! 2. Chunking it into small pieces (simulating live audio)
//! 3. Feeding chunks incrementally to the pipeline
//! 4. Displaying partial transcriptions as they become available
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
use std::path::Path;
use std::time::Instant;

use mlx_nemo::asr::{CacheAwareRNNTPipeline, StreamingModeInfo};
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> mlx_nemo::Result<()> {
    // Initialize tracing subscriber
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

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

    // Load pipeline (detect directory vs .nemo file)
    println!("Loading pipeline from: {}", model_path);
    let start = Instant::now();
    let model_path_ref = Path::new(model_path);
    let mut pipeline = if model_path_ref.is_dir() {
        CacheAwareRNNTPipeline::restore_from_directory(model_path_ref, mode_idx)?
    } else {
        CacheAwareRNNTPipeline::restore(model_path_ref, mode_idx)?
    };
    println!("Pipeline loaded in {:.2}s", start.elapsed().as_secs_f32());

    let sample_rate = pipeline.sample_rate();
    println!("Sample rate: {} Hz", sample_rate);

    // List available modes
    let modes = pipeline.streaming_modes();
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

    // Print streaming config
    let config = pipeline.streaming_config();
    println!("\nStreaming configuration (mode {}):", mode_idx);
    println!(
        "  Chunk size: {} frames (first), {} frames (subsequent)",
        config.chunk_size.0, config.chunk_size.1
    );
    println!(
        "  Shift size: {} frames (first), {} frames (subsequent)",
        config.shift_size.0, config.shift_size.1
    );
    println!("  Valid output length: {} frames", config.valid_out_len);

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
    let samples = if file_sample_rate != sample_rate {
        println!(
            "Resampling from {} Hz to {} Hz...",
            file_sample_rate, sample_rate
        );
        resample_audio(&samples, file_sample_rate, sample_rate)?
    } else {
        samples
    };

    // Calculate chunk size in samples
    // Each chunk includes the lookahead and current 80ms frames (8x downsampled mel frames)
    // mode 0: 1120ms chunk (112 mel frames), 1040ms lookahead
    let chunk_samples = (sample_rate as f32 * chunk_ms as f32 / 1000.0) as usize;
    // let chunk_samples = config.shift_size.0 * 160;
    println!(
        "\nProcessing in {} ms chunks ({} samples)",
        chunk_ms, chunk_samples
    );

    // Process audio in chunks (simulating live streaming)
    println!("\n--- Streaming Transcription ---");
    let start = Instant::now();
    let mut last_text = String::new();
    let mut chunks_fed = 0;
    let mut transcribe_steps = 0;

    for (chunk_idx, chunk) in samples.chunks(chunk_samples).enumerate() {
        // Log chunk info with first few samples
        let preview: Vec<String> = chunk.iter().take(5).map(|s| format!("{:.6}", s)).collect();
        let max_abs = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let time_offset = (chunk_idx * chunk_samples) as f32 / sample_rate as f32;
        eprintln!(
            "CHUNK {} @ {:.2}s len={} max_abs={:.6} samples=[{}{}]",
            chunk_idx,
            time_offset,
            chunk.len(),
            max_abs,
            preview.join(", "),
            if chunk.len() > 5 { ", ..." } else { "" }
        );

        // Feed audio chunk to pipeline
        pipeline.add_audio(chunk);
        chunks_fed += 1;

        // Check if this is the last chunk
        let is_last = (chunk_idx + 1) * chunk_samples >= samples.len();
        if is_last {
            pipeline.end_stream();
        }

        // Process any available transcription steps
        while let Some(output) = pipeline.transcribe_step()? {
            // Display step info for debugging
            let new_tokens_str: Vec<String> = output.new_tokens.iter().map(|t| t.to_string()).collect();
            eprintln!(
                "STEP {} new_tokens=[{}] total_tokens={}",
                transcribe_steps,
                new_tokens_str.join(", "),
                output.token_count
            );
            transcribe_steps += 1;

            // Display partial result if text changed
            if !output.text.is_empty() && output.text != last_text {
                print!("\r\x1b[K{}\n", output.text); // Clear line and print new text
                io::stdout().flush().unwrap();
                last_text = output.text.clone();
            }

            if output.is_final {
                println!(); // New line after final output
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let rtf = elapsed / audio_duration;

    println!("\n--- Final Result ---");
    println!("Text: {}", last_text);

    println!("\n--- Statistics ---");
    println!("Audio duration: {:.2}s", audio_duration);
    println!("Processing time: {:.2}s", elapsed);
    println!("Real-time factor: {:.2}x", rtf);
    println!("Audio chunks fed: {}", chunks_fed);
    println!("Transcribe steps: {}", transcribe_steps);

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
