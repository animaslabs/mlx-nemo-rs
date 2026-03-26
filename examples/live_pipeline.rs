//! Example: Live microphone transcription using CacheAwareRNNTPipeline.
//!
//! This example captures audio from your microphone in real-time and transcribes
//! speech continuously using the pipeline API, displaying partial results as they
//! become available.
//!
//! Usage:
//!   cargo run --example live_pipeline -- --model path/to/streaming-model.nemo
//!   cargo run --example live_pipeline -- --model path/to/mlx-model-directory
//!
//! Options:
//!   --model <path>      Path to NeMo streaming model (.nemo file or MLX directory)
//!   --mode <idx>        Streaming mode index (default: 0)
//!   --device <name>     Audio input device name (default: system default)
//!   --list-devices      List available audio input devices and exit
//!   --chunk-ms <ms>     Processing interval in milliseconds (default: 100)
//!
//! Requirements:
//!   - A streaming-capable NeMo model (e.g., nemotron-speech-streaming-en-0.6b)
//!   - Microphone access permissions
//!
//! Press Ctrl+C to stop and see the final transcription.

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use mlx_nemo::asr::{CacheAwareRNNTPipeline, StreamingModeInfo};

fn main() -> mlx_nemo::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_path: Option<&str> = None;
    let mut mode_idx: usize = 0;
    let mut chunk_ms: u32 = 100;
    let mut list_devices = false;
    let mut device_name: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(&args[i]);
                }
            }
            "--mode" => {
                i += 1;
                if i < args.len() {
                    mode_idx = args[i].parse().unwrap_or(0);
                }
            }
            "--chunk-ms" => {
                i += 1;
                if i < args.len() {
                    chunk_ms = args[i].parse().unwrap_or(100);
                }
            }
            "--device" => {
                i += 1;
                if i < args.len() {
                    device_name = Some(&args[i]);
                }
            }
            "--list-devices" => {
                list_devices = true;
            }
            _ => {}
        }
        i += 1;
    }

    // Initialize audio host
    let host = cpal::default_host();

    // List devices if requested
    if list_devices {
        println!("Available audio input devices:");
        println!("{}", "-".repeat(50));
        match host.input_devices() {
            Ok(devices) => {
                for device in devices {
                    let name = get_device_name(&device);
                    let default_config = device
                        .default_input_config()
                        .map(|c| format!("{} Hz, {} ch", c.sample_rate(), c.channels()))
                        .unwrap_or_else(|_| "N/A".to_string());
                    println!("  {} ({})", name, default_config);
                }
            }
            Err(e) => {
                eprintln!("Error listing devices: {}", e);
            }
        }
        return Ok(());
    }

    // Check for model path
    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: {} --model <model> [options]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>      Path to NeMo streaming model (.nemo file or MLX directory)");
        eprintln!("  --mode <idx>        Streaming mode index (default: 0)");
        eprintln!("  --device <name>     Audio input device name (default: system default)");
        eprintln!("  --list-devices      List available audio input devices and exit");
        eprintln!("  --chunk-ms <ms>     Processing interval in milliseconds (default: 100)");
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

    let target_rate = pipeline.sample_rate();
    println!("Sample rate: {} Hz", target_rate);

    // Show available streaming modes
    let modes = pipeline.streaming_modes();
    if !modes.is_empty() {
        println!("\nAvailable streaming modes:");
        print_modes(&modes);
    }

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

    // Get input device
    let device = if let Some(name) = device_name {
        host.input_devices()
            .map_err(|e| mlx_nemo::Error::audio(format!("Failed to enumerate devices: {}", e)))?
            .find(|d| get_device_name(d).contains(name))
            .ok_or_else(|| {
                mlx_nemo::Error::audio(format!(
                    "Audio device '{}' not found. Use --list-devices to see available devices.",
                    name
                ))
            })?
    } else {
        host.default_input_device().ok_or_else(|| {
            mlx_nemo::Error::audio(
                "No default audio input device available. Check microphone permissions.",
            )
        })?
    };

    let device_name_str = get_device_name(&device);
    println!("\nAudio device: {}", device_name_str);

    // Get audio configuration
    let (audio_config, source_rate, channels) = get_input_config(&device, target_rate)?;
    let needs_resampling = source_rate != target_rate;

    println!(
        "Input config: {} Hz, {} channel(s){}",
        source_rate,
        channels,
        if needs_resampling {
            format!(" (resampling to {} Hz)", target_rate)
        } else {
            String::new()
        }
    );

    // Calculate chunk sizes (matching streaming.rs approach)
    let chunk_samples = (target_rate as f32 * chunk_ms as f32 / 1000.0) as usize;
    let source_chunk_samples = (source_rate as f32 * chunk_ms as f32 / 1000.0) as usize;

    println!(
        "\nProcessing in {} ms chunks ({} samples at target rate)",
        chunk_ms, chunk_samples
    );

    // Create ring buffer (2 seconds of audio at source rate)
    let buffer_capacity = source_rate as usize * 2;
    let ring_buffer = HeapRb::<f32>::new(buffer_capacity);
    let (mut producer, mut consumer) = ring_buffer.split();

    // Setup Ctrl+C handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    ctrlc::set_handler(move || {
        println!("\n\nStopping...");
        running_clone.store(false, Ordering::SeqCst);
    })
    .map_err(|e| mlx_nemo::Error::audio(format!("Failed to set Ctrl+C handler: {}", e)))?;

    // Build input stream
    let channels_clone = channels;
    let stream = device
        .build_input_stream(
            &audio_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert to mono if stereo (average channels)
                if channels_clone > 1 {
                    for chunk in data.chunks(channels_clone as usize) {
                        let mono = chunk.iter().sum::<f32>() / channels_clone as f32;
                        let _ = producer.try_push(mono);
                    }
                } else {
                    // Mono: push all samples, drop if buffer is full
                    for &sample in data {
                        let _ = producer.try_push(sample);
                    }
                }
            },
            move |err| {
                eprintln!("Audio stream error: {}", err);
            },
            None,
        )
        .map_err(|e| mlx_nemo::Error::audio(format!("Failed to build input stream: {}", e)))?;

    // Start audio stream
    stream
        .play()
        .map_err(|e| mlx_nemo::Error::audio(format!("Failed to start audio stream: {}", e)))?;

    // Setup resampler if needed (using same approach as streaming.rs)
    let mut resampler: Option<SincFixedIn<f32>> = if needs_resampling {
        Some(create_resampler(source_rate, target_rate, source_chunk_samples)?)
    } else {
        None
    };

    // Buffer to accumulate samples from ring buffer
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(source_chunk_samples * 2);

    println!("\nListening... Press Ctrl+C to stop.\n");

    let start_time = Instant::now();
    let mut last_text = String::new();
    let mut chunks_fed = 0;
    let mut transcribe_steps = 0;

    let debug = std::env::var("NEMO_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Main processing loop
    while running.load(Ordering::SeqCst) {
        // Drain available samples from ring buffer
        while let Some(sample) = consumer.try_pop() {
            audio_buffer.push(sample);
        }

        // Process when we have enough samples
        let required_samples = if needs_resampling {
            source_chunk_samples
        } else {
            chunk_samples
        };

        if audio_buffer.len() >= required_samples {
            // Take exactly the required number of samples (matching streaming.rs)
            let chunk: Vec<f32> = audio_buffer.drain(..required_samples).collect();

            // Log chunk info (matching streaming.rs debug output)
            if debug {
                let preview: Vec<String> = chunk.iter().take(5).map(|s| format!("{:.6}", s)).collect();
                let max_abs = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                let time_offset = chunks_fed as f32 * chunk_ms as f32 / 1000.0;
                eprintln!(
                    "CHUNK {} @ {:.2}s required_samples={} len={} max_abs={:.6} samples=[{}{}]",
                    chunks_fed,
                    time_offset,
                    required_samples,
                    chunk.len(),
                    max_abs,
                    preview.join(", "),
                    if chunk.len() > 5 { ", ..." } else { "" }
                );
            }

            // Resample if needed
            let chunk = if let Some(ref mut rs) = resampler {
                tracing::info!("RESAMPLING");
                resample_chunk(rs, &chunk)?
            } else {
                chunk
            };

            // Feed audio to pipeline (matching streaming.rs)
            pipeline.add_audio(&chunk);
            chunks_fed += 1;
        }

        // Process any available transcription steps unconditionally
        // This ensures we process audio as it accumulates in the pipeline's internal buffer,
        // rather than only when we add new audio (matching streaming.rs behavior)
        while let Some(output) = pipeline.transcribe_step()? {
            if debug {
                let new_tokens_str: Vec<String> =
                    output.new_tokens.iter().map(|t| t.to_string()).collect();
                eprintln!(
                    "STEP {} new_tokens=[{}] total_tokens={}",
                    transcribe_steps,
                    new_tokens_str.join(", "),
                    output.token_count
                );
            }
            transcribe_steps += 1;

            // Display partial result if text changed
            if !output.text.is_empty() && output.text != last_text {
                print!("\r\x1b[K{}", output.text);
                io::stdout().flush().unwrap();
                last_text = output.text.clone();
            }
        }

        // Small sleep to avoid busy-waiting
        thread::sleep(Duration::from_millis(10));
    }

    // Stop audio stream
    drop(stream);

    // Process any remaining audio in buffer
    if !audio_buffer.is_empty() {
        if debug {
            eprintln!("Processing {} remaining samples in buffer", audio_buffer.len());
        }
        let chunk = if let Some(ref mut rs) = resampler {
            resample_chunk(rs, &audio_buffer)?
        } else {
            audio_buffer
        };

        if !chunk.is_empty() {
            pipeline.add_audio(&chunk);
            chunks_fed += 1;
        }
    }

    // Signal end of stream and process remaining steps (matching streaming.rs)
    pipeline.end_stream();

    while let Some(output) = pipeline.transcribe_step()? {
        if debug {
            let new_tokens_str: Vec<String> =
                output.new_tokens.iter().map(|t| t.to_string()).collect();
            eprintln!(
                "FINAL STEP {} new_tokens=[{}] total_tokens={}",
                transcribe_steps,
                new_tokens_str.join(", "),
                output.token_count
            );
        }
        transcribe_steps += 1;

        if !output.text.is_empty() && output.text != last_text {
            print!("\r\x1b[K{}", output.text);
            io::stdout().flush().unwrap();
            last_text = output.text.clone();
        }

        if output.is_final {
            println!(); // New line after final output
            break;
        }
    }

    let elapsed = start_time.elapsed().as_secs_f32();

    println!("\n--- Final Result ---");
    if !last_text.is_empty() {
        println!("{}", last_text);
    } else {
        println!("(no speech detected)");
    }

    println!("\n--- Statistics ---");
    println!("Duration: {:.2}s", elapsed);
    println!("Audio chunks fed: {}", chunks_fed);
    println!("Transcribe steps: {}", transcribe_steps);
    if chunks_fed > 0 {
        let audio_duration = chunks_fed as f32 * chunk_ms as f32 / 1000.0;
        let rtf = elapsed / audio_duration;
        println!("Real-time factor: {:.2}x", rtf);
    }

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

/// Get an appropriate input configuration for the device.
/// Returns (StreamConfig, source_sample_rate, num_channels).
fn get_input_config(
    device: &cpal::Device,
    target_rate: u32,
) -> mlx_nemo::Result<(StreamConfig, u32, u16)> {
    let supported_configs = device
        .supported_input_configs()
        .map_err(|e| mlx_nemo::Error::audio(format!("Failed to get supported configs: {}", e)))?;

    // Collect all supported configs
    let configs: Vec<_> = supported_configs.collect();

    // First, try to find a config that supports our target rate with mono
    for config_range in &configs {
        if config_range.channels() == 1
            && config_range.min_sample_rate() <= target_rate
            && config_range.max_sample_rate() >= target_rate
            && config_range.sample_format() == SampleFormat::F32
        {
            let config = config_range.clone().with_sample_rate(target_rate);
            return Ok((config.into(), target_rate, 1));
        }
    }

    // Try stereo at target rate
    for config_range in &configs {
        if config_range.channels() == 2
            && config_range.min_sample_rate() <= target_rate
            && config_range.max_sample_rate() >= target_rate
            && config_range.sample_format() == SampleFormat::F32
        {
            let config = config_range.clone().with_sample_rate(target_rate);
            return Ok((config.into(), target_rate, 2));
        }
    }

    // Fall back to default config (will need resampling)
    let default_config = device
        .default_input_config()
        .map_err(|e| mlx_nemo::Error::audio(format!("Failed to get default config: {}", e)))?;

    // For non-f32 formats, we need to handle conversion
    // For simplicity, try to find an f32 config at default rate
    let default_rate = default_config.sample_rate();
    let default_channels = default_config.channels();

    for config_range in &configs {
        if config_range.sample_format() == SampleFormat::F32
            && config_range.min_sample_rate() <= default_rate
            && config_range.max_sample_rate() >= default_rate
        {
            let config = config_range.clone().with_sample_rate(default_rate);
            return Ok((config.into(), default_rate, config_range.channels()));
        }
    }

    // Last resort: use default config as-is (may not work for all formats)
    Ok((default_config.config(), default_rate, default_channels))
}

/// Get device name, handling deprecated API gracefully.
fn get_device_name(device: &cpal::Device) -> String {
    #[allow(deprecated)]
    device.name().unwrap_or_else(|_| "<unknown>".to_string())
}

/// Create a resampler matching the streaming.rs approach.
fn create_resampler(
    from_rate: u32,
    to_rate: u32,
    chunk_size: usize,
) -> mlx_nemo::Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        chunk_size,
        1, // mono
    )
    .map_err(|e| mlx_nemo::Error::audio(format!("Failed to create resampler: {}", e)))?;

    Ok(resampler)
}

/// Resample a chunk of audio using the same approach as streaming.rs.
fn resample_chunk(
    resampler: &mut SincFixedIn<f32>,
    input: &[f32],
) -> mlx_nemo::Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let waves_in = vec![input.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| mlx_nemo::Error::audio(format!("Resampling failed: {}", e)))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}
