//! Example: Live microphone transcription using a NeMo streaming ASR model.
//!
//! This example captures audio from your microphone in real-time and transcribes
//! speech continuously, displaying partial results as they become available.
//!
//! Usage:
//!   cargo run --example live -- --model path/to/streaming-model.nemo
//!
//! Options:
//!   --model <path>      Path to NeMo streaming model (.nemo file)
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

use mlx_voice::asr::{EncDecRNNTModel, StreamingModeInfo};

fn main() -> mlx_voice::Result<()> {
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
        eprintln!("Usage: {} --model <model.nemo> [options]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>      Path to NeMo streaming model (.nemo file)");
        eprintln!("  --mode <idx>        Streaming mode index (default: 0)");
        eprintln!("  --device <name>     Audio input device name (default: system default)");
        eprintln!("  --list-devices      List available audio input devices and exit");
        eprintln!("  --chunk-ms <ms>     Processing interval in milliseconds (default: 100)");
        std::process::exit(1);
    });

    // Load model
    println!("Loading model from: {}", model_path);
    let start = Instant::now();
    let mut model = EncDecRNNTModel::restore(model_path)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    let target_rate = model.sample_rate();
    println!("Sample rate: {} Hz", target_rate);

    // Check streaming support
    if !model.supports_streaming() {
        eprintln!("Error: Model does not support streaming inference.");
        eprintln!("Use a streaming-capable model like Nemotron.");
        std::process::exit(1);
    }

    // Show available streaming modes
    let modes = model.streaming_modes();
    if !modes.is_empty() {
        println!("\nAvailable streaming modes:");
        print_modes(&modes);
    }

    // Get input device
    let device = if let Some(name) = device_name {
        host.input_devices()
            .map_err(|e| mlx_voice::Error::audio(format!("Failed to enumerate devices: {}", e)))?
            .find(|d| get_device_name(d).contains(name))
            .ok_or_else(|| {
                mlx_voice::Error::audio(format!(
                    "Audio device '{}' not found. Use --list-devices to see available devices.",
                    name
                ))
            })?
    } else {
        host.default_input_device().ok_or_else(|| {
            mlx_voice::Error::audio(
                "No default audio input device available. Check microphone permissions.",
            )
        })?
    };

    let device_name_str = get_device_name(&device);
    println!("\nAudio device: {}", device_name_str);

    // Get audio configuration
    let (config, source_rate, channels) = get_input_config(&device, target_rate)?;
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

    // Initialize streaming session
    println!("\nInitializing streaming (mode {})", mode_idx);
    let mut session = model.init_streaming(mode_idx, 1)?;
    println!(
        "Chunk size: {} frames (first), {} frames (subsequent)",
        session.config.chunk_size_for_step(true),
        session.config.chunk_size_for_step(false)
    );
    println!(
        "Shift size: {} frames (first), {} frames (subsequent)",
        session.config.shift_size_for_step(true),
        session.config.shift_size_for_step(false)
    );
    println!(
        "Valid output length: {} frames, Lookahead: {} frames",
        session.config.valid_out_len, session.config.lookahead_frames
    );
    println!(
        "Cache sizes: channel={}, time={}",
        session.config.last_channel_cache_size, session.config.last_time_cache_size
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
    .map_err(|e| mlx_voice::Error::audio(format!("Failed to set Ctrl+C handler: {}", e)))?;

    // Build input stream
    let channels_clone = channels;
    let stream = device
        .build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert to mono if stereo (take first channel only)
                if channels_clone > 1 {
                    for chunk in data.chunks(channels_clone as usize) {
                        // Take first channel, drop if buffer is full
                        let _ = producer.try_push(chunk[0]);
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
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to build input stream: {}", e)))?;

    // Start audio stream
    stream
        .play()
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to start audio stream: {}", e)))?;

    // Setup resampler if needed
    let mut resampler = if needs_resampling {
        Some(create_resampler(source_rate, target_rate)?)
    } else {
        None
    };

    // Processing parameters
    let chunk_samples = (target_rate as usize * chunk_ms as usize) / 1000;
    let source_chunk_samples = (source_rate as usize * chunk_ms as usize) / 1000;

    println!("\nListening... Press Ctrl+C to stop.\n");

    let start_time = Instant::now();
    let mut last_text = String::new();
    let mut chunks_processed = 0;
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(source_chunk_samples * 2);

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
            let chunk: Vec<f32> = audio_buffer.drain(..required_samples).collect();
            let input_len = chunk.len();

            // Resample if needed
            let chunk = if let Some(ref mut rs) = resampler {
                resample_chunk(rs, &chunk)?
            } else {
                chunk
            };

            // Debug: show what we're feeding to the model
            let debug = std::env::var("NEMO_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false);
            if debug {
                eprintln!(
                    "[LIVE DEBUG] chunk {}: input {} samples -> resampled {} samples",
                    chunks_processed,
                    input_len,
                    chunk.len()
                );
            }

            // Process through model
            let output = model.process_chunk(&mut session, &chunk)?;
            chunks_processed += 1;

            // Display partial result if text changed
            if !output.text.is_empty() && output.text[0] != last_text {
                if debug {
                    eprintln!("[LIVE DEBUG] text changed: \"{}\"", output.text[0]);
                }
                print!("\r\x1b[K{}", output.text[0]);
                io::stdout().flush().unwrap();
                last_text = output.text[0].clone();
            }
        }

        // Small sleep to avoid busy-waiting
        thread::sleep(Duration::from_millis(10));
    }

    // Stop audio stream
    drop(stream);

    // Process any remaining audio in buffer
    if !audio_buffer.is_empty() {
        let chunk = if let Some(ref mut rs) = resampler {
            resample_chunk(rs, &audio_buffer)?
        } else {
            audio_buffer
        };

        if !chunk.is_empty() {
            let _ = model.process_chunk(&mut session, &chunk)?;
        }
    }

    // Finalize and get final result
    let final_output = model.finalize_streaming(session)?;

    let elapsed = start_time.elapsed().as_secs_f32();

    println!("\n\n--- Final Result ---");
    if !final_output.text.is_empty() && !final_output.text[0].is_empty() {
        println!("{}", final_output.text[0]);
    } else {
        println!("(no speech detected)");
    }

    println!("\n--- Statistics ---");
    println!("Duration: {:.2}s", elapsed);
    println!("Chunks processed: {}", chunks_processed);
    if chunks_processed > 0 {
        let rtf = elapsed / (chunks_processed as f32 * chunk_ms as f32 / 1000.0);
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
) -> mlx_voice::Result<(StreamConfig, u32, u16)> {
    let supported_configs = device
        .supported_input_configs()
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to get supported configs: {}", e)))?;

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
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to get default config: {}", e)))?;

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

/// Get the device's human-readable name.
fn get_device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

/// Create a resampler for converting from source to target sample rate.
fn create_resampler(from_rate: u32, to_rate: u32) -> mlx_voice::Result<rubato::Fft<f32>> {
    let chunk_size = 1024;
    let resampler = rubato::Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        1, // channels
        rubato::FixedSync::Input,
    )
    .map_err(|e| mlx_voice::Error::audio(format!("Failed to create resampler: {}", e)))?;

    Ok(resampler)
}

/// Resample a chunk of audio.
fn resample_chunk(resampler: &mut rubato::Fft<f32>, input: &[f32]) -> mlx_voice::Result<Vec<f32>> {
    use rubato::{audioadapter_buffers::direct::InterleavedSlice, Resampler};

    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Fft with FixedSync::Input requires specific input sizes, so we may need to pad
    let input_frames_needed = resampler.input_frames_next();

    let mut padded_input = input.to_vec();
    if padded_input.len() < input_frames_needed {
        padded_input.resize(input_frames_needed, 0.0);
    }

    // Process in chunks of input_frames_needed
    let mut output = Vec::new();
    for chunk in padded_input.chunks(input_frames_needed) {
        if chunk.len() < input_frames_needed {
            // Pad the last chunk
            let mut padded = chunk.to_vec();
            padded.resize(input_frames_needed, 0.0);
            let waves_in = InterleavedSlice::new(&padded, 1, padded.len())
                .map_err(|e| mlx_voice::Error::audio(format!("Invalid resampler input: {}", e)))?;
            match resampler.process(&waves_in, None) {
                Ok(waves_out) => output.extend(waves_out.take_data()),
                Err(e) => {
                    return Err(mlx_voice::Error::audio(format!("Resampling failed: {}", e)));
                }
            }
        } else {
            let waves_in = InterleavedSlice::new(chunk, 1, chunk.len())
                .map_err(|e| mlx_voice::Error::audio(format!("Invalid resampler input: {}", e)))?;
            match resampler.process(&waves_in, None) {
                Ok(waves_out) => output.extend(waves_out.take_data()),
                Err(e) => {
                    return Err(mlx_voice::Error::audio(format!("Resampling failed: {}", e)));
                }
            }
        }
    }

    Ok(output)
}
