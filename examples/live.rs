//! Example: Live microphone transcription with a StreamingSession.
//!
//! This example captures audio from your microphone in real-time and transcribes
//! speech continuously, printing text as it is decoded. The session resamples
//! the microphone's rate to the model's.
//!
//! Usage:
//!   cargo run --release --example live -- --model path/to/streaming-model.nemo
//!   cargo run --release --example live -- --model path/to/mlx-model-directory
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use mlx_voice::asr::{EncDecRNNTModel, StreamingModeInfo, StreamingOptions};
use ringbuf::{
    HeapRb,
    traits::{Consumer, Producer, Split},
};
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> mlx_voice::Result<()> {
    // Initialize tracing subscriber
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

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
        eprintln!(
            "  --model <path>      Path to NeMo streaming model (.nemo file or MLX directory)"
        );
        eprintln!("  --mode <idx>        Streaming mode index (default: 0)");
        eprintln!("  --device <name>     Audio input device name (default: system default)");
        eprintln!("  --list-devices      List available audio input devices and exit");
        eprintln!("  --chunk-ms <ms>     Processing interval in milliseconds (default: 100)");
        std::process::exit(1);
    });

    println!("Loading model from: {}", model_path);
    let start = Instant::now();
    let mut model = EncDecRNNTModel::from_pretrained(model_path)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    let target_rate = model.sample_rate();
    println!("Sample rate: {} Hz", target_rate);

    // Show available streaming modes
    let modes = model.streaming_modes();
    if !modes.is_empty() {
        println!("\nAvailable streaming modes:");
        print_modes(&modes);
    }

    let mut session = model.stream(StreamingOptions::default().mode(mode_idx))?;

    // Print streaming config
    let config = session.streaming_config();
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

    // Feed the session in chunks of `chunk_ms` at the device rate
    let source_chunk_samples = (source_rate as f32 * chunk_ms as f32 / 1000.0) as usize;
    println!(
        "\nProcessing in {} ms chunks ({} samples at {} Hz)",
        chunk_ms, source_chunk_samples, source_rate
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
            audio_config,
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
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to build input stream: {}", e)))?;

    // Start audio stream
    stream
        .play()
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to start audio stream: {}", e)))?;

    // Buffer to accumulate samples from ring buffer
    let mut audio_buffer: Vec<f32> = Vec::with_capacity(source_chunk_samples * 2);

    println!("\nListening... Press Ctrl+C to stop.\n");

    let start_time = Instant::now();
    let mut chunks_fed = 0;
    let mut transcribe_steps = 0;
    let mut warned_behind = false;

    let debug = std::env::var("NEMO_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Main processing loop
    while running.load(Ordering::SeqCst) {
        // Drain available samples from ring buffer
        while let Some(sample) = consumer.try_pop() {
            audio_buffer.push(sample);
        }

        if audio_buffer.len() >= source_chunk_samples {
            let chunk: Vec<f32> = audio_buffer.drain(..source_chunk_samples).collect();
            if debug {
                let max_abs = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                eprintln!(
                    "CHUNK {} len={} max_abs={:.6}",
                    chunks_fed,
                    chunk.len(),
                    max_abs
                );
            }
            session.push_audio(&chunk, source_rate)?;
            chunks_fed += 1;
        }

        // Decode whatever the session has buffered
        while let Some(update) = session.next_update()? {
            if debug {
                let ids: Vec<usize> = update.tokens.iter().map(|t| t.id).collect();
                eprintln!("STEP {} new_tokens={:?}", transcribe_steps, ids);
            }
            transcribe_steps += 1;
            print!("{}", update.delta);
            io::stdout().flush().unwrap();
        }

        // Decoding slower than real time shows up as a growing backlog
        let behind = session.buffered_samples() as f32 / target_rate as f32;
        if behind > 2.0 && !warned_behind {
            eprintln!("\n(transcription is {behind:.1}s behind real time)");
            warned_behind = true;
        }

        // Small sleep to avoid busy-waiting
        thread::sleep(Duration::from_millis(10));
    }

    // Stop audio stream
    drop(stream);

    // Drain remaining samples from ring buffer that weren't popped in the main loop
    while let Some(sample) = consumer.try_pop() {
        audio_buffer.push(sample);
    }

    // Decode the remaining audio
    if !audio_buffer.is_empty() {
        session.push_audio(&audio_buffer, source_rate)?;
        chunks_fed += 1;
    }
    session.end_audio()?;
    while let Some(update) = session.next_update()? {
        transcribe_steps += 1;
        print!("{}", update.delta);
        io::stdout().flush().unwrap();
    }
    println!();

    let elapsed = start_time.elapsed().as_secs_f32();

    let transcript = session.transcript();
    println!("\n--- Final Result ---");
    if !transcript.text.is_empty() {
        println!("{}", transcript.text);
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
