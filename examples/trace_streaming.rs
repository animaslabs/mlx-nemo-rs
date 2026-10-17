//! Trace streaming pipeline values for comparison with Python.
//!
//! Run with: RUST_LOG=info cargo run --example trace_streaming --release

use mlx_voice::asr::CacheAwareRNNTPipeline;
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> mlx_voice::Result<()> {
    // Initialize tracing
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let model_path =
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";
    let audio_path = "samples/2086-149220-0033.wav";
    let mode_idx = 2; // Same as Python: att_context_size = [70, 1]

    println!("Loading pipeline from: {}", model_path);
    let mut pipeline = CacheAwareRNNTPipeline::restore(model_path, mode_idx)?;

    println!("\n=== Model Info ===");
    println!("Sample rate: {} Hz", pipeline.sample_rate());
    println!("Blank ID: {}", pipeline.blank_id());

    // Print streaming config
    let config = pipeline.streaming_config();
    println!("\n=== Streaming Config ===");
    println!("chunk_size: {:?}", config.chunk_size);
    println!("shift_size: {:?}", config.shift_size);
    println!("valid_out_len: {}", config.valid_out_len);
    println!("pre_encode_cache_size: {:?}", config.pre_encode_cache_size);
    println!("drop_extra_pre_encoded: {}", config.drop_extra_pre_encoded);
    println!("cache_drop_size: {}", config.cache_drop_size);
    println!(
        "last_channel_cache_size: {}",
        config.last_channel_cache_size
    );
    println!("last_time_cache_size: {}", config.last_time_cache_size);

    // Load audio
    let (samples, sr) = load_audio(audio_path)?;
    println!("\n=== Audio ===");
    println!("Path: {}", audio_path);
    println!("Duration: {:.2}s", samples.len() as f32 / sr as f32);
    println!("Samples: {}", samples.len());

    // Set audio
    pipeline.set_audio_file(audio_path)?;

    // Process chunks
    println!("\n=== Processing Chunks ===");
    let mut chunk_idx = 0;
    let max_chunks = 30;

    while let Some(output) = pipeline.transcribe_step()? {
        println!("\n--- Chunk {} ---", chunk_idx);
        println!("New tokens: {:?}", output.new_tokens);
        println!(
            "All tokens ({} total): {:?}",
            output.tokens.len(),
            if output.tokens.len() > 20 {
                format!("{:?}...", &output.tokens[..20])
            } else {
                format!("{:?}", output.tokens)
            }
        );

        if !output.text.is_empty() {
            println!("Text: '{}'", output.text);
        }

        if output.is_final {
            println!("\n=== FINAL ===");
            println!("Final text: '{}'", output.text);
            break;
        }

        chunk_idx += 1;
        if chunk_idx >= max_chunks {
            println!("\n(stopping after {} chunks)", max_chunks);
            break;
        }
    }

    Ok(())
}

fn load_audio(path: &str) -> mlx_voice::Result<(Vec<f32>, u32)> {
    use hound::WavReader;

    let reader = WavReader::open(path)
        .map_err(|e| mlx_voice::Error::audio(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| mlx_voice::Error::audio(format!("Failed to read samples: {}", e)))?;

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
                .map_err(|e| mlx_voice::Error::audio(format!("Failed to read samples: {}", e)))?;

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
