//! Debug script to compare subsampling outputs between Rust and Python.
//!
//! Usage:
//!   cargo run --release --example debug_subsampling -- --model path/to/model.nemo audio.wav

use mlx_voice::asr::modules::{ConformerEncoder, MelSpectrogram, MelSpectrogramConfig};
use mlx_voice::asr::config::AsrModelConfig;
use mlx_voice::core::checkpoint::NemoCheckpoint;
use mlx_rs::Array;

fn main() -> mlx_voice::Result<()> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<&str> = None;
    let mut audio_path: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(&args[i]);
            }
            arg if !arg.starts_with('-') => {
                audio_path = Some(arg);
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.expect("Usage: --model <model.nemo> <audio.wav>");
    let audio_path = audio_path.expect("Usage: --model <model.nemo> <audio.wav>");

    println!("=== Loading Audio: {} ===", audio_path);
    let (samples, sr) = mlx_voice::audio::load_wav(audio_path)?;
    println!("Sample rate: {}", sr);
    println!("Duration: {:.2}s ({} samples)", samples.len() as f32 / sr as f32, samples.len());

    println!("\n=== Loading Model: {} ===", model_path);
    let checkpoint = NemoCheckpoint::open(model_path)?;
    let config: AsrModelConfig = checkpoint.load_config()?;

    // Create preprocessor
    let mut preproc_config: MelSpectrogramConfig = config.preprocessor.clone().into();
    preproc_config.dither = 0.0;  // Disable dither for inference
    let mut mel_proc = MelSpectrogram::new(preproc_config.clone())?;

    // Create encoder (has subsampling layer)
    let mut encoder = ConformerEncoder::new(config.encoder.clone())?;

    // Load encoder weights (.nemo files are not quantized)
    let weights = checkpoint.load_weights()?;
    encoder.load_weights(&weights, None)?;

    // Streaming parameters (matching Python debug script)
    let hop_samples = 160;  // 10ms at 16kHz
    let chunk_size = 112;   // MEL frames per encoder chunk
    let pre_encode_cache = 9;  // subsampling_factor + 1 = 8 + 1 = 9

    let first_chunk_audio_samples = (chunk_size + pre_encode_cache) * hop_samples;  // 121 * 160 = 19360
    let subsequent_chunk_audio_samples = chunk_size * hop_samples;  // 112 * 160 = 17920

    println!("\n=== Streaming Config ===");
    println!("  Chunk size: {} MEL frames", chunk_size);
    println!("  Pre-encode cache: {} frames", pre_encode_cache);
    println!("  First chunk audio: {} samples ({:.1}ms)", first_chunk_audio_samples, first_chunk_audio_samples as f32 / sr as f32 * 1000.0);
    println!("  Subsequent chunk audio: {} samples ({:.1}ms)", subsequent_chunk_audio_samples, subsequent_chunk_audio_samples as f32 / sr as f32 * 1000.0);

    println!("\n=== Processing Audio ===");

    let mut audio_offset = 0;
    let mut encoder_call = 0;

    while audio_offset < samples.len() && encoder_call < 8 {
        // Determine audio needed for this chunk
        let audio_needed = if encoder_call == 0 {
            first_chunk_audio_samples
        } else {
            subsequent_chunk_audio_samples
        };

        // Get audio for this chunk
        let chunk_end = (audio_offset + audio_needed).min(samples.len());
        let mut chunk: Vec<f32> = samples[audio_offset..chunk_end].to_vec();

        // Skip if too short
        if chunk.len() < hop_samples {
            break;
        }

        // Pad if needed for last chunk
        if chunk.len() < audio_needed {
            chunk.resize(audio_needed, 0.0);
        }

        // Compute MEL for this chunk
        let audio_array = mlx_rs::Array::from_slice(&chunk, &[1, chunk.len() as i32]);
        let (mel, _) = mel_proc.forward(&audio_array, None)?;
        mel.eval()?;

        // mel is [batch=1, n_mels, frames]
        let n_mels = mel.dim(1) as usize;
        let num_mel_frames = mel.dim(2) as usize;

        // Remove batch dim and transpose: [1, n_mels, T] -> [n_mels, T] -> [T, n_mels]
        use mlx_rs::ops::indexing::IndexOp;
        let mel_2d = mel.index(0);
        let mel_transposed = mlx_rs::ops::transpose_axes(&mel_2d, &[1, 0])?;
        mel_transposed.eval()?;

        // Add batch dimension: [T, n_mels] -> [1, T, n_mels]
        let mel_batch = mel_transposed.reshape(&[1, num_mel_frames as i32, n_mels as i32])?;
        mel_batch.eval()?;

        let mel_len = Array::from_slice(&[num_mel_frames as i32], &[1]);

        // Run through encoder's subsampling layer only
        // We use forward_batch which just does subsampling + conformer layers
        // But we want ONLY subsampling...
        // Let me access the subsampling directly
        let (subsampled, sub_len) = encoder.forward_subsampling_only(&mel_batch, &mel_len)?;
        subsampled.eval()?;
        sub_len.eval()?;

        // Print results
        let out_flat = subsampled.reshape(&[-1])?;
        out_flat.eval()?;
        let out_slice: &[f32] = out_flat.as_slice();
        let total_elements = out_slice.len();

        // First 5 values
        let n = 5.min(total_elements);
        let first_samples: Vec<String> = out_slice[..n].iter().map(|x| format!("{:.6}", x)).collect();

        // Middle 5 values
        let mid_start = total_elements / 2;
        let mid_end = (mid_start + 5).min(total_elements);
        let mid_samples: Vec<String> = out_slice[mid_start..mid_end].iter().map(|x| format!("{:.6}", x)).collect();

        // Last 5 values
        let last_start = total_elements.saturating_sub(5);
        let last_samples: Vec<String> = out_slice[last_start..].iter().map(|x| format!("{:.6}", x)).collect();

        // Statistics
        let out_min = out_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = out_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let out_mean: f32 = out_slice.iter().sum::<f32>() / total_elements as f32;
        let out_std: f32 = (out_slice.iter().map(|&v| (v - out_mean).powi(2)).sum::<f32>() / total_elements as f32).sqrt();

        println!("\nEncoder call {}:", encoder_call);
        println!("  Audio samples: {} ({:.1}ms)", chunk.len(), chunk.len() as f32 / sr as f32 * 1000.0);
        println!("  MEL frames: {}", num_mel_frames);
        println!("  MEL input shape: {:?}", mel_batch.shape());
        println!("  Subsampling output shape: {:?} (total: {})", subsampled.shape(), total_elements);
        println!("  first 5: [{}]", first_samples.join(", "));
        println!("  mid 5 (@{}): [{}]", mid_start, mid_samples.join(", "));
        println!("  last 5 (@{}): [{}]", last_start, last_samples.join(", "));
        println!("  stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}", out_min, out_max, out_mean, out_std);

        // For encoder call 0, print detailed comparison values
        if encoder_call == 0 {
            // Output is [1, T, D] where T=17, D=1024
            let t_dim = subsampled.dim(1) as usize;
            let d_dim = subsampled.dim(2) as usize;

            // Print specific time steps
            println!("\n  Detailed output (for comparison):");

            // Time step 0
            println!("  Time step 0:");
            let t0_samples: Vec<String> = out_slice[..5].iter().map(|x| format!("{:.6}", x)).collect();
            let t0_mid: Vec<String> = out_slice[512..517].iter().map(|x| format!("{:.6}", x)).collect();
            let t0_end: Vec<String> = out_slice[1019..1024].iter().map(|x| format!("{:.6}", x)).collect();
            println!("    features 0-4: [{}]", t0_samples.join(", "));
            println!("    features 512-516: [{}]", t0_mid.join(", "));
            println!("    features 1019-1023: [{}]", t0_end.join(", "));

            // Middle time step
            let mid_t = t_dim / 2;
            let mid_offset = mid_t * d_dim;
            println!("  Time step {}:", mid_t);
            let tm_samples: Vec<String> = out_slice[mid_offset..mid_offset+5].iter().map(|x| format!("{:.6}", x)).collect();
            let tm_mid: Vec<String> = out_slice[mid_offset+512..mid_offset+517].iter().map(|x| format!("{:.6}", x)).collect();
            let tm_end: Vec<String> = out_slice[mid_offset+1019..mid_offset+1024].iter().map(|x| format!("{:.6}", x)).collect();
            println!("    features 0-4: [{}]", tm_samples.join(", "));
            println!("    features 512-516: [{}]", tm_mid.join(", "));
            println!("    features 1019-1023: [{}]", tm_end.join(", "));

            // Last time step
            let last_t = t_dim - 1;
            let last_offset = last_t * d_dim;
            println!("  Time step {}:", last_t);
            let tl_samples: Vec<String> = out_slice[last_offset..last_offset+5].iter().map(|x| format!("{:.6}", x)).collect();
            let tl_mid: Vec<String> = out_slice[last_offset+512..last_offset+517].iter().map(|x| format!("{:.6}", x)).collect();
            let tl_end: Vec<String> = out_slice[last_offset+1019..last_offset+1024].iter().map(|x| format!("{:.6}", x)).collect();
            println!("    features 0-4: [{}]", tl_samples.join(", "));
            println!("    features 512-516: [{}]", tl_mid.join(", "));
            println!("    features 1019-1023: [{}]", tl_end.join(", "));

            // First 50 values for detailed comparison
            println!("\n  === All first 50 values (flattened) ===");
            for i in 0..50.min(out_slice.len()) {
                println!("    [{}] = {:.6}", i, out_slice[i]);
            }

            // Print values at each time step (matching Python format)
            // Valid time steps are 0 to t_dim-2 (last one is padding in NeMo)
            let valid_t = t_dim - 1;  // NeMo considers last step as padding
            println!("\n  === Valid output (time steps 0 to {}) ===", valid_t - 1);
            for t in 0..valid_t {
                let offset = t * d_dim;
                let vals: Vec<String> = out_slice[offset..offset+5].iter().map(|x| format!("{:.6}", x)).collect();
                println!("  T{:02} [0:5]: [{}]", t, vals.join(", "));
            }
        }

        audio_offset += audio_needed;
        encoder_call += 1;
    }

    println!("\n=== Done: {} encoder calls ===", encoder_call);

    Ok(())
}
