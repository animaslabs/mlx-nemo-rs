//! Debug script to compare encoder outputs between Rust and Python.
//!
//! Usage:
//!   cargo run --release --example debug_encoder -- --model path/to/model.nemo audio.wav

use mlx_rs::Array;
use mlx_voice::asr::config::AsrModelConfig;
use mlx_voice::asr::modules::{ConformerEncoder, MelSpectrogram, MelSpectrogramConfig};
use mlx_voice::core::checkpoint::NemoCheckpoint;

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
    println!("Audio: {} samples at {} Hz", samples.len(), sr);

    println!("\n=== Loading Model: {} ===", model_path);
    let checkpoint = NemoCheckpoint::open(model_path)?;
    let config: AsrModelConfig = checkpoint.load_config()?;

    // Create preprocessor
    let mut preproc_config: MelSpectrogramConfig = config.preprocessor.clone().into();
    preproc_config.dither = 0.0;
    let mel_proc = MelSpectrogram::new(preproc_config.clone())?;

    // Create encoder
    let mut encoder = ConformerEncoder::new(config.encoder.clone())?;

    // Load weights (.nemo files are not quantized)
    let weights = checkpoint.load_weights()?;
    encoder.load_weights(&weights, None)?;

    println!("\nEncoder config:");
    println!("  d_model: {}", encoder.d_model());
    println!("  n_layers: {}", encoder.num_layers());
    println!("  subsampling_factor: {}", encoder.subsampling_factor());

    // Streaming parameters
    let hop_samples = 160;
    let chunk_size = 112;
    let pre_encode_cache = 9;
    let first_chunk_audio = (chunk_size + pre_encode_cache) * hop_samples; // 19360
    let subsequent_chunk_audio = chunk_size * hop_samples; // 17920

    println!("\n=== Processing Audio ===");

    let mut audio_offset = 0;
    let mut encoder_call = 0;

    while audio_offset < samples.len() && encoder_call < 3 {
        // Determine audio needed
        let audio_needed = if encoder_call == 0 {
            first_chunk_audio
        } else {
            subsequent_chunk_audio
        };

        // Get audio chunk
        let chunk_end = (audio_offset + audio_needed).min(samples.len());
        let mut chunk: Vec<f32> = samples[audio_offset..chunk_end].to_vec();

        if chunk.len() < hop_samples {
            break;
        }

        // Pad if needed
        if chunk.len() < audio_needed {
            chunk.resize(audio_needed, 0.0);
        }

        // Compute MEL
        let audio_array = Array::from_slice(&chunk, &[1, chunk.len() as i32]);
        let (mel, _) = mel_proc.forward(&audio_array, None)?;
        mel.eval()?;

        // mel is [batch=1, n_mels, frames]
        let n_mels = mel.dim(1) as usize;
        let num_mel_frames = mel.dim(2) as usize;

        // Already in correct shape [1, n_mels, T] for encoder
        let mel_batch = mel;
        mel_batch.eval()?;

        // Use NeMo's mel_len formula: ceil(audio_len / hop) = audio_len / hop (for exact multiples)
        // For 19360 samples with hop=160: mel_len = ceil(19360/160) = 121
        let valid_mel_len = (chunk.len() as f32 / 160.0).ceil() as i32;
        let mel_len = Array::from_slice(&[valid_mel_len], &[1]);

        println!(
            "  Using mel_len: {} (actual frames: {})",
            valid_mel_len, num_mel_frames
        );

        // For first encoder call, use debug forward to get intermediate values
        if encoder_call == 0 {
            let debug_out = encoder.forward_debug(&mel_batch, &mel_len)?;

            // Print intermediate values
            println!("\n  === Debug Output ===");

            // After subsampling
            let sub_flat = debug_out.after_subsampling.reshape(&[-1])?;
            sub_flat.eval()?;
            let sub_slice: &[f32] = sub_flat.as_slice();
            let d_model = debug_out.after_subsampling.dim(2) as usize;
            let t_dim = debug_out.after_subsampling.dim(1) as usize;
            println!(
                "  After subsampling: shape={:?}",
                debug_out.after_subsampling.shape()
            );
            let t0_sub: Vec<String> = sub_slice[..5].iter().map(|x| format!("{:.6}", x)).collect();
            let t1_sub: Vec<String> = sub_slice[d_model..d_model + 5]
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            println!("    T00[0:5]: [{}]", t0_sub.join(", "));
            println!("    T01[0:5]: [{}]", t1_sub.join(", "));
            // Also print T15 (last valid position) and T16 (padding)
            if t_dim > 15 {
                let t15_offset = 15 * d_model;
                let t15_sub: Vec<String> = sub_slice[t15_offset..t15_offset + 5]
                    .iter()
                    .map(|x| format!("{:.6}", x))
                    .collect();
                println!("    T15[0:5]: [{}]", t15_sub.join(", "));
            }
            if t_dim > 16 {
                let t16_offset = 16 * d_model;
                let t16_sub: Vec<String> = sub_slice[t16_offset..t16_offset + 5]
                    .iter()
                    .map(|x| format!("{:.6}", x))
                    .collect();
                println!("    T16[0:5]: [{}]", t16_sub.join(", "));
            }

            // Xscale
            println!("  Xscale: {}", encoder.xscale());

            // After scaling
            let scale_flat = debug_out.after_scaling.reshape(&[-1])?;
            scale_flat.eval()?;
            let scale_slice: &[f32] = scale_flat.as_slice();
            println!(
                "  After scaling: shape={:?}",
                debug_out.after_scaling.shape()
            );
            let t0_scale: Vec<String> = scale_slice[..5]
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            println!("    T00[0:5]: [{}]", t0_scale.join(", "));
            // Also print T16 (last valid position)
            if t_dim > 16 {
                let t16_offset = 16 * d_model;
                let t16_scale: Vec<String> = scale_slice[t16_offset..t16_offset + 5]
                    .iter()
                    .map(|x| format!("{:.6}", x))
                    .collect();
                println!("    T16[0:5]: [{}]", t16_scale.join(", "));
            }

            // Positional encoding
            let pe_flat = debug_out.pos_emb.reshape(&[-1])?;
            pe_flat.eval()?;
            let pe_slice: &[f32] = pe_flat.as_slice();
            println!("  Pos emb: shape={:?}", debug_out.pos_emb.shape());
            let pe_t0: Vec<String> = pe_slice[..5].iter().map(|x| format!("{:.6}", x)).collect();
            println!("    PE[0][0:5]: [{}]", pe_t0.join(", "));
            let pe_d = debug_out.pos_emb.dim(2) as usize;
            let pe_t1: Vec<String> = pe_slice[pe_d..pe_d + 5]
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            println!("    PE[1][0:5]: [{}]", pe_t1.join(", "));

            // Layer outputs
            for (i, layer_out) in debug_out.layer_outputs.iter().enumerate() {
                let layer_flat = layer_out.reshape(&[-1])?;
                layer_flat.eval()?;
                let layer_slice: &[f32] = layer_flat.as_slice();
                let t0_layer: Vec<String> = layer_slice[..5]
                    .iter()
                    .map(|x| format!("{:.6}", x))
                    .collect();
                let t1_layer: Vec<String> = layer_slice[d_model..d_model + 5]
                    .iter()
                    .map(|x| format!("{:.6}", x))
                    .collect();
                println!("  After layer {}: T00[0:5]: [{}]", i, t0_layer.join(", "));
                println!("                  T01[0:5]: [{}]", t1_layer.join(", "));
            }
        }

        // Run full encoder (batch mode)
        let (encoded, encoded_len) = encoder.forward(&mel_batch, &mel_len)?;
        encoded.eval()?;
        encoded_len.eval()?;

        // encoded is [B, D, T], transpose to [B, T, D]
        let encoded_t = mlx_rs::ops::transpose_axes(&encoded, &[0, 2, 1])?;
        encoded_t.eval()?;

        let enc_flat = encoded_t.reshape(&[-1])?;
        enc_flat.eval()?;
        let enc_slice: &[f32] = enc_flat.as_slice();

        let t_dim = encoded_t.dim(1) as usize;
        let d_dim = encoded_t.dim(2) as usize;
        let valid_t_arr: &[i32] = encoded_len.as_slice();
        let valid_t = valid_t_arr[0] as usize;

        println!("\nEncoder call {}:", encoder_call);
        println!("  Audio samples: {}", chunk.len());
        println!(
            "  MEL shape: [{}, {}, {}], mel_len: {}",
            1, n_mels, num_mel_frames, num_mel_frames
        );
        println!(
            "  Encoded shape: [{}, {}, {}], encoded_len: {}",
            1, d_dim, t_dim, valid_t
        );
        println!("  Valid time steps: {}", valid_t);

        // Statistics
        let enc_min = enc_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let enc_max = enc_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let enc_mean: f32 = enc_slice.iter().sum::<f32>() / enc_slice.len() as f32;
        let enc_std: f32 = (enc_slice
            .iter()
            .map(|&v| (v - enc_mean).powi(2))
            .sum::<f32>()
            / enc_slice.len() as f32)
            .sqrt();

        println!(
            "  Stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}",
            enc_min, enc_max, enc_mean, enc_std
        );

        // Print first few time steps
        println!("\n  === Valid encoder output ===");
        let print_count = valid_t.min(8);
        for t in 0..print_count {
            let offset = t * d_dim;
            let vals: Vec<String> = enc_slice[offset..offset + 5]
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            println!("  T{:02} [0:5]: [{}]", t, vals.join(", "));
        }

        if valid_t > 8 {
            println!("  ...");
            // Last valid time step
            let t = valid_t - 1;
            let offset = t * d_dim;
            let vals: Vec<String> = enc_slice[offset..offset + 5]
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            println!("  T{:02} [0:5]: [{}]", t, vals.join(", "));
        }

        audio_offset += audio_needed;
        encoder_call += 1;
    }

    println!("\n=== Done: {} encoder calls ===", encoder_call);

    Ok(())
}
