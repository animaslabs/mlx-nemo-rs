//! Test mel spectrogram computation for comparison with Python NeMo.

use mlx_voice::asr::config::{Normalize, WindowType};
use mlx_voice::asr::modules::{MelConfig, MelProcessor};

fn main() -> mlx_voice::Result<()> {
    // Load audio
    let (samples, sr) = mlx_voice::audio::load_wav("samples/2086-149220-0033.wav")?;
    println!("Audio: {} samples at {} Hz", samples.len(), sr);
    println!(
        "Audio stats: min={:.6}, max={:.6}, mean={:.6}",
        samples.iter().cloned().fold(f32::INFINITY, f32::min),
        samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        samples.iter().sum::<f32>() / samples.len() as f32
    );

    // Create preprocessor with NeMo config
    // NeMo's FilterbankFeatures has a hardcoded default of preemph=0.97
    // Python NeMo only applies dither in training mode, not inference
    let mut config = MelConfig::default();
    config.sample_rate = 16000;
    config.window_size = 0.025;
    config.window_stride = 0.01;
    config.window = WindowType::Hann;
    config.features = 128;
    config.n_fft = 512;
    config.log = true;
    config.normalize = Normalize::Disabled;
    config.dither = 0.0; // Disable dither (Python only dithers in training mode)
    config.preemph = Some(0.97); // NeMo default - FilterbankFeatures has preemph=0.97 hardcoded
    config.fmin = 0.0;
    config.fmax = None;

    println!("\nPreprocessor config:");
    println!("  preemph: {:?}", config.preemph);
    println!("  dither: {}", config.dither);

    let mel_proc = MelProcessor::new(config)?;

    // Process first chunk (same as Python streaming: 19360 samples for 121 mel frames)
    let chunk_samples = 19360;
    let audio_chunk = &samples[..chunk_samples.min(samples.len())];
    println!("\nProcessing audio chunk: {} samples", audio_chunk.len());

    let audio_array = mlx_rs::Array::from_slice(audio_chunk, &[1, audio_chunk.len() as i32]);
    let (mel, _) = mel_proc.forward(&audio_array, None)?;
    mel.eval()?;

    println!("Mel output shape: {:?}", mel.shape());

    // Remove batch dimension for analysis: [1, n_mels, frames] -> [n_mels, frames]
    use mlx_rs::ops::indexing::IndexOp;
    let mel = mel.index(0);

    let mel_flat = mel.reshape(&[-1])?;
    mel_flat.eval()?;
    let mel_slice: &[f32] = mel_flat.as_slice();
    let min = mel_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = mel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = mel_slice.iter().sum::<f32>() / mel_slice.len() as f32;
    let std: f32 = (mel_slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
        / mel_slice.len() as f32)
        .sqrt();

    println!(
        "Mel stats: min={:.6}, max={:.6}, mean={:.6}, std={:.6}",
        min, max, mean, std
    );

    let positive_count = mel_slice.iter().filter(|&&v| v > 0.0).count();
    println!(
        "Number of positive values: {} out of {}",
        positive_count,
        mel_slice.len()
    );

    // Print first 10 values of first frame
    // mel shape is [n_mels, frames], so first frame is mel[:, 0]
    let n_mels = mel.dim(0) as usize;
    println!("\nFirst mel frame (first 10 bins):");
    for i in 0..10.min(n_mels) {
        // mel is [n_mels, frames], flattened is row-major so mel[i, 0] = mel_slice[i * frames]
        let frames = mel.dim(1) as usize;
        print!("{:.6} ", mel_slice[i * frames]);
    }
    println!();

    // ===== Test multiple chunks like streaming =====
    println!("\n=== Testing Multi-chunk Streaming ===");

    let mut config2 = MelConfig::default();
    config2.sample_rate = 16000;
    config2.window_size = 0.025;
    config2.window_stride = 0.01;
    config2.window = WindowType::Hann;
    config2.features = 128;
    config2.n_fft = 512;
    config2.log = true;
    config2.normalize = Normalize::Disabled;
    config2.dither = 0.0;
    config2.preemph = Some(0.97); // NeMo default
    config2.fmin = 0.0;
    config2.fmax = None;

    let mel_proc2 = MelProcessor::new(config2)?;

    // Process audio in multiple chunks (simulating streaming)
    let audio_chunk_size = 19360; // ~1210ms
    let mut total_positive = 0;

    for (chunk_idx, audio_chunk) in samples.chunks(audio_chunk_size).enumerate() {
        let audio_array = mlx_rs::Array::from_slice(audio_chunk, &[1, audio_chunk.len() as i32]);
        let (mel_batch, _) = mel_proc2.forward(&audio_array, None)?;
        mel_batch.eval()?;
        // Remove batch dimension: [1, n_mels, frames] -> [n_mels, frames]
        let mel_chunk = mel_batch.index(0);

        let mel_flat = mel_chunk.reshape(&[-1])?;
        mel_flat.eval()?;
        let mel_slice: &[f32] = mel_flat.as_slice();
        let min = mel_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = mel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let positive = mel_slice.iter().filter(|&&v| v > 0.0).count();
        total_positive += positive;

        println!(
            "Chunk {}: shape={:?}, min={:.6}, max={:.6}, positive={}",
            chunk_idx,
            mel_chunk.shape(),
            min,
            max,
            positive
        );
    }

    println!(
        "\nTotal positive values across all chunks: {}",
        total_positive
    );

    Ok(())
}
