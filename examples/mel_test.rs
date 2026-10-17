//! Test mel spectrogram computation for comparison with Python NeMo.

use mlx_voice::asr::modules::{MelSpectrogram, MelSpectrogramConfig};
use mlx_voice::asr::streaming::StreamingMelBuffer;

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
    let config = MelSpectrogramConfig {
        sample_rate: 16000,
        window_size: 0.025,
        window_stride: 0.01,
        window: "hann".to_string(),
        features: 128,
        n_fft: 512,
        log: true,
        normalize: "NA".to_string(),
        dither: 0.0,         // Disable dither (Python only dithers in training mode)
        preemph: Some(0.97), // NeMo default - FilterbankFeatures has preemph=0.97 hardcoded
        fmin: 0.0,
        fmax: None,
    };

    println!("\nPreprocessor config:");
    println!("  preemph: {:?}", config.preemph);
    println!("  dither: {}", config.dither);

    let mel_proc = MelSpectrogram::new(config)?;

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

    // ===== Test StreamingMelBuffer =====
    println!("\n=== Testing StreamingMelBuffer ===");

    let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 128, 512);

    // Append mel to buffer
    buffer.append_mel(&mel)?;
    println!(
        "Buffer after append: total_frames={}, buffer_idx={}",
        buffer.total_frames, buffer.buffer_idx
    );

    // Test extraction with streaming params (chunk_size=112, pre_encode_cache=9)
    let chunk_size = 112;
    let shift_size = 112;
    let pre_encode_cache = 9;
    let total_needed = chunk_size + pre_encode_cache;

    println!(
        "\nAttempting to extract: chunk_size={}, pre_encode_cache={}, total_needed={}",
        chunk_size, pre_encode_cache, total_needed
    );

    if let Some((extracted, len, _valid_frames)) =
        buffer.extract_chunk(chunk_size, shift_size, pre_encode_cache)
    {
        extracted.eval()?;
        println!(
            "Extracted chunk shape: {:?}, len: {}",
            extracted.shape(),
            len
        );

        let extracted_flat = extracted.reshape(&[-1])?;
        extracted_flat.eval()?;
        let ex_slice: &[f32] = extracted_flat.as_slice();
        let ex_min = ex_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let ex_max = ex_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex_mean: f32 = ex_slice.iter().sum::<f32>() / ex_slice.len() as f32;

        println!(
            "Extracted stats: min={:.6}, max={:.6}, mean={:.6}",
            ex_min, ex_max, ex_mean
        );

        let ex_positive = ex_slice.iter().filter(|&&v| v > 0.0).count();
        println!(
            "Positive values in extracted: {} out of {}",
            ex_positive,
            ex_slice.len()
        );

        // Print first frame of extracted
        println!("\nFirst frame of extracted (first 10 bins):");
        let ex_frames = extracted.dim(1) as usize;
        for i in 0..10.min(n_mels) {
            print!("{:.6} ", ex_slice[i * ex_frames]);
        }
        println!();
    } else {
        println!("ERROR: Failed to extract chunk!");
    }

    // ===== Test multiple chunks like streaming =====
    println!("\n=== Testing Multi-chunk Streaming ===");

    let config2 = MelSpectrogramConfig {
        sample_rate: 16000,
        window_size: 0.025,
        window_stride: 0.01,
        window: "hann".to_string(),
        features: 128,
        n_fft: 512,
        log: true,
        normalize: "NA".to_string(),
        dither: 0.0,
        preemph: Some(0.97), // NeMo default
        fmin: 0.0,
        fmax: None,
    };

    let mel_proc2 = MelSpectrogram::new(config2)?;
    let mut buffer2 = StreamingMelBuffer::new(16000, 0.025, 0.01, 128, 512);

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

        buffer2.append_mel(&mel_chunk)?;
    }

    println!(
        "\nTotal positive values across all chunks: {}",
        total_positive
    );
    println!("Buffer total_frames: {}", buffer2.total_frames);

    Ok(())
}
