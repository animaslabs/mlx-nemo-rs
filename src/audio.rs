//! Audio processing utilities for speech recognition.
//!
//! This module provides audio loading, preprocessing, and feature extraction
//! utilities optimized for ASR models.

use std::path::Path;

use mlx_rs::Array;

use crate::core::error::{Error, Result};

/// Audio file loading utilities.
pub mod loader {
    use super::*;

    /// Load audio from a WAV file.
    #[cfg(feature = "asr")]
    pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
        use hound::WavReader;

        let path = path.as_ref();
        let reader = WavReader::open(path).map_err(|e| {
            Error::audio(format!(
                "Failed to open WAV file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as usize;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                let samples: Vec<f32> = reader
                    .into_samples::<f32>()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| Error::audio(format!("Failed to read float samples: {}", e)))?;

                // Convert stereo to mono by averaging
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
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| Error::audio(format!("Failed to read int samples: {}", e)))?;

                // Convert to float and handle channels
                if channels > 1 {
                    samples
                        .chunks(channels)
                        .map(|chunk| {
                            chunk.iter().map(|&s| s as f32).sum::<f32>()
                                / (channels as f32 * max_val)
                        })
                        .collect()
                } else {
                    samples.iter().map(|&s| s as f32 / max_val).collect()
                }
            }
        };

        Ok((samples, sample_rate))
    }

    /// Load audio from a WAV file (fallback when asr feature is disabled).
    #[cfg(not(feature = "asr"))]
    pub fn load_wav(_path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
        Err(Error::not_implemented(
            "Audio loading requires the 'asr' feature",
        ))
    }

    /// Save audio to a WAV file.
    #[cfg(feature = "asr")]
    pub fn save_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
        use hound::WavWriter;

        let path = path.as_ref();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let writer = WavWriter::create(path, spec).map_err(|e| {
            Error::audio(format!(
                "Failed to create WAV file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let mut writer = writer;
        for &sample in samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| Error::audio(format!("Failed to write sample: {}", e)))?;
        }

        writer
            .finalize()
            .map_err(|e| Error::audio(format!("Failed to finalize WAV file: {}", e)))?;

        Ok(())
    }

    #[cfg(not(feature = "asr"))]
    pub fn save_wav(_path: impl AsRef<Path>, _samples: &[f32], _sample_rate: u32) -> Result<()> {
        Err(Error::not_implemented(
            "Audio saving requires the 'asr' feature",
        ))
    }
}

/// Audio resampling utilities.
pub mod resample {
    use super::*;

    /// Resample audio to a target sample rate.
    #[cfg(feature = "asr")]
    pub fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        use rubato::{
            Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
            WindowFunction,
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

        let ratio = to_rate as f64 / from_rate as f64;
        let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, samples.len(), 1)
            .map_err(|e| Error::audio(format!("Failed to create resampler: {}", e)))?;

        let waves_in = vec![samples.to_vec()];
        let waves_out = resampler
            .process(&waves_in, None)
            .map_err(|e| Error::audio(format!("Resampling failed: {}", e)))?;

        Ok(waves_out.into_iter().next().unwrap_or_default())
    }

    #[cfg(not(feature = "asr"))]
    pub fn resample_audio(_samples: &[f32], _from_rate: u32, _to_rate: u32) -> Result<Vec<f32>> {
        Err(Error::not_implemented(
            "Audio resampling requires the 'asr' feature",
        ))
    }
}

/// Audio preprocessing utilities.
pub mod preprocess {

    /// Normalize audio to [-1, 1] range.
    pub fn normalize(samples: &mut [f32]) {
        let max_val = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        if max_val > 0.0 {
            for sample in samples.iter_mut() {
                *sample /= max_val;
            }
        }
    }

    /// Apply pre-emphasis filter to audio.
    pub fn pre_emphasis(samples: &mut [f32], coeff: f32) {
        if samples.len() <= 1 {
            return;
        }

        for i in (1..samples.len()).rev() {
            samples[i] = samples[i] - coeff * samples[i - 1];
        }
    }

    /// Remove DC offset from audio.
    pub fn remove_dc_offset(samples: &mut [f32]) {
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        for sample in samples.iter_mut() {
            *sample -= mean;
        }
    }

    /// Apply window function to audio.
    pub fn apply_window(samples: &mut [f32], window_type: WindowType) {
        let window = generate_window(samples.len(), window_type);
        for (sample, &window_val) in samples.iter_mut().zip(window.iter()) {
            *sample *= window_val;
        }
    }

    /// Generate window function.
    pub fn generate_window(length: usize, window_type: WindowType) -> Vec<f32> {
        match window_type {
            WindowType::Hann => (0..length)
                .map(|i| {
                    0.5 * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos())
                })
                .collect(),
            WindowType::Hamming => (0..length)
                .map(|i| {
                    0.54 - 0.46
                        * (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos()
                })
                .collect(),
            WindowType::Blackman => (0..length)
                .map(|i| {
                    let n = i as f32;
                    let len = length as f32;
                    0.42 - 0.5 * (2.0 * std::f32::consts::PI * n / (len - 1.0)).cos()
                        + 0.08 * (4.0 * std::f32::consts::PI * n / (len - 1.0)).cos()
                })
                .collect(),
            WindowType::Rectangular => vec![1.0; length],
        }
    }

    /// Window function types.
    #[derive(Debug, Clone, Copy)]
    pub enum WindowType {
        Hann,
        Hamming,
        Blackman,
        Rectangular,
    }
}

/// Audio feature extraction utilities.
pub mod features {
    use super::*;

    /// Extract mel spectrogram features from audio.
    pub fn mel_spectrogram(
        audio: &[f32],
        _sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        _f_min: f32,
        _f_max: f32,
    ) -> Result<Array> {
        // This is a simplified implementation
        // In practice, you'd use a more sophisticated mel filter bank

        let n_frames = (audio.len() - n_fft) / hop_length + 1;
        let mut spectrogram = vec![0.0f32; n_frames * n_mels];

        // Simple STFT + mel filter bank (placeholder implementation)
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = start + n_fft;

            if end <= audio.len() {
                let frame = &audio[start..end];

                // Apply window
                let window = preprocess::generate_window(n_fft, preprocess::WindowType::Hann);
                let windowed_frame: Vec<f32> = frame
                    .iter()
                    .zip(window.iter())
                    .map(|(&x, &w)| x * w)
                    .collect();

                // Compute FFT (simplified - just magnitude spectrum)
                let fft_result = compute_fft(&windowed_frame)?;

                // Apply mel filter bank (simplified)
                for mel_idx in 0..n_mels.min(fft_result.len() / 2) {
                    let mel_val = fft_result[mel_idx];
                    spectrogram[frame_idx * n_mels + mel_idx] = mel_val;
                }
            }
        }

        Ok(Array::from_slice(
            &spectrogram,
            &[n_frames as i32, n_mels as i32],
        ))
    }

    /// Compute FFT magnitude spectrum (simplified implementation).
    fn compute_fft(samples: &[f32]) -> Result<Vec<f32>> {
        // This is a placeholder for a proper FFT implementation
        // In practice, you'd use a library like rustfft or implement Cooley-Tukey

        let n = samples.len();
        let mut spectrum = vec![0.0f32; n / 2 + 1];

        // Simple DFT for demonstration (very slow!)
        for k in 0..=n / 2 {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, &sample) in samples.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt();
        }

        Ok(spectrum)
    }

    /// Extract MFCC features from audio.
    pub fn mfcc(
        audio: &[f32],
        sample_rate: u32,
        n_mfcc: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> Result<Array> {
        // First compute mel spectrogram
        let mel_spec = mel_spectrogram(
            audio,
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            0.0,
            sample_rate as f32 / 2.0,
        )?;

        // Apply log compression
        let log_mel = &mel_spec; // Placeholder for log operation

        // Apply DCT to get MFCCs (simplified)
        let mel_data = vec![0.0f32; log_mel.shape().iter().product::<i32>() as usize]; // Placeholder
        let (n_frames, n_mel_bins) = (log_mel.shape()[0] as usize, log_mel.shape()[1] as usize);

        let mut mfcc_features = vec![0.0f32; n_frames * n_mfcc];

        for frame_idx in 0..n_frames {
            for mfcc_idx in 0..n_mfcc.min(n_mel_bins) {
                let mut mfcc_val = 0.0f32;

                for mel_idx in 0..n_mel_bins {
                    let angle = std::f32::consts::PI * mfcc_idx as f32 * (mel_idx as f32 + 0.5)
                        / n_mel_bins as f32;
                    mfcc_val += mel_data[frame_idx * n_mel_bins + mel_idx] * angle.cos();
                }

                mfcc_features[frame_idx * n_mfcc + mfcc_idx] = mfcc_val;
            }
        }

        Ok(Array::from_slice(
            &mfcc_features,
            &[n_frames as i32, n_mfcc as i32],
        ))
    }
}

/// Audio batching utilities.
pub mod batching {
    use super::*;

    /// Batch multiple audio sequences with padding.
    pub fn batch_audio(audio_sequences: &[Vec<f32>], padding_value: f32) -> Result<(Array, Array)> {
        if audio_sequences.is_empty() {
            return Err(Error::audio("Cannot batch empty audio sequences"));
        }

        let batch_size = audio_sequences.len();
        let max_length = audio_sequences
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(0);

        let mut batched_data = vec![padding_value; batch_size * max_length];
        let mut lengths = Vec::with_capacity(batch_size);

        for (batch_idx, sequence) in audio_sequences.iter().enumerate() {
            let offset = batch_idx * max_length;
            let seq_len = sequence.len();

            batched_data[offset..offset + seq_len].copy_from_slice(sequence);
            lengths.push(seq_len as i32);
        }

        let audio_tensor =
            Array::from_slice(&batched_data, &[batch_size as i32, max_length as i32]);
        let lengths_tensor = Array::from_slice(&lengths, &[batch_size as i32]);

        Ok((audio_tensor, lengths_tensor))
    }

    /// Unbatch audio sequences.
    pub fn unbatch_audio(batched_audio: &Array, lengths: &[usize]) -> Result<Vec<Vec<f32>>> {
        let shape = batched_audio.shape();
        if shape.len() != 2 {
            return Err(Error::audio("Batched audio must be 2-dimensional"));
        }

        let batch_size = shape[0] as usize;
        let max_length = shape[1] as usize;

        if lengths.len() != batch_size {
            return Err(Error::audio("Lengths array must match batch size"));
        }

        let batched_data = vec![0.0f32; batched_audio.shape().iter().product::<i32>() as usize]; // Placeholder
        let mut sequences = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let seq_len = lengths[batch_idx].min(max_length);
            let offset = batch_idx * max_length;
            let sequence = batched_data[offset..offset + seq_len].to_vec();
            sequences.push(sequence);
        }

        Ok(sequences)
    }
}

pub use batching::{batch_audio, unbatch_audio};
pub use features::{mel_spectrogram, mfcc};
/// Re-export commonly used audio utilities.
pub use loader::{load_wav, save_wav};
pub use preprocess::{apply_window, normalize, pre_emphasis, remove_dc_offset, WindowType};
pub use resample::resample_audio;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_generation() {
        let hann_window = preprocess::generate_window(100, preprocess::WindowType::Hann);
        assert_eq!(hann_window.len(), 100);

        let hamming_window = preprocess::generate_window(100, preprocess::WindowType::Hamming);
        assert_eq!(hamming_window.len(), 100);

        let rect_window = preprocess::generate_window(100, preprocess::WindowType::Rectangular);
        assert_eq!(rect_window.len(), 100);
        assert!(rect_window.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_audio_preprocessing() {
        let mut samples = vec![0.5, -0.3, 0.8, -0.1, 0.2];

        // Test normalization
        preprocess::normalize(&mut samples);
        let max_val = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        assert!((max_val - 1.0).abs() < 1e-6);

        // Test pre-emphasis
        let mut samples2 = vec![0.5, -0.3, 0.8, -0.1, 0.2];
        let original = samples2.clone();
        preprocess::pre_emphasis(&mut samples2, 0.97);
        assert_ne!(samples2, original);
    }

    #[test]
    fn test_audio_batching() {
        let sequences = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5],
            vec![0.6, 0.7, 0.8, 0.9],
        ];

        let (batched, lengths) = batching::batch_audio(&sequences, 0.0).unwrap();
        assert_eq!(batched.shape(), &[3, 4]);
        assert_eq!(lengths.shape(), &[3]);

        lengths.eval().unwrap();
        let lengths_data: &[i32] = lengths.as_slice();
        assert_eq!(lengths_data, &[3, 2, 4]);
    }
}
