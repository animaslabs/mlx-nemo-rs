//! Mel spectrogram audio preprocessing.
//!
//! Converts raw audio waveforms to log-Mel spectrogram features.

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::asr::config::PreprocessorConfig;
use crate::core::error::{Error, Result};

/// Mel spectrogram feature extractor.
///
/// This preprocessor converts raw audio waveforms to log-Mel spectrogram
/// features, matching NeMo's `AudioToMelSpectrogramPreprocessor`.
pub struct MelSpectrogram {
    config: PreprocessorConfig,
    /// Mel filterbank matrix [n_mels, n_fft/2 + 1]
    mel_filterbank: Array,
    /// Window function [win_length]
    window: Array,
    /// Window length in samples
    win_length: usize,
    /// Hop length in samples
    hop_length: usize,
}

impl MelSpectrogram {
    /// Create a new Mel spectrogram preprocessor from configuration.
    pub fn new(config: PreprocessorConfig) -> Result<Self> {
        let sample_rate = config.sample_rate as f32;
        let win_length = (config.window_size * sample_rate) as usize;
        let hop_length = (config.window_stride * sample_rate) as usize;
        let n_fft = config.n_fft;
        let n_mels = config.features;

        // Create window function
        let window = Self::create_window(&config.window, win_length)?;

        // Create Mel filterbank
        let lowfreq = config.lowfreq.unwrap_or(0.0);
        let highfreq = config.highfreq.unwrap_or(sample_rate / 2.0);
        let mel_filterbank =
            Self::create_mel_filterbank(n_mels, n_fft, sample_rate, lowfreq, highfreq)?;

        Ok(Self {
            config,
            mel_filterbank,
            window,
            win_length,
            hop_length,
        })
    }

    /// Process a batch of audio waveforms to Mel spectrograms.
    ///
    /// # Arguments
    /// * `audio` - Audio waveform tensor [batch, time] or [time]
    /// * `lengths` - Optional lengths for each sample in the batch [batch]
    ///
    /// # Returns
    /// * Mel spectrogram features [batch, n_mels, frames]
    /// * Output lengths [batch]
    pub fn forward(&self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Ensure audio is 2D [batch, time]
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            audio.clone()
        };

        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        // Apply optional preemphasis
        let audio = if let Some(coef) = self.config.preemph {
            self.preemphasis(&audio, coef)?
        } else {
            audio
        };

        // Add dither for numerical stability
        let audio = if self.config.dither > 0.0 {
            let dither_noise =
                mlx_rs::random::normal::<f32>(audio.shape(), None, Some(self.config.dither), None)?;
            &audio + &dither_noise
        } else {
            audio
        };

        // Compute STFT
        let stft_result = self.stft(&audio)?;

        // Compute power spectrum
        let power_spectrum = self.complex_abs_squared(&stft_result)?;

        // Apply Mel filterbank
        // power_spectrum: [batch, freq_bins, frames]
        // mel_filterbank: [n_mels, freq_bins]
        // result: [batch, n_mels, frames]
        let mel_spec = mlx_rs::ops::einsum(
            "mf,bft->bmt",
            &[self.mel_filterbank.clone(), power_spectrum],
        )?;

        // Apply log with guard for numerical stability
        // NeMo uses log_zero_guard_type="add": log(x + guard) rather than log(max(x, guard))
        // NeMo's log_zero_guard = 2^-24
        let mel_spec = &mel_spec + &Array::from_f32(2.0_f32.powi(-24));
        let log_mel_spec = mlx_rs::ops::log(&mel_spec)?;

        // Normalize if requested
        let log_mel_spec = match self.config.normalize.as_str() {
            "per_feature" => self.normalize_per_feature(&log_mel_spec)?,
            _ => log_mel_spec,
        };

        // Compute output lengths
        let output_lengths = if let Some(lens) = lengths {
            self.compute_output_lengths(lens)?
        } else {
            // Compute based on total audio length with center padding
            // PyTorch formula: n_frames = floor((padded_len - n_fft) / hop_length) + 1
            let audio_len_usize = audio_len as usize;
            let n_fft = self.config.n_fft;
            let padded_len = audio_len_usize + n_fft; // center padding adds n_fft/2 on each side
            let n_frames = (padded_len - n_fft) / self.hop_length + 1;
            mlx_rs::ops::full::<i32>(&[batch_size], &Array::from_int(n_frames as i32))?
        };

        Ok((log_mel_spec, output_lengths))
    }

    /// Get the output feature dimension.
    pub fn output_dim(&self) -> usize {
        self.config.features
    }

    /// Get reference to the mel filterbank for debugging.
    pub fn filterbank(&self) -> &Array {
        &self.mel_filterbank
    }

    /// Process audio and return BOTH pre-normalized and post-normalized mel spectrograms.
    /// Used for debugging to compare with reference implementations.
    pub fn forward_debug(
        &self,
        audio: &Array,
        lengths: Option<&Array>,
    ) -> Result<(Array, Array, Array)> {
        // Ensure audio is 2D [batch, time]
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            audio.clone()
        };

        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        // Apply optional preemphasis
        let audio = if let Some(coef) = self.config.preemph {
            self.preemphasis(&audio, coef)?
        } else {
            audio
        };

        // Note: In NeMo, dither is only applied in training mode
        // For inference (eval mode), dither is skipped. So we skip it here.

        // Compute STFT
        let stft_result = self.stft(&audio)?;

        // Compute power spectrum
        let power_spectrum = self.complex_abs_squared(&stft_result)?;

        // Apply Mel filterbank
        let mel_spec = mlx_rs::ops::einsum(
            "mf,bft->bmt",
            &[self.mel_filterbank.clone(), power_spectrum],
        )?;

        // Apply log with guard for numerical stability
        // NeMo uses log_zero_guard_type="add": log(x + guard) rather than log(max(x, guard))
        // NeMo's log_zero_guard = 2^-24
        let mel_spec = &mel_spec + &Array::from_f32(2.0_f32.powi(-24));
        let log_mel_spec = mlx_rs::ops::log(&mel_spec)?;

        // Save pre-normalized
        let pre_norm_mel = log_mel_spec.clone();

        // Normalize if requested
        let log_mel_spec = match self.config.normalize.as_str() {
            "per_feature" => self.normalize_per_feature(&log_mel_spec)?,
            _ => log_mel_spec,
        };

        // Compute output lengths
        let output_lengths = if let Some(lens) = lengths {
            self.compute_output_lengths(lens)?
        } else {
            let audio_len_usize = audio_len as usize;
            let n_fft = self.config.n_fft;
            let padded_len = audio_len_usize + n_fft;
            let n_frames = (padded_len - n_fft) / self.hop_length + 1;
            mlx_rs::ops::full::<i32>(&[batch_size], &Array::from_int(n_frames as i32))?
        };

        Ok((pre_norm_mel, log_mel_spec, output_lengths))
    }

    // Private helper methods

    fn create_window(window_type: &str, length: usize) -> Result<Array> {
        let n = length as f32;
        match window_type {
            "hann" => {
                // Hann window: 0.5 * (1 - cos(2*pi*n / (N-1)))
                let indices: Vec<f32> = (0..length).map(|i| i as f32).collect();
                let indices = Array::from_slice(&indices, &[length as i32]);
                let scale = Array::from_f32(2.0 * std::f32::consts::PI / (n - 1.0));
                let cos_arg = &indices * &scale;
                let cos_vals = mlx_rs::ops::cos(&cos_arg)?;
                let one_minus_cos = &Array::from_f32(1.0) - &cos_vals;
                let window = &Array::from_f32(0.5) * &one_minus_cos;
                Ok(window)
            }
            "hamming" => {
                // Hamming window: 0.54 - 0.46 * cos(2*pi*n / (N-1))
                let indices: Vec<f32> = (0..length).map(|i| i as f32).collect();
                let indices = Array::from_slice(&indices, &[length as i32]);
                let scale = Array::from_f32(2.0 * std::f32::consts::PI / (n - 1.0));
                let cos_arg = &indices * &scale;
                let cos_vals = mlx_rs::ops::cos(&cos_arg)?;
                let scaled_cos = &Array::from_f32(0.46) * &cos_vals;
                let window = &Array::from_f32(0.54) - &scaled_cos;
                Ok(window)
            }
            _ => Err(Error::config(format!(
                "Unknown window type: {}",
                window_type
            ))),
        }
    }

    fn create_mel_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        lowfreq: f32,
        highfreq: f32,
    ) -> Result<Array> {
        let n_freqs = n_fft / 2 + 1;

        // Slaney mel scale (used by librosa and NeMo)
        // This is NOT the HTK mel scale!
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = (min_log_hz - 0.0) / f_sp;
        let logstep = (6.4_f32).ln() / 27.0;

        let hz_to_mel = |hz: f32| -> f32 {
            if hz >= min_log_hz {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            } else {
                hz / f_sp
            }
        };

        let mel_to_hz = |mel: f32| -> f32 {
            if mel >= min_log_mel {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            } else {
                mel * f_sp
            }
        };

        let mel_low = hz_to_mel(lowfreq);
        let mel_high = hz_to_mel(highfreq);

        // Create mel points (n_mels + 2 points for triangular filters)
        // Linearly spaced in mel scale
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * (i as f32) / ((n_mels + 1) as f32))
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // FFT frequencies
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();

        // Compute frequency differences (n_mels + 1 differences)
        let fdiff: Vec<f32> = hz_points.windows(2).map(|w| w[1] - w[0]).collect();

        // Create filterbank matrix using librosa's method:
        // lower = (f - f_left) / (f_center - f_left)
        // upper = (f_right - f) / (f_right - f_center)
        // value = max(0, min(lower, upper))
        let mut filterbank = vec![0.0f32; n_mels * n_freqs];

        for m in 0..n_mels {
            let f_left = hz_points[m];
            let _f_center = hz_points[m + 1];
            let f_right = hz_points[m + 2];

            for (k, &freq) in fft_freqs.iter().enumerate() {
                // lower slope: (f - f_left) / (f_center - f_left) = -(f_left - f) / fdiff[m]
                // which is (freq - f_left) / fdiff[m]
                let lower = (freq - f_left) / fdiff[m];
                // upper slope: (f_right - f) / (f_right - f_center) = (f_right - freq) / fdiff[m+1]
                let upper = (f_right - freq) / fdiff[m + 1];

                // Take minimum of lower and upper, clipped to >= 0
                let value = lower.min(upper).max(0.0);
                filterbank[m * n_freqs + k] = value;
            }
        }

        // Apply Slaney normalization: multiply each filter by 2 / (f_right - f_left)
        for m in 0..n_mels {
            let f_left = hz_points[m];
            let f_right = hz_points[m + 2];
            let enorm = 2.0 / (f_right - f_left);

            for k in 0..n_freqs {
                filterbank[m * n_freqs + k] *= enorm;
            }
        }

        let filterbank = Array::from_slice(&filterbank, &[n_mels as i32, n_freqs as i32]);
        Ok(filterbank)
    }

    fn preemphasis(&self, audio: &Array, coef: f32) -> Result<Array> {
        // y[n] = x[n] - coef * x[n-1]
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        // Shift audio by 1: prepend zeros and remove last element
        let zeros = Array::zeros::<f32>(&[batch_size, 1])?;

        // Slice audio to get all but last sample using indexing API
        let sliced = audio.index((.., ..(audio_len - 1)));

        let shifted = mlx_rs::ops::concatenate_axis(&[&zeros, &sliced], 1)?;
        let coef_arr = Array::from_f32(coef);
        let scaled_shifted = &coef_arr * &shifted;
        let result = audio - &scaled_shifted;
        Ok(result)
    }

    fn stft(&self, audio: &Array) -> Result<Array> {
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];
        let n_fft = self.config.n_fft;

        // Apply center padding (pad by n_fft//2 on each side with ZEROS)
        // This matches NeMo's torch.stft with center=True, pad_mode='constant'
        let pad_size = n_fft / 2;

        // Zero padding on both sides
        audio.eval()?;
        let audio_data: &[f32] = audio.as_slice();

        let mut padded_data =
            Vec::with_capacity(batch_size as usize * (audio_len as usize + 2 * pad_size));

        for b in 0..batch_size as usize {
            let batch_offset = b * audio_len as usize;

            // Left padding: zeros
            padded_data.extend(std::iter::repeat_n(0.0, pad_size));

            // Original audio
            padded_data
                .extend_from_slice(&audio_data[batch_offset..batch_offset + audio_len as usize]);

            // Right padding: zeros
            padded_data.extend(std::iter::repeat_n(0.0, pad_size));
        }

        let padded_len = audio_len as usize + 2 * pad_size;
        let audio = Array::from_slice(&padded_data, &[batch_size, padded_len as i32]);

        // Calculate number of frames with the padded audio
        // PyTorch formula: n_frames = floor((padded_len - n_fft) / hop_length) + 1
        let hop_length = self.hop_length as i32;
        let n_frames = (padded_len as i32 - n_fft as i32) / hop_length + 1;

        // Create frames of length n_fft (not win_length!) - PyTorch extracts n_fft samples per frame
        let mut frames_data = Vec::with_capacity((batch_size * n_frames * n_fft as i32) as usize);

        audio.eval()?;
        let audio_data: &[f32] = audio.as_slice();

        for b in 0..batch_size {
            let batch_offset = b * padded_len as i32;
            for frame_idx in 0..n_frames {
                let start = frame_idx * hop_length;
                for i in 0..n_fft {
                    let idx = (batch_offset as usize) + (start as usize) + i;
                    frames_data.push(audio_data[idx]);
                }
            }
        }

        let frames = Array::from_slice(&frames_data, &[batch_size, n_frames, n_fft as i32]);

        // Create centered window: [zeros, window, zeros]
        // PyTorch centers the window within n_fft when win_length < n_fft
        let centered_window = if self.win_length < n_fft {
            let left_pad = (n_fft - self.win_length) / 2;
            let right_pad = n_fft - self.win_length - left_pad;
            let left_zeros = Array::zeros::<f32>(&[left_pad as i32])?;
            let right_zeros = Array::zeros::<f32>(&[right_pad as i32])?;
            mlx_rs::ops::concatenate_axis(&[&left_zeros, &self.window, &right_zeros], 0)?
        } else {
            self.window.clone()
        };

        // Apply centered window
        let windowed = &frames * &centered_window;

        // Compute FFT
        // windowed shape: [batch, frames, n_fft]
        let fft_result = mlx_rs::fft::rfft(&windowed, Some(n_fft as i32), Some(2))?;

        // Transpose to [batch, freq_bins, frames]
        let fft_result = mlx_rs::ops::transpose_axes(&fft_result, &[0, 2, 1])?;

        Ok(fft_result)
    }

    fn complex_abs_squared(&self, complex_tensor: &Array) -> Result<Array> {
        // For complex FFT output, compute |x|^2 = real^2 + imag^2
        let real = mlx_rs::ops::real(complex_tensor)?;
        let imag = mlx_rs::ops::imag(complex_tensor)?;
        let real_sq = &real * &real;
        let imag_sq = &imag * &imag;
        let result = &real_sq + &imag_sq;
        Ok(result)
    }

    fn normalize_per_feature(&self, features: &Array) -> Result<Array> {
        // Normalize each feature (mel bin) to zero mean and unit variance
        // features shape: [batch, n_mels, frames]
        let mean = mlx_rs::ops::mean_axis(features, 2, true)?;
        let centered = features - &mean;
        let centered_sq = &centered * &centered;
        let var = mlx_rs::ops::mean_axis(&centered_sq, 2, true)?;
        let var_plus_eps = &var + &Array::from_f32(1e-5);
        let std = mlx_rs::ops::sqrt(&var_plus_eps)?;
        let normalized = &centered / &std;
        Ok(normalized)
    }

    fn compute_output_lengths(&self, input_lengths: &Array) -> Result<Array> {
        // With center padding: padded_length = input_length + n_fft
        // PyTorch formula: n_frames = floor((padded_length - n_fft) / hop_length) + 1
        //                          = floor(input_length / hop_length) + 1
        let hop_len = self.hop_length as i32;

        let divided = mlx_rs::ops::floor_divide(input_lengths, Array::from_int(hop_len))?;
        let output_lengths = &divided + &Array::from_int(1);
        Ok(output_lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to check if two f32 values are approximately equal
    fn approx_eq(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    /// Helper to check if two arrays are approximately equal element-wise
    fn arrays_approx_eq(a: &[f32], b: &[f32], atol: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, atol))
    }

    /// Create a default test config
    fn test_config() -> PreprocessorConfig {
        PreprocessorConfig {
            sample_rate: 16000,
            window_size: 0.025,  // 400 samples
            window_stride: 0.01, // 160 samples
            window: "hann".to_string(),
            features: 80,
            n_fft: 512,
            log: true,
            normalize: "per_feature".to_string(),
            dither: 0.0, // Disable dither for deterministic tests
            pad_to: 0,
            preemph: Some(0.97),
            lowfreq: Some(0.0),
            highfreq: None,
            target: None,
        }
    }

    // ==================== Window Function Tests ====================

    #[test]
    fn test_hann_window_length() {
        let window = MelSpectrogram::create_window("hann", 400).unwrap();
        assert_eq!(window.shape(), &[400]);
    }

    #[test]
    fn test_hann_window_values() {
        let window = MelSpectrogram::create_window("hann", 400).unwrap();
        window.eval().unwrap();
        let values: &[f32] = window.as_slice();

        // Hann window: first and last should be 0, middle should be ~1
        assert!(approx_eq(values[0], 0.0, 1e-6), "First value should be 0");
        assert!(approx_eq(values[399], 0.0, 1e-6), "Last value should be 0");
        assert!(
            approx_eq(values[200], 1.0, 1e-4),
            "Middle value should be ~1, got {}",
            values[200]
        );

        // Sum should be approximately N/2 for Hann window
        let sum: f32 = values.iter().sum();
        assert!(
            approx_eq(sum, 199.5, 0.1),
            "Sum should be ~199.5, got {}",
            sum
        );
    }

    #[test]
    fn test_hann_window_symmetry() {
        let window = MelSpectrogram::create_window("hann", 400).unwrap();
        window.eval().unwrap();
        let values: &[f32] = window.as_slice();

        // Window should be symmetric: w[i] == w[N-1-i]
        for i in 0..200 {
            assert!(
                approx_eq(values[i], values[399 - i], 1e-6),
                "Window not symmetric at index {}: {} != {}",
                i,
                values[i],
                values[399 - i]
            );
        }
    }

    #[test]
    fn test_hamming_window_values() {
        let window = MelSpectrogram::create_window("hamming", 400).unwrap();
        window.eval().unwrap();
        let values: &[f32] = window.as_slice();

        // Hamming window: first and last should be ~0.08, middle should be ~1
        assert!(
            approx_eq(values[0], 0.08, 1e-5),
            "First value should be ~0.08, got {}",
            values[0]
        );
        assert!(
            approx_eq(values[399], 0.08, 1e-5),
            "Last value should be ~0.08, got {}",
            values[399]
        );
        assert!(
            approx_eq(values[200], 1.0, 1e-4),
            "Middle value should be ~1, got {}",
            values[200]
        );
    }

    #[test]
    fn test_unknown_window_error() {
        let result = MelSpectrogram::create_window("blackman", 400);
        assert!(result.is_err(), "Should error on unknown window type");
    }

    // ==================== Mel Filterbank Tests ====================

    #[test]
    fn test_filterbank_shape() {
        let filterbank =
            MelSpectrogram::create_mel_filterbank(80, 512, 16000.0, 0.0, 8000.0).unwrap();
        // n_mels=80, n_freqs = n_fft/2 + 1 = 257
        assert_eq!(filterbank.shape(), &[80, 257]);
    }

    #[test]
    fn test_filterbank_non_negative() {
        let filterbank =
            MelSpectrogram::create_mel_filterbank(80, 512, 16000.0, 0.0, 8000.0).unwrap();
        filterbank.eval().unwrap();
        let values: &[f32] = filterbank.as_slice();

        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min_val >= 0.0,
            "Filterbank should be non-negative, min={}",
            min_val
        );
    }

    #[test]
    fn test_filterbank_triangular_shape() {
        // Each mel filter should be triangular: values increase then decrease
        let filterbank =
            MelSpectrogram::create_mel_filterbank(80, 512, 16000.0, 0.0, 8000.0).unwrap();
        filterbank.eval().unwrap();
        let values: &[f32] = filterbank.as_slice();

        // Check a middle filter (filter 40)
        let n_freqs = 257;
        let filter_40: Vec<f32> = values[40 * n_freqs..(40 + 1) * n_freqs].to_vec();

        // Find non-zero values
        let non_zero: Vec<(usize, f32)> = filter_40
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 1e-10)
            .map(|(i, &v)| (i, v))
            .collect();

        assert!(!non_zero.is_empty(), "Filter should have non-zero values");

        // Find peak
        let (peak_idx, peak_val) = non_zero
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        assert!(*peak_val > 0.0, "Filter should have a positive peak");

        // Verify triangular shape: values before peak should be increasing
        let before_peak: Vec<&(usize, f32)> =
            non_zero.iter().filter(|(i, _)| i < peak_idx).collect();
        if before_peak.len() > 1 {
            for w in before_peak.windows(2) {
                assert!(
                    w[1].1 >= w[0].1 - 1e-6,
                    "Values before peak should be non-decreasing"
                );
            }
        }
    }

    // ==================== Preemphasis Tests ====================

    #[test]
    fn test_preemphasis_formula() {
        // y[n] = x[n] - 0.97 * x[n-1]
        // For impulse [1, 0, 0, 0, 0], output should be [1, -0.97, 0, 0, 0]
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let input = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0], &[1, 5]);
        let output = mel_spec.preemphasis(&input, 0.97).unwrap();
        output.eval().unwrap();
        let values: &[f32] = output.as_slice();

        let expected = [1.0f32, -0.97, 0.0, 0.0, 0.0];
        assert!(
            arrays_approx_eq(values, &expected, 1e-5),
            "Preemphasis output mismatch: {:?} vs {:?}",
            values,
            expected
        );
    }

    #[test]
    fn test_preemphasis_first_sample() {
        // First sample should be unchanged (x[-1] = 0, so y[0] = x[0])
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let input = Array::from_slice(&[0.5f32, 0.3, 0.1, -0.2, 0.4], &[1, 5]);
        let output = mel_spec.preemphasis(&input, 0.97).unwrap();
        output.eval().unwrap();
        let values: &[f32] = output.as_slice();

        assert!(
            approx_eq(values[0], 0.5, 1e-6),
            "First sample should be unchanged: {} vs 0.5",
            values[0]
        );
    }

    #[test]
    fn test_preemphasis_batch() {
        // Preemphasis should work on batched input
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let input = Array::from_slice(
            &[
                1.0f32, 0.0, 0.0, 0.0, 0.0, // batch 0
                0.5, 0.5, 0.5, 0.5, 0.5, // batch 1
            ],
            &[2, 5],
        );
        let output = mel_spec.preemphasis(&input, 0.97).unwrap();
        output.eval().unwrap();

        assert_eq!(output.shape(), &[2, 5], "Output shape should match input");
    }

    #[test]
    fn test_preemphasis_ramp() {
        // Test with ramp signal from golden data
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let input = Array::from_slice(&[0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5], &[1, 6]);
        let output = mel_spec.preemphasis(&input, 0.97).unwrap();
        output.eval().unwrap();
        let values: &[f32] = output.as_slice();

        // Expected from golden data (approximately)
        // y[0] = 0.0
        // y[1] = 0.1 - 0.97*0.0 = 0.1
        // y[2] = 0.2 - 0.97*0.1 = 0.103
        // y[3] = 0.3 - 0.97*0.2 = 0.106
        // etc.
        assert!(approx_eq(values[0], 0.0, 1e-5));
        assert!(approx_eq(values[1], 0.1, 1e-5));
        assert!(approx_eq(values[2], 0.103, 1e-3));
    }

    // ==================== Output Length Calculation Tests ====================

    #[test]
    fn test_output_length_single() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // 16000 samples -> floor(16000/160) + 1 = 101 frames
        let input_len = Array::from_slice(&[16000i32], &[1]);
        let output_len = mel_spec.compute_output_lengths(&input_len).unwrap();
        output_len.eval().unwrap();
        let values: &[i32] = output_len.as_slice();

        assert_eq!(values[0], 101, "16000 samples should produce 101 frames");
    }

    #[test]
    fn test_output_length_batch() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Test multiple lengths from golden data
        let input_lens = Array::from_slice(&[1600i32, 8000, 16000, 24000, 32000, 48000], &[6]);
        let output_lens = mel_spec.compute_output_lengths(&input_lens).unwrap();
        output_lens.eval().unwrap();
        let values: &[i32] = output_lens.as_slice();

        let expected = [11, 51, 101, 151, 201, 301];
        assert_eq!(values, expected, "Output lengths mismatch");
    }

    #[test]
    fn test_output_length_edge_cases() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Small lengths
        let input_lens = Array::from_slice(&[160i32, 320, 480], &[3]);
        let output_lens = mel_spec.compute_output_lengths(&input_lens).unwrap();
        output_lens.eval().unwrap();
        let values: &[i32] = output_lens.as_slice();

        let expected = [2, 3, 4];
        assert_eq!(values, expected, "Edge case lengths mismatch");
    }

    // ==================== Log Zero Guard Tests ====================

    #[test]
    fn test_log_guard_value() {
        // NeMo uses 2^-24 as the log zero guard
        let guard = 2.0_f32.powi(-24);
        let expected = 5.960464477539063e-08_f32;
        assert!(
            approx_eq(guard, expected, 1e-15),
            "Log guard should be 2^-24 = {}, got {}",
            expected,
            guard
        );
    }

    #[test]
    fn test_log_guard_add_behavior() {
        // NeMo uses log(x + guard), not log(max(x, guard))
        // This makes a difference for small positive values
        let guard = 2.0_f32.powi(-24);

        // For x = 0: log(0 + guard) vs log(max(0, guard))
        // Both should give the same result
        let x = 0.0_f32;
        let log_add = (x + guard).ln();
        let log_max = guard.max(x).ln();
        assert!(
            approx_eq(log_add, log_max, 1e-6),
            "For x=0, both methods should be equal"
        );

        // For x = guard/2: log(guard/2 + guard) != log(max(guard/2, guard))
        let x = guard / 2.0;
        let log_add = (x + guard).ln();
        let log_max = guard.max(x).ln();
        assert!(
            !approx_eq(log_add, log_max, 1e-6),
            "For x=guard/2, methods should differ: add={}, max={}",
            log_add,
            log_max
        );
    }

    // ==================== Per-Feature Normalization Tests ====================

    #[test]
    fn test_normalize_output_shape() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Create synthetic input [batch, n_mels, frames]
        let input = Array::from_slice(&vec![1.0f32; 2 * 80 * 100], &[2, 80, 100]);
        let output = mel_spec.normalize_per_feature(&input).unwrap();

        assert_eq!(output.shape(), &[2, 80, 100], "Shape should be preserved");
    }

    #[test]
    fn test_normalize_zero_mean() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Create synthetic input with non-zero mean
        let mut data = Vec::with_capacity(80 * 100);
        for i in 0..80 {
            for j in 0..100 {
                data.push((i as f32) + (j as f32) * 0.1);
            }
        }
        let input = Array::from_slice(&data, &[1, 80, 100]);
        let output = mel_spec.normalize_per_feature(&input).unwrap();
        output.eval().unwrap();

        // Compute mean along time axis for each feature
        let mean = mlx_rs::ops::mean_axis(&output, 2, false).unwrap();
        mean.eval().unwrap();
        let mean_values: &[f32] = mean.as_slice();

        // All means should be approximately 0
        for (i, &m) in mean_values.iter().enumerate() {
            assert!(
                approx_eq(m, 0.0, 1e-4),
                "Mean of feature {} should be ~0, got {}",
                i,
                m
            );
        }
    }

    #[test]
    fn test_normalize_unit_variance() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Create synthetic input with varying variance
        let mut data = Vec::with_capacity(80 * 100);
        for i in 0..80 {
            for j in 0..100 {
                // Use a pattern that creates different variances per feature
                data.push(((i * 100 + j) as f32).sin() * (i as f32 + 1.0));
            }
        }
        let input = Array::from_slice(&data, &[1, 80, 100]);
        let output = mel_spec.normalize_per_feature(&input).unwrap();
        output.eval().unwrap();

        // Compute variance along time axis for each feature
        let mean = mlx_rs::ops::mean_axis(&output, 2, true).unwrap();
        let centered = &output - &mean;
        let centered_sq = &centered * &centered;
        let var = mlx_rs::ops::mean_axis(&centered_sq, 2, false).unwrap();
        var.eval().unwrap();
        let var_values: &[f32] = var.as_slice();

        // All variances should be approximately 1
        for (i, &v) in var_values.iter().enumerate() {
            assert!(
                approx_eq(v, 1.0, 0.1),
                "Variance of feature {} should be ~1, got {}",
                i,
                v
            );
        }
    }

    // ==================== STFT Tests ====================

    #[test]
    fn test_stft_output_shape() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // 16000 samples -> 101 frames, 257 freq bins
        let audio = Array::from_slice(&vec![0.0f32; 16000], &[1, 16000]);
        let stft_result = mel_spec.stft(&audio).unwrap();

        // Output should be [batch, freq_bins, frames]
        // freq_bins = n_fft/2 + 1 = 257
        // frames = floor(16000/160) + 1 = 101
        assert_eq!(stft_result.shape()[0], 1, "Batch size should be 1");
        assert_eq!(stft_result.shape()[1], 257, "Should have 257 freq bins");
        assert_eq!(stft_result.shape()[2], 101, "Should have 101 frames");
    }

    #[test]
    fn test_stft_batch() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Batch of 2
        let audio = Array::from_slice(&vec![0.0f32; 2 * 16000], &[2, 16000]);
        let stft_result = mel_spec.stft(&audio).unwrap();

        assert_eq!(stft_result.shape()[0], 2, "Batch size should be 2");
    }

    #[test]
    fn test_stft_sine_wave_peak() {
        // A pure sine wave should have energy concentrated at its frequency
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let sr = 16000.0;
        let freq = 440.0; // A4 note
        let duration = 0.1;
        let n_samples = (sr * duration) as usize;

        // Generate sine wave
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / sr).sin())
            .collect();

        let audio = Array::from_slice(&samples, &[1, n_samples as i32]);
        let stft_result = mel_spec.stft(&audio).unwrap();

        // Compute power spectrum
        let power = mel_spec.complex_abs_squared(&stft_result).unwrap();
        power.eval().unwrap();

        // Average power across frames
        let avg_power = mlx_rs::ops::mean_axis(&power, 2, false).unwrap();
        avg_power.eval().unwrap();
        let power_values: &[f32] = avg_power.as_slice();

        // Find the bin with maximum power
        let (max_bin, _) = power_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Expected bin for 440 Hz: round(440 * 512 / 16000) = 14
        let expected_bin = (freq * 512.0 / sr).round() as usize;

        // Allow for some tolerance (within 2 bins)
        assert!(
            (max_bin as i32 - expected_bin as i32).abs() <= 2,
            "Peak should be near bin {} (440 Hz), got bin {}",
            expected_bin,
            max_bin
        );
    }

    // ==================== Full Pipeline Tests ====================

    #[test]
    fn test_forward_output_shape() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let audio = Array::from_slice(&vec![0.1f32; 16000], &[1, 16000]);
        let (features, lengths) = mel_spec.forward(&audio, None).unwrap();

        // Output should be [batch, n_mels, frames]
        assert_eq!(features.shape()[0], 1, "Batch size should be 1");
        assert_eq!(features.shape()[1], 80, "Should have 80 mel features");
        assert_eq!(features.shape()[2], 101, "Should have 101 frames");

        lengths.eval().unwrap();
        let len_values: &[i32] = lengths.as_slice();
        assert_eq!(len_values[0], 101, "Length should be 101");
    }

    #[test]
    fn test_forward_1d_input() {
        // Single audio without batch dimension should work
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let audio = Array::from_slice(&vec![0.1f32; 16000], &[16000]);
        let (features, _) = mel_spec.forward(&audio, None).unwrap();

        // Should automatically add batch dimension
        assert_eq!(features.shape()[0], 1, "Should add batch dimension");
        assert_eq!(features.shape()[1], 80, "Should have 80 mel features");
    }

    #[test]
    fn test_forward_deterministic() {
        // With dither=0, output should be deterministic
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Use a simple pattern, not zeros (zeros could cause numerical issues)
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 1600]);

        let (features1, _) = mel_spec.forward(&audio, None).unwrap();
        let (features2, _) = mel_spec.forward(&audio, None).unwrap();

        features1.eval().unwrap();
        features2.eval().unwrap();

        let values1: &[f32] = features1.as_slice();
        let values2: &[f32] = features2.as_slice();

        assert!(
            arrays_approx_eq(values1, values2, 1e-6),
            "Output should be deterministic with dither=0"
        );
    }

    #[test]
    fn test_forward_with_lengths() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        let audio = Array::from_slice(&vec![0.1f32; 2 * 16000], &[2, 16000]);
        let lengths = Array::from_slice(&[16000i32, 8000], &[2]);

        let (_, output_lengths) = mel_spec.forward(&audio, Some(&lengths)).unwrap();
        output_lengths.eval().unwrap();
        let len_values: &[i32] = output_lengths.as_slice();

        // 16000 -> 101, 8000 -> 51
        assert_eq!(len_values[0], 101);
        assert_eq!(len_values[1], 51);
    }

    #[test]
    fn test_forward_batch_consistency() {
        // Processing as batch should match processing individually
        // This matches NeMo's test_AudioToMelSpectrogramPreprocessor_batch
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();

        // Generate two different audio samples
        let samples1: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.002).sin()).collect();
        let samples2: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.003).cos()).collect();

        // Process individually
        let audio1 = Array::from_slice(&samples1, &[1, 8000]);
        let audio2 = Array::from_slice(&samples2, &[1, 8000]);
        let (feat1, _) = mel_spec.forward(&audio1, None).unwrap();
        let (feat2, _) = mel_spec.forward(&audio2, None).unwrap();

        // Process as batch
        let mut batch_samples = samples1.clone();
        batch_samples.extend(&samples2);
        let audio_batch = Array::from_slice(&batch_samples, &[2, 8000]);
        let (feat_batch, _) = mel_spec.forward(&audio_batch, None).unwrap();

        feat1.eval().unwrap();
        feat2.eval().unwrap();
        feat_batch.eval().unwrap();

        let values1: &[f32] = feat1.as_slice();
        let values2: &[f32] = feat2.as_slice();
        let batch_values: &[f32] = feat_batch.as_slice();

        // First half of batch should match first individual
        let n_elements = values1.len();
        assert!(
            arrays_approx_eq(values1, &batch_values[..n_elements], 1e-5),
            "Batch[0] should match individual[0]"
        );

        // Second half of batch should match second individual
        // Use a more lenient tolerance for batch processing differences
        let batch1_slice = &batch_values[n_elements..];
        let max_diff = values2
            .iter()
            .zip(batch1_slice.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            arrays_approx_eq(values2, batch1_slice, 1e-4),
            "Batch[1] should match individual[1], max diff: {}",
            max_diff
        );
    }

    #[test]
    fn test_output_dim() {
        let config = test_config();
        let mel_spec = MelSpectrogram::new(config).unwrap();
        assert_eq!(mel_spec.output_dim(), 80);
    }
}
