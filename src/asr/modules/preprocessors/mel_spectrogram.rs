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
        let mel_spec = &mel_spec + &Array::from_f32(5.960464477539063e-08);
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
        let mel_spec = &mel_spec + &Array::from_f32(5.960464477539063e-08);
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
            let f_center = hz_points[m + 1];
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
            for _ in 0..pad_size {
                padded_data.push(0.0);
            }

            // Original audio
            for i in 0..audio_len as usize {
                padded_data.push(audio_data[batch_offset + i]);
            }

            // Right padding: zeros
            for _ in 0..pad_size {
                padded_data.push(0.0);
            }
        }

        let padded_len = audio_len as usize + 2 * pad_size;
        let audio = Array::from_slice(&padded_data, &[batch_size as i32, padded_len as i32]);

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

        let frames = Array::from_slice(
            &frames_data,
            &[batch_size as i32, n_frames as i32, n_fft as i32],
        );

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

        let divided = mlx_rs::ops::floor_divide(input_lengths, &Array::from_int(hop_len))?;
        let output_lengths = &divided + &Array::from_int(1);
        Ok(output_lengths)
    }
}
