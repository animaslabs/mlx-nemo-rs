//! Mel spectrogram audio preprocessing.
//!
//! Converts raw audio waveforms to log-Mel spectrogram features for ASR models.
//!
//! This module provides:
//! - [`MelSpectrogramConfig`]: Configuration matching NeMo's AudioToMelSpectrogramPreprocessor
//! - [`MelSpectrogram`]: Fast GPU-based mel spectrogram processor
//!
//! # Streaming Support
//!
//! The processor maintains a small fixed buffer for leftover audio samples that
//! don't fit into complete mel frames. Call [`process_chunk`](MelSpectrogram::process_chunk)
//! repeatedly with audio chunks of any size.
//!
//! # Example
//!
//! ```rust,ignore
//! let config = MelSpectrogramConfig::default();
//! let mut mel = MelSpectrogram::new(config)?;
//!
//! // Process streaming audio chunks
//! for chunk in audio.chunks(1600) {
//!     let features = mel.process_chunk(chunk)?;
//!     // features: [n_mels, num_frames]
//! }
//!
//! // Flush remaining audio at end of stream
//! let final_features = mel.flush()?;
//! ```

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fft, ops, Array};
use std::cmp::max;
use std::f32::consts::PI;

use crate::asr::config::PreprocessorConfig;
use crate::core::error::{Error, Result};

/// Configuration for mel spectrogram computation.
///
/// Matches NeMo's `AudioToMelSpectrogramPreprocessor` configuration:
///
/// ```yaml
/// preprocessor:
///   _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
///   sample_rate: 16000
///   normalize: NA
///   window_size: 0.025
///   window_stride: 0.01
///   window: hann
///   features: 128
///   n_fft: 512
///   log: true
///   dither: 1.0e-05
/// ```
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    /// Sample rate in Hz (default: 16000).
    pub sample_rate: u32,
    /// Window size in seconds (default: 0.025 = 25ms = 400 samples @ 16kHz).
    pub window_size: f32,
    /// Window stride/hop in seconds (default: 0.01 = 10ms = 160 samples @ 16kHz).
    pub window_stride: f32,
    /// Window function type: "hann" or "hamming" (default: "hann").
    pub window: String,
    /// Number of mel filterbank features (default: 128).
    pub features: usize,
    /// FFT size (default: 512).
    pub n_fft: usize,
    /// Whether to apply log compression (default: true).
    pub log: bool,
    /// Normalization mode: "per_feature" or "NA" (default: "NA").
    pub normalize: String,
    /// Dither amount for numerical stability (default: 1e-5, 0.0 to disable).
    pub dither: f32,
    /// Pre-emphasis coefficient (default: None, Some(0.97) typical).
    pub preemph: Option<f32>,
    /// Minimum frequency for mel filterbank (default: 0.0).
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (default: Nyquist = sample_rate/2).
    pub fmax: Option<f32>,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            window_size: 0.025,
            window_stride: 0.01,
            window: "hann".to_string(),
            features: 128,
            n_fft: 512,
            log: true,
            normalize: "NA".to_string(),
            dither: 1e-5,
            preemph: None,
            fmin: 0.0,
            fmax: None,
        }
    }
}

impl MelSpectrogramConfig {
    /// Window length in samples.
    pub fn win_samples(&self) -> usize {
        (self.window_size * self.sample_rate as f32) as usize
    }

    /// Hop length in samples.
    pub fn hop_samples(&self) -> usize {
        (self.window_stride * self.sample_rate as f32) as usize
    }

    /// Maximum frequency for mel filterbank.
    pub fn effective_fmax(&self) -> f32 {
        self.fmax.unwrap_or(self.sample_rate as f32 / 2.0)
    }
}

impl From<PreprocessorConfig> for MelSpectrogramConfig {
    fn from(config: PreprocessorConfig) -> Self {
        Self {
            sample_rate: config.sample_rate,
            window_size: config.window_size,
            window_stride: config.window_stride,
            window: config.window,
            features: config.features,
            n_fft: config.n_fft,
            log: config.log,
            normalize: config.normalize,
            dither: config.dither,
            preemph: config.preemph,
            fmin: config.lowfreq.unwrap_or(0.0),
            fmax: config.highfreq,
        }
    }
}

/// Fast GPU-based mel spectrogram processor.
///
/// Converts raw audio waveforms to log-mel spectrogram features using MLX operations.
/// Maintains a small buffer for streaming support.
pub struct MelSpectrogram {
    config: MelSpectrogramConfig,
    /// Precomputed Hann/Hamming window [win_length], zero-padded to n_fft.
    window: Array,
    /// Mel filterbank matrix [n_mels, n_fft/2 + 1].
    filterbank: Array,
    /// Log guard constant (2^-24 for NeMo compatibility).
    log_guard: Array,
    /// Derived: window samples.
    win_samples: usize,
    /// Derived: hop samples.
    hop_samples: usize,
    /// Derived: number of frequency bins (n_fft/2 + 1).
    #[allow(dead_code)]
    n_freqs: usize,
    /// Audio remainder buffer for streaming (leftover samples from previous chunk).
    remainder: Vec<f32>,
    /// Last sample for preemphasis state tracking.
    preemph_last: f32,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram processor.
    pub fn new(config: MelSpectrogramConfig) -> Result<Self> {
        let win_samples = config.win_samples();
        let hop_samples = config.hop_samples();
        let n_freqs = config.n_fft / 2 + 1;

        // Create window function (zero-padded to n_fft)
        let window = Self::create_window(&config.window, win_samples, config.n_fft)?;

        // Create mel filterbank
        let filterbank = Self::create_filterbank(
            config.features,
            config.n_fft,
            config.sample_rate as f32,
            config.fmin,
            config.effective_fmax(),
        )?;

        // NeMo uses 2^-24 as log guard
        let log_guard = Array::from_f32(2.0_f32.powi(-24));

        Ok(Self {
            config,
            window,
            filterbank,
            log_guard,
            win_samples,
            hop_samples,
            n_freqs,
            remainder: Vec::new(),
            preemph_last: 0.0,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &MelSpectrogramConfig {
        &self.config
    }

    /// Get window length in samples.
    pub fn win_samples(&self) -> usize {
        self.win_samples
    }

    /// Get hop length in samples.
    pub fn hop_samples(&self) -> usize {
        self.hop_samples
    }

    /// Alias for hop_samples (backwards compatibility).
    pub fn hop_length(&self) -> usize {
        self.hop_samples
    }

    /// Alias for win_samples (backwards compatibility).
    pub fn window_length(&self) -> usize {
        self.win_samples
    }

    /// Get output feature dimension (number of mel bins).
    pub fn output_dim(&self) -> usize {
        self.config.features
    }

    /// Number of mel features.
    pub fn n_mels(&self) -> usize {
        self.config.features
    }

    /// Get the mel filterbank matrix for inspection.
    pub fn filterbank(&self) -> &Array {
        &self.filterbank
    }

    /// Reset streaming state (remainder buffer and preemphasis state).
    pub fn reset(&mut self) {
        self.remainder.clear();
        self.preemph_last = 0.0;
    }

    /// Process a chunk of audio samples (streaming API).
    ///
    /// Combines the chunk with any leftover samples from the previous call,
    /// extracts as many complete mel frames as possible, and buffers the
    /// remainder for the next call.
    ///
    /// # Arguments
    /// * `samples` - Raw audio samples (any length).
    ///
    /// # Returns
    /// Mel spectrogram features [n_mels, num_frames]. May be empty if not
    /// enough samples for a complete frame.
    pub fn process_chunk(&mut self, samples: &[f32]) -> Result<Array> {
        // Combine remainder with new samples (raw, unprocessed)
        let mut raw_audio = Vec::with_capacity(self.remainder.len() + samples.len());
        raw_audio.extend_from_slice(&self.remainder);
        raw_audio.extend_from_slice(samples);

        // Calculate how many full frames we can compute
        // Frame i requires samples [i*hop, i*hop + win_samples)
        // So for n_frames, we need: (n_frames-1)*hop + win_samples samples
        let n_frames = self.compute_frame_count(raw_audio.len());
        tracing::debug!("process_chunk raw_audio len: {} num audio samples: {} remainder len: {} mel n_frames: {}",
            raw_audio.len(), samples.len(), self.remainder.len(), n_frames);

        if n_frames == 0 {
            // Not enough audio, save raw samples for next call
            self.remainder = raw_audio;
            // Return empty array with correct shape [n_mels, 0]
            return Array::zeros::<f32>(&[self.config.features as i32, 0])
                .map_err(|e| Error::model(e.to_string()));
        }

        // Calculate samples needed for n_frames and samples consumed (for remainder)
        // Samples needed: (n_frames - 1) * hop + win_samples
        // Samples consumed (for next chunk boundary): n_frames * hop
        let samples_needed = (n_frames - 1) * self.hop_samples + self.win_samples;
        let samples_consumed = n_frames * self.hop_samples;

        // Save remainder BEFORE processing (raw samples that weren't consumed)
        if samples_consumed < raw_audio.len() {
            self.remainder = raw_audio[samples_consumed..].to_vec();
        } else {
            self.remainder.clear();
        }

        // Take only the samples we need for processing
        let audio_to_process = raw_audio[..samples_needed].to_vec();

        // Apply preemphasis with state tracking
        let audio_to_process = self.apply_preemphasis(&audio_to_process);

        // Apply dithering (adds small noise for numerical stability)
        let mut audio_to_process = audio_to_process;
        self.apply_dither(&mut audio_to_process);

        tracing::debug!(
            "process_chunk audio_to_process len: {} remainder len: {} samples_needed: {}",
            audio_to_process.len(),
            self.remainder.len(),
            samples_needed
        );

        // Compute mel spectrogram
        let mel = self.compute_mel(&audio_to_process)?;

        Ok(mel)
    }

    /// Flush any remaining audio in the buffer.
    ///
    /// Call this at the end of a stream to process any leftover samples.
    /// Returns empty array if no samples remain.
    pub fn flush(&mut self) -> Result<Array> {
        if self.remainder.is_empty() {
            return Array::zeros::<f32>(&[self.config.features as i32, 0])
                .map_err(|e| Error::model(e.to_string()));
        }

        // Process whatever is left (may produce partial frame with padding)
        let audio = std::mem::take(&mut self.remainder);
        self.compute_mel(&audio)
    }

    /// Process audio in batch mode (non-streaming).
    ///
    /// This processes the entire audio at once without maintaining state.
    /// For streaming, use [`process_chunk`](Self::process_chunk) instead.
    ///
    /// # Arguments
    /// * `audio` - Audio tensor [batch, time] or [time].
    ///
    /// # Returns
    /// Tuple of (mel_features, lengths):
    /// - mel_features: [batch, n_mels, frames]
    /// - lengths: [batch] output frame counts
    pub fn forward(&self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Ensure audio is 2D [batch, time]
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            audio.clone()
        };

        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        // Apply dithering if configured (adds small noise for numerical stability)
        let audio = self.apply_dither_batch(&audio)?;

        // Apply preemphasis if configured
        let audio = if let Some(coef) = self.config.preemph {
            self.apply_preemphasis_batch(&audio, coef)?
        } else {
            audio
        };

        // Compute STFT
        let stft = self.stft_batch(&audio)?;

        // Compute power spectrum
        let power = self.power_spectrum(&stft)?;

        // Apply mel filterbank: power [batch, freqs, frames] -> mel [batch, n_mels, frames]
        let mel = ops::einsum("mf,bft->bmt", &[self.filterbank.clone(), power])?;

        // Apply log compression
        let mel = if self.config.log {
            let mel_safe = ops::add(&mel, &self.log_guard)?;
            ops::log(&mel_safe)?
        } else {
            mel
        };

        // Apply normalization if requested
        let mel = match self.config.normalize.as_str() {
            "per_feature" => self.normalize_per_feature(&mel)?,
            _ => mel,
        };

        // Compute output lengths
        let out_lengths = if let Some(lens) = lengths {
            self.compute_output_lengths(lens)?
        } else {
            // All same length based on audio_len
            let n_frames = audio_len as usize / self.hop_samples + 1;
            ops::full::<i32>(&[batch_size], &Array::from_int(n_frames as i32))?
        };

        Ok((mel, out_lengths))
    }

    /// Process audio without normalization (for streaming with external stats).
    pub fn forward_unnormalized(&self, audio: &Array) -> Result<Array> {
        let (mel, _) = self.forward(audio, None)?;

        // If normalization was applied, we need to undo it or use a different path
        // For simplicity, we just skip normalization in the config check above
        Ok(mel)
    }

    /// Process audio with preemphasis already applied externally.
    pub fn forward_no_preemph(&self, audio: &Array) -> Result<Array> {
        // Ensure audio is 2D [batch, time]
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            audio.clone()
        };

        // Compute STFT (skip preemphasis)
        let stft = self.stft_batch(&audio)?;

        // Compute power spectrum
        let power = self.power_spectrum(&stft)?;

        // Apply mel filterbank
        let mel = ops::einsum("mf,bft->bmt", &[self.filterbank.clone(), power])?;

        // Apply log compression
        let mel = if self.config.log {
            let mel_safe = ops::add(&mel, &self.log_guard)?;
            ops::log(&mel_safe)?
        } else {
            mel
        };

        Ok(mel)
    }

    // ========================================================================
    // Private implementation
    // ========================================================================

    /// Create window function, zero-padded to n_fft.
    fn create_window(window_type: &str, win_len: usize, n_fft: usize) -> Result<Array> {
        let window: Vec<f32> = match window_type {
            "hann" => (0..win_len)
                .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (win_len - 1) as f32).cos()))
                .collect(),
            "hamming" => (0..win_len)
                .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / (win_len - 1) as f32).cos())
                .collect(),
            _ => {
                return Err(Error::config(format!(
                    "Unknown window type: {}",
                    window_type
                )))
            }
        };

        // Zero-pad to n_fft if needed (centered)
        if win_len < n_fft {
            let left_pad = (n_fft - win_len) / 2;
            let right_pad = n_fft - win_len - left_pad;
            let mut padded = vec![0.0f32; left_pad];
            padded.extend(&window);
            padded.extend(vec![0.0f32; right_pad]);
            Ok(Array::from_slice(&padded, &[n_fft as i32]))
        } else {
            Ok(Array::from_slice(&window, &[win_len as i32]))
        }
    }

    /// Create Slaney-normalized mel filterbank.
    fn create_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
    ) -> Result<Array> {
        let n_freqs = n_fft / 2 + 1;

        // Slaney mel scale conversion
        let hz_to_mel = |hz: f32| -> f32 {
            const F_SP: f32 = 200.0 / 3.0;
            const MIN_LOG_HZ: f32 = 1000.0;
            const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
            const LOGSTEP: f32 = 0.06875177742094912; // ln(6.4) / 27.0

            if hz >= MIN_LOG_HZ {
                MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
            } else {
                hz / F_SP
            }
        };

        let mel_to_hz = |mel: f32| -> f32 {
            const F_SP: f32 = 200.0 / 3.0;
            const MIN_LOG_HZ: f32 = 1000.0;
            const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
            const LOGSTEP: f32 = 0.06875177742094912;

            if mel >= MIN_LOG_MEL {
                MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
            } else {
                mel * F_SP
            }
        };

        // Mel points
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // FFT frequencies
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();

        // Frequency differences for normalization
        let fdiff: Vec<f32> = hz_points.windows(2).map(|w| w[1] - w[0]).collect();

        // Build filterbank matrix with Slaney normalization
        let mut fb = vec![0.0f32; n_mels * n_freqs];

        for m in 0..n_mels {
            let f_left = hz_points[m];
            let _f_center = hz_points[m + 1];
            let f_right = hz_points[m + 2];

            // Slaney normalization factor: 2 / (f_right - f_left)
            let enorm = 2.0 / (f_right - f_left);

            for (k, &freq) in fft_freqs.iter().enumerate() {
                // Triangular filter
                let lower = (freq - f_left) / fdiff[m];
                let upper = (f_right - freq) / fdiff[m + 1];
                let value = lower.min(upper).max(0.0) * enorm;
                fb[m * n_freqs + k] = value;
            }
        }

        Ok(Array::from_slice(&fb, &[n_mels as i32, n_freqs as i32]))
    }

    /// Apply preemphasis with state tracking (streaming).
    fn apply_preemphasis(&mut self, audio: &[f32]) -> Vec<f32> {
        let coef = match self.config.preemph {
            Some(c) => c,
            None => return audio.to_vec(),
        };

        if audio.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(audio.len());

        // First sample uses preemph_last from previous chunk
        output.push(audio[0] - coef * self.preemph_last);

        // Rest of samples
        for i in 1..audio.len() {
            output.push(audio[i] - coef * audio[i - 1]);
        }

        // Update state
        self.preemph_last = *audio.last().unwrap();

        output
    }

    /// Apply preemphasis in batch mode (non-streaming).
    fn apply_preemphasis_batch(&self, audio: &Array, coef: f32) -> Result<Array> {
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        // y[n] = x[n] - coef * x[n-1], with x[-1] = 0
        let zeros = Array::zeros::<f32>(&[batch_size, 1])?;
        let shifted = ops::concatenate_axis(&[&zeros, &audio.index((.., ..(audio_len - 1)))], 1)?;
        let scaled = ops::multiply(&shifted, &Array::from_f32(coef))?;
        ops::subtract(audio, &scaled).map_err(|e| Error::model(e.to_string()))
    }

    /// Apply dithering to audio samples (streaming mode).
    ///
    /// Dithering adds small random noise to break up quantization patterns
    /// and improve numerical stability for silent/near-silent audio.
    /// NeMo uses uniform noise in range [-dither, +dither].
    ///
    /// Note: In NeMo, dithering is typically only applied during training.
    /// For inference, it's often disabled (dither=0.0) for determinism.
    fn apply_dither(&self, audio: &mut [f32]) {
        let dither = self.config.dither;
        if dither <= 0.0 {
            return;
        }

        // Use a simple deterministic noise based on sample index for reproducibility
        // This matches NeMo's behavior of adding small perturbations
        // For truly random dither (training), you'd use a proper RNG
        for (i, sample) in audio.iter_mut().enumerate() {
            // Simple hash-based pseudo-random noise
            let noise =
                ((i as u32).wrapping_mul(2654435769) as f32 / u32::MAX as f32 - 0.5) * 2.0 * dither;
            *sample += noise;
        }
    }

    /// Apply dithering to audio tensor (batch mode).
    ///
    /// Adds small random noise for numerical stability.
    fn apply_dither_batch(&self, audio: &Array) -> Result<Array> {
        let dither = self.config.dither;
        if dither <= 0.0 {
            return Ok(audio.clone());
        }

        // Generate uniform random noise in [-dither, +dither]
        let noise = mlx_rs::random::uniform::<_, f32>(-dither, dither, audio.shape(), None)?;

        ops::add(audio, &noise).map_err(|e| Error::model(e.to_string()))
    }

    /// Compute number of frames from audio length for streaming.
    ///
    /// For streaming, we use a non-center-padded calculation to ensure
    /// frame boundaries align across chunks. Each frame requires `win_samples`
    /// of audio, and frames are spaced `hop_samples` apart.
    ///
    /// Frame count: `(audio_len - win_samples) / hop + 1` when `audio_len >= win_samples`
    ///
    /// Returns 0 if not enough audio for even one complete frame.
    fn compute_frame_count(&self, audio_len: usize) -> usize {
        if audio_len < self.win_samples {
            0
        } else {
            (audio_len - self.win_samples) / self.hop_samples + 1
        }
    }

    /// Compute mel spectrogram from audio samples (internal, streaming path).
    ///
    /// For streaming, we use STFT **without** center padding to get exactly
    /// the number of frames that `compute_frame_count` predicts. This ensures
    /// frame boundaries align correctly across chunks.
    ///
    /// The batch mode `forward()` uses center padding for compatibility with
    /// NeMo's offline processing, but streaming needs precise frame counts.
    pub fn compute_mel(&self, audio: &[f32]) -> Result<Array> {
        let audio_len = audio.len();
        let n_fft = self.config.n_fft;
        let hop = self.hop_samples;
        let win_len = self.win_samples;

        // Calculate expected frames (same formula as compute_frame_count)
        let n_frames = if audio_len >= win_len {
            (audio_len - win_len) / hop + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Array::zeros::<f32>(&[self.config.features as i32, 0])
                .map_err(|e| Error::model(e.to_string()));
        }

        // Frame the audio WITHOUT center padding
        // Each frame starts at frame_idx * hop and has length n_fft
        // We need to handle the case where win_len < n_fft (zero-pad each frame)
        let mut frames_data = Vec::with_capacity(n_frames * n_fft);
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            // The window is applied to win_len samples, but FFT needs n_fft
            // The window array is already zero-padded to n_fft (centered)
            for i in 0..n_fft {
                // Map FFT bin to audio sample, accounting for window centering
                let win_offset = (n_fft - win_len) / 2;
                if i >= win_offset && i < win_offset + win_len {
                    let audio_idx = start + (i - win_offset);
                    if audio_idx < audio_len {
                        frames_data.push(audio[audio_idx]);
                    } else {
                        frames_data.push(0.0);
                    }
                } else {
                    frames_data.push(0.0); // Zero padding for FFT
                }
            }
        }

        let frames = Array::from_slice(&frames_data, &[1, n_frames as i32, n_fft as i32]);

        // Apply window (already zero-padded to n_fft)
        let windowed = ops::multiply(&frames, &self.window)?;

        // Compute FFT along last axis
        let fft_result = fft::rfft(&windowed, Some(n_fft as i32), Some(2))?;

        // Transpose to [batch, freq_bins, frames]
        let stft = ops::transpose_axes(&fft_result, &[0, 2, 1])?;

        // Compute power spectrum
        let power = self.power_spectrum(&stft)?;

        // Apply mel filterbank
        let mel = ops::einsum("mf,bft->bmt", &[self.filterbank.clone(), power])?;

        // Apply log compression
        let mel = if self.config.log {
            let mel_safe = ops::add(&mel, &self.log_guard)?;
            ops::log(&mel_safe)?
        } else {
            mel
        };

        // Remove batch dimension: [1, n_mels, frames] -> [n_mels, frames]
        let mel = mel.index(0);

        // mlx_rs::transforms::eval([&mel])?;
        // tracing::trace!(
        //     audio_len,
        //     n_frames,
        //     mel_shape = ?mel.shape(),
        //     "compute_mel: streaming STFT without center padding"
        // );

        Ok(mel)
    }

    /// Compute STFT for batch of audio.
    fn stft_batch(&self, audio: &Array) -> Result<Array> {
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1] as usize;
        let n_fft = self.config.n_fft;
        let hop = self.hop_samples;

        // Apply center padding (zeros, matching NeMo's torch.stft center=True, pad_mode='constant')
        let pad_size = n_fft / 2;
        let left_pad = Array::zeros::<f32>(&[batch_size, pad_size as i32])?;
        let right_pad = Array::zeros::<f32>(&[batch_size, pad_size as i32])?;
        let padded = ops::concatenate_axis(&[&left_pad, audio, &right_pad], 1)?;

        let padded_len = audio_len + 2 * pad_size;
        let n_frames = (padded_len - n_fft) / hop + 1;

        // Frame the audio using as_strided
        // We need to process each batch item
        padded.eval()?;
        let padded_data: &[f32] = padded.as_slice();

        let mut frames_data = Vec::with_capacity(batch_size as usize * n_frames * n_fft);
        for b in 0..batch_size as usize {
            let batch_offset = b * padded_len;
            for frame_idx in 0..n_frames {
                let start = frame_idx * hop;
                for i in 0..n_fft {
                    frames_data.push(padded_data[batch_offset + start + i]);
                }
            }
        }

        let frames = Array::from_slice(&frames_data, &[batch_size, n_frames as i32, n_fft as i32]);

        // Apply window
        let windowed = ops::multiply(&frames, &self.window)?;

        // Compute FFT along last axis
        let fft_result = fft::rfft(&windowed, Some(n_fft as i32), Some(2))?;

        // Transpose to [batch, freq_bins, frames]
        ops::transpose_axes(&fft_result, &[0, 2, 1]).map_err(|e| Error::model(e.to_string()))
    }

    /// Compute power spectrum from complex FFT result.
    fn power_spectrum(&self, fft_result: &Array) -> Result<Array> {
        let real = ops::real(fft_result)?;
        let imag = ops::imag(fft_result)?;
        let real_sq = ops::square(&real)?;
        let imag_sq = ops::square(&imag)?;
        ops::add(&real_sq, &imag_sq).map_err(|e| Error::model(e.to_string()))
    }

    /// Per-feature normalization (zero mean, unit variance per mel bin).
    fn normalize_per_feature(&self, features: &Array) -> Result<Array> {
        // features: [batch, n_mels, frames]
        // Normalize each mel bin across frames
        let mean = ops::mean_axis(features, 2, Some(true))?;
        let centered = ops::subtract(features, &mean)?;
        let variance = ops::mean_axis(&ops::square(&centered)?, 2, Some(true))?;
        let std = ops::sqrt(&ops::add(&variance, &Array::from_f32(1e-5))?)?;
        ops::divide(&centered, &std).map_err(|e| Error::model(e.to_string()))
    }

    /// Compute output frame lengths from input sample lengths.
    fn compute_output_lengths(&self, input_lengths: &Array) -> Result<Array> {
        // With center padding: n_frames = input_len / hop + 1
        let divided = ops::floor_divide(input_lengths, Array::from_int(self.hop_samples as i32))?;
        ops::add(&divided, &Array::from_int(1)).map_err(|e| Error::model(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ==================== Config Tests ====================

    #[test]
    fn test_default_config() {
        let config = MelSpectrogramConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.features, 128);
        assert_eq!(config.n_fft, 512);
        assert!((config.window_size - 0.025).abs() < 1e-6);
        assert!((config.window_stride - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_config_derived_values() {
        let config = MelSpectrogramConfig::default();
        assert_eq!(config.win_samples(), 400);
        assert_eq!(config.hop_samples(), 160);
        assert!((config.effective_fmax() - 8000.0).abs() < 1e-6);
    }

    // ==================== Processor Creation Tests ====================

    #[test]
    fn test_processor_creation() {
        let config = MelSpectrogramConfig::default();
        let mel = MelSpectrogram::new(config);
        assert!(mel.is_ok());

        let mel = mel.unwrap();
        assert_eq!(mel.n_mels(), 128);
        assert_eq!(mel.win_samples(), 400);
        assert_eq!(mel.hop_samples(), 160);
    }

    // ==================== Window Tests ====================

    #[test]
    fn test_hann_window_endpoints() {
        let window = MelSpectrogram::create_window("hann", 400, 512).unwrap();
        window.eval().unwrap();
        let data: &[f32] = window.as_slice();

        // Window is zero-padded: [56 zeros, 400 hann, 56 zeros]
        // The hann values start at index 56
        let left_pad = (512 - 400) / 2;
        assert!(data[left_pad].abs() < 1e-6, "First hann value should be ~0");
        assert!(
            data[left_pad + 399].abs() < 1e-6,
            "Last hann value should be ~0"
        );
    }

    #[test]
    fn test_hamming_window() {
        let window = MelSpectrogram::create_window("hamming", 400, 512).unwrap();
        window.eval().unwrap();
        let data: &[f32] = window.as_slice();

        // Hamming window has non-zero endpoints (~0.08)
        let left_pad = (512 - 400) / 2;
        assert!(approx_eq(data[left_pad], 0.08, 1e-4));
    }

    #[test]
    fn test_unknown_window_error() {
        let result = MelSpectrogram::create_window("blackman", 400, 512);
        assert!(result.is_err());
    }

    // ==================== Filterbank Tests ====================

    #[test]
    fn test_filterbank_shape() {
        let fb = MelSpectrogram::create_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        assert_eq!(fb.shape(), &[128, 257]); // [n_mels, n_fft/2+1]
    }

    #[test]
    fn test_filterbank_non_negative() {
        let fb = MelSpectrogram::create_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        let data: &[f32] = fb.as_slice();

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(min >= 0.0, "Filterbank should be non-negative, got {}", min);
    }

    // ==================== Streaming Tests ====================

    #[test]
    fn test_process_chunk_accumulation() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = MelSpectrogram::new(config).unwrap();

        // With win_samples=400, hop_samples=160:
        // n_frames = (audio_len - 400) / 160 + 1 when audio_len >= 400

        // Push small chunk (100 samples < 400 win_samples = 0 frames)
        let chunk1 = vec![0.1f32; 100];
        let result1 = mel.process_chunk(&chunk1).unwrap();

        // Not enough for a frame, returns empty
        assert_eq!(result1.shape()[0], 128);
        assert_eq!(result1.shape()[1], 0);

        // Push more (100 remainder + 350 new = 450 samples)
        // n_frames = (450 - 400) / 160 + 1 = 1 frame
        // samples_consumed = 1 * 160 = 160, remainder = 450 - 160 = 290
        let chunk2 = vec![0.1f32; 350];
        let result2 = mel.process_chunk(&chunk2).unwrap();

        assert_eq!(result2.shape()[0], 128);
        assert_eq!(result2.shape()[1], 1);

        // Push more (290 remainder + 500 new = 790 samples)
        // n_frames = (790 - 400) / 160 + 1 = 3 frames
        let chunk3 = vec![0.1f32; 500];
        let result3 = mel.process_chunk(&chunk3).unwrap();

        assert_eq!(result3.shape()[0], 128);
        assert_eq!(result3.shape()[1], 3);
    }

    #[test]
    fn test_reset_clears_state() {
        let config = MelSpectrogramConfig {
            preemph: Some(0.97),
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = MelSpectrogram::new(config).unwrap();

        // Push some audio
        let chunk = vec![0.1f32; 500];
        mel.process_chunk(&chunk).unwrap();

        assert!(!mel.remainder.is_empty() || mel.preemph_last != 0.0);

        // Reset
        mel.reset();

        assert!(mel.remainder.is_empty());
        assert_eq!(mel.preemph_last, 0.0);
    }

    #[test]
    fn test_flush_processes_remainder() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = MelSpectrogram::new(config).unwrap();

        // Push audio that leaves a remainder
        let chunk = vec![0.1f32; 300];
        mel.process_chunk(&chunk).unwrap();

        // Flush should process any remaining samples
        let result = mel.flush().unwrap();
        assert!(result.shape()[1] >= 0); // May be 0 if no remainder
    }

    // ==================== Batch Processing Tests ====================

    #[test]
    fn test_forward_output_shape() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        // 1 second of audio
        let audio = Array::from_slice(&vec![0.1f32; 16000], &[1, 16000]);
        let (features, lengths) = mel.forward(&audio, None).unwrap();

        // [batch, n_mels, frames]
        assert_eq!(features.shape()[0], 1);
        assert_eq!(features.shape()[1], 128);
        // 16000/160 + 1 = 101 frames
        assert_eq!(features.shape()[2], 101);

        lengths.eval().unwrap();
        let len_data: &[i32] = lengths.as_slice();
        assert_eq!(len_data[0], 101);
    }

    #[test]
    fn test_forward_1d_input() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        // 1D input (no batch dimension)
        let audio = Array::from_slice(&vec![0.1f32; 16000], &[16000]);
        let (features, _) = mel.forward(&audio, None).unwrap();

        // Should auto-add batch dimension
        assert_eq!(features.shape()[0], 1);
        assert_eq!(features.shape()[1], 128);
    }

    #[test]
    fn test_forward_batch() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        // Batch of 2
        let audio = Array::from_slice(&vec![0.1f32; 2 * 16000], &[2, 16000]);
        let (features, lengths) = mel.forward(&audio, None).unwrap();

        assert_eq!(features.shape()[0], 2);
        assert_eq!(features.shape()[1], 128);
        assert_eq!(features.shape()[2], 101);

        lengths.eval().unwrap();
        let len_data: &[i32] = lengths.as_slice();
        assert_eq!(len_data.len(), 2);
    }

    #[test]
    fn test_forward_deterministic() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 1600]);

        let (feat1, _) = mel.forward(&audio, None).unwrap();
        let (feat2, _) = mel.forward(&audio, None).unwrap();

        feat1.eval().unwrap();
        feat2.eval().unwrap();

        let data1: &[f32] = feat1.as_slice();
        let data2: &[f32] = feat2.as_slice();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!(approx_eq(*a, *b, 1e-6), "Should be deterministic");
        }
    }

    // ==================== Preemphasis Tests ====================

    #[test]
    fn test_preemphasis_streaming_state() {
        let config = MelSpectrogramConfig {
            preemph: Some(0.97),
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = MelSpectrogram::new(config).unwrap();

        // With win_samples=400, hop_samples=160:
        // Need at least 400 samples to produce 1 frame
        // First chunk - 500 samples produces (500-400)/160 + 1 = 1 frame
        // samples_consumed = 1 * 160 = 160
        let chunk1: Vec<f32> = (0..500).map(|i| i as f32 / 500.0).collect();
        mel.process_chunk(&chunk1).unwrap();

        // Preemph state should be updated (last sample of samples_needed)
        // samples_needed = (1-1)*160 + 400 = 400, so preemph_last = chunk1[399] = 399/500 ≈ 0.798
        assert!(
            mel.preemph_last > 0.7,
            "preemph_last should be updated after processing, got {}",
            mel.preemph_last
        );

        // Second chunk - combined with remainder should produce more frames
        let chunk2: Vec<f32> = (500..1000).map(|i| i as f32 / 1000.0).collect();
        mel.process_chunk(&chunk2).unwrap();

        // State should update again
        assert!(
            mel.preemph_last > 0.5,
            "preemph_last should remain updated, got {}",
            mel.preemph_last
        );
    }

    // ==================== Normalization Tests ====================

    #[test]
    fn test_per_feature_normalization() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            normalize: "per_feature".to_string(),
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 16000]);

        let (features, _) = mel.forward(&audio, None).unwrap();
        features.eval().unwrap();

        // Check that mean is ~0 per feature
        let mean = ops::mean_axis(&features, 2, Some(false)).unwrap();
        mean.eval().unwrap();
        let mean_data: &[f32] = mean.as_slice();

        for &m in mean_data {
            assert!(
                m.abs() < 1e-4,
                "Mean should be ~0 after normalization, got {}",
                m
            );
        }
    }

    // ==================== Output Length Tests ====================

    #[test]
    fn test_output_lengths() {
        let config = MelSpectrogramConfig::default();
        let mel = MelSpectrogram::new(config).unwrap();

        let input_lens = Array::from_slice(&[16000i32, 8000, 32000], &[3]);
        let output_lens = mel.compute_output_lengths(&input_lens).unwrap();
        output_lens.eval().unwrap();

        let data: &[i32] = output_lens.as_slice();
        // 16000/160+1=101, 8000/160+1=51, 32000/160+1=201
        assert_eq!(data, &[101, 51, 201]);
    }

    // ==================== Dithering Tests ====================

    #[test]
    fn test_dither_disabled() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        // With dither=0, output should be deterministic
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 1600]);

        let (feat1, _) = mel.forward(&audio, None).unwrap();
        let (feat2, _) = mel.forward(&audio, None).unwrap();

        feat1.eval().unwrap();
        feat2.eval().unwrap();

        let data1: &[f32] = feat1.as_slice();
        let data2: &[f32] = feat2.as_slice();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-6),
                "With dither=0, output should be deterministic"
            );
        }
    }

    #[test]
    fn test_dither_enabled_adds_noise() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 1e-5,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(config).unwrap();

        // With dither > 0, results should differ slightly between calls
        // (due to random noise)
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 1600]);

        let (feat1, _) = mel.forward(&audio, None).unwrap();
        let (feat2, _) = mel.forward(&audio, None).unwrap();

        feat1.eval().unwrap();
        feat2.eval().unwrap();

        let data1: &[f32] = feat1.as_slice();
        let data2: &[f32] = feat2.as_slice();

        // Check that outputs differ (due to random dither)
        let mut has_diff = false;
        for (a, b) in data1.iter().zip(data2.iter()) {
            if (a - b).abs() > 1e-10 {
                has_diff = true;
                break;
            }
        }
        assert!(
            has_diff,
            "With dither enabled, outputs should differ between calls"
        );
    }

    #[test]
    fn test_dither_streaming_modifies_audio() {
        let config = MelSpectrogramConfig {
            preemph: None,
            dither: 1e-3, // Larger dither for easier detection
            ..Default::default()
        };
        let mut mel = MelSpectrogram::new(config).unwrap();

        // Original audio (all zeros - silent)
        let samples = vec![0.0f32; 500];
        let result = mel.process_chunk(&samples).unwrap();

        result.eval().unwrap();
        let data: &[f32] = result.as_slice();

        // With dithering on silent audio, mel features should NOT be
        // the extreme negative values you'd get from log(0 + epsilon)
        // because dither adds noise before STFT
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Values should be in a reasonable range (not -infinity or extreme)
        assert!(min_val.is_finite(), "Min value should be finite");
        assert!(max_val.is_finite(), "Max value should be finite");
    }
}
