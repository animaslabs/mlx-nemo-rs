//! Mel spectrogram audio preprocessing.
//!
//! Converts raw audio waveforms to log-Mel spectrogram features for ASR models.
//!
//! This module provides:
//! - [`MelConfig`]: Configuration matching NeMo's AudioToMelSpectrogramPreprocessor
//! - [`MelProcessor`]: Cached, stateless mel spectrogram processor
//!
//! # Example
//!
//! ```rust,ignore
//! let config = MelConfig::default();
//! let mel = MelProcessor::new(config)?;
//!
//! // Process audio (stateless - no streaming state)
//! let (features, lengths) = mel.forward(&audio, None)?;
//! // features: [batch, n_mels, frames]
//! ```

use std::f32::consts::PI;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fft, ops, Array};

use crate::asr::config::PreprocessorConfig;
use crate::core::error::{Error, Result};

/// Configuration for mel spectrogram computation.
///
/// Matches NeMo's `AudioToMelSpectrogramPreprocessor` configuration.
#[derive(Debug, Clone)]
pub struct MelConfig {
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

impl Default for MelConfig {
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

impl MelConfig {
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

impl From<PreprocessorConfig> for MelConfig {
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

/// Cached, stateless mel spectrogram processor.
///
/// Caches the mel filterbank and window function for efficient repeated use.
/// This processor is **stateless** - it does not maintain any streaming buffers.
/// For streaming, use `StreamingMelProcessor` to handle audio buffering and
/// incremental mel computation.
pub struct MelProcessor {
    config: MelConfig,
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
}

impl MelProcessor {
    /// Create a new mel spectrogram processor.
    pub fn new(config: MelConfig) -> Result<Self> {
        let win_samples = config.win_samples();
        let hop_samples = config.hop_samples();

        // Create window function (zero-padded to n_fft)
        let window = Self::create_window_padded(&config.window, win_samples, config.n_fft)?;

        // Create mel filterbank
        let filterbank = Self::create_mel_filterbank(
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
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &MelConfig {
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

    /// Get the mel filterbank matrix.
    pub fn filterbank(&self) -> &Array {
        &self.filterbank
    }

    /// Calculate number of output frames for given audio length.
    ///
    /// With center padding: frames = audio_len / hop + 1
    pub fn num_frames(&self, audio_len: usize) -> usize {
        if audio_len == 0 {
            0
        } else {
            audio_len / self.hop_samples + 1
        }
    }

    /// Process audio to mel spectrogram (batch API).
    ///
    /// # Arguments
    /// * `audio` - Audio tensor [batch, time] or [time].
    /// * `lengths` - Optional input lengths [batch].
    ///
    /// # Returns
    /// Tuple of (mel_features, output_lengths):
    /// - mel_features: [batch, n_mels, frames]
    /// - output_lengths: [batch] frame counts
    pub fn forward(&self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Ensure audio is 2D [batch, time]
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            audio.clone()
        };

        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1] as usize;

        // Apply dithering if configured
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
        let n_frames = audio_len / self.hop_samples + 1;
        let out_lengths = if let Some(lens) = lengths {
            self.compute_output_lengths(lens)?
        } else {
            ops::full::<i32>(&[batch_size], &Array::from_int(n_frames as i32))?
        };

        Ok((mel, out_lengths))
    }

    // ========================================================================
    // Private: Window and Filterbank Creation
    // ========================================================================

    /// Create a window function.
    fn create_window(window_type: &str, length: usize) -> Result<Vec<f32>> {
        let n = length as f32;
        let values: Vec<f32> = match window_type {
            "hann" => (0..length)
                .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1.0)).cos()))
                .collect(),
            "hamming" => (0..length)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1.0)).cos())
                .collect(),
            _ => {
                return Err(Error::config(format!(
                    "Unknown window type: {}",
                    window_type
                )));
            }
        };
        Ok(values)
    }

    /// Create a window function with centered zero-padding to n_fft.
    fn create_window_padded(window_type: &str, win_length: usize, n_fft: usize) -> Result<Array> {
        let win_values = Self::create_window(window_type, win_length)?;

        if win_length >= n_fft {
            return Ok(Array::from_slice(&win_values, &[win_length as i32]));
        }

        // Center the window with zero-padding
        let left_pad = (n_fft - win_length) / 2;
        let mut padded = vec![0.0f32; n_fft];
        padded[left_pad..left_pad + win_length].copy_from_slice(&win_values);

        Ok(Array::from_slice(&padded, &[n_fft as i32]))
    }

    /// Create Slaney-normalized mel filterbank.
    ///
    /// Uses Slaney normalization (area normalization) to ensure consistent
    /// energy across mel bands, matching NeMo's torchaudio mel filterbank.
    fn create_mel_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        f_min: f32,
        f_max: f32,
    ) -> Result<Array> {
        let n_freqs = n_fft / 2 + 1;

        // Slaney mel scale constants
        const F_SP: f32 = 200.0 / 3.0;
        const MIN_LOG_HZ: f32 = 1000.0;
        const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
        const LOGSTEP: f32 = 0.06875177742094912; // ln(6.4) / 27.0

        let hz_to_mel = |hz: f32| -> f32 {
            if hz >= MIN_LOG_HZ {
                MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
            } else {
                hz / F_SP
            }
        };

        let mel_to_hz = |mel: f32| -> f32 {
            if mel >= MIN_LOG_MEL {
                MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
            } else {
                mel * F_SP
            }
        };

        // FFT frequencies: i * sample_rate / n_fft
        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();

        // Mel points
        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Frequency differences for normalization
        let fdiff: Vec<f32> = hz_points.windows(2).map(|w| w[1] - w[0]).collect();

        // Build filterbank with Slaney normalization
        let mut filterbank = vec![0.0f32; n_mels * n_freqs];

        for m in 0..n_mels {
            let f_left = hz_points[m];
            let f_right = hz_points[m + 2];

            // Slaney normalization factor: 2 / (f_right - f_left)
            let enorm = 2.0 / (f_right - f_left);

            for (k, &freq) in fft_freqs.iter().enumerate() {
                // Triangular filter with proper slopes
                let lower = (freq - f_left) / fdiff[m];
                let upper = (f_right - freq) / fdiff[m + 1];
                let value = lower.min(upper).max(0.0) * enorm;
                filterbank[m * n_freqs + k] = value;
            }
        }

        Ok(Array::from_slice(
            &filterbank,
            &[n_mels as i32, n_freqs as i32],
        ))
    }

    // ========================================================================
    // Private: Audio Processing
    // ========================================================================

    /// Apply preemphasis in batch mode.
    fn apply_preemphasis_batch(&self, audio: &Array, coef: f32) -> Result<Array> {
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1];

        // y[n] = x[n] - coef * x[n-1], with x[-1] = 0
        let zeros = Array::zeros::<f32>(&[batch_size, 1])?;
        let shifted = ops::concatenate_axis(&[&zeros, &audio.index((.., ..(audio_len - 1)))], 1)?;
        let scaled = ops::multiply(&shifted, &Array::from_f32(coef))?;
        ops::subtract(audio, &scaled).map_err(|e| Error::model(e.to_string()))
    }

    /// Apply dithering to audio tensor (batch mode).
    fn apply_dither_batch(&self, audio: &Array) -> Result<Array> {
        let dither = self.config.dither;
        if dither <= 0.0 {
            return Ok(audio.clone());
        }

        let noise = mlx_rs::random::uniform::<_, f32>(-dither, dither, audio.shape(), None)?;
        ops::add(audio, &noise).map_err(|e| Error::model(e.to_string()))
    }

    /// Compute STFT for batch of audio with center padding.
    fn stft_batch(&self, audio: &Array) -> Result<Array> {
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1] as usize;
        let n_fft = self.config.n_fft;
        let hop = self.hop_samples;

        // Apply center padding (zeros, matching NeMo's torch.stft center=True)
        let pad_size = n_fft / 2;
        let left_pad = Array::zeros::<f32>(&[batch_size, pad_size as i32])?;
        let right_pad = Array::zeros::<f32>(&[batch_size, pad_size as i32])?;
        let padded = ops::concatenate_axis(&[&left_pad, audio, &right_pad], 1)?;

        let padded_len = audio_len + 2 * pad_size;
        let n_frames = (padded_len - n_fft) / hop + 1;

        // Frame the audio
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
        let mean = ops::mean_axis(features, 2, Some(true))?;
        let centered = ops::subtract(features, &mean)?;
        let variance = ops::mean_axis(&ops::square(&centered)?, 2, Some(true))?;
        let std = ops::sqrt(&ops::add(&variance, &Array::from_f32(1e-5))?)?;
        ops::divide(&centered, &std).map_err(|e| Error::model(e.to_string()))
    }

    /// Compute output frame lengths from input sample lengths.
    fn compute_output_lengths(&self, input_lengths: &Array) -> Result<Array> {
        let divided = ops::floor_divide(input_lengths, Array::from_int(self.hop_samples as i32))?;
        ops::add(&divided, &Array::from_int(1)).map_err(|e| Error::model(e.to_string()))
    }
}

// Type aliases for backwards compatibility
pub type MelSpectrogram = MelProcessor;
pub type MelSpectrogramConfig = MelConfig;

// ============================================================================
// Streaming Mel Processor
// ============================================================================

use crate::asr::streaming::{FeatureBuffer, StreamingConfig};

/// Configuration for streaming mel processing.
#[derive(Debug, Clone)]
pub struct StreamingMelConfig {
    /// Base mel configuration
    pub mel: MelConfig,
    /// Chunk size in mel frames (new frames per chunk)
    pub chunk_size: usize,
    /// Shift size in mel frames (how much to advance per chunk)
    pub shift_size: usize,
    /// Pre-encode cache size in mel frames (overlap from previous chunk)
    pub pre_encode_cache: usize,
}

impl StreamingMelConfig {
    /// Create streaming config from MelConfig and StreamingConfig.
    pub fn new(mel: MelConfig, streaming: &StreamingConfig) -> Self {
        Self {
            mel,
            chunk_size: streaming.chunk_size.1,
            shift_size: streaming.shift_size.1,
            pre_encode_cache: streaming.pre_encode_cache_size.1,
        }
    }
}

/// Streaming mel spectrogram processor.
///
/// Combines audio buffering, mel computation, and chunk extraction into
/// a single component. This replaces the dual overlap mechanism that was
/// previously split between `RequestStreamer` (audio overlap) and
/// `StreamingMelBuffer` (mel frame overlap).
///
/// # Architecture
///
/// ```text
/// Audio samples → add_audio() → audio_buffer → MelProcessor → mel_buffer → next_chunk()
///                                    ↑                              ↓
///                            STFT overlap               pre_encode_cache overlap
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let config = StreamingMelConfig::new(mel_config, &streaming_config);
/// let mut processor = StreamingMelProcessor::new(config)?;
///
/// // Add audio incrementally
/// processor.add_audio(&audio_chunk)?;
///
/// // Extract chunks as they become available
/// while let Some(buffer) = processor.next_chunk() {
///     // Process buffer through encoder
/// }
///
/// // Signal end of stream
/// processor.end_stream();
///
/// // Get remaining chunks
/// while let Some(buffer) = processor.next_chunk() {
///     // Process final chunks
/// }
/// ```
pub struct StreamingMelProcessor {
    /// Internal mel processor for computing spectrograms
    processor: MelProcessor,
    /// Configuration
    config: StreamingMelConfig,

    // Audio buffer state
    /// Accumulated audio samples
    audio_buffer: Vec<f32>,
    /// Audio samples to keep for STFT overlap (window - hop samples)
    audio_overlap: Vec<f32>,

    // Mel buffer state
    /// Accumulated mel frames [n_mels, time]
    mel_buffer: Option<Array>,
    /// Total mel frames in buffer
    mel_total_frames: usize,
    /// Current extraction position
    mel_buffer_idx: usize,
    /// Offset for indexing after trimming
    mel_buffer_offset: usize,

    // Stream state
    /// Whether this is the first chunk
    is_first_chunk: bool,
    /// Whether end of stream has been signaled
    stream_ended: bool,
    /// Whether all data has been processed
    is_done: bool,
    /// Whether we've initialized with silence
    initialized: bool,
}

impl StreamingMelProcessor {
    /// Create a new streaming mel processor.
    pub fn new(config: StreamingMelConfig) -> Result<Self> {
        let processor = MelProcessor::new(config.mel.clone())?;

        Ok(Self {
            processor,
            config,
            audio_buffer: Vec::new(),
            audio_overlap: Vec::new(),
            mel_buffer: None,
            mel_total_frames: 0,
            mel_buffer_idx: 0,
            mel_buffer_offset: 0,
            is_first_chunk: true,
            stream_ended: false,
            is_done: false,
            initialized: false,
        })
    }

    /// Get the underlying mel processor.
    pub fn processor(&self) -> &MelProcessor {
        &self.processor
    }

    /// Get the streaming configuration.
    pub fn config(&self) -> &StreamingMelConfig {
        &self.config
    }

    /// Number of mel features.
    pub fn n_mels(&self) -> usize {
        self.processor.n_mels()
    }

    /// Get the audio chunk size that maps to one mel chunk.
    ///
    /// Returns the number of audio samples to add for each chunk.
    pub fn audio_chunk_size(&self) -> usize {
        self.config.chunk_size * self.processor.hop_samples()
    }

    /// Get remaining mel frames to be extracted.
    pub fn remaining_frames(&self) -> usize {
        self.mel_total_frames.saturating_sub(self.mel_buffer_idx)
    }

    /// Check if the processor is done (all data processed after end_stream).
    pub fn is_done(&self) -> bool {
        self.is_done
    }

    /// Add audio samples for processing.
    ///
    /// Audio is buffered and converted to mel frames when enough samples
    /// are available. The processor handles STFT overlap internally.
    pub fn add_audio(&mut self, samples: &[f32]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Initialize with silence for pre_encode_cache on first add
        if !self.initialized {
            self.initialize_with_silence()?;
            self.initialized = true;
        }

        // Add to audio buffer
        self.audio_buffer.extend_from_slice(samples);

        // Process audio into mel when we have enough
        self.process_pending_audio()?;

        Ok(())
    }

    /// Signal end of the audio stream.
    ///
    /// After calling this, `next_chunk()` will return remaining chunks
    /// with `is_last = true` on the final one.
    pub fn end_stream(&mut self) {
        tracing::debug!(
            audio_remaining = self.audio_buffer.len(),
            mel_remaining = self.remaining_frames(),
            "Stream ended"
        );
        self.stream_ended = true;
    }

    /// Extract the next mel chunk if available.
    ///
    /// Returns `None` if not enough frames are available yet.
    /// Call `add_audio()` to provide more data, or `end_stream()` to
    /// get final chunks with padding.
    pub fn next_chunk(&mut self) -> Option<FeatureBuffer> {
        if self.is_done {
            return None;
        }

        // Initialize if not done yet
        if !self.initialized {
            if let Err(e) = self.initialize_with_silence() {
                tracing::error!("Failed to initialize: {}", e);
                return None;
            }
            self.initialized = true;
        }

        // Process any pending audio
        if let Err(e) = self.process_pending_audio() {
            tracing::error!("Failed to process audio: {}", e);
            return None;
        }

        // Try to extract a chunk
        self.try_extract_chunk()
    }

    /// Reset for a new audio stream.
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.audio_overlap.clear();
        self.mel_buffer = None;
        self.mel_total_frames = 0;
        self.mel_buffer_idx = 0;
        self.mel_buffer_offset = 0;
        self.is_first_chunk = true;
        self.stream_ended = false;
        self.is_done = false;
        self.initialized = false;

        tracing::debug!("StreamingMelProcessor reset");
    }

    // ========================================================================
    // Private Implementation
    // ========================================================================

    /// Initialize mel buffer with silence for pre_encode_cache.
    fn initialize_with_silence(&mut self) -> Result<()> {
        let pre_encode_cache = self.config.pre_encode_cache;
        if pre_encode_cache == 0 {
            return Ok(());
        }

        // LOG_MEL_ZERO = ln(2^-24) ≈ -16.6355 (matches NeMo's log_zero_guard)
        const LOG_MEL_ZERO: f32 = -16.635532;

        let n_mels = self.processor.n_mels();
        let silence = ops::full::<f32>(
            &[n_mels as i32, pre_encode_cache as i32],
            &Array::from_f32(LOG_MEL_ZERO),
        )?;

        self.mel_buffer = Some(silence);
        self.mel_total_frames = pre_encode_cache;

        tracing::debug!(
            pre_encode_cache,
            "Initialized with silence frames"
        );

        Ok(())
    }

    /// Get the number of lookback samples for STFT continuity.
    fn stft_lookback_samples(&self) -> usize {
        // One hop of lookback for STFT windowing continuity
        self.processor.hop_samples()
    }

    /// Build audio buffer with STFT lookback for mel computation.
    fn build_audio_with_lookback(&mut self, chunk: &[f32]) -> Vec<f32> {
        let lookback = self.stft_lookback_samples();

        // Initialize overlap buffer with zeros if empty
        if self.audio_overlap.is_empty() {
            self.audio_overlap = vec![0.0f32; lookback];
        }

        // Build buffer: [lookback] + [chunk]
        let mut buffer = Vec::with_capacity(lookback + chunk.len());
        buffer.extend_from_slice(&self.audio_overlap);
        buffer.extend_from_slice(chunk);

        // Update overlap for next iteration
        if chunk.len() >= lookback {
            self.audio_overlap = chunk[chunk.len() - lookback..].to_vec();
        } else {
            // Short chunk - shift and append
            let shift = chunk.len();
            self.audio_overlap.drain(..shift);
            self.audio_overlap.extend_from_slice(chunk);
        }

        buffer
    }

    /// Process pending audio into mel frames.
    fn process_pending_audio(&mut self) -> Result<()> {
        let chunk_audio_size = self.audio_chunk_size();

        // Process complete chunks
        while self.audio_buffer.len() >= chunk_audio_size {
            let audio_chunk: Vec<f32> = self.audio_buffer.drain(..chunk_audio_size).collect();
            self.compute_mel_for_chunk(&audio_chunk)?;
        }

        // Process remaining audio if stream ended
        if self.stream_ended && !self.audio_buffer.is_empty() {
            let remaining = self.audio_buffer.clone();
            self.audio_buffer.clear();
            self.compute_mel_for_chunk(&remaining)?;
        }

        Ok(())
    }

    /// Compute mel for an audio chunk and append to mel buffer.
    fn compute_mel_for_chunk(&mut self, audio_chunk: &[f32]) -> Result<()> {
        if audio_chunk.is_empty() {
            return Ok(());
        }

        // Build audio buffer with STFT lookback
        let audio_buffer = self.build_audio_with_lookback(audio_chunk);

        // Compute mel spectrogram
        let audio_array = Array::from_slice(&audio_buffer, &[audio_buffer.len() as i32]);
        let (mel, _lengths) = self.processor.forward(&audio_array, None)?;

        // Remove batch dim: [1, n_mels, frames] -> [n_mels, frames]
        let mel = mel.index(0);
        mel.eval()?;

        let n_frames = mel.dim(1) as usize;
        if n_frames == 0 {
            return Ok(());
        }

        // Python's approach: trim to expected length, take last chunk_size frames
        let feature_chunk_len = self.config.chunk_size;
        let expected_len = feature_chunk_len + 1;

        let mel_trimmed = if n_frames > expected_len {
            mel.index((.., ..expected_len as i32))
        } else {
            mel
        };
        let trimmed_frames = mel_trimmed.dim(1) as usize;

        let mel_to_append = if trimmed_frames > feature_chunk_len {
            let start = (trimmed_frames - feature_chunk_len) as i32;
            mel_trimmed.index((.., start..))
        } else {
            mel_trimmed
        };

        let appended_frames = mel_to_append.dim(1) as usize;

        // Append to mel buffer
        self.mel_buffer = match &self.mel_buffer {
            Some(existing) => Some(ops::concatenate_axis(&[existing, &mel_to_append], 1)?),
            None => Some(mel_to_append),
        };
        self.mel_total_frames += appended_frames;

        tracing::trace!(
            audio_chunk_len = audio_chunk.len(),
            mel_computed = n_frames,
            mel_appended = appended_frames,
            mel_total = self.mel_total_frames,
            "Computed mel from audio chunk"
        );

        Ok(())
    }

    /// Try to extract a mel chunk.
    fn try_extract_chunk(&mut self) -> Option<FeatureBuffer> {
        let chunk_size = self.config.chunk_size;
        let shift_size = self.config.shift_size;
        let pre_encode_cache = self.config.pre_encode_cache;
        let n_mels = self.processor.n_mels();

        let remaining = self.remaining_frames();
        let required = chunk_size + pre_encode_cache;

        // Not enough frames yet
        if remaining < required {
            if self.is_first_chunk && remaining > 0 {
                // Pad first chunk if needed
                let frames_to_pad = required - remaining;
                tracing::debug!(remaining, required, frames_to_pad, "Padding first chunk");

                if let Err(e) = self.pad_mel_buffer(frames_to_pad) {
                    tracing::error!("Failed to pad: {}", e);
                    return None;
                }
            } else if self.stream_ended && remaining > 0 {
                // Pad final chunk
                let frames_to_pad = required - remaining;
                tracing::debug!(remaining, required, frames_to_pad, "Padding final chunk");

                if let Err(e) = self.pad_mel_buffer(frames_to_pad) {
                    tracing::error!("Failed to pad: {}", e);
                    return None;
                }
            } else {
                // No frames or waiting for more
                if self.stream_ended && remaining == 0 {
                    self.is_done = true;
                }
                return None;
            }
        }

        // Extract chunk from mel buffer
        let mel_buffer = self.mel_buffer.as_ref()?;

        let extract_start = self.mel_buffer_idx;
        let extract_end = self.mel_total_frames.min(extract_start + chunk_size + pre_encode_cache);
        let total_extract = extract_end - extract_start;

        // Convert to local indices
        let local_start = extract_start.saturating_sub(self.mel_buffer_offset);
        let local_end = local_start + total_extract;

        let mel_chunk = mel_buffer.index((.., local_start as i32..local_end as i32));

        // Advance buffer index by shift_size
        self.mel_buffer_idx += shift_size;

        // Trim old frames to bound memory
        if let Err(e) = self.trim_old_frames(pre_encode_cache) {
            tracing::error!("Failed to trim: {}", e);
        }

        let is_first = self.is_first_chunk;
        self.is_first_chunk = false;

        let is_last = self.stream_ended && self.remaining_frames() < chunk_size;
        if is_last {
            self.is_done = true;
        }

        tracing::debug!(
            is_first,
            is_last,
            valid_frames = total_extract,
            pre_encode_cache,
            mel_buffer_idx = self.mel_buffer_idx,
            "Extracted chunk"
        );

        Some(
            FeatureBuffer::new(mel_chunk, total_extract)
                .with_first(is_first)
                .with_last(is_last)
                .with_pre_encode_cache(pre_encode_cache),
        )
    }

    /// Pad mel buffer with zeros.
    fn pad_mel_buffer(&mut self, frames: usize) -> Result<()> {
        if frames == 0 {
            return Ok(());
        }

        let n_mels = self.processor.n_mels();
        let padding = Array::zeros::<f32>(&[n_mels as i32, frames as i32])?;

        self.mel_buffer = match &self.mel_buffer {
            Some(existing) => Some(ops::concatenate_axis(&[existing, &padding], 1)?),
            None => Some(padding),
        };
        self.mel_total_frames += frames;

        Ok(())
    }

    /// Trim old mel frames that are no longer needed.
    fn trim_old_frames(&mut self, keep_frames: usize) -> Result<()> {
        let mel_buffer = match &self.mel_buffer {
            Some(buf) => buf,
            None => return Ok(()),
        };

        let local_idx = self.mel_buffer_idx.saturating_sub(self.mel_buffer_offset);
        let can_trim = local_idx.saturating_sub(keep_frames);

        if can_trim > 0 {
            let trimmed = mel_buffer.index((.., can_trim as i32..));
            self.mel_buffer = Some(trimmed);
            self.mel_buffer_offset += can_trim;
        }

        Ok(())
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
        let config = MelConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.features, 128);
        assert_eq!(config.n_fft, 512);
        assert!((config.window_size - 0.025).abs() < 1e-6);
        assert!((config.window_stride - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_config_derived_values() {
        let config = MelConfig::default();
        assert_eq!(config.win_samples(), 400);
        assert_eq!(config.hop_samples(), 160);
        assert!((config.effective_fmax() - 8000.0).abs() < 1e-6);
    }

    // ==================== Processor Creation Tests ====================

    #[test]
    fn test_processor_creation() {
        let config = MelConfig::default();
        let mel = MelProcessor::new(config);
        assert!(mel.is_ok());

        let mel = mel.unwrap();
        assert_eq!(mel.n_mels(), 128);
        assert_eq!(mel.win_samples(), 400);
        assert_eq!(mel.hop_samples(), 160);
    }

    #[test]
    fn test_num_frames() {
        let config = MelConfig::default();
        let mel = MelProcessor::new(config).unwrap();

        // 16000 samples -> 16000/160 + 1 = 101 frames
        assert_eq!(mel.num_frames(16000), 101);
        // 8000 samples -> 8000/160 + 1 = 51 frames
        assert_eq!(mel.num_frames(8000), 51);
        // Empty
        assert_eq!(mel.num_frames(0), 0);
    }

    // ==================== Window Tests ====================

    #[test]
    fn test_hann_window_endpoints() {
        let window = MelProcessor::create_window_padded("hann", 400, 512).unwrap();
        window.eval().unwrap();
        let data: &[f32] = window.as_slice();

        let left_pad = (512 - 400) / 2;
        assert!(data[left_pad].abs() < 1e-6, "First hann value should be ~0");
        assert!(
            data[left_pad + 399].abs() < 1e-6,
            "Last hann value should be ~0"
        );
    }

    #[test]
    fn test_hamming_window() {
        let window = MelProcessor::create_window_padded("hamming", 400, 512).unwrap();
        window.eval().unwrap();
        let data: &[f32] = window.as_slice();

        let left_pad = (512 - 400) / 2;
        assert!(approx_eq(data[left_pad], 0.08, 1e-4));
    }

    // ==================== Filterbank Tests ====================

    #[test]
    fn test_filterbank_shape() {
        let fb = MelProcessor::create_mel_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        assert_eq!(fb.shape(), &[128, 257]); // [n_mels, n_fft/2+1]
    }

    #[test]
    fn test_filterbank_non_negative() {
        let fb = MelProcessor::create_mel_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        let data: &[f32] = fb.as_slice();

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(min >= 0.0, "Filterbank should be non-negative, got {}", min);
    }

    // ==================== Batch Processing Tests ====================

    #[test]
    fn test_forward_output_shape() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

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
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

        // 1D input (no batch dimension)
        let audio = Array::from_slice(&vec![0.1f32; 16000], &[16000]);
        let (features, _) = mel.forward(&audio, None).unwrap();

        // Should auto-add batch dimension
        assert_eq!(features.shape()[0], 1);
        assert_eq!(features.shape()[1], 128);
    }

    #[test]
    fn test_forward_batch() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

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
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

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
    fn test_preemphasis_applied() {
        // With preemphasis enabled, output should differ from disabled
        let config_with = MelConfig {
            preemph: Some(0.97),
            dither: 0.0,
            ..Default::default()
        };
        let config_without = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };

        let mel_with = MelProcessor::new(config_with).unwrap();
        let mel_without = MelProcessor::new(config_without).unwrap();

        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio = Array::from_slice(&samples, &[1, 1600]);

        let (feat_with, _) = mel_with.forward(&audio, None).unwrap();
        let (feat_without, _) = mel_without.forward(&audio, None).unwrap();

        feat_with.eval().unwrap();
        feat_without.eval().unwrap();

        let data_with: &[f32] = feat_with.as_slice();
        let data_without: &[f32] = feat_without.as_slice();

        // Values should differ
        let mut has_diff = false;
        for (a, b) in data_with.iter().zip(data_without.iter()) {
            if (a - b).abs() > 1e-4 {
                has_diff = true;
                break;
            }
        }
        assert!(has_diff, "Preemphasis should change output");
    }

    // ==================== Normalization Tests ====================

    #[test]
    fn test_per_feature_normalization() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            normalize: "per_feature".to_string(),
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

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
        let config = MelConfig::default();
        let mel = MelProcessor::new(config).unwrap();

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
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

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
        let config = MelConfig {
            preemph: None,
            dither: 1e-5,
            ..Default::default()
        };
        let mel = MelProcessor::new(config).unwrap();

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
}
