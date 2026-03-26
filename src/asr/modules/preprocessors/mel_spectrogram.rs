//! Mel spectrogram audio preprocessing.
//!
//! Converts raw audio waveforms to log-Mel spectrogram features for ASR models.
//!
//! This module provides:
//! - [`MelConfig`]: Configuration matching NeMo's AudioToMelSpectrogramPreprocessor
//! - [`MelProcessor`]: Cached, stateless mel spectrogram processor
//! - [`StreamingMel`]: Simple streaming processor with rolling mel buffer
//!
//! # Batch Processing Example
//!
//! ```rust,ignore
//! let config = MelConfig::default();
//! let mel = MelProcessor::new(config)?;
//!
//! // Process audio (stateless - no streaming state)
//! let (features, lengths) = mel.forward(&audio, None)?;
//! // features: [batch, n_mels, frames]
//! ```
//!
//! # Streaming Processing Example
//!
//! ```rust,ignore
//! let mut mel = StreamingMel::new(config)?;
//!
//! for audio_chunk in audio_chunks {
//!     mel.add_audio(&audio_chunk)?;
//!
//!     while mel.total_frames() >= chunk_size {
//!         let frames = mel.get_frames(0, chunk_size)?;
//!         // Process frames through encoder...
//!         mel.trim(shift_size);
//!     }
//! }
//! ```

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{fft, ops, Array};

use crate::asr::config::PreprocessorConfig;
use crate::asr::dsp;
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
    /// Precomputed Hann/Hamming window [n_fft], center-padded.
    /// Used for both batch and streaming processing.
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

        // Create window function (center-padded to n_fft)
        // Used for both batch and streaming processing
        let window = dsp::create_window_padded(&config.window, win_samples, config.n_fft)?;

        // Create mel filterbank using dsp module
        let filterbank = dsp::mel_filterbank(
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
        let audio = if self.config.dither > 0.0 {
            dsp::dither(&audio, self.config.dither)?
        } else {
            audio
        };

        // Apply preemphasis if configured
        let audio = if let Some(coef) = self.config.preemph {
            dsp::preemphasis(&audio, coef)?
        } else {
            audio
        };

        // Compute STFT with center padding
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
    // Private: Audio Processing
    // ========================================================================

    /// Compute STFT for batch of audio with center padding.
    fn stft_batch(&self, audio: &Array) -> Result<Array> {
        let batch_size = audio.shape()[0];
        let audio_len = audio.shape()[1] as usize;
        let n_fft = self.config.n_fft;
        let hop = self.hop_samples;

        // Apply signal-level center padding (zeros, matching NeMo's torch.stft center=True)
        // This is different from FFT frame centering - it pads the entire signal so that
        // the first and last frames are centered on the signal boundaries.
        let pad_size = n_fft / 2;
        let left_pad = Array::zeros::<f32>(&[batch_size, pad_size as i32])?;
        let right_pad = Array::zeros::<f32>(&[batch_size, pad_size as i32])?;
        let padded = ops::concatenate_axis(&[&left_pad, audio, &right_pad], 1)?;

        // Frame the audio using dsp module
        // frame_audio_batch always center-pads each FFT frame (audio centered within n_fft)
        let frames = dsp::frame_audio_batch(&padded, n_fft, hop, n_fft)?;

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
// Stateless Streaming API
// ============================================================================

/// State for stateless streaming mel processing.
///
/// This struct holds the minimal state needed for continuous streaming:
/// - Audio overlap buffer for STFT window continuity
/// - Last sample for preemphasis filter continuity
///
/// The state is passed explicitly to streaming functions and returned updated,
/// keeping all processing stateless from the function's perspective.
#[derive(Debug, Clone)]
pub struct MelStreamState {
    /// Audio overlap buffer (window_samples - hop_samples = 240 samples by default).
    /// This ensures STFT window continuity across chunk boundaries.
    pub overlap: Vec<f32>,
    /// Last sample from previous chunk for preemphasis continuity.
    /// Only used when preemphasis is enabled (preemph != None).
    pub last_sample: f32,
}

impl MelStreamState {
    /// Create a new streaming state for the given processor.
    ///
    /// Initializes the overlap buffer with zeros and last_sample to 0.
    pub fn new(processor: &MelProcessor) -> Self {
        let overlap_size = processor.win_samples().saturating_sub(processor.hop_samples());
        Self {
            overlap: vec![0.0f32; overlap_size],
            last_sample: 0.0,
        }
    }

    /// Create a new streaming state with explicit overlap size.
    pub fn with_overlap_size(overlap_size: usize) -> Self {
        Self {
            overlap: vec![0.0f32; overlap_size],
            last_sample: 0.0,
        }
    }

    /// Reset the state to initial values.
    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
        self.last_sample = 0.0;
    }

    /// Get the overlap buffer size.
    pub fn overlap_len(&self) -> usize {
        self.overlap.len()
    }
}

/// Calculate how many complete mel frames can be computed from the available audio.
///
/// # Arguments
/// * `overlap_len` - Number of samples in the overlap buffer
/// * `new_audio_len` - Number of new audio samples
/// * `window_samples` - STFT window size in samples (e.g., 400)
/// * `hop_samples` - STFT hop size in samples (e.g., 160)
///
/// # Returns
/// Number of complete frames that can be computed.
///
/// # Example
/// ```rust,ignore
/// // With 240 samples overlap and 1600 new samples:
/// // Total = 1840, need 400 for first frame, then 160 per additional
/// // frames = (1840 - 400) / 160 + 1 = 10 frames
/// let frames = frames_available(240, 1600, 400, 160);
/// assert_eq!(frames, 10);
/// ```
#[inline]
pub fn frames_available(
    overlap_len: usize,
    new_audio_len: usize,
    window_samples: usize,
    hop_samples: usize,
) -> usize {
    let total = overlap_len + new_audio_len;
    if total < window_samples {
        0
    } else {
        (total - window_samples) / hop_samples + 1
    }
}

/// Process audio chunk in streaming mode, returning complete mel frames and updated state.
///
/// This is a **stateless** function - all state is passed explicitly and returned updated.
/// The caller is responsible for managing the state between calls.
///
/// # Arguments
/// * `processor` - The mel processor (contains cached filterbank and window)
/// * `audio` - New audio samples to process
/// * `state` - Current streaming state (overlap buffer, preemphasis state)
///
/// # Returns
/// Tuple of (mel_frames, updated_state):
/// - `mel_frames`: `Some(Array)` with shape [n_mels, n_frames] if any complete frames were computed,
///                 `None` if not enough audio for a complete frame
/// - `updated_state`: New state to use for the next call
///
/// # No Center Padding
///
/// Unlike the batch `forward()` method, this function does **not** apply center padding.
/// This means:
/// - The first frame covers samples [0, window_samples)
/// - Each subsequent frame shifts by hop_samples
/// - This produces slightly fewer frames than center-padded batch mode
///
/// # Example
///
/// ```rust,ignore
/// let processor = MelProcessor::new(config)?;
/// let mut state = MelStreamState::new(&processor);
///
/// for chunk in audio_chunks {
///     let (frames, new_state) = process_audio_streaming(&processor, &chunk, &state)?;
///     state = new_state;
///
///     if let Some(mel) = frames {
///         // mel has shape [n_mels, n_frames]
///         process_mel_chunk(mel);
///     }
/// }
/// ```
pub fn process_audio_streaming(
    processor: &MelProcessor,
    audio: &[f32],
    state: &MelStreamState,
) -> Result<(Option<Array>, MelStreamState)> {
    let win_samples = processor.win_samples();
    let hop_samples = processor.hop_samples();
    let overlap_size = win_samples.saturating_sub(hop_samples);

    // Calculate how many frames we can produce
    let n_frames = frames_available(state.overlap.len(), audio.len(), win_samples, hop_samples);

    tracing::debug!(
        audio_len = audio.len(),
        overlap_len = state.overlap.len(),
        total_available = state.overlap.len() + audio.len(),
        win_samples,
        hop_samples,
        overlap_size,
        n_frames,
        "process_audio_streaming: calculating frames"
    );

    if n_frames == 0 {
        // Not enough audio for a complete frame - accumulate into overlap
        let mut new_overlap = state.overlap.clone();
        new_overlap.extend_from_slice(audio);

        // Keep only the last overlap_size samples if we've accumulated too much
        // (This shouldn't happen in normal operation, but handle it gracefully)
        if new_overlap.len() > overlap_size + win_samples {
            let keep_from = new_overlap.len() - overlap_size - win_samples;
            tracing::warn!(
                new_overlap_len = new_overlap.len(),
                max_expected = overlap_size + win_samples,
                trimming_from = keep_from,
                "process_audio_streaming: overflow, trimming accumulated audio"
            );
            new_overlap.drain(..keep_from);
        }

        tracing::debug!(
            new_overlap_len = new_overlap.len(),
            "process_audio_streaming: not enough audio, accumulating"
        );

        let new_state = MelStreamState {
            overlap: new_overlap,
            last_sample: if audio.is_empty() {
                state.last_sample
            } else {
                *audio.last().unwrap()
            },
        };

        return Ok((None, new_state));
    }

    // Combine overlap buffer with new audio
    let mut combined = Vec::with_capacity(state.overlap.len() + audio.len());
    combined.extend_from_slice(&state.overlap);
    combined.extend_from_slice(audio);

    // Process frames
    let mel = processor.process_frames(&combined, n_frames, state.last_sample)?;

    // Calculate new overlap: keep the last overlap_size samples from the audio we consumed
    let samples_consumed = (n_frames - 1) * hop_samples + win_samples;
    let new_overlap_start = samples_consumed.saturating_sub(overlap_size);
    let new_overlap = if new_overlap_start < combined.len() {
        combined[new_overlap_start..samples_consumed.min(combined.len())].to_vec()
    } else {
        vec![0.0f32; overlap_size]
    };

    // Ensure overlap is exactly overlap_size (pad with zeros if needed)
    let mut final_overlap = new_overlap;
    if final_overlap.len() < overlap_size {
        let mut padded = vec![0.0f32; overlap_size - final_overlap.len()];
        padded.extend(final_overlap);
        final_overlap = padded;
    } else if final_overlap.len() > overlap_size {
        final_overlap = final_overlap[final_overlap.len() - overlap_size..].to_vec();
    }

    // Update last_sample for preemphasis continuity
    let new_last_sample = if samples_consumed > 0 && samples_consumed <= combined.len() {
        combined[samples_consumed - 1]
    } else if !audio.is_empty() {
        *audio.last().unwrap()
    } else {
        state.last_sample
    };

    let new_state = MelStreamState {
        overlap: final_overlap.clone(),
        last_sample: new_last_sample,
    };

    tracing::debug!(
        n_frames,
        samples_consumed,
        combined_len = combined.len(),
        new_overlap_len = final_overlap.len(),
        leftover_samples = combined.len().saturating_sub(samples_consumed),
        "process_audio_streaming: frames computed"
    );

    Ok((Some(mel), new_state))
}

impl MelProcessor {
    /// Process multiple frames for streaming.
    ///
    /// Note: This does NOT apply signal-level center padding (which is only for batch mode).
    /// However, it DOES use center-padded FFT frames (audio centered within each n_fft frame),
    /// which is required for proper STFT alignment with the center-padded window.
    ///
    /// # Arguments
    /// * `audio` - Audio buffer with enough samples for n_frames
    /// * `n_frames` - Number of frames to compute
    /// * `last_sample` - Last sample from previous chunk (for preemphasis continuity)
    ///
    /// # Returns
    /// Mel features with shape [n_mels, n_frames]
    pub fn process_frames(&self, audio: &[f32], n_frames: usize, last_sample: f32) -> Result<Array> {
        if n_frames == 0 {
            return Ok(Array::zeros::<f32>(&[self.config.features as i32, 0])?);
        }

        let win_samples = self.win_samples;
        let hop_samples = self.hop_samples;
        let n_fft = self.config.n_fft;

        // Verify we have enough audio
        let required = (n_frames - 1) * hop_samples + win_samples;
        if audio.len() < required {
            return Err(Error::config(format!(
                "Not enough audio: have {}, need {} for {} frames",
                audio.len(),
                required,
                n_frames
            )));
        }

        // Convert to MLX array for processing
        let audio_arr = Array::from_slice(audio, &[audio.len() as i32]);

        // Apply preemphasis if configured
        let audio_arr = if let Some(coef) = self.config.preemph {
            // For streaming preemphasis, we need to prepend the last_sample
            let last_arr = Array::from_slice(&[last_sample], &[1]);
            let extended = ops::concatenate(&[&last_arr, &audio_arr])?;
            let preemph = dsp::preemphasis(&extended, coef)?;
            // Remove the first sample (it was just for continuity)
            preemph.index(1..)
        } else {
            audio_arr
        };

        // Apply dithering if configured
        let audio_arr = if self.config.dither > 0.0 {
            dsp::dither(&audio_arr, self.config.dither)?
        } else {
            audio_arr
        };

        // Frame the audio using dsp module
        // frame_audio always center-pads each FFT frame (audio centered within n_fft)
        let frames = dsp::frame_audio(&audio_arr, win_samples, hop_samples, n_fft)?;

        // Apply window (same center-padded window used for batch processing)
        let windowed = ops::multiply(&frames, &self.window)?;

        // Compute FFT along last axis
        let fft_result = fft::rfft(&windowed, Some(n_fft as i32), Some(1))?;

        // Compute power spectrum
        let power = self.power_spectrum(&fft_result)?;

        // Apply mel filterbank: power [frames, freqs] -> mel [frames, n_mels]
        // Transpose power to [freqs, frames] for matmul
        let power_t = ops::transpose(&power)?;
        let mel = ops::matmul(&self.filterbank, &power_t)?;

        // mel is now [n_mels, frames]

        // Apply log compression
        let mel = if self.config.log {
            let mel_safe = ops::add(&mel, &self.log_guard)?;
            ops::log(&mel_safe)?
        } else {
            mel
        };

        // Apply normalization if requested
        let mel = match self.config.normalize.as_str() {
            "per_feature" => {
                // Per-feature norm over time axis (axis 1)
                let mean = ops::mean_axis(&mel, 1, Some(true))?;
                let centered = ops::subtract(&mel, &mean)?;
                let variance = ops::mean_axis(&ops::square(&centered)?, 1, Some(true))?;
                let std = ops::sqrt(&ops::add(&variance, &Array::from_f32(1e-5))?)?;
                ops::divide(&centered, &std)?
            }
            _ => mel,
        };

        mel.eval()?;
        Ok(mel)
    }

    /// Process a single frame without center padding (for streaming).
    ///
    /// # Arguments
    /// * `frame_audio` - Exactly win_samples (e.g., 400) audio samples
    /// * `last_sample` - Last sample from previous frame (for preemphasis continuity)
    ///
    /// # Returns
    /// Mel features with shape [n_mels]
    pub fn process_frame(&self, frame_audio: &[f32], last_sample: f32) -> Result<Array> {
        if frame_audio.len() != self.win_samples {
            return Err(Error::config(format!(
                "Frame must have exactly {} samples, got {}",
                self.win_samples,
                frame_audio.len()
            )));
        }

        let mel = self.process_frames(frame_audio, 1, last_sample)?;
        // Squeeze from [n_mels, 1] to [n_mels]
        mel.reshape(&[self.config.features as i32])
            .map_err(|e| Error::model(e.to_string()))
    }
}

// ============================================================================
// Streaming Configuration for Pipeline Integration
// ============================================================================

use crate::asr::streaming::StreamingConfig;

/// Configuration for streaming mel processing in the pipeline.
///
/// This contains encoder-level streaming parameters that determine
/// how mel frames are chunked and overlapped for the encoder.
#[derive(Debug, Clone)]
pub struct StreamingMelConfig {
    /// Base mel configuration
    pub mel: MelConfig,
    /// Chunk size in mel frames (new frames per chunk)
    pub chunk_size: usize,
    /// Shift size in mel frames (how much to advance per chunk)
    pub shift_size: usize,
    /// Pre-encode cache size in mel frames for first chunk
    pub pre_encode_cache_first: usize,
    /// Pre-encode cache size in mel frames for subsequent chunks (overlap from previous chunk)
    pub pre_encode_cache: usize,
}

impl StreamingMelConfig {
    /// Create streaming config from MelConfig and StreamingConfig.
    pub fn new(mel: MelConfig, streaming: &StreamingConfig) -> Self {
        Self {
            mel,
            chunk_size: streaming.chunk_size.1,
            shift_size: streaming.shift_size.1,
            pre_encode_cache_first: streaming.pre_encode_cache_size.0,
            pre_encode_cache: streaming.pre_encode_cache_size.1,
        }
    }

    /// Get the audio chunk size in samples that maps to one mel chunk.
    pub fn audio_chunk_size(&self, hop_samples: usize) -> usize {
        self.chunk_size * hop_samples
    }

    /// Get the pre-encode cache size for a given chunk.
    pub fn pre_encode_cache_for_chunk(&self, is_first: bool) -> usize {
        if is_first {
            self.pre_encode_cache_first
        } else {
            self.pre_encode_cache
        }
    }
}

// ============================================================================
// StreamingMel: Simple Streaming API
// ============================================================================

/// Simple streaming mel processor with rolling buffer.
///
/// This struct provides a clean API for streaming mel spectrogram processing:
/// - Add audio chunks with [`add_audio`](Self::add_audio)
/// - Get mel frames with [`get_frames`](Self::get_frames)
/// - Trim processed frames with [`trim`](Self::trim)
///
/// # Example
///
/// ```rust,ignore
/// let mut mel = StreamingMel::new(config)?;
///
/// for audio_chunk in audio_stream {
///     mel.add_audio(&audio_chunk)?;
///
///     while mel.total_frames() >= chunk_size + overlap {
///         let chunk = mel.get_frames(0, chunk_size)?;
///         // Process through encoder...
///
///         // Trim processed frames, keep overlap
///         mel.trim(chunk_size - overlap);
///     }
/// }
/// ```
pub struct StreamingMel {
    /// The underlying mel processor
    processor: MelProcessor,

    /// Audio lookback buffer for STFT window continuity.
    /// Size: window_samples - hop_samples (e.g., 240 samples)
    audio_lookback: Vec<f32>,

    /// Last audio sample for preemphasis continuity
    last_sample: f32,

    /// Rolling mel buffer - stores flattened [n_mels * total_frames] data
    mel_frames: Vec<f32>,

    /// Total mel frames in buffer
    total_frames: usize,

    /// Number of mel features
    n_mels: usize,

    /// Window length in samples
    win_samples: usize,

    /// Hop length in samples
    hop_samples: usize,
}

impl StreamingMel {
    /// Create a new streaming mel processor.
    ///
    /// # Arguments
    /// * `config` - Mel spectrogram configuration
    pub fn new(config: MelConfig) -> Result<Self> {
        let processor = MelProcessor::new(config)?;
        let n_mels = processor.n_mels();
        let win_samples = processor.win_samples();
        let hop_samples = processor.hop_samples();
        let overlap_size = dsp::overlap_size(win_samples, hop_samples);

        Ok(Self {
            processor,
            audio_lookback: vec![0.0f32; overlap_size],
            last_sample: 0.0,
            mel_frames: Vec::new(),
            total_frames: 0,
            n_mels,
            win_samples,
            hop_samples,
        })
    }

    /// Add audio chunk and compute mel frames.
    ///
    /// Returns the number of new mel frames produced.
    ///
    /// # Arguments
    /// * `audio` - New audio samples to process
    pub fn add_audio(&mut self, audio: &[f32]) -> Result<usize> {
        if audio.is_empty() {
            return Ok(0);
        }

        // Combine lookback buffer with new audio
        let mut combined = Vec::with_capacity(self.audio_lookback.len() + audio.len());
        combined.extend_from_slice(&self.audio_lookback);
        combined.extend_from_slice(audio);

        // Calculate how many frames we can produce
        let n_frames = dsp::num_frames(combined.len(), self.win_samples, self.hop_samples);

        if n_frames == 0 {
            // Not enough audio - update lookback buffer
            self.audio_lookback = combined;
            if let Some(&last) = audio.last() {
                self.last_sample = last;
            }
            return Ok(0);
        }

        // Process frames
        let mel = self.processor.process_frames(&combined, n_frames, self.last_sample)?;
        mel.eval()?;

        // Extract mel data and append to buffer
        let mel_data: &[f32] = mel.as_slice();
        // mel is [n_mels, n_frames], stored in row-major order
        // We need to transpose to [n_frames, n_mels] for our buffer
        for frame in 0..n_frames {
            for mel_bin in 0..self.n_mels {
                self.mel_frames.push(mel_data[mel_bin * n_frames + frame]);
            }
        }
        self.total_frames += n_frames;

        // Update lookback buffer
        let samples_consumed = (n_frames - 1) * self.hop_samples + self.win_samples;
        let overlap_size = dsp::overlap_size(self.win_samples, self.hop_samples);

        if samples_consumed >= combined.len() {
            // All samples consumed, keep last overlap_size from end
            let start = combined.len().saturating_sub(overlap_size);
            self.audio_lookback = combined[start..].to_vec();
        } else {
            // Keep unconsumed samples plus some overlap
            let keep_start = samples_consumed.saturating_sub(overlap_size);
            self.audio_lookback = combined[keep_start..].to_vec();
        }

        // Ensure lookback is correct size
        if self.audio_lookback.len() < overlap_size {
            let mut padded = vec![0.0f32; overlap_size - self.audio_lookback.len()];
            padded.extend(&self.audio_lookback);
            self.audio_lookback = padded;
        } else if self.audio_lookback.len() > overlap_size + self.win_samples {
            // Trim excess
            let trim_from = self.audio_lookback.len() - overlap_size - self.win_samples;
            self.audio_lookback.drain(..trim_from);
        }

        // Update last sample
        if let Some(&last) = audio.last() {
            self.last_sample = last;
        }

        Ok(n_frames)
    }

    /// Get mel frames as Array [n_mels, count].
    ///
    /// # Arguments
    /// * `start` - Start frame index
    /// * `end` - End frame index (exclusive)
    ///
    /// # Returns
    /// Array with shape [n_mels, end - start]
    pub fn get_frames(&self, start: usize, end: usize) -> Result<Array> {
        if start >= self.total_frames {
            return Ok(Array::zeros::<f32>(&[self.n_mels as i32, 0])?);
        }

        let end = end.min(self.total_frames);
        let count = end - start;

        if count == 0 {
            return Ok(Array::zeros::<f32>(&[self.n_mels as i32, 0])?);
        }

        // Extract frames from buffer
        // Buffer is stored as [frame0_mel0, frame0_mel1, ..., frame1_mel0, ...]
        // We need to reshape to [n_mels, count]
        let mut output = Vec::with_capacity(self.n_mels * count);

        for mel_bin in 0..self.n_mels {
            for frame in start..end {
                let idx = frame * self.n_mels + mel_bin;
                output.push(self.mel_frames[idx]);
            }
        }

        Ok(Array::from_slice(&output, &[self.n_mels as i32, count as i32]))
    }

    /// Get mel frames for a range, with optional padding.
    ///
    /// If the range extends beyond available frames, pads with the specified value.
    ///
    /// # Arguments
    /// * `start` - Start frame index
    /// * `count` - Number of frames to get
    /// * `pad_value` - Value to use for padding
    pub fn get_frames_padded(&self, start: usize, count: usize, pad_value: f32) -> Result<Array> {
        let available_start = start.min(self.total_frames);
        let available_end = (start + count).min(self.total_frames);
        let available_count = available_end.saturating_sub(available_start);
        let pad_before = start.saturating_sub(available_start);
        let pad_after = count - available_count - pad_before;

        let mut output = Vec::with_capacity(self.n_mels * count);

        for mel_bin in 0..self.n_mels {
            // Pad before
            for _ in 0..pad_before {
                output.push(pad_value);
            }
            // Actual frames
            for frame in available_start..available_end {
                let idx = frame * self.n_mels + mel_bin;
                output.push(self.mel_frames[idx]);
            }
            // Pad after
            for _ in 0..pad_after {
                output.push(pad_value);
            }
        }

        Ok(Array::from_slice(&output, &[self.n_mels as i32, count as i32]))
    }

    /// Total mel frames available.
    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Number of mel features.
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Trim frames before `keep_from`, adjusts all indices.
    ///
    /// This frees memory for frames that are no longer needed.
    ///
    /// # Arguments
    /// * `trim_count` - Number of frames to trim from the beginning
    pub fn trim(&mut self, trim_count: usize) {
        if trim_count == 0 || trim_count >= self.total_frames {
            if trim_count >= self.total_frames {
                self.mel_frames.clear();
                self.total_frames = 0;
            }
            return;
        }

        // Remove first trim_count frames
        let elements_to_remove = trim_count * self.n_mels;
        self.mel_frames.drain(..elements_to_remove);
        self.total_frames -= trim_count;
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        let overlap_size = dsp::overlap_size(self.win_samples, self.hop_samples);
        self.audio_lookback = vec![0.0f32; overlap_size];
        self.last_sample = 0.0;
        self.mel_frames.clear();
        self.total_frames = 0;
    }

    /// Get a reference to the underlying processor.
    pub fn processor(&self) -> &MelProcessor {
        &self.processor
    }

    /// Get the audio lookback buffer size.
    pub fn lookback_size(&self) -> usize {
        self.audio_lookback.len()
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
        let window = dsp::create_window_padded("hann", 400, 512).unwrap();
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
        let window = dsp::create_window_padded("hamming", 400, 512).unwrap();
        window.eval().unwrap();
        let data: &[f32] = window.as_slice();

        let left_pad = (512 - 400) / 2;
        assert!(approx_eq(data[left_pad], 0.08, 1e-4));
    }

    // ==================== Filterbank Tests ====================

    #[test]
    fn test_filterbank_shape() {
        let fb = dsp::mel_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
        fb.eval().unwrap();
        assert_eq!(fb.shape(), &[128, 257]); // n_mels * (n_fft/2+1)
    }

    #[test]
    fn test_filterbank_non_negative() {
        let fb = dsp::mel_filterbank(128, 512, 16000.0, 0.0, 8000.0).unwrap();
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

    // ==================== Stateless Streaming Tests ====================

    #[test]
    fn test_frames_available() {
        // Default: window=400, hop=160, overlap=240
        // With 240 overlap and 1600 new samples:
        // total = 1840, need 400 for first frame
        // frames = (1840 - 400) / 160 + 1 = 10
        assert_eq!(super::frames_available(240, 1600, 400, 160), 10);

        // Not enough for a frame
        assert_eq!(super::frames_available(240, 100, 400, 160), 0);

        // Exactly one frame
        // 240 + 160 = 400, which is exactly one window
        assert_eq!(super::frames_available(240, 160, 400, 160), 1);

        // Two frames
        // 240 + 320 = 560, need 400 for first, then 160 more for second
        // (560 - 400) / 160 + 1 = 2
        assert_eq!(super::frames_available(240, 320, 400, 160), 2);
    }

    #[test]
    fn test_mel_stream_state_creation() {
        let config = MelConfig::default();
        let processor = MelProcessor::new(config).unwrap();
        let state = super::MelStreamState::new(&processor);

        // Default: window=400, hop=160, overlap = 400-160 = 240
        assert_eq!(state.overlap.len(), 240);
        assert_eq!(state.last_sample, 0.0);
    }

    #[test]
    fn test_process_frames_single() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let processor = MelProcessor::new(config).unwrap();

        // Create exactly one window worth of audio (400 samples)
        let audio: Vec<f32> = (0..400).map(|i| (i as f32 * 0.01).sin()).collect();

        let mel = processor.process_frames(&audio, 1, 0.0).unwrap();
        mel.eval().unwrap();

        // Output should be [n_mels, 1]
        assert_eq!(mel.shape(), &[128, 1]);
    }

    #[test]
    fn test_process_frames_multiple() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let processor = MelProcessor::new(config).unwrap();

        // Create enough audio for 5 frames
        // First frame needs 400, then 4 more frames need 4*160 = 640 more
        // Total: 400 + 640 = 1040 samples
        let audio: Vec<f32> = (0..1040).map(|i| (i as f32 * 0.01).sin()).collect();

        let mel = processor.process_frames(&audio, 5, 0.0).unwrap();
        mel.eval().unwrap();

        // Output should be [n_mels, 5]
        assert_eq!(mel.shape(), &[128, 5]);
    }

    #[test]
    fn test_process_audio_streaming_basic() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let processor = MelProcessor::new(config).unwrap();
        let state = super::MelStreamState::new(&processor);

        // Create enough audio for several frames
        // State has 240 overlap (zeros), we add 1600 samples
        // Total = 1840, frames = (1840 - 400) / 160 + 1 = 10
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();

        let (mel_opt, new_state) =
            super::process_audio_streaming(&processor, &audio, &state).unwrap();

        // Should produce mel frames
        assert!(mel_opt.is_some());
        let mel = mel_opt.unwrap();
        mel.eval().unwrap();

        // Output should have 10 frames (from the frames_available calculation)
        assert_eq!(mel.shape()[0], 128); // n_mels
        assert_eq!(mel.shape()[1], 10); // n_frames

        // New state should have updated overlap
        assert_eq!(new_state.overlap.len(), 240);
    }

    #[test]
    fn test_process_audio_streaming_not_enough_audio() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let processor = MelProcessor::new(config).unwrap();
        let state = super::MelStreamState::new(&processor);

        // Not enough audio for even one frame
        // State overlap is 240 (zeros), add only 100 samples = 340 total < 400
        let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();

        let (mel_opt, new_state) =
            super::process_audio_streaming(&processor, &audio, &state).unwrap();

        // Should not produce mel frames
        assert!(mel_opt.is_none());

        // New state should accumulate the audio
        assert!(new_state.overlap.len() >= 240);
    }

    #[test]
    fn test_process_audio_streaming_continuity() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let processor = MelProcessor::new(config).unwrap();
        let mut state = super::MelStreamState::new(&processor);

        // Process audio in chunks and verify continuity
        let full_audio: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.001).sin()).collect();

        // Process in two chunks of 1600 samples each
        let chunk1 = &full_audio[0..1600];
        let chunk2 = &full_audio[1600..3200];

        let (mel1_opt, state1) =
            super::process_audio_streaming(&processor, chunk1, &state).unwrap();
        state = state1;

        let (mel2_opt, _state2) =
            super::process_audio_streaming(&processor, chunk2, &state).unwrap();

        // Both chunks should produce mel frames
        assert!(mel1_opt.is_some());
        assert!(mel2_opt.is_some());

        let mel1 = mel1_opt.unwrap();
        let mel2 = mel2_opt.unwrap();
        mel1.eval().unwrap();
        mel2.eval().unwrap();

        // Total frames should be reasonable for 3200 samples
        let total_frames = mel1.shape()[1] + mel2.shape()[1];
        // With 240 initial overlap and 3200 new samples = 3440 total
        // (3440 - 400) / 160 + 1 = 20 frames approximately
        assert!(total_frames >= 15, "Expected ~20 frames, got {}", total_frames);
    }

    // ==================== StreamingMel Tests ====================

    #[test]
    fn test_streaming_mel_creation() {
        let config = MelConfig::default();
        let mel = StreamingMel::new(config).unwrap();

        assert_eq!(mel.n_mels(), 128);
        assert_eq!(mel.total_frames(), 0);
        assert_eq!(mel.lookback_size(), 240); // window - hop = 400 - 160
    }

    #[test]
    fn test_streaming_mel_add_audio() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Add 1600 samples
        // With 240 lookback (zeros) + 1600 new = 1840 total
        // Frames = (1840 - 400) / 160 + 1 = 10
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let frames_added = mel.add_audio(&audio).unwrap();

        assert_eq!(frames_added, 10);
        assert_eq!(mel.total_frames(), 10);
    }

    #[test]
    fn test_streaming_mel_get_frames() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Add audio to produce frames
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        mel.add_audio(&audio).unwrap();

        // Get frames [0, 5)
        let frames = mel.get_frames(0, 5).unwrap();
        frames.eval().unwrap();

        assert_eq!(frames.shape(), &[128, 5]);
    }

    #[test]
    fn test_streaming_mel_trim() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Add audio to produce frames
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        mel.add_audio(&audio).unwrap();
        assert_eq!(mel.total_frames(), 10);

        // Trim first 5 frames
        mel.trim(5);
        assert_eq!(mel.total_frames(), 5);

        // Get remaining frames
        let frames = mel.get_frames(0, 5).unwrap();
        frames.eval().unwrap();
        assert_eq!(frames.shape(), &[128, 5]);
    }

    #[test]
    fn test_streaming_mel_reset() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Add audio
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        mel.add_audio(&audio).unwrap();
        assert!(mel.total_frames() > 0);

        // Reset
        mel.reset();
        assert_eq!(mel.total_frames(), 0);
        assert_eq!(mel.lookback_size(), 240);
    }

    #[test]
    fn test_streaming_mel_not_enough_audio() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Add only 100 samples - not enough for a frame
        // Need 400 samples for first frame, have 240 (zeros) + 100 = 340
        let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let frames_added = mel.add_audio(&audio).unwrap();

        assert_eq!(frames_added, 0);
        assert_eq!(mel.total_frames(), 0);
    }

    #[test]
    fn test_streaming_mel_multiple_chunks() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Process in multiple chunks
        let audio1: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        let audio2: Vec<f32> = (0..1600).map(|i| ((i + 1600) as f32 * 0.001).sin()).collect();

        let frames1 = mel.add_audio(&audio1).unwrap();
        let frames2 = mel.add_audio(&audio2).unwrap();

        // Should produce reasonable number of frames
        assert!(frames1 > 0);
        assert!(frames2 > 0);
        assert_eq!(mel.total_frames(), frames1 + frames2);
    }

    #[test]
    fn test_streaming_mel_get_frames_padded() {
        let config = MelConfig {
            preemph: None,
            dither: 0.0,
            ..Default::default()
        };
        let mut mel = StreamingMel::new(config).unwrap();

        // Add audio
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.001).sin()).collect();
        mel.add_audio(&audio).unwrap();
        let total = mel.total_frames();

        // Request more frames than available, with padding
        let frames = mel.get_frames_padded(0, total + 5, -16.0).unwrap();
        frames.eval().unwrap();

        // Should have requested count
        assert_eq!(frames.shape(), &[128, (total + 5) as i32]);

        // Check that padding values are present
        let data: &[f32] = frames.as_slice();
        // Last 5 frames should be padded (for each mel bin)
        let last_frame_start = 127 * (total + 5) + total; // mel bin 127, frame 'total'
        assert!(approx_eq(data[last_frame_start], -16.0, 1e-6));
    }
}
