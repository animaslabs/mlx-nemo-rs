//! Streaming ASR support.
//!
//! This module provides types and utilities for cache-aware streaming
//! ASR inference, compatible with NeMo's streaming architecture.
//!
//! # Architecture
//!
//! Streaming inference uses three cache tensors maintained across chunks:
//! - `cache_last_channel`: Stores key/value states for multi-head attention
//! - `cache_last_time`: Stores left context for causal convolutions
//! - `cache_last_channel_len`: Tracks actual cache fill level per sample
//!
//! # Example
//!
//! ```rust,ignore
//! use mlx_nemo::asr::{EncDecCTCModel, StreamingSession};
//!
//! let mut model = EncDecCTCModel::restore("model.nemo")?;
//!
//! // Check streaming modes
//! for mode in model.streaming_modes() {
//!     println!("Mode {}: latency={}ms", mode.mode_idx, mode.latency_ms);
//! }
//!
//! // Initialize streaming
//! let mut session = model.init_streaming(0, 1)?;
//!
//! // Process audio chunks
//! loop {
//!     let audio_chunk = get_audio_chunk(); // ~160ms of audio
//!     let output = model.process_chunk(&mut session, &audio_chunk)?;
//!     print!("{}", output.text);
//!     
//!     if end_of_audio {
//!         let final_output = model.finalize_streaming(session)?;
//!         println!("\nFinal: {}", final_output.text);
//!         break;
//!     }
//! }
//! ```

use mlx_rs::Array;

use crate::asr::config::EncoderConfig;
use crate::core::error::{Error, Result};

/// Streaming configuration computed from model config.
///
/// This mirrors NeMo's `CacheAwareStreamingConfig` dataclass.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Size of each chunk at each step (frames after subsampling).
    /// Tuple of (first_chunk_size, subsequent_chunk_size).
    pub chunk_size: (usize, usize),

    /// Size of shift in each step (frames).
    /// Tuple of (first_shift, subsequent_shift).
    pub shift_size: (usize, usize),

    /// Number of frames to drop from the cache per step.
    pub cache_drop_size: usize,

    /// Size of cache for MHA layers (attention context).
    pub last_channel_cache_size: usize,

    /// Size of cache for convolution layers (kernel_size - 1).
    pub last_time_cache_size: usize,

    /// Number of valid output frames per chunk (matching offline mode).
    pub valid_out_len: usize,

    /// Cache size for pre-encoding (subsampling).
    /// Tuple of (first_chunk_cache, subsequent_chunk_cache).
    pub pre_encode_cache_size: (usize, usize),

    /// Number of extra pre-encoded frames to drop.
    pub drop_extra_pre_encoded: usize,

    /// Lookahead frames required (future context).
    pub lookahead_frames: usize,

    /// Current streaming mode index.
    pub mode_idx: usize,

    /// Left attention context (frames).
    pub left_context: i32,

    /// Right attention context (frames).
    pub right_context: i32,
}

impl StreamingConfig {
    /// Get chunk size for a given step (first or subsequent).
    pub fn chunk_size_for_step(&self, is_first: bool) -> usize {
        if is_first {
            self.chunk_size.0
        } else {
            self.chunk_size.1
        }
    }

    /// Get shift size for a given step (first or subsequent).
    pub fn shift_size_for_step(&self, is_first: bool) -> usize {
        if is_first {
            self.shift_size.0
        } else {
            self.shift_size.1
        }
    }

    /// Get pre-encode cache size for a given step.
    pub fn pre_encode_cache_for_step(&self, is_first: bool) -> usize {
        if is_first {
            self.pre_encode_cache_size.0
        } else {
            self.pre_encode_cache_size.1
        }
    }
}

/// Encoder cache state for streaming inference.
///
/// Contains the three cache tensors needed for cache-aware streaming:
/// - Attention (channel) cache for key/value states
/// - Convolution (time) cache for causal conv left context
/// - Length tracking for proper masking
#[derive(Debug, Clone)]
pub struct EncoderCache {
    /// Cache for MHA layers: [num_layers, batch, cache_len, d_model]
    pub last_channel: Array,

    /// Cache for conv layers: [num_layers, batch, d_model, time_cache]
    pub last_time: Array,

    /// Actual fill level per sample: [batch]
    pub last_channel_len: Array,
}

impl EncoderCache {
    /// Create a new encoder cache with given dimensions.
    ///
    /// # Arguments
    /// * `num_layers` - Number of encoder layers
    /// * `batch_size` - Batch size
    /// * `channel_cache_size` - Size of attention cache
    /// * `time_cache_size` - Size of convolution cache (kernel_size - 1)
    /// * `d_model` - Model dimension
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        channel_cache_size: usize,
        time_cache_size: usize,
        d_model: usize,
    ) -> Result<Self> {
        let last_channel = mlx_rs::ops::zeros::<f32>(&[
            num_layers as i32,
            batch_size as i32,
            channel_cache_size as i32,
            d_model as i32,
        ])?;

        let last_time = mlx_rs::ops::zeros::<f32>(&[
            num_layers as i32,
            batch_size as i32,
            d_model as i32,
            time_cache_size as i32,
        ])?;

        let last_channel_len = mlx_rs::ops::zeros::<i64>(&[batch_size as i32])?;

        Ok(Self {
            last_channel,
            last_time,
            last_channel_len,
        })
    }

    /// Get the batch size from cache dimensions.
    pub fn batch_size(&self) -> usize {
        self.last_channel.dim(1) as usize
    }

    /// Get the number of layers from cache dimensions.
    pub fn num_layers(&self) -> usize {
        self.last_channel.dim(0) as usize
    }

    /// Get the channel cache size.
    pub fn channel_cache_size(&self) -> usize {
        self.last_channel.dim(2) as usize
    }

    /// Get the time cache size.
    pub fn time_cache_size(&self) -> usize {
        self.last_time.dim(3) as usize
    }
}

/// Information about a streaming mode.
#[derive(Debug, Clone)]
pub struct StreamingModeInfo {
    /// Index of this streaming mode.
    pub mode_idx: usize,

    /// Left attention context (frames, -1 for unlimited).
    pub left_context: i32,

    /// Right attention context / lookahead (frames).
    pub right_context: i32,

    /// Estimated latency in milliseconds (based on lookahead).
    pub latency_ms: f32,

    /// Human-readable description.
    pub description: String,
}

impl StreamingModeInfo {
    /// Create from context sizes.
    pub fn from_context(
        mode_idx: usize,
        left_context: i32,
        right_context: i32,
        frame_duration_ms: f32,
    ) -> Self {
        let latency_ms = if right_context >= 0 {
            right_context as f32 * frame_duration_ms
        } else {
            f32::INFINITY
        };

        let description = if right_context == 0 {
            format!("Zero lookahead (lowest latency), left={}", left_context)
        } else if right_context < 0 {
            format!("Unlimited lookahead, left={}", left_context)
        } else {
            format!(
                "Lookahead={} frames (~{:.0}ms), left={}",
                right_context, latency_ms, left_context
            )
        };

        Self {
            mode_idx,
            left_context,
            right_context,
            latency_ms,
            description,
        }
    }
}

/// Output from streaming inference.
#[derive(Debug, Clone)]
pub struct StreamingOutput {
    /// Log probabilities (if requested): [batch, time, vocab]
    pub log_probs: Option<Array>,

    /// Merged token IDs for this chunk.
    pub tokens: Vec<Vec<usize>>,

    /// Partial transcription text.
    pub text: Vec<String>,

    /// Whether this is the final output.
    pub is_final: bool,
}

impl Default for StreamingOutput {
    fn default() -> Self {
        Self {
            log_probs: None,
            tokens: Vec::new(),
            text: Vec::new(),
            is_final: false,
        }
    }
}

/// Streaming audio buffer that mirrors NeMo's CacheAwareStreamingAudioBuffer.
///
/// This buffer incrementally computes mel spectrograms from audio and provides chunks
/// with proper overlap (pre_encode_cache) for streaming inference.
///
/// # NeMo Streaming Flow
///
/// NeMo's streaming works as follows:
/// 1. Audio is converted to mel spectrogram incrementally
/// 2. Mel frames are stored in a buffer
/// 3. Chunks are extracted with overlap: `[pre_encode_cache frames | chunk_size frames]`
/// 4. Buffer index advances by `shift_size` frames
///
/// For example with mode 0:
/// - First chunk: 105 mel frames (pre_encode_cache=0)
/// - Subsequent: 112 + 9 = 121 mel frames (pre_encode_cache=9)
/// - Shift: 112 frames per step (after first)
///
/// # Memory Efficiency
///
/// This implementation uses incremental mel computation to maintain O(1) memory:
/// - Audio buffer only keeps unprocessed samples plus overlap for STFT windowing
/// - Mel buffer can be trimmed after chunks are extracted
#[derive(Debug, Clone)]
pub struct StreamingMelBuffer {
    /// Pre-computed mel frames stored as [time, n_mels] (internal format for efficient frame access)
    /// This is transposed from the preprocessor output [batch, n_mels, time]
    pub mel_buffer: Vec<f32>,

    /// Number of mel features (e.g., 128)
    pub n_mels: usize,

    /// Total mel frames in buffer (including any that have been trimmed from the start)
    pub total_frames: usize,

    /// Current position in mel buffer (analogous to NeMo's buffer_idx)
    pub buffer_idx: usize,

    /// Sliding window audio buffer - only keeps samples needed for next mel computation.
    /// Contains: overlap samples from previous computation + new unprocessed samples.
    audio_buffer: Vec<f32>,

    /// Total number of audio samples that have been fed to this stream.
    /// This is cumulative and never decreases - used for frame count calculations.
    pub total_audio_samples: usize,

    /// Sample rate
    pub sample_rate: u32,

    /// Window size in samples
    pub window_samples: usize,

    /// Hop size in samples
    pub hop_samples: usize,

    /// FFT size
    pub n_fft: usize,

    /// Offset for mel_buffer indexing after trimming.
    /// When we trim old mel frames, this tracks how many frames were removed.
    mel_buffer_offset: usize,
}

impl StreamingMelBuffer {
    /// Create a new streaming mel buffer.
    pub fn new(
        sample_rate: u32,
        window_size: f32,
        window_stride: f32,
        n_mels: usize,
        n_fft: usize,
    ) -> Self {
        let window_samples = (window_size * sample_rate as f32) as usize;
        let hop_samples = (window_stride * sample_rate as f32) as usize;

        Self {
            mel_buffer: Vec::new(),
            n_mels,
            total_frames: 0,
            buffer_idx: 0,
            audio_buffer: Vec::new(),
            total_audio_samples: 0,
            sample_rate,
            window_samples,
            hop_samples,
            n_fft,
            mel_buffer_offset: 0,
        }
    }

    /// Add audio samples to the buffer for incremental processing.
    pub fn add_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
        self.total_audio_samples += samples.len();
    }

    /// Get the pending audio samples that need mel computation.
    /// Returns audio with necessary overlap for STFT continuity.
    pub fn get_pending_audio(&self) -> &[f32] {
        &self.audio_buffer
    }

    /// Get total audio samples in buffer (pending processing).
    pub fn audio_samples_available(&self) -> usize {
        self.audio_buffer.len()
    }

    /// Check if there is enough pending audio to compute at least one mel frame.
    ///
    /// We need at least window_samples worth of audio to compute any frames.
    pub fn has_pending_audio(&self) -> bool {
        self.audio_buffer.len() >= self.window_samples
    }

    /// Append pre-computed mel frames to the buffer.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram data in [batch=1, n_mels, time] format (flattened row-major)
    /// * `num_frames` - Number of time frames
    ///
    /// The input is in row-major [batch, n_mels, time] format from the preprocessor.
    /// We store internally as [time, n_mels] for efficient frame-by-frame access.
    pub fn append_mel(&mut self, mel: &[f32], num_frames: usize) {
        // Input mel is [1, n_mels, time] flattened row-major, so:
        // mel[mel_idx * time + frame_idx] = feature mel_idx at frame frame_idx
        //
        // We want to store as [time, n_mels] for easier frame extraction:
        // buffer[frame_idx * n_mels + mel_idx] = feature mel_idx at frame frame_idx
        let n_mels = self.n_mels;
        for frame_idx in 0..num_frames {
            for mel_idx in 0..n_mels {
                // Input index: [batch=0, mel_idx, frame_idx] = mel_idx * num_frames + frame_idx
                let input_idx = mel_idx * num_frames + frame_idx;
                if input_idx < mel.len() {
                    self.mel_buffer.push(mel[input_idx]);
                }
            }
        }
        self.total_frames += num_frames;
    }

    /// Drain excess audio samples from buffer, keeping enough for STFT overlap.
    ///
    /// After computing mel frames, call this to free memory from audio samples
    /// that are no longer needed. We keep window_samples worth of overlap to
    /// ensure STFT continuity at chunk boundaries.
    pub fn drain_processed_audio(&mut self) {
        // Keep enough samples for STFT overlap (window_samples worth)
        // This ensures continuity at chunk boundaries
        let keep_samples = self.window_samples;

        if self.audio_buffer.len() > keep_samples {
            let drain_count = self.audio_buffer.len() - keep_samples;
            self.audio_buffer.drain(..drain_count);
        }
    }

    /// Trim old mel frames that are no longer needed.
    ///
    /// Call this periodically to free memory from mel frames that have been
    /// processed and are no longer needed for pre_encode_cache.
    ///
    /// # Arguments
    /// * `keep_frames` - Minimum number of frames to keep before buffer_idx
    ///                   (should be >= max pre_encode_cache size)
    pub fn trim_old_frames(&mut self, keep_frames: usize) {
        // Calculate how many frames we can safely remove
        // We need to keep at least `keep_frames` before the current buffer_idx
        let buffer_local_idx = self.buffer_idx.saturating_sub(self.mel_buffer_offset);
        let can_trim = buffer_local_idx.saturating_sub(keep_frames);

        if can_trim > 0 {
            // Remove old frames from the beginning of mel_buffer
            let elements_to_remove = can_trim * self.n_mels;
            self.mel_buffer.drain(..elements_to_remove);
            self.mel_buffer_offset += can_trim;
        }
    }

    /// Get the number of remaining mel frames to process.
    pub fn remaining_frames(&self) -> usize {
        self.total_frames.saturating_sub(self.buffer_idx)
    }

    /// Check if the buffer is empty (no more frames to process).
    pub fn is_empty(&self) -> bool {
        self.buffer_idx >= self.total_frames
    }

    /// Extract a chunk of mel frames for processing.
    ///
    /// This mirrors NeMo's CacheAwareStreamingAudioBuffer iteration:
    /// - Extracts `pre_encode_cache + chunk_size` frames starting from `buffer_idx - pre_encode_cache`
    /// - Advances `buffer_idx` by `shift_size`
    ///
    /// # Arguments
    /// * `chunk_size` - Number of new mel frames in the chunk
    /// * `shift_size` - Number of frames to advance
    /// * `pre_encode_cache` - Number of frames to prepend from previous context
    ///
    /// # Returns
    /// * `Some((mel_data, actual_length))` - Mel chunk in [n_mels, time] format for encoder input
    /// * `None` if not enough frames available
    pub fn extract_chunk(
        &mut self,
        chunk_size: usize,
        shift_size: usize,
        pre_encode_cache: usize,
    ) -> Option<(Vec<f32>, usize)> {
        // Check if we have enough frames for this chunk
        if self.buffer_idx + chunk_size > self.total_frames {
            // Not enough frames - check if we have any remaining for final chunk
            if self.buffer_idx >= self.total_frames {
                return None;
            }
            tracing::warn!("PARTIAL CHUNK!!");
            // Handle partial final chunk
            let remaining = self.total_frames - self.buffer_idx;
            return self.extract_chunk(remaining, remaining, pre_encode_cache);
        }

        // Calculate start position accounting for pre_encode_cache
        // Also account for mel_buffer_offset when frames have been trimmed
        let cache_start = self.buffer_idx.saturating_sub(pre_encode_cache);
        let cache_frames_available = self.buffer_idx - cache_start;

        // Total frames to extract: cache + chunk
        let total_extract = cache_frames_available + chunk_size;
        tracing::trace!("total_extract {}", total_extract);

        // Internal buffer is [time, n_mels]: buffer[frame * n_mels + mel_idx]
        // Output needs to be [n_mels, time] for encoder: output[mel_idx * time + frame_idx]
        let mut mel_chunk = vec![0.0f32; self.n_mels * total_extract];

        for frame_offset in 0..total_extract {
            // Global frame index
            let frame = cache_start + frame_offset;
            // Local index in mel_buffer (accounting for trimmed frames)
            let local_frame = frame.saturating_sub(self.mel_buffer_offset);

            for mel_idx in 0..self.n_mels {
                // Source: buffer[local_frame * n_mels + mel_idx]
                let src_idx = local_frame * self.n_mels + mel_idx;
                // Destination: output[mel_idx * total_extract + frame_idx]
                let dst_idx = mel_idx * total_extract + frame_offset;
                if src_idx < self.mel_buffer.len() {
                    mel_chunk[dst_idx] = self.mel_buffer[src_idx];
                }
            }
        }

        // Advance buffer index
        self.buffer_idx += shift_size;

        Some((mel_chunk, total_extract))
    }

    /// Reset the buffer for a new audio stream.
    pub fn reset(&mut self) {
        self.mel_buffer.clear();
        self.total_frames = 0;
        self.buffer_idx = 0;
        self.audio_buffer.clear();
        self.total_audio_samples = 0;
        self.mel_buffer_offset = 0;
    }

    /// Get memory usage statistics for debugging.
    pub fn memory_stats(&self) -> (usize, usize) {
        let audio_bytes = self.audio_buffer.len() * std::mem::size_of::<f32>();
        let mel_bytes = self.mel_buffer.len() * std::mem::size_of::<f32>();
        (audio_bytes, mel_bytes)
    }
}

/// Legacy MelBuffer for backward compatibility with CTC streaming.
/// Wraps audio buffering with running normalization stats.
#[derive(Debug, Clone)]
pub struct MelBuffer {
    /// Accumulated audio samples.
    pub audio_buffer: Vec<f32>,

    /// Running sum for mean computation.
    pub feature_sum: Vec<f32>,

    /// Running sum of squares for std computation.
    pub feature_sum_sq: Vec<f32>,

    /// Number of frames processed (for running stats).
    pub frames_processed: usize,

    /// Sample rate.
    pub sample_rate: u32,

    /// Window size in samples.
    pub window_samples: usize,

    /// Hop size in samples.
    pub hop_samples: usize,

    /// Number of mel features.
    pub n_mels: usize,
}

impl MelBuffer {
    /// Create a new mel buffer.
    pub fn new(sample_rate: u32, window_size: f32, window_stride: f32, n_mels: usize) -> Self {
        let window_samples = (window_size * sample_rate as f32) as usize;
        let hop_samples = (window_stride * sample_rate as f32) as usize;

        Self {
            audio_buffer: Vec::new(),
            feature_sum: vec![0.0; n_mels],
            feature_sum_sq: vec![0.0; n_mels],
            frames_processed: 0,
            sample_rate,
            window_samples,
            hop_samples,
            n_mels,
        }
    }

    /// Add audio samples to the buffer.
    pub fn add_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
    }

    /// Get the number of complete frames available.
    pub fn available_frames(&self) -> usize {
        if self.audio_buffer.len() < self.window_samples {
            0
        } else {
            (self.audio_buffer.len() - self.window_samples) / self.hop_samples + 1
        }
    }

    /// Consume audio for a given number of mel frames.
    ///
    /// For streaming, we need to consume `chunk_size` frames worth of audio
    /// but only advance by `shift_size` frames worth.
    ///
    /// # Arguments
    /// * `chunk_frames` - Number of mel frames to extract
    /// * `shift_frames` - Number of mel frames to advance the buffer
    ///
    /// # Returns
    /// Audio samples for computing `chunk_frames` mel frames
    pub fn consume_frames(&mut self, chunk_frames: usize, shift_frames: usize) -> Vec<f32> {
        // Samples needed to compute chunk_frames mel frames
        let samples_needed =
            self.window_samples + (chunk_frames.saturating_sub(1)) * self.hop_samples;
        let samples_to_take = samples_needed.min(self.audio_buffer.len());

        // Copy the audio (don't drain yet)
        let audio: Vec<f32> = self.audio_buffer[..samples_to_take].to_vec();

        // Advance by shift_frames worth of samples
        let samples_to_advance = shift_frames * self.hop_samples;
        let drain_count = samples_to_advance.min(self.audio_buffer.len());
        self.audio_buffer.drain(..drain_count);

        audio
    }

    /// Update running feature statistics for normalization.
    pub fn update_stats(&mut self, features: &[f32]) {
        // features: [n_mels * num_frames] in row-major order
        let num_frames = features.len() / self.n_mels;

        for frame_idx in 0..num_frames {
            for mel_idx in 0..self.n_mels {
                let val = features[frame_idx * self.n_mels + mel_idx];
                self.feature_sum[mel_idx] += val;
                self.feature_sum_sq[mel_idx] += val * val;
            }
        }
        self.frames_processed += num_frames;
    }

    /// Get current mean and std for normalization.
    pub fn get_norm_stats(&self) -> (Vec<f32>, Vec<f32>) {
        if self.frames_processed == 0 {
            return (vec![0.0; self.n_mels], vec![1.0; self.n_mels]);
        }

        let n = self.frames_processed as f32;
        let mean: Vec<f32> = self.feature_sum.iter().map(|s| s / n).collect();
        let std: Vec<f32> = self
            .feature_sum_sq
            .iter()
            .zip(&self.feature_sum)
            .map(|(sq, s)| {
                let variance = (sq / n) - (s / n).powi(2);
                variance.max(1e-10).sqrt()
            })
            .collect();

        (mean, std)
    }

    /// Reset the buffer state.
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.feature_sum.fill(0.0);
        self.feature_sum_sq.fill(0.0);
        self.frames_processed = 0;
    }
}

/// Compute streaming configuration from encoder config.
///
/// This implements NeMo's `setup_streaming_params` logic.
pub fn compute_streaming_config(
    encoder_config: &EncoderConfig,
    mode_idx: usize,
    max_context: usize,
) -> Result<StreamingConfig> {
    // Get attention context size for the selected mode
    let (left_context, right_context) = encoder_config
        .att_context_size
        .as_ref()
        .and_then(|cs| cs.get_mode(mode_idx))
        .ok_or_else(|| {
            Error::config(format!(
                "Invalid streaming mode index {} or no att_context_size configured",
                mode_idx
            ))
        })?;

    let att_context_style = encoder_config
        .att_context_style
        .as_deref()
        .unwrap_or("regular");

    let n_layers = encoder_config.n_layers;
    let subsampling_factor = encoder_config.subsampling_factor;

    // Compute conv context size
    let conv_kernel = encoder_config.conv_kernel_size;
    let conv_context = if encoder_config.is_causal_conv() {
        (conv_kernel - 1, 0) // all padding on left for causal
    } else {
        ((conv_kernel - 1) / 2, (conv_kernel - 1) / 2)
    };

    // Compute lookahead_steps and cache_drop_size based on attention style
    // This follows NeMo's setup_streaming_params() logic exactly
    let (lookahead_steps, cache_drop_size) = if att_context_style == "chunked_limited" {
        // Chunked limited: lookahead is just right context, no cache drop
        (right_context.max(0) as usize, 0)
    } else {
        // Regular style: lookahead accumulates through layers
        let lookahead = if right_context >= 0 {
            (right_context as usize) * n_layers + conv_context.1 * n_layers
        } else {
            0
        };
        (lookahead, lookahead)
    };

    // Channel cache size
    let last_channel_cache_size = if left_context >= 0 {
        left_context as usize
    } else {
        max_context
    };

    // Time cache size is conv_kernel - 1 for causal convolutions
    let last_time_cache_size = conv_context.0;

    // Compute sampling frames from subsampling
    // NeMo's ConvSubsampling.get_sampling_frames() returns [1, subsampling_factor]
    let sampling_frames = (1, subsampling_factor);

    // Chunk size in pre-subsampled frames
    // NeMo: chunk_size = [sampling_frames[0] + subsampling_factor * lookahead_steps,
    //                     sampling_frames[1] + subsampling_factor * lookahead_steps]
    let chunk_size = (
        sampling_frames.0 + subsampling_factor * lookahead_steps,
        sampling_frames.1 + subsampling_factor * lookahead_steps,
    );

    // Shift size - how many pre-subsampled frames to advance each step
    let shift_size = (
        sampling_frames.0 + sampling_frames.1 * lookahead_steps.saturating_sub(cache_drop_size),
        sampling_frames.1 + sampling_frames.1 * lookahead_steps.saturating_sub(cache_drop_size),
    );

    // Valid output length - how many OUTPUT frames are valid per chunk
    let valid_out_len = if sampling_frames.1 > 0 {
        (shift_size.1 - sampling_frames.1) / subsampling_factor + 1
    } else {
        shift_size.1 / subsampling_factor
    };

    // Pre-encode cache size
    // NeMo's dw_striding subsampling returns [0, subsampling_factor + 1]
    // This is different from what we had before!
    let pre_encode_cache = (0, subsampling_factor + 1);

    // Drop extra pre-encoded frames after subsampling
    // NeMo: drop_extra_pre_encoded = 1 + (pre_encode_cache_size[1] - 1) // subsampling_factor
    let drop_extra = if pre_encode_cache.1 >= 1 {
        1 + (pre_encode_cache.1 - 1) / subsampling_factor
    } else {
        0
    };

    tracing::debug!(
        "Streaming config: mode={}, style={}, lookahead_steps={}, cache_drop={}, \
         chunk_size={:?}, shift_size={:?}, valid_out_len={}, pre_encode={:?}, drop_extra={}",
        mode_idx,
        att_context_style,
        lookahead_steps,
        cache_drop_size,
        chunk_size,
        shift_size,
        valid_out_len,
        pre_encode_cache,
        drop_extra
    );

    Ok(StreamingConfig {
        chunk_size,
        shift_size,
        cache_drop_size,
        last_channel_cache_size,
        last_time_cache_size,
        valid_out_len,
        pre_encode_cache_size: pre_encode_cache,
        drop_extra_pre_encoded: drop_extra,
        lookahead_frames: lookahead_steps,
        mode_idx,
        left_context,
        right_context,
    })
}

/// Get all available streaming modes from encoder config.
pub fn get_streaming_modes(
    encoder_config: &EncoderConfig,
    frame_duration_ms: f32,
) -> Vec<StreamingModeInfo> {
    let att_context_size = match &encoder_config.att_context_size {
        Some(cs) => cs,
        None => return Vec::new(),
    };

    let num_modes = att_context_size.num_modes();
    let mut modes = Vec::with_capacity(num_modes);

    for mode_idx in 0..num_modes {
        if let Some((left, right)) = att_context_size.get_mode(mode_idx) {
            modes.push(StreamingModeInfo::from_context(
                mode_idx,
                left,
                right,
                frame_duration_ms,
            ));
        }
    }

    modes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_cache_creation() {
        let cache = EncoderCache::new(24, 1, 70, 8, 1024).unwrap();
        assert_eq!(cache.num_layers(), 24);
        assert_eq!(cache.batch_size(), 1);
        assert_eq!(cache.channel_cache_size(), 70);
        assert_eq!(cache.time_cache_size(), 8);
    }

    #[test]
    fn test_mel_buffer() {
        let mut buffer = MelBuffer::new(16000, 0.025, 0.01, 80);
        assert_eq!(buffer.window_samples, 400);
        assert_eq!(buffer.hop_samples, 160);

        // Add 1 second of audio
        buffer.add_audio(&vec![0.0f32; 16000]);

        // Should have (16000 - 400) / 160 + 1 = 98 frames available
        assert_eq!(buffer.available_frames(), 98);
    }

    #[test]
    fn test_streaming_mode_info() {
        let mode = StreamingModeInfo::from_context(0, 70, 13, 10.0);
        assert_eq!(mode.mode_idx, 0);
        assert_eq!(mode.left_context, 70);
        assert_eq!(mode.right_context, 13);
        assert!((mode.latency_ms - 130.0).abs() < 0.1);
    }

    #[test]
    fn test_streaming_config_chunk_sizes() {
        let config = StreamingConfig {
            chunk_size: (100, 120),
            shift_size: (80, 100),
            cache_drop_size: 0,
            last_channel_cache_size: 70,
            last_time_cache_size: 8,
            valid_out_len: 10,
            pre_encode_cache_size: (0, 9),
            drop_extra_pre_encoded: 2,
            lookahead_frames: 13,
            mode_idx: 0,
            left_context: 70,
            right_context: 13,
        };

        assert_eq!(config.chunk_size_for_step(true), 100);
        assert_eq!(config.chunk_size_for_step(false), 120);
        assert_eq!(config.shift_size_for_step(true), 80);
        assert_eq!(config.shift_size_for_step(false), 100);
        assert_eq!(config.pre_encode_cache_for_step(true), 0);
        assert_eq!(config.pre_encode_cache_for_step(false), 9);
    }

    #[test]
    fn test_streaming_mel_buffer_incremental() {
        let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 80, 512);
        assert_eq!(buffer.window_samples, 400);
        assert_eq!(buffer.hop_samples, 160);

        // Add 1 second of audio
        buffer.add_audio(&vec![0.0f32; 16000]);
        assert_eq!(buffer.audio_samples_available(), 16000);
        assert!(buffer.has_pending_audio(), "Should have pending audio");

        // Simulate appending mel frames (what preprocessor would return)
        // 16000 samples -> floor(16000/160) + 1 = 101 frames
        let dummy_mel = vec![0.0f32; 80 * 101]; // [n_mels, time] format
        buffer.append_mel(&dummy_mel, 101);
        assert_eq!(buffer.total_frames, 101);

        // Drain processed audio
        buffer.drain_processed_audio();

        // After processing, audio buffer should be trimmed to just overlap
        assert!(
            buffer.audio_samples_available() <= buffer.window_samples,
            "Audio buffer should be bounded after processing"
        );

        // Add more audio
        buffer.add_audio(&vec![0.0f32; 1600]); // 0.1s more

        // Should still have pending audio (overlap + new = 400 + 1600 = 2000 samples)
        assert!(
            buffer.has_pending_audio(),
            "Should have pending audio after adding more"
        );
        assert_eq!(buffer.audio_samples_available(), 2000);
    }

    #[test]
    fn test_streaming_mel_buffer_trim() {
        let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 80, 512);

        // Add mel frames
        let dummy_mel = vec![1.0f32; 80 * 100];
        buffer.append_mel(&dummy_mel, 100);
        assert_eq!(buffer.total_frames, 100);

        // Advance buffer_idx (simulating chunk extraction)
        buffer.buffer_idx = 50;

        // Trim, keeping 10 frames before buffer_idx
        buffer.trim_old_frames(10);

        // Should have trimmed 40 frames (50 - 10)
        assert_eq!(buffer.mel_buffer_offset, 40);

        // Mel buffer should now have 60 frames worth of data (100 - 40)
        assert_eq!(buffer.mel_buffer.len(), 60 * 80);

        // total_frames should still reflect logical count
        assert_eq!(buffer.total_frames, 100);
    }

    #[test]
    fn test_streaming_mel_buffer_memory_bounded() {
        let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 80, 512);

        // Simulate a long stream by repeatedly adding and processing audio
        for _ in 0..100 {
            // Add 0.1s of audio
            buffer.add_audio(&vec![0.0f32; 1600]);

            // Simulate mel computation and processing
            if buffer.has_pending_audio() {
                // Compute how many frames from current audio buffer
                // Formula: audio_len / hop + 1
                let audio_len = buffer.audio_samples_available();
                let computed_frames = audio_len / buffer.hop_samples + 1;

                // Skip overlap frames (except on first iteration)
                let overlap_frames = if buffer.total_frames == 0 {
                    0
                } else {
                    buffer.window_samples / buffer.hop_samples
                };
                let new_frames = computed_frames.saturating_sub(overlap_frames);

                if new_frames > 0 {
                    let dummy_mel = vec![0.0f32; 80 * new_frames];
                    buffer.append_mel(&dummy_mel, new_frames);
                    buffer.drain_processed_audio();

                    // Advance buffer_idx
                    buffer.buffer_idx = buffer.total_frames;

                    // Trim old frames (keep 20 for cache)
                    buffer.trim_old_frames(20);
                }
            }
        }

        // After processing 10 seconds of audio, memory should still be bounded
        let (audio_bytes, mel_bytes) = buffer.memory_stats();

        // Audio buffer should be O(window_samples) ~ 400 * 4 = 1.6KB
        assert!(
            audio_bytes < 10_000,
            "Audio buffer should be bounded, got {} bytes",
            audio_bytes
        );

        // Mel buffer should be O(keep_frames * n_mels) ~ 20 * 80 * 4 = 6.4KB
        assert!(
            mel_bytes < 50_000,
            "Mel buffer should be bounded, got {} bytes",
            mel_bytes
        );
    }
}
