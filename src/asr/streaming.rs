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

/// Buffer for incremental mel spectrogram computation.
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

    /// Consume audio for a given number of frames.
    /// Returns the audio slice and updates the buffer, keeping proper overlap.
    ///
    /// The STFT requires `window_samples` of audio for each frame, but frames
    /// advance by `hop_samples`. So for `num_frames` frames we need:
    ///   `window_samples + (num_frames - 1) * hop_samples` samples
    ///
    /// After processing, we keep `window_samples - hop_samples` samples of overlap
    /// for the next chunk, which corresponds to advancing by `num_frames * hop_samples`.
    pub fn consume_frames(&mut self, num_frames: usize) -> Vec<f32> {
        // Total samples needed to produce num_frames
        let samples_needed =
            self.window_samples + (num_frames.saturating_sub(1)) * self.hop_samples;
        let samples_to_take = samples_needed.min(self.audio_buffer.len());

        // Copy out the audio for processing (don't drain yet)
        let audio: Vec<f32> = self.audio_buffer[..samples_to_take].to_vec();

        // Advance the buffer by the number of samples that correspond to num_frames
        // This keeps the overlap (window_samples - hop_samples) for the next chunk
        let samples_to_advance = num_frames * self.hop_samples;
        let samples_to_drain = samples_to_advance.min(self.audio_buffer.len());
        self.audio_buffer.drain(..samples_to_drain);

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
        // lookahead_steps = att_context_size[1] (right context)
        (right_context.max(0) as usize, 0)
    } else {
        // Regular style: lookahead accumulates through layers
        // lookahead_steps = att_context_size[1] * n_layers + conv_context[1] * n_layers
        let lookahead = if right_context >= 0 {
            (right_context as usize) * n_layers + conv_context.1 * n_layers
        } else {
            0
        };
        // cache_drop_size = lookahead_steps for regular style
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
    // For dw_striding with factor 8: sampling_frames = [7, 8] (first, subsequent)
    // This matches NeMo's ConvSubsampling.get_sampling_frames()
    let sampling_frames = match subsampling_factor {
        2 => (1, 2),
        4 => (3, 4),
        8 => (7, 8),
        _ => (subsampling_factor - 1, subsampling_factor),
    };

    // Chunk size in pre-subsampled frames
    // NeMo: chunk_size = [sampling_frames[0] + subsampling_factor * lookahead_steps,
    //                     sampling_frames[1] + subsampling_factor * lookahead_steps]
    let chunk_size = (
        sampling_frames.0 + subsampling_factor * lookahead_steps,
        sampling_frames.1 + subsampling_factor * lookahead_steps,
    );

    // Shift size - how many pre-subsampled frames to advance each step
    // NeMo: shift_size = [sampling_frames[0] + sampling_frames[1] * (lookahead_steps - cache_drop_size),
    //                     sampling_frames[1] + sampling_frames[1] * (lookahead_steps - cache_drop_size)]
    let shift_size = (
        sampling_frames.0 + sampling_frames.1 * lookahead_steps.saturating_sub(cache_drop_size),
        sampling_frames.1 + sampling_frames.1 * lookahead_steps.saturating_sub(cache_drop_size),
    );

    // Valid output length - how many OUTPUT frames are valid per chunk
    // NeMo: valid_out_len = (shift_size[1] - sampling_frames[1]) // subsampling_factor + 1
    // For chunked_limited with lookahead_steps = right_context and cache_drop_size = 0:
    //   shift_size[1] = sampling_frames[1] + sampling_frames[1] * lookahead_steps
    //                 = sampling_frames[1] * (1 + lookahead_steps)
    //   valid_out_len = (sampling_frames[1] * (1 + lookahead_steps) - sampling_frames[1]) / subsampling_factor + 1
    //                 = (sampling_frames[1] * lookahead_steps) / subsampling_factor + 1
    //                 = lookahead_steps + 1 (when sampling_frames[1] == subsampling_factor)
    let valid_out_len = if sampling_frames.1 > 0 {
        (shift_size.1 - sampling_frames.1) / subsampling_factor + 1
    } else {
        shift_size.1 / subsampling_factor
    };

    // Pre-encode cache size (for subsampling overlap)
    // For dw_striding: cache = kernel_size - 1 for each conv layer
    let pre_encode_cache = match subsampling_factor {
        8 => (6, 7), // 3 conv layers with kernel 3
        4 => (4, 5),
        2 => (2, 3),
        _ => (0, 0),
    };

    // Drop extra pre-encoded frames after subsampling
    // NeMo: drop_extra_pre_encoded = 1 + (pre_encode_cache_size[1] - 1) // subsampling_factor
    let drop_extra = if pre_encode_cache.1 >= 1 {
        1 + (pre_encode_cache.1 - 1) / subsampling_factor
    } else {
        0
    };

    tracing::debug!(
        "Streaming config: mode={}, style={}, lookahead_steps={}, cache_drop={}, \
         chunk_size={:?}, shift_size={:?}, valid_out_len={}, drop_extra={}",
        mode_idx,
        att_context_style,
        lookahead_steps,
        cache_drop_size,
        chunk_size,
        shift_size,
        valid_out_len,
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
            pre_encode_cache_size: (6, 7),
            drop_extra_pre_encoded: 1,
            lookahead_frames: 13,
            mode_idx: 0,
            left_context: 70,
            right_context: 13,
        };

        assert_eq!(config.chunk_size_for_step(true), 100);
        assert_eq!(config.chunk_size_for_step(false), 120);
        assert_eq!(config.shift_size_for_step(true), 80);
        assert_eq!(config.shift_size_for_step(false), 100);
    }
}
