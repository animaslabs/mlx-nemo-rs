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
/// This buffer stores mel spectrogram frames and provides chunks with proper
/// overlap (pre_encode_cache) for streaming inference.
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
/// # MLX Array Storage
///
/// Mel features are stored as MLX Arrays in `[n_mels, time]` format for efficient
/// GPU operations. Since MLX uses lazy evaluation, concatenation overhead is minimal.
#[derive(Debug, Clone)]
pub struct StreamingMelBuffer {
    /// Pre-computed mel frames stored as [n_mels, time] MLX Array.
    pub mel_buffer: Option<Array>,

    /// Number of mel features (e.g., 128)
    pub n_mels: usize,

    /// Total mel frames in buffer (including any that have been trimmed from the start)
    pub total_frames: usize,

    /// Current position in mel buffer (analogous to NeMo's buffer_idx)
    pub buffer_idx: usize,

    /// Total number of audio samples that have been fed to this stream.
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
            mel_buffer: None,
            n_mels,
            total_frames: 0,
            buffer_idx: 0,
            total_audio_samples: 0,
            sample_rate,
            window_samples,
            hop_samples,
            n_fft,
            mel_buffer_offset: 0,
        }
    }

    /// Append pre-computed mel frames to the buffer.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram Array in [n_mels, time] format from preprocessor
    pub fn append_mel(&mut self, mel: &Array) -> Result<()> {
        let num_frames = mel.dim(1) as usize;
        if num_frames == 0 {
            return Ok(());
        }

        self.total_frames += num_frames;

        // Concatenate with existing buffer
        self.mel_buffer = match &self.mel_buffer {
            Some(existing) => Some(mlx_rs::ops::concatenate_axis(&[existing, mel], 1)?),
            None => Some(mel.clone()),
        };

        Ok(())
    }

    /// Trim old mel frames that are no longer needed.
    ///
    /// Call this periodically to free memory from mel frames that have been
    /// processed and are no longer needed for pre_encode_cache.
    ///
    /// # Arguments
    /// * `keep_frames` - Minimum number of frames to keep before buffer_idx
    ///                   (should be >= max pre_encode_cache size)
    pub fn trim_old_frames(&mut self, keep_frames: usize) -> Result<()> {
        let mel_buffer = match &self.mel_buffer {
            Some(buf) => buf,
            None => return Ok(()),
        };

        // Calculate how many frames we can safely remove
        let buffer_local_idx = self.buffer_idx.saturating_sub(self.mel_buffer_offset);
        let can_trim = buffer_local_idx.saturating_sub(keep_frames);

        if can_trim > 0 {
            use mlx_rs::ops::indexing::IndexOp;
            let trimmed = mel_buffer.index((.., can_trim as i32..));
            self.mel_buffer = Some(trimmed);
            self.mel_buffer_offset += can_trim;
        }

        Ok(())
    }

    /// Get the number of remaining mel frames to process.
    pub fn remaining_frames(&self) -> usize {
        self.total_frames.saturating_sub(self.buffer_idx)
    }

    /// Check if the buffer is empty (no more frames to process).
    pub fn is_empty(&self) -> bool {
        self.buffer_idx >= self.total_frames
    }

    /// Get the current number of frames stored in the buffer.
    pub fn buffer_frames(&self) -> usize {
        self.mel_buffer
            .as_ref()
            .map(|b| b.dim(1) as usize)
            .unwrap_or(0)
    }

    /// Extract a chunk of mel frames for processing.
    ///
    /// This mirrors NeMo's CacheAwareStreamingAudioBuffer iteration:
    /// - Extracts `pre_encode_cache + chunk_size` frames
    /// - Advances `buffer_idx` by `shift_size`
    ///
    /// # Arguments
    /// * `chunk_size` - Number of new mel frames in the chunk
    /// * `shift_size` - Number of frames to advance
    /// * `pre_encode_cache` - Number of frames to prepend from previous context
    ///
    /// # Returns
    /// * `Some((mel_array, total_frames, valid_frames))` - Mel chunk Array in [n_mels, time] format
    /// * `None` if not enough frames available
    pub fn extract_chunk(
        &mut self,
        chunk_size: usize,
        shift_size: usize,
        pre_encode_cache: usize,
    ) -> Option<(Array, usize, usize)> {

        // Check if we have enough frames for this chunk
        // For the first chunk (buffer_idx=0), we just need chunk_size frames (no overlap yet)
        // For subsequent chunks, we need chunk_size new frames (pre_encode_cache comes from overlap)
        let frames_available = self.total_frames.saturating_sub(self.buffer_idx);

        // First chunk doesn't need pre_encode_cache (nothing to overlap with)
        // Subsequent chunks get pre_encode_cache from overlap with previous chunk
        let frames_required = chunk_size;

        if frames_available < frames_required {
            // // Not enough frames for a full chunk
            // if self.buffer_idx >= self.total_frames {
            //     return None;
            // }
            tracing::warn!("PARTIAL CHUNK!!");
            // Handle partial final chunk
            let remaining = frames_available;
            return self.extract_chunk(remaining, remaining, pre_encode_cache);
        }

        let mel_buffer = self.mel_buffer.as_ref()?;

        // Calculate extraction range
        // We extract chunk_size + pre_encode_cache frames starting from buffer_idx.
        // The pre_encode_cache provides context for the encoder's subsampling layer.
        //
        // buffer_idx tracks the extraction start position. By advancing by shift_size
        // (not the full extraction size), we maintain proper overlap between chunks.
        let extract_start = self.buffer_idx;
        let extract_end = self.total_frames.min(extract_start + chunk_size + pre_encode_cache);

        // Total frames to extract
        let total_extract = extract_end - extract_start;

        tracing::trace!(
            self.buffer_idx,
            self.total_frames,
            self.mel_buffer_offset,
            frames_available,
            frames_required,
            extract_start,
            extract_end,
            total_extract,
            chunk_size,
            pre_encode_cache,
            "Extracting mel frames"
        );

        // Convert global frame indices to local buffer indices
        let local_start = extract_start.saturating_sub(self.mel_buffer_offset);
        let local_end = local_start + total_extract;

        tracing::trace!(
            local_start,
            local_end,
            mel_buffer_shape = ?mel_buffer.shape(),
            "Local extraction indices"
        );

        // Extract slice from buffer: mel_buffer[:, local_start:local_end]
        use mlx_rs::ops::indexing::IndexOp;
        let mel_chunk = mel_buffer.index((.., local_start as i32..local_end as i32));

        // Advance buffer index by shift_size, NOT total_extract.
        // This matches NeMo's streaming behavior where the buffer advances by shift_size
        // each step, creating overlap between consecutive chunks.
        // The overlap is essential for proper streaming recognition.
        self.buffer_idx += shift_size;

        // All extracted frames are valid (matching Python's expected_feature_buffer_len)
        let valid_frames = total_extract;

        Some((mel_chunk, total_extract, valid_frames))
    }

    /// Reset the buffer for a new audio stream.
    pub fn reset(&mut self) {
        tracing::debug!("StreamingMelBuffer::reset");
        self.mel_buffer = None;
        self.total_frames = 0;
        self.buffer_idx = 0;
        self.total_audio_samples = 0;
        self.mel_buffer_offset = 0;
    }

    /// Initialize the buffer with silence frames.
    ///
    /// This matches Python's BatchedCacheFeatureBufferer which initializes
    /// the feature_buffer to LOG_MEL_ZERO. The silence value is computed as
    /// ln(2^-24) ≈ -16.6355 to match NeMo's log_zero_guard_value.
    ///
    /// # Arguments
    /// * `n_frames` - Number of silence frames to initialize
    pub fn initialize_with_silence(&mut self, n_frames: usize) -> Result<()> {
        if n_frames == 0 {
            return Ok(());
        }

        // LOG_MEL_ZERO = ln(2^-24) ≈ -16.6355 (matches NeMo's log_zero_guard)
        const LOG_MEL_ZERO: f32 = -16.635532;

        let silence_mel = mlx_rs::ops::full::<f32>(
            &[self.n_mels as i32, n_frames as i32],
            &mlx_rs::Array::from_f32(LOG_MEL_ZERO),
        )?;
        self.append_mel(&silence_mel)?;

        tracing::debug!(
            n_frames,
            LOG_MEL_ZERO,
            "Initialized mel buffer with silence frames"
        );

        Ok(())
    }

    /// Get memory statistics for the buffer.
    ///
    /// # Returns
    /// A tuple of (audio_bytes, mel_bytes) representing approximate memory usage.
    pub fn memory_stats(&self) -> (usize, usize) {
        // No audio buffer in StreamingMelBuffer (audio is processed immediately)
        let audio_bytes = 0;

        // Mel buffer: [n_mels, time] of f32
        let mel_bytes = self
            .mel_buffer
            .as_ref()
            .map(|b| b.size() * std::mem::size_of::<f32>())
            .unwrap_or(0);

        (audio_bytes, mel_bytes)
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
    // JJK what to do with chunk 0?
    let sampling_frames = (subsampling_factor, subsampling_factor);

    // Chunk size in pre-subsampled frames
    // NeMo: chunk_size = [sampling_frames[0] + subsampling_factor * lookahead_steps,
    //                     sampling_frames[1] + subsampling_factor * lookahead_steps]
    let chunk_size = (
        sampling_frames.1 + subsampling_factor * lookahead_steps,
        sampling_frames.1 + subsampling_factor * lookahead_steps,
    );
    // JJK: for chunk aware streaming, make chunk size = shift size
    // let chunk_size = (
    //     subsampling_factor * (lookahead_steps + 1),
    //     subsampling_factor * (lookahead_steps + 1),
    // );


    // Shift size - how many pre-subsampled frames to advance each step
    let shift_size = (
        sampling_frames.1 + subsampling_factor * lookahead_steps.saturating_sub(cache_drop_size),
        sampling_frames.1 + subsampling_factor * lookahead_steps.saturating_sub(cache_drop_size),
    );
    // JJK hardcode them to equal for chunk aware streaming
    // let shift_size = chunk_size;

    // Valid output length - how many OUTPUT frames are valid per chunk
    let valid_out_len = if sampling_frames.1 > 0 {
        (shift_size.1 - sampling_frames.1) / subsampling_factor + 1
    } else {
        shift_size.1 / subsampling_factor
    };

    // Pre-encode cache size (mel frames to prepend for streaming)
    // NeMo's dw_striding subsampling.get_streaming_cache_size() returns [0, subsampling_factor + 1]
    // For 8x subsampling: [0, 9]
    // For 4x subsampling: [0, 5]
    // let pre_encode_cache = (subsampling_factor + 1, subsampling_factor + 1);
    // let pre_encode_cache = (subsampling_factor + 1, subsampling_factor + 1);
    let pre_encode_cache = (subsampling_factor + 4, subsampling_factor + 4);

    // Drop extra pre-encoded frames after subsampling
    // NeMo: drop_extra_pre_encoded = 1 + (pre_encode_cache_size[1] - 1) // subsampling_factor
    // For 9 frames with 8x subsampling: 1 + (9-1)//8 = 1 + 1 = 2
    // let drop_extra = 1;
    let drop_extra = 1;
    // if pre_encode_cache.1 >= 1 {
    //     // 1 + (pre_encode_cache.1 - 1) / subsampling_factor
    //     // JJK test - just drop one
    //     // (pre_encode_cache.1 + 1) / subsampling_factor
    //     1
    // } else {
    //     0
    // };

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
    fn test_streaming_mel_buffer_trim() {
        let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 80, 512);

        // Add mel frames as Array [n_mels=80, time=100]
        let dummy_mel = Array::from_slice(&vec![1.0f32; 80 * 100], &[80, 100]);
        buffer.append_mel(&dummy_mel).unwrap();
        assert_eq!(buffer.total_frames, 100);

        // Advance buffer_idx (simulating chunk extraction)
        buffer.buffer_idx = 50;

        // Trim, keeping 10 frames before buffer_idx
        buffer.trim_old_frames(10).unwrap();

        // Should have trimmed 40 frames (50 - 10)
        assert_eq!(buffer.mel_buffer_offset, 40);

        // Mel buffer should now have 60 frames worth of data (100 - 40)
        assert_eq!(buffer.buffer_frames(), 60);

        // total_frames should still reflect logical count
        assert_eq!(buffer.total_frames, 100);
    }

    #[test]
    fn test_streaming_mel_buffer_append_and_extract() {
        let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 80, 512);

        // Create mel data [n_mels=80, time=50] with incrementing values per frame
        let mut mel_data = vec![0.0f32; 80 * 50];
        for t in 0..50 {
            for m in 0..80 {
                mel_data[m * 50 + t] = t as f32; // Each time frame has its frame index as value
            }
        }
        let mel = Array::from_slice(&mel_data, &[80, 50]);
        buffer.append_mel(&mel).unwrap();

        assert_eq!(buffer.total_frames, 50);
        assert_eq!(buffer.buffer_frames(), 50);

        // Extract a chunk: 20 frames, shift by 20, no pre_encode_cache
        // Extracts from buffer_idx (0) to buffer_idx + chunk_size (20)
        let (chunk, total, valid) = buffer.extract_chunk(20, 20, 0).unwrap();
        assert_eq!(total, 20); // Extracts chunk_size frames
        assert_eq!(valid, 20); // All frames are valid
        assert_eq!(chunk.shape(), &[80, 20]);

        // Verify buffer_idx advanced by shift_size
        assert_eq!(buffer.buffer_idx, 20);
        assert_eq!(buffer.remaining_frames(), 30);

        // Extract another chunk with pre_encode_cache=5
        // Extracts from buffer_idx (20) to buffer_idx + chunk_size + pre_encode_cache (45)
        let (chunk2, total2, valid2) = buffer.extract_chunk(20, 20, 5).unwrap();
        // Should get 20 + 5 = 25 frames (frames 20-44)
        assert_eq!(total2, 25);
        assert_eq!(valid2, 25);
        assert_eq!(chunk2.shape(), &[80, 25]);

        // Verify extracted values start at frame 20
        chunk2.eval().unwrap();
        let chunk2_data: &[f32] = chunk2.as_slice();
        assert_eq!(chunk2_data[0] as i32, 20); // First frame is 20
        assert_eq!(chunk2_data[24] as i32, 44); // Last frame is 44
    }

    #[test]
    fn test_streaming_mel_buffer_reset() {
        let mut buffer = StreamingMelBuffer::new(16000, 0.025, 0.01, 80, 512);

        let mel = Array::from_slice(&vec![1.0f32; 80 * 30], &[80, 30]);
        buffer.append_mel(&mel).unwrap();
        buffer.buffer_idx = 10;
        buffer.total_audio_samples = 5000;

        buffer.reset();

        assert!(buffer.mel_buffer.is_none());
        assert_eq!(buffer.total_frames, 0);
        assert_eq!(buffer.buffer_idx, 0);
        assert_eq!(buffer.total_audio_samples, 0);
        assert_eq!(buffer.mel_buffer_offset, 0);
    }
}
