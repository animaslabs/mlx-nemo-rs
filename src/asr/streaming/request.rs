//! Feature buffer type for streaming ASR inference.

use mlx_rs::Array;

/// Pre-computed feature buffer containing mel spectrogram frames.
///
/// This mirrors NeMo's `FeatureBuffer` dataclass for streaming mel input.
#[derive(Debug, Clone)]
pub struct FeatureBuffer {
    /// Mel spectrogram features [n_mels, time]
    pub features: Array,
    /// Whether this is the first buffer in the stream
    pub is_first: bool,
    /// Whether this is the last buffer in the stream
    pub is_last: bool,
    /// Valid number of frames (excluding right padding)
    pub valid_size: usize,
    /// Number of pre-encode cache frames (left context)
    pub pre_encode_cache: usize,
}

impl FeatureBuffer {
    /// Create a new feature buffer.
    pub fn new(features: Array, valid_size: usize) -> Self {
        Self {
            features,
            is_first: false,
            is_last: false,
            valid_size,
            pre_encode_cache: 0,
        }
    }

    /// Mark this buffer as the first in the stream.
    pub fn with_first(mut self, is_first: bool) -> Self {
        self.is_first = is_first;
        self
    }

    /// Mark this buffer as the last in the stream.
    pub fn with_last(mut self, is_last: bool) -> Self {
        self.is_last = is_last;
        self
    }

    /// Set the pre-encode cache size.
    pub fn with_pre_encode_cache(mut self, cache_size: usize) -> Self {
        self.pre_encode_cache = cache_size;
        self
    }

    /// Get the total number of frames (including padding and cache).
    pub fn size(&self) -> usize {
        self.features.dim(1) as usize
    }

    /// Get the number of new frames (excluding pre-encode cache).
    pub fn new_frames(&self) -> usize {
        self.size().saturating_sub(self.pre_encode_cache)
    }
}
