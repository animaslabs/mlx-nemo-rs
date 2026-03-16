//! Request types for streaming ASR inference.
//!
//! This module provides request abstractions that mirror NeMo's
//! `Frame` and `FeatureBuffer` types for streaming pipelines.

use mlx_rs::Array;

/// Audio frame request containing raw audio samples.
///
/// This mirrors NeMo's `Frame` dataclass for streaming audio input.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Raw audio samples (mono, f32)
    pub samples: Vec<f32>,
    /// Whether this is the first frame in the stream
    pub is_first: bool,
    /// Whether this is the last frame in the stream
    pub is_last: bool,
    /// Valid length of samples (excluding padding). If None, all samples are valid.
    pub length: Option<usize>,
}

impl Frame {
    /// Create a new frame with samples.
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            is_first: false,
            is_last: false,
            length: None,
        }
    }

    /// Create the first frame in a stream.
    pub fn first(samples: Vec<f32>) -> Self {
        Self {
            samples,
            is_first: true,
            is_last: false,
            length: None,
        }
    }

    /// Create the last frame in a stream.
    pub fn last(samples: Vec<f32>) -> Self {
        Self {
            samples,
            is_first: false,
            is_last: true,
            length: None,
        }
    }

    /// Mark this frame as the first in the stream.
    pub fn with_first(mut self, is_first: bool) -> Self {
        self.is_first = is_first;
        self
    }

    /// Mark this frame as the last in the stream.
    pub fn with_last(mut self, is_last: bool) -> Self {
        self.is_last = is_last;
        self
    }

    /// Set the valid length (for padded frames).
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    /// Get the total size of the frame (including padding).
    pub fn size(&self) -> usize {
        self.samples.len()
    }

    /// Get the valid size of the frame (excluding padding).
    pub fn valid_size(&self) -> usize {
        self.length.unwrap_or(self.samples.len())
    }
}

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

/// Request type for streaming inference.
///
/// A request can be either raw audio samples (Frame) or pre-computed
/// mel spectrogram features (FeatureBuffer).
#[derive(Debug, Clone)]
pub enum Request {
    /// Raw audio frame
    Frame(Frame),
    /// Pre-computed feature buffer
    FeatureBuffer(FeatureBuffer),
}

impl Request {
    /// Check if this is the first request in the stream.
    pub fn is_first(&self) -> bool {
        match self {
            Request::Frame(f) => f.is_first,
            Request::FeatureBuffer(fb) => fb.is_first,
        }
    }

    /// Check if this is the last request in the stream.
    pub fn is_last(&self) -> bool {
        match self {
            Request::Frame(f) => f.is_last,
            Request::FeatureBuffer(fb) => fb.is_last,
        }
    }
}

impl From<Frame> for Request {
    fn from(frame: Frame) -> Self {
        Request::Frame(frame)
    }
}

impl From<FeatureBuffer> for Request {
    fn from(buffer: FeatureBuffer) -> Self {
        Request::FeatureBuffer(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_creation() {
        let samples = vec![0.1, 0.2, 0.3];
        let frame = Frame::new(samples.clone());
        assert_eq!(frame.size(), 3);
        assert_eq!(frame.valid_size(), 3);
        assert!(!frame.is_first);
        assert!(!frame.is_last);
    }

    #[test]
    fn test_frame_first_last() {
        let first = Frame::first(vec![0.1]);
        assert!(first.is_first);
        assert!(!first.is_last);

        let last = Frame::last(vec![0.2]);
        assert!(!last.is_first);
        assert!(last.is_last);
    }

    #[test]
    fn test_frame_with_length() {
        let frame = Frame::new(vec![0.0; 100]).with_length(80);
        assert_eq!(frame.size(), 100);
        assert_eq!(frame.valid_size(), 80);
    }

    #[test]
    fn test_request_is_first_last() {
        let frame = Frame::first(vec![0.1]);
        let request: Request = frame.into();
        assert!(request.is_first());
        assert!(!request.is_last());
    }
}
