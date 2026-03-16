//! Streaming ASR support.
//!
//! This module provides types and utilities for cache-aware streaming
//! ASR inference, compatible with NeMo's streaming architecture.
//!
//! # Submodules
//!
//! - `core`: Core streaming types (StreamingConfig, EncoderCache, etc.)
//! - `request`: Request types (Frame, FeatureBuffer)
//! - `streamer`: Request streamer for audio/mel processing

mod core;
mod request;
mod streamer;

// Re-export core types
pub use self::core::{
    compute_streaming_config, get_streaming_modes, EncoderCache, MelBuffer, StreamingConfig,
    StreamingMelBuffer, StreamingModeInfo, StreamingOutput,
};

// Re-export request types
pub use request::{FeatureBuffer, Frame, Request};

// Re-export streamer
pub use streamer::RequestStreamer;
