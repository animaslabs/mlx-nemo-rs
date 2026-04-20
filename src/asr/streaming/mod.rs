//! Streaming ASR support.
//!
//! This module provides types and utilities for cache-aware streaming
//! ASR inference, compatible with NeMo's streaming architecture.

mod core;
mod request;
mod sortformer_state;

// Re-export core types
pub use self::core::{
    compute_streaming_config, get_streaming_modes, EncoderCache, StreamingConfig,
    StreamingMelBuffer, StreamingModeInfo, StreamingOutput,
};

// Re-export request types
pub use request::FeatureBuffer;

// Re-export sortformer streaming state
pub use sortformer_state::StreamingSortformerState;
