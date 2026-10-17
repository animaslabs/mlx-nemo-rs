//! Core abstractions and utilities for mlx-voice.
//!
//! This module provides the foundational types and traits used across
//! all collections (ASR, TTS, LLM, etc.).

pub mod array;
pub mod checkpoint;
pub mod config;
pub mod error;
pub mod source;
internal_mod!(
    /// Loading checkpoint tensors into model layers.
    loading
);
internal_mod!(
    /// PyTorch checkpoint (`.ckpt`) parsing.
    pytorch
);
internal_mod!(
    /// Standard and quantized linear and embedding layers.
    quantization
);
internal_mod!(
    /// Weight layouts and safetensors loading.
    weights
);

pub use array::ArrayExt;
pub use error::{Error, Result};
pub use loading::{LoadOptions, LoadReport};
pub use quantization::QuantizationConfig;
pub use source::{HubRepo, ModelSource};
pub use weights::WeightFormat;

#[cfg(feature = "internals")]
pub use loading::Weights;
#[cfg(feature = "internals")]
pub use quantization::{EmbeddingLayer, LinearLayer};
