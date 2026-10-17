//! Core abstractions and utilities for mlx-voice.
//!
//! Errors, model sources and loading options. Checkpoint parsing and the
//! other building blocks are public with the `internals` feature.

internal_mod!(
    /// Array helpers.
    array
);
internal_mod!(
    /// `.nemo` archives and safetensors model directories.
    checkpoint
);
internal_mod!(
    /// YAML config helpers.
    config
);
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

pub use error::{Error, Result};
pub use loading::{LoadOptions, LoadReport};
pub use source::{HubRepo, ModelSource};

#[cfg(feature = "internals")]
pub use array::ArrayExt;
#[cfg(not(feature = "internals"))]
pub(crate) use array::ArrayExt;
#[cfg(feature = "internals")]
pub use loading::Weights;
#[cfg(feature = "internals")]
pub use quantization::{EmbeddingLayer, LinearLayer, QuantizationConfig};
#[cfg(feature = "internals")]
pub use weights::WeightFormat;
