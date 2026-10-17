//! Core abstractions and utilities for mlx-voice.
//!
//! This module provides the foundational types and traits used across
//! all collections (ASR, TTS, LLM, etc.).

pub mod array;
pub mod checkpoint;
pub mod config;
pub mod error;
pub mod loading;
pub mod pytorch;
pub mod quantization;
pub mod weights;

pub use array::ArrayExt;
pub use error::{Error, Result};
pub use loading::{LoadOptions, LoadReport, Weights};
pub use quantization::{EmbeddingLayer, LinearLayer, QuantizationConfig};
pub use weights::WeightFormat;
