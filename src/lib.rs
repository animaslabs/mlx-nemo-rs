//! # mlx-nemo
//!
//! A Rust implementation of NVIDIA NeMo for Apple Silicon, built on MLX.
//!
//! This library provides speech AI capabilities including:
//! - Automatic Speech Recognition (ASR)
//! - Text-to-Speech (TTS) - coming soon
//! - Large Language Models (LLM) - coming soon
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mlx_nemo::asr::EncDecCTCModel;
//!
//! // Load a pretrained model from a .nemo file
//! let model = EncDecCTCModel::restore("path/to/model.nemo")?;
//!
//! // Transcribe audio
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! println!("{}", transcriptions[0]);
//! ```

pub mod core;

#[cfg(feature = "asr")]
pub mod asr;

pub mod audio;
pub mod math;
pub mod utils;

// Re-export commonly used types
pub use core::error::{Error, Result};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::core::error::{Error, Result};

    #[cfg(feature = "asr")]
    pub use crate::asr::prelude::*;
}
