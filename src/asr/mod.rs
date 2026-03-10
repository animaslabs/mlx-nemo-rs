//! Automatic Speech Recognition (ASR) collection.
//!
//! This module provides ASR models and utilities compatible with NVIDIA NeMo's
//! ASR collection, including:
//!
//! - CTC-based models (EncDecCTCModel)
//! - Conformer/FastConformer encoders
//! - Audio preprocessing (Mel spectrogram)
//! - Greedy and beam search decoding
//!
//! # Example
//!
//! ```rust,ignore
//! use mlx_nemo::asr::EncDecCTCModel;
//!
//! let model = EncDecCTCModel::restore("path/to/model.nemo")?;
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! ```

pub mod config;
pub mod decoding;
pub mod models;
pub mod modules;

// Re-exports
pub use config::*;
pub use models::{DebugOutput, EncDecCTCModel};
pub use modules::MelSpectrogram;

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::EncDecCTCModel;
}
