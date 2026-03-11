//! Automatic Speech Recognition (ASR) collection.
//!
//! This module provides ASR models and utilities compatible with NVIDIA NeMo's
//! ASR collection, including:
//!
//! - CTC-based models (`EncDecCTCModel`)
//! - RNN-T/Transducer models (`EncDecRNNTModel`)
//! - Conformer/FastConformer encoders
//! - Audio preprocessing (Mel spectrogram)
//! - Greedy and beam search decoding
//!
//! # Example (CTC Model)
//!
//! ```rust,ignore
//! use mlx_nemo::asr::EncDecCTCModel;
//!
//! let model = EncDecCTCModel::restore("parakeet-ctc-0.6b.nemo")?;
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! ```
//!
//! # Example (RNN-T Model)
//!
//! ```rust,ignore
//! use mlx_nemo::asr::EncDecRNNTModel;
//!
//! let model = EncDecRNNTModel::restore("parakeet-rnnt-0.6b.nemo")?;
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! ```

pub mod config;
pub mod decoding;
pub mod models;
pub mod modules;
pub mod streaming;

// Re-exports
pub use config::*;
pub use models::{
    CtcStreamingSession, DebugOutput, EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy,
    RnntStreamingSession,
};
pub use modules::MelSpectrogram;
pub use streaming::{
    EncoderCache, MelBuffer, StreamingConfig, StreamingMelBuffer, StreamingModeInfo,
    StreamingOutput,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::{EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy};
}
