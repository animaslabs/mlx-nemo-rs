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
//! - Streaming pipelines (`CacheAwareRNNTPipeline`)
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
//! # Example (RNN-T Streaming Pipeline)
//!
//! ```rust,ignore
//! use mlx_nemo::asr::CacheAwareRNNTPipeline;
//!
//! let mut pipeline = CacheAwareRNNTPipeline::restore("model.nemo", 0)?;
//! pipeline.set_audio_file("audio.wav")?;
//! let text = pipeline.run()?;
//! ```

pub mod config;
pub mod decoding;
pub mod models;
pub mod modules;
pub mod pipeline;
pub mod streaming;

// Re-exports
pub use config::*;
pub use models::{
    CtcStreamingSession, DebugOutput, EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy,
    RnntStreamingSession,
};
// Primary exports (new names)
pub use modules::{MelConfig, MelProcessor};
// Backwards-compatible alias
pub use modules::MelSpectrogram;
pub use pipeline::{CacheAwareRNNTPipeline, TranscribeOutput};
pub use streaming::{
    EncoderCache, FeatureBuffer, StreamingConfig, StreamingMelBuffer, StreamingModeInfo,
    StreamingOutput,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::{EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy};
    pub use super::pipeline::CacheAwareRNNTPipeline;
}
