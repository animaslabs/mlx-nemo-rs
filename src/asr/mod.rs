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
pub mod mel_stream;
pub mod models;
pub mod modules;
pub mod pipeline;
pub mod streaming;

// Re-exports
pub use config::*;
pub use mel_stream::{MelStream, MelStreamConfig, MelStreamIter};
pub use models::{
    CtcStreamingSession, DebugOutput, EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy,
    RnntStreamingSession,
};
pub use modules::MelSpectrogram;
pub use pipeline::{CacheAwareRNNTPipeline, TranscribeOutput};
pub use streaming::{
    EncoderCache, FeatureBuffer, Frame, MelBuffer, Request, RequestStreamer, StreamingConfig,
    StreamingMelBuffer, StreamingModeInfo, StreamingOutput,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::{EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy};
    pub use super::pipeline::CacheAwareRNNTPipeline;
}
