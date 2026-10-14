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
//! ```no_run
//! # fn main() -> mlx_nemo::Result<()> {
//! use mlx_nemo::asr::EncDecCTCModel;
//!
//! let mut model = EncDecCTCModel::restore("parakeet-ctc-0.6b.nemo")?;
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example (RNN-T Streaming Pipeline)
//!
//! ```no_run
//! # fn main() -> mlx_nemo::Result<()> {
//! use mlx_nemo::asr::CacheAwareRNNTPipeline;
//!
//! let mut pipeline = CacheAwareRNNTPipeline::restore("model.nemo", 0)?;
//! pipeline.set_audio_file("audio.wav")?;
//! let text = pipeline.run()?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod decoding;
pub mod dsp;
pub mod models;
pub mod modules;
pub mod pipeline;
pub mod streaming;
pub mod vad;

// Re-exports
pub use config::*;
pub use models::{
    CtcStreamingSession, DebugOutput, EncDecCTCModel, EncDecClassificationModel, EncDecRNNTModel,
    RnntDecodingStrategy, RnntStreamingSession, SortformerEncLabelModel,
};
pub use vad::{SpeechSegment, VadPostProcessConfig, VadPostProcessor};
// Primary exports (new names)
pub use modules::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};
// Backwards-compatible alias
pub use modules::MelSpectrogram;
pub use pipeline::{CacheAwareRNNTPipeline, MultiTalkerOutput, MultiTalkerPipeline, SpeakerSegment, TranscribeOutput};
pub use streaming::{
    EncoderCache, FeatureBuffer, StreamingConfig, StreamingMelBuffer, StreamingModeInfo,
    StreamingOutput, StreamingSortformerState,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::{EncDecCTCModel, EncDecRNNTModel, RnntDecodingStrategy};
    pub use super::pipeline::CacheAwareRNNTPipeline;
}
