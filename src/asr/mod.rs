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
//! # fn main() -> mlx_voice::Result<()> {
//! use mlx_voice::asr::EncDecCTCModel;
//!
//! let mut model = EncDecCTCModel::from_pretrained("parakeet-ctc-0.6b.nemo")?;
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example (RNN-T Streaming Pipeline)
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! use mlx_voice::asr::CacheAwareRNNTPipeline;
//!
//! let mut pipeline = CacheAwareRNNTPipeline::from_pretrained("model.nemo", 0)?;
//! pipeline.set_audio_file("audio.wav")?;
//! let text = pipeline.run()?;
//! # Ok(())
//! # }
//! ```

pub mod config;
internal_mod!(
    /// Greedy and beam search decoding, detokenization and token merging.
    decoding
);
internal_mod!(
    /// Signal processing helpers (windows, mel filterbanks, framing).
    dsp
);
pub mod models;
internal_mod!(
    /// Neural network building blocks: preprocessors, encoders, decoders.
    modules
);
pub mod pipeline;
internal_mod!(
    /// Cache-aware streaming state and configuration.
    streaming
);
pub mod tokenizer;
pub mod transcript;
pub mod vad;

// Re-exports
pub use config::AsrModelConfig;
pub use models::{
    CtcStreamingSession, EncDecCTCModel, EncDecClassificationModel, EncDecRNNTModel,
    RnntStreamingSession, SortformerEncLabelModel,
};
pub use modules::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};
pub use pipeline::{
    CacheAwareRNNTPipeline, MultiTalkerConfig, MultiTalkerMode, MultiTalkerOutput,
    MultiTalkerPipeline, SpeakerSegment, SpeakerWord, TranscribeOutput,
};
pub use streaming::{StreamingConfig, StreamingModeInfo, StreamingOutput};
pub use tokenizer::Tokenizer;
pub use transcript::{AudioInput, DecodingStrategy, Token, TranscribeOptions, Transcript, Word};
pub use vad::{SpeechSegment, VadPostProcessConfig, VadPostProcessor};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::{EncDecCTCModel, EncDecRNNTModel};
    pub use super::pipeline::CacheAwareRNNTPipeline;
    pub use super::tokenizer::Tokenizer;
    pub use super::transcript::{AudioInput, DecodingStrategy, TranscribeOptions, Transcript};
}
