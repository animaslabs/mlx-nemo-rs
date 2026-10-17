//! Automatic Speech Recognition (ASR) collection.
//!
//! This module provides ASR models and utilities compatible with NVIDIA NeMo's
//! ASR collection, including:
//!
//! - CTC-based models (`EncDecCtcModel`)
//! - RNN-T/Transducer models (`EncDecRnntModel`)
//! - Conformer/FastConformer encoders
//! - Audio preprocessing (Mel spectrogram)
//! - Greedy and beam search decoding
//! - Cache-aware streaming (`StreamingSession`) and multi-talker transcription
//!
//! # Example (CTC Model)
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! use mlx_voice::asr::EncDecCtcModel;
//!
//! let mut model = EncDecCtcModel::from_pretrained("parakeet-ctc-0.6b.nemo")?;
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! # Ok(())
//! # }
//! ```
//!
//! # Example (RNN-T streaming)
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! use mlx_voice::asr::{EncDecRnntModel, StreamingOptions};
//!
//! let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
//! let transcript = model.transcribe_streaming("audio.wav", StreamingOptions::default())?;
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
    EncDecClassificationModel, EncDecCtcModel, EncDecRnntModel, SortformerEncLabelModel,
};
pub use modules::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};
pub use pipeline::{
    MultiTalkerConfig, MultiTalkerMode, MultiTalkerOutput, MultiTalkerPipeline, SpeakerSegment,
    SpeakerWord, StreamUpdate, StreamingOptions, StreamingSession,
};
pub use streaming::{StreamingConfig, StreamingModeInfo};
pub use tokenizer::Tokenizer;
pub use transcript::{
    AudioInput, DecoderType, DecodingStrategy, Token, TranscribeOptions, Transcript, Word,
};
pub use vad::{SpeechSegment, VadPostProcessConfig, VadPostProcessor};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::config::AsrModelConfig;
    pub use super::models::{EncDecCtcModel, EncDecRnntModel};
    pub use super::pipeline::{StreamingOptions, StreamingSession};
    pub use super::tokenizer::Tokenizer;
    pub use super::transcript::{
        AudioInput, DecoderType, DecodingStrategy, TranscribeOptions, Transcript,
    };
}

// Models and sessions are moved to worker threads (audio callbacks feed a
// queue that another thread drains into a session), so keep them `Send`.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<EncDecCtcModel>();
    assert_send::<EncDecRnntModel>();
    assert_send::<EncDecClassificationModel>();
    assert_send::<SortformerEncLabelModel>();
    assert_send::<StreamingSession<EncDecRnntModel>>();
    assert_send::<StreamingSession<&mut EncDecRnntModel>>();
    assert_send::<MultiTalkerPipeline>();
    assert_send::<StreamUpdate>();
    assert_send::<MultiTalkerOutput>();
    assert_send::<Transcript>();
    assert_send::<Tokenizer>();
    assert_send::<crate::audio::StreamResampler>();
};
