//! ASR neural network modules.
//!
//! This module contains the building blocks for ASR models:
//! - Preprocessors (audio to spectrogram)
//! - Encoders (Conformer, FastConformer, etc.)
//! - Decoders (CTC projection, RNN-T prediction/joint networks)

pub mod decoders;
pub mod encoders;
pub mod language_prompt;
pub mod preprocessors;
pub mod sortformer;
pub mod speaker_kernel;

pub use decoders::CTCDecoder;
pub use decoders::{LstmState, RNNTDecoder, RNNTJoint, StatelessTransducerDecoder};
pub use encoders::ConformerEncoder;
pub use encoders::TransformerEncoder;
pub use language_prompt::{LanguagePrompt, AUTO_LANGUAGE};
pub use sortformer::SortformerModules;
pub use speaker_kernel::{SpeakerKernels, SpeakerTargets};

// Primary exports (new names)
pub use preprocessors::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};

// Stateless streaming exports
pub use preprocessors::{
    frames_available, process_audio_streaming, MelStreamState, StreamingMelConfig,
};

// Backwards-compatible aliases
pub use preprocessors::{MelSpectrogram, MelSpectrogramConfig};
