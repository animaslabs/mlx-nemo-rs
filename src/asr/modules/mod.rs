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

pub use decoders::CtcDecoder;
pub use decoders::{DecoderState, PredictionNetwork};
pub use decoders::{LstmState, RnntDecoder, RnntJoint, StatelessTransducerDecoder};
pub use encoders::ConformerEncoder;
pub use encoders::TransformerEncoder;
pub use language_prompt::{AUTO_LANGUAGE, LanguagePrompt};
pub use sortformer::SortformerModules;
pub use speaker_kernel::{SpeakerKernels, SpeakerTargets};

pub use preprocessors::{MelConfig, MelProcessor, SpectrumTransform, StreamingMel};
pub(crate) use preprocessors::StreamingMelConfig;
