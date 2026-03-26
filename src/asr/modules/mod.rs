//! ASR neural network modules.
//!
//! This module contains the building blocks for ASR models:
//! - Preprocessors (audio to spectrogram)
//! - Encoders (Conformer, FastConformer, etc.)
//! - Decoders (CTC projection, RNN-T prediction/joint networks)

pub mod decoders;
pub mod encoders;
pub mod preprocessors;

pub use decoders::CTCDecoder;
pub use decoders::{LstmState, RNNTDecoder, RNNTJoint, StatelessTransducerDecoder};
pub use encoders::ConformerEncoder;

// Primary exports (new names)
pub use preprocessors::{MelConfig, MelProcessor};

// Streaming exports
pub use preprocessors::{StreamingMelConfig, StreamingMelProcessor};

// Backwards-compatible aliases
pub use preprocessors::{MelSpectrogram, MelSpectrogramConfig};
