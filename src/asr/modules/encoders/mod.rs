//! ASR encoder modules.
//!
//! Encoders transform preprocessed audio features into higher-level
//! acoustic representations.

pub mod attention;
mod conformer;

pub use conformer::reset_encoder_counter;
pub use conformer::ConformerEncoder;
pub use conformer::DebugEncoderOutput;
