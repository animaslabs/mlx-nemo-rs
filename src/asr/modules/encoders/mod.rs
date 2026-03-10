//! ASR encoder modules.
//!
//! Encoders transform preprocessed audio features into higher-level
//! acoustic representations.

mod conformer;

pub use conformer::ConformerEncoder;
