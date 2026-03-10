//! ASR encoder modules.
//!
//! Encoders transform preprocessed audio features into higher-level
//! acoustic representations.

mod attention;
mod conformer;

pub use conformer::ConformerEncoder;
