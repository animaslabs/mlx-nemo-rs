//! Decoding strategies for ASR models.
//!
//! This module provides different decoding strategies for converting
//! model output probabilities to text transcriptions.

mod greedy;

pub use greedy::greedy_decode;
