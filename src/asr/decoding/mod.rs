//! Decoding strategies for ASR models.
//!
//! This module provides different decoding strategies for converting
//! model output probabilities to text transcriptions.
//!
//! ## CTC Decoding
//! - `greedy_decode`: Greedy CTC decoding with blank collapsing
//!
//! ## RNN-T Decoding
//! - `rnnt_greedy_decode`: Greedy transducer decoding
//! - `rnnt_beam_decode`: Beam search transducer decoding
//!
//! ## Streaming Merging
//! - `lcs_merge_buffer`: LCS-based token merging for streaming
//! - `greedy_ctc_merge`: Simple CTC blank collapsing

mod greedy;
mod merge;
mod rnnt_beam;
mod rnnt_greedy;

pub use greedy::greedy_decode;
pub use merge::{greedy_ctc_merge, lcs_merge_buffer, LcsAlignmentResult};
pub use rnnt_beam::rnnt_beam_decode;
pub use rnnt_greedy::{rnnt_greedy_decode, rnnt_greedy_decode_batched};
