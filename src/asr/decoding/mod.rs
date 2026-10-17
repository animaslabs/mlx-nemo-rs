//! Decoding strategies for ASR models.
//!
//! This module provides different decoding strategies for converting
//! model output probabilities to text transcriptions.
//!
//! ## CTC Decoding
//! - `greedy_decode`: Greedy CTC decoding with blank collapsing
//! - `ctc_frame_labels`, `ctc_collapse`: Streaming greedy CTC decoding
//!
//! ## RNN-T Decoding
//! - `rnnt_greedy_decode`: Greedy transducer decoding (batch)
//! - `rnnt_beam_decode`: Beam search transducer decoding
//! - `RnntStreamingDecoder`: Streaming RNN-T decoder
//!
//! ## TDT Decoding
//! - `TdtGreedyDecoder`: Greedy token-and-duration transducer decoding

mod detokenize;
mod greedy;
mod rnnt_beam;
mod rnnt_greedy;
mod rnnt_streaming;
mod tdt_greedy;

pub use detokenize::{
    IncrementalDetokenizer, decode_text, is_lang_tag, is_punctuation_char, lang_tag_table,
    last_lang_tag, punctuation_table, strip_lang_tags, word_start_table,
};
pub use greedy::{ctc_collapse, ctc_frame_labels, greedy_decode};
pub use rnnt_beam::rnnt_beam_decode;
pub use rnnt_greedy::{rnnt_greedy_decode, rnnt_greedy_decode_batched};
pub use rnnt_streaming::{DecodeStepOutput, RnntStreamingDecoder};
pub use tdt_greedy::{TdtGreedyDecoder, refine_punctuation_timestamps};

/// A decoded token and the encoder frames `[start_frame, end_frame)` it was
/// emitted on.
///
/// TDT tokens span the frames their predicted duration advances, which can
/// be none (`start_frame == end_frame`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimedToken {
    /// Token id
    pub id: usize,
    /// First encoder frame
    pub start_frame: usize,
    /// One past the last encoder frame
    pub end_frame: usize,
}
