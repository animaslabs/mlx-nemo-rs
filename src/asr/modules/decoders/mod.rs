//! ASR decoder modules.
//!
//! Decoders project encoded features to output vocabulary logits.

mod ctc;

pub use ctc::CTCDecoder;
