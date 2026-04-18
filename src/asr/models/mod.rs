//! ASR model implementations.
//!
//! This module provides high-level ASR model implementations:
//! - `EncDecCTCModel`: CTC-based encoder-decoder model
//! - `EncDecRNNTModel`: RNN-T (Transducer) encoder-decoder model

mod enc_dec_classification;
mod enc_dec_ctc;
mod enc_dec_rnnt;

pub use enc_dec_classification::EncDecClassificationModel;
pub use enc_dec_ctc::{CtcStreamingSession, DebugOutput, EncDecCTCModel};
pub use enc_dec_rnnt::{
    DecoderState, EncDecRNNTModel, PredictionNetwork, RnntDecodingStrategy, RnntStreamingSession,
};
