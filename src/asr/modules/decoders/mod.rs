//! ASR decoder modules.
//!
//! Decoders project encoded features to output vocabulary logits.
//!
//! - `CTCDecoder`: Simple linear projection for CTC models
//! - `StatelessTransducerDecoder`: Stateless prediction network for RNN-T models
//! - `RNNTDecoder`: LSTM-based prediction network for RNN-T models
//! - `RNNTJoint`: Joint network for RNN-T models

mod ctc;
mod rnnt;

pub use ctc::CTCDecoder;
pub use rnnt::{JointActivation, LstmState, RNNTDecoder, RNNTJoint, StatelessTransducerDecoder};
