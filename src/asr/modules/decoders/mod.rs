//! ASR decoder modules.
//!
//! Decoders project encoded features to output vocabulary logits.
//!
//! - `CtcDecoder`: Simple linear projection for CTC models
//! - `StatelessTransducerDecoder`: Stateless prediction network for RNN-T models
//! - `RnntDecoder`: LSTM-based prediction network for RNN-T models
//! - `RnntJoint`: Joint network for RNN-T models

mod ctc;
mod prediction;
mod rnnt;

pub use ctc::CtcDecoder;
pub use prediction::{DecoderState, PredictionNetwork};
pub use rnnt::{JointActivation, LstmState, RnntDecoder, RnntJoint, StatelessTransducerDecoder};
