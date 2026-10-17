//! RNN-T prediction network: a stateless or LSTM decoder behind one interface.

use mlx_rs::Array;

use super::{LstmState, RnntDecoder, StatelessTransducerDecoder};
use crate::core::error::{Error, Result};
use crate::core::loading::Weights;
use crate::core::quantization::QuantizationConfig;

/// Prediction network type (stateless or LSTM-based).
#[derive(Debug)]
pub enum PredictionNetwork {
    /// Stateless decoder (embedding-based)
    Stateless(StatelessTransducerDecoder),
    /// LSTM-based decoder
    Lstm(RnntDecoder),
}

/// Decoder state type.
#[derive(Debug, Clone)]
pub enum DecoderState {
    /// Stateless decoder state (context buffer)
    Stateless(Array),
    /// LSTM decoder state (hidden + cell)
    Lstm(LstmState),
}

impl PredictionNetwork {
    /// Get the prediction network hidden dimension.
    pub fn pred_hidden(&self) -> usize {
        match self {
            Self::Stateless(d) => d.pred_hidden(),
            Self::Lstm(d) => d.pred_hidden(),
        }
    }

    /// Get the blank token index.
    pub fn blank_id(&self) -> usize {
        match self {
            Self::Stateless(d) => d.blank_id(),
            Self::Lstm(d) => d.blank_id(),
        }
    }

    /// Get the vocabulary size (excluding blank).
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Stateless(d) => d.vocab_size(),
            Self::Lstm(d) => d.vocab_size(),
        }
    }

    /// Initialize decoder state for a batch.
    pub fn initialize_state(&self, batch_size: usize) -> Result<DecoderState> {
        match self {
            Self::Stateless(d) => Ok(DecoderState::Stateless(d.initialize_state(batch_size)?)),
            Self::Lstm(d) => Ok(DecoderState::Lstm(d.initialize_state(batch_size)?)),
        }
    }

    /// Initialize decoder state with warmup, matching NeMo's add_sos=True behavior.
    ///
    /// NeMo's decoder.predict(y=None, state=None, add_sos=True) runs the LSTM twice
    /// with blank inputs to "warm up" the state before decoding. This method
    /// replicates that behavior by:
    /// 1. Initialize state with zeros
    /// 2. Run predict_one(None) to get first output and updated state
    /// 3. Run predict_one(None) again to get the warmed up state
    ///
    /// Returns the prediction output and state after the warmup.
    pub fn initialize_state_with_warmup(&self, batch_size: usize) -> Result<(Array, DecoderState)> {
        // Initialize with zeros
        let state = self.initialize_state(batch_size)?;

        // First pass: blank input with zeros state
        let (_, state1) = self.predict_one(None, &state)?;

        // Second pass: blank input with updated state (matches NeMo's add_sos=True output[-1])
        let (output, state2) = self.predict_one(None, &state1)?;

        Ok((output, state2))
    }

    /// Run a single-step prediction for decoding.
    ///
    /// # Arguments
    /// * `y` - Single token index [batch, 1] or None for blank/SOS
    /// * `state` - Previous decoder state
    ///
    /// # Returns
    /// * `(output, new_state)` where output is [batch, 1, pred_hidden]
    pub fn predict_one(
        &self,
        y: Option<&Array>,
        state: &DecoderState,
    ) -> Result<(Array, DecoderState)> {
        match (self, state) {
            (Self::Stateless(decoder), DecoderState::Stateless(s)) => {
                let (output, new_state) = decoder.predict_one(y, s)?;
                Ok((output, DecoderState::Stateless(new_state)))
            }
            (Self::Lstm(decoder), DecoderState::Lstm(s)) => {
                let (output, new_state) = decoder.predict_one(y, s)?;
                Ok((output, DecoderState::Lstm(new_state)))
            }
            _ => Err(Error::model("Decoder and state type mismatch".to_string())),
        }
    }

    /// Load weights from a checkpoint.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `prefix` - Key prefix for this layer
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        match self {
            Self::Stateless(d) => d.load_weights(weights, prefix, quant),
            Self::Lstm(d) => d.load_weights(weights, prefix, quant),
        }
    }

    /// Check if this is an LSTM-based decoder.
    pub fn is_lstm(&self) -> bool {
        matches!(self, Self::Lstm(_))
    }
}
