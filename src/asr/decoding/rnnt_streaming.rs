//! Streaming RNN-T greedy decoding.
//!
//! This module provides a streaming-aware RNN-T decoder that mirrors
//! NeMo's `RNNTGreedyDecoder` for cache-aware streaming inference.
//!
//! Unlike the batch decoder in `rnnt_greedy.rs`, this decoder:
//! - Maintains state across multiple calls (decoder state, last token)
//! - Processes chunks of encoder output incrementally
//! - Tracks labels including blanks (for future endpointing support)

use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
use mlx_rs::Array;
use tracing::{debug, info, trace};

use crate::asr::models::{DecoderState, PredictionNetwork};
use crate::asr::modules::{LstmState, RNNTJoint};
use crate::core::error::Result;

/// Compute L2 norm of a 1D slice
fn l2_norm(slice: &[f32]) -> f32 {
    slice.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Log LSTM decoder state for debugging
fn log_lstm_state(state: &LstmState, label: &str, step: usize, frame: usize, symbols: usize) {
    if let (Ok(h_flat), Ok(c_flat)) = (
        state.hidden.reshape(&[-1]),
        state.cell.reshape(&[-1]),
    ) {
        h_flat.eval().ok();
        c_flat.eval().ok();
        let h_slice: &[f32] = h_flat.as_slice();
        let c_slice: &[f32] = c_flat.as_slice();
        let h_norm = l2_norm(h_slice);
        let c_norm = l2_norm(c_slice);
        let h_first5: Vec<f32> = h_slice.iter().take(5).copied().collect();
        let c_first5: Vec<f32> = c_slice.iter().take(5).copied().collect();

        info!(
            step,
            frame,
            symbols,
            label,
            h_norm = format!("{:.6}", h_norm),
            c_norm = format!("{:.6}", c_norm),
            h_first5 = ?h_first5.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>(),
            c_first5 = ?c_first5.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>(),
            "decoder state"
        );
    }
}

/// Maximum number of symbols that can be emitted per encoder frame.
/// This is a safety limit to prevent infinite loops.
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

/// Output from a streaming decode step.
#[derive(Debug, Clone, Default)]
pub struct DecodeStepOutput {
    /// New tokens emitted in this step
    pub new_tokens: Vec<usize>,
    /// Timestep (encoder frame index) for each new token
    pub timesteps: Vec<usize>,
    /// Labels including blanks for each encoder frame (for endpointing)
    pub labels: Vec<usize>,
    /// Total blank emissions in this step
    pub blank_count: usize,
    /// Total non-blank emissions in this step
    pub non_blank_count: usize,
}

impl DecodeStepOutput {
    /// Check if any tokens were emitted.
    pub fn has_tokens(&self) -> bool {
        !self.new_tokens.is_empty()
    }
}

/// Streaming RNN-T greedy decoder.
///
/// This decoder processes encoder output frame-by-frame and maintains
/// state for streaming inference. It mirrors NeMo's `RNNTGreedyDecoder`.
#[derive(Debug, Clone)]
pub struct RnntStreamingDecoder {
    /// Blank token ID (typically vocab_size)
    blank_id: usize,
    /// Maximum symbols per encoder frame
    max_symbols_per_step: usize,
}

impl RnntStreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new(blank_id: usize) -> Self {
        Self {
            blank_id,
            max_symbols_per_step: DEFAULT_MAX_SYMBOLS_PER_STEP,
        }
    }

    /// Set the maximum symbols per step.
    pub fn with_max_symbols(mut self, max_symbols: usize) -> Self {
        self.max_symbols_per_step = max_symbols;
        self
    }

    /// Get the blank token ID.
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Decode encoder frames and update state.
    ///
    /// This is the main entry point for streaming decoding. It processes
    /// `valid_frames` of encoder output and updates the decoder state.
    ///
    /// # Arguments
    /// * `encoder_output` - Encoder output [1, time, d_model]
    /// * `valid_frames` - Number of valid frames to decode
    /// * `decoder` - Prediction network
    /// * `decoder_state` - Mutable decoder state (updated in place)
    /// * `joint` - Joint network
    /// * `last_token` - Mutable last token (updated in place)
    ///
    /// # Returns
    /// * `DecodeStepOutput` containing new tokens and labels
    pub fn decode_frames(
        &self,
        encoder_output: &Array,
        valid_frames: usize,
        decoder: &PredictionNetwork,
        decoder_state: &mut DecoderState,
        joint: &RNNTJoint,
        last_token: &mut Option<usize>,
    ) -> Result<DecodeStepOutput> {
        encoder_output.eval()?;

        let mut output = DecodeStepOutput {
            labels: vec![self.blank_id; valid_frames],
            ..Default::default()
        };

        trace!(
            valid_frames,
            blank_id = self.blank_id,
            max_symbols = self.max_symbols_per_step,
            "decode_frames start"
        );

        // Track global step for detailed logging
        let mut global_step: usize = 0;

        for t in 0..valid_frames {
            // Get encoder frame: [1, 1, d_model]
            let enc_frame = encoder_output.index((0i32..1, t as i32..t as i32 + 1, ..));
            enc_frame.eval()?;

            // Log encoder frame values for comparison
            let enc_flat = enc_frame.reshape(&[-1])?;
            enc_flat.eval()?;
            let enc_slice: &[f32] = enc_flat.as_slice();
            let enc_norm = l2_norm(enc_slice);
            let enc_first10: Vec<f32> = enc_slice.iter().take(10).copied().collect();

            info!(
                frame = t,
                enc_norm = format!("{:.6}", enc_norm),
                enc_first10 = ?enc_first10.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>(),
                "decoder input encoder frame"
            );

            let mut symbols_emitted = 0;

            loop {
                // Build last token array if we have one
                let last_token_arr =
                    last_token.map(|tok| Array::from_slice(&[tok as i32], &[1, 1]));

                info!(
                    frame = t,
                    step = global_step,
                    symbols_emitted,
                    last_token = ?*last_token,
                    "prediction network input"
                );

                // Log decoder state before prediction
                if let DecoderState::Lstm(ref lstm_state) = decoder_state {
                    log_lstm_state(lstm_state, "before", global_step, t, symbols_emitted);
                }

                // Get prediction output
                let (pred_output, new_state) =
                    decoder.predict_one(last_token_arr.as_ref(), decoder_state)?;
                pred_output.eval()?;

                // Log prediction output
                let pred_flat = pred_output.reshape(&[-1])?;
                pred_flat.eval()?;
                let pred_slice: &[f32] = pred_flat.as_slice();
                let pred_norm = l2_norm(pred_slice);
                let pred_first10: Vec<f32> = pred_slice.iter().take(10).copied().collect();

                info!(
                    frame = t,
                    step = global_step,
                    pred_norm = format!("{:.6}", pred_norm),
                    pred_first10 = ?pred_first10.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>(),
                    "prediction network output"
                );

                // Compute joint output: [1, vocab_size+1]
                let log_probs = joint.forward_one(&enc_frame, &pred_output)?;
                log_probs.eval()?;

                // Get argmax prediction
                let prediction = argmax_axis(&log_probs, -1, false)?;
                prediction.eval()?;
                let pred_token = prediction.as_slice::<u32>()[0] as usize;

                // Log joint output and prediction
                let lp_slice: &[f32] = log_probs.as_slice();
                let blank_prob = lp_slice
                    .get(self.blank_id)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                let pred_prob = lp_slice
                    .get(pred_token)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);

                // Get top-5 predictions
                let mut indexed: Vec<(usize, f32)> = lp_slice.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();

                info!(
                    frame = t,
                    step = global_step,
                    pred_token,
                    pred_prob = format!("{:.6}", pred_prob),
                    blank_prob = format!("{:.6}", blank_prob),
                    is_blank = pred_token == self.blank_id,
                    top5 = ?top5.iter().map(|(i, p)| format!("{}:{:.4}", i, p)).collect::<Vec<_>>(),
                    "joint output"
                );

                if pred_token == self.blank_id {
                    // Blank: advance to next encoder frame
                    output.blank_count += 1;
                    output.labels[t] = self.blank_id;
                    global_step += 1;
                    break;
                }

                // Non-blank: emit token and update state
                output.new_tokens.push(pred_token);
                output.timesteps.push(t);
                output.non_blank_count += 1;
                output.labels[t] = pred_token;

                *last_token = Some(pred_token);
                *decoder_state = new_state;
                symbols_emitted += 1;
                global_step += 1;

                // Log decoder state after token emission
                if let DecoderState::Lstm(ref lstm_state) = decoder_state {
                    log_lstm_state(lstm_state, "after", global_step, t, symbols_emitted);
                }

                info!(
                    frame = t,
                    step = global_step,
                    token = pred_token,
                    symbols_emitted,
                    "token emitted"
                );

                if symbols_emitted >= self.max_symbols_per_step {
                    debug!(frame = t, symbols_emitted, "max symbols per step reached");
                    break;
                }
            }
        }

        debug!(
            new_tokens = output.new_tokens.len(),
            blank_count = output.blank_count,
            non_blank_count = output.non_blank_count,
            "decode_frames complete"
        );

        Ok(output)
    }

    /// Decode a single encoder frame.
    ///
    /// Convenience method for decoding one frame at a time.
    pub fn decode_frame(
        &self,
        encoder_frame: &Array,
        decoder: &PredictionNetwork,
        decoder_state: &mut DecoderState,
        joint: &RNNTJoint,
        last_token: &mut Option<usize>,
    ) -> Result<DecodeStepOutput> {
        // Ensure frame has batch dimension: [1, 1, d_model]
        let frame = if encoder_frame.ndim() == 2 {
            encoder_frame.reshape(&[1, 1, encoder_frame.dim(1)])?
        } else {
            encoder_frame.clone()
        };

        self.decode_frames(&frame, 1, decoder, decoder_state, joint, last_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_step_output_default() {
        let output = DecodeStepOutput::default();
        assert!(!output.has_tokens());
        assert_eq!(output.blank_count, 0);
        assert_eq!(output.non_blank_count, 0);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = RnntStreamingDecoder::new(1024);
        assert_eq!(decoder.blank_id(), 1024);
        assert_eq!(decoder.max_symbols_per_step, DEFAULT_MAX_SYMBOLS_PER_STEP);

        let decoder = decoder.with_max_symbols(5);
        assert_eq!(decoder.max_symbols_per_step, 5);
    }
}
