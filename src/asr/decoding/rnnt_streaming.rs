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
use mlx_rs::ops::{self, max_axis};
use mlx_rs::Array;
use tracing::debug;

use crate::asr::models::{DecoderState, PredictionNetwork};
use crate::asr::modules::RNNTJoint;
use crate::core::ArrayExt;
use crate::core::error::Result;

/// Encoder frames scored per GPU round trip while the prediction is unchanged.
const DECODE_WINDOW: usize = 32;

/// Maximum number of symbols that can be emitted per encoder frame.
/// This is a safety limit to prevent infinite loops.
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

/// Default blank threshold (disabled).
/// When > 0, requires blank log-probability to exceed best non-blank by this margin.
/// This helps prevent the model from over-predicting blanks during streaming.
const DEFAULT_BLANK_THRESHOLD: f32 = 0.0;

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
    /// Blank threshold: log-probability margin required for blank to be emitted.
    /// When > 0, blank is only emitted if blank_log_prob - best_non_blank_log_prob > threshold.
    /// This helps prevent over-predicting blanks during streaming.
    /// - 0.0 = disabled (default argmax behavior)
    /// - 0.5 = blank must be ~1.6x more probable than best non-blank
    /// - 1.0 = blank must be ~2.7x more probable
    /// - 2.0 = blank must be ~7.4x more probable
    blank_threshold: f32,
}

impl RnntStreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new(blank_id: usize) -> Self {
        Self {
            blank_id,
            max_symbols_per_step: DEFAULT_MAX_SYMBOLS_PER_STEP,
            blank_threshold: DEFAULT_BLANK_THRESHOLD,
        }
    }

    /// Set the maximum symbols per step.
    pub fn with_max_symbols(mut self, max_symbols: usize) -> Self {
        self.max_symbols_per_step = max_symbols;
        self
    }

    /// Set the blank threshold.
    ///
    /// When > 0, blank is only emitted if blank_log_prob - best_non_blank_log_prob > threshold.
    /// This helps prevent the model from over-predicting blanks during streaming.
    ///
    /// Recommended values:
    /// - 0.0 = disabled (default argmax behavior)
    /// - 0.5 = blank must be ~1.6x more probable than best non-blank (mild)
    /// - 1.0 = blank must be ~2.7x more probable (moderate)
    /// - 2.0 = blank must be ~7.4x more probable (aggressive)
    pub fn with_blank_threshold(mut self, threshold: f32) -> Self {
        self.blank_threshold = threshold;
        self
    }

    /// Get the blank token ID.
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Get the current blank threshold.
    pub fn blank_threshold(&self) -> f32 {
        self.blank_threshold
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
    /// * `masked_token_ids` - Optional token IDs that are never emitted
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
        masked_token_ids: Option<&[usize]>,
    ) -> Result<DecodeStepOutput> {
        let enc_proj = if valid_frames > 0 {
            let frames = encoder_output.index((0, ..valid_frames as i32, ..));
            joint.project_encoder(&frames)?
        } else {
            Array::zeros::<f32>(&[0])?
        };
        self.decode_projected(
            &enc_proj,
            valid_frames,
            decoder,
            decoder_state,
            joint,
            last_token,
            masked_token_ids,
        )
    }

    /// Greedy-decode encoder frames already projected by
    /// [`RNNTJoint::project_encoder`] ([time, joint_hidden]).
    ///
    /// Equivalent to frame-by-frame greedy decoding, but while the prediction
    /// is unchanged (after a blank) the joint scores up to `DECODE_WINDOW`
    /// frames at once and the first non-blank frame is found on the host, so
    /// there is one GPU round trip per emitted token or window of blanks rather
    /// than one per frame. The prediction network only reruns after a token.
    pub fn decode_projected(
        &self,
        enc_proj: &Array,
        valid_frames: usize,
        decoder: &PredictionNetwork,
        decoder_state: &mut DecoderState,
        joint: &RNNTJoint,
        last_token: &mut Option<usize>,
        masked_token_ids: Option<&[usize]>,
    ) -> Result<DecodeStepOutput> {
        let blank = self.blank_id;
        let mut output = DecodeStepOutput {
            labels: vec![blank; valid_frames],
            ..Default::default()
        };
        if valid_frames == 0 {
            return Ok(output);
        }

        // Additive mask over the logits that keeps only emittable non-blank tokens
        let vocab = joint.vocab_size();
        let mut non_blank_mask = vec![0.0f32; vocab];
        for id in masked_token_ids.unwrap_or_default().iter().copied().chain([blank]) {
            if id < vocab {
                non_blank_mask[id] = f32::NEG_INFINITY;
            }
        }
        let non_blank_mask = Array::from_slice(&non_blank_mask, &[vocab as i32]);
        let blank_arr = Array::from_int(blank as i32);
        let threshold = Array::from_f32(self.blank_threshold);

        // Prediction for the current history, and the state after consuming
        // `last_token` (committed only once a new token is emitted)
        let predict = |token: Option<usize>, state: &DecoderState| -> Result<(Array, DecoderState)> {
            let token = token.map(|t| Array::from_slice(&[t as i32], &[1, 1]));
            let (pred, next_state) = decoder.predict_one(token.as_ref(), state)?;
            Ok((joint.project_prediction(&pred.reshape(&[1, -1])?)?, next_state))
        };
        let (mut pred_proj, mut next_state) = predict(*last_token, decoder_state)?;

        let mut t = 0;
        let mut symbols = 0;
        while t < valid_frames {
            let window = DECODE_WINDOW.min(valid_frames - t);
            let enc = enc_proj.index((t as i32..(t + window) as i32, ..));
            let logits = joint.logits_from_projections(&enc, &pred_proj)?;

            // Blank wins when it beats the best allowed token by more than the
            // threshold (argmax at threshold 0; ties go to the token, as argmax
            // with blank last would)
            let non_blank = ops::add(&logits, &non_blank_mask)?;
            let best = argmax_axis(&non_blank, -1, false)?;
            let margin = ops::subtract(
                &logits.index((.., blank as i32)),
                &max_axis(&non_blank, -1, false)?,
            )?;
            let choice = ops::select(margin.gt(&threshold)?, &blank_arr, &best)?;
            let choice: Vec<u32> = choice.as_dtype(mlx_rs::Dtype::Uint32)?.to_vec_row_major()?;

            let Some(k) = choice.iter().position(|&c| c as usize != blank) else {
                output.blank_count += window;
                t += window;
                symbols = 0;
                continue;
            };
            if k > 0 {
                output.blank_count += k;
                t += k;
                symbols = 0;
            }

            let token = choice[k] as usize;
            output.new_tokens.push(token);
            output.timesteps.push(t);
            output.non_blank_count += 1;
            if symbols == 0 {
                output.labels[t] = token;
            }
            *last_token = Some(token);
            let (proj, state) = predict(Some(token), &next_state)?;
            *decoder_state = std::mem::replace(&mut next_state, state);
            pred_proj = proj;

            symbols += 1;
            if symbols >= self.max_symbols_per_step {
                t += 1;
                symbols = 0;
            }
        }

        debug!(
            frames = valid_frames,
            new_tokens = output.new_tokens.len(),
            blank_count = output.blank_count,
            "DECODER: decode_frames complete"
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
        masked_token_ids: Option<&[usize]>,
    ) -> Result<DecodeStepOutput> {
        // Ensure frame has batch dimension: [1, 1, d_model]
        let frame = if encoder_frame.ndim() == 2 {
            encoder_frame.reshape(&[1, 1, encoder_frame.dim(1)])?
        } else {
            encoder_frame.clone()
        };

        self.decode_frames(
            &frame,
            1,
            decoder,
            decoder_state,
            joint,
            last_token,
            masked_token_ids,
        )
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
        assert_eq!(decoder.blank_threshold(), DEFAULT_BLANK_THRESHOLD);

        let decoder = decoder.with_max_symbols(5);
        assert_eq!(decoder.max_symbols_per_step, 5);
    }

    #[test]
    fn test_decoder_blank_threshold() {
        let decoder = RnntStreamingDecoder::new(1024)
            .with_blank_threshold(1.5);
        assert_eq!(decoder.blank_id(), 1024);
        assert_eq!(decoder.blank_threshold(), 1.5);

        // Can chain with other builder methods
        let decoder = RnntStreamingDecoder::new(512)
            .with_max_symbols(3)
            .with_blank_threshold(2.0);
        assert_eq!(decoder.blank_id(), 512);
        assert_eq!(decoder.max_symbols_per_step, 3);
        assert_eq!(decoder.blank_threshold(), 2.0);
    }
}
