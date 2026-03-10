//! Beam search RNN-T decoding.
//!
//! Implements beam search decoding for RNN-T models, which maintains
//! multiple hypothesis candidates and expands the most promising ones.

use std::cmp::Ordering;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::asr::modules::{RNNTJoint, StatelessTransducerDecoder};
use crate::core::error::Result;

/// Maximum number of symbols that can be emitted per encoder frame.
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

/// A hypothesis in beam search.
#[derive(Clone)]
struct Hypothesis {
    /// Token sequence so far
    tokens: Vec<usize>,
    /// Log probability score
    score: f32,
    /// Decoder state
    state: Array,
}

impl Hypothesis {
    fn new(state: Array) -> Self {
        Self {
            tokens: Vec::new(),
            score: 0.0,
            state,
        }
    }

    fn with_token(&self, token: usize, log_prob: f32, new_state: Array) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        Self {
            tokens,
            score: self.score + log_prob,
            state: new_state,
        }
    }

    fn with_blank(&self, log_prob: f32) -> Self {
        Self {
            tokens: self.tokens.clone(),
            score: self.score + log_prob,
            state: self.state.clone(),
        }
    }
}

// For max-heap ordering by score
impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Hypothesis {}

impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (higher scores first)
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Perform beam search RNN-T decoding.
///
/// # Arguments
/// * `encoder_output` - Encoder output [batch, time, encoder_hidden]
/// * `encoder_lengths` - Valid lengths for each sequence [batch]
/// * `decoder` - The prediction network
/// * `joint` - The joint network
/// * `blank_id` - Blank token ID
/// * `beam_size` - Number of hypotheses to keep
/// * `max_symbols_per_step` - Maximum symbols per encoder frame
///
/// # Returns
/// * Vector of decoded token sequences (best hypothesis for each batch item)
pub fn rnnt_beam_decode(
    encoder_output: &Array,
    encoder_lengths: &Array,
    decoder: &StatelessTransducerDecoder,
    joint: &RNNTJoint,
    blank_id: usize,
    beam_size: usize,
    max_symbols_per_step: Option<usize>,
) -> Result<Vec<Vec<usize>>> {
    encoder_output.eval()?;
    encoder_lengths.eval()?;

    let batch_size = encoder_output.shape()[0] as usize;
    let max_symbols = max_symbols_per_step.unwrap_or(DEFAULT_MAX_SYMBOLS_PER_STEP);

    // Get encoder lengths
    let lengths_i32 = encoder_lengths.as_type::<i32>()?;
    lengths_i32.eval()?;
    let lengths_data: Vec<i32> = lengths_i32.as_slice::<i32>().to_vec();

    let mut results = Vec::with_capacity(batch_size);

    // Process each sequence independently
    for b in 0..batch_size {
        let seq_len = lengths_data[b] as usize;
        let best_hyp = beam_search_single(
            encoder_output,
            b,
            seq_len,
            decoder,
            joint,
            blank_id,
            beam_size,
            max_symbols,
        )?;
        results.push(best_hyp);
    }

    Ok(results)
}

/// Beam search for a single sequence.
fn beam_search_single(
    encoder_output: &Array,
    batch_idx: usize,
    seq_len: usize,
    decoder: &StatelessTransducerDecoder,
    joint: &RNNTJoint,
    blank_id: usize,
    beam_size: usize,
    max_symbols: usize,
) -> Result<Vec<usize>> {
    let vocab_size = joint.vocab_size();

    // Initialize with empty hypothesis
    let init_state = decoder.initialize_state(1)?;
    let mut beams = vec![Hypothesis::new(init_state)];

    // Process each encoder frame
    for t in 0..seq_len {
        // Get encoder frame: [1, 1, encoder_hidden]
        let enc_frame = encoder_output.index((
            batch_idx as i32..batch_idx as i32 + 1,
            t as i32..t as i32 + 1,
            ..,
        ));

        // Expand hypotheses for this frame
        let mut new_beams: Vec<Hypothesis> = Vec::new();

        for hyp in &beams {
            // Each hypothesis can emit multiple symbols before moving to next frame
            let expanded = expand_hypothesis(
                &enc_frame,
                hyp,
                decoder,
                joint,
                blank_id,
                vocab_size,
                beam_size,
                max_symbols,
            )?;
            new_beams.extend(expanded);
        }

        // Keep top beam_size hypotheses
        new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        new_beams.truncate(beam_size);

        if new_beams.is_empty() {
            // Keep at least the best hypothesis from before
            if let Some(best) = beams.first() {
                new_beams.push(best.clone());
            }
        }

        beams = new_beams;
    }

    // Return best hypothesis
    Ok(beams.first().map(|h| h.tokens.clone()).unwrap_or_default())
}

/// Expand a hypothesis by emitting symbols until blank.
fn expand_hypothesis(
    enc_frame: &Array,
    hyp: &Hypothesis,
    decoder: &StatelessTransducerDecoder,
    joint: &RNNTJoint,
    blank_id: usize,
    _vocab_size: usize,
    beam_size: usize,
    max_symbols: usize,
) -> Result<Vec<Hypothesis>> {
    let mut results = Vec::new();
    let mut frontier = vec![hyp.clone()];

    // Expand until we have enough blank-terminated hypotheses
    for _ in 0..max_symbols {
        let mut new_frontier = Vec::new();

        for h in &frontier {
            // Get prediction output
            let last_token = if h.tokens.is_empty() {
                None
            } else {
                Some(last_token_array(&h.tokens)?)
            };

            let (pred_output, new_state) = decoder.predict_one(last_token.as_ref(), &h.state)?;

            // Compute joint probabilities
            let log_probs = joint.forward_one(enc_frame, &pred_output)?;
            log_probs.eval()?;

            // Get log probabilities as a slice
            let log_probs_data: Vec<f32> = log_probs.as_slice::<f32>().to_vec();

            // Add blank hypothesis (terminates this expansion)
            let blank_log_prob = log_probs_data[blank_id];
            results.push(h.with_blank(blank_log_prob));

            // Add non-blank hypotheses (continue expansion)
            let mut token_scores: Vec<(usize, f32)> = log_probs_data
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != blank_id)
                .map(|(i, &p)| (i, p))
                .collect();

            // Sort by score and take top k
            token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            token_scores.truncate(beam_size);

            for (token, log_prob) in token_scores {
                new_frontier.push(h.with_token(token, log_prob, new_state.clone()));
            }
        }

        if new_frontier.is_empty() {
            break;
        }

        // Keep only top beam_size candidates for next iteration
        new_frontier.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        new_frontier.truncate(beam_size);
        frontier = new_frontier;
    }

    // Add any remaining frontier hypotheses with blank
    for h in &frontier {
        // These didn't emit blank within max_symbols, force blank
        results.push(h.clone());
    }

    Ok(results)
}

/// Create an array containing the last token for prediction.
fn last_token_array(tokens: &[usize]) -> Result<Array> {
    let last = tokens.last().copied().unwrap_or(0) as i32;
    Ok(Array::from_slice(&[last], &[1, 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypothesis_ordering() {
        let state = Array::zeros::<f32>(&[1, 1]).unwrap();
        let h1 = Hypothesis {
            tokens: vec![],
            score: -1.0,
            state: state.clone(),
        };
        let h2 = Hypothesis {
            tokens: vec![],
            score: -2.0,
            state: state.clone(),
        };

        // h1 should be "greater" (higher priority) due to higher score
        assert!(h1 > h2);
    }
}
