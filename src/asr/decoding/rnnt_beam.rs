//! Beam search RNN-T decoding.
//!
//! A port of NeMo's default `BeamRNNTInfer` search (`search_type: default`,
//! after Graves 2012). For each encoder frame it keeps expanding the best
//! hypothesis: ending it with blank moves it on to the next frame, and each of
//! the `beam_size` best tokens extends it within the frame. The frame is done
//! once `beam_size` hypotheses that ended in blank outscore every hypothesis
//! still being expanded.

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

use crate::asr::modules::{DecoderState, PredictionNetwork};
use super::TimedToken;
use crate::asr::modules::RnntJoint;
use crate::core::ArrayExt;
use crate::core::error::{Error, Result};

/// Expansions per frame, as a multiple of the beam size, before the search
/// moves on with the best hypotheses so far. NeMo has no limit (scores only
/// fall, so the search always ends); this bounds the work on a pathological
/// frame.
const MAX_EXPANSIONS_PER_BEAM: usize = 10;

/// A hypothesis in beam search.
#[derive(Clone)]
struct Hypothesis {
    /// Token sequence so far
    tokens: Vec<usize>,
    /// Encoder frame each token was emitted on
    frames: Vec<usize>,
    /// Log probability score
    score: f32,
    /// Decoder state before the last token in `tokens` was consumed
    state: DecoderState,
    /// Prediction for `tokens`, projected into the joint space, with the
    /// decoder state after it. Computed when the hypothesis is first expanded.
    prediction: Option<(Array, DecoderState)>,
}

/// Perform beam search RNN-T decoding.
///
/// # Arguments
/// * `encoder_output` - Encoder output [batch, time, encoder_hidden]
/// * `encoder_lengths` - Valid lengths for each sequence `[batch]`
/// * `decoder` - The prediction network
/// * `joint` - The joint network
/// * `blank_id` - Blank token ID
/// * `beam_size` - Number of hypotheses to keep
///
/// # Returns
/// * The best hypothesis for each batch item, chosen by length-normalized
///   score as in NeMo (`score_norm: true`), with the frame of each token
pub fn rnnt_beam_decode(
    encoder_output: &Array,
    encoder_lengths: &Array,
    decoder: &PredictionNetwork,
    joint: &RnntJoint,
    blank_id: usize,
    beam_size: usize,
) -> Result<Vec<Vec<TimedToken>>> {
    if beam_size == 0 {
        return Err(Error::decoding("beam size must be at least 1"));
    }

    let lengths: Vec<i32> = encoder_lengths
        .as_dtype(mlx_rs::Dtype::Int32)?
        .to_vec_row_major()?;

    let mut results = Vec::with_capacity(lengths.len());
    for (b, &len) in lengths.iter().enumerate() {
        let seq_len = (len.max(0) as usize).min(encoder_output.dim(1) as usize);
        let frames = encoder_output.index((b as i32, ..seq_len as i32, ..));
        let enc_proj = joint.project_encoder(&frames)?;
        results.push(beam_search_single(
            &enc_proj, seq_len, decoder, joint, blank_id, beam_size,
        )?);
    }

    Ok(results)
}

/// Beam search over one sequence of projected encoder frames [time, joint_hidden].
fn beam_search_single(
    enc_proj: &Array,
    seq_len: usize,
    decoder: &PredictionNetwork,
    joint: &RnntJoint,
    blank_id: usize,
    beam_size: usize,
) -> Result<Vec<TimedToken>> {
    let beam_k = beam_size.min(joint.vocab_size().saturating_sub(1)).max(1);
    let max_expansions = beam_size * MAX_EXPANSIONS_PER_BEAM;

    let mut kept = vec![Hypothesis {
        tokens: Vec::new(),
        frames: Vec::new(),
        score: 0.0,
        state: decoder.initialize_state(1)?,
        prediction: None,
    }];

    for t in 0..seq_len {
        let enc_t = enc_proj.index((t as i32..t as i32 + 1, ..));
        let mut hyps = std::mem::take(&mut kept);
        let mut done = false;

        for _ in 0..max_expansions {
            let mut best = hyps.swap_remove(best_index(&hyps));
            let (pred_proj, next_state) = match best.prediction.take() {
                Some(prediction) => prediction,
                None => predict(decoder, joint, best.tokens.last().copied(), &best.state)?,
            };

            let logits = joint.logits_from_projections(&enc_t, &pred_proj)?;
            let log_probs: Vec<f32> = mlx_rs::nn::log_softmax(&logits, -1)?
                .as_dtype(mlx_rs::Dtype::Float32)?
                .to_vec_row_major()?;

            for (token, log_prob) in top_k_non_blank(&log_probs, blank_id, beam_k) {
                let mut tokens = best.tokens.clone();
                tokens.push(token);
                let mut frames = best.frames.clone();
                frames.push(t);
                hyps.push(Hypothesis {
                    tokens,
                    frames,
                    score: best.score + log_prob,
                    state: next_state.clone(),
                    prediction: None,
                });
            }

            // Ending in blank keeps the same tokens, so the prediction carries over
            best.score += log_probs[blank_id];
            best.prediction = Some((pred_proj, next_state));
            kept.push(best);

            // Stop once enough finished hypotheses beat every unfinished one
            if hyps.is_empty() {
                done = true;
                break;
            }
            let hyps_max = hyps[best_index(&hyps)].score;
            if kept.iter().filter(|h| h.score > hyps_max).count() >= beam_size {
                kept.retain(|h| h.score > hyps_max);
                done = true;
                break;
            }
        }

        if !done {
            kept.sort_by(|a, b| b.score.total_cmp(&a.score));
            kept.truncate(beam_size);
        }
    }

    // NeMo's hypotheses start with a blank, so its length norm counts one more
    // token than they emit
    let best = kept
        .iter()
        .max_by(|a, b| {
            let norm = |h: &Hypothesis| h.score / (h.tokens.len() + 1) as f32;
            norm(a).total_cmp(&norm(b))
        })
        .map(|h| {
            h.tokens
                .iter()
                .zip(&h.frames)
                .map(|(&id, &t)| TimedToken {
                    id,
                    start_frame: t,
                    end_frame: t + 1,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(best)
}

/// Prediction for a history ending in `token` (`None` for the start of the
/// sequence), projected into the joint space, and the decoder state after it.
fn predict(
    decoder: &PredictionNetwork,
    joint: &RnntJoint,
    token: Option<usize>,
    state: &DecoderState,
) -> Result<(Array, DecoderState)> {
    let token = token.map(|t| Array::from_slice(&[t as i32], &[1, 1]));
    let (pred, next_state) = decoder.predict_one(token.as_ref(), state)?;
    Ok((
        joint.project_prediction(&pred.reshape(&[1, -1])?)?,
        next_state,
    ))
}

/// Index of the highest-scoring hypothesis. `hyps` must not be empty.
fn best_index(hyps: &[Hypothesis]) -> usize {
    hyps.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        .map(|(i, _)| i)
        .expect("beam search always has a hypothesis to expand")
}

/// The `k` most probable tokens other than blank, best first.
fn top_k_non_blank(log_probs: &[f32], blank_id: usize, k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = log_probs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(i, _)| i != blank_id)
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_non_blank_skips_blank() {
        let log_probs = [-0.1, -3.0, -0.5, -2.0];
        assert_eq!(
            top_k_non_blank(&log_probs, 0, 2),
            vec![(2, -0.5), (3, -2.0)]
        );
        assert_eq!(top_k_non_blank(&log_probs, 3, 10).len(), 3);
    }
}
