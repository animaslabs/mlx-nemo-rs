//! Greedy CTC decoding.
//!
//! Simple greedy decoding that selects the most probable token at each timestep.

use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::{Array, ops};

use super::TimedToken;
use crate::core::ArrayExt;
use crate::core::error::Result;

/// Perform greedy CTC decoding.
///
/// # Arguments
/// * `log_probs` - Log probabilities [batch, time, vocab_size]
/// * `lengths` - Sequence lengths `[batch]`
/// * `blank_id` - Blank token ID (typically 0 or vocab_size-1)
///
/// # Returns
/// * The tokens of each batch item, each spanning the frames it was repeated on
pub fn greedy_decode(
    log_probs: &Array,
    lengths: &Array,
    blank_id: usize,
) -> Result<Vec<Vec<TimedToken>>> {
    // Evaluate to ensure we can access the data
    log_probs.eval()?;
    lengths.eval()?;

    let batch_size = log_probs.shape()[0] as usize;
    let max_len = log_probs.shape()[1] as usize;
    let _vocab_size = log_probs.shape()[2] as usize;

    // Get the argmax predictions
    let predictions = argmax_axis(log_probs, 2, false)?;
    predictions.eval()?;

    // Get lengths as a vector - convert to i32 if needed
    let lengths_i32 = lengths.as_type::<i32>()?;
    lengths_i32.eval()?;
    let lengths_data: Vec<i32> = lengths_i32.to_vec_row_major::<i32>()?;

    // Get predictions as a flat vector
    let predictions_u32 = predictions.as_type::<u32>()?;
    predictions_u32.eval()?;
    let pred_data: Vec<u32> = predictions_u32.to_vec_row_major::<u32>()?;

    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let seq_len = lengths_data[b] as usize;
        let offset = b * max_len;

        let labels: Vec<usize> = pred_data[offset..offset + seq_len]
            .iter()
            .map(|&id| id as usize)
            .collect();
        let mut tokens = Vec::new();
        ctc_collapse(&labels, blank_id, 0, &mut None, &mut tokens);
        results.push(tokens);
    }

    Ok(results)
}

/// The greedy label of each frame of CTC log probabilities `[time, vocab]`,
/// before collapsing repeats and blanks.
///
/// Blank wins a frame only when it beats the best allowed token by more than
/// `blank_threshold` nats (plain argmax at 0), and `masked_token_ids` are
/// never chosen.
pub fn ctc_frame_labels(
    log_probs: &Array,
    blank_id: usize,
    blank_threshold: f32,
    masked_token_ids: Option<&[usize]>,
) -> Result<Vec<usize>> {
    if log_probs.size() == 0 {
        return Ok(Vec::new());
    }
    let vocab = log_probs.dim(-1) as usize;
    let mut mask = vec![0.0f32; vocab];
    for id in masked_token_ids
        .unwrap_or_default()
        .iter()
        .copied()
        .chain([blank_id])
    {
        if id < vocab {
            mask[id] = f32::NEG_INFINITY;
        }
    }
    let non_blank = ops::add(log_probs, Array::from_slice(&mask, &[vocab as i32]))?;
    let best = argmax_axis(&non_blank, -1, false)?;
    let margin = ops::subtract(
        log_probs.index((.., blank_id as i32)),
        ops::max_axis(&non_blank, -1, false)?,
    )?;
    let labels = ops::select(
        ops::gt(&margin, Array::from_f32(blank_threshold))?,
        Array::from_int(blank_id as i32),
        best.as_type::<i32>()?,
    )?;
    Ok(labels
        .to_vec_row_major::<i32>()?
        .into_iter()
        .map(|id| id as usize)
        .collect())
}

/// Append the tokens of greedy frame `labels` to `tokens`, collapsing repeats
/// and blanks. `prev` is the label of the frame before the first one (`None`
/// at the start of a stream) and is updated to the last label, so a stream
/// collapses the same across chunks as in one piece. The first label is
/// encoder frame `first_frame`.
pub fn ctc_collapse(
    labels: &[usize],
    blank_id: usize,
    first_frame: usize,
    prev: &mut Option<usize>,
    tokens: &mut Vec<TimedToken>,
) {
    for (i, &label) in labels.iter().enumerate() {
        let frame = first_frame + i;
        if label != blank_id {
            match tokens.last_mut() {
                Some(last) if *prev == Some(label) => last.end_frame = frame + 1,
                _ => tokens.push(TimedToken {
                    id: label,
                    start_frame: frame,
                    end_frame: frame + 1,
                }),
            }
        }
        *prev = Some(label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctc_collapse_across_chunks() {
        let blank = 9;
        // "a a _ a b b" split after the second frame and after the fifth
        let mut tokens = Vec::new();
        let mut prev = None;
        ctc_collapse(&[1, 1], blank, 0, &mut prev, &mut tokens);
        ctc_collapse(&[blank, 1, 2], blank, 2, &mut prev, &mut tokens);
        ctc_collapse(&[2], blank, 5, &mut prev, &mut tokens);
        let spans: Vec<_> = tokens
            .iter()
            .map(|t| (t.id, t.start_frame, t.end_frame))
            .collect();
        assert_eq!(spans, [(1, 0, 2), (1, 3, 4), (2, 4, 6)]);
    }

    #[test]
    fn test_ctc_frame_labels() {
        let blank = 2;
        let log_probs = Array::from_slice(
            &[
                -0.1f32, -3.0, -2.5, // token 0
                -3.0, -2.0, -0.2, // blank by 1.8
                -9.0, -9.0, -0.1, // blank by 8.9
            ],
            &[3, 3],
        );
        let labels = |threshold, masked: Option<&[usize]>| {
            ctc_frame_labels(&log_probs, blank, threshold, masked).unwrap()
        };
        assert_eq!(labels(0.0, None), [0, 2, 2]);
        assert_eq!(labels(2.0, None), [0, 1, 2]);
        assert_eq!(labels(0.0, Some(&[0])), [2, 2, 2]);
        assert_eq!(labels(5.0, Some(&[0])), [1, 1, 2]);
    }
}
