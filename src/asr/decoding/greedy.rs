//! Greedy CTC decoding.
//!
//! Simple greedy decoding that selects the most probable token at each timestep.

use mlx_rs::ops::indexing::argmax_axis;
use mlx_rs::Array;

use crate::core::ArrayExt;
use crate::core::error::Result;

/// Perform greedy CTC decoding.
///
/// # Arguments
/// * `log_probs` - Log probabilities [batch, time, vocab_size]
/// * `lengths` - Sequence lengths [batch]
/// * `blank_id` - Blank token ID (typically 0 or vocab_size-1)
///
/// # Returns
/// * Vector of token sequences for each batch item
pub fn greedy_decode(
    log_probs: &Array,
    lengths: &Array,
    blank_id: usize,
) -> Result<Vec<Vec<usize>>> {
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

        let mut tokens = Vec::new();
        let mut prev_token = blank_id;

        for t in 0..seq_len {
            let token = pred_data[offset + t] as usize;

            // CTC decoding: skip blanks and repeated tokens
            if token != blank_id && token != prev_token {
                tokens.push(token);
            }

            prev_token = token;
        }

        results.push(tokens);
    }

    Ok(results)
}

/// Perform greedy CTC decoding with timestamps.
///
/// Returns token sequences along with their start/end frame indices.
pub fn greedy_decode_with_timestamps(
    log_probs: &Array,
    lengths: &Array,
    blank_id: usize,
) -> Result<Vec<Vec<(usize, usize, usize)>>> {
    log_probs.eval()?;
    lengths.eval()?;

    let batch_size = log_probs.shape()[0] as usize;
    let max_len = log_probs.shape()[1] as usize;

    let predictions = argmax_axis(log_probs, 2, false)?;
    predictions.eval()?;

    let lengths_data: Vec<i32> = lengths.to_vec_row_major::<i32>()?;
    let pred_data: Vec<i32> = predictions.to_vec_row_major::<i32>()?;

    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let seq_len = lengths_data[b] as usize;
        let offset = b * max_len;

        let mut tokens_with_timestamps = Vec::new();
        let mut prev_token = blank_id;
        let mut token_start = 0;

        for t in 0..seq_len {
            let token = pred_data[offset + t] as usize;

            if token != prev_token {
                if prev_token != blank_id {
                    // End of previous token
                    tokens_with_timestamps.push((prev_token, token_start, t));
                }
                if token != blank_id {
                    // Start of new token
                    token_start = t;
                }
            }

            prev_token = token;
        }

        // Handle last token
        if prev_token != blank_id {
            tokens_with_timestamps.push((prev_token, token_start, seq_len));
        }

        results.push(tokens_with_timestamps);
    }

    Ok(results)
}
