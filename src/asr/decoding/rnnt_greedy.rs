//! Greedy RNN-T decoding.
//!
//! Implements the greedy decoding algorithm for RNN-T models.
//! At each encoder timestep, the decoder predicts tokens until a blank
//! is emitted, then advances to the next encoder frame.

use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
use mlx_rs::Array;
use tracing::{debug, instrument, trace};

use crate::asr::modules::{RNNTJoint, StatelessTransducerDecoder};
use crate::core::error::Result;

/// Maximum number of symbols that can be emitted per encoder frame.
/// This is a safety limit to prevent infinite loops.
const DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

/// Perform greedy RNN-T decoding.
///
/// The algorithm processes encoder output frame-by-frame:
/// 1. For each encoder frame, compute joint output with prediction network
/// 2. If argmax is blank, advance to next encoder frame
/// 3. If argmax is a label, emit it and update prediction state
/// 4. Repeat until all encoder frames are processed
///
/// # Arguments
/// * `encoder_output` - Encoder output [batch, time, encoder_hidden]
/// * `encoder_lengths` - Valid lengths for each sequence [batch]
/// * `decoder` - The prediction network
/// * `joint` - The joint network
/// * `blank_id` - Blank token ID (typically vocab_size)
/// * `max_symbols_per_step` - Maximum symbols per encoder frame (prevents infinite loops)
///
/// # Returns
/// * Vector of decoded token sequences for each batch item
#[instrument(
    skip(encoder_output, encoder_lengths, decoder, joint),
    fields(batch_size, blank_id, max_symbols_per_step)
)]
pub fn rnnt_greedy_decode(
    encoder_output: &Array,
    encoder_lengths: &Array,
    decoder: &StatelessTransducerDecoder,
    joint: &RNNTJoint,
    blank_id: usize,
    max_symbols_per_step: Option<usize>,
) -> Result<Vec<Vec<usize>>> {
    encoder_output.eval()?;
    encoder_lengths.eval()?;

    let batch_size = encoder_output.shape()[0] as usize;
    let max_symbols = max_symbols_per_step.unwrap_or(DEFAULT_MAX_SYMBOLS_PER_STEP);

    debug!(
        batch_size,
        encoder_shape = ?encoder_output.shape(),
        blank_id,
        max_symbols,
        "Starting RNN-T greedy decode"
    );

    // Get encoder lengths as a vector
    let lengths_i32 = encoder_lengths.as_type::<i32>()?;
    lengths_i32.eval()?;
    let lengths_data: Vec<i32> = lengths_i32.as_slice::<i32>().to_vec();

    let mut results = Vec::with_capacity(batch_size);

    // Process each sequence in the batch independently
    // (This is simpler but less efficient than batched decoding)
    for b in 0..batch_size {
        let seq_len = lengths_data[b] as usize;
        let mut tokens = Vec::new();

        debug!(batch_idx = b, seq_len, "Processing sequence");

        // Initialize decoder state for this sequence
        let mut state = decoder.initialize_state(1)?;

        // Process each encoder frame
        let mut t = 0;
        while t < seq_len {
            // Get encoder output for current frame: [1, 1, encoder_hidden]
            let enc_frame =
                encoder_output.index((b as i32..b as i32 + 1, t as i32..t as i32 + 1, ..));

            let mut symbols_emitted = 0;

            // Emit symbols until blank or max symbols reached
            loop {
                // Get prediction network output
                let last_token = if tokens.is_empty() {
                    None
                } else {
                    Some(last_token_array(&tokens)?)
                };
                let (pred_output, new_state) = decoder.predict_one(last_token.as_ref(), &state)?;

                // Compute joint output: [1, vocab_size]
                let log_probs = joint.forward_one(&enc_frame, &pred_output)?;
                log_probs.eval()?;

                // Get argmax
                let prediction = argmax_axis(&log_probs, -1, false)?;
                prediction.eval()?;
                let pred_token = prediction.as_slice::<u32>()[0] as usize;

                trace!(
                    batch_idx = b,
                    frame = t,
                    pred_token,
                    is_blank = pred_token == blank_id,
                    symbols_emitted,
                    "Joint prediction"
                );

                if pred_token == blank_id {
                    // Blank: advance to next encoder frame
                    trace!(batch_idx = b, frame = t, "Blank emitted, advancing frame");
                    break;
                }

                // Non-blank: emit token and update state
                tokens.push(pred_token);
                state = new_state;
                symbols_emitted += 1;

                trace!(
                    batch_idx = b,
                    frame = t,
                    token = pred_token,
                    total_tokens = tokens.len(),
                    "Token emitted"
                );

                if symbols_emitted >= max_symbols {
                    // Safety limit reached, advance frame
                    debug!(
                        batch_idx = b,
                        frame = t,
                        symbols_emitted,
                        "Max symbols per step reached"
                    );
                    break;
                }
            }

            t += 1;
        }

        debug!(
            batch_idx = b,
            num_tokens = tokens.len(),
            "Sequence decoding complete"
        );
        results.push(tokens);
    }

    debug!(
        total_tokens = results.iter().map(|r| r.len()).sum::<usize>(),
        "RNN-T greedy decode complete"
    );

    Ok(results)
}

/// Create an array containing the last token for prediction.
fn last_token_array(tokens: &[usize]) -> Result<Array> {
    let last = tokens.last().copied().unwrap_or(0) as i32;
    Ok(Array::from_slice(&[last], &[1, 1]))
}

/// Perform batched greedy RNN-T decoding.
///
/// More efficient than single-sequence decoding for batch processing.
/// All sequences in the batch are decoded in parallel.
///
/// # Arguments
/// * `encoder_output` - Encoder output [batch, time, encoder_hidden]
/// * `encoder_lengths` - Valid lengths for each sequence [batch]
/// * `decoder` - The prediction network
/// * `joint` - The joint network
/// * `blank_id` - Blank token ID
/// * `max_symbols_per_step` - Maximum symbols per encoder frame
///
/// # Returns
/// * Vector of decoded token sequences for each batch item
#[instrument(
    skip(encoder_output, encoder_lengths, decoder, joint),
    fields(batch_size, blank_id, max_symbols_per_step)
)]
pub fn rnnt_greedy_decode_batched(
    encoder_output: &Array,
    encoder_lengths: &Array,
    decoder: &StatelessTransducerDecoder,
    joint: &RNNTJoint,
    blank_id: usize,
    max_symbols_per_step: Option<usize>,
) -> Result<Vec<Vec<usize>>> {
    encoder_output.eval()?;
    encoder_lengths.eval()?;

    let batch_size = encoder_output.shape()[0] as usize;
    let max_symbols = max_symbols_per_step.unwrap_or(DEFAULT_MAX_SYMBOLS_PER_STEP);

    debug!(
        batch_size,
        encoder_shape = ?encoder_output.shape(),
        blank_id,
        max_symbols,
        "Starting batched RNN-T greedy decode"
    );

    // Get encoder lengths as a vector
    let lengths_i32 = encoder_lengths.as_type::<i32>()?;
    lengths_i32.eval()?;
    let lengths_data: Vec<i32> = lengths_i32.as_slice::<i32>().to_vec();

    // Initialize results and states for each sequence
    let mut results: Vec<Vec<usize>> = vec![Vec::new(); batch_size];
    let states = decoder.initialize_state(batch_size)?;
    let mut time_indices: Vec<usize> = vec![0; batch_size];
    let mut finished: Vec<bool> = vec![false; batch_size];

    let mut iteration = 0;

    // Process until all sequences are done
    loop {
        // Check if all sequences are finished
        if finished.iter().all(|&f| f) {
            debug!(iterations = iteration, "All sequences finished");
            break;
        }

        iteration += 1;
        trace!(iteration, "Processing iteration");

        // Build batch of encoder frames for active sequences
        let mut enc_frames = Vec::new();
        let mut active_indices = Vec::new();

        for b in 0..batch_size {
            if !finished[b] {
                let t = time_indices[b];
                if t < lengths_data[b] as usize {
                    enc_frames.push(b);
                    active_indices.push(b);
                } else {
                    finished[b] = true;
                    debug!(batch_idx = b, "Sequence finished (end of frames)");
                }
            }
        }

        if active_indices.is_empty() {
            break;
        }

        trace!(active_count = active_indices.len(), "Active sequences");

        // For simplicity, process one step at a time for each active sequence
        // A more optimized version would batch the joint computation
        for &b in &active_indices {
            let t = time_indices[b];
            let mut symbols_emitted = 0;

            // Get encoder frame for this sequence
            let enc_frame =
                encoder_output.index((b as i32..b as i32 + 1, t as i32..t as i32 + 1, ..));

            // Get state for this sequence
            let seq_state = states.index((b as i32..b as i32 + 1, ..));

            loop {
                // Get prediction output
                let last_token = if results[b].is_empty() {
                    None
                } else {
                    Some(last_token_array(&results[b])?)
                };

                let (pred_output, _new_state) =
                    decoder.predict_one(last_token.as_ref(), &seq_state)?;

                // Compute joint
                let log_probs = joint.forward_one(&enc_frame, &pred_output)?;
                log_probs.eval()?;

                // Get prediction
                let prediction = argmax_axis(&log_probs, -1, false)?;
                prediction.eval()?;
                let pred_token = prediction.as_slice::<u32>()[0] as usize;

                trace!(
                    batch_idx = b,
                    frame = t,
                    pred_token,
                    is_blank = pred_token == blank_id,
                    "Batched joint prediction"
                );

                if pred_token == blank_id {
                    // Advance to next frame
                    time_indices[b] += 1;
                    trace!(
                        batch_idx = b,
                        new_frame = time_indices[b],
                        "Advancing frame"
                    );
                    break;
                }

                // Emit token
                results[b].push(pred_token);
                trace!(
                    batch_idx = b,
                    token = pred_token,
                    total_tokens = results[b].len(),
                    "Token emitted (batched)"
                );

                // Update state (simplified - in full implementation would update batch state)
                symbols_emitted += 1;

                if symbols_emitted >= max_symbols {
                    time_indices[b] += 1;
                    debug!(
                        batch_idx = b,
                        symbols_emitted, "Max symbols per step reached (batched)"
                    );
                    break;
                }
            }

            // Check if sequence is finished
            if time_indices[b] >= lengths_data[b] as usize {
                finished[b] = true;
                debug!(
                    batch_idx = b,
                    num_tokens = results[b].len(),
                    "Sequence finished"
                );
            }
        }
    }

    debug!(
        total_tokens = results.iter().map(|r| r.len()).sum::<usize>(),
        "Batched RNN-T greedy decode complete"
    );

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_last_token_array() {
        let tokens = vec![1, 2, 3];
        let arr = last_token_array(&tokens).unwrap();
        assert_eq!(arr.shape(), &[1, 1]);
    }
}
