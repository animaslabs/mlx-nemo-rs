//! ASR model implementations.
//!
//! This module provides high-level ASR model implementations:
//! - `EncDecCTCModel`: CTC-based encoder-decoder model
//! - `EncDecRNNTModel`: RNN-T (Transducer) encoder-decoder model

mod enc_dec_classification;
mod enc_dec_ctc;
mod enc_dec_rnnt;
mod sortformer_enc_label;

pub use enc_dec_classification::EncDecClassificationModel;
pub use enc_dec_ctc::{CtcStreamingSession, EncDecCTCModel};
pub use enc_dec_rnnt::{EncDecRNNTModel, RnntStreamingSession};
pub use sortformer_enc_label::SortformerEncLabelModel;

use mlx_rs::Array;

use crate::asr::decoding::TimedToken;
use crate::asr::transcript::AudioInput;
use crate::core::error::Result;

/// Decode `inputs` at `sample_rate` in batches of `batch_size` clips, with
/// `decode` mapping one batch of clips to each clip's tokens. Empty clips
/// decode to no tokens without running the model.
fn decode_in_batches<'a, I>(
    inputs: I,
    sample_rate: u32,
    batch_size: usize,
    mut decode: impl FnMut(&[Vec<f32>]) -> Result<Vec<Vec<TimedToken>>>,
) -> Result<Vec<Vec<TimedToken>>>
where
    I: IntoIterator,
    I::Item: Into<AudioInput<'a>>,
{
    let inputs: Vec<AudioInput<'a>> = inputs.into_iter().map(Into::into).collect();
    let mut results = vec![Vec::new(); inputs.len()];
    for (batch_idx, batch) in inputs.chunks(batch_size.max(1)).enumerate() {
        let clips = batch
            .iter()
            .map(|input| input.load(sample_rate))
            .collect::<Result<Vec<_>>>()?;
        let nonempty: Vec<usize> = (0..clips.len()).filter(|&i| !clips[i].is_empty()).collect();
        if nonempty.is_empty() {
            continue;
        }
        let to_decode: Vec<Vec<f32>> = nonempty.iter().map(|&i| clips[i].clone()).collect();
        for (i, tokens) in nonempty.into_iter().zip(decode(&to_decode)?) {
            results[batch_idx * batch_size.max(1) + i] = tokens;
        }
    }
    Ok(results)
}

/// Zero-pad clips into a `[batch, samples]` array, with their lengths `[batch]`.
fn pad_batch(clips: &[Vec<f32>]) -> (Array, Array) {
    let max_len = clips.iter().map(Vec::len).max().unwrap_or(0);
    let mut padded = vec![0.0f32; clips.len() * max_len];
    for (row, clip) in padded.chunks_mut(max_len.max(1)).zip(clips) {
        row[..clip.len()].copy_from_slice(clip);
    }
    let lengths: Vec<i32> = clips.iter().map(|c| c.len() as i32).collect();
    (
        Array::from_slice(&padded, &[clips.len() as i32, max_len as i32]),
        Array::from_slice(&lengths, &[clips.len() as i32]),
    )
}
