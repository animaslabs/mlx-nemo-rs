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
pub use enc_dec_rnnt::{EncDecRNNTModel, RnntDecodingStrategy, RnntStreamingSession};
pub use sortformer_enc_label::SortformerEncLabelModel;

use std::path::Path;

use mlx_rs::Array;

use crate::audio::{load_wav, resample};
use crate::core::error::Result;

/// Load WAV files as mono audio at `sample_rate`.
fn load_clips(paths: &[impl AsRef<Path>], sample_rate: u32) -> Result<Vec<Vec<f32>>> {
    paths
        .iter()
        .map(|path| {
            let (samples, file_sample_rate) = load_wav(path.as_ref())?;
            if file_sample_rate == sample_rate {
                Ok(samples)
            } else {
                resample(&samples, file_sample_rate, sample_rate)
            }
        })
        .collect()
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
