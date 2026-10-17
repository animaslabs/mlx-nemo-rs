//! Streaming state for SortFormer diarization model.
//!
//! Manages the speaker cache, FIFO buffer, and silence embedding used for
//! streaming diarization inference. This follows NeMo's synchronous
//! `SortformerModules.streaming_update()` and `_compress_spkcache()`.
//!
//! The speaker cache and FIFO hold *pre-encode* embeddings (the output of the
//! conformer subsampling stage). Every step re-runs the conformer over
//! `[spkcache, fifo, chunk]`, which is what keeps speaker indices stable
//! across chunks.

use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;

use crate::asr::config::SortformerModulesConfig;
use crate::core::ArrayExt;
use crate::core::error::Result;

/// Streaming state for the SortFormer diarizer (batch size 1).
#[derive(Debug)]
pub struct StreamingSortformerState {
    /// Long-term speaker memory [1, spkcache frames, d_model]
    pub spkcache: Array,
    /// Speaker predictions for cached frames [1, spkcache frames, num_spks]
    pub spkcache_preds: Option<Array>,
    /// Recent embeddings FIFO [1, fifo frames, d_model]
    pub fifo: Array,
    /// Speaker predictions for FIFO frames [1, fifo frames, num_spks]
    pub fifo_preds: Array,
    /// Running mean silence embedding [1, d_model]
    pub mean_sil_emb: Array,
    /// Count of silence frames seen
    pub n_sil_frames: usize,
    /// Whether the speaker cache has been compressed at least once
    pub spkcache_compressed: bool,

    params: SortformerModulesConfig,
    d_model: usize,
}

impl StreamingSortformerState {
    /// Create an empty streaming state.
    ///
    /// `d_model` is the pre-encode embedding dimension (the conformer `d_model`).
    pub fn new(params: SortformerModulesConfig, d_model: usize) -> Result<Self> {
        let d = d_model as i32;
        let s = params.num_spks as i32;
        Ok(Self {
            spkcache: ops::zeros::<f32>(&[1, 0, d])?,
            spkcache_preds: None,
            fifo: ops::zeros::<f32>(&[1, 0, d])?,
            fifo_preds: ops::zeros::<f32>(&[1, 0, s])?,
            mean_sil_emb: ops::zeros::<f32>(&[1, d])?,
            n_sil_frames: 0,
            spkcache_compressed: false,
            params,
            d_model,
        })
    }

    /// Number of frames currently in the speaker cache.
    pub fn spkcache_len(&self) -> usize {
        self.spkcache.dim(1) as usize
    }

    /// Number of frames currently in the FIFO.
    pub fn fifo_len(&self) -> usize {
        self.fifo.dim(1) as usize
    }

    /// Concatenate `[spkcache, fifo, chunk]` along time.
    pub fn concat_with_chunk(&self, chunk: &Array) -> Result<Array> {
        ops::concatenate(&[&self.spkcache, &self.fifo, chunk], 1).map_err(Into::into)
    }

    /// Update the state with a new chunk and return the chunk's predictions.
    ///
    /// Port of NeMo's synchronous `streaming_update()`.
    ///
    /// # Arguments
    /// * `chunk` - Pre-encode embeddings of the chunk [1, lc + chunk_len + rc, d_model]
    /// * `preds` - Predictions for `[spkcache, fifo, chunk]` [1, total, num_spks]
    /// * `lc`, `rc` - Left/right context frames in `chunk` excluded from the update
    ///
    /// # Returns
    /// Speaker probabilities for the central chunk frames [1, chunk_len, num_spks]
    pub fn update(&mut self, chunk: &Array, preds: &Array, lc: usize, rc: usize) -> Result<Array> {
        let spkcache_len = self.spkcache_len();
        let fifo_len = self.fifo_len();
        let chunk_len = (chunk.dim(1) as usize).saturating_sub(lc + rc);

        let (sc, fl, l) = (spkcache_len as i32, fifo_len as i32, lc as i32);
        let cl = chunk_len as i32;

        self.fifo_preds = preds.index((.., sc..sc + fl, ..));
        let chunk = chunk.index((.., l..l + cl, ..));
        let chunk_preds = preds.index((.., sc + fl + l..sc + fl + l + cl, ..));

        // Append chunk to FIFO
        self.fifo = ops::concatenate(&[&self.fifo, &chunk], 1)?;
        self.fifo_preds = ops::concatenate(&[&self.fifo_preds, &chunk_preds], 1)?;

        if fifo_len + chunk_len > self.params.fifo_len {
            // Pop the oldest frames out of the FIFO into the speaker cache
            let pop_out_len = self
                .params
                .spkcache_update_period
                .max((chunk_len + fifo_len).saturating_sub(self.params.fifo_len))
                .min(fifo_len + chunk_len) as i32;

            let pop_out_embs = self.fifo.index((.., ..pop_out_len, ..));
            let pop_out_preds = self.fifo_preds.index((.., ..pop_out_len, ..));
            self.update_silence_profile(&pop_out_embs, &pop_out_preds)?;

            self.fifo = self.fifo.index((.., pop_out_len.., ..));
            self.fifo_preds = self.fifo_preds.index((.., pop_out_len.., ..));

            self.spkcache = ops::concatenate(&[&self.spkcache, &pop_out_embs], 1)?;
            let spkcache_preds = match (&self.spkcache_preds, self.spkcache_compressed) {
                (Some(cached), true) => ops::concatenate(&[cached, &pop_out_preds], 1)?,
                // Before the first compression, use fresh predictions for every cache frame
                _ => {
                    let fresh = preds.index((.., ..sc, ..));
                    ops::concatenate(&[&fresh, &pop_out_preds], 1)?
                }
            };
            self.spkcache_preds = Some(spkcache_preds);

            if self.spkcache_len() > self.params.spkcache_len {
                self.compress_spkcache()?;
                self.spkcache_compressed = true;
            }
        }

        Ok(chunk_preds)
    }

    /// Update the running mean silence embedding from popped-out frames.
    ///
    /// A frame is silence if the sum of its speaker probabilities is below
    /// `sil_threshold`.
    fn update_silence_profile(&mut self, embs: &Array, preds: &Array) -> Result<()> {
        let n_frames = preds.dim(1) as usize;
        let n_spk = preds.dim(2) as usize;
        let preds_data: Vec<f32> = preds.to_vec_row_major()?;

        let sil_mask: Vec<f32> = (0..n_frames)
            .map(|t| {
                let sum: f32 = preds_data[t * n_spk..(t + 1) * n_spk].iter().sum();
                if sum < self.params.sil_threshold {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let sil_count = sil_mask.iter().filter(|&&m| m > 0.0).count();
        if sil_count == 0 {
            return Ok(());
        }

        let mask = Array::from_slice(&sil_mask, &[1, n_frames as i32, 1]);
        let sil_sum = ops::multiply(embs, &mask)?.sum_axis(1, false)?; // [1, d_model]
        let old_sum = ops::multiply(
            &self.mean_sil_emb,
            Array::from_f32(self.n_sil_frames as f32),
        )?;
        self.n_sil_frames += sil_count;
        self.mean_sil_emb = ops::divide(
            &ops::add(&old_sum, &sil_sum)?,
            Array::from_f32(self.n_sil_frames as f32),
        )?;
        Ok(())
    }

    /// Compress the speaker cache down to `spkcache_len` frames.
    ///
    /// Port of NeMo's `_compress_spkcache()` (inference path, no speaker
    /// permutation). Frames are scored per speaker, boosted so every speaker
    /// keeps some frames, and the top `spkcache_len` (speaker, frame) pairs are
    /// kept. The result is ordered by speaker, then by time. Unused slots, and
    /// `spkcache_sil_frames_per_spk` slots per speaker, hold the mean silence
    /// embedding with zero predictions.
    fn compress_spkcache(&mut self) -> Result<()> {
        let p = &self.params;
        let preds = self
            .spkcache_preds
            .as_ref()
            .expect("speaker cache predictions are set before compression");
        let n_frames = preds.dim(1) as usize;
        let n_spk = preds.dim(2) as usize;
        let preds_data: Vec<f32> = preds.to_vec_row_major()?;

        let per_spk = (p.spkcache_len / n_spk).saturating_sub(p.spkcache_sil_frames_per_spk);
        let strong_boost = (per_spk as f32 * p.strong_boost_rate).floor() as usize;
        let weak_boost = (per_spk as f32 * p.weak_boost_rate).floor() as usize;
        let min_pos_scores = (per_spk as f32 * p.min_pos_scores_rate).floor() as usize;

        // scores[s][t]
        let mut scores = log_pred_scores(&preds_data, n_frames, n_spk, p.pred_score_threshold);
        disable_low_scores(&preds_data, &mut scores, n_frames, n_spk, min_pos_scores);

        if p.scores_boost_latest > 0.0 {
            for spk_scores in scores.iter_mut() {
                for score in spk_scores.iter_mut().skip(p.spkcache_len) {
                    *score += p.scores_boost_latest;
                }
            }
        }

        // Strong boost so each speaker keeps some frames, weak boost to keep
        // any one speaker from dominating
        let ln_half = 0.5f32.ln();
        for spk_scores in scores.iter_mut() {
            boost_topk(spk_scores, strong_boost, -2.0 * ln_half);
            boost_topk(spk_scores, weak_boost, -ln_half);
        }

        // Append reserved silence slots per speaker
        let sil = p.spkcache_sil_frames_per_spk;
        for spk_scores in scores.iter_mut() {
            spk_scores.extend(std::iter::repeat(f32::INFINITY).take(sil));
        }
        let padded_frames = n_frames + sil;

        // Top spkcache_len over the speaker-major flattened scores
        let mut flat: Vec<(usize, f32)> = scores.iter().flatten().copied().enumerate().collect();
        flat.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut topk: Vec<Option<usize>> = flat
            .iter()
            .take(p.spkcache_len)
            .map(|&(idx, val)| (val != f32::NEG_INFINITY).then_some(idx))
            .collect();
        // Valid indices in original order, disabled placeholders last
        topk.sort_by_key(|idx| idx.unwrap_or(usize::MAX));

        let mut gather_idx = Vec::with_capacity(p.spkcache_len);
        let mut disabled = Vec::with_capacity(p.spkcache_len);
        for idx in topk {
            match idx.map(|i| i % padded_frames) {
                Some(frame) if frame < n_frames => {
                    gather_idx.push(frame as i32);
                    disabled.push(0.0f32);
                }
                _ => {
                    gather_idx.push(0);
                    disabled.push(1.0);
                }
            }
        }

        let k = gather_idx.len() as i32;
        let idx_arr = Array::from_slice(&gather_idx, &[k]);
        let disabled_arr = Array::from_slice(&disabled, &[1, k, 1]);
        let enabled_arr = ops::subtract(Array::from_f32(1.0), &disabled_arr)?;

        let embs = ops::indexing::take_axis(&self.spkcache, &idx_arr, 1)?;
        let sil_emb = self.mean_sil_emb.reshape(&[1, 1, self.d_model as i32])?;
        self.spkcache = ops::add(
            &ops::multiply(&embs, &enabled_arr)?,
            &ops::multiply(&sil_emb, &disabled_arr)?,
        )?;

        let gathered_preds = ops::indexing::take_axis(preds, &idx_arr, 1)?;
        self.spkcache_preds = Some(ops::multiply(&gathered_preds, &enabled_arr)?);

        Ok(())
    }
}

/// Per-frame speaker scores, high for confident non-overlapped speech.
///
/// Returns `scores[speaker][frame]`.
fn log_pred_scores(preds: &[f32], n_frames: usize, n_spk: usize, threshold: f32) -> Vec<Vec<f32>> {
    let ln_half = 0.5f32.ln();
    let mut scores = vec![vec![0.0f32; n_frames]; n_spk];
    for t in 0..n_frames {
        let row = &preds[t * n_spk..(t + 1) * n_spk];
        let log_1_sum: f32 = row.iter().map(|&p| (1.0 - p).max(threshold).ln()).sum();
        for (s, &prob) in row.iter().enumerate() {
            let log_p = prob.max(threshold).ln();
            let log_1_p = (1.0 - prob).max(threshold).ln();
            scores[s][t] = log_p - log_1_p + log_1_sum - ln_half;
        }
    }
    scores
}

/// Disable non-speech frames, and overlapped frames for speakers that have
/// enough clean frames.
fn disable_low_scores(
    preds: &[f32],
    scores: &mut [Vec<f32>],
    n_frames: usize,
    n_spk: usize,
    min_pos_scores: usize,
) {
    for s in 0..n_spk {
        for t in 0..n_frames {
            if preds[t * n_spk + s] <= 0.5 {
                scores[s][t] = f32::NEG_INFINITY;
            }
        }
        let n_pos = scores[s].iter().filter(|&&v| v > 0.0).count();
        if n_pos >= min_pos_scores {
            for t in 0..n_frames {
                let is_speech = preds[t * n_spk + s] > 0.5;
                if is_speech && scores[s][t] <= 0.0 {
                    scores[s][t] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Add `amount` to the `k` highest scores (scores at -inf stay -inf).
fn boost_topk(scores: &mut [f32], k: usize, amount: f32) {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in order.iter().take(k) {
        scores[i] += amount;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boost_topk_keeps_neg_inf() {
        let mut scores = vec![1.0, f32::NEG_INFINITY, 3.0];
        boost_topk(&mut scores, 3, 1.0);
        assert_eq!(scores, vec![2.0, f32::NEG_INFINITY, 4.0]);
    }

    #[test]
    fn test_disable_low_scores_drops_silence() {
        // 2 frames, 2 speakers: frame 0 speaker 0 only, frame 1 silence
        let preds = vec![0.9, 0.1, 0.1, 0.1];
        let mut scores = log_pred_scores(&preds, 2, 2, 0.25);
        disable_low_scores(&preds, &mut scores, 2, 2, 1);
        assert!(scores[0][0] > 0.0);
        assert_eq!(scores[0][1], f32::NEG_INFINITY);
        assert_eq!(scores[1][0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_streaming_update_compresses_to_spkcache_len() -> Result<()> {
        let mut params = SortformerModulesConfig::default();
        params.num_spks = 2;
        params.spkcache_len = 8;
        params.fifo_len = 4;
        params.spkcache_update_period = 4;
        params.spkcache_sil_frames_per_spk = 1;
        let mut state = StreamingSortformerState::new(params, 3)?;
        for step in 0..6 {
            let chunk = ops::ones::<f32>(&[1, 4, 3])?;
            let total = (state.spkcache_len() + state.fifo_len() + 4) as i32;
            // Alternate which speaker is active
            let row = if step % 2 == 0 {
                [0.9f32, 0.1]
            } else {
                [0.1, 0.9]
            };
            let data: Vec<f32> = (0..total).flat_map(|_| row).collect();
            let preds = Array::from_slice(&data, &[1, total, 2]);
            let chunk_preds = state.update(&chunk, &preds, 0, 0)?;
            assert_eq!(chunk_preds.shape(), &[1, 4, 2]);
            assert!(state.spkcache_len() <= 8);
            assert!(state.fifo_len() <= 4);
        }
        assert!(state.spkcache_compressed);
        assert_eq!(state.spkcache_len(), 8);
        Ok(())
    }
}
