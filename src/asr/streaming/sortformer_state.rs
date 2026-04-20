//! Streaming state for SortFormer diarization model.
//!
//! Manages the speaker cache, FIFO buffer, and silence embedding
//! used for streaming diarization inference.

use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::core::error::Result;

/// Streaming state for the SortFormer diarizer.
pub struct StreamingSortformerState {
    /// Long-term speaker memory [batch, spkcache_len, fc_d_model]
    pub spkcache: Array,
    /// Speaker predictions for cached frames [batch, spkcache_len, num_spks]
    pub spkcache_preds: Array,
    /// Recent embeddings FIFO [batch, fifo_len, fc_d_model]
    pub fifo: Array,
    /// Speaker predictions for FIFO frames [batch, fifo_len, num_spks]
    pub fifo_preds: Array,
    /// Running mean silence embedding [batch, fc_d_model]
    pub mean_sil_emb: Array,
    /// Count of silence frames seen [batch]
    pub n_sil_frames: Array,

    /// Maximum FIFO length
    fifo_len: usize,
    /// Maximum speaker cache length
    spkcache_len: usize,
    /// Number of speakers
    num_spks: usize,
    /// Conformer output dimension
    fc_d_model: usize,
    /// Number of frames currently filled in the FIFO
    fifo_filled: usize,
    /// Number of frames currently filled in the speaker cache
    spkcache_filled: usize,
}

impl StreamingSortformerState {
    /// Create a new streaming state with zero-initialized buffers.
    pub fn new(
        batch_size: usize,
        fc_d_model: usize,
        num_spks: usize,
        spkcache_len: usize,
        fifo_len: usize,
    ) -> Result<Self> {
        let b = batch_size as i32;
        let d = fc_d_model as i32;
        let s = num_spks as i32;
        let sc = spkcache_len as i32;
        let fl = fifo_len as i32;

        Ok(Self {
            spkcache: ops::zeros::<f32>(&[b, sc, d])?,
            spkcache_preds: ops::zeros::<f32>(&[b, sc, s])?,
            fifo: ops::zeros::<f32>(&[b, fl, d])?,
            fifo_preds: ops::zeros::<f32>(&[b, fl, s])?,
            mean_sil_emb: ops::zeros::<f32>(&[b, d])?,
            n_sil_frames: ops::zeros::<f32>(&[b])?,
            fifo_len,
            spkcache_len,
            num_spks,
            fc_d_model,
            fifo_filled: 0,
            spkcache_filled: 0,
        })
    }

    /// Update the streaming state with a new chunk of embeddings and predictions.
    ///
    /// This implements NeMo's `_streaming_update()`:
    /// 1. Append new chunk to FIFO
    /// 2. If FIFO overflows, move oldest frames to speaker cache
    /// 3. If speaker cache overflows, compress using importance scoring
    /// 4. Update the silence embedding
    ///
    /// # Arguments
    /// * `chunk_embs` - New chunk embeddings [batch, chunk_len, fc_d_model]
    /// * `chunk_preds` - New chunk predictions [batch, chunk_len, num_spks]
    ///
    /// # Returns
    /// Ok(()) - state is updated in place
    pub fn update(
        &mut self,
        chunk_embs: &Array,
        chunk_preds: &Array,
    ) -> Result<()> {
        let chunk_len = chunk_embs.dim(1) as usize;

        // Update silence embedding: frames where all speakers are below 0.5
        self.update_silence_embedding(chunk_embs, chunk_preds)?;

        // Append chunk to FIFO
        let new_fifo_len = self.fifo_filled + chunk_len;

        if new_fifo_len <= self.fifo_len {
            // FIFO has room, just append
            // Place chunk_embs at fifo[fifo_filled:fifo_filled+chunk_len]
            let start = self.fifo_filled as i32;
            let end = new_fifo_len as i32;

            // Build new FIFO by concatenating: fifo[:start], chunk, fifo[end:]
            let before = self.fifo.index((.., ..start, ..));
            let after = self.fifo.index((.., end.., ..));
            self.fifo = ops::concatenate_axis(&[&before, chunk_embs, &after], 1)?;

            let before_p = self.fifo_preds.index((.., ..start, ..));
            let after_p = self.fifo_preds.index((.., end.., ..));
            self.fifo_preds = ops::concatenate_axis(&[&before_p, chunk_preds, &after_p], 1)?;

            self.fifo_filled = new_fifo_len;
        } else {
            // FIFO overflows: move oldest frames to speaker cache
            let overflow = new_fifo_len - self.fifo_len;

            // Frames to move to speaker cache: fifo[:overflow]
            let overflow_embs = self.fifo.index((.., ..overflow as i32, ..));
            let overflow_preds = self.fifo_preds.index((.., ..overflow as i32, ..));

            // Update speaker cache with overflow frames
            self.update_spkcache(&overflow_embs, &overflow_preds)?;

            // Shift FIFO left and append new chunk
            let remaining = self.fifo.index((.., overflow as i32.., ..));
            let remaining_p = self.fifo_preds.index((.., overflow as i32.., ..));

            self.fifo = ops::concatenate_axis(&[&remaining, chunk_embs], 1)?;
            self.fifo_preds = ops::concatenate_axis(&[&remaining_p, chunk_preds], 1)?;

            // Trim to fifo_len if needed
            if self.fifo.dim(1) as usize > self.fifo_len {
                let fl = self.fifo_len as i32;
                self.fifo = self.fifo.index((.., ..fl, ..));
                self.fifo_preds = self.fifo_preds.index((.., ..fl, ..));
            }

            self.fifo_filled = self.fifo_len;
        }

        Ok(())
    }

    /// Update speaker cache with new frames.
    fn update_spkcache(&mut self, embs: &Array, preds: &Array) -> Result<()> {
        let new_frames = embs.dim(1) as usize;
        let new_total = self.spkcache_filled + new_frames;

        if new_total <= self.spkcache_len {
            // Cache has room, just append
            let start = self.spkcache_filled as i32;
            let end = new_total as i32;

            let before = self.spkcache.index((.., ..start, ..));
            let after = self.spkcache.index((.., end.., ..));
            self.spkcache = ops::concatenate_axis(&[&before, embs, &after], 1)?;

            let before_p = self.spkcache_preds.index((.., ..start, ..));
            let after_p = self.spkcache_preds.index((.., end.., ..));
            self.spkcache_preds = ops::concatenate_axis(&[&before_p, preds, &after_p], 1)?;

            self.spkcache_filled = new_total;
        } else {
            // Cache overflows: compress by keeping most important frames
            // First, fill remaining slots
            let remaining_slots = self.spkcache_len - self.spkcache_filled;
            if remaining_slots > 0 {
                let start = self.spkcache_filled as i32;
                let fill_embs = embs.index((.., ..remaining_slots as i32, ..));
                let fill_preds = preds.index((.., ..remaining_slots as i32, ..));

                let before = self.spkcache.index((.., ..start, ..));
                self.spkcache = ops::concatenate_axis(&[&before, &fill_embs], 1)?;

                let before_p = self.spkcache_preds.index((.., ..start, ..));
                self.spkcache_preds = ops::concatenate_axis(&[&before_p, &fill_preds], 1)?;

                self.spkcache_filled = self.spkcache_len;
            }

            // Now compress
            self.compress_spkcache()?;

            // Add remaining overflow frames
            if remaining_slots < new_frames {
                let remaining_embs = embs.index((.., remaining_slots as i32.., ..));
                let remaining_preds = preds.index((.., remaining_slots as i32.., ..));
                // Recursive call for remaining frames
                self.update_spkcache(&remaining_embs, &remaining_preds)?;
            }
        }

        Ok(())
    }

    /// Compress the speaker cache by keeping the most important frames per speaker.
    ///
    /// Importance is scored using max speaker probability per frame.
    /// Low-importance frames are replaced with the mean silence embedding.
    fn compress_spkcache(&mut self) -> Result<()> {
        if self.spkcache_filled == 0 {
            return Ok(());
        }

        let filled = self.spkcache_filled as i32;

        // Get filled portion
        let embs = self.spkcache.index((.., ..filled, ..));
        let preds = self.spkcache_preds.index((.., ..filled, ..));

        // Score each frame by its maximum speaker probability
        // Higher max prob = more important (more likely to be a speaker turn)
        preds.eval()?;
        let max_probs = preds.max_axis(-1, false)?; // [batch, filled]
        max_probs.eval()?;

        // Keep top half of frames by importance
        let keep_count = self.spkcache_len / 2;
        let keep_count = keep_count.max(1).min(self.spkcache_filled);

        // For simplicity, sort by importance and keep top-K
        // We operate on batch=0 (single batch for streaming)
        let scores = max_probs.index(0); // [filled]
        scores.eval()?;
        let scores_data: &[f32] = scores.as_slice();

        // Get indices sorted by score (descending)
        let mut indices: Vec<(usize, f32)> = scores_data.iter().copied().enumerate().collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top-K indices, sorted by original position to maintain temporal order
        let mut keep_indices: Vec<usize> = indices[..keep_count].iter().map(|(i, _)| *i).collect();
        keep_indices.sort();

        // Build kept embeddings and predictions
        let keep_idx_arr = Array::from_slice(
            &keep_indices.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            &[keep_count as i32],
        );

        let kept_embs = ops::indexing::take_axis(&embs.index(0), &keep_idx_arr, 0)?;
        let kept_preds = ops::indexing::take_axis(&preds.index(0), &keep_idx_arr, 0)?;

        // Add batch dimension back
        let kept_embs = kept_embs.reshape(&[1, keep_count as i32, self.fc_d_model as i32])?;
        let kept_preds = kept_preds.reshape(&[1, keep_count as i32, self.num_spks as i32])?;

        // Pad with silence embedding to fill remaining slots
        let pad_count = self.spkcache_len - keep_count;
        if pad_count > 0 {
            let sil = self.mean_sil_emb.reshape(&[1, 1, self.fc_d_model as i32])?;
            let sil_pad = ops::broadcast_to(&sil, &[1, pad_count as i32, self.fc_d_model as i32])?;
            let sil_preds = ops::zeros::<f32>(&[1, pad_count as i32, self.num_spks as i32])?;

            self.spkcache = ops::concatenate_axis(&[&kept_embs, &sil_pad], 1)?;
            self.spkcache_preds = ops::concatenate_axis(&[&kept_preds, &sil_preds], 1)?;
        } else {
            self.spkcache = kept_embs;
            self.spkcache_preds = kept_preds;
        }

        self.spkcache_filled = keep_count;

        Ok(())
    }

    /// Update the running mean silence embedding.
    ///
    /// Silence frames are those where all speaker probabilities are below 0.5.
    fn update_silence_embedding(&mut self, embs: &Array, preds: &Array) -> Result<()> {
        // Find silence frames: max(preds, axis=-1) < 0.5
        let max_probs = preds.max_axis(-1, false)?; // [batch, time]
        let threshold = Array::from_f32(0.5);
        let is_silence = ops::lt(&max_probs, &threshold)?; // [batch, time]

        is_silence.eval()?;
        let sil_mask = is_silence.index(0); // [time]
        sil_mask.eval()?;
        let sil_data: &[bool] = sil_mask.as_slice();

        let n_new_sil = sil_data.iter().filter(|&&b| b).count();
        if n_new_sil == 0 {
            return Ok(());
        }

        // Sum silence embeddings
        let is_silence_expanded = is_silence.reshape(&[1, embs.dim(1), 1])?;
        let zero = Array::from_f32(0.0);
        let sil_embs = ops::r#where(&is_silence_expanded, embs, &zero)?;
        let sil_sum = sil_embs.sum_axis(1, false)?; // [batch, fc_d_model]

        // Update running mean: new_mean = (old_mean * old_count + new_sum) / (old_count + new_count)
        let old_count = &self.n_sil_frames;
        let new_count = Array::from_f32(n_new_sil as f32);
        let total_count = ops::add(old_count, &new_count)?;

        let weighted_old = ops::multiply(&self.mean_sil_emb, &old_count.reshape(&[1, 1])?)?;
        let total_sum = ops::add(&weighted_old, &sil_sum)?;
        let total_count_expanded = total_count.reshape(&[1, 1])?;

        // Avoid division by zero
        let eps = Array::from_f32(1e-8);
        let safe_count = ops::add(&total_count_expanded, &eps)?;
        self.mean_sil_emb = ops::divide(&total_sum, &safe_count)?;
        self.n_sil_frames = total_count;

        Ok(())
    }

    /// Get the concatenated context for the conformer encoder.
    ///
    /// Returns [spkcache_filled, fifo_filled] concatenated as [batch, total, fc_d_model].
    /// Only includes filled portions.
    pub fn get_context(&self) -> Result<Array> {
        let sc_filled = self.spkcache_filled as i32;
        let fifo_filled = self.fifo_filled as i32;

        if sc_filled == 0 && fifo_filled == 0 {
            // Return empty tensor
            return ops::zeros::<f32>(&[1, 0, self.fc_d_model as i32]).map_err(Into::into);
        }

        let parts: Vec<Array> = Vec::new();
        let mut refs: Vec<&Array> = Vec::new();
        let sc_part;
        let fifo_part;

        if sc_filled > 0 {
            sc_part = self.spkcache.index((.., ..sc_filled, ..));
            refs.push(&sc_part);
        }
        if fifo_filled > 0 {
            fifo_part = self.fifo.index((.., ..fifo_filled, ..));
            refs.push(&fifo_part);
        }

        if refs.is_empty() {
            return ops::zeros::<f32>(&[1, 0, self.fc_d_model as i32]).map_err(Into::into);
        }

        let _ = parts;
        ops::concatenate_axis(&refs, 1).map_err(Into::into)
    }

    /// Get the total number of context frames (spkcache + fifo filled).
    pub fn context_len(&self) -> usize {
        self.spkcache_filled + self.fifo_filled
    }
}
