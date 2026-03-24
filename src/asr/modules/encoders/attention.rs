//! Relative positional multi-head attention for Conformer.
//!
//! This module implements Transformer-XL style relative positional attention
//! as used in NeMo's Conformer ASR models.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{ops, Array};

use crate::core::error::{Error, Result};
use crate::core::quantization::{LinearLayer, QuantizationConfig, load_linear_layer};

/// Global counter for attention forward calls (for debugging)
static ATTENTION_CALL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset the attention call counter (call at start of streaming session)
pub fn reset_attention_counter() {
    ATTENTION_CALL_COUNT.store(0, Ordering::SeqCst);
}

/// Relative positional encoding generator.
///
/// Generates sinusoidal positional encodings for relative positions.
/// For a sequence of length L, generates encodings for positions
/// from (L-1) down to -(L-1), giving 2L-1 total position embeddings.
pub struct RelPositionalEncoding {
    d_model: usize,
    max_len: usize,
    /// Scale factor: sqrt(d_model) when xscaling is enabled
    xscale: f32,
    /// Precomputed positional encodings: [1, 2*max_len-1, d_model]
    pe: Array,
}

impl RelPositionalEncoding {
    /// Create a new relative positional encoding generator.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `max_len` - Maximum sequence length to support
    /// * `xscaling` - Whether to scale input by sqrt(d_model)
    pub fn new(d_model: usize, max_len: usize, xscaling: bool) -> Result<Self> {
        let xscale = if xscaling {
            (d_model as f32).sqrt()
        } else {
            1.0
        };

        // Generate positional encodings
        let pe = Self::create_pe(d_model, max_len)?;

        Ok(Self {
            d_model,
            max_len,
            xscale,
            pe,
        })
    }

    /// Create sinusoidal positional encodings.
    ///
    /// Generates positions from (max_len-1) down to -(max_len-1).
    ///
    /// NeMo's layout: pe[:, 0::2] = sin, pe[:, 1::2] = cos
    /// This means even indices get sin, odd indices get cos.
    fn create_pe(d_model: usize, max_len: usize) -> Result<Array> {
        let pe_len = 2 * max_len - 1;

        // Create position indices: [max_len-1, max_len-2, ..., 1, 0, -1, ..., -(max_len-1)]
        let positions: Vec<f32> = (-(max_len as i32 - 1)..=(max_len as i32 - 1))
            .rev()
            .map(|p| p as f32)
            .collect();
        let positions = Array::from_slice(&positions, &[pe_len as i32, 1]);

        // Compute div_term: exp(arange(0, d_model, 2) * -(ln(10000) / d_model))
        let div_term: Vec<f32> = (0..d_model)
            .step_by(2)
            .map(|i| (-(i as f32) * (10000.0_f32.ln() / d_model as f32)).exp())
            .collect();
        let div_term = Array::from_slice(&div_term, &[1, (d_model / 2) as i32]);

        // Compute angles: positions * div_term -> [pe_len, d_model/2]
        let angles = ops::matmul(&positions, &div_term)?;

        // Compute sin and cos: each is [pe_len, d_model/2]
        let pe_sin = ops::sin(&angles)?;
        let pe_cos = ops::cos(&angles)?;

        // NeMo interleaving: pe[:, 0::2] = sin, pe[:, 1::2] = cos
        // Stack along last axis then reshape to interleave: [sin0, cos0, sin1, cos1, ...]
        // This gives the same result as NeMo's indexing pattern.
        let pe_stacked = ops::stack_axis(&[pe_sin, pe_cos], -1)?;
        let pe = pe_stacked.reshape(&[pe_len as i32, d_model as i32])?;

        // Add batch dimension: [1, pe_len, d_model]
        let pe = pe.reshape(&[1, pe_len as i32, d_model as i32])?;

        tracing::debug!(
            pe_len,
            d_model,
            max_len,
            pe_shape = ?pe.shape(),
            "created relative positional encodings"
        );

        Ok(pe)
    }

    /// Get positional encoding for a given sequence length.
    ///
    /// Returns positional encodings centered around position 0,
    /// covering positions from (seq_len-1) to -(seq_len-1).
    ///
    /// # Arguments
    /// * `seq_len` - Current sequence length (total context: cache + current input)
    ///
    /// # Returns
    /// Positional encodings: [1, 2*seq_len-1, d_model]
    ///
    /// # NeMo Reference
    /// In NeMo's RelPositionalEncoding.forward():
    /// ```python
    /// input_len = x.size(1) + cache_len
    /// center_pos = self.pe.size(1) // 2 + 1
    /// start_pos = center_pos - input_len
    /// end_pos = center_pos + input_len - 1
    /// pos_emb = self.pe[:, start_pos:end_pos]
    /// ```
    /// This extracts positions centered at 0, giving 2*input_len-1 positions.
    pub fn get_encoding(&self, seq_len: usize) -> Result<Array> {
        if seq_len > self.max_len {
            return Err(Error::model(format!(
                "Sequence length {} exceeds maximum positional encoding length {}",
                seq_len, self.max_len
            )));
        }

        // Calculate slice indices to extract the relevant portion
        // Full PE has positions: [max_len-1, ..., 0, ..., -(max_len-1)]
        // We want positions: [seq_len-1, ..., 0, ..., -(seq_len-1)]
        //
        // NeMo's calculation:
        //   center_pos = pe.size(1) // 2 + 1 = (2*max_len-1) // 2 + 1 = max_len
        //   start_pos = center_pos - input_len = max_len - seq_len
        //   end_pos = center_pos + input_len - 1 = max_len + seq_len - 1
        //   length = end_pos - start_pos = 2*seq_len - 1
        let pe_len = 2 * self.max_len - 1;
        let out_len = 2 * seq_len - 1;
        let start = (pe_len - out_len) / 2;
        let end = start + out_len;

        tracing::trace!(
            seq_len,
            pe_len,
            out_len,
            start,
            end,
            max_len = self.max_len,
            "extracting positional encoding slice"
        );

        // Slice: pe[:, start:end, :]
        let pe_slice = self.pe.index((.., start as i32..end as i32, ..));

        Ok(pe_slice)
    }

    /// Get the scale factor for input scaling.
    pub fn xscale(&self) -> f32 {
        self.xscale
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

/// Relative position multi-head attention.
///
/// Implements Transformer-XL style relative positional attention where:
/// - Content attention: (Q + bias_u) @ K^T
/// - Position attention: (Q + bias_v) @ P^T with rel_shift
///
/// Position biases (pos_bias_u, pos_bias_v) are stored per-layer.
pub struct RelPositionMultiHeadAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,

    /// Query projection (supports quantized weights)
    linear_q: LinearLayer,
    /// Key projection (supports quantized weights)
    linear_k: LinearLayer,
    /// Value projection (supports quantized weights)
    linear_v: LinearLayer,
    /// Output projection (supports quantized weights)
    linear_out: LinearLayer,
    /// Position embedding projection (no bias, supports quantized weights)
    linear_pos: LinearLayer,

    /// Position bias for content attention [n_heads, head_dim]
    pos_bias_u: Array,
    /// Position bias for position attention [n_heads, head_dim]
    pos_bias_v: Array,

    /// Cache drop size for streaming (frames to drop from cache per step)
    pub cache_drop_size: usize,

    /// Attention context size (left, right) for streaming masking
    /// This is stored here for debugging but masking is done at encoder level
    pub att_context_size: Option<(i32, i32)>,
}

impl RelPositionMultiHeadAttention {
    /// Create a new relative position multi-head attention module.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `n_heads` - Number of attention heads
    /// * `use_bias` - Whether to include bias in Q/K/V/Out linear projections
    pub fn new(d_model: usize, n_heads: usize, use_bias: bool) -> Result<Self> {
        let head_dim = d_model / n_heads;

        // Create zero-initialized linear projections (will be loaded from weights)
        let create_linear = |with_bias: bool| -> Result<LinearLayer> {
            let weight = ops::zeros::<f32>(&[d_model as i32, d_model as i32])?;
            let bias = if with_bias {
                Some(ops::zeros::<f32>(&[d_model as i32])?)
            } else {
                None
            };
            Ok(LinearLayer::standard(weight, bias))
        };

        let linear_q = create_linear(use_bias)?;
        let linear_k = create_linear(use_bias)?;
        let linear_v = create_linear(use_bias)?;
        let linear_out = create_linear(use_bias)?;
        let linear_pos = create_linear(false)?; // Position projection always without bias

        // Initialize position biases (will be loaded from weights)
        let pos_bias_u = ops::zeros::<f32>(&[n_heads as i32, head_dim as i32])?;
        let pos_bias_v = ops::zeros::<f32>(&[n_heads as i32, head_dim as i32])?;

        Ok(Self {
            d_model,
            n_heads,
            head_dim,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            cache_drop_size: 0,
            att_context_size: None,
        })
    }

    /// Get head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get number of heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Forward pass with relative positional attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `pos_emb` - Positional embeddings [1, 2*seq_len-1, d_model]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, d_model]
    pub fn forward(&mut self, x: &Array, pos_emb: &Array) -> Result<Array> {
        let debug_attn = std::env::var("DEBUG_ATTENTION").is_ok();

        let batch_size = x.dim(0) as i32;
        let seq_len = x.dim(1) as i32;
        let n_heads = self.n_heads as i32;
        let head_dim = self.head_dim as i32;

        if debug_attn {
            // Debug: print input x at positions 0 and 16
            x.eval()?;
            let x_flat = x.reshape(&[-1])?;
            x_flat.eval()?;
            let x_slice: &[f32] = x_flat.as_slice();
            let d = self.d_model;
            eprintln!("DEBUG ATTN input x shape: {:?}", x.shape());
            eprintln!("DEBUG ATTN input x[0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                x_slice[0], x_slice[1], x_slice[2], x_slice[3], x_slice[4]);
            let x_pos_16 = 16 * d;
            if x_slice.len() > x_pos_16 + 4 {
                eprintln!("DEBUG ATTN input x[0,16,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                    x_slice[x_pos_16], x_slice[x_pos_16+1], x_slice[x_pos_16+2], x_slice[x_pos_16+3], x_slice[x_pos_16+4]);
            }
        }

        // Project Q, K, V: [batch, seq_len, d_model]
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;

        if debug_attn {
            // Debug: print Q/K/V projections before reshape
            q.eval()?;
            k.eval()?;
            v.eval()?;
            let q_flat = q.reshape(&[-1])?;
            let k_flat = k.reshape(&[-1])?;
            let v_flat = v.reshape(&[-1])?;
            q_flat.eval()?;
            k_flat.eval()?;
            v_flat.eval()?;
            let q_slice: &[f32] = q_flat.as_slice();
            let k_slice: &[f32] = k_flat.as_slice();
            let v_slice: &[f32] = v_flat.as_slice();
            let d = self.d_model;
            eprintln!("DEBUG ATTN Q[0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                q_slice[0], q_slice[1], q_slice[2], q_slice[3], q_slice[4]);
            eprintln!("DEBUG ATTN Q[0,1,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                q_slice[d], q_slice[d+1], q_slice[d+2], q_slice[d+3], q_slice[d+4]);
            eprintln!("DEBUG ATTN K[0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                k_slice[0], k_slice[1], k_slice[2], k_slice[3], k_slice[4]);
            // Also print K at position 16 (last valid position)
            let k_pos_16 = 16 * d;
            eprintln!("DEBUG ATTN K[0,16,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                k_slice[k_pos_16], k_slice[k_pos_16+1], k_slice[k_pos_16+2], k_slice[k_pos_16+3], k_slice[k_pos_16+4]);
            eprintln!("DEBUG ATTN V[0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                v_slice[0], v_slice[1], v_slice[2], v_slice[3], v_slice[4]);

            // Debug: print position biases
            self.pos_bias_u.eval()?;
            self.pos_bias_v.eval()?;
            let bias_u_flat = self.pos_bias_u.reshape(&[-1])?;
            let bias_v_flat = self.pos_bias_v.reshape(&[-1])?;
            bias_u_flat.eval()?;
            bias_v_flat.eval()?;
            let bu_slice: &[f32] = bias_u_flat.as_slice();
            let bv_slice: &[f32] = bias_v_flat.as_slice();
            eprintln!("DEBUG ATTN pos_bias_u[0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bu_slice[0], bu_slice[1], bu_slice[2], bu_slice[3], bu_slice[4]);
            eprintln!("DEBUG ATTN pos_bias_v[0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bv_slice[0], bv_slice[1], bv_slice[2], bv_slice[3], bv_slice[4]);
        }

        // Reshape to multi-head: [batch, seq_len, n_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, n_heads, head_dim])?;
        let k = k.reshape(&[batch_size, seq_len, n_heads, head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, n_heads, head_dim])?;

        // Transpose to [batch, n_heads, seq_len, head_dim]
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        // Project positional embeddings: [1, 2*seq_len-1, d_model]
        let p = self.linear_pos.forward(pos_emb)?;
        let pos_len = pos_emb.dim(1) as i32; // 2*seq_len - 1

        // Reshape p to multi-head: [1, 2*seq_len-1, n_heads, head_dim]
        // Then transpose to [1, n_heads, 2*seq_len-1, head_dim]
        let p = p.reshape(&[1, pos_len, n_heads, head_dim])?;
        let p = ops::transpose_axes(&p, &[0, 2, 1, 3])?;

        // Add position biases to query
        // pos_bias_u/v: [n_heads, head_dim] -> [1, n_heads, 1, head_dim]
        let bias_u = self.pos_bias_u.reshape(&[1, n_heads, 1, head_dim])?;
        let bias_v = self.pos_bias_v.reshape(&[1, n_heads, 1, head_dim])?;

        // q_with_bias_u: [batch, n_heads, seq_len, head_dim]
        let q_with_bias_u = &q + &bias_u;
        let q_with_bias_v = &q + &bias_v;

        // Content attention: (Q + u) @ K^T -> [batch, n_heads, seq_len, seq_len]
        let k_t = ops::transpose_axes(&k, &[0, 1, 3, 2])?;
        let matrix_ac = ops::matmul(&q_with_bias_u, &k_t)?;

        // Position attention: (Q + v) @ P^T -> [batch, n_heads, seq_len, 2*seq_len-1]
        let p_t = ops::transpose_axes(&p, &[0, 1, 3, 2])?;
        let matrix_bd = ops::matmul(&q_with_bias_v, &p_t)?;

        // Apply rel_shift to align relative positions
        if debug_attn {
            // Debug: print matrix_bd shape and values BEFORE rel_shift
            matrix_bd.eval()?;
            eprintln!("DEBUG ATTN matrix_bd BEFORE rel_shift shape: {:?}", matrix_bd.shape());
            let bd_pre_flat = matrix_bd.reshape(&[-1])?;
            bd_pre_flat.eval()?;
            let bd_pre: &[f32] = bd_pre_flat.as_slice();
            let pos_len = matrix_bd.dim(3) as usize;
            // Row 0 values at positions that will become [0, 1, 2, ..., 16] after rel_shift
            eprintln!("DEBUG ATTN matrix_bd[0,0,0,:5] BEFORE shift: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bd_pre[0], bd_pre[1], bd_pre[2], bd_pre[3], bd_pre[4]);
            eprintln!("DEBUG ATTN matrix_bd[0,0,0,-5:] BEFORE shift: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bd_pre[pos_len-5], bd_pre[pos_len-4], bd_pre[pos_len-3], bd_pre[pos_len-2], bd_pre[pos_len-1]);
        }

        let matrix_bd = Self::rel_shift(&matrix_bd)?;

        if debug_attn {
            // Debug: print matrix_bd shape and values AFTER rel_shift but BEFORE truncation
            matrix_bd.eval()?;
            eprintln!("DEBUG ATTN matrix_bd AFTER rel_shift shape: {:?}", matrix_bd.shape());
            let bd_post_flat = matrix_bd.reshape(&[-1])?;
            bd_post_flat.eval()?;
            let bd_post: &[f32] = bd_post_flat.as_slice();
            let pos_len = matrix_bd.dim(3) as usize;
            eprintln!("DEBUG ATTN matrix_bd[0,0,0,:5] AFTER shift: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bd_post[0], bd_post[1], bd_post[2], bd_post[3], bd_post[4]);
            eprintln!("DEBUG ATTN matrix_bd[0,0,0, 12:17]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bd_post[12], bd_post[13], bd_post[14], bd_post[15], bd_post[16]);
        }

        // Truncate to seq_len (take first seq_len columns)
        // matrix_bd: [batch, n_heads, seq_len, 2*seq_len-1] -> [batch, n_heads, seq_len, seq_len]
        let matrix_bd = matrix_bd.index((.., .., .., ..seq_len));

        if debug_attn {
            // Debug: print matrix_ac and matrix_bd (content and position attention)
            matrix_ac.eval()?;
            matrix_bd.eval()?;
            let ac_flat = matrix_ac.reshape(&[-1])?;
            let bd_flat = matrix_bd.reshape(&[-1])?;
            ac_flat.eval()?;
            bd_flat.eval()?;
            let ac_slice: &[f32] = ac_flat.as_slice();
            let bd_slice: &[f32] = bd_flat.as_slice();
            // Head 0, query 0, first 5 keys
            eprintln!("DEBUG ATTN matrix_ac[0,0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                ac_slice[0], ac_slice[1], ac_slice[2], ac_slice[3], ac_slice[4]);
            eprintln!("DEBUG ATTN matrix_bd[0,0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                bd_slice[0], bd_slice[1], bd_slice[2], bd_slice[3], bd_slice[4]);
            eprintln!("DEBUG ATTN matrix_ac shape: {:?}", matrix_ac.shape());
            eprintln!("DEBUG ATTN matrix_bd shape: {:?}", matrix_bd.shape());
            // Also print last 5 positions
            eprintln!("DEBUG ATTN matrix_ac[0,0,0,12:17]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                ac_slice[12], ac_slice[13], ac_slice[14], ac_slice[15], ac_slice[16]);
        }

        // Combine and scale
        let scale = (head_dim as f32).sqrt();
        let scores = (&matrix_ac + &matrix_bd) / Array::from_f32(scale);

        if debug_attn {
            // Debug: print scores before softmax
            scores.eval()?;
            let scores_flat = scores.reshape(&[-1])?;
            scores_flat.eval()?;
            let s_slice: &[f32] = scores_flat.as_slice();
            eprintln!("DEBUG ATTN scores shape: {:?}", scores.shape());
            eprintln!("DEBUG ATTN scores[0,0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                s_slice[0], s_slice[1], s_slice[2], s_slice[3], s_slice[4]);
            // Also show last 5 scores for row 0
            let row_size = seq_len as usize;
            if row_size > 5 {
                eprintln!("DEBUG ATTN scores[0,0,0,-5:]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                    s_slice[row_size-5], s_slice[row_size-4], s_slice[row_size-3], s_slice[row_size-2], s_slice[row_size-1]);
            }
        }

        // Softmax
        let attn_weights = ops::softmax_axis(&scores, -1, None)?;

        if debug_attn {
            // Debug: print attention weights
            attn_weights.eval()?;
            let aw_flat = attn_weights.reshape(&[-1])?;
            aw_flat.eval()?;
            let aw_slice: &[f32] = aw_flat.as_slice();
            eprintln!("DEBUG ATTN attn_weights[0,0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                aw_slice[0], aw_slice[1], aw_slice[2], aw_slice[3], aw_slice[4]);
        }

        // Apply attention to values: [batch, n_heads, seq_len, head_dim]
        let context = ops::matmul(&attn_weights, &v)?;

        // Transpose back: [batch, seq_len, n_heads, head_dim]
        let context = ops::transpose_axes(&context, &[0, 2, 1, 3])?;

        // Reshape: [batch, seq_len, d_model]
        let context = context.reshape(&[batch_size, seq_len, self.d_model as i32])?;

        // Output projection
        let output = self.linear_out.forward(&context)?;

        if debug_attn {
            // Debug: print final attention output
            output.eval()?;
            let out_flat = output.reshape(&[-1])?;
            out_flat.eval()?;
            let out_slice: &[f32] = out_flat.as_slice();
            let d = self.d_model;
            eprintln!("DEBUG ATTN output[0,0,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                out_slice[0], out_slice[1], out_slice[2], out_slice[3], out_slice[4]);
            eprintln!("DEBUG ATTN output[0,1,:5]: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                out_slice[d], out_slice[d+1], out_slice[d+2], out_slice[d+3], out_slice[d+4]);
        }

        Ok(output)
    }

    /// Forward pass with cache support for streaming inference.
    ///
    /// This implements NeMo's cache-aware streaming attention (arXiv:2312.17279):
    ///
    /// 1. **Cache Content**: The cache stores `h_k` values - the **input to self-attention**
    ///    (i.e., the layer-normed hidden states from previous chunks). This is NOT the
    ///    projected K/V values, and NOT the attention output.
    ///
    /// 2. **K/V Computation**: K and V are computed from `concat(cache, current_input)`.
    ///    This means cached states get re-projected through linear_k/linear_v each chunk,
    ///    which is intentional: it allows relative positional encoding to work correctly
    ///    across the full context window.
    ///
    /// 3. **Cache Update**: Uses sliding window - old frames are dropped and new input
    ///    frames are appended. The cache size is bounded by `left_context`.
    ///
    /// 4. **Attention Mask**: Unfilled cache positions (during warmup) are masked out
    ///    to prevent attending to zero-padding.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model] (hidden states h_k to attend over)
    /// * `pos_emb` - Positional embeddings [1, 2*total_len-1, d_model]
    /// * `cache` - Optional full cache buffer [batch, cache_size, d_model] containing
    ///             previous chunks' attention inputs (h values)
    /// * `cache_filled_len` - How many frames of cache are valid (0 if no cache)
    /// * `att_mask` - Optional attention mask [batch, seq_len, total_len] where True=masked
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated cache tensor [batch, cache_size, d_model] for next chunk
    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        pos_emb: &Array,
        cache: Option<&Array>,
        _cache_filled_len: i32,
        att_mask: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        let batch_size = x.dim(0) as i32;
        let seq_len = x.dim(1) as i32;
        let d_model = x.dim(2) as i32;
        let n_heads = self.n_heads as i32;
        let head_dim = self.head_dim as i32;

        let call_count = ATTENTION_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        let layer_idx = call_count % 24;
        let encoder_call = call_count / 24;

        // Log dimensions for first few calls to help debug
        if layer_idx == 0 && encoder_call < 5 {
            tracing::debug!(
                encoder_call,
                batch_size,
                seq_len,
                d_model,
                pos_emb_shape = ?pos_emb.shape(),
                cache_present = cache.is_some(),
                cache_drop_size = self.cache_drop_size,
                "attention forward_with_cache dimensions"
            );
        }

        // NeMo cache-aware streaming (arXiv:2312.17279 Section 3.3):
        //
        // The cache stores h_k values (input to self-attention layer). For K/V computation:
        //   key = value = concat(cache, current_input)
        //
        // This re-projects cached h values through linear_k/linear_v each chunk.
        // This is intentional and correct: relative positional encoding needs to be
        // computed over the full context (cache + current), so we can't cache K/V directly.
        //
        // We use the FULL cache buffer (including unfilled/zero positions) and rely
        // on the attention mask to hide unfilled positions during warmup.

        let (kv_input, total_len, new_cache) = if let Some(cache_tensor) = cache {
            let cache_size = cache_tensor.dim(1) as i32;

            // NeMo's update_cache: key = value = concat(cache, input)
            // Use FULL cache buffer, not just filled portion
            let kv_input = ops::concatenate_axis(&[cache_tensor, x], 1)?;
            let total = cache_size + seq_len;

            // Update cache: sliding window
            // q_keep_size = seq_len - cache_drop_size (frames to keep from input)
            // The cache stores values at the END of the buffer
            // new_cache = concat(old_cache[q_keep_size:], x[:q_keep_size])
            let q_keep_size = (seq_len - self.cache_drop_size as i32).max(0);

            // For the new cache, we slide the window:
            // - Drop first q_keep_size frames from the old cache
            // - Append first q_keep_size frames from current input
            let new_cache = if q_keep_size > 0 {
                let cache_keep = cache_tensor.index((.., q_keep_size.., ..));
                let x_keep = x.index((.., ..q_keep_size, ..));
                ops::concatenate_axis(&[&cache_keep, &x_keep], 1)?
            } else {
                cache_tensor.clone()
            };

            (kv_input, total, Some(new_cache))
        } else {
            // No cache at all - non-streaming or first chunk
            let q_keep_size = (seq_len - self.cache_drop_size as i32).max(0);
            let new_cache = if q_keep_size > 0 {
                x.index((.., ..q_keep_size, ..))
            } else {
                ops::zeros::<f32>(&[batch_size, 0, d_model])?
            };
            (x.clone(), seq_len, Some(new_cache))
        };

        // Verify positional encoding dimensions match expectations
        // pos_emb should be [1, 2*total_len-1, d_model]
        let expected_pos_len = 2 * total_len - 1;
        let actual_pos_len = pos_emb.dim(1) as i32;
        if layer_idx == 0 && actual_pos_len != expected_pos_len {
            tracing::warn!(
                encoder_call,
                total_len,
                expected_pos_len,
                actual_pos_len,
                seq_len,
                "positional encoding length mismatch!"
            );
        }

        // Project Q from input only (NeMo: query = input, unchanged)
        let q = self.linear_q.forward(x)?;

        // Project K, V from concatenated cache+input (NeMo style)
        let k = self.linear_k.forward(&kv_input)?;
        let v = self.linear_v.forward(&kv_input)?;

        // Reshape Q to multi-head: [batch, seq_len, n_heads, head_dim]
        let q = q.reshape(&[batch_size, seq_len, n_heads, head_dim])?;
        // Reshape K, V to multi-head: [batch, total_len, n_heads, head_dim]
        let k = k.reshape(&[batch_size, total_len, n_heads, head_dim])?;
        let v = v.reshape(&[batch_size, total_len, n_heads, head_dim])?;

        // Transpose to [batch, n_heads, len, head_dim]
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        // Project positional embeddings
        let p = self.linear_pos.forward(pos_emb)?;
        let pos_len = pos_emb.dim(1) as i32;
        let p = p.reshape(&[1, pos_len, n_heads, head_dim])?;
        let p = ops::transpose_axes(&p, &[0, 2, 1, 3])?;

        // Add position biases to query
        let bias_u = self.pos_bias_u.reshape(&[1, n_heads, 1, head_dim])?;
        let bias_v = self.pos_bias_v.reshape(&[1, n_heads, 1, head_dim])?;
        let q_with_bias_u = &q + &bias_u;
        let q_with_bias_v = &q + &bias_v;

        // Content attention: (Q + u) @ K^T -> [batch, n_heads, seq_len, total_len]
        let k_t = ops::transpose_axes(&k, &[0, 1, 3, 2])?;
        let matrix_ac = ops::matmul(&q_with_bias_u, &k_t)?;

        // Position attention: (Q + v) @ P^T
        let p_t = ops::transpose_axes(&p, &[0, 1, 3, 2])?;
        let matrix_bd = ops::matmul(&q_with_bias_v, &p_t)?;

        // Apply rel_shift and truncate to total_len
        let matrix_bd = Self::rel_shift(&matrix_bd)?;
        let matrix_bd = matrix_bd.index((.., .., .., ..total_len));

        // Debug: check matrix_bd stats after rel_shift (only at specific points)
        let encoder_call = call_count / 24;
        if call_count % 24 == 0 && (encoder_call < 5 || encoder_call == 35 || encoder_call == 135) {
            matrix_bd.eval()?;
            let bd_flat = matrix_bd.reshape(&[-1])?;
            bd_flat.eval()?;
            let bd_slice: &[f32] = bd_flat.as_slice();
            let bd_min = bd_slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let bd_max = bd_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let bd_mean: f32 = bd_slice.iter().sum::<f32>() / bd_slice.len() as f32;
            tracing::trace!(
                encoder_call,
                seq_len,
                total_len,
                bd_min,
                bd_max,
                bd_mean,
                "rel_shift matrix_bd stats"
            );
        }

        // Combine and scale
        let scale = (head_dim as f32).sqrt();
        let mut scores = (&matrix_ac + &matrix_bd) / Array::from_f32(scale);

        // Apply attention mask if provided
        let mask_expanded = if let Some(mask) = att_mask {
            // mask: [batch, seq_len, total_len] -> [batch, 1, seq_len, total_len]
            let mask = mask.reshape(&[batch_size, 1, seq_len, total_len])?;
            // NeMo convention: mask=True means VALID (can attend), False means MASKED
            // where(mask, scores, -inf) keeps scores where mask=True, sets -inf where mask=False
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            scores = ops::r#where(&mask, &scores, &neg_inf)?;
            Some(mask)
        } else {
            None
        };

        // Softmax
        let mut attn_weights = ops::softmax_axis(&scores, -1, None)?;

        // Zero out attention weights for masked positions (Python: .masked_fill(mask, 0.0))
        // This handles the case where all positions are masked (softmax produces NaN)
        // and ensures masked positions don't contribute to the output
        if let Some(mask) = mask_expanded {
            let zero = Array::from_f32(0.0);
            attn_weights = ops::r#where(&mask, &attn_weights, &zero)?;
        }

        // Debug: print attention stats periodically (only layer 0, every ~100 encoder calls)
        // Each encoder call has 24 layers, so call_count/24 gives encoder call number
        if call_count % 24 == 0
            && (encoder_call < 10
                || encoder_call % 50 == 0
                || (encoder_call >= 130 && encoder_call <= 145))
        {
            attn_weights.eval()?;
            let aw_flat = attn_weights.reshape(&[-1])?;
            aw_flat.eval()?;
            let aw_slice: &[f32] = aw_flat.as_slice();
            let aw_min = aw_slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let aw_max = aw_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let aw_mean: f32 = aw_slice.iter().sum::<f32>() / aw_slice.len() as f32;

            // Check attention entropy (low entropy = attending to few positions)
            let entropy: f32 = aw_slice
                .iter()
                .filter(|&&x| x > 1e-10)
                .map(|&x| -x * x.ln())
                .sum::<f32>()
                / (n_heads * seq_len) as f32;

            // Print first row of attention weights for head 0 (where is query 0 attending?)
            let first_row = attn_weights.index((0, 0, 0, ..));
            first_row.eval()?;
            let row_slice: &[f32] = first_row.as_slice();
            let top_5: Vec<(usize, f32)> = {
                let mut indexed: Vec<_> =
                    row_slice.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.into_iter().take(5).collect()
            };
            tracing::trace!(
                encoder_call,
                layer = 0,
                seq_len,
                total_len,
                pos_emb_len = pos_emb.dim(1),
                aw_min,
                aw_max,
                aw_mean,
                entropy,
                ?top_5,
                "attention weights stats"
            );
        }

        // Apply attention to values: [batch, n_heads, seq_len, head_dim]
        let context = ops::matmul(&attn_weights, &v)?;

        // Transpose back: [batch, seq_len, n_heads, head_dim]
        let context = ops::transpose_axes(&context, &[0, 2, 1, 3])?;

        // Reshape: [batch, seq_len, d_model]
        let context = context.reshape(&[batch_size, seq_len, self.d_model as i32])?;

        // Output projection
        let output = self.linear_out.forward(&context)?;

        Ok((output, new_cache))
    }

    /// Relative shift operation to align position attention scores.
    ///
    /// Transforms position attention from shape where each row has positions
    /// [L-1, L-2, ..., 0, ..., -(L-1)] to a skewed matrix where row i has
    /// the correct relative positions to keys 0..L.
    ///
    /// This follows NeMo's implementation which handles variable position lengths
    /// (important for streaming where pos_len = 2*total_len-1 != 2*seq_len-1).
    ///
    /// # Arguments
    /// * `x` - Position attention scores [batch, n_heads, seq_len, pos_len]
    ///
    /// # Returns
    /// Shifted scores [batch, n_heads, seq_len, pos_len]
    fn rel_shift(x: &Array) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let n_heads = shape[1] as i32;
        let seq_len = shape[2] as i32;
        let pos_len = shape[3] as i32;

        // Step 1: Pad with zeros on the left of last dimension
        // [batch, n_heads, seq_len, pos_len] -> [batch, n_heads, seq_len, pos_len+1]
        let zeros = ops::zeros::<f32>(&[batch, n_heads, seq_len, 1])?;
        let x_padded = ops::concatenate_axis(&[&zeros, x], -1)?;

        // Step 2: Reshape to [batch, n_heads, pos_len+1, seq_len]
        // This is a transposition-via-reshape that shifts positions
        let x_reshaped = x_padded.reshape(&[batch, n_heads, pos_len + 1, seq_len])?;

        // Step 3: Slice from index 1 in dimension 2 (skip first row)
        // [batch, n_heads, pos_len+1, seq_len] -> [batch, n_heads, pos_len, seq_len]
        let x_sliced = x_reshaped.index((.., .., 1.., ..));

        // Step 4: Reshape back to [batch, n_heads, seq_len, pos_len]
        let x_shifted = x_sliced.reshape(&[batch, n_heads, seq_len, pos_len])?;

        Ok(x_shifted)
    }

    /// Load weights from a state dict.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `prefix` - Key prefix for this layer
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // Load linear projections (supports both standard and quantized)
        if let Some(layer) = load_linear_layer(weights, &format!("{}.linear_q", prefix), quant) {
            self.linear_q = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.linear_k", prefix), quant) {
            self.linear_k = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.linear_v", prefix), quant) {
            self.linear_v = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.linear_out", prefix), quant) {
            self.linear_out = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.linear_pos", prefix), quant) {
            self.linear_pos = layer;
        }

        // Load position biases (per-layer, not shared, not quantized)
        if let Some(bias_u) = weights.get(&format!("{}.pos_bias_u", prefix)) {
            self.pos_bias_u = bias_u.clone();
        }
        if let Some(bias_v) = weights.get(&format!("{}.pos_bias_v", prefix)) {
            self.pos_bias_v = bias_v.clone();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rel_positional_encoding_creation() {
        let pe = RelPositionalEncoding::new(512, 100, true).unwrap();
        assert!((pe.xscale() - 512.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_rel_positional_encoding_shape() {
        let pe = RelPositionalEncoding::new(512, 100, false).unwrap();
        let encoding = pe.get_encoding(50).unwrap();
        let shape = encoding.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 99); // 2*50 - 1
        assert_eq!(shape[2], 512);
    }

    #[test]
    fn test_rel_shift() {
        // Create a simple test tensor
        let x = Array::from_slice(&[1.0_f32; 28], &[1, 1, 4, 7]); // seq_len=4, pos_len=7
        let shifted = RelPositionMultiHeadAttention::rel_shift(&x).unwrap();
        let shape = shifted.shape();
        assert_eq!(shape, &[1, 1, 4, 7]);
    }

    #[test]
    fn test_rel_shift_correctness() {
        // Test with known values to verify rel_shift behavior
        // For seq_len=3, pos_len=5 (2*3-1)
        // Input matrix_bd has shape [1, 1, 3, 5] where each row i
        // has scores for relative positions [2, 1, 0, -1, -2]
        // After rel_shift, row i should have the correct alignment

        // Row 0 (query at position 0): needs rel_pos [0, -1, -2] for keys [0, 1, 2]
        // Row 1 (query at position 1): needs rel_pos [1, 0, -1] for keys [0, 1, 2]
        // Row 2 (query at position 2): needs rel_pos [2, 1, 0] for keys [0, 1, 2]

        // Input: each row has [pos_2, pos_1, pos_0, pos_-1, pos_-2]
        // Let's use unique values: row 0 = [0,1,2,3,4], row 1 = [5,6,7,8,9], row 2 = [10,11,12,13,14]
        let input_data: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, // row 0: positions [2, 1, 0, -1, -2]
            5.0, 6.0, 7.0, 8.0, 9.0, // row 1: positions [2, 1, 0, -1, -2]
            10.0, 11.0, 12.0, 13.0, 14.0, // row 2: positions [2, 1, 0, -1, -2]
        ];
        let x = Array::from_slice(&input_data, &[1, 1, 3, 5]);

        let shifted = RelPositionMultiHeadAttention::rel_shift(&x).unwrap();

        // Evaluate and get values
        mlx_rs::transforms::eval([&shifted]).unwrap();
        let shifted_data: Vec<f32> = shifted.as_slice().to_vec();

        // After shift and taking first 3 columns:
        // Row 0 should have rel_pos [0, -1, -2] = values [2, 3, 4]
        // Row 1 should have rel_pos [1, 0, -1] = values [6, 7, 8]
        // Row 2 should have rel_pos [2, 1, 0] = values [10, 11, 12]

        // The full shifted tensor has shape [1, 1, 3, 5]
        // Extract and verify first 3 columns of each row
        assert_eq!(
            shifted_data[0], 2.0,
            "Row 0, col 0 should be pos_0 (value 2)"
        );
        assert_eq!(
            shifted_data[1], 3.0,
            "Row 0, col 1 should be pos_-1 (value 3)"
        );
        assert_eq!(
            shifted_data[2], 4.0,
            "Row 0, col 2 should be pos_-2 (value 4)"
        );

        assert_eq!(
            shifted_data[5], 6.0,
            "Row 1, col 0 should be pos_1 (value 6)"
        );
        assert_eq!(
            shifted_data[6], 7.0,
            "Row 1, col 1 should be pos_0 (value 7)"
        );
        assert_eq!(
            shifted_data[7], 8.0,
            "Row 1, col 2 should be pos_-1 (value 8)"
        );

        assert_eq!(
            shifted_data[10], 10.0,
            "Row 2, col 0 should be pos_2 (value 10)"
        );
        assert_eq!(
            shifted_data[11], 11.0,
            "Row 2, col 1 should be pos_1 (value 11)"
        );
        assert_eq!(
            shifted_data[12], 12.0,
            "Row 2, col 2 should be pos_0 (value 12)"
        );
    }

    #[test]
    fn test_rel_positional_encoding_values() {
        // Test that the positional encoding produces correct sin/cos interleaving
        // NeMo uses: pe[:, 0::2] = sin, pe[:, 1::2] = cos
        let d_model = 4;
        let max_len = 3;
        let pe = RelPositionalEncoding::new(d_model, max_len, false).unwrap();

        // Get encoding for seq_len=3 -> output shape [1, 5, 4] (2*3-1 = 5 positions)
        let encoding = pe.get_encoding(3).unwrap();
        encoding.eval().unwrap();

        let shape = encoding.shape();
        assert_eq!(shape, &[1, 5, 4]);

        // The positions are: [2, 1, 0, -1, -2] (from pos 2 down to pos -2)
        let data: &[f32] = encoding.as_slice();

        // For d_model=4, we have div_term for i=0 and i=2:
        // div_term[0] = exp(0) = 1.0
        // div_term[1] = exp(-2 * ln(10000) / 4) = exp(-2.3026) ≈ 0.1
        let div_term_0 = 1.0_f32;
        let div_term_1 = (-2.0_f32 * 10000.0_f32.ln() / 4.0).exp();

        // Position 2 (index 0 in output): angles = [2*1.0, 2*div_term_1]
        // pe[0, 0] = sin(2*1.0) = sin(2)
        // pe[0, 1] = cos(2*1.0) = cos(2)
        // pe[0, 2] = sin(2*div_term_1)
        // pe[0, 3] = cos(2*div_term_1)
        let pos = 2.0_f32;
        let expected_sin0 = (pos * div_term_0).sin();
        let expected_cos0 = (pos * div_term_0).cos();
        let expected_sin1 = (pos * div_term_1).sin();
        let expected_cos1 = (pos * div_term_1).cos();

        // data layout: [row0, row1, row2, row3, row4] where each row has 4 elements
        // row 0 = position 2
        assert!(
            (data[0] - expected_sin0).abs() < 1e-5,
            "pe[0,0] should be sin(2*1.0), got {} expected {}",
            data[0],
            expected_sin0
        );
        assert!(
            (data[1] - expected_cos0).abs() < 1e-5,
            "pe[0,1] should be cos(2*1.0), got {} expected {}",
            data[1],
            expected_cos0
        );
        assert!(
            (data[2] - expected_sin1).abs() < 1e-5,
            "pe[0,2] should be sin(2*div_term_1), got {} expected {}",
            data[2],
            expected_sin1
        );
        assert!(
            (data[3] - expected_cos1).abs() < 1e-5,
            "pe[0,3] should be cos(2*div_term_1), got {} expected {}",
            data[3],
            expected_cos1
        );

        // Position 0 (index 2 in output, the center position)
        let pos_0_row = 2 * 4; // row index 2, 4 elements per row
        let expected_sin0_at_0 = 0.0_f32.sin(); // sin(0) = 0
        let expected_cos0_at_0 = 0.0_f32.cos(); // cos(0) = 1

        assert!(
            (data[pos_0_row] - expected_sin0_at_0).abs() < 1e-5,
            "pe at position 0 should have sin(0)=0, got {}",
            data[pos_0_row]
        );
        assert!(
            (data[pos_0_row + 1] - expected_cos0_at_0).abs() < 1e-5,
            "pe at position 0 should have cos(0)=1, got {}",
            data[pos_0_row + 1]
        );
    }

    #[test]
    fn test_rel_positional_encoding_streaming_dimensions() {
        // Test that streaming dimensions work correctly
        // Simulates: cache_size=70, seq_len=14, total_len=84
        let d_model = 512;
        let max_len = 5000;
        let pe = RelPositionalEncoding::new(d_model, max_len, false).unwrap();

        // Streaming scenario
        let cache_size = 70;
        let seq_len = 14;
        let total_len = cache_size + seq_len;

        let encoding = pe.get_encoding(total_len).unwrap();
        let shape = encoding.shape();

        // Should be [1, 2*84-1, 512] = [1, 167, 512]
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 2 * total_len as i32 - 1);
        assert_eq!(shape[2], d_model as i32);
    }
}
