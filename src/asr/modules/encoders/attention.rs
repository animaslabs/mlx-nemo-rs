//! Relative positional multi-head attention for Conformer.
//!
//! This module implements Transformer-XL style relative positional attention
//! as used in NeMo's Conformer ASR models.

use std::collections::HashMap;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{builder::Builder, module::Module, nn, ops, Array};

use crate::core::error::{Error, Result};

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

        // Compute sin and cos
        let pe_sin = ops::sin(&angles)?;
        let pe_cos = ops::cos(&angles)?;

        // Interleave sin and cos: [pe_len, d_model]
        // Stack along last axis then reshape
        let pe_stacked = ops::stack_axis(&[pe_sin, pe_cos], -1)?;
        let pe = pe_stacked.reshape(&[pe_len as i32, d_model as i32])?;

        // Add batch dimension: [1, pe_len, d_model]
        let pe = pe.reshape(&[1, pe_len as i32, d_model as i32])?;

        Ok(pe)
    }

    /// Get positional encoding for a given sequence length.
    ///
    /// Returns positional encodings centered around position 0,
    /// covering positions from (seq_len-1) to -(seq_len-1).
    ///
    /// # Arguments
    /// * `seq_len` - Current sequence length
    ///
    /// # Returns
    /// Positional encodings: [1, 2*seq_len-1, d_model]
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
        let pe_len = 2 * self.max_len - 1;
        let out_len = 2 * seq_len - 1;
        let start = (pe_len - out_len) / 2;
        let end = start + out_len;

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

    /// Query projection
    linear_q: nn::Linear,
    /// Key projection
    linear_k: nn::Linear,
    /// Value projection
    linear_v: nn::Linear,
    /// Output projection
    linear_out: nn::Linear,
    /// Position embedding projection (no bias)
    linear_pos: nn::Linear,

    /// Position bias for content attention [n_heads, head_dim]
    pos_bias_u: Array,
    /// Position bias for position attention [n_heads, head_dim]
    pos_bias_v: Array,
}

impl RelPositionMultiHeadAttention {
    /// Create a new relative position multi-head attention module.
    pub fn new(d_model: usize, n_heads: usize) -> Result<Self> {
        let head_dim = d_model / n_heads;

        // Create linear projections
        let linear_q = nn::Linear::new(d_model as i32, d_model as i32)?;
        let linear_k = nn::Linear::new(d_model as i32, d_model as i32)?;
        let linear_v = nn::Linear::new(d_model as i32, d_model as i32)?;
        let linear_out = nn::Linear::new(d_model as i32, d_model as i32)?;

        // Position projection without bias
        let linear_pos = nn::LinearBuilder::new(d_model as i32, d_model as i32)
            .bias(false)
            .build()?;

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
        })
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
        let batch_size = x.dim(0) as i32;
        let seq_len = x.dim(1) as i32;
        let n_heads = self.n_heads as i32;
        let head_dim = self.head_dim as i32;

        // Project Q, K, V: [batch, seq_len, d_model]
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;

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
        let matrix_bd = Self::rel_shift(&matrix_bd)?;

        // Truncate to seq_len (take first seq_len columns)
        // matrix_bd: [batch, n_heads, seq_len, 2*seq_len-1] -> [batch, n_heads, seq_len, seq_len]
        let matrix_bd = matrix_bd.index((.., .., .., ..seq_len));

        // Combine and scale
        let scale = (head_dim as f32).sqrt();
        let scores = (&matrix_ac + &matrix_bd) / Array::from_f32(scale);

        // Softmax
        let attn_weights = ops::softmax_axis(&scores, -1, None)?;

        // Apply attention to values: [batch, n_heads, seq_len, head_dim]
        let context = ops::matmul(&attn_weights, &v)?;

        // Transpose back: [batch, seq_len, n_heads, head_dim]
        let context = ops::transpose_axes(&context, &[0, 2, 1, 3])?;

        // Reshape: [batch, seq_len, d_model]
        let context = context.reshape(&[batch_size, seq_len, self.d_model as i32])?;

        // Output projection
        let output = self.linear_out.forward(&context)?;

        Ok(output)
    }

    /// Relative shift operation to align position attention scores.
    ///
    /// Transforms position attention from shape where each row has positions
    /// [L-1, L-2, ..., 0, ..., -(L-1)] to a skewed matrix where row i has
    /// the correct relative positions to keys 0..L.
    ///
    /// # Arguments
    /// * `x` - Position attention scores [batch, n_heads, seq_len, 2*seq_len-1]
    ///
    /// # Returns
    /// Shifted scores [batch, n_heads, seq_len, 2*seq_len-1]
    fn rel_shift(x: &Array) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let n_heads = shape[1] as i32;
        let seq_len = shape[2] as i32;
        let pos_len = shape[3] as i32; // 2*seq_len - 1

        // Step 1: Pad with zeros on the left of last dimension
        // [batch, n_heads, seq_len, 2*seq_len-1] -> [batch, n_heads, seq_len, 2*seq_len]
        let zeros = ops::zeros::<f32>(&[batch, n_heads, seq_len, 1])?;
        let x_padded = ops::concatenate_axis(&[&zeros, x], -1)?;

        // Step 2: Reshape to [batch, n_heads, 2*seq_len, seq_len]
        let x_reshaped = x_padded.reshape(&[batch, n_heads, 2 * seq_len, seq_len])?;

        // Step 3: Slice from index 1 in dimension 2 (skip first row)
        // [batch, n_heads, 2*seq_len, seq_len] -> [batch, n_heads, 2*seq_len-1, seq_len]
        let x_sliced = x_reshaped.index((.., .., 1.., ..));

        // Step 4: Reshape back to [batch, n_heads, seq_len, 2*seq_len-1]
        let x_shifted = x_sliced.reshape(&[batch, n_heads, seq_len, pos_len])?;

        Ok(x_shifted)
    }

    /// Load weights from a state dict.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        load_linear(&mut self.linear_q, weights, &format!("{}.linear_q", prefix));
        load_linear(&mut self.linear_k, weights, &format!("{}.linear_k", prefix));
        load_linear(&mut self.linear_v, weights, &format!("{}.linear_v", prefix));
        load_linear(
            &mut self.linear_out,
            weights,
            &format!("{}.linear_out", prefix),
        );

        // linear_pos has no bias
        if let Some(weight) = weights.get(&format!("{}.linear_pos.weight", prefix)) {
            self.linear_pos.weight.value = weight.clone();
        }

        // Load position biases (per-layer, not shared)
        if let Some(bias_u) = weights.get(&format!("{}.pos_bias_u", prefix)) {
            self.pos_bias_u = bias_u.clone();
        }
        if let Some(bias_v) = weights.get(&format!("{}.pos_bias_v", prefix)) {
            self.pos_bias_v = bias_v.clone();
        }

        Ok(())
    }
}

/// Helper to load Linear weights
fn load_linear(layer: &mut nn::Linear, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        layer.weight.value = weight.clone();
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
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
}
