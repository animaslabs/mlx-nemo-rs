//! Quantization support for MLX models.
//!
//! This module provides types for handling quantized weights in MLX format.
//! MLX stores quantized weights as:
//! - `{layer}.weight`: U32 packed bits, shape `(out, in / (32/bits))`
//! - `{layer}.scales`: F16, shape `(out, in / group_size)`
//! - `{layer}.biases`: F16, shape `(out, in / group_size)` - quantization biases
//! - `{layer}.bias`: F16, shape `(out)` - regular layer bias (optional)

use std::collections::HashMap;

use mlx_rs::ops::{self, indexing};
use mlx_rs::Array;

use crate::core::ArrayExt;
use crate::core::error::Result;

/// Quantization configuration from model config.json.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Bits per weight (typically 4 or 8)
    pub bits: i32,
    /// Group size for quantization (typically 64)
    pub group_size: i32,
}

impl QuantizationConfig {
    /// Create a new quantization config.
    pub fn new(bits: i32, group_size: i32) -> Self {
        Self { bits, group_size }
    }
}

/// Linear layer that can be standard (float) or quantized.
pub enum LinearLayer {
    /// Standard float linear layer
    Standard {
        weight: Array,
        bias: Option<Array>,
    },
    /// Quantized linear layer
    Quantized {
        /// Packed quantized weights [out, in / (32/bits)]
        weight: Array,
        /// Quantization scales [out, in / group_size]
        scales: Array,
        /// Quantization biases [out, in / group_size]
        biases: Array,
        /// Optional layer bias [out]
        bias: Option<Array>,
        /// Bits per weight
        bits: i32,
        /// Group size
        group_size: i32,
    },
}

impl LinearLayer {
    /// Create a standard (non-quantized) linear layer.
    pub fn standard(weight: Array, bias: Option<Array>) -> Self {
        Self::Standard { weight, bias }
    }

    /// Create a quantized linear layer.
    pub fn quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        bias: Option<Array>,
        bits: i32,
        group_size: i32,
    ) -> Self {
        Self::Quantized {
            weight,
            scales,
            biases,
            bias,
            bits,
            group_size,
        }
    }

    /// Check if this layer is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Forward pass through the linear layer.
    ///
    /// For quantized layers, uses `mlx_rs::ops::quantized_matmul`.
    /// For standard layers, uses regular matmul.
    ///
    /// # Arguments
    /// * `x` - Input tensor [..., in_features]
    ///
    /// # Returns
    /// * Output tensor [..., out_features]
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Standard { weight, bias } => {
                // Standard linear: y = x @ W^T + b
                let weight_t = weight.transpose()?;
                let mut result = ops::matmul(x, &weight_t)?;
                if let Some(b) = bias {
                    result = ops::add(&result, b)?;
                }
                Ok(result)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                bias,
                bits,
                group_size,
            } => {
                // Quantized linear using MLX's quantized_matmul
                // quantized_matmul computes: x @ dequantize(weight, scales, biases)^T
                let mut result = ops::quantized_matmul(
                    x,
                    weight,
                    scales,
                    biases,
                    true,        // transpose
                    *group_size,
                    *bits,
                )?;
                if let Some(b) = bias {
                    result = ops::add(&result, b)?;
                }
                Ok(result)
            }
        }
    }
}

/// Embedding layer that can be standard (float) or quantized.
pub enum EmbeddingLayer {
    /// Standard float embedding
    Standard(Array),
    /// Quantized embedding
    Quantized {
        /// Packed quantized weights [vocab_size, embed_dim / (32/bits)]
        weight: Array,
        /// Quantization scales [vocab_size, embed_dim / group_size]
        scales: Array,
        /// Quantization biases [vocab_size, embed_dim / group_size]
        biases: Array,
        /// Bits per weight
        bits: i32,
        /// Group size
        group_size: i32,
    },
}

impl EmbeddingLayer {
    /// Create a standard (non-quantized) embedding layer.
    pub fn standard(weight: Array) -> Self {
        Self::Standard(weight)
    }

    /// Create a quantized embedding layer.
    pub fn quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        bits: i32,
        group_size: i32,
    ) -> Self {
        Self::Quantized {
            weight,
            scales,
            biases,
            bits,
            group_size,
        }
    }

    /// Check if this embedding is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Get the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::Standard(weight) => weight.dim(-1) as usize,
            Self::Quantized {
                weight,
                bits,
                ..
            } => {
                // weight shape is [vocab, embed_dim / (32/bits)]
                // So embed_dim = weight.dim(-1) * (32/bits)
                let packed_dim = weight.dim(-1) as i32;
                (packed_dim * (32 / bits)) as usize
            }
        }
    }

    /// Lookup embeddings for the given indices.
    ///
    /// For quantized embeddings, dequantizes the selected rows.
    ///
    /// # Arguments
    /// * `indices` - Token indices [...] (any shape)
    ///
    /// # Returns
    /// * Embeddings [..., embed_dim]
    pub fn lookup(&self, indices: &Array) -> Result<Array> {
        match self {
            Self::Standard(weight) => {
                // Standard embedding lookup using take_axis
                let flat_indices = indices.reshape(&[-1])?;
                let flat_embeddings = indexing::take_axis(weight, &flat_indices, 0)?;

                // Reshape back to [..., embed_dim]
                let embed_dim = weight.dim(-1);
                let mut out_shape: Vec<i32> = indices.shape().to_vec();
                out_shape.push(embed_dim);
                flat_embeddings.reshape(&out_shape).map_err(Into::into)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                bits,
                group_size,
            } => {
                // For quantized embeddings, we need to:
                // 1. Take the quantized rows for the indices
                // 2. Dequantize those rows
                let flat_indices = indices.reshape(&[-1])?;

                // Take quantized weight rows
                let quant_rows = indexing::take_axis(weight, &flat_indices, 0)?;
                let scale_rows = indexing::take_axis(scales, &flat_indices, 0)?;
                let bias_rows = indexing::take_axis(biases, &flat_indices, 0)?;

                // Dequantize: weight_float = scales * (weight_int - biases)
                // MLX dequantize function handles unpacking
                let dequantized = ops::dequantize(
                    &quant_rows,
                    &scale_rows,
                    &bias_rows,
                    *group_size,
                    *bits,
                )?;

                // Reshape back to [..., embed_dim]
                let embed_dim = dequantized.dim(-1);
                let mut out_shape: Vec<i32> = indices.shape().to_vec();
                out_shape.push(embed_dim);
                dequantized.reshape(&out_shape).map_err(Into::into)
            }
        }
    }
}

/// Load a linear layer from weights, auto-detecting quantization.
///
/// If `{prefix}.scales` exists in weights, loads as quantized.
/// Otherwise loads as standard float layer.
///
/// # Arguments
/// * `weights` - Weight dictionary
/// * `prefix` - Weight key prefix (e.g., "encoder.layers.0.ff.linear1")
/// * `quant` - Optional quantization config (for bits/group_size)
pub fn load_linear_layer(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Option<LinearLayer> {
    let weight_key = format!("{}.weight", prefix);
    let scales_key = format!("{}.scales", prefix);
    let biases_key = format!("{}.biases", prefix);
    let bias_key = format!("{}.bias", prefix);

    // Check if quantized by looking for scales
    if let (Some(weight), Some(scales), Some(qbiases)) = (
        weights.get(&weight_key),
        weights.get(&scales_key),
        weights.get(&biases_key),
    ) {
        // Quantized layer
        let bias = weights.get(&bias_key).cloned();
        let (bits, group_size) = quant
            .map(|q| (q.bits, q.group_size))
            .unwrap_or((8, 64)); // Default to 8-bit, group_size 64

        tracing::debug!(
            prefix,
            weight_shape = ?weight.shape(),
            scales_shape = ?scales.shape(),
            bits,
            group_size,
            has_bias = bias.is_some(),
            "loading quantized linear layer"
        );

        Some(LinearLayer::quantized(
            weight.clone(),
            scales.clone(),
            qbiases.clone(),
            bias,
            bits,
            group_size,
        ))
    } else if let Some(weight) = weights.get(&weight_key) {
        // Standard float layer
        let bias = weights.get(&bias_key).cloned();

        // Handle 3D pointwise conv weights [O, 1, I] by squeezing to [O, I]
        let weight = if weight.ndim() == 3 && weight.dim(1) == 1 {
            weight.squeeze_axes(&[1]).ok()?
        } else {
            weight.clone()
        };

        tracing::trace!(
            prefix,
            weight_shape = ?weight.shape(),
            has_bias = bias.is_some(),
            "loading standard linear layer"
        );

        Some(LinearLayer::standard(weight, bias))
    } else {
        tracing::warn!(prefix, "linear layer weights not found");
        None
    }
}

/// Load an embedding layer from weights, auto-detecting quantization.
///
/// If `{prefix}.scales` exists in weights, loads as quantized.
/// Otherwise loads as standard float embedding.
///
/// # Arguments
/// * `weights` - Weight dictionary
/// * `prefix` - Weight key prefix (e.g., "decoder.embed")
/// * `quant` - Optional quantization config (for bits/group_size)
pub fn load_embedding_layer(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Option<EmbeddingLayer> {
    let weight_key = format!("{}.weight", prefix);
    let scales_key = format!("{}.scales", prefix);
    let biases_key = format!("{}.biases", prefix);

    // Check if quantized by looking for scales
    if let (Some(weight), Some(scales), Some(qbiases)) = (
        weights.get(&weight_key),
        weights.get(&scales_key),
        weights.get(&biases_key),
    ) {
        // Quantized embedding
        let (bits, group_size) = quant
            .map(|q| (q.bits, q.group_size))
            .unwrap_or((8, 64));

        tracing::debug!(
            prefix,
            weight_shape = ?weight.shape(),
            scales_shape = ?scales.shape(),
            bits,
            group_size,
            "loading quantized embedding layer"
        );

        Some(EmbeddingLayer::quantized(
            weight.clone(),
            scales.clone(),
            qbiases.clone(),
            bits,
            group_size,
        ))
    } else if let Some(weight) = weights.get(&weight_key) {
        // Standard float embedding
        tracing::trace!(
            prefix,
            weight_shape = ?weight.shape(),
            "loading standard embedding layer"
        );

        Some(EmbeddingLayer::standard(weight.clone()))
    } else {
        tracing::warn!(prefix, "embedding layer weights not found");
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_linear_forward() {
        // Create a simple 2x3 weight matrix
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let bias = Array::from_slice(&[0.1f32, 0.2], &[2]);
        let layer = LinearLayer::standard(weight, Some(bias));

        // Input: [1, 3]
        let x = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3]);
        let y = layer.forward(&x).unwrap();
        y.eval().unwrap();

        // Expected: [1, 2] = [[1+2+3+0.1, 4+5+6+0.2]] = [[6.1, 15.2]]
        assert_eq!(y.shape(), &[1, 2]);
        let y_data: Vec<f32> = y.to_vec_row_major().unwrap();
        assert!((y_data[0] - 6.1).abs() < 1e-5);
        assert!((y_data[1] - 15.2).abs() < 1e-5);
    }

    #[test]
    fn test_standard_embedding_lookup() {
        // Create a 4x3 embedding matrix (4 tokens, 3-dim embeddings)
        let weight = Array::from_slice(
            &[
                1.0f32, 2.0, 3.0,   // token 0
                4.0, 5.0, 6.0,      // token 1
                7.0, 8.0, 9.0,      // token 2
                10.0, 11.0, 12.0,   // token 3
            ],
            &[4, 3],
        );
        let layer = EmbeddingLayer::standard(weight);

        // Lookup tokens [1, 3]
        let indices = Array::from_slice(&[1i32, 3], &[2]);
        let embeddings = layer.lookup(&indices).unwrap();
        embeddings.eval().unwrap();

        assert_eq!(embeddings.shape(), &[2, 3]);
        let data: Vec<f32> = embeddings.to_vec_row_major().unwrap();
        // Token 1: [4, 5, 6]
        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 6.0).abs() < 1e-5);
        // Token 3: [10, 11, 12]
        assert!((data[3] - 10.0).abs() < 1e-5);
        assert!((data[4] - 11.0).abs() < 1e-5);
        assert!((data[5] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::new(8, 64);
        assert_eq!(config.bits, 8);
        assert_eq!(config.group_size, 64);
    }
}
