//! CTC decoder implementation.
//!
//! Projects encoder output to vocabulary logits for CTC decoding.

use std::collections::HashMap;

use mlx_rs::ops;
use mlx_rs::Array;

use crate::asr::config::DecoderConfig;
use crate::core::error::Result;
use crate::core::quantization::{LinearLayer, QuantizationConfig, load_linear_layer};

/// CTC decoder that projects encoder output to vocabulary logits.
pub struct CTCDecoder {
    #[allow(dead_code)]
    config: DecoderConfig,
    /// Linear projection to vocabulary size (supports quantized weights)
    decoder: LinearLayer,
    /// Vocabulary size (including blank token)
    vocab_size: usize,
}

impl CTCDecoder {
    /// Create a new CTC decoder.
    ///
    /// # Arguments
    /// * `config` - Decoder configuration
    /// * `feat_in` - Input feature dimension from encoder
    /// * `vocab_size` - Vocabulary size (including blank token)
    pub fn new(config: DecoderConfig, feat_in: usize, vocab_size: usize) -> Result<Self> {
        // Initialize with zeros (will be loaded from weights)
        let weight = ops::zeros::<f32>(&[vocab_size as i32, feat_in as i32])?;
        let bias = Some(ops::zeros::<f32>(&[vocab_size as i32])?);
        let decoder = LinearLayer::standard(weight, bias);

        Ok(Self {
            config,
            decoder,
            vocab_size,
        })
    }

    /// Forward pass - project encoder output to log probabilities.
    ///
    /// # Arguments
    /// * `encoded` - Encoder output [batch, d_model, time]
    ///
    /// # Returns
    /// * Log probabilities [batch, time, vocab_size]
    pub fn forward(&self, encoded: &Array) -> Result<Array> {
        // Transpose from [B, D, T] to [B, T, D]
        let x = ops::transpose_axes(encoded, &[0, 2, 1])?;

        // Project to vocabulary size
        let logits = self.decoder.forward(&x)?;

        // Apply log softmax for CTC
        let log_probs = mlx_rs::nn::log_softmax(&logits, -1)?;

        Ok(log_probs)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
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
        // Try to load using the quantization-aware loader first
        let linear_prefix = format!("{}.decoder_layers.0", prefix);
        if let Some(layer) = load_linear_layer(weights, &linear_prefix, quant) {
            self.decoder = layer;
            tracing::debug!("Loaded CTC decoder from {}", linear_prefix);
            return Ok(());
        }

        // Fall back to manual loading for NeMo Conv1d format
        // NeMo uses "decoder_layers.0" for the CTC projection
        // It's a Conv1d layer with kernel_size=1, shape [vocab_size, d_model, 1]
        let weight_key = format!("{}.decoder_layers.0.weight", prefix);
        let bias_key = format!("{}.decoder_layers.0.bias", prefix);

        // Try to find weights
        if let Some(weight) = weights.get(&weight_key) {
            // NeMo Conv1d weight shape is [O, I, K] = [vocab_size, d_model, 1]
            // We need to squeeze the last dimension for Linear: [vocab_size, d_model]
            let weight = if weight.ndim() == 3 {
                // Squeeze kernel dimension: [O, I, 1] -> [O, I]
                weight.squeeze_axes(&[-1]).unwrap_or_else(|_| weight.clone())
            } else {
                weight.clone()
            };

            let bias = weights.get(&bias_key).cloned();
            self.decoder = LinearLayer::standard(weight, bias);
        }

        Ok(())
    }
}
