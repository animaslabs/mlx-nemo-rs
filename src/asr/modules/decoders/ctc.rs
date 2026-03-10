//! CTC decoder implementation.
//!
//! Projects encoder output to vocabulary logits for CTC decoding.

use std::collections::HashMap;

use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::Array;

use crate::asr::config::DecoderConfig;
use crate::core::error::Result;

/// CTC decoder that projects encoder output to vocabulary logits.
pub struct CTCDecoder {
    #[allow(dead_code)]
    config: DecoderConfig,
    /// Linear projection to vocabulary size
    decoder: nn::Linear,
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
        let decoder = nn::Linear::new(feat_in as i32, vocab_size as i32)?;

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
    pub fn forward(&mut self, encoded: &Array) -> Result<Array> {
        // Transpose from [B, D, T] to [B, T, D]
        let x = mlx_rs::ops::transpose_axes(encoded, &[0, 2, 1])?;

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
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        // NeMo uses "decoder_layers.0" for the CTC projection
        // It's a Conv1d layer with kernel_size=1, shape [vocab_size, d_model, 1]
        let weight_key = format!("{}.decoder_layers.0.weight", prefix);
        let bias_key = format!("{}.decoder_layers.0.bias", prefix);

        // Try to find weights
        if let Some(weight) = weights.get(&weight_key) {
            // NeMo Conv1d weight shape is [O, I, K] = [vocab_size, d_model, 1]
            // We need to squeeze the last dimension for Linear: [vocab_size, d_model]
            if weight.ndim() == 3 {
                // Squeeze kernel dimension: [O, I, 1] -> [O, I]
                if let Ok(squeezed) = weight.squeeze_axes(&[-1]) {
                    self.decoder.weight.value = squeezed;
                }
            } else {
                self.decoder.weight.value = weight.clone();
            }
        }

        if let Some(bias) = weights.get(&bias_key) {
            self.decoder.bias.value = Some(bias.clone());
        }

        Ok(())
    }
}
