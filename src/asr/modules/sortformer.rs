//! SortFormer speaker probability head.
//!
//! Neural layers that take the transformer encoder output and produce
//! per-frame speaker activity probabilities.
//!
//! Architecture:
//!   encoder_output [B, T, fc_d_model]
//!   → encoder_proj (optional, if fc_d_model != tf_d_model)
//!   → ... (transformer encoder processes in parent model)
//!   → hidden_out [B, T, tf_d_model]
//!   → forward_speaker_sigmoids:
//!       ReLU → first_hidden_to_hidden → ReLU → single_hidden_to_spks → Sigmoid
//!   → preds [B, T, num_spks]

use std::collections::HashMap;

use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::Array;

use crate::asr::config::SortformerModulesConfig;
use crate::core::error::Result;
use crate::core::quantization::{load_linear_layer, LinearLayer, QuantizationConfig};

/// SortFormer speaker head modules.
pub struct SortformerModules {
    /// Optional projection from conformer dim to transformer dim
    pub encoder_proj: Option<LinearLayer>,
    /// Hidden → hidden layer
    first_hidden_to_hidden: LinearLayer,
    /// Hidden → speaker logits layer
    single_hidden_to_spks: LinearLayer,
    #[allow(dead_code)]
    num_spks: usize,
    #[allow(dead_code)]
    fc_d_model: usize,
    #[allow(dead_code)]
    tf_d_model: usize,
}

impl SortformerModules {
    /// Create new SortformerModules from configuration.
    pub fn new(config: &SortformerModulesConfig) -> Result<Self> {
        let fc = config.fc_d_model as i32;
        let tf = config.tf_d_model as i32;
        let num_spks = config.num_spks as i32;

        // encoder_proj: only created if fc_d_model != tf_d_model
        let encoder_proj = if config.fc_d_model != config.tf_d_model {
            let w = ops::zeros::<f32>(&[tf, fc])?;
            let b = ops::zeros::<f32>(&[tf])?;
            Some(LinearLayer::standard(w, Some(b)))
        } else {
            None
        };

        // first_hidden_to_hidden: Linear(tf_d_model → tf_d_model)
        let w1 = ops::zeros::<f32>(&[tf, tf])?;
        let b1 = ops::zeros::<f32>(&[tf])?;
        let first_hidden_to_hidden = LinearLayer::standard(w1, Some(b1));

        // single_hidden_to_spks: Linear(tf_d_model → num_spks)
        let w2 = ops::zeros::<f32>(&[num_spks, tf])?;
        let b2 = ops::zeros::<f32>(&[num_spks])?;
        let single_hidden_to_spks = LinearLayer::standard(w2, Some(b2));

        Ok(Self {
            encoder_proj,
            first_hidden_to_hidden,
            single_hidden_to_spks,
            num_spks: config.num_spks,
            fc_d_model: config.fc_d_model,
            tf_d_model: config.tf_d_model,
        })
    }

    /// Project encoder output from fc_d_model to tf_d_model (if dimensions differ).
    ///
    /// # Arguments
    /// * `encoder_output` - [B, T, fc_d_model]
    ///
    /// # Returns
    /// * [B, T, tf_d_model]
    pub fn project_encoder_output(&self, encoder_output: &Array) -> Result<Array> {
        if let Some(ref proj) = self.encoder_proj {
            proj.forward(encoder_output)
        } else {
            Ok(encoder_output.clone())
        }
    }

    /// Compute speaker activity sigmoids from transformer hidden output.
    ///
    /// # Arguments
    /// * `hidden_out` - Transformer encoder output [B, T, tf_d_model]
    ///
    /// # Returns
    /// * Speaker probabilities [B, T, num_spks] in range [0, 1]
    pub fn forward_speaker_sigmoids(&self, hidden_out: &Array) -> Result<Array> {
        let zero = Array::from_f32(0.0);

        // ReLU
        let x = ops::maximum(hidden_out, &zero)?;

        // first_hidden_to_hidden
        let x = self.first_hidden_to_hidden.forward(&x)?;

        // ReLU
        let x = ops::maximum(&x, &zero)?;

        // single_hidden_to_spks
        let x = self.single_hidden_to_spks.forward(&x)?;

        // Sigmoid
        nn::sigmoid(&x).map_err(Into::into)
    }

    /// Load weights from checkpoint.
    ///
    /// Weight prefix: `sortformer_modules.{submodule}`
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        if let Some(layer) = load_linear_layer(weights, &format!("{}.encoder_proj.0", prefix), quant) {
            self.encoder_proj = Some(layer);
        }

        if let Some(layer) = load_linear_layer(weights, &format!("{}.first_hidden_to_hidden.0", prefix), quant) {
            self.first_hidden_to_hidden = layer;
        }

        if let Some(layer) = load_linear_layer(weights, &format!("{}.single_hidden_to_spks", prefix), quant) {
            self.single_hidden_to_spks = layer;
        }

        Ok(())
    }
}
