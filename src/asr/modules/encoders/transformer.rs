//! Standard Transformer encoder for SortFormer diarization.
//!
//! This implements NeMo's `TransformerEncoder` used by the SortFormer model.
//! It is distinct from the ConformerEncoder (which adds convolution modules).
//!
//! Architecture per block (post-LN mode, pre_ln=False):
//!   x → MultiHeadAttention → residual → LayerNorm → FFN → residual → LayerNorm

use std::collections::HashMap;

use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::Array;

use crate::asr::config::TransformerEncoderConfig;
use crate::core::error::Result;
use crate::core::quantization::{load_linear_layer, LinearLayer, QuantizationConfig};

/// Multi-head attention (standard, no relative positional encoding).
///
/// NeMo's MultiHeadAttention uses sqrt(sqrt(head_dim)) scaling applied
/// to both Q and K (rather than 1/sqrt(head_dim) on the dot product).
struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    /// Scale factor: sqrt(sqrt(head_dim)), applied to both Q and K
    scale: f32,

    query_proj: LinearLayer,
    key_proj: LinearLayer,
    value_proj: LinearLayer,
    out_proj: LinearLayer,
}

impl MultiHeadAttention {
    fn new(hidden_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        // NeMo uses sqrt(sqrt(head_dim)) applied to both Q and K
        let scale = (head_dim as f32).sqrt().sqrt();

        // Placeholder weights - will be loaded from checkpoint
        let q_weight = ops::zeros::<f32>(&[hidden_size as i32, hidden_size as i32])?;
        let q_bias = ops::zeros::<f32>(&[hidden_size as i32])?;
        let k_weight = ops::zeros::<f32>(&[hidden_size as i32, hidden_size as i32])?;
        let k_bias = ops::zeros::<f32>(&[hidden_size as i32])?;
        let v_weight = ops::zeros::<f32>(&[hidden_size as i32, hidden_size as i32])?;
        let v_bias = ops::zeros::<f32>(&[hidden_size as i32])?;
        let o_weight = ops::zeros::<f32>(&[hidden_size as i32, hidden_size as i32])?;
        let o_bias = ops::zeros::<f32>(&[hidden_size as i32])?;

        Ok(Self {
            num_heads,
            head_dim,
            scale,
            query_proj: LinearLayer::standard(q_weight, Some(q_bias)),
            key_proj: LinearLayer::standard(k_weight, Some(k_bias)),
            value_proj: LinearLayer::standard(v_weight, Some(v_bias)),
            out_proj: LinearLayer::standard(o_weight, Some(o_bias)),
        })
    }

    fn forward(
        &self,
        query: &Array,
        key: &Array,
        value: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let batch = query.dim(0);
        let seq_len_q = query.dim(1);
        let seq_len_k = key.dim(1);
        let num_heads = self.num_heads as i32;
        let head_dim = self.head_dim as i32;

        // Project Q, K, V
        let q = self.query_proj.forward(query)?;
        let k = self.key_proj.forward(key)?;
        let v = self.value_proj.forward(value)?;

        // Reshape to [B, seq, num_heads, head_dim] then transpose to [B, num_heads, seq, head_dim]
        let q = q.reshape(&[batch, seq_len_q, num_heads, head_dim])?;
        let q = ops::transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len_k, num_heads, head_dim])?;
        let k = ops::transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len_k, num_heads, head_dim])?;
        let v = ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        // Apply sqrt(sqrt(head_dim)) scaling to both Q and K
        let scale_arr = Array::from_f32(self.scale);
        let q = ops::divide(&q, &scale_arr)?;
        let k = ops::divide(&k, &scale_arr)?;

        // Attention scores: Q @ K^T -> [B, num_heads, seq_q, seq_k]
        let k_t = ops::transpose_axes(&k, &[0, 1, 3, 2])?;
        let mut scores = ops::matmul(&q, &k_t)?;

        // Apply mask if provided
        if let Some(mask) = mask {
            // mask shape: [B, 1, seq_q, seq_k] or [1, 1, seq_q, seq_k]
            // True = attend, False = mask out
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            scores = ops::select(mask, &scores, &neg_inf)?;
        }

        // Softmax
        let attn_weights = ops::softmax_axis(&scores, -1, None)?;

        // Weighted sum: [B, num_heads, seq_q, head_dim]
        let context = ops::matmul(&attn_weights, &v)?;

        // Transpose back and reshape: [B, seq_q, hidden_size]
        let context = ops::transpose_axes(&context, &[0, 2, 1, 3])?;
        let hidden_size = num_heads * head_dim;
        let context = context.reshape(&[batch, seq_len_q, hidden_size])?;

        // Output projection
        self.out_proj.forward(&context)
    }

    fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) {
        if let Some(layer) = load_linear_layer(weights, &format!("{}.query_net", prefix), quant) {
            self.query_proj = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.key_net", prefix), quant) {
            self.key_proj = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.value_net", prefix), quant) {
            self.value_proj = layer;
        }
        if let Some(layer) =
            load_linear_layer(weights, &format!("{}.out_projection", prefix), quant)
        {
            self.out_proj = layer;
        }
    }
}

/// Position-wise feed-forward network.
///
/// Linear(hidden→inner) → activation → Linear(inner→hidden)
struct PositionWiseFF {
    linear1: LinearLayer,
    linear2: LinearLayer,
    activation: String,
}

impl PositionWiseFF {
    fn new(hidden_size: usize, inner_size: usize, activation: &str) -> Result<Self> {
        let h = hidden_size as i32;
        let i = inner_size as i32;

        let w1 = ops::zeros::<f32>(&[i, h])?;
        let b1 = ops::zeros::<f32>(&[i])?;
        let w2 = ops::zeros::<f32>(&[h, i])?;
        let b2 = ops::zeros::<f32>(&[h])?;

        Ok(Self {
            linear1: LinearLayer::standard(w1, Some(b1)),
            linear2: LinearLayer::standard(w2, Some(b2)),
            activation: activation.to_string(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = self.linear1.forward(x)?;

        h = match self.activation.as_str() {
            "gelu" => nn::gelu(&h)?,
            _ => ops::maximum(&h, &Array::from_f32(0.0))?, // relu default
        };

        self.linear2.forward(&h)
    }

    fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) {
        if let Some(layer) = load_linear_layer(weights, &format!("{}.dense_in", prefix), quant) {
            self.linear1 = layer;
        }
        if let Some(layer) = load_linear_layer(weights, &format!("{}.dense_out", prefix), quant) {
            self.linear2 = layer;
        }
    }
}

/// Helper to load LayerNorm weights.
fn load_layer_norm(layer: &mut nn::LayerNorm, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        layer.weight.value = Some(weight.clone());
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
    }
}

/// A single Transformer encoder block (post-LN mode).
///
/// Forward:
///   x → MHA(x, x, x) → + x → LayerNorm → FFN → + residual → LayerNorm
struct TransformerEncoderBlock {
    self_attn: MultiHeadAttention,
    ffn: PositionWiseFF,
    layer_norm_1: nn::LayerNorm,
    layer_norm_2: nn::LayerNorm,
    pre_ln: bool,
}

impl TransformerEncoderBlock {
    fn new(config: &TransformerEncoderConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let inner_size = config.inner_size;
        let num_heads = config.num_attention_heads;

        Ok(Self {
            self_attn: MultiHeadAttention::new(hidden_size, num_heads)?,
            ffn: PositionWiseFF::new(hidden_size, inner_size, &config.hidden_act)?,
            layer_norm_1: nn::LayerNorm::new(hidden_size as i32)?,
            layer_norm_2: nn::LayerNorm::new(hidden_size as i32)?,
            pre_ln: config.pre_ln,
        })
    }

    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        if self.pre_ln {
            // Pre-LN: LayerNorm → MHA → residual, LayerNorm → FFN → residual
            let residual = x.clone();
            let normed = self.layer_norm_1.forward(x)?;
            let attn_out = self.self_attn.forward(&normed, &normed, &normed, mask)?;
            let x = ops::add(&residual, &attn_out)?;

            let residual = x.clone();
            let normed = self.layer_norm_2.forward(&x)?;
            let ffn_out = self.ffn.forward(&normed)?;
            ops::add(&residual, &ffn_out).map_err(Into::into)
        } else {
            // Post-LN: MHA → residual → LayerNorm, FFN → residual → LayerNorm
            let residual = x.clone();
            let attn_out = self.self_attn.forward(x, x, x, mask)?;
            let x = ops::add(&residual, &attn_out)?;
            let x = self.layer_norm_1.forward(&x)?;

            let residual = x.clone();
            let ffn_out = self.ffn.forward(&x)?;
            let x = ops::add(&residual, &ffn_out)?;
            self.layer_norm_2.forward(&x).map_err(Into::into)
        }
    }

    fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) {
        // NeMo weight prefix: {prefix}.first_sub_layer (MHA), {prefix}.second_sub_layer (FFN)
        self.self_attn
            .load_weights(weights, &format!("{}.first_sub_layer", prefix), quant);
        self.ffn
            .load_weights(weights, &format!("{}.second_sub_layer", prefix), quant);
        load_layer_norm(
            &mut self.layer_norm_1,
            weights,
            &format!("{}.layer_norm_1", prefix),
        );
        load_layer_norm(
            &mut self.layer_norm_2,
            weights,
            &format!("{}.layer_norm_2", prefix),
        );
    }
}

/// Standard Transformer encoder.
///
/// Stack of N TransformerEncoderBlocks with optional final LayerNorm.
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderBlock>,
    final_norm: Option<nn::LayerNorm>,
    #[allow(dead_code)]
    config: TransformerEncoderConfig,
}

impl TransformerEncoder {
    /// Create a new TransformerEncoder from configuration.
    pub fn new(config: TransformerEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerEncoderBlock::new(&config)?);
        }

        // Final layer norm: used when pre_ln=False and pre_ln_final_layer_norm=True
        let final_norm = if !config.pre_ln && config.pre_ln_final_layer_norm {
            Some(nn::LayerNorm::new(config.hidden_size as i32)?)
        } else {
            None
        };

        Ok(Self {
            layers,
            final_norm,
            config,
        })
    }

    /// Forward pass through the transformer encoder.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, hidden_size]
    /// * `mask` - Optional attention mask [batch, 1, seq_len, seq_len] or broadcastable
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, hidden_size]
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let mut out = x.clone();
        for layer in &mut self.layers {
            out = layer.forward(&out, mask)?;
        }

        if let Some(ref mut norm) = self.final_norm {
            // The final norm is NOT applied here because post-LN blocks
            // already have LayerNorm at the end. The pre_ln_final_layer_norm
            // config option in NeMo adds an EXTRA norm at the end of the encoder
            // only when using pre-LN mode. For post-LN mode (our case),
            // each block already ends with LayerNorm.
            // But NeMo's code does: if pre_ln_final_layer_norm: self.final_layer_norm = LayerNorm
            // and applies it unconditionally at the end.
            // Let's follow NeMo exactly.
            out = norm.forward(&out)?;
        }

        Ok(out)
    }

    /// Load weights from a checkpoint.
    ///
    /// Weight prefix: `transformer_encoder.layers.{i}.{submodule}`
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{}.layers.{}", prefix, i), quant);
        }

        if let Some(ref mut norm) = self.final_norm {
            load_layer_norm(norm, weights, &format!("{}.final_layer_norm", prefix));
        }

        Ok(())
    }
}
