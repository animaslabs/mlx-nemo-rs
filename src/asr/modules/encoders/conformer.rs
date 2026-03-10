//! Conformer encoder implementation.
//!
//! The Conformer architecture combines convolution and self-attention
//! for effective speech feature encoding.

use std::collections::HashMap;

use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::asr::config::EncoderConfig;
use crate::core::error::Result;

/// Conformer encoder for ASR.
///
/// This implements the Conformer architecture as used in NeMo's ASR models,
/// including FastConformer variants with depthwise striding subsampling.
pub struct ConformerEncoder {
    config: EncoderConfig,

    /// Input projection / subsampling layers
    subsampling: Subsampling,

    /// Conformer blocks
    layers: Vec<ConformerBlock>,

    /// Final layer normalization
    final_norm: nn::LayerNorm,
}

impl ConformerEncoder {
    /// Create a new Conformer encoder from configuration.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let d_model = config.d_model;
        let n_layers = config.n_layers;

        // Create subsampling layer
        let subsampling = Subsampling::new(&config)?;

        // Create conformer blocks
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(ConformerBlock::new(&config)?);
        }

        // Final layer norm
        let final_norm = nn::LayerNorm::new(d_model as i32)?;

        Ok(Self {
            config,
            subsampling,
            layers,
            final_norm,
        })
    }

    /// Forward pass through the encoder.
    ///
    /// # Arguments
    /// * `features` - Input features [batch, features, time]
    /// * `lengths` - Sequence lengths [batch]
    ///
    /// # Returns
    /// * Encoded features [batch, d_model, time']
    /// * Output lengths [batch]
    pub fn forward(&mut self, features: &Array, lengths: &Array) -> Result<(Array, Array)> {
        // Transpose from [B, F, T] to [B, T, F] for processing
        let x = mlx_rs::ops::transpose_axes(features, &[0, 2, 1])?;

        // Subsampling
        let (mut x, lengths) = self.subsampling.forward(&x, lengths)?;

        // Process through conformer blocks
        for layer in &mut self.layers {
            x = layer.forward(&x, &lengths)?;
        }

        // Final normalization
        x = self.final_norm.forward(&x)?;

        // Transpose back to [B, D, T]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

        Ok((x, lengths))
    }

    /// Get output dimension.
    pub fn output_dim(&self) -> usize {
        self.config.d_model
    }

    /// Get subsampling factor.
    pub fn subsampling_factor(&self) -> usize {
        self.config.subsampling_factor
    }

    /// Load weights from a state dict.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<()> {
        // Load subsampling weights
        self.subsampling.load_weights(weights, "pre_encode")?;

        // Load conformer block weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("layers.{}", i))?;
        }

        // Load final norm weights
        if let Some(_weight) = weights.get("final_norm.weight") {
            // TODO: Set the layer norm weight
        }
        if let Some(_bias) = weights.get("final_norm.bias") {
            // TODO: Set the layer norm bias
        }

        Ok(())
    }
}

/// Subsampling layer for reducing sequence length.
struct Subsampling {
    #[allow(dead_code)]
    subsampling_factor: usize,
    /// Convolutional layers for subsampling
    conv_layers: Vec<nn::Conv1d>,
    /// Output projection
    out_proj: nn::Linear,
}

impl Subsampling {
    fn new(config: &EncoderConfig) -> Result<Self> {
        let feat_in = config.feat_in;
        let d_model = config.d_model;
        let subsampling_factor = config.subsampling_factor;

        // For dw_striding (depthwise striding) subsampling
        // Typical configuration: 2 conv layers with stride 2 each for 4x subsampling
        // or 3 conv layers for 8x subsampling
        let n_conv_layers = match subsampling_factor {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 2,
        };

        let conv_channels = config.subsampling_conv_channels.unwrap_or(256);

        let mut conv_layers = Vec::new();
        let mut in_channels = feat_in;

        for i in 0..n_conv_layers {
            let out_channels = if i == n_conv_layers - 1 {
                conv_channels
            } else {
                conv_channels
            };

            let conv = nn::Conv1dBuilder::new(in_channels as i32, out_channels as i32, 3)
                .stride(2)
                .padding(1)
                .build()?;

            conv_layers.push(conv);
            in_channels = out_channels;
        }

        // Output projection to d_model
        let out_proj = nn::Linear::new(conv_channels as i32, d_model as i32)?;

        Ok(Self {
            subsampling_factor,
            conv_layers,
            out_proj,
        })
    }

    fn forward(&mut self, x: &Array, lengths: &Array) -> Result<(Array, Array)> {
        // x: [B, T, F]
        // Transpose to [B, F, T] for conv1d
        let mut x = mlx_rs::ops::transpose_axes(x, &[0, 2, 1])?;

        // Apply conv layers with ReLU
        for conv in &mut self.conv_layers {
            x = conv.forward(&x)?;
            x = mlx_rs::nn::relu(&x)?;
        }

        // Transpose back to [B, T', F]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

        // Project to d_model
        let x = self.out_proj.forward(&x)?;

        // Update lengths
        let mut new_lengths = lengths.clone();
        for _ in &self.conv_layers {
            // Each conv with stride 2: new_len = (len + 1) / 2
            let len_plus_one = &new_lengths + &Array::from_int(1);
            new_lengths = mlx_rs::ops::floor_divide(&len_plus_one, &Array::from_int(2))?;
        }

        Ok((x, new_lengths))
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Array>, _prefix: &str) -> Result<()> {
        // TODO: Implement weight loading for subsampling layers
        Ok(())
    }
}

/// Single Conformer block.
struct ConformerBlock {
    #[allow(dead_code)]
    d_model: usize,

    /// First feed-forward module (half-step)
    ff1: FeedForward,
    ff1_norm: nn::LayerNorm,

    /// Multi-head self-attention
    self_attn: MultiHeadAttention,
    self_attn_norm: nn::LayerNorm,

    /// Convolution module
    conv: ConvolutionModule,
    conv_norm: nn::LayerNorm,

    /// Second feed-forward module (half-step)
    ff2: FeedForward,
    ff2_norm: nn::LayerNorm,

    /// Final layer norm
    final_norm: nn::LayerNorm,
}

impl ConformerBlock {
    fn new(config: &EncoderConfig) -> Result<Self> {
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let ff_dim = (d_model as f32 * config.ff_expansion_factor) as usize;
        let conv_kernel = config.conv_kernel_size;
        let dropout = config.dropout;

        Ok(Self {
            d_model,
            ff1: FeedForward::new(d_model, ff_dim, dropout)?,
            ff1_norm: nn::LayerNorm::new(d_model as i32)?,
            self_attn: MultiHeadAttention::new(d_model, n_heads, dropout)?,
            self_attn_norm: nn::LayerNorm::new(d_model as i32)?,
            conv: ConvolutionModule::new(d_model, conv_kernel, dropout)?,
            conv_norm: nn::LayerNorm::new(d_model as i32)?,
            ff2: FeedForward::new(d_model, ff_dim, dropout)?,
            ff2_norm: nn::LayerNorm::new(d_model as i32)?,
            final_norm: nn::LayerNorm::new(d_model as i32)?,
        })
    }

    fn forward(&mut self, x: &Array, _lengths: &Array) -> Result<Array> {
        // x: [B, T, D]

        // Feed-forward 1 (half-step residual)
        let residual = x.clone();
        let ff1_out = self.ff1_norm.forward(x)?;
        let ff1_out = self.ff1.forward(&ff1_out)?;
        let half = Array::from_f32(0.5);
        let scaled_ff1 = &half * &ff1_out;
        let x = &residual + &scaled_ff1;

        // Self-attention (full residual)
        let residual = x.clone();
        let attn_out = self.self_attn_norm.forward(&x)?;
        let attn_out = self.self_attn.forward(&attn_out)?;
        let x = &residual + &attn_out;

        // Convolution (full residual)
        let residual = x.clone();
        let conv_out = self.conv_norm.forward(&x)?;
        let conv_out = self.conv.forward(&conv_out)?;
        let x = &residual + &conv_out;

        // Feed-forward 2 (half-step residual)
        let residual = x.clone();
        let ff2_out = self.ff2_norm.forward(&x)?;
        let ff2_out = self.ff2.forward(&ff2_out)?;
        let scaled_ff2 = &half * &ff2_out;
        let x = &residual + &scaled_ff2;

        // Final layer norm
        let x = self.final_norm.forward(&x)?;

        Ok(x)
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Array>, _prefix: &str) -> Result<()> {
        // TODO: Implement weight loading for conformer blocks
        Ok(())
    }
}

/// Feed-forward module with expansion.
struct FeedForward {
    linear1: nn::Linear,
    linear2: nn::Linear,
    #[allow(dead_code)]
    dropout: f32,
}

impl FeedForward {
    fn new(d_model: usize, ff_dim: usize, dropout: f32) -> Result<Self> {
        Ok(Self {
            linear1: nn::Linear::new(d_model as i32, ff_dim as i32)?,
            linear2: nn::Linear::new(ff_dim as i32, d_model as i32)?,
            dropout,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let x = self.linear1.forward(x)?;
        let x = mlx_rs::nn::silu(&x)?; // silu is the same as swish
                                       // Skip dropout during inference
        let x = self.linear2.forward(&x)?;
        Ok(x)
    }
}

/// Multi-head self-attention module.
struct MultiHeadAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
}

impl MultiHeadAttention {
    fn new(d_model: usize, n_heads: usize, _dropout: f32) -> Result<Self> {
        let head_dim = d_model / n_heads;

        Ok(Self {
            d_model,
            n_heads,
            head_dim,
            q_proj: nn::Linear::new(d_model as i32, d_model as i32)?,
            k_proj: nn::Linear::new(d_model as i32, d_model as i32)?,
            v_proj: nn::Linear::new(d_model as i32, d_model as i32)?,
            out_proj: nn::Linear::new(d_model as i32, d_model as i32)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let n_heads = self.n_heads as i32;
        let head_dim = self.head_dim as i32;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [B, T, D] -> [B, H, T, D/H]
        let q = q.reshape(&[batch_size, seq_len, n_heads, head_dim])?;
        let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?;

        let k = k.reshape(&[batch_size, seq_len, n_heads, head_dim])?;
        let k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3])?;

        let v = v.reshape(&[batch_size, seq_len, n_heads, head_dim])?;
        let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = Array::from_f32(1.0 / (head_dim as f32).sqrt());
        let k_transposed = mlx_rs::ops::transpose_axes(&k, &[0, 1, 3, 2])?;
        let attn_weights = mlx_rs::ops::matmul(&q, &k_transposed)?;
        let attn_weights = &attn_weights * &scale;
        let attn_weights = mlx_rs::ops::softmax_axis(&attn_weights, -1, None)?;

        // Apply attention to values
        let attn_output = mlx_rs::ops::matmul(&attn_weights, &v)?;

        // Reshape back: [B, H, T, D/H] -> [B, T, D]
        let attn_output = mlx_rs::ops::transpose_axes(&attn_output, &[0, 2, 1, 3])?;
        let attn_output = attn_output.reshape(&[batch_size, seq_len, self.d_model as i32])?;

        // Output projection
        let output = self.out_proj.forward(&attn_output)?;

        Ok(output)
    }
}

/// Convolution module for Conformer.
struct ConvolutionModule {
    d_model: usize,
    pointwise_conv1: nn::Conv1d,
    depthwise_conv: nn::Conv1d,
    pointwise_conv2: nn::Conv1d,
    batch_norm: nn::BatchNorm,
}

impl ConvolutionModule {
    fn new(d_model: usize, kernel_size: usize, _dropout: f32) -> Result<Self> {
        let padding = (kernel_size - 1) / 2;

        Ok(Self {
            d_model,
            // Pointwise expansion (d_model -> 2*d_model)
            pointwise_conv1: nn::Conv1dBuilder::new(d_model as i32, (2 * d_model) as i32, 1)
                .build()?,
            // Depthwise convolution
            depthwise_conv: nn::Conv1dBuilder::new(
                d_model as i32,
                d_model as i32,
                kernel_size as i32,
            )
            .padding(padding as i32)
            .groups(d_model as i32)
            .build()?,
            // Pointwise compression (d_model -> d_model)
            pointwise_conv2: nn::Conv1dBuilder::new(d_model as i32, d_model as i32, 1).build()?,
            batch_norm: nn::BatchNorm::new(d_model as i32)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, D]
        // Transpose to [B, D, T] for conv1d
        let x = mlx_rs::ops::transpose_axes(x, &[0, 2, 1])?;

        // Pointwise conv 1 with GLU activation
        let x = self.pointwise_conv1.forward(&x)?;

        // GLU: split in half along channel dim, apply sigmoid to second half, multiply
        let d_model = self.d_model as i32;
        let batch = x.shape()[0];
        let time = x.shape()[2];
        // Slice first half: x[:, :d_model, :]
        let x1 = x.index((.., ..d_model, ..));
        // Slice second half: x[:, d_model:, :]
        let x2 = x.index((.., d_model.., ..));
        let sigmoid_x2 = mlx_rs::ops::sigmoid(&x2)?;
        let x = &x1 * &sigmoid_x2;

        // Depthwise conv
        let x = self.depthwise_conv.forward(&x)?;
        let x = self.batch_norm.forward(&x)?;
        let x = mlx_rs::nn::silu(&x)?; // silu is the same as swish

        // Pointwise conv 2
        let x = self.pointwise_conv2.forward(&x)?;

        // Transpose back to [B, T, D]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

        Ok(x)
    }
}
