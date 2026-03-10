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

use super::attention::{RelPositionMultiHeadAttention, RelPositionalEncoding};
use crate::asr::config::EncoderConfig;
use crate::core::error::Result;

/// Helper to load LayerNorm weights
fn load_layer_norm(layer: &mut nn::LayerNorm, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        layer.weight.value = Some(weight.clone());
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
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

/// Helper to load Conv1d weights
///
/// PyTorch Conv1d weight shape: (out_channels, in_channels/groups, kernel_size)
/// MLX Conv1d weight shape: (out_channels, kernel_size, in_channels/groups)
/// So we need to transpose axes 1 and 2.
fn load_conv1d(layer: &mut nn::Conv1d, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        // Transpose from PyTorch (O, I, K) to MLX (O, K, I)
        if let Ok(transposed) = weight.transpose_axes(&[0, 2, 1]) {
            layer.weight.value = transposed;
        }
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
    }
}

/// Helper to load BatchNorm weights
fn load_batch_norm(layer: &mut nn::BatchNorm, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        layer.weight.value = Some(weight.clone());
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
    }
    if let Some(running_mean) = weights.get(&format!("{}.running_mean", prefix)) {
        layer.running_mean.value = Some(running_mean.clone());
    }
    if let Some(running_var) = weights.get(&format!("{}.running_var", prefix)) {
        layer.running_var.value = Some(running_var.clone());
    }
}

/// Conformer encoder for ASR.
///
/// This implements the Conformer architecture as used in NeMo's ASR models,
/// including FastConformer variants with depthwise striding subsampling.
pub struct ConformerEncoder {
    config: EncoderConfig,

    /// Input projection / subsampling layers
    subsampling: Subsampling,

    /// Relative positional encoding generator
    pos_enc: RelPositionalEncoding,

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
        let pos_emb_max_len = config.pos_emb_max_len;
        let xscaling = config.xscaling.unwrap_or(true);

        // Create subsampling layer
        let subsampling = Subsampling::new(&config)?;

        // Create relative positional encoding
        let pos_enc = RelPositionalEncoding::new(d_model, pos_emb_max_len, xscaling)?;

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
            pos_enc,
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

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale);
        }

        // Generate positional encoding for this sequence length
        let seq_len = x.dim(1) as usize;
        let pos_emb = self.pos_enc.get_encoding(seq_len)?;

        // Process through conformer blocks with positional info
        for layer in &mut self.layers {
            x = layer.forward(&x, &pos_emb)?;
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
        // Load subsampling weights (NeMo uses encoder.pre_encode prefix)
        self.subsampling
            .load_weights(weights, "encoder.pre_encode")?;

        // Load conformer block weights (NeMo uses encoder.layers.N prefix)
        // Position biases are per-layer (not shared), loaded in ConformerBlock
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("encoder.layers.{}", i))?;
        }

        // Load final norm weights
        load_layer_norm(&mut self.final_norm, weights, "encoder.final_norm");

        Ok(())
    }
}

/// Helper to load Conv2d weights
///
/// PyTorch Conv2d weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
/// MLX Conv2d weight shape: (out_channels, kernel_h, kernel_w, in_channels/groups)
/// So we need to transpose axes 1,2,3 -> 2,3,1
fn load_conv2d(layer: &mut nn::Conv2d, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        // Transpose from PyTorch (O, I, H, W) to MLX (O, H, W, I)
        if let Ok(transposed) = weight.transpose_axes(&[0, 2, 3, 1]) {
            layer.weight.value = transposed;
        }
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
    }
}

/// Subsampling layer for reducing sequence length using depthwise striding.
///
/// This implements the `dw_striding` subsampling method used in NeMo's Conformer encoder.
/// It uses 2D convolutions to downsample the mel spectrogram in both time and frequency.
struct Subsampling {
    subsampling_factor: usize,
    conv_channels: usize,

    /// First conv: regular Conv2d(1, conv_channels, 3x3, stride=2)
    first_conv: nn::Conv2d,

    /// Depthwise convolutions (one for each additional striding level)
    /// Conv2d(conv_channels, conv_channels, 3x3, stride=2, groups=conv_channels)
    depthwise_convs: Vec<nn::Conv2d>,

    /// Pointwise convolutions after each depthwise conv
    /// Conv2d(conv_channels, conv_channels, 1x1)
    pointwise_convs: Vec<nn::Conv2d>,

    /// Output projection: Linear(conv_channels * reduced_feat, d_model)
    out_proj: nn::Linear,
}

impl Subsampling {
    fn new(config: &EncoderConfig) -> Result<Self> {
        let feat_in = config.feat_in;
        let d_model = config.d_model;
        let subsampling_factor = config.subsampling_factor;
        let conv_channels = config.subsampling_conv_channels.unwrap_or(256);

        // Number of striding operations: log2(subsampling_factor)
        let n_strides = match subsampling_factor {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 2,
        };

        let kernel_size = 3;
        let padding = 1; // (kernel_size - 1) / 2

        // First conv: 1 input channel (mono spectrogram) -> conv_channels
        let first_conv = nn::Conv2dBuilder::new(1, conv_channels as i32, kernel_size)
            .stride(2)
            .padding(padding)
            .build()?;

        // Depthwise + pointwise pairs for remaining strides
        let mut depthwise_convs = Vec::new();
        let mut pointwise_convs = Vec::new();

        for _ in 0..(n_strides - 1) {
            // Depthwise conv: groups = channels for depthwise convolution
            let dw_conv =
                nn::Conv2dBuilder::new(conv_channels as i32, conv_channels as i32, kernel_size)
                    .stride(2)
                    .padding(padding)
                    .groups(conv_channels as i32)
                    .build()?;
            depthwise_convs.push(dw_conv);

            // Pointwise conv: 1x1 convolution to mix channels
            let pw_conv =
                nn::Conv2dBuilder::new(conv_channels as i32, conv_channels as i32, 1).build()?;
            pointwise_convs.push(pw_conv);
        }

        // Calculate output feature dimension after all striding
        // Each stride halves the dimension: feat_out = feat_in / subsampling_factor
        let feat_out = feat_in / subsampling_factor;
        let linear_in = conv_channels * feat_out;

        let out_proj = nn::Linear::new(linear_in as i32, d_model as i32)?;

        Ok(Self {
            subsampling_factor,
            conv_channels,
            first_conv,
            depthwise_convs,
            pointwise_convs,
            out_proj,
        })
    }

    fn forward(&mut self, x: &Array, lengths: &Array) -> Result<(Array, Array)> {
        // x: [B, T, F] where F is number of mel features (e.g., 80)
        // Add channel dimension: [B, T, F] -> [B, T, F, 1] for MLX Conv2d
        // MLX Conv2d expects (N, H, W, C) format
        let x = x.reshape(&[-1, x.dim(1) as i32, x.dim(2) as i32, 1])?;

        // First conv layer with ReLU
        let mut x = self.first_conv.forward(&x)?;
        x = mlx_rs::nn::relu(&x)?;

        // Apply depthwise + pointwise pairs with ReLU
        for (dw_conv, pw_conv) in self
            .depthwise_convs
            .iter_mut()
            .zip(self.pointwise_convs.iter_mut())
        {
            x = dw_conv.forward(&x)?;
            x = pw_conv.forward(&x)?;
            x = mlx_rs::nn::relu(&x)?;
        }

        // x is now [B, T', F', C] where T' = T/subsampling_factor, F' = F/subsampling_factor
        // Flatten frequency and channel dimensions: [B, T', F' * C]
        let batch = x.dim(0);
        let time = x.dim(1);
        let x = x.reshape(&[batch as i32, time as i32, -1])?;

        // Project to d_model: [B, T', d_model]
        let x = self.out_proj.forward(&x)?;

        // Update lengths: each stride halves the length
        let mut new_lengths = lengths.clone();
        for _ in 0..self.depthwise_convs.len() + 1 {
            new_lengths = mlx_rs::ops::floor_divide(&new_lengths, &Array::from_int(2))?;
        }

        Ok((x, new_lengths))
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        // Load first conv: encoder.pre_encode.conv.0
        load_conv2d(&mut self.first_conv, weights, &format!("{}.conv.0", prefix));

        // Load depthwise and pointwise conv pairs
        // NeMo layout: conv.0 (first), activation, conv.2 (dw), conv.3 (pw), activation, ...
        // Layer indices: 0=first, 1=activation, 2=dw1, 3=pw1, 4=activation, 5=dw2, 6=pw2, ...
        for (i, (dw_conv, pw_conv)) in self
            .depthwise_convs
            .iter_mut()
            .zip(self.pointwise_convs.iter_mut())
            .enumerate()
        {
            let dw_idx = 2 + i * 3; // 2, 5, 8, ...
            let pw_idx = 3 + i * 3; // 3, 6, 9, ...
            load_conv2d(dw_conv, weights, &format!("{}.conv.{}", prefix, dw_idx));
            load_conv2d(pw_conv, weights, &format!("{}.conv.{}", prefix, pw_idx));
        }

        // Load output projection
        let proj_prefix = format!("{}.out", prefix);
        if let Some(weight) = weights.get(&format!("{}.weight", proj_prefix)) {
            self.out_proj.weight.value = weight.clone();
        }
        if let Some(bias) = weights.get(&format!("{}.bias", proj_prefix)) {
            self.out_proj.bias.value = Some(bias.clone());
        }

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

    /// Multi-head self-attention with relative positional encoding
    self_attn: RelPositionMultiHeadAttention,
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
            self_attn: RelPositionMultiHeadAttention::new(d_model, n_heads)?,
            self_attn_norm: nn::LayerNorm::new(d_model as i32)?,
            conv: ConvolutionModule::new(d_model, conv_kernel, dropout)?,
            conv_norm: nn::LayerNorm::new(d_model as i32)?,
            ff2: FeedForward::new(d_model, ff_dim, dropout)?,
            ff2_norm: nn::LayerNorm::new(d_model as i32)?,
            final_norm: nn::LayerNorm::new(d_model as i32)?,
        })
    }

    fn forward(&mut self, x: &Array, pos_emb: &Array) -> Result<Array> {
        // x: [B, T, D]

        // Feed-forward 1 (half-step residual)
        let residual = x.clone();
        let ff1_out = self.ff1_norm.forward(x)?;
        let ff1_out = self.ff1.forward(&ff1_out)?;
        let half = Array::from_f32(0.5);
        let scaled_ff1 = &half * &ff1_out;
        let x = &residual + &scaled_ff1;

        // Self-attention with relative positional encoding (full residual)
        let residual = x.clone();
        let attn_out = self.self_attn_norm.forward(&x)?;
        let attn_out = self.self_attn.forward(&attn_out, pos_emb)?;
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

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        // Load feed-forward 1 (NeMo uses "feed_forward1" not "ff1")
        self.ff1
            .load_weights(weights, &format!("{}.feed_forward1", prefix))?;
        load_layer_norm(
            &mut self.ff1_norm,
            weights,
            &format!("{}.norm_feed_forward1", prefix),
        );

        // Load self-attention
        self.self_attn
            .load_weights(weights, &format!("{}.self_attn", prefix))?;
        load_layer_norm(
            &mut self.self_attn_norm,
            weights,
            &format!("{}.norm_self_att", prefix),
        );

        // Load convolution module
        self.conv
            .load_weights(weights, &format!("{}.conv", prefix))?;
        load_layer_norm(
            &mut self.conv_norm,
            weights,
            &format!("{}.norm_conv", prefix),
        );

        // Load feed-forward 2 (NeMo uses "feed_forward2" not "ff2")
        self.ff2
            .load_weights(weights, &format!("{}.feed_forward2", prefix))?;
        load_layer_norm(
            &mut self.ff2_norm,
            weights,
            &format!("{}.norm_feed_forward2", prefix),
        );

        // Load final norm (NeMo uses "norm_out")
        load_layer_norm(
            &mut self.final_norm,
            weights,
            &format!("{}.norm_out", prefix),
        );

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

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        load_linear(&mut self.linear1, weights, &format!("{}.linear1", prefix));
        load_linear(&mut self.linear2, weights, &format!("{}.linear2", prefix));
        Ok(())
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

        // Create batch norm - keep in training mode for now
        // TODO: Investigate why eval mode produces empty output
        let batch_norm = nn::BatchNorm::new(d_model as i32)?;

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
            batch_norm,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, D] - MLX Conv1d expects (N, L, C_in) format, no transpose needed

        // Pointwise conv 1 with GLU activation
        // Output: [B, T, 2*D]
        let x = self.pointwise_conv1.forward(x)?;

        // GLU: split in half along channel dim (axis 2), apply sigmoid to second half, multiply
        let d_model = self.d_model as i32;
        // Slice first half: x[:, :, :d_model]
        let x1 = x.index((.., .., ..d_model));
        // Slice second half: x[:, :, d_model:]
        let x2 = x.index((.., .., d_model..));
        let sigmoid_x2 = mlx_rs::ops::sigmoid(&x2)?;
        let x = &x1 * &sigmoid_x2;

        // Depthwise conv - x is [B, T, D]
        let x = self.depthwise_conv.forward(&x)?;
        let x = self.batch_norm.forward(&x)?;
        let x = mlx_rs::nn::silu(&x)?; // silu is the same as swish

        // Pointwise conv 2
        let x = self.pointwise_conv2.forward(&x)?;

        Ok(x)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        load_conv1d(
            &mut self.pointwise_conv1,
            weights,
            &format!("{}.pointwise_conv1", prefix),
        );
        load_conv1d(
            &mut self.depthwise_conv,
            weights,
            &format!("{}.depthwise_conv", prefix),
        );
        load_conv1d(
            &mut self.pointwise_conv2,
            weights,
            &format!("{}.pointwise_conv2", prefix),
        );
        load_batch_norm(
            &mut self.batch_norm,
            weights,
            &format!("{}.batch_norm", prefix),
        );
        Ok(())
    }
}
