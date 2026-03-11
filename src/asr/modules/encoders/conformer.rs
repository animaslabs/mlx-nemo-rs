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

/// Apply causal padding to a 4D tensor [B, T, F, C].
///
/// This matches NeMo's CausalConv2D padding behavior, which pads BOTH time and
/// frequency dimensions the same way:
/// - left_pad = kernel_size - 1 (for causal behavior on time, and extra preservation on frequency)
/// - right_pad = (kernel_size - 1) // 2 (additional padding for output size preservation)
///
/// This asymmetric padding ensures the output size is: ceil((input + left_pad + right_pad - kernel) / stride) + 1
/// which gives us the same output dimensions as NeMo's CausalConv2D.
fn causal_pad_2d(x: &Array, kernel_size: usize, _stride: usize) -> Result<Array> {
    let batch = x.dim(0) as i32;
    let time = x.dim(1) as i32;
    let freq = x.dim(2) as i32;
    let channels = x.dim(3) as i32;

    // NeMo's CausalConv2D uses the same padding for both time and frequency:
    // left_pad = kernel_size - 1
    // right_pad = (kernel_size - 1) // 2
    let left_pad = (kernel_size - 1) as i32;
    let right_pad = ((kernel_size - 1) / 2) as i32;

    // Step 1: Pad time dimension (axis 1): [B, T, F, C] -> [B, T+left_pad+right_pad, F, C]
    let time_zeros_left = mlx_rs::ops::zeros::<f32>(&[batch, left_pad, freq, channels])?;
    let time_zeros_right = mlx_rs::ops::zeros::<f32>(&[batch, right_pad, freq, channels])?;
    let x = mlx_rs::ops::concatenate_axis(&[&time_zeros_left, x, &time_zeros_right], 1)?;

    // Step 2: Pad frequency dimension (axis 2)
    let new_time = time + left_pad + right_pad;
    let freq_zeros_left = mlx_rs::ops::zeros::<f32>(&[batch, new_time, left_pad, channels])?;
    let freq_zeros_right = mlx_rs::ops::zeros::<f32>(&[batch, new_time, right_pad, channels])?;
    let x = mlx_rs::ops::concatenate_axis(&[&freq_zeros_left, &x, &freq_zeros_right], 2)?;

    Ok(x)
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

        // NOTE: NeMo's ConformerEncoder does NOT have a final_norm at the encoder level.
        // The norm_out is inside each ConformerBlock, applied at the end of the block.
        // So we skip the final_norm here.

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

    // ==================== Streaming Support ====================

    /// Check if this encoder supports streaming inference.
    pub fn supports_streaming(&self) -> bool {
        self.config.is_streaming()
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Get convolution cache size (kernel_size - 1).
    pub fn conv_cache_size(&self) -> usize {
        if !self.layers.is_empty() {
            self.layers[0].conv_cache_size()
        } else {
            0
        }
    }

    /// Setup streaming parameters and update layer cache drop sizes.
    ///
    /// This should be called before streaming inference to configure
    /// the cache drop behavior for attention and convolution layers.
    pub fn setup_streaming(&mut self, cache_drop_size: usize) {
        for layer in &mut self.layers {
            layer.self_attn.cache_drop_size = cache_drop_size;
            layer.conv.cache_drop_size = cache_drop_size;
        }
    }

    /// Forward pass with streaming cache support.
    ///
    /// # Arguments
    /// * `features` - Input features [batch, features, time]
    /// * `lengths` - Sequence lengths [batch]
    /// * `cache_last_channel` - MHA cache [num_layers, batch, cache_len, d_model]
    /// * `cache_last_time` - Conv cache [num_layers, batch, d_model, time_cache]
    /// * `cache_last_channel_len` - Cache fill level [batch]
    ///
    /// # Returns
    /// * Encoded features [batch, d_model, time']
    /// * Output lengths [batch]
    /// * Updated MHA cache
    /// * Updated conv cache
    /// * Updated cache fill level
    pub fn forward_streaming(
        &mut self,
        features: &Array,
        lengths: &Array,
        cache_last_channel: &Array,
        cache_last_time: &Array,
        cache_last_channel_len: &Array,
        drop_extra_pre_encoded: usize,
    ) -> Result<(Array, Array, Array, Array, Array)> {
        let debug = std::env::var("NEMO_DEBUG")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Transpose from [B, F, T] to [B, T, F] for processing
        let x = mlx_rs::ops::transpose_axes(features, &[0, 2, 1])?;

        let input_frames = x.dim(1);

        // Subsampling
        let (mut x, mut lengths) = self.subsampling.forward(&x, lengths)?;

        let subsampled_frames = x.dim(1);

        // Drop extra pre-encoded frames for streaming
        // This is needed because the subsampling convolutions produce extra frames
        // at chunk boundaries that should be discarded to match offline behavior.
        // NeMo applies this whenever cache is provided (even on first chunk when cache is empty).
        // The cache tensor being passed indicates we're in streaming mode.
        if drop_extra_pre_encoded > 0 {
            let drop = drop_extra_pre_encoded as i32;
            x = x.index((.., drop.., ..));
            lengths = mlx_rs::ops::clip(
                &(&lengths - &Array::from_int(drop)),
                (&Array::from_int(0), ()),
            )?;
            if debug {
                eprintln!(
                    "[DEBUG]   encoder: dropped {} extra pre-encoded frames, now {} frames",
                    drop_extra_pre_encoded,
                    x.dim(1)
                );
            }
        }

        if debug {
            eprintln!(
                "[DEBUG]   encoder: input {} frames -> subsampled {} frames (factor {})",
                input_frames, subsampled_frames, self.config.subsampling_factor
            );
        }

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale);
        }

        let seq_len = x.dim(1);
        let cache_size = cache_last_channel.dim(2); // Allocated cache size

        // Get the actual filled cache length (use first batch element, assume all same)
        cache_last_channel_len.eval()?;
        let filled_cache_len = cache_last_channel_len.as_slice::<i64>()[0].max(0) as i32;

        // NeMo uses the ALLOCATED cache size (not filled) for positional encoding.
        // This is because the relative positional encoding needs to cover the maximum
        // possible attention span, regardless of how much cache is currently filled.
        let pos_len = cache_size + seq_len;

        if debug {
            eprintln!(
                "[DEBUG]   encoder: seq_len={}, cache_size={}, filled_cache={}, pos_len={}",
                seq_len, cache_size, filled_cache_len, pos_len
            );
        }

        // Generate positional encoding for cache_size + seq_len (matching NeMo)
        let pos_emb = self.pos_enc.get_encoding(pos_len as usize)?;

        // Initialize output caches
        let mut cache_channel_next = Vec::with_capacity(self.layers.len());
        let mut cache_time_next = Vec::with_capacity(self.layers.len());

        // Process through conformer blocks with cache
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Extract per-layer cache: [batch, cache_size, d_model] and [batch, d_model, time_cache]
            let layer_cache_channel_full = cache_last_channel.index((layer_idx as i32, .., .., ..));
            let layer_cache_time = cache_last_time.index((layer_idx as i32, .., .., ..));

            // Only use the filled portion of the channel cache
            // Cache is stored with newest at the end, so we take the last `filled_cache_len` frames
            let cache_start = (cache_size - filled_cache_len).max(0);
            let layer_cache_channel = if filled_cache_len > 0 {
                layer_cache_channel_full.index((.., cache_start.., ..))
            } else {
                // Empty cache - create empty tensor with right shape
                mlx_rs::ops::zeros::<f32>(&[x.dim(0) as i32, 0, x.dim(2) as i32])?
            };

            let (x_out, new_channel, new_time) = layer.forward_with_cache(
                &x,
                &pos_emb,
                if filled_cache_len > 0 {
                    Some(&layer_cache_channel)
                } else {
                    None
                },
                // Only pass time cache if we have some context (not the first chunk)
                // On first chunk, the conv module should apply its own causal padding
                if filled_cache_len > 0 {
                    Some(&layer_cache_time)
                } else {
                    None
                },
                None, // att_mask
            )?;

            x = x_out;

            // For the output cache, we need to maintain the full cache size
            // The layer returns cache of size [batch, filled_len + seq_len - drop_size, d_model]
            // We need to pad or truncate to cache_size
            if let Some(nc) = new_channel {
                let nc_len = nc.dim(1);
                let padded_cache = if nc_len < cache_size {
                    // Pad with zeros on the left
                    let pad_size = cache_size - nc_len;
                    let zeros =
                        mlx_rs::ops::zeros::<f32>(&[nc.dim(0) as i32, pad_size, nc.dim(2) as i32])?;
                    mlx_rs::ops::concatenate_axis(&[&zeros, &nc], 1)?
                } else if nc_len > cache_size {
                    // Truncate from the left (keep newest)
                    let start = nc_len - cache_size;
                    nc.index((.., start.., ..))
                } else {
                    nc
                };
                cache_channel_next.push(padded_cache);
            }
            if let Some(nt) = new_time {
                cache_time_next.push(nt);
            }
        }

        // Transpose back to [B, D, T]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

        // Stack caches: [num_layers, batch, ...]
        let cache_channel_next = if !cache_channel_next.is_empty() {
            let refs: Vec<&Array> = cache_channel_next.iter().collect();
            mlx_rs::ops::stack_axis(&refs, 0)?
        } else {
            cache_last_channel.clone()
        };

        let cache_time_next = if !cache_time_next.is_empty() {
            let refs: Vec<&Array> = cache_time_next.iter().collect();
            mlx_rs::ops::stack_axis(&refs, 0)?
        } else {
            cache_last_time.clone()
        };

        // Update cache length: clamp(old_len + seq_len, max=cache_size)
        let cache_channel_len_next = mlx_rs::ops::clip(
            &(cache_last_channel_len + &Array::from_int(seq_len as i32)),
            ((), &Array::from_int(cache_size)),
        )?;

        Ok((
            x,
            lengths,
            cache_channel_next,
            cache_time_next,
            cache_channel_len_next,
        ))
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
    #[allow(dead_code)]
    subsampling_factor: usize,
    #[allow(dead_code)]
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

    /// Whether to use causal (left-only) padding for streaming
    is_causal: bool,

    /// Kernel size for manual causal padding
    kernel_size: usize,

    /// Stride for length calculation
    stride: usize,
}

impl Subsampling {
    fn new(config: &EncoderConfig) -> Result<Self> {
        let feat_in = config.feat_in;
        let d_model = config.d_model;
        let subsampling_factor = config.subsampling_factor;
        let conv_channels = config.subsampling_conv_channels.unwrap_or(256);
        let is_causal = config.is_causal_downsampling();

        // Number of striding operations: log2(subsampling_factor)
        let n_strides = match subsampling_factor {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 2,
        };

        let kernel_size: usize = 3;
        let stride: usize = 2;

        // For causal: no padding in conv (we'll pad manually with asymmetric padding)
        // For non-causal: symmetric padding
        let padding = if is_causal { 0 } else { 1 }; // (kernel_size - 1) / 2

        // First conv: 1 input channel (mono spectrogram) -> conv_channels
        let first_conv = nn::Conv2dBuilder::new(1, conv_channels as i32, kernel_size as i32)
            .stride(stride as i32)
            .padding(padding)
            .build()?;

        // Depthwise + pointwise pairs for remaining strides
        let mut depthwise_convs = Vec::new();
        let mut pointwise_convs = Vec::new();

        for _ in 0..(n_strides - 1) {
            // Depthwise conv: groups = channels for depthwise convolution
            let dw_conv = nn::Conv2dBuilder::new(
                conv_channels as i32,
                conv_channels as i32,
                kernel_size as i32,
            )
            .stride(stride as i32)
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
        // We need to compute the actual conv output size, not just feat_in / subsampling_factor
        // Conv output: floor((input + 2*padding - kernel) / stride) + 1
        // For causal mode with our padding: we add (kernel-1) left and (kernel-1)/2 symmetric on freq
        // so freq padding is effectively (kernel-1)/2 = 1 for kernel=3
        // NeMo's causal subsampling uses ceil-mode-like rounding to preserve dimensions
        let freq_padding = (kernel_size - 1) / 2; // Always symmetric for frequency dimension
        let feat_out = Self::compute_freq_output_size(
            feat_in,
            n_strides,
            kernel_size,
            stride,
            freq_padding,
            is_causal,
        );
        let linear_in = conv_channels * feat_out;

        let out_proj = nn::Linear::new(linear_in as i32, d_model as i32)?;

        Ok(Self {
            subsampling_factor,
            conv_channels,
            first_conv,
            depthwise_convs,
            pointwise_convs,
            out_proj,
            is_causal,
            kernel_size,
            stride,
        })
    }

    /// Compute the frequency dimension output size after subsampling convolutions.
    ///
    /// For NeMo's CausalConv2D with kernel=3, stride=2, the padding is:
    /// - left_pad = kernel - 1 = 2
    /// - right_pad = (kernel - 1) / 2 = 1
    ///
    /// Output size = floor((input + left_pad + right_pad - kernel) / stride) + 1
    ///             = floor((input + 2 + 1 - 3) / 2) + 1
    ///             = floor(input / 2) + 1
    fn compute_freq_output_size(
        feat_in: usize,
        n_strides: usize,
        kernel_size: usize,
        stride: usize,
        _freq_padding: usize,
        is_causal: bool,
    ) -> usize {
        let mut f = feat_in;
        for _ in 0..n_strides {
            if is_causal {
                // CausalConv2D: output = floor(input / stride) + 1
                // This comes from: floor((input + (k-1) + (k-1)/2 - k) / stride) + 1
                //                = floor((input + k - 1 + floor((k-1)/2) - k) / stride) + 1
                // For k=3: floor((input + 2 + 1 - 3) / 2) + 1 = floor(input / 2) + 1
                f = f / stride + 1;
            } else {
                // Standard conv with symmetric padding: floor((input + 2*pad - kernel) / stride) + 1
                let pad = (kernel_size - 1) / 2;
                f = (f + 2 * pad - kernel_size) / stride + 1;
            }
        }
        f
    }

    fn forward(&mut self, x: &Array, lengths: &Array) -> Result<(Array, Array)> {
        // x: [B, T, F] where F is number of mel features (e.g., 80 or 128)
        // Add channel dimension: [B, T, F] -> [B, T, F, 1] for MLX Conv2d
        // MLX Conv2d expects (N, H, W, C) format
        let mut x = x.reshape(&[-1, x.dim(1) as i32, x.dim(2) as i32, 1])?;

        // Apply causal padding if needed before first conv
        if self.is_causal {
            x = causal_pad_2d(&x, self.kernel_size, self.stride)?;
        }

        // First conv layer with ReLU
        x = self.first_conv.forward(&x)?;
        x = mlx_rs::nn::relu(&x)?;

        // Get parameters for loop since we need to avoid borrow issues
        let is_causal = self.is_causal;
        let kernel_size = self.kernel_size;
        let stride = self.stride;

        // Apply depthwise + pointwise pairs with ReLU
        for (dw_conv, pw_conv) in self
            .depthwise_convs
            .iter_mut()
            .zip(self.pointwise_convs.iter_mut())
        {
            // Apply causal padding before each depthwise conv
            if is_causal {
                x = causal_pad_2d(&x, kernel_size, stride)?;
            }
            x = dw_conv.forward(&x)?;
            x = pw_conv.forward(&x)?;
            x = mlx_rs::nn::relu(&x)?;
        }

        // x is now [B, T', F', C] where T' = T/subsampling_factor, F' = F/subsampling_factor
        // In MLX (N, H, W, C), H=T', W=F', C=conv_channels
        // NeMo does: [B, C, T, F] -> permute to [B, T, C, F] -> flatten to [B, T, C*F]
        // We have: [B, T', F', C] -> transpose to [B, T', C, F'] -> flatten to [B, T', C*F']
        let batch = x.dim(0);
        let time = x.dim(1);

        // Transpose to [B, T', C, F'] to match NeMo's C*F flattening order
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 1, 3, 2])?;

        // Flatten to [B, T', C*F']
        let x = x.reshape(&[batch as i32, time as i32, -1])?;

        // Project to d_model: [B, T', d_model]
        let x = self.out_proj.forward(&x)?;

        // Update lengths based on padding mode
        let mut new_lengths = lengths.clone();
        let n_convs = self.depthwise_convs.len() + 1;

        if is_causal {
            // For CausalConv2D: output = floor(input / stride) + 1
            // This is equivalent to: (input + stride - 1) / stride when we want ceiling behavior
            // Actually, it's just floor(input / stride) + 1
            // So we compute: input // 2 + 1 for each conv
            for _ in 0..n_convs {
                new_lengths = &mlx_rs::ops::floor_divide(&new_lengths, &Array::from_int(2))?
                    + &Array::from_int(1);
            }
        } else {
            // Standard: each stride just halves the length (with symmetric padding preserving dims)
            for _ in 0..n_convs {
                new_lengths = mlx_rs::ops::floor_divide(&new_lengths, &Array::from_int(2))?;
            }
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

        // Check if causal convolution is requested
        let is_causal = config.is_causal_conv();

        Ok(Self {
            d_model,
            ff1: FeedForward::new(d_model, ff_dim, dropout)?,
            ff1_norm: nn::LayerNorm::new(d_model as i32)?,
            self_attn: RelPositionMultiHeadAttention::new(d_model, n_heads)?,
            self_attn_norm: nn::LayerNorm::new(d_model as i32)?,
            conv: ConvolutionModule::new(
                d_model,
                conv_kernel,
                dropout,
                &config.conv_norm_type,
                is_causal,
            )?,
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

    /// Forward pass with cache support for streaming inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `pos_emb` - Positional embeddings
    /// * `cache_last_channel` - Optional MHA cache [batch, cache_len, d_model]
    /// * `cache_last_time` - Optional conv cache [batch, d_model, time_cache]
    /// * `att_mask` - Optional attention mask
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated MHA cache
    /// * Updated conv cache
    fn forward_with_cache(
        &mut self,
        x: &Array,
        pos_emb: &Array,
        cache_last_channel: Option<&Array>,
        cache_last_time: Option<&Array>,
        att_mask: Option<&Array>,
    ) -> Result<(Array, Option<Array>, Option<Array>)> {
        // Feed-forward 1 (half-step residual)
        let residual = x.clone();
        let ff1_out = self.ff1_norm.forward(x)?;
        let ff1_out = self.ff1.forward(&ff1_out)?;
        let half = Array::from_f32(0.5);
        let scaled_ff1 = &half * &ff1_out;
        let x = &residual + &scaled_ff1;

        // Self-attention with cache
        let residual = x.clone();
        let attn_out = self.self_attn_norm.forward(&x)?;
        let (attn_out, new_cache_channel) =
            self.self_attn
                .forward_with_cache(&attn_out, pos_emb, cache_last_channel, att_mask)?;
        let x = &residual + &attn_out;

        // Convolution with cache
        let residual = x.clone();
        let conv_out = self.conv_norm.forward(&x)?;
        let (conv_out, new_cache_time) =
            self.conv.forward_with_cache(&conv_out, cache_last_time)?;
        let x = &residual + &conv_out;

        // Feed-forward 2 (half-step residual)
        let residual = x.clone();
        let ff2_out = self.ff2_norm.forward(&x)?;
        let ff2_out = self.ff2.forward(&ff2_out)?;
        let scaled_ff2 = &half * &ff2_out;
        let x = &residual + &scaled_ff2;

        // Final layer norm
        let x = self.final_norm.forward(&x)?;

        Ok((x, new_cache_channel, new_cache_time))
    }

    /// Get the convolution cache size for this block.
    fn conv_cache_size(&self) -> usize {
        self.conv.cache_size()
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

/// Normalization type for convolution module.
enum ConvNorm {
    BatchNorm(nn::BatchNorm),
    LayerNorm(nn::LayerNorm),
}

impl ConvNorm {
    fn forward(&mut self, x: &Array) -> mlx_rs::error::Result<Array> {
        match self {
            ConvNorm::BatchNorm(bn) => bn.forward(x),
            ConvNorm::LayerNorm(ln) => ln.forward(x),
        }
    }
}

/// Convolution module for Conformer.
struct ConvolutionModule {
    d_model: usize,
    pointwise_conv1: nn::Conv1d,
    depthwise_conv: nn::Conv1d,
    pointwise_conv2: nn::Conv1d,
    norm: ConvNorm,
    /// Whether this is a causal convolution (for streaming)
    is_causal: bool,
    /// Kernel size for causal padding calculation
    kernel_size: usize,
    /// Cache drop size for streaming
    pub cache_drop_size: usize,
}

impl ConvolutionModule {
    fn new(
        d_model: usize,
        kernel_size: usize,
        _dropout: f32,
        norm_type: &str,
        is_causal: bool,
    ) -> Result<Self> {
        // For non-causal: symmetric padding (kernel_size - 1) / 2 on each side
        // For causal: all padding on the left (kernel_size - 1), none on right
        // With causal, we'll handle padding manually in forward()
        let padding = if is_causal {
            0 // We'll pad manually in forward
        } else {
            (kernel_size - 1) / 2
        };

        // Create normalization layer based on type
        let norm = match norm_type {
            "layer_norm" => ConvNorm::LayerNorm(nn::LayerNorm::new(d_model as i32)?),
            _ => ConvNorm::BatchNorm(nn::BatchNorm::new(d_model as i32)?),
        };

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
            norm,
            is_causal,
            kernel_size,
            cache_drop_size: 0,
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

        // Apply causal padding if needed (pad left only)
        let x = if self.is_causal {
            // Pad on the left (time dimension) with zeros
            // x is [B, T, D], we need to pad T by (kernel_size - 1) on the left
            let pad_size = self.kernel_size - 1;
            let batch = x.dim(0);
            let d = x.dim(2);

            // Create zero padding: [B, pad_size, D]
            let zeros = mlx_rs::ops::zeros::<f32>(&[batch as i32, pad_size as i32, d as i32])?;
            mlx_rs::ops::concatenate_axis(&[&zeros, &x], 1)?
        } else {
            x
        };

        // Depthwise conv - x is [B, T, D] (or [B, T+pad, D] if causal)
        let x = self.depthwise_conv.forward(&x)?;
        let x = self.norm.forward(&x)?;
        let x = mlx_rs::nn::silu(&x)?; // silu is the same as swish

        // Pointwise conv 2
        let x = self.pointwise_conv2.forward(&x)?;

        Ok(x)
    }

    /// Forward pass with cache support for streaming inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `cache` - Optional cache tensor [batch, d_model, cache_len] for left context
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated cache tensor [batch, d_model, new_cache_len] (if cache was provided)
    fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        // Pointwise conv 1 with GLU activation
        let x = self.pointwise_conv1.forward(x)?;

        // GLU: split in half along channel dim
        let d_model = self.d_model as i32;
        let x1 = x.index((.., .., ..d_model));
        let x2 = x.index((.., .., d_model..));
        let sigmoid_x2 = mlx_rs::ops::sigmoid(&x2)?;
        let x = &x1 * &sigmoid_x2;

        // For streaming: use cache instead of padding
        let (x, new_cache) = if let Some(cache_tensor) = cache {
            // cache: [batch, d_model, cache_len]
            // x after GLU: [batch, seq_len, d_model]
            // Need to transpose x to [batch, d_model, seq_len] for concatenation
            let x_transposed = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

            // Concatenate: [batch, d_model, cache_len + seq_len]
            let x_with_cache = mlx_rs::ops::concatenate_axis(&[cache_tensor, &x_transposed], 2)?;

            // Update cache: keep the last (kernel_size - 1) frames
            let cache_size = self.kernel_size - 1;
            let total_len = x_with_cache.dim(2) as usize;
            let cache_start = total_len.saturating_sub(cache_size);
            let new_cache = x_with_cache.index((.., .., cache_start as i32..));

            // Transpose back to [batch, total_len, d_model] for conv
            let x_for_conv = mlx_rs::ops::transpose_axes(&x_with_cache, &[0, 2, 1])?;

            (x_for_conv, Some(new_cache))
        } else if self.is_causal {
            // No cache but causal: pad on the left
            let pad_size = self.kernel_size - 1;
            let batch = x.dim(0);
            let d = x.dim(2);
            let zeros = mlx_rs::ops::zeros::<f32>(&[batch as i32, pad_size as i32, d as i32])?;
            let x_padded = mlx_rs::ops::concatenate_axis(&[&zeros, &x], 1)?;
            (x_padded, None)
        } else {
            (x, None)
        };

        // Depthwise conv (no additional padding needed when using cache)
        let x = self.depthwise_conv.forward(&x)?;
        let x = self.norm.forward(&x)?;
        let x = mlx_rs::nn::silu(&x)?;

        // Pointwise conv 2
        let x = self.pointwise_conv2.forward(&x)?;

        Ok((x, new_cache))
    }

    /// Get the cache size needed for this convolution module.
    fn cache_size(&self) -> usize {
        self.kernel_size - 1
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

        // Load normalization weights based on type
        match &mut self.norm {
            ConvNorm::BatchNorm(bn) => {
                load_batch_norm(bn, weights, &format!("{}.batch_norm", prefix));
            }
            ConvNorm::LayerNorm(ln) => {
                load_layer_norm(ln, weights, &format!("{}.batch_norm", prefix));
            }
        }
        Ok(())
    }
}
