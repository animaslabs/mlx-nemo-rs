//! Jasper/ConvASR encoder implementation.
//!
//! Implements NeMo's ConvASREncoder with JasperBlock sub-modules,
//! used by MarbleNet for Voice Activity Detection.
//!
//! Architecture: sequence of 1D conv blocks with optional depthwise separable
//! convolutions and residual connections.

use std::collections::HashMap;

use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::Array;

use crate::asr::config::{JasperBlockConfig, JasperEncoderConfig};
use crate::core::error::Result;

/// Helper to load Conv1d weights (already transposed from PyTorch by weights.rs).
fn load_conv1d(layer: &mut nn::Conv1d, weights: &HashMap<String, Array>, prefix: &str) {
    if let Some(weight) = weights.get(&format!("{}.weight", prefix)) {
        layer.weight.value = weight.clone();
    }
    if let Some(bias) = weights.get(&format!("{}.bias", prefix)) {
        layer.bias.value = Some(bias.clone());
    }
}

/// Helper to load BatchNorm weights.
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

/// A single sub-block within a JasperBlock.
///
/// For separable convolutions: DepthwiseConv1d -> PointwiseConv1d(1x1) -> BatchNorm
/// For regular convolutions: Conv1d -> BatchNorm
struct SubBlock {
    /// Depthwise conv (only for separable blocks)
    dw_conv: Option<nn::Conv1d>,
    /// Pointwise (1x1) conv for separable, or regular conv for non-separable
    conv: nn::Conv1d,
    /// Batch normalization
    bn: nn::BatchNorm,
}

impl SubBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        separable: bool,
    ) -> Result<Self> {
        let padding = (kernel_size - 1) / 2 * dilation;

        if separable {
            // Depthwise conv: groups = in_channels
            let dw_conv = nn::Conv1dBuilder::new(
                in_channels as i32,
                in_channels as i32,
                kernel_size as i32,
            )
            .padding(padding as i32)
            .stride(stride as i32)
            .dilation(dilation as i32)
            .groups(in_channels as i32)
            .bias(false)
            .build()?;

            // Pointwise conv: 1x1
            let conv = nn::Conv1dBuilder::new(in_channels as i32, out_channels as i32, 1)
                .bias(false)
                .build()?;

            let bn = nn::BatchNorm::new(out_channels as i32)?;

            Ok(Self {
                dw_conv: Some(dw_conv),
                conv,
                bn,
            })
        } else {
            // Regular conv
            let conv = nn::Conv1dBuilder::new(
                in_channels as i32,
                out_channels as i32,
                kernel_size as i32,
            )
            .padding(padding as i32)
            .stride(stride as i32)
            .dilation(dilation as i32)
            .bias(false)
            .build()?;

            let bn = nn::BatchNorm::new(out_channels as i32)?;

            Ok(Self {
                dw_conv: None,
                conv,
                bn,
            })
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut out = x.clone();
        if let Some(ref mut dw) = self.dw_conv {
            out = dw.forward(&out)?;
        }
        out = self.conv.forward(&out)?;
        out = self.bn.forward(&out)?;
        Ok(out)
    }
}

/// Residual connection branch: PointwiseConv1d(1x1) -> BatchNorm
struct ResidualBranch {
    conv: nn::Conv1d,
    bn: nn::BatchNorm,
}

impl ResidualBranch {
    fn new(in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv =
            nn::Conv1dBuilder::new(in_channels as i32, out_channels as i32, 1)
                .bias(false)
                .build()?;
        let bn = nn::BatchNorm::new(out_channels as i32)?;
        Ok(Self { conv, bn })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let out = self.conv.forward(x)?;
        self.bn.forward(&out).map_err(Into::into)
    }
}

/// A Jasper block containing multiple sub-blocks and optional residual connection.
struct JasperBlock {
    sub_blocks: Vec<SubBlock>,
    residual: Option<Vec<ResidualBranch>>,
    out_channels: usize,
    stride: usize,
}

impl JasperBlock {
    fn new(config: &JasperBlockConfig, in_channels: usize) -> Result<Self> {
        let kernel_size = config.kernel[0];
        let stride = config.stride[0];
        let dilation = config.dilation[0];
        let out_channels = config.filters;

        let mut sub_blocks = Vec::with_capacity(config.repeat);
        for i in 0..config.repeat {
            let sub_in = if i == 0 { in_channels } else { out_channels };
            sub_blocks.push(SubBlock::new(
                sub_in,
                out_channels,
                kernel_size,
                if i == 0 { stride } else { 1 },
                dilation,
                config.separable,
            )?);
        }

        let residual = if config.residual {
            // Single residual branch from in_channels to out_channels
            Some(vec![ResidualBranch::new(in_channels, out_channels)?])
        } else {
            None
        };

        Ok(Self {
            sub_blocks,
            residual,
            out_channels,
            stride,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual_input = x.clone();

        // Process sub-blocks: all but last get ReLU after BN
        let n = self.sub_blocks.len();
        let mut out = x.clone();
        for (i, sub) in self.sub_blocks.iter_mut().enumerate() {
            out = sub.forward(&out)?;
            if i < n - 1 {
                out = mlx_rs::ops::maximum(&out, &Array::from_f32(0.0))?; // ReLU
            }
        }

        // Add residual
        if let Some(ref mut res_branches) = self.residual {
            // Handle stride mismatch between residual input and output
            let mut res = residual_input;
            for branch in res_branches.iter_mut() {
                res = branch.forward(&res)?;
            }
            // If stride > 1, the main path output has different time dimension.
            // The residual branch conv is 1x1 with stride=1, so we need to
            // downsample the residual to match.
            if self.stride > 1 {
                // Subsample residual to match output time dimension
                // Take every `stride`-th frame: res[:, ::stride, :]
                let t = res.dim(1) as usize;
                let indices: Vec<i32> = (0..t)
                    .step_by(self.stride)
                    .map(|i| i as i32)
                    .collect();
                let idx = Array::from_slice(&indices, &[indices.len() as i32]);
                res = mlx_rs::ops::indexing::take_axis(&res, &idx, 1)?;
            }
            out = mlx_rs::ops::add(&out, &res)?;
        }

        // Final ReLU
        out = mlx_rs::ops::maximum(&out, &Array::from_f32(0.0))?;

        Ok(out)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        // NeMo mconv index layout for separable blocks:
        // Per sub-block: dw_conv, pw_conv, bn, relu (no params), dropout (no params)
        // = 5 indices per sub-block (except last sub-block has no relu/dropout = 3 indices)
        //
        // For non-separable blocks:
        // Per sub-block: conv, bn, relu (no params), dropout (no params)
        // = 4 indices per sub-block (except last = 2 indices)
        let is_separable = self.sub_blocks.first().map_or(false, |s| s.dw_conv.is_some());
        let n = self.sub_blocks.len();

        let mut mconv_idx = 0;
        for (i, sub) in self.sub_blocks.iter_mut().enumerate() {
            let is_last = i == n - 1;

            if is_separable {
                // dw_conv
                if let Some(ref mut dw) = sub.dw_conv {
                    load_conv1d(dw, weights, &format!("{}.mconv.{}.conv", prefix, mconv_idx));
                    mconv_idx += 1;
                }
                // pw_conv
                load_conv1d(&mut sub.conv, weights, &format!("{}.mconv.{}.conv", prefix, mconv_idx));
                mconv_idx += 1;
                // bn
                load_batch_norm(&mut sub.bn, weights, &format!("{}.mconv.{}", prefix, mconv_idx));
                mconv_idx += 1;
            } else {
                // conv
                load_conv1d(&mut sub.conv, weights, &format!("{}.mconv.{}.conv", prefix, mconv_idx));
                mconv_idx += 1;
                // bn
                load_batch_norm(&mut sub.bn, weights, &format!("{}.mconv.{}", prefix, mconv_idx));
                mconv_idx += 1;
            }

            if !is_last {
                // relu (no params) + dropout (no params)
                mconv_idx += 2;
            }
        }

        // Load residual weights
        if let Some(ref mut res_branches) = self.residual {
            for (res_idx, branch) in res_branches.iter_mut().enumerate() {
                // res.{res_idx}.0 = conv, res.{res_idx}.1 = bn
                load_conv1d(
                    &mut branch.conv,
                    weights,
                    &format!("{}.res.{}.0.conv", prefix, res_idx),
                );
                load_batch_norm(
                    &mut branch.bn,
                    weights,
                    &format!("{}.res.{}.1", prefix, res_idx),
                );
            }
        }
    }
}

/// Jasper/ConvASR encoder.
///
/// Consists of a sequence of JasperBlocks, each containing depthwise separable
/// or regular 1D convolutions with batch normalization.
pub struct JasperEncoder {
    blocks: Vec<JasperBlock>,
    output_dim: usize,
}

impl JasperEncoder {
    /// Create a new JasperEncoder from configuration.
    pub fn new(config: &JasperEncoderConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.jasper.len());
        let mut in_channels = config.feat_in;

        for block_config in &config.jasper {
            let block = JasperBlock::new(block_config, in_channels)?;
            in_channels = block.out_channels;
            blocks.push(block);
        }

        let output_dim = in_channels;

        Ok(Self { blocks, output_dim })
    }

    /// Get the output dimension (number of channels).
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Compute the total stride of the encoder (product of all block strides).
    pub fn total_stride(&self) -> usize {
        self.blocks.iter().map(|b| b.stride).product()
    }

    /// Forward pass through the encoder.
    ///
    /// # Arguments
    /// * `features` - Input features [B, T, n_mels] (MLX Conv1d format: N, L, C)
    /// * `lengths` - Input lengths [B]
    ///
    /// # Returns
    /// * (encoded, encoded_lengths) where encoded is [B, T', output_dim]
    pub fn forward(&mut self, features: &Array, lengths: &Array) -> Result<(Array, Array)> {
        let mut x = features.clone();
        let mut lens = lengths.clone();

        for block in &mut self.blocks {
            x = block.forward(&x)?;
            if block.stride > 1 {
                // Update lengths for stride: new_len = (len - 1) / stride + 1
                let one = Array::from_int(1);
                let stride_arr = Array::from_int(block.stride as i32);
                let subtracted = &lens - &one;
                let divided = mlx_rs::ops::floor_divide(&subtracted, &stride_arr)?;
                lens = &divided + &one;
            }
        }

        Ok((x, lens))
    }

    /// Load weights from a state dict.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_weights(weights, &format!("encoder.encoder.{}", i));
        }
    }
}
