//! Conformer encoder implementation.
//!
//! The Conformer architecture combines convolution and self-attention
//! for effective speech feature encoding.

use std::collections::HashMap;
use std::sync::LazyLock;

use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;

use super::attention::{
    AttentionContext, RelPositionMultiHeadAttention, RelPositionalEncoding, key_padding_mask,
};
use crate::asr::config::{AttContextStyle, ConvNormType, EncoderConfig};
use crate::asr::modules::speaker_kernel::{SpeakerKernels, SpeakerTargets};
use crate::core::ArrayExt;
use crate::core::error::Result;
use crate::core::loading::{Weights, load_batch_norm, load_conv1d, load_conv2d, load_layer_norm};
use crate::core::quantization::{LinearLayer, QuantizationConfig};

/// Debug switches, read from the environment once rather than on every call
static DEBUG_MEL_MASK: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("DEBUG_MEL_MASK").is_some());
static MLX_VOICE_DEBUG: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("MLX_VOICE_DEBUG").is_some());
static DEBUG_LAYER0: LazyLock<bool> = LazyLock::new(|| std::env::var_os("DEBUG_LAYER0").is_some());
static DEBUG_CONV: LazyLock<bool> = LazyLock::new(|| std::env::var_os("DEBUG_CONV").is_some());


/// Debug output from encoder forward pass.
#[derive(Debug)]
pub struct DebugEncoderOutput {
    /// Output after subsampling [B, T, D]
    pub after_subsampling: Array,
    /// Output after xscaling [B, T, D]
    pub after_scaling: Array,
    /// Positional embeddings [1, 2*T-1, D]
    pub pos_emb: Array,
    /// Outputs after each layer (first few) [B, T, D]
    pub layer_outputs: Vec<Array>,
    /// Final encoder output [B, D, T]
    pub final_output: Array,
    /// Output lengths
    pub lengths: Array,
}

/// Apply causal padding to a 4D tensor [B, T, F, C].
///
/// This matches NeMo's CausalConv2D padding behavior, which pads BOTH time and
/// frequency dimensions the same way:
/// - left_pad = kernel_size - 1 (causal padding for past context)
/// - right_pad = stride - 1 (additional padding for proper output size)
///
/// NeMo's CausalConv2D uses F.pad with format (left, right, top, bottom):
///   F.pad(x, pad=(left_pad, right_pad, left_pad, right_pad))
///
/// For kernel=3, stride=2:
/// - left_pad = 3 - 1 = 2
/// - right_pad = 2 - 1 = 1
///
/// This gives output size = floor((input + 2 + 1 - 3) / 2) + 1 = floor(input / 2) + 1
fn causal_pad_2d(x: &Array, kernel_size: usize, stride: usize) -> Result<Array> {
    // NeMo's CausalConv2D padding formula:
    // left_pad = kernel_size - 1
    // right_pad = stride - 1
    let left_pad = (kernel_size - 1) as i32;
    let right_pad = (stride - 1) as i32;

    // Pad time (axis 1) and frequency (axis 2) the same way: [B, T, F, C]
    Ok(mlx_rs::ops::pad(
        x,
        &[(0, 0), (left_pad, right_pad), (left_pad, right_pad), (0, 0)],
        None,
        None,
    )?)
}

/// Create attention mask for streaming inference.
///
/// This implements NeMo's `_create_masks()` function for 'chunked_limited' style
/// attention context windowing and offset-based masking for unfilled cache positions.
///
/// # Arguments
/// * `att_context_size` - (left_context, right_context) in frames
/// * `att_context_style` - "chunked_limited" or "regular"
/// * `max_audio_length` - Total length: filled_cache + seq_len
/// * `cache_len` - Full cache buffer size (for slicing the mask)
/// * `filled_cache_len` - How many cache frames are actually valid
/// * `seq_len` - Current input sequence length
///
/// # Returns
/// Boolean mask [1, seq_len, max_audio_length] where True = masked (ignore)
fn create_attention_mask(
    att_context_size: (i32, i32),
    att_context_style: &str,
    max_audio_length: i32,
    cache_len: i32,
    filled_cache_len: i32,
    tensor_seq_len: i32,
    valid_seq_len: i32,
) -> Result<Array> {
    tracing::trace!(
        ?att_context_size,
        style = att_context_style,
        max_audio_length,
        cache_len,
        filled_cache_len,
        tensor_seq_len,
        valid_seq_len,
        "creating streaming attention mask"
    );

    // Create base attention mask [1, max_audio_length, max_audio_length]
    // In NeMo, this starts as all True (all positions visible),
    // then gets masked based on context style
    let mut att_mask = mlx_rs::ops::ones::<bool>(&[1, max_audio_length, max_audio_length])?;

    let (left_context, right_context) = att_context_size;

    if att_context_style == "chunked_limited" {
        // Chunked limited masking (NeMo conformer_encoder.py lines 803-822)
        if right_context == -1 {
            // Unlimited right context - only apply left context via triu
            if left_context >= 0 {
                att_mask = apply_triu(&att_mask, -left_context)?;
            }
        } else {
            // Chunked limited: chunk_size = right_context + 1
            let chunk_size = right_context + 1;
            let left_chunks_num = if left_context >= 0 {
                left_context / chunk_size
            } else {
                10000 // effectively unlimited
            };

            // Create chunk-based visibility mask
            att_mask = create_chunked_limited_mask(max_audio_length, chunk_size, left_chunks_num)?;
        }
    } else {
        // Regular style - simple triangular masking
        if left_context >= 0 {
            att_mask = apply_triu(&att_mask, -left_context)?;
        }
        if right_context >= 0 {
            att_mask = apply_tril(&att_mask, right_context)?;
        }
    }

    // Apply offset masking for unfilled cache positions (NeMo lines 831-835)
    // offset = cache_len - filled_cache_len (how many zeros at start of cache)
    let offset = cache_len - filled_cache_len;

    tracing::trace!(
        offset,
        cache_len,
        filled_cache_len,
        "mask offset calculation"
    );
    if offset < 0 {
        tracing::warn!("Negative offset detected! This indicates cache overflow.");
    }

    if offset > 0 {
        // Create pad_mask: positions >= offset are valid
        // First create range [0, 1, 2, ..., max_audio_length-1]
        let range_vals: Vec<i32> = (0..max_audio_length).collect();
        let range = Array::from_slice(&range_vals, &[1, max_audio_length]);

        // pad_mask[i] = (range[i] >= offset)
        let offset_arr = Array::from_int(offset);
        let valid_positions = range.ge(&offset_arr)?;

        // Broadcast to [1, max_audio_length, max_audio_length] and AND with att_mask
        // For each query position, keys must be at valid positions
        let valid_positions = valid_positions.reshape(&[1, 1, max_audio_length])?;
        att_mask = att_mask.logical_and(&valid_positions)?;
    }

    // Slice to get [1, seq_len, max_audio_length]
    // This removes the cache rows, keeping only query rows for the current input
    // NeMo: att_mask = att_mask[:, cache_len:]
    //
    // With max_audio_length = cache_size + seq_len:
    // - Positions 0..cache_len are cache positions
    // - Positions cache_len..max_audio_length are current input positions
    // We want the mask for queries at positions cache_len..max_audio_length
    let mut att_mask = att_mask.index((.., cache_len.., ..));

    // Mask out positions beyond valid_seq_len
    // Python NeMo uses the `length` parameter to mask extra tensor frames.
    // Query positions >= valid_seq_len should be fully masked (can't attend to anything).
    // This zeros out the attention output for padding frames.
    if valid_seq_len < tensor_seq_len {
        tracing::trace!(
            valid_seq_len,
            tensor_seq_len,
            "masking query positions beyond valid length"
        );

        // Create row mask: [tensor_seq_len] where True for rows 0..valid_seq_len
        let row_indices: Vec<i32> = (0..tensor_seq_len).collect();
        let row_indices = Array::from_slice(&row_indices, &[tensor_seq_len]);
        let valid_len_arr = Array::from_int(valid_seq_len);
        let row_valid = row_indices.lt(&valid_len_arr)?;

        // Reshape to [1, tensor_seq_len, 1] for broadcasting
        let row_valid = row_valid.reshape(&[1, tensor_seq_len, 1])?;

        // AND with attention mask: rows >= valid_seq_len become all False
        att_mask = att_mask.logical_and(&row_valid)?;
    }

    // No inversion needed: mask is True where attention is ALLOWED
    // The attention layer will use: where(mask, scores, -inf) to keep scores where mask=True

    Ok(att_mask)
}

/// Apply upper triangular masking (triu) to attention mask.
///
/// Sets positions below the diagonal to False.
/// diagonal=0 means main diagonal, negative shifts down.
fn apply_triu(mask: &Array, diagonal: i32) -> Result<Array> {
    let shape = mask.shape();
    let rows = shape[1];
    let cols = shape[2];

    // Create row indices [0, 1, 2, ...]
    let row_indices: Vec<i32> = (0..rows).collect();
    let row_arr = Array::from_slice(&row_indices, &[rows, 1]);

    // Create col indices [0, 1, 2, ...]
    let col_indices: Vec<i32> = (0..cols).collect();
    let col_arr = Array::from_slice(&col_indices, &[1, cols]);

    // Upper triangular: col >= row + diagonal
    // Equivalent to: col - row >= diagonal
    let diff = &col_arr - &row_arr;
    let diag_arr = Array::from_int(diagonal);
    let triu_mask = diff.ge(&diag_arr)?;

    // Reshape to [1, rows, cols] and AND with mask
    let triu_mask = triu_mask.reshape(&[1, rows, cols])?;
    Ok(mask.logical_and(&triu_mask)?)
}

/// Apply lower triangular masking (tril) to attention mask.
///
/// Sets positions above the diagonal to False.
/// diagonal=0 means main diagonal, positive shifts up.
fn apply_tril(mask: &Array, diagonal: i32) -> Result<Array> {
    let shape = mask.shape();
    let rows = shape[1];
    let cols = shape[2];

    // Create row indices [0, 1, 2, ...]
    let row_indices: Vec<i32> = (0..rows).collect();
    let row_arr = Array::from_slice(&row_indices, &[rows, 1]);

    // Create col indices [0, 1, 2, ...]
    let col_indices: Vec<i32> = (0..cols).collect();
    let col_arr = Array::from_slice(&col_indices, &[1, cols]);

    // Lower triangular: col <= row + diagonal
    // Equivalent to: col - row <= diagonal
    let diff = &col_arr - &row_arr;
    let diag_arr = Array::from_int(diagonal);
    let tril_mask = diff.le(&diag_arr)?;

    // Reshape to [1, rows, cols] and AND with mask
    let tril_mask = tril_mask.reshape(&[1, rows, cols])?;
    Ok(mask.logical_and(&tril_mask)?)
}

/// Create chunked limited attention mask.
///
/// Each position can only attend to positions within its own chunk
/// and a number of preceding chunks.
///
/// NeMo implementation (conformer_encoder.py lines 816-822):
/// ```python
/// chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=device)
/// chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
/// diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
/// chunked_limited_mask = torch.logical_and(
///     torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
/// )
/// ```
fn create_chunked_limited_mask(
    max_audio_length: i32,
    chunk_size: i32,
    left_chunks_num: i32,
) -> Result<Array> {
    // Create position indices and compute chunk indices
    let indices: Vec<i32> = (0..max_audio_length).collect();
    let chunk_idx: Vec<i32> = indices.iter().map(|&i| i / chunk_size).collect();

    // Create 2D diff_chunks: diff_chunks[i, j] = chunk_idx[i] - chunk_idx[j]
    // This is query_chunk - key_chunk
    let mut diff_chunks = Vec::with_capacity((max_audio_length * max_audio_length) as usize);
    for &qi in &chunk_idx {
        for &ki in &chunk_idx {
            diff_chunks.push(qi - ki);
        }
    }
    let diff_chunks = Array::from_slice(&diff_chunks, &[max_audio_length, max_audio_length]);

    // Mask: 0 <= diff_chunks <= left_chunks_num
    // i.e., query can attend to key if key is in same chunk or up to left_chunks_num chunks back
    let zero = Array::from_int(0);
    let left_limit = Array::from_int(left_chunks_num);

    let ge_zero = diff_chunks.ge(&zero)?;
    let le_left = diff_chunks.le(&left_limit)?;
    let mask = ge_zero.logical_and(&le_left)?;

    // Reshape to [1, max_audio_length, max_audio_length]
    Ok(mask.reshape(&[1, max_audio_length, max_audio_length])?)
}

/// Apply speaker kernels before `layer_idx`, if the model has them and targets are set.
fn apply_speaker_kernels(
    kernels: Option<&SpeakerKernels>,
    targets: Option<&SpeakerTargets>,
    layer_idx: usize,
    x: Array,
) -> Result<Array> {
    match (kernels, targets) {
        (Some(kernels), Some(targets)) if kernels.applies_to(layer_idx) => {
            kernels.apply(layer_idx, &x, targets)
        }
        _ => Ok(x),
    }
}

/// Conformer encoder for ASR.
///
/// This implements the Conformer architecture as used in NeMo's ASR models,
/// including FastConformer variants with depthwise striding subsampling.
#[derive(Debug)]
pub struct ConformerEncoder {
    config: EncoderConfig,

    /// Input projection / subsampling layers
    subsampling: Subsampling,

    /// Relative positional encoding generator
    pos_enc: RelPositionalEncoding,

    /// Conformer blocks
    layers: Vec<ConformerBlock>,

    /// Attention context size (left, right) for streaming
    /// Parsed from config.att_context_size
    att_context_size: Option<(i32, i32)>,

    /// Attention context style ("regular" or "chunked_limited")
    att_context_style: AttContextStyle,

    /// Speaker kernels (multitalker ASR models only)
    speaker_kernels: Option<SpeakerKernels>,

    /// Speaker targets used by the speaker kernels on the next forward pass
    speaker_targets: Option<SpeakerTargets>,

    /// Dtype of the weights and activations (see `set_dtype`)
    dtype: mlx_rs::Dtype,

    /// Streaming attention masks by (filled cache, frames, valid frames):
    /// `None` when every position is visible, else an additive f32 mask, or a
    /// boolean one (true = visible) when some query rows see nothing
    streaming_masks: HashMap<(i32, i32, i32), Option<Array>>,
}

impl ConformerEncoder {
    /// Create a new Conformer encoder from configuration.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;

        let d_model = config.d_model;
        let n_layers = config.n_layers;
        let pos_emb_max_len = config.pos_emb_max_len;
        let xscaling = config.xscaling;

        // Create subsampling layer
        let subsampling = Subsampling::new(&config)?;

        // Create relative positional encoding
        let pos_enc = RelPositionalEncoding::new(d_model, pos_emb_max_len, xscaling)?;

        // Create conformer blocks
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(ConformerBlock::new(&config)?);
        }

        // Extract attention context configuration for streaming
        let att_context_size = config
            .att_context_size
            .as_ref()
            .and_then(|cs| cs.default_mode());
        let att_context_style = config.att_context_style.clone();

        Ok(Self {
            config,
            subsampling,
            pos_enc,
            layers,
            att_context_size,
            att_context_style,
            speaker_kernels: None,
            speaker_targets: None,
            dtype: mlx_rs::Dtype::Float32,
            streaming_masks: HashMap::new(),
        })
    }

    /// Run the encoder's weights and activations in `dtype` (e.g. `Dtype::Bfloat16`).
    ///
    /// For the offline [`forward`](Self::forward) pass: inputs are cast to
    /// `dtype` and the output back to float32. The streaming caches stay float32.
    pub fn set_dtype(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        self.subsampling.set_dtype(dtype)?;
        self.pos_enc.set_dtype(dtype)?;
        for layer in &mut self.layers {
            layer.set_dtype(dtype)?;
        }
        self.dtype = dtype;
        Ok(())
    }

    /// Set the streaming mode's attention context size.
    ///
    /// This should be called after loading the model and before streaming inference
    /// to select the appropriate context size for the desired latency mode.
    pub fn set_streaming_mode(&mut self, mode_idx: usize) {
        if let Some(ctx_size) = self
            .config
            .att_context_size
            .as_ref()
            .and_then(|cs| cs.mode(mode_idx))
        {
            tracing::debug!(
                mode_idx,
                left = ctx_size.0,
                right = ctx_size.1,
                "Setting encoder streaming mode"
            );
            if self.att_context_size != Some(ctx_size) {
                self.att_context_size = Some(ctx_size);
                // Cached masks were built for the previous context
                self.streaming_masks.clear();
            }
        } else {
            tracing::warn!(
                mode_idx,
                "Invalid streaming mode index, keeping current context"
            );
        }
    }

    /// Forward pass through the encoder.
    ///
    /// # Arguments
    /// * `features` - Input features [batch, features, time]
    /// * `lengths` - Sequence lengths `[batch]`
    ///
    /// # Returns
    /// * Encoded features [batch, d_model, time']
    /// * Output lengths `[batch]`
    pub fn forward(&mut self, features: &Array, lengths: &Array) -> Result<(Array, Array)> {
        // Transpose from [B, F, T] to [B, T, F] for processing
        let mut x = mlx_rs::ops::transpose_axes(features, &[0, 2, 1])?;

        // Mask MEL frames beyond lengths to zero (NeMo's preprocessor does this)
        // This is important because padding frames can have non-zero values from center-padded STFT.
        // We mask BEFORE subsampling so the convolution produces natural values at padding positions
        // (small but non-zero from the receptive field extending into valid frames).
        let mel_time = x.dim(1) as i32;
        let mel_indices_data: Vec<i32> = (0..mel_time).collect();
        let mel_indices = mlx_rs::Array::from_slice(&mel_indices_data, &[1, mel_time]);
        let mel_lengths_expanded = lengths.reshape(&[-1, 1])?;
        let mel_mask = mlx_rs::ops::ge(&mel_indices, &mel_lengths_expanded)?;
        let mel_mask = mel_mask.reshape(&[x.dim(0) as i32, mel_time, 1])?;

        let debug_mask = *DEBUG_MEL_MASK;
        if debug_mask {
            x.eval()?;
            mel_mask.eval()?;
            let mask_flat: Vec<bool> = mel_mask.to_vec_row_major()?;
            let num_masked = mask_flat.iter().filter(|&&b| b).count();
            let x_flat = x.reshape(&[-1])?;
            x_flat.eval()?;
            let x_slice: Vec<f32> = x_flat.to_vec_row_major()?;
            let n_features = x.dim(2) as usize;
            let last_frame_offset = (mel_time as usize - 1) * n_features;
            let before_mask: Vec<f32> = x_slice[last_frame_offset..last_frame_offset + 5].to_vec();
            tracing::debug!(
                x_shape = ?x.shape(),
                mel_time,
                lengths = ?lengths.to_vec_row_major::<i32>()?,
                num_masked,
                ?before_mask,
                "MEL masking before"
            );
        }

        x = mlx_rs::ops::select(&mel_mask, mlx_rs::Array::from_f32(0.0), &x)?;

        if debug_mask {
            x.eval()?;
            let x_flat = x.reshape(&[-1])?;
            x_flat.eval()?;
            let x_slice: Vec<f32> = x_flat.to_vec_row_major()?;
            let n_features = x.dim(2) as usize;
            let last_frame_offset = (mel_time as usize - 1) * n_features;
            let after_mask: Vec<f32> = x_slice[last_frame_offset..last_frame_offset + 5].to_vec();
            tracing::debug!(?after_mask, "MEL masking after");
        }

        // Subsampling
        let x = x.as_dtype(self.dtype)?;
        let (mut x, lengths) = self.subsampling.forward(&x, lengths)?;

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale).as_dtype(x.dtype())?;
        }

        // Generate positional encoding for this sequence length
        let seq_len = x.dim(1) as usize;
        let pos_emb = self.pos_enc.get_encoding(seq_len)?;

        // Process through conformer blocks with positional info
        let context = self.offline_attention_context();
        let padding = PaddingMask::new(&lengths, seq_len as i32)?;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            x = apply_speaker_kernels(
                self.speaker_kernels.as_ref(),
                self.speaker_targets.as_ref(),
                layer_idx,
                x,
            )?;
            x = layer.forward(&x, &pos_emb, context, padding.as_ref())?;
        }

        // NOTE: NeMo's ConformerEncoder does NOT have a final_norm at the encoder level.
        // The norm_out is inside each ConformerBlock, applied at the end of the block.
        // So we skip the final_norm here.

        // Transpose back to [B, D, T]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?.as_dtype(mlx_rs::Dtype::Float32)?;

        Ok((x, lengths))
    }

    /// The attention context offline passes use: NeMo masks whole-file attention
    /// to the default `att_context_size` (the first configured mode), so
    /// cache-aware models see the same context offline as when streaming in
    /// that mode. Streaming in another mode doesn't change it. `None` (full
    /// attention) for [-1, -1] models such as parakeet.
    fn offline_attention_context(&self) -> Option<AttentionContext> {
        self.config
            .att_context_size
            .as_ref()
            .and_then(|cs| cs.default_mode())
            .and_then(|size| AttentionContext::new(size, self.att_context_style.as_str()))
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
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &Weights,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // Load subsampling weights (NeMo uses encoder.pre_encode prefix)
        // Conv layers are not quantized, but output projection may be
        self.subsampling
            .load_weights(weights, "encoder.pre_encode", quant)?;

        // Load conformer block weights (NeMo uses encoder.layers.N prefix)
        // Position biases are per-layer (not shared), loaded in ConformerBlock
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("encoder.layers.{}", i), quant)?;
        }

        // Speaker kernels live at the model root (spk_kernels.*, bg_spk_kernels.*)
        self.speaker_kernels = SpeakerKernels::load(weights, quant)?;

        Ok(())
    }

    /// Whether this encoder has speaker kernels (a multitalker ASR model).
    pub fn has_speaker_kernels(&self) -> bool {
        self.speaker_kernels.is_some()
    }

    /// Set the speaker targets used by the speaker kernels.
    ///
    /// Masks are [batch, time] at encoder frame rate and apply to subsequent
    /// forward passes until cleared. Ignored by models without speaker kernels.
    pub fn set_speaker_targets(&mut self, targets: SpeakerTargets) {
        self.speaker_targets = Some(targets);
    }

    /// Clear the speaker targets.
    pub fn clear_speaker_targets(&mut self) {
        self.speaker_targets = None;
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

    /// Get the attention context style.
    pub fn att_context_style(&self) -> &AttContextStyle {
        &self.att_context_style
    }

    /// Check if this is a chunked_limited style encoder.
    pub fn is_chunked_limited(&self) -> bool {
        self.att_context_style == AttContextStyle::ChunkedLimited
    }

    /// Forward pass through subsampling layer only.
    ///
    /// This is useful for debugging to compare subsampling outputs with Python.
    ///
    /// # Arguments
    /// * `features` - Input features [batch, time, features]
    /// * `lengths` - Sequence lengths `[batch]`
    ///
    /// # Returns
    /// * Subsampled features [batch, time', d_model]
    /// * Output lengths `[batch]`
    pub fn forward_subsampling_only(
        &mut self,
        features: &Array,
        lengths: &Array,
    ) -> Result<(Array, Array)> {
        self.subsampling.forward(features, lengths)
    }

    /// Forward pass starting from already-subsampled embeddings.
    ///
    /// This skips the subsampling stage and runs only the conformer blocks.
    /// Used by SortFormer's streaming mode where embeddings from multiple sources
    /// (speaker cache, FIFO, chunk) are concatenated before running the conformer.
    ///
    /// # Arguments
    /// * `embeddings` - Pre-subsampled embeddings [batch, time, d_model]
    /// * `lengths` - Valid sequence lengths `[batch]`
    ///
    /// # Returns
    /// * Encoded features [batch, d_model, time]
    /// * Output lengths `[batch]`
    pub fn forward_from_embeddings(
        &mut self,
        embeddings: &Array,
        lengths: &Array,
    ) -> Result<(Array, Array)> {
        let mut x = embeddings.clone();

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale).as_dtype(x.dtype())?;
        }

        // Generate positional encoding for this sequence length
        let seq_len = x.dim(1) as usize;
        let pos_emb = self.pos_enc.get_encoding(seq_len)?;

        // Process through conformer blocks
        let context = self.offline_attention_context();
        let padding = PaddingMask::new(lengths, seq_len as i32)?;
        for layer in &mut self.layers {
            x = layer.forward(&x, &pos_emb, context, padding.as_ref())?;
        }

        // Transpose to [B, D, T]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

        Ok((x, lengths.clone()))
    }

    /// Get positional encoding for a given sequence length (for debugging).
    pub fn get_pos_encoding(&self, seq_len: usize) -> Result<Array> {
        self.pos_enc.get_encoding(seq_len)
    }

    /// Get the xscale factor (for debugging).
    pub fn xscale(&self) -> f32 {
        self.pos_enc.xscale()
    }

    /// Forward pass with debug output.
    ///
    /// Returns intermediate values for comparison with Python.
    pub fn forward_debug(
        &mut self,
        features: &Array,
        lengths: &Array,
    ) -> Result<DebugEncoderOutput> {
        // Transpose from [B, F, T] to [B, T, F] for processing
        let mut x = mlx_rs::ops::transpose_axes(features, &[0, 2, 1])?;

        // Mask MEL frames beyond lengths to zero (NeMo's preprocessor does this)
        let mel_time = x.dim(1) as i32;
        let mel_indices_data: Vec<i32> = (0..mel_time).collect();
        let mel_indices = mlx_rs::Array::from_slice(&mel_indices_data, &[1, mel_time]);
        let mel_lengths_expanded = lengths.reshape(&[-1, 1])?;
        let mel_mask = mlx_rs::ops::ge(&mel_indices, &mel_lengths_expanded)?;
        let mel_mask = mel_mask.reshape(&[x.dim(0) as i32, mel_time, 1])?;
        x = mlx_rs::ops::select(&mel_mask, mlx_rs::Array::from_f32(0.0), &x)?;

        // Subsampling
        let (mut x, lengths) = self.subsampling.forward(&x, lengths)?;
        x.eval()?;
        let after_subsampling = x.clone();

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale).as_dtype(x.dtype())?;
        }
        x.eval()?;
        let after_scaling = x.clone();

        // Generate positional encoding
        let seq_len = x.dim(1) as usize;
        let pos_emb = self.pos_enc.get_encoding(seq_len)?;
        pos_emb.eval()?;

        // Process through conformer blocks
        let context = self.offline_attention_context();
        let padding = PaddingMask::new(&lengths, seq_len as i32)?;
        let mut layer_outputs = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, &pos_emb, context, padding.as_ref())?;
            x.eval()?;
            if i < 3 {
                // Only save first few layers
                layer_outputs.push(x.clone());
            }
        }

        // Transpose back to [B, D, T]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;
        x.eval()?;

        Ok(DebugEncoderOutput {
            after_subsampling,
            after_scaling,
            pos_emb,
            layer_outputs,
            final_output: x,
            lengths,
        })
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
    /// This implements NeMo's streaming forward with:
    /// 1. Attention mask creation based on att_context_size and att_context_style
    /// 2. Full cache buffer passed to layers (not just filled portion)
    /// 3. Proper cache update with sliding window
    ///
    /// # Arguments
    /// * `features` - Input features [batch, features, time]
    /// * `lengths` - Sequence lengths `[batch]`
    /// * `cache_last_channel` - MHA cache [num_layers, batch, cache_len, d_model]
    /// * `cache_last_time` - Conv cache [num_layers, batch, d_model, time_cache]
    /// * `cache_last_channel_len` - Cache fill level `[batch]`
    /// * `drop_extra_pre_encoded` - Frames to drop after subsampling
    ///
    /// # Returns
    /// * Encoded features [batch, d_model, time']
    /// * Output lengths `[batch]`
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
        // Transpose from [B, F, T] to [B, T, F] for processing
        let x = mlx_rs::ops::transpose_axes(features, &[0, 2, 1])?;

        // Lengths and the cache fill level are host arrays (this function
        // returns them that way too), so reading them doesn't wait on the GPU.
        let input_lengths: Vec<i32> = lengths.as_dtype(mlx_rs::Dtype::Int32)?.to_vec_row_major()?;
        let filled_cache_lens: Vec<i64> = cache_last_channel_len
            .as_dtype(mlx_rs::Dtype::Int64)?
            .to_vec_row_major()?;

        // Subsampling
        let mut x = self
            .subsampling
            .forward_frames_chunked(&x, SUBSAMPLING_CHUNK_FRAMES)?;

        // Drop the frames the pre-encode cache produced (NeMo also shortens
        // `length` by the same amount)
        if drop_extra_pre_encoded > 0 {
            x = x.index((.., drop_extra_pre_encoded as i32.., ..));
        }
        let out_lengths: Vec<i32> = input_lengths
            .iter()
            .map(|&len| {
                let len = self.subsampling.output_length(len.max(0) as usize);
                len.saturating_sub(drop_extra_pre_encoded) as i32
            })
            .collect();

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale).as_dtype(x.dtype())?;
        }

        // The mask is shared across the batch, so it uses the first stream's lengths
        let tensor_seq_len = x.dim(1) as i32;
        let valid_seq_len = out_lengths[0].min(tensor_seq_len);
        let cache_size = cache_last_channel.dim(2);
        let filled_cache_len = filled_cache_lens[0].clamp(0, cache_size as i64) as i32;

        // Keys/values cover the full cache buffer plus the current frames
        let total_len = cache_size + tensor_seq_len;
        let pos_emb = self.pos_enc.get_encoding(total_len as usize)?;

        // NeMo: att_mask = _create_masks(...); att_mask = att_mask[:, cache_len:]
        let att_mask = match self.att_context_size {
            Some(ctx) => {
                let key = (filled_cache_len, tensor_seq_len, valid_seq_len);
                if !self.streaming_masks.contains_key(&key) {
                    let mask = create_attention_mask(
                        ctx,
                        self.att_context_style.as_str(),
                        total_len,
                        cache_size,
                        filled_cache_len,
                        tensor_seq_len,
                        valid_seq_len,
                    )?;
                    // Skip masking entirely once nothing is hidden. Otherwise, when
                    // every query row sees some key (rows only go blank past the
                    // valid length), pass an additive mask the fused attention
                    // kernel can take.
                    let all_visible = mlx_rs::ops::all(&mask, None)?.item_exact::<bool>();
                    let mask = if all_visible {
                        None
                    } else if valid_seq_len < tensor_seq_len {
                        Some(mask)
                    } else {
                        let hidden = Array::from_f32(f32::NEG_INFINITY);
                        let additive = mlx_rs::ops::select(&mask, Array::from_f32(0.0), &hidden)?;
                        additive.eval()?;
                        Some(additive)
                    };
                    self.streaming_masks.insert(key, mask);
                }
                self.streaming_masks[&key].clone()
            }
            None => None,
        };

        // Create padding mask for conv module: True where position >= valid_seq_len
        // This zeros out padding frames after GLU, matching Python NeMo behavior
        let pad_mask = if tensor_seq_len > valid_seq_len {
            // Create indices [0, 1, 2, ..., tensor_seq_len - 1]
            let indices = mlx_rs::ops::arange::<_, i32>(0, tensor_seq_len, 1)?;
            // mask[i] = (i >= valid_seq_len) => True for padding positions
            let valid_len_arr = Array::from_int(valid_seq_len);
            let mask = mlx_rs::ops::ge(&indices, &valid_len_arr)?;
            // Expand to [1, tensor_seq_len] for batch broadcasting
            let mask = mask.reshape(&[1, tensor_seq_len])?;
            Some(mask)
        } else {
            None
        };

        // Initialize output caches
        let mut cache_channel_next = Vec::with_capacity(self.layers.len());
        let mut cache_time_next = Vec::with_capacity(self.layers.len());

        // Process through conformer blocks with cache
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            x = apply_speaker_kernels(
                self.speaker_kernels.as_ref(),
                self.speaker_targets.as_ref(),
                layer_idx,
                x,
            )?;

            // Extract per-layer cache: [batch, cache_size, d_model] and [batch, d_model, time_cache]
            let layer_cache_channel_full = cache_last_channel.index((layer_idx as i32, .., .., ..));
            let layer_cache_time = cache_last_time.index((layer_idx as i32, .., .., ..));

            // Pass full cache buffer to layer, along with filled length
            // The attention layer will extract the filled portion internally
            // Always pass conv cache, even on first chunk with zeros.
            // Python's NeMo passes zeros cache which triggers cache path (concat zeros + input),
            // not the no-cache path (pad input). This ensures consistent cache population.
            let (x_out, new_channel, new_time) = layer.forward_with_cache(
                &x,
                &pos_emb,
                Some(&layer_cache_channel_full),
                filled_cache_len,
                Some(&layer_cache_time), // Always pass conv cache
                att_mask.as_ref(),
                pad_mask.as_ref(),
            )?;

            x = x_out;

            // Update cache: the attention layer returns new_cache from sliding window
            // We need to store it properly in the full cache buffer
            //
            // The attention returns: new_cache = concat(old_filled[q_keep_size:], x[:q_keep_size])
            // We need to place this at the END of the full cache buffer
            if let Some(nc) = new_channel {
                let nc_len = nc.dim(1);
                let updated_cache = if nc_len < cache_size {
                    // Pad with zeros at the beginning to maintain cache_size
                    let pad_size = cache_size - nc_len;
                    let batch = nc.dim(0);
                    let d_model = nc.dim(2);
                    let zeros = mlx_rs::ops::zeros::<f32>(&[batch, pad_size, d_model])?;
                    mlx_rs::ops::concatenate(&[&zeros, &nc], 1)?
                } else if nc_len > cache_size {
                    // Truncate from the left (keep newest)
                    let start = nc_len - cache_size;
                    nc.index((.., start.., ..))
                } else {
                    nc
                };
                cache_channel_next.push(updated_cache);
            }
            if let Some(nt) = new_time {
                cache_time_next.push(nt);
            }
        }

        // Debug encoder output stats (before final transpose)
        // Transpose back to [B, D, T]
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

        // Stack caches: [num_layers, batch, ...]
        let cache_channel_next = if !cache_channel_next.is_empty() {
            let refs: Vec<&Array> = cache_channel_next.iter().collect();
            mlx_rs::ops::stack(&refs, 0)?
        } else {
            cache_last_channel.clone()
        };

        let cache_time_next = if !cache_time_next.is_empty() {
            let refs: Vec<&Array> = cache_time_next.iter().collect();
            mlx_rs::ops::stack(&refs, 0)?
        } else {
            cache_last_time.clone()
        };

        // NeMo: clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
        // with cache_keep_size = frames - cache_drop_size
        let keep = (tensor_seq_len as i64
            - self
                .layers
                .first()
                .map_or(0, |l| l.self_attn.cache_drop_size) as i64)
            .max(0);
        let next_lens: Vec<i64> = filled_cache_lens
            .iter()
            .map(|&len| (len + keep).min(cache_size as i64))
            .collect();
        let cache_channel_len_next = Array::from_slice(&next_lens, &[next_lens.len() as i32]);
        let lengths = Array::from_slice(&out_lengths, &[out_lengths.len() as i32]);

        Ok((
            x,
            lengths,
            cache_channel_next,
            cache_time_next,
            cache_channel_len_next,
        ))
    }
}

/// Cast every parameter of an `nn` module to `dtype`.
fn cast_module(module: &mut impl ModuleParameters, dtype: mlx_rs::Dtype) -> Result<()> {
    for (_, param) in module.parameters_mut().flatten() {
        *param = param.as_dtype(dtype)?;
    }
    Ok(())
}

/// Mel frames per subsampling chunk for long inputs (~41 s at 10 ms hops).
/// Shorter inputs run in one pass.
const SUBSAMPLING_CHUNK_FRAMES: usize = 4096;

/// Subsampling layer for reducing sequence length using depthwise striding.
///
/// This implements the `dw_striding` subsampling method used in NeMo's Conformer encoder.
/// It uses 2D convolutions to downsample the mel spectrogram in both time and frequency.
#[derive(Debug)]
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
    out_proj: LinearLayer,

    /// Whether to use causal (left-only) padding for streaming
    is_causal: bool,

    /// Kernel size for manual causal padding
    kernel_size: usize,

    /// Stride for length calculation
    stride: usize,
}

impl Subsampling {
    fn set_dtype(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        cast_module(&mut self.first_conv, dtype)?;
        for conv in self
            .depthwise_convs
            .iter_mut()
            .chain(self.pointwise_convs.iter_mut())
        {
            cast_module(conv, dtype)?;
        }
        self.out_proj.cast(dtype)
    }

    #[tracing::instrument(skip(config), fields(feat_in = config.feat_in, d_model = config.d_model, subsampling_factor = config.subsampling_factor))]
    fn new(config: &EncoderConfig) -> Result<Self> {
        let feat_in = config.feat_in;
        let d_model = config.d_model;
        let subsampling_factor = config.subsampling_factor;
        let conv_channels = config.subsampling_conv_channels.unwrap_or(256);
        let is_causal = config.causal_downsampling;

        // Number of striding operations: log2(subsampling_factor)
        let n_strides = match subsampling_factor {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 2,
        };

        // Kernel size is 3x3 and stride is 2 for FastConformer
        let kernel_size: usize = 3;
        let stride: usize = 2;

        // First conv: 1 input channel (mono spectrogram) -> conv_channels
        // For causal: padding=0 (manual causal padding applied before each conv)
        // For non-causal: padding=1 (symmetric padding to preserve frequency dimension)
        let conv_padding = if is_causal {
            0
        } else {
            (kernel_size / 2) as i32
        };
        let first_conv = nn::Conv2dBuilder::new(1, conv_channels as i32, kernel_size as i32)
            .stride(stride as i32)
            .padding(conv_padding)
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
            .padding(conv_padding)
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

        // Create placeholder LinearLayer - weights will be loaded later
        let out_proj = LinearLayer::standard(
            mlx_rs::ops::zeros::<f32>(&[d_model as i32, linear_in as i32])?,
            Some(mlx_rs::ops::zeros::<f32>(&[d_model as i32])?),
        );

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
    /// NeMo's CausalConv2D applies the same asymmetric padding to BOTH dimensions:
    /// - left_pad = kernel - 1 = 2 (for k=3)
    /// - right_pad = (kernel - 1) // 2 = 1 (for k=3)
    /// - total padding = 3
    ///
    /// Output size = floor((input + left_pad + right_pad - kernel) / stride) + 1
    ///             = floor((input + 2 + 1 - 3) / 2) + 1
    ///             = floor(input / 2) + 1
    ///
    /// Example for 128 mel bins with 3 stages (causal mode):
    ///   128 -> floor(128/2) + 1 = 65
    ///   65  -> floor(65/2) + 1 = 33
    ///   33  -> floor(33/2) + 1 = 17
    ///
    /// Note: This differs from the documentation which suggests symmetric padding
    /// for frequency, but matches NeMo's actual CausalConv2D implementation.
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

    /// Debug helper to print array statistics
    #[allow(dead_code)]
    fn debug_print_array(name: &str, arr: &Array, show_samples: usize) {
        if arr.eval().is_err() {
            tracing::info!("DEBUG subsampling {}: failed to eval", name);
            return;
        }

        let shape = arr.shape();
        tracing::info!("DEBUG subsampling {}: shape={:?}", name, shape);

        if arr.size() == 0 {
            tracing::info!("  {} is empty", name);
            return;
        }

        // Get data as f32
        let Ok(data) = arr.to_vec_row_major::<f32>() else {
            tracing::info!("  {} could not be read", name);
            return;
        };
        let n = data.len();

        // Calculate stats
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = data.iter().sum::<f32>() / n as f32;

        let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();

        tracing::info!(
            "  min={:.6} max={:.6} mean={:.6} std={:.6}",
            min,
            max,
            mean,
            std
        );

        // First few samples
        let samples: usize = show_samples.min(n);
        let first: Vec<String> = data[..samples]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect();
        tracing::info!("  first {}: [{}]", samples, first.join(", "));

        // Last few samples
        if n > samples {
            let last: Vec<String> = data[n - samples..]
                .iter()
                .map(|x| format!("{:.6}", x))
                .collect();
            tracing::info!("  last {}: [{}]", samples, last.join(", "));
        }
    }

    /// Calculate the output length of one causal subsampling conv.
    ///
    /// NeMo's calc_length with CausalConv2D padding (left `k - 1`, right `s - 1`):
    /// `floor((input + (k - 1) + (s - 1) - k) / s) + 1`, elementwise. The kernel
    /// size cancels out.
    fn calc_conv_output_size(input_size: &Array, stride: i32) -> Result<Array> {
        let padded = input_size + Array::from_int(stride - 2);
        let divided = mlx_rs::ops::floor_divide(&padded, Array::from_int(stride))?;
        Ok(divided + Array::from_int(1))
    }

    /// Output length after subsampling for an input of `frames` mel frames.
    ///
    /// Host-side twin of [`Self::output_lengths`].
    fn output_length(&self, frames: usize) -> usize {
        let n_convs = self.depthwise_convs.len() + 1;
        let (kernel, stride) = (self.kernel_size as i64, self.stride as i64);
        let mut len = frames as i64;
        for _ in 0..n_convs {
            len = if self.is_causal {
                (len + (kernel - 1) + (stride - 1) - kernel).div_euclid(stride) + 1
            } else {
                (len - 1).div_euclid(2) + 1
            };
        }
        len.max(0) as usize
    }

    fn forward(&mut self, x: &Array, lengths: &Array) -> Result<(Array, Array)> {
        let frames = self.forward_frames_chunked(x, SUBSAMPLING_CHUNK_FRAMES)?;
        Ok((frames, self.output_lengths(lengths)?))
    }

    /// Run the conv stack and projection over time chunks of `chunk_frames` mel
    /// frames, so long inputs never hold the full-resolution conv activations
    /// (several GB for a 20 min file) at once.
    ///
    /// The result is exact. Each output frame depends on only a few input frames
    /// (15 for 8x subsampling). Every chunk starts on a multiple of the
    /// subsampling factor, so its strides line up with the full computation, and
    /// it runs with twice the receptive field of real neighbouring input on each
    /// side. The outputs that see the chunk's own edge padding fall in that
    /// context and are cropped. Chunks touching the true start or end pad
    /// exactly as the full input does.
    fn forward_frames_chunked(&mut self, x: &Array, chunk_frames: usize) -> Result<Array> {
        let total = x.dim(1) as usize;
        let factor = self.subsampling_factor;
        let chunk = chunk_frames.max(factor) / factor * factor;
        if total <= chunk {
            return self.forward_frames(x);
        }
        // Receptive field of the stacked stride-2 convs, in input frames (15 for 8x),
        // doubled and rounded up to a multiple of the factor to keep chunks aligned.
        let receptive_field = 1 + (self.kernel_size - 1) * (factor - 1);
        let context = (2 * receptive_field).div_ceil(factor) * factor;
        let mut outputs = Vec::new();
        let mut start = 0;
        while start < total {
            let end = (start + chunk).min(total);
            let left = context.min(start);
            let right = context.min(total - end);
            let piece = x.index((.., (start - left) as i32..(end + right) as i32, ..));
            let out = self.forward_frames(&piece)?;
            let first = (left / factor) as i32;
            let out = if end == total {
                out.index((.., first.., ..))
            } else {
                out.index((.., first..first + ((end - start) / factor) as i32, ..))
            };
            // Materialize each chunk so its conv activations are freed before the next.
            out.eval()?;
            outputs.push(out);
            start = end;
        }
        Ok(mlx_rs::ops::concatenate(&outputs, 1)?)
    }

    #[cfg(test)]
    fn out_proj_in_features(&self) -> i32 {
        match &self.out_proj {
            LinearLayer::Standard { weight, .. } => weight.dim(1),
            _ => unreachable!("test subsampling uses a standard projection"),
        }
    }

    /// Output lengths after subsampling.
    fn output_lengths(&self, lengths: &Array) -> Result<Array> {
        let n_convs = self.depthwise_convs.len() + 1;
        let mut new_lengths = lengths.as_dtype(mlx_rs::Dtype::Int32)?;
        for _ in 0..n_convs {
            new_lengths = if self.is_causal {
                // CausalConv2D: left_pad=k-1, right_pad=s-1
                Self::calc_conv_output_size(&new_lengths, self.stride as i32)?
            } else {
                // Symmetric padding of 1 with kernel 3, stride 2 (NeMo's calc_length):
                // out = floor((len + 2 - 3) / 2) + 1, which rounds odd lengths up
                let shrunk = mlx_rs::ops::subtract(&new_lengths, Array::from_int(1))?;
                mlx_rs::ops::add(
                    &mlx_rs::ops::floor_divide(&shrunk, Array::from_int(2))?,
                    Array::from_int(1),
                )?
            };
        }
        if *MLX_VOICE_DEBUG {
            new_lengths.eval()?;
            let len_data: Vec<i32> = new_lengths.to_vec_row_major()?;
            tracing::info!("DEBUG subsampling: output_lengths={:?}", len_data);
        }
        Ok(new_lengths)
    }

    /// Conv stack and output projection: [B, T, F] mel features to [B, T', d_model].
    fn forward_frames(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, F] where F is number of mel features (e.g., 80 or 128)
        let debug_enabled = *MLX_VOICE_DEBUG;

        tracing::debug!(
            input_shape = ?x.shape(),
            is_causal = self.is_causal,
            "subsampling forward start"
        );

        if debug_enabled {
            tracing::info!(
                "DEBUG subsampling: input shape {:?} is_causal={} kernel={} stride={}",
                x.shape(),
                self.is_causal,
                self.kernel_size,
                self.stride
            );
            Self::debug_print_array("input", x, 5);
        }

        // Add channel dimension: [B, T, F] -> [B, T, F, 1] for MLX Conv2d
        // MLX Conv2d expects (N, H, W, C) format
        let mut x = x.reshape(&[-1, x.dim(1), x.dim(2), 1])?;

        if debug_enabled {
            tracing::info!(
                "DEBUG subsampling: after reshape to [B,T,F,1]: {:?}",
                x.shape()
            );
        }

        // Apply causal padding if needed before first conv
        if self.is_causal {
            x = causal_pad_2d(&x, self.kernel_size, self.stride)?;
            if debug_enabled {
                tracing::info!(
                    "DEBUG subsampling: after causal_pad_2d (k={}, s={}): {:?}",
                    self.kernel_size,
                    self.stride,
                    x.shape()
                );
            }
        }

        // First conv layer with ReLU
        x = self.first_conv.forward(&x)?;
        x = mlx_rs::nn::relu(&x)?;

        if debug_enabled {
            tracing::info!(
                "DEBUG subsampling: after first_conv + ReLU: {:?}",
                x.shape()
            );
            Self::debug_print_array("after_first_conv", &x, 5);
        }

        // Get parameters for loop since we need to avoid borrow issues
        let is_causal = self.is_causal;
        let kernel_size = self.kernel_size;
        let stride = self.stride;

        // Apply depthwise + pointwise pairs with ReLU
        for (i, (dw_conv, pw_conv)) in self
            .depthwise_convs
            .iter_mut()
            .zip(self.pointwise_convs.iter_mut())
            .enumerate()
        {
            // Apply causal padding before each depthwise conv
            if is_causal {
                x = causal_pad_2d(&x, kernel_size, stride)?;
                if debug_enabled {
                    tracing::info!(
                        "DEBUG subsampling: after causal_pad_2d for dw_conv_{}: {:?}",
                        i,
                        x.shape()
                    );
                }
            }
            x = dw_conv.forward(&x)?;
            x = pw_conv.forward(&x)?;
            x = mlx_rs::nn::relu(&x)?;

            if debug_enabled {
                tracing::info!(
                    "DEBUG subsampling: after dw_conv_{} + pw_conv + ReLU: {:?}",
                    i,
                    x.shape()
                );
                Self::debug_print_array(&format!("after_conv_pair_{}", i), &x, 5);
            }
        }

        // x is now [B, T', F', C] where T' = T/subsampling_factor, F' = F/subsampling_factor
        // In MLX (N, H, W, C), H=T', W=F', C=conv_channels
        // NeMo does: [B, C, T, F] -> permute to [B, T, C, F] -> flatten to [B, T, C*F]
        // We have: [B, T', F', C] -> transpose to [B, T', C, F'] -> flatten to [B, T', C*F']
        let batch = x.dim(0);
        let time = x.dim(1);

        if debug_enabled {
            tracing::info!(
                "DEBUG subsampling: before transpose, shape {:?} (batch={}, time={})",
                x.shape(),
                batch,
                time
            );
        }

        // Transpose to [B, T', C, F'] to match NeMo's C*F flattening order
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 1, 3, 2])?;

        if debug_enabled {
            tracing::info!(
                "DEBUG subsampling: after transpose to [B,T,C,F]: {:?}",
                x.shape()
            );
        }

        // Flatten to [B, T', C*F']
        let x = x.reshape(&[batch, time, -1])?;

        if debug_enabled {
            tracing::info!(
                "DEBUG subsampling: after flatten to [B,T,C*F]: {:?}",
                x.shape()
            );
            Self::debug_print_array("before_out_proj", &x, 5);
        }

        // Project to d_model: [B, T', d_model]
        let x = self.out_proj.forward(&x)?;

        if debug_enabled {
            tracing::info!("DEBUG subsampling: after out_proj: {:?}", x.shape());
            Self::debug_print_array("after_out_proj", &x, 5);
        }

        tracing::trace!(output_shape = ?x.shape(), "subsampling forward complete");

        Ok(x)
    }

    fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // Load first conv: encoder.pre_encode.conv.0
        load_conv2d(&mut self.first_conv, weights, &format!("{}.conv.0", prefix))?;

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
            load_conv2d(dw_conv, weights, &format!("{}.conv.{}", prefix, dw_idx))?;
            load_conv2d(pw_conv, weights, &format!("{}.conv.{}", prefix, pw_idx))?;
        }

        // Load output projection (may be quantized)
        if let Some(layer) = weights.fetch_linear(&format!("{}.out", prefix), quant)? {
            self.out_proj = layer;
        }

        Ok(())
    }
}

/// Masks keeping padded frames (beyond each sequence's length) out of the
/// valid ones in offline conformer layers, like NeMo's `pad_mask`: hidden
/// from attention as keys and zeroed before the depthwise convolution.
/// Padded frames' own outputs are left unspecified.
struct PaddingMask {
    /// Additive mask over attention keys [batch, 1, 1, time]
    keys: Array,
    /// Whether each frame is valid [batch, time, 1]
    valid: Array,
}

impl PaddingMask {
    /// Masks for encoder frames with valid `lengths` `[batch]`, or `None` when
    /// no frame is padding.
    fn new(lengths: &Array, seq_len: i32) -> Result<Option<Self>> {
        let lengths: Vec<i32> = lengths.as_dtype(mlx_rs::Dtype::Int32)?.to_vec_row_major()?;
        let Some(keys) = key_padding_mask(&lengths, seq_len)? else {
            return Ok(None);
        };
        let batch = lengths.len() as i32;
        let frames = mlx_rs::ops::arange::<_, i32>(0, seq_len, 1)?.reshape(&[1, seq_len, 1])?;
        let lengths = Array::from_slice(&lengths, &[batch, 1, 1]);
        let valid = mlx_rs::ops::lt(&frames, &lengths)?;
        Ok(Some(Self { keys, valid }))
    }
}

/// Single Conformer block.
#[derive(Debug)]
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
    fn set_dtype(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        self.ff1.set_dtype(dtype)?;
        self.ff2.set_dtype(dtype)?;
        self.self_attn.set_dtype(dtype)?;
        self.conv.set_dtype(dtype)?;
        for norm in [
            &mut self.ff1_norm,
            &mut self.self_attn_norm,
            &mut self.conv_norm,
            &mut self.ff2_norm,
            &mut self.final_norm,
        ] {
            cast_module(norm, dtype)?;
        }
        Ok(())
    }

    fn new(config: &EncoderConfig) -> Result<Self> {
        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let ff_dim = (d_model as f32 * config.ff_expansion_factor) as usize;
        let conv_kernel = config.conv_kernel_size;
        let dropout = config.dropout;
        let use_bias = config.use_bias;

        // Check if causal convolution is requested
        let is_causal = config.is_causal_conv();

        Ok(Self {
            d_model,
            ff1: FeedForward::new(d_model, ff_dim, dropout, use_bias)?,
            ff1_norm: nn::LayerNorm::new(d_model as i32)?,
            self_attn: RelPositionMultiHeadAttention::new(d_model, n_heads, use_bias)?,
            self_attn_norm: nn::LayerNorm::new(d_model as i32)?,
            conv: ConvolutionModule::new(
                d_model,
                conv_kernel,
                dropout,
                &config.conv_norm_type,
                is_causal,
                use_bias,
            )?,
            conv_norm: nn::LayerNorm::new(d_model as i32)?,
            ff2: FeedForward::new(d_model, ff_dim, dropout, use_bias)?,
            ff2_norm: nn::LayerNorm::new(d_model as i32)?,
            final_norm: nn::LayerNorm::new(d_model as i32)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        pos_emb: &Array,
        context: Option<AttentionContext>,
        padding: Option<&PaddingMask>,
    ) -> Result<Array> {
        // x: [B, T, D]
        let debug_layer = *DEBUG_LAYER0;

        // Feed-forward 1 (half-step residual)
        let residual = x.clone();
        let ff1_normed = self.ff1_norm.forward(x)?;

        if debug_layer {
            ff1_normed.eval()?;
            let flat = ff1_normed.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after ff1_norm T00[:5]");
        }

        let ff1_out = self.ff1.forward(&ff1_normed)?;

        if debug_layer {
            ff1_out.eval()?;
            let flat = ff1_out.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after ff1 T00[:5]");
        }

        let half = Array::from_f32(0.5).as_dtype(ff1_out.dtype())?;
        let scaled_ff1 = &half * &ff1_out;
        let x = &residual + &scaled_ff1;

        if debug_layer {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let d = x.dim(2) as usize;
            let t00: Vec<f32> = slice[..5].to_vec();
            let t16_offset = 16 * d;
            let t16: Option<Vec<f32>> = if slice.len() > t16_offset + 4 {
                Some(slice[t16_offset..t16_offset + 5].to_vec())
            } else {
                None
            };
            tracing::debug!(?t00, ?t16, "after FF1+residual");
        }

        // Self-attention with relative positional encoding (full residual)
        let residual = x.clone();
        let attn_normed = self.self_attn_norm.forward(&x)?;

        if debug_layer {
            attn_normed.eval()?;
            let flat = attn_normed.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let d = attn_normed.dim(2) as usize;
            let t00: Vec<f32> = slice[..5].to_vec();
            let t16_offset = 16 * d;
            let t16: Option<Vec<f32>> = if slice.len() > t16_offset + 4 {
                Some(slice[t16_offset..t16_offset + 5].to_vec())
            } else {
                None
            };
            tracing::debug!(?t00, ?t16, "after self_attn_norm");
        }

        let attn_out =
            self.self_attn
                .forward(&attn_normed, pos_emb, context, padding.map(|p| &p.keys))?;

        if debug_layer {
            attn_out.eval()?;
            let flat = attn_out.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after self_attn T00[:5]");
        }

        let x = &residual + &attn_out;

        if debug_layer {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after self_attn+residual T00[:5]");
        }

        // Convolution (full residual)
        let residual = x.clone();
        let conv_out = self.conv_norm.forward(&x)?;

        if debug_layer {
            conv_out.eval()?;
            let flat = conv_out.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after conv_norm T00[:5]");
        }

        let conv_out = self.conv.forward(&conv_out, padding.map(|p| &p.valid))?;

        if debug_layer {
            conv_out.eval()?;
            let flat = conv_out.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after conv T00[:5]");
        }

        let x = &residual + &conv_out;

        if debug_layer {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after conv+residual T00[:5]");
        }

        // Feed-forward 2 (half-step residual)
        let residual = x.clone();
        let ff2_out = self.ff2_norm.forward(&x)?;

        if debug_layer {
            ff2_out.eval()?;
            let flat = ff2_out.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after ff2_norm T00[:5]");
        }

        let ff2_out = self.ff2.forward(&ff2_out)?;

        if debug_layer {
            ff2_out.eval()?;
            let flat = ff2_out.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after ff2 T00[:5]");
        }

        let scaled_ff2 = &half * &ff2_out;
        let x = &residual + &scaled_ff2;

        if debug_layer {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after ff2+residual T00[:5]");
        }

        // Final layer norm
        let x = self.final_norm.forward(&x)?;

        if debug_layer {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "after final_norm T00[:5]");
        }

        Ok(x)
    }

    /// Forward pass with cache support for streaming inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `pos_emb` - Positional embeddings
    /// * `cache_last_channel` - Optional full MHA cache buffer [batch, cache_size, d_model]
    /// * `cache_filled_len` - How many frames of MHA cache are valid
    /// * `cache_last_time` - Optional conv cache [batch, d_model, time_cache]
    /// * `att_mask` - Optional attention mask [batch, seq_len, total_len] where True=masked
    /// * `pad_mask` - Optional padding mask [batch, seq_len] where True=padding (for conv module)
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated MHA cache
    /// * Updated conv cache
    #[allow(clippy::too_many_arguments)]
    fn forward_with_cache(
        &mut self,
        x: &Array,
        pos_emb: &Array,
        cache_last_channel: Option<&Array>,
        cache_filled_len: i32,
        cache_last_time: Option<&Array>,
        att_mask: Option<&Array>,
        pad_mask: Option<&Array>,
    ) -> Result<(Array, Option<Array>, Option<Array>)> {
        tracing::trace!(
            x_shape = ?x.shape(),
            pos_emb_shape = ?pos_emb.shape(),
            cache_last_channel_shape = ?cache_last_channel.map(|c| c.shape().to_vec()),
            cache_filled_len,
            cache_last_time_shape = ?cache_last_time.map(|c| c.shape().to_vec()),
            has_att_mask = att_mask.is_some(),
            has_pad_mask = pad_mask.is_some(),
            "ConformerBlock::forward_with_cache entry"
        );

        // Feed-forward 1 (half-step residual)
        let residual = x.clone();
        let ff1_norm_out = self.ff1_norm.forward(x)?;
        let ff1_out = self.ff1.forward(&ff1_norm_out)?;
        let half = Array::from_f32(0.5).as_dtype(ff1_out.dtype())?;
        let scaled_ff1 = &half * &ff1_out;
        let x = &residual + &scaled_ff1;

        tracing::trace!(x_shape = ?x.shape(), "after ff1");

        // Self-attention with cache
        let residual = x.clone();
        let mha_norm_out = self.self_attn_norm.forward(&x)?;
        let (attn_out, new_cache_channel) = self.self_attn.forward_with_cache(
            &mha_norm_out,
            pos_emb,
            cache_last_channel,
            cache_filled_len,
            att_mask,
        )?;
        let x = &residual + &attn_out;

        tracing::trace!(
            attn_out_shape = ?attn_out.shape(),
            new_cache_channel_shape = ?new_cache_channel.as_ref().map(|c| c.shape().to_vec()),
            "after self-attention"
        );

        // Convolution with cache
        let residual = x.clone();
        let conv_norm_out = self.conv_norm.forward(&x)?;
        let (conv_out, new_cache_time) =
            self.conv
                .forward_with_cache(&conv_norm_out, cache_last_time, pad_mask)?;
        let x = &residual + &conv_out;

        tracing::trace!(
            conv_out_shape = ?conv_out.shape(),
            new_cache_time_shape = ?new_cache_time.as_ref().map(|c| c.shape().to_vec()),
            "after convolution"
        );

        // Feed-forward 2 (half-step residual)
        let residual = x.clone();
        let ff2_norm_out = self.ff2_norm.forward(&x)?;
        let ff2_out = self.ff2.forward(&ff2_norm_out)?;
        let scaled_ff2 = &half * &ff2_out;
        let x = &residual + &scaled_ff2;

        tracing::trace!(x_shape = ?x.shape(), "after ff2");

        // Final layer norm
        let x = self.final_norm.forward(&x)?;

        tracing::trace!(x_shape = ?x.shape(), "ConformerBlock::forward_with_cache output");

        Ok((x, new_cache_channel, new_cache_time))
    }

    /// Get the convolution cache size for this block.
    fn conv_cache_size(&self) -> usize {
        self.conv.cache_size()
    }

    fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // Load feed-forward 1 (NeMo uses "feed_forward1" not "ff1")
        self.ff1
            .load_weights(weights, &format!("{}.feed_forward1", prefix), quant)?;
        load_layer_norm(
            &mut self.ff1_norm,
            weights,
            &format!("{}.norm_feed_forward1", prefix),
        )?;

        // Load self-attention (linear projections may be quantized)
        self.self_attn
            .load_weights(weights, &format!("{}.self_attn", prefix), quant)?;
        load_layer_norm(
            &mut self.self_attn_norm,
            weights,
            &format!("{}.norm_self_att", prefix),
        )?;

        // Load convolution module (pointwise convs may be quantized)
        self.conv
            .load_weights(weights, &format!("{}.conv", prefix), quant)?;
        load_layer_norm(
            &mut self.conv_norm,
            weights,
            &format!("{}.norm_conv", prefix),
        )?;

        // Load feed-forward 2 (NeMo uses "feed_forward2" not "ff2")
        self.ff2
            .load_weights(weights, &format!("{}.feed_forward2", prefix), quant)?;
        load_layer_norm(
            &mut self.ff2_norm,
            weights,
            &format!("{}.norm_feed_forward2", prefix),
        )?;

        // Load final norm (NeMo uses "norm_out")
        load_layer_norm(
            &mut self.final_norm,
            weights,
            &format!("{}.norm_out", prefix),
        )?;

        Ok(())
    }
}

/// Feed-forward module with expansion.
#[derive(Debug)]
struct FeedForward {
    linear1: LinearLayer,
    linear2: LinearLayer,
    #[allow(dead_code)]
    dropout: f32,
}

impl FeedForward {
    fn set_dtype(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        self.linear1.cast(dtype)?;
        self.linear2.cast(dtype)
    }

    fn new(d_model: usize, ff_dim: usize, dropout: f32, use_bias: bool) -> Result<Self> {
        // Create zero-initialized linear layers (will be loaded from weights)
        let create_linear =
            |in_dim: usize, out_dim: usize, with_bias: bool| -> Result<LinearLayer> {
                let weight = mlx_rs::ops::zeros::<f32>(&[out_dim as i32, in_dim as i32])?;
                let bias = if with_bias {
                    Some(mlx_rs::ops::zeros::<f32>(&[out_dim as i32])?)
                } else {
                    None
                };
                Ok(LinearLayer::standard(weight, bias))
            };

        Ok(Self {
            linear1: create_linear(d_model, ff_dim, use_bias)?,
            linear2: create_linear(ff_dim, d_model, use_bias)?,
            dropout,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.linear1.forward(x)?;
        let x = mlx_rs::nn::silu(&x)?; // silu is the same as swish
        // Skip dropout during inference
        let x = self.linear2.forward(&x)?;
        Ok(x)
    }

    fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        if let Some(layer) = weights.fetch_linear(&format!("{}.linear1", prefix), quant)? {
            self.linear1 = layer;
        }
        if let Some(layer) = weights.fetch_linear(&format!("{}.linear2", prefix), quant)? {
            self.linear2 = layer;
        }
        Ok(())
    }
}

/// Normalization type for convolution module.
#[derive(Debug)]
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
#[derive(Debug)]
struct ConvolutionModule {
    d_model: usize,
    pointwise_conv1: LinearLayer,
    depthwise_conv: nn::Conv1d,
    pointwise_conv2: LinearLayer,
    norm: ConvNorm,
    /// Whether this is a causal convolution (for streaming)
    is_causal: bool,
    /// Kernel size for causal padding calculation
    kernel_size: usize,
    /// Cache drop size for streaming
    pub cache_drop_size: usize,
}

impl ConvolutionModule {
    fn set_dtype(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        self.pointwise_conv1.cast(dtype)?;
        self.pointwise_conv2.cast(dtype)?;
        cast_module(&mut self.depthwise_conv, dtype)?;
        match &mut self.norm {
            ConvNorm::BatchNorm(bn) => {
                cast_module(bn, dtype)?;
                bn.eps = bn.eps.as_dtype(dtype)?;
            }
            ConvNorm::LayerNorm(ln) => cast_module(ln, dtype)?,
        }
        Ok(())
    }

    fn new(
        d_model: usize,
        kernel_size: usize,
        _dropout: f32,
        norm_type: &ConvNormType,
        is_causal: bool,
        use_bias: bool,
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
            ConvNormType::LayerNorm => ConvNorm::LayerNorm(nn::LayerNorm::new(d_model as i32)?),
            _ => {
                // Inference mode: normalize with the loaded running stats. In
                // training mode mlx-rs normalizes with each input's own stats
                // and overwrites the running stats on every call.
                let mut bn = nn::BatchNorm::new(d_model as i32)?;
                bn.training = false;
                ConvNorm::BatchNorm(bn)
            }
        };

        // Create zero-initialized pointwise linear layers (will be loaded from weights)
        let pw1_weight = mlx_rs::ops::zeros::<f32>(&[(2 * d_model) as i32, d_model as i32])?;
        let pw1_bias = Some(mlx_rs::ops::zeros::<f32>(&[(2 * d_model) as i32])?);
        let pw2_weight = mlx_rs::ops::zeros::<f32>(&[d_model as i32, d_model as i32])?;
        let pw2_bias = Some(mlx_rs::ops::zeros::<f32>(&[d_model as i32])?);

        Ok(Self {
            d_model,
            // Pointwise expansion (d_model -> 2*d_model) as linear layer
            pointwise_conv1: LinearLayer::standard(pw1_weight, pw1_bias),
            // Depthwise convolution
            depthwise_conv: nn::Conv1dBuilder::new(
                d_model as i32,
                d_model as i32,
                kernel_size as i32,
            )
            .padding(padding as i32)
            .groups(d_model as i32)
            .bias(use_bias)
            .build()?,
            // Pointwise compression (d_model -> d_model) as linear layer
            pointwise_conv2: LinearLayer::standard(pw2_weight, pw2_bias),
            norm,
            is_causal,
            kernel_size,
            cache_drop_size: 0,
        })
    }

    /// `valid` [batch, time, 1] marks frames that aren't padding; padded frames
    /// are zeroed before the depthwise conv so they don't reach valid ones.
    fn forward(&mut self, x: &Array, valid: Option<&Array>) -> Result<Array> {
        // x: [B, T, D] - MLX Conv1d expects (N, L, C_in) format, no transpose needed
        let debug_conv = *DEBUG_CONV;

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "CONV input T00[:5]");
        }

        // Pointwise conv 1 with GLU activation
        // Output: [B, T, 2*D]
        let x = self.pointwise_conv1.forward(x)?;

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, shape = ?x.shape(), "CONV after pointwise1");
        }

        // GLU: split in half along channel dim (axis 2), apply sigmoid to second half, multiply
        let d_model = self.d_model as i32;
        // Slice first half: x[:, :, :d_model]
        let x1 = x.index((.., .., ..d_model));
        // Slice second half: x[:, :, d_model:]
        let x2 = x.index((.., .., d_model..));
        let sigmoid_x2 = mlx_rs::ops::sigmoid(&x2)?;
        let mut x = &x1 * &sigmoid_x2;
        if let Some(valid) = valid {
            let zero = Array::from_f32(0.0).as_dtype(x.dtype())?;
            x = mlx_rs::ops::select(valid, &x, &zero)?;
        }

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "CONV after GLU T00[:5]");
        }

        // Apply causal padding if needed (pad left only)
        let x = if self.is_causal {
            // Pad on the left (time dimension) with zeros
            // x is [B, T, D], we need to pad T by (kernel_size - 1) on the left
            let pad_size = self.kernel_size - 1;
            let batch = x.dim(0);
            let d = x.dim(2);

            // Create zero padding: [B, pad_size, D]
            let zeros = mlx_rs::ops::zeros::<f32>(&[batch as i32, pad_size as i32, d as i32])?;
            mlx_rs::ops::concatenate(&[&zeros, &x], 1)?
        } else {
            x
        };

        // Depthwise conv - x is [B, T, D] (or [B, T+pad, D] if causal)
        let x = self.depthwise_conv.forward(&x)?;

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "CONV after depthwise T00[:5]");
        }

        let x = self.norm.forward(&x)?;

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "CONV after norm T00[:5]");
        }

        let x = mlx_rs::nn::silu(&x)?; // silu is the same as swish

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "CONV after silu T00[:5]");
        }

        // Pointwise conv 2
        let x = self.pointwise_conv2.forward(&x)?;

        if debug_conv {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: Vec<f32> = flat.to_vec_row_major()?;
            let t00: Vec<f32> = slice[..5].to_vec();
            tracing::debug!(?t00, "CONV after pointwise2 (final) T00[:5]");
        }

        Ok(x)
    }

    /// Forward pass with cache support for streaming inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `cache` - Optional cache tensor [batch, d_model, cache_len] for left context
    /// * `pad_mask` - Optional padding mask [batch, seq_len] where True = padding (to be zeroed)
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated cache tensor [batch, d_model, new_cache_len] (if cache was provided)
    fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: Option<&Array>,
        pad_mask: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        tracing::debug!(
            x_shape = ?x.shape(),
            cache_shape = ?cache.map(|c| c.shape().to_vec()),
            has_pad_mask = pad_mask.is_some(),
            kernel_size = self.kernel_size,
            d_model = self.d_model,
            "ConvModule::forward_with_cache entry"
        );

        // Pointwise conv 1 with GLU activation
        let x = self.pointwise_conv1.forward(x)?;

        // GLU: split in half along channel dim
        let d_model = self.d_model as i32;
        let x1 = x.index((.., .., ..d_model));
        let x2 = x.index((.., .., d_model..));
        let sigmoid_x2 = mlx_rs::ops::sigmoid(&x2)?;
        let mut x = &x1 * &sigmoid_x2;

        // Apply padding mask after GLU - this matches Python NeMo's behavior
        // where padding frames are zeroed out before depthwise conv and cache update
        if let Some(mask) = pad_mask {
            // pad_mask: [batch, seq_len] where True = padding
            // x: [batch, seq_len, d_model]
            // Expand mask to [batch, seq_len, 1] for broadcasting
            let mask_expanded = mask.reshape(&[mask.dim(0), mask.dim(1), 1])?;
            // Where mask is True (padding), set to 0.0
            x = mlx_rs::ops::select(&mask_expanded, Array::from_f32(0.0), &x)?;
        }

        // For streaming: use cache instead of padding
        let (x, new_cache) = if let Some(cache_tensor) = cache {
            // cache: [batch, d_model, cache_len]
            // x after GLU: [batch, seq_len, d_model]
            // Need to transpose x to [batch, d_model, seq_len] for concatenation
            let x_transposed = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1])?;

            // Concatenate: [batch, d_model, cache_len + seq_len]
            let x_with_cache = mlx_rs::ops::concatenate(&[cache_tensor, &x_transposed], 2)?;

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
            let x_padded = mlx_rs::ops::concatenate(&[&zeros, &x], 1)?;
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

    fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // Pointwise convs load as LinearLayer (may be quantized)
        if let Some(layer) = weights.fetch_linear(&format!("{}.pointwise_conv1", prefix), quant)? {
            self.pointwise_conv1 = layer;
        }

        // Depthwise conv stays as nn::Conv1d (not quantizable)
        load_conv1d(
            &mut self.depthwise_conv,
            weights,
            &format!("{}.depthwise_conv", prefix),
        )?;

        if let Some(layer) = weights.fetch_linear(&format!("{}.pointwise_conv2", prefix), quant)? {
            self.pointwise_conv2 = layer;
        }

        // Load normalization weights based on type
        match &mut self.norm {
            ConvNorm::BatchNorm(bn) => {
                load_batch_norm(bn, weights, &format!("{}.batch_norm", prefix))?;
            }
            ConvNorm::LayerNorm(ln) => {
                load_layer_norm(ln, weights, &format!("{}.batch_norm", prefix))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chunked subsampling must reproduce the single-pass result exactly,
    /// including a ragged final chunk, for both padding styles.
    #[test]
    fn test_chunked_subsampling_matches_single_pass() {
        for causal in [false, true] {
            let mut config = EncoderConfig::default();
            config.feat_in = 16;
            config.d_model = 8;
            config.subsampling_factor = 8;
            config.subsampling_conv_channels = Some(4);
            config.causal_downsampling = causal;
            let mut sub = Subsampling::new(&config).unwrap();
            // Constructor weights are placeholders until a checkpoint loads; give the
            // depthwise convs the real [channels, k, k, 1] shape.
            for (i, dw) in sub.depthwise_convs.iter_mut().enumerate() {
                let key = mlx_rs::random::key(100 + i as u64).unwrap();
                dw.weight.value =
                    mlx_rs::random::normal::<f32>(&[4, 3, 3, 1], None, None, &key).unwrap() * 0.5;
            }
            let key = mlx_rs::random::key(7).unwrap();
            let linear_in = sub.out_proj_in_features();
            sub.out_proj = LinearLayer::standard(
                mlx_rs::random::normal::<f32>(&[8, linear_in], None, None, &key).unwrap(),
                None,
            );

            let key = mlx_rs::random::key(3).unwrap();
            let x = mlx_rs::random::normal::<f32>(&[1, 203, 16], None, None, &key).unwrap();
            let single = sub.forward_frames(&x).unwrap();
            let chunked = sub.forward_frames_chunked(&x, 40).unwrap();
            assert_eq!(chunked.shape(), single.shape(), "causal={causal}");

            let a: Vec<f32> = single.to_vec_row_major().unwrap();
            let b: Vec<f32> = chunked.to_vec_row_major().unwrap();
            assert!(
                a.iter().any(|v| v.abs() > 1e-3),
                "degenerate output (causal={causal})"
            );
            let max_diff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            assert!(max_diff < 1e-5, "causal={causal}: max diff {max_diff}");
        }
    }
}
