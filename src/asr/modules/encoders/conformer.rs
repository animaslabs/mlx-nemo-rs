//! Conformer encoder implementation.
//!
//! The Conformer architecture combines convolution and self-attention
//! for effective speech feature encoding.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::attention::{RelPositionMultiHeadAttention, RelPositionalEncoding};
use crate::asr::config::EncoderConfig;
use crate::core::error::Result;

/// Global counter for encoder forward calls (for debugging)
static ENCODER_CALL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset the encoder call counter (call at start of streaming session)
pub fn reset_encoder_counter() {
    ENCODER_CALL_COUNT.store(0, Ordering::SeqCst);
}

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
    _seq_len: i32,
) -> Result<Array> {
    let debug = std::env::var("NEMO_DEBUG_MASK")
        .map(|v| v == "1")
        .unwrap_or(false);

    if debug {
        eprintln!("[MASK DEBUG] att_context_size={:?}, style={}, max_audio_len={}, cache_len={}, filled={}, seq_len={}",
                 att_context_size, att_context_style, max_audio_length, cache_len, filled_cache_len, _seq_len);
    }

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

    if debug {
        eprintln!(
            "[MASK DEBUG] offset={} (cache_len={} - filled={})",
            offset, cache_len, filled_cache_len
        );
        if offset < 0 {
            eprintln!("[MASK WARNING] Negative offset detected! This indicates cache overflow.");
        }
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
    let att_mask = att_mask.index((.., cache_len.., ..));

    // Invert: NeMo returns inverted mask at the end (True means ignore)
    let att_mask = mlx_rs::ops::logical_not(&att_mask)?;

    if debug {
        att_mask.eval()?;
        // Count how many positions are visible (False = visible after inversion)
        let flat = att_mask.reshape(&[-1])?;
        flat.eval()?;
        let flat_slice: &[bool] = flat.as_slice();
        let visible_count = flat_slice.iter().filter(|&&x| !x).count();
        let total = flat_slice.len();
        eprintln!(
            "[MASK DEBUG] output shape={:?}, visible={}/{} ({:.1}%)",
            att_mask.shape(),
            visible_count,
            total,
            100.0 * visible_count as f64 / total as f64
        );
    }

    Ok(att_mask)
}

/// Apply upper triangular masking (triu) to attention mask.
///
/// Sets positions below the diagonal to False.
/// diagonal=0 means main diagonal, negative shifts down.
fn apply_triu(mask: &Array, diagonal: i32) -> Result<Array> {
    let shape = mask.shape();
    let rows = shape[1] as i32;
    let cols = shape[2] as i32;

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
    let rows = shape[1] as i32;
    let cols = shape[2] as i32;

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

    /// Attention context size (left, right) for streaming
    /// Parsed from config.att_context_size
    att_context_size: Option<(i32, i32)>,

    /// Attention context style ("regular" or "chunked_limited")
    att_context_style: String,
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

        // Extract attention context configuration for streaming
        let att_context_size = config
            .att_context_size
            .as_ref()
            .and_then(|cs| cs.get_default());
        let att_context_style = config
            .att_context_style
            .clone()
            .unwrap_or_else(|| "regular".to_string());

        Ok(Self {
            config,
            subsampling,
            pos_enc,
            layers,
            final_norm,
            att_context_size,
            att_context_style,
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
    /// This implements NeMo's streaming forward with:
    /// 1. Attention mask creation based on att_context_size and att_context_style
    /// 2. Full cache buffer passed to layers (not just filled portion)
    /// 3. Proper cache update with sliding window
    ///
    /// # Arguments
    /// * `features` - Input features [batch, features, time]
    /// * `lengths` - Sequence lengths [batch]
    /// * `cache_last_channel` - MHA cache [num_layers, batch, cache_len, d_model]
    /// * `cache_last_time` - Conv cache [num_layers, batch, d_model, time_cache]
    /// * `cache_last_channel_len` - Cache fill level [batch]
    /// * `drop_extra_pre_encoded` - Frames to drop after subsampling
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
        // Transpose from [B, F, T] to [B, T, F] for processing
        let x = mlx_rs::ops::transpose_axes(features, &[0, 2, 1])?;

        // Subsampling
        let (mut x, mut lengths) = self.subsampling.forward(&x, lengths)?;

        // Drop extra pre-encoded frames for streaming
        // This is needed because the subsampling convolutions produce extra frames
        // at chunk boundaries that should be discarded to match offline behavior.
        // NeMo applies this on ALL chunks (whenever cache_last_channel is not None),
        // which in streaming mode is always.
        if drop_extra_pre_encoded > 0 {
            let drop = drop_extra_pre_encoded as i32;
            x = x.index((.., drop.., ..));
            lengths = mlx_rs::ops::clip(
                &(&lengths - &Array::from_int(drop)),
                (&Array::from_int(0), ()),
            )?;
        }

        // Scale input by sqrt(d_model) if xscaling is enabled
        let xscale = self.pos_enc.xscale();
        if xscale != 1.0 {
            x = &x * Array::from_f32(xscale);
        }

        let seq_len = x.dim(1) as i32;
        let cache_size = cache_last_channel.dim(2) as i32; // Allocated cache size

        // Get the actual filled cache length (use first batch element, assume all same)
        cache_last_channel_len.eval()?;
        let filled_cache_len = cache_last_channel_len.as_slice::<i64>()[0].max(0) as i32;

        // Total length for K/V is cache_size + seq_len (full buffer, not just filled)
        // NeMo uses full buffer and relies on mask to hide unfilled positions
        let total_len = cache_size + seq_len;

        // Generate positional encoding for total length (full buffer + current)
        // This should cover positions for all K/V that attention will see
        let pos_emb = self.pos_enc.get_encoding(total_len as usize)?;

        // Create attention mask if we have context configuration
        // NeMo: att_mask = _create_masks(...); att_mask = att_mask[:, cache_len:]
        //
        // The mask is created for the full buffer (cache_size + seq_len = total_len),
        // and offset masking hides the unfilled cache positions.
        let att_mask = if let Some(ctx) = self.att_context_size {
            Some(create_attention_mask(
                ctx,
                &self.att_context_style,
                total_len,        // Full buffer: cache_size + seq_len
                cache_size,       // Full cache buffer size (for slicing)
                filled_cache_len, // How much is filled (for offset masking)
                seq_len,          // Current sequence length
            )?)
        } else {
            None
        };

        // Initialize output caches
        let mut cache_channel_next = Vec::with_capacity(self.layers.len());
        let mut cache_time_next = Vec::with_capacity(self.layers.len());

        // Track encoder calls for debugging
        let enc_call = ENCODER_CALL_COUNT.fetch_add(1, Ordering::SeqCst);

        // Debug encoder input stats
        let debug_layers = std::env::var("NEMO_DEBUG_LAYERS")
            .map(|v| v == "1")
            .unwrap_or(false);
        if debug_layers && (enc_call < 5 || enc_call == 135 || enc_call == 200 || enc_call == 270) {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: &[f32] = flat.as_slice();
            let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let var: f32 =
                slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
            let std = var.sqrt();
            eprintln!(
                "[ENCODER DEBUG] enc={}, seq_len={}, encoder_input: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
                enc_call, seq_len, min, max, mean, std
            );
        }

        // Process through conformer blocks with cache
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
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
                layer_idx,
                enc_call,
            )?;

            x = x_out;

            // Update cache: the attention layer returns new_cache from sliding window
            // We need to store it properly in the full cache buffer
            //
            // The attention returns: new_cache = concat(old_filled[q_keep_size:], x[:q_keep_size])
            // We need to place this at the END of the full cache buffer
            if let Some(nc) = new_channel {
                let nc_len = nc.dim(1) as i32;
                let updated_cache = if nc_len < cache_size {
                    // Pad with zeros at the beginning to maintain cache_size
                    let pad_size = cache_size - nc_len;
                    let batch = nc.dim(0) as i32;
                    let d_model = nc.dim(2) as i32;
                    let zeros = mlx_rs::ops::zeros::<f32>(&[batch, pad_size, d_model])?;
                    mlx_rs::ops::concatenate_axis(&[&zeros, &nc], 1)?
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

        // Debug encoder output stats
        if debug_layers && (enc_call < 5 || enc_call == 135 || enc_call == 200 || enc_call == 270) {
            x.eval()?;
            let flat = x.reshape(&[-1])?;
            flat.eval()?;
            let slice: &[f32] = flat.as_slice();
            let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let var: f32 =
                slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
            let std = var.sqrt();
            eprintln!(
                "[ENCODER DEBUG] enc={}, encoder_output: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
                enc_call, min, max, mean, std
            );
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

        // Debug: track cache length
        let debug_cache_len = std::env::var("NEMO_DEBUG_CACHE_LEN")
            .map(|v| v == "1")
            .unwrap_or(false);

        if debug_cache_len {
            eprintln!(
                "[CACHE_LEN DEBUG] seq_len={}, cache_size={}, filled_before={}",
                seq_len, cache_size, filled_cache_len
            );
        }

        // Update cache length: clamp(old_len + seq_len, max=cache_size)
        let cache_channel_len_next = mlx_rs::ops::clip(
            &(cache_last_channel_len + &Array::from_int(seq_len)),
            ((), &Array::from_int(cache_size)),
        )?;

        // Verify and debug the result
        cache_channel_len_next.eval()?;
        let filled_after = cache_channel_len_next.as_slice::<i64>()[0];

        if debug_cache_len {
            eprintln!(
                "[CACHE_LEN DEBUG] filled_after={}, expected_max={}",
                filled_after, cache_size
            );

            // Flag potential issues
            if filled_after > cache_size as i64 {
                eprintln!(
                    "[CACHE_LEN WARNING] Cache length {} exceeds cache_size {}!",
                    filled_after, cache_size
                );
            }
        }

        // Safety: ensure cache length never exceeds cache_size
        // This guards against any issues with the clip operation
        // Note: cache_last_channel_len is i64, so we need to match that type
        let cache_channel_len_next = if filled_after > cache_size as i64 {
            if debug_cache_len {
                eprintln!(
                    "[CACHE_LEN FIX] Clamping {} to {}",
                    filled_after, cache_size
                );
            }
            Array::from_slice(&[cache_size as i64], &[1])
        } else {
            cache_channel_len_next
        };

        // Debug: check cache statistics
        let debug_cache = std::env::var("NEMO_DEBUG_CACHE")
            .map(|v| v == "1")
            .unwrap_or(false);
        if debug_cache {
            // Check first layer's new cache
            let first_cache = cache_channel_next.index((0, .., .., ..));
            first_cache.eval()?;
            let fc_flat = first_cache.reshape(&[-1])?;
            fc_flat.eval()?;
            let fc_slice: &[f32] = fc_flat.as_slice();
            let fc_min = fc_slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let fc_max = fc_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let fc_mean: f32 = fc_slice.iter().sum::<f32>() / fc_slice.len() as f32;
            let fc_var: f32 = fc_slice.iter().map(|&x| (x - fc_mean).powi(2)).sum::<f32>()
                / fc_slice.len() as f32;
            eprintln!(
                "[CACHE DEBUG] layer0 cache: min={:.4}, max={:.4}, mean={:.4}, var={:.6}",
                fc_min, fc_max, fc_mean, fc_var
            );
        }

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
    /// * `cache_last_channel` - Optional full MHA cache buffer [batch, cache_size, d_model]
    /// * `cache_filled_len` - How many frames of MHA cache are valid
    /// * `cache_last_time` - Optional conv cache [batch, d_model, time_cache]
    /// * `att_mask` - Optional attention mask [batch, seq_len, total_len] where True=masked
    /// * `layer_idx` - Layer index for debugging
    /// * `enc_call` - Encoder call count for debugging
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
        cache_filled_len: i32,
        cache_last_time: Option<&Array>,
        att_mask: Option<&Array>,
        layer_idx: usize,
        enc_call: u64,
    ) -> Result<(Array, Option<Array>, Option<Array>)> {
        let debug_layers = std::env::var("NEMO_DEBUG_LAYERS")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Helper to compute tensor stats
        let compute_stats = |arr: &Array, name: &str| -> Result<()> {
            if !debug_layers {
                return Ok(());
            }
            arr.eval()?;
            let flat = arr.reshape(&[-1])?;
            flat.eval()?;
            let slice: &[f32] = flat.as_slice();
            let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let var: f32 =
                slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
            let std = var.sqrt();
            eprintln!(
                "[LAYER DEBUG] enc={}, layer={}, {}: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
                enc_call, layer_idx, name, min, max, mean, std
            );
            Ok(())
        };

        // Only debug specific encoder calls and layers to avoid too much output
        let should_debug = debug_layers
            && (layer_idx == 0 || layer_idx == 11 || layer_idx == 23)
            && (enc_call < 5 || enc_call == 135 || enc_call == 200 || enc_call == 270);

        if should_debug {
            compute_stats(x, "input")?;
        }

        // Feed-forward 1 (half-step residual)
        let residual = x.clone();
        let ff1_out = self.ff1_norm.forward(x)?;
        if should_debug {
            compute_stats(&ff1_out, "ff1_norm")?;
        }
        let ff1_out = self.ff1.forward(&ff1_out)?;
        if should_debug {
            compute_stats(&ff1_out, "ff1_out")?;
        }
        let half = Array::from_f32(0.5);
        let scaled_ff1 = &half * &ff1_out;
        let x = &residual + &scaled_ff1;
        if should_debug {
            compute_stats(&x, "after_ff1")?;
        }

        // Self-attention with cache
        let residual = x.clone();
        let attn_out = self.self_attn_norm.forward(&x)?;
        if should_debug {
            compute_stats(&attn_out, "attn_norm")?;
        }
        let (attn_out, new_cache_channel) = self.self_attn.forward_with_cache(
            &attn_out,
            pos_emb,
            cache_last_channel,
            cache_filled_len,
            att_mask,
        )?;
        if should_debug {
            compute_stats(&attn_out, "attn_out")?;
        }
        let x = &residual + &attn_out;
        if should_debug {
            compute_stats(&x, "after_attn")?;
        }

        // Convolution with cache
        let residual = x.clone();
        let conv_out = self.conv_norm.forward(&x)?;
        if should_debug {
            compute_stats(&conv_out, "conv_norm")?;
        }
        let (conv_out, new_cache_time) =
            self.conv.forward_with_cache(&conv_out, cache_last_time)?;
        if should_debug {
            compute_stats(&conv_out, "conv_out")?;
        }

        let x = &residual + &conv_out;
        if should_debug {
            compute_stats(&x, "after_conv")?;
        }

        // Feed-forward 2 (half-step residual)
        let residual = x.clone();
        let ff2_out = self.ff2_norm.forward(&x)?;
        if should_debug {
            compute_stats(&ff2_out, "ff2_norm")?;
        }
        let ff2_out = self.ff2.forward(&ff2_out)?;
        if should_debug {
            compute_stats(&ff2_out, "ff2_out")?;
        }
        let scaled_ff2 = &half * &ff2_out;
        let x = &residual + &scaled_ff2;
        if should_debug {
            compute_stats(&x, "after_ff2")?;
        }

        // Final layer norm
        let x = self.final_norm.forward(&x)?;
        if should_debug {
            compute_stats(&x, "final_out")?;
        }

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
