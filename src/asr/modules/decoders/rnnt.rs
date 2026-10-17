//! RNN-T decoder components.
//!
//! This module implements the Transducer decoder components:
//! - `StatelessTransducerDecoder`: The prediction network (label model)
//! - `RNNTJoint`: The joint network that combines encoder and prediction outputs

use mlx_rs::Array;
use mlx_rs::ops::{self, indexing::IndexOp};
use tracing::{debug, instrument, trace, trace_span};

use crate::asr::config::{DecoderConfig, JointConfig};
use crate::asr::modules::LanguagePrompt;
use crate::core::error::{Error, Result};
use crate::core::loading::Weights;
use crate::core::quantization::{EmbeddingLayer, LinearLayer, QuantizationConfig};

/// Stateless Transducer Decoder (Prediction Network).
///
/// A prediction network that uses embeddings of previous tokens (context)
/// to predict the next token. Unlike LSTM-based decoders, this is stateless
/// and simply concatenates/sums embeddings of the context window.
///
/// This is the decoder used by Parakeet RNNT models.
#[derive(Debug)]
pub struct StatelessTransducerDecoder {
    /// Embedding layer: vocab_size + 1 (blank) -> pred_hidden (supports quantized weights)
    embedding: EmbeddingLayer,
    /// Hidden dimension of prediction network
    pred_hidden: usize,
    /// Context size (number of previous tokens to consider)
    context_size: usize,
    /// Vocabulary size (excluding blank)
    vocab_size: usize,
    /// Blank token index (typically vocab_size)
    blank_idx: usize,
}

impl StatelessTransducerDecoder {
    /// Create a new stateless transducer decoder.
    ///
    /// # Arguments
    /// * `config` - Decoder configuration
    /// * `vocab_size` - Vocabulary size (excluding blank token)
    pub fn new(config: &DecoderConfig, vocab_size: usize) -> Result<Self> {
        let pred_hidden = config
            .prednet
            .as_ref()
            .map(|p| p.pred_hidden)
            .unwrap_or(640);

        let context_size = config.context_size.unwrap_or(2);
        let blank_idx = vocab_size;

        // Initialize embedding with zeros (will be loaded from weights)
        // Shape: [vocab_size + 1, pred_hidden] (includes blank token)
        let embedding_weight = Array::zeros::<f32>(&[(vocab_size + 1) as i32, pred_hidden as i32])?;
        let embedding = EmbeddingLayer::standard(embedding_weight);

        Ok(Self {
            embedding,
            pred_hidden,
            context_size,
            vocab_size,
            blank_idx,
        })
    }

    /// Get the prediction network hidden dimension.
    pub fn pred_hidden(&self) -> usize {
        self.pred_hidden
    }

    /// Get the blank token index.
    pub fn blank_idx(&self) -> usize {
        self.blank_idx
    }

    /// Get the vocabulary size (excluding blank).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the context size.
    pub fn context_size(&self) -> usize {
        self.context_size
    }

    /// Initialize decoder state for a batch.
    ///
    /// The state is a context buffer of previous token indices.
    /// Initialized with blank tokens.
    ///
    /// # Arguments
    /// * `batch_size` - Number of sequences in the batch
    ///
    /// # Returns
    /// * State tensor of shape [batch_size, context_size - 1] filled with blank_idx
    #[instrument(skip(self), fields(batch_size, context_size = self.context_size, blank_idx = self.blank_idx))]
    pub fn initialize_state(&self, batch_size: usize) -> Result<Array> {
        // State contains context_size - 1 previous tokens (the current token completes it)
        let state_len = self.context_size.saturating_sub(1).max(1);
        // Create state filled with blank_idx
        let fill_value = Array::from_int(self.blank_idx as i32);
        let state = Array::full::<i32>(&[batch_size as i32, state_len as i32], &fill_value)?;
        debug!(state_shape = ?state.shape(), "Initialized stateless decoder state");
        Ok(state)
    }

    /// Predict the next output given input tokens and optional state.
    ///
    /// # Arguments
    /// * `y` - Input token indices [batch, seq_len] or None for SOS
    /// * `state` - Previous context state [batch, context_size - 1] or None
    /// * `add_sos` - Whether to prepend a start-of-sequence (zero) embedding
    ///
    /// # Returns
    /// * `(output, new_state)` where:
    ///   - output: [batch, seq_len (+ 1 if add_sos), pred_hidden]
    ///   - new_state: [batch, context_size - 1] with updated context
    #[instrument(skip(self, y, state), fields(
        has_input = y.is_some(),
        has_state = state.is_some(),
        add_sos,
        pred_hidden = self.pred_hidden
    ))]
    pub fn predict(
        &self,
        y: Option<&Array>,
        state: Option<&Array>,
        add_sos: bool,
        batch_size: Option<usize>,
    ) -> Result<(Array, Array)> {
        let batch = if let Some(y) = y {
            y.shape()[0] as usize
        } else if let Some(state) = state {
            state.shape()[0] as usize
        } else {
            batch_size.unwrap_or(1)
        };

        trace!(batch_size = batch, "Stateless decoder predict");

        if let Some(y) = y {
            // y: [B, U] -> embeddings: [B, U, H]
            trace!(input_shape = ?y.shape(), "Predicting with tokens");
            self.predict_with_tokens(y, state, add_sos)
        } else {
            // No input tokens - return zero embedding (like pad/blank)
            trace!("Predicting without tokens (zero embedding)");
            let zeros = Array::zeros::<f32>(&[batch as i32, 1, self.pred_hidden as i32])?;
            let new_state = if let Some(s) = state {
                s.clone()
            } else {
                self.initialize_state(batch)?
            };

            if add_sos {
                // Prepend SOS (zero vector)
                let sos = Array::zeros::<f32>(&[batch as i32, 1, self.pred_hidden as i32])?;
                let output = ops::concatenate(&[&sos, &zeros], 1)?;
                trace!(output_shape = ?output.shape(), "Output with SOS prepended");
                Ok((output, new_state))
            } else {
                Ok((zeros, new_state))
            }
        }
    }

    /// Predict with actual input tokens.
    fn predict_with_tokens(
        &self,
        y: &Array,
        state: Option<&Array>,
        add_sos: bool,
    ) -> Result<(Array, Array)> {
        let batch = y.shape()[0] as usize;
        let seq_len = y.shape()[1] as usize;

        // Initialize or use provided state
        let prev_state = if let Some(s) = state {
            s.clone()
        } else {
            self.initialize_state(batch)?
        };

        // For stateless decoder, we need to look at context_size tokens
        // State has context_size - 1 previous tokens, y has current tokens

        // The stateless approach: for each position, sum embeddings of context window
        // This is a simplified version - NeMo's StatelessNet does this more efficiently

        // Build context by concatenating state and input
        // context: [batch, context_size - 1 + seq_len]
        let context = ops::concatenate(&[&prev_state, &y.as_type::<i32>()?], 1)?;

        // For each output position, we need embeddings of context_size tokens
        let mut outputs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Get context window ending at position i + context_size - 1
            let start = i;
            let end = i + self.context_size;

            // Slice context window
            let window = self.slice_context(&context, start, end)?;

            // Get embeddings and sum them
            let emb = self.embed(&window)?;
            let summed = ops::sum_axis(&emb, 1, false)?; // Sum over context dimension

            outputs.push(summed);
        }

        // Stack outputs: [batch, seq_len, pred_hidden]
        let output = ops::stack(&outputs.iter().collect::<Vec<_>>(), 1)?;

        // Update state: last context_size - 1 tokens from context
        let state_start = seq_len;
        let state_end = seq_len + self.context_size - 1;
        let new_state = self.slice_context(&context, state_start, state_end)?;

        if add_sos {
            // Prepend SOS (zero vector)
            let sos = Array::zeros::<f32>(&[batch as i32, 1, self.pred_hidden as i32])?;
            let output = ops::concatenate(&[&sos, &output], 1)?;
            Ok((output, new_state))
        } else {
            Ok((output, new_state))
        }
    }

    /// Single-step prediction for greedy/beam decoding.
    ///
    /// # Arguments
    /// * `y` - Single token index [batch, 1] or None for blank
    /// * `state` - Previous context state [batch, context_size - 1]
    ///
    /// # Returns
    /// * `(output, new_state)` where:
    ///   - output: [batch, 1, pred_hidden]
    ///   - new_state: [batch, context_size - 1]
    #[instrument(skip(self, y, state), level = "trace", fields(
        has_token = y.is_some(),
        state_shape = ?state.shape(),
        context_size = self.context_size
    ))]
    pub fn predict_one(&self, y: Option<&Array>, state: &Array) -> Result<(Array, Array)> {
        let _span = trace_span!("stateless_predict_one").entered();
        let batch = state.shape()[0] as usize;

        if let Some(y) = y {
            trace!(token_shape = ?y.shape(), "Single-step prediction with token");

            // Build full context: [state, y]
            let y_i32 = y.as_type::<i32>()?;
            let context = ops::concatenate(&[state, &y_i32], 1)?;

            // Get embeddings of last context_size tokens
            let ctx_start = 0;
            let ctx_end = self.context_size;
            let window = self.slice_context(&context, ctx_start, ctx_end)?;

            // Sum embeddings
            let emb = self.embed(&window)?;
            let output = ops::sum_axis(&emb, 1, true)?; // [batch, 1, pred_hidden]

            // New state: last context_size - 1 tokens
            let new_state = self.slice_context(&context, 1, self.context_size)?;

            trace!(output_shape = ?output.shape(), new_state_shape = ?new_state.shape(), "Prediction complete");
            Ok((output, new_state))
        } else {
            trace!("Single-step prediction with blank token");
            // No token - return blank embedding
            let fill_value = Array::from_int(self.blank_idx as i32);
            let blank_idx = Array::full::<i32>(&[batch as i32, 1], &fill_value)?;
            let blank_emb = self.embed(&blank_idx)?;

            // State stays the same when outputting blank
            trace!(output_shape = ?blank_emb.shape(), "Blank embedding returned");
            Ok((blank_emb, state.clone()))
        }
    }

    /// Slice a 2D array along axis 1.
    fn slice_context(&self, arr: &Array, start: usize, end: usize) -> Result<Array> {
        // Use IndexOp to slice: arr[:, start:end]
        Ok(arr.index((.., (start as i32)..(end as i32))))
    }

    /// Get embeddings for token indices.
    fn embed(&self, indices: &Array) -> Result<Array> {
        // indices: [batch, seq_len] -> [batch, seq_len, pred_hidden]
        // Use EmbeddingLayer::lookup which handles both standard and quantized embeddings
        self.embedding.lookup(indices)
    }

    /// Load weights from a checkpoint.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `prefix` - Key prefix for this layer
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // NeMo key: decoder.prediction.embed
        let prefixes = [
            format!("{}.prediction.embed", prefix),
            format!("{}.embed", prefix),
        ];
        if let Some(layer) = weights.fetch_embedding_any(&prefixes, quant)? {
            self.embedding = layer;
        }

        Ok(())
    }
}

/// LSTM-based RNN-T Decoder (Prediction Network).
///
/// A prediction network that uses an LSTM to model the label sequence.
/// This is used by models like parakeet-rnnt-0.6b.
///
/// The architecture is:
/// - Embedding layer: vocab_size + 1 -> pred_hidden
/// - LSTM layers (typically 2 layers)
///
/// State consists of (hidden, cell) tensors for each LSTM layer.
#[derive(Debug)]
pub struct RNNTDecoder {
    /// Embedding layer: vocab_size + 1 (blank) -> pred_hidden (supports quantized weights)
    embedding: EmbeddingLayer,
    /// LSTM layers - list of (weight_ih, weight_hh, bias_ih, bias_hh) (not quantized)
    lstm_layers: Vec<LstmLayer>,
    /// Hidden dimension of prediction network
    pred_hidden: usize,
    /// Number of LSTM layers
    num_layers: usize,
    /// Vocabulary size (excluding blank)
    vocab_size: usize,
    /// Blank token index (typically vocab_size)
    blank_idx: usize,
}

/// Weights for a single LSTM layer.
#[derive(Debug)]
struct LstmLayer {
    /// Input-hidden weight: [4 * hidden_size, input_size]
    weight_ih: Array,
    /// Hidden-hidden weight: [4 * hidden_size, hidden_size]
    weight_hh: Array,
    /// Input-hidden bias: [4 * hidden_size]
    bias_ih: Array,
    /// Hidden-hidden bias: [4 * hidden_size]
    bias_hh: Array,
    /// `[weight_ih, weight_hh]` transposed: [input_size + hidden_size, 4 * hidden_size]
    fused_weight_t: Array,
    /// `bias_ih + bias_hh`: [4 * hidden_size]
    fused_bias: Array,
}

impl LstmLayer {
    fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let gate_size = (4 * hidden_size) as i32;
        let mut layer = Self {
            weight_ih: Array::zeros::<f32>(&[gate_size, input_size as i32])?,
            weight_hh: Array::zeros::<f32>(&[gate_size, hidden_size as i32])?,
            bias_ih: Array::zeros::<f32>(&[gate_size])?,
            bias_hh: Array::zeros::<f32>(&[gate_size])?,
            fused_weight_t: Array::zeros::<f32>(&[(input_size + hidden_size) as i32, gate_size])?,
            fused_bias: Array::zeros::<f32>(&[gate_size])?,
        };
        layer.fuse()?;
        Ok(layer)
    }

    /// Combine the input and hidden projections so a step is one matmul.
    fn fuse(&mut self) -> Result<()> {
        self.fused_weight_t =
            ops::concatenate(&[&self.weight_ih, &self.weight_hh], 1)?.transpose()?;
        self.fused_bias = ops::add(&self.bias_ih, &self.bias_hh)?;
        mlx_rs::transforms::eval([&self.fused_weight_t, &self.fused_bias])?;
        Ok(())
    }
}

/// LSTM decoder state for a single batch item.
/// Contains hidden and cell states for each layer.
#[derive(Debug, Clone)]
pub struct LstmState {
    /// Hidden states per layer: [num_layers, batch, hidden_size]
    pub hidden: Array,
    /// Cell states per layer: [num_layers, batch, hidden_size]
    pub cell: Array,
}

impl RNNTDecoder {
    /// Create a new LSTM-based transducer decoder.
    ///
    /// # Arguments
    /// * `config` - Decoder configuration
    /// * `vocab_size` - Vocabulary size (excluding blank token)
    pub fn new(config: &DecoderConfig, vocab_size: usize) -> Result<Self> {
        let pred_hidden = config
            .prednet
            .as_ref()
            .map(|p| p.pred_hidden)
            .unwrap_or(640);

        let num_layers = config
            .prednet
            .as_ref()
            .and_then(|p| p.pred_rnn_layers)
            .unwrap_or(2);

        let blank_idx = vocab_size;

        // Initialize embedding with zeros (will be loaded from weights)
        // Shape: [vocab_size + 1, pred_hidden] (includes blank token)
        let embedding_weight = Array::zeros::<f32>(&[(vocab_size + 1) as i32, pred_hidden as i32])?;
        let embedding = EmbeddingLayer::standard(embedding_weight);

        // Initialize LSTM layers (not quantized)
        // All layers have the same input/hidden size for NeMo models
        let lstm_layers = (0..num_layers)
            .map(|_| LstmLayer::new(pred_hidden, pred_hidden))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embedding,
            lstm_layers,
            pred_hidden,
            num_layers,
            vocab_size,
            blank_idx,
        })
    }

    /// Get the prediction network hidden dimension.
    pub fn pred_hidden(&self) -> usize {
        self.pred_hidden
    }

    /// Get the blank token index.
    pub fn blank_idx(&self) -> usize {
        self.blank_idx
    }

    /// Get the vocabulary size (excluding blank).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the number of LSTM layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Initialize decoder state for a batch.
    ///
    /// # Arguments
    /// * `batch_size` - Number of sequences in the batch
    ///
    /// # Returns
    /// * LstmState with zero-initialized hidden and cell states
    #[instrument(skip(self), fields(batch_size, num_layers = self.num_layers, pred_hidden = self.pred_hidden))]
    pub fn initialize_state(&self, batch_size: usize) -> Result<LstmState> {
        // Hidden and cell states: [num_layers, batch, hidden_size]
        let hidden = Array::zeros::<f32>(&[
            self.num_layers as i32,
            batch_size as i32,
            self.pred_hidden as i32,
        ])?;
        let cell = Array::zeros::<f32>(&[
            self.num_layers as i32,
            batch_size as i32,
            self.pred_hidden as i32,
        ])?;
        debug!(
            hidden_shape = ?hidden.shape(),
            cell_shape = ?cell.shape(),
            "Initialized LSTM decoder state"
        );
        Ok(LstmState { hidden, cell })
    }

    /// Run a single-step prediction for decoding.
    ///
    /// # Arguments
    /// * `y` - Single token index [batch, 1] or None for blank/SOS
    /// * `state` - Previous LSTM state
    ///
    /// # Returns
    /// * `(output, new_state)` where:
    ///   - output: [batch, 1, pred_hidden]
    ///   - new_state: Updated LstmState
    #[instrument(skip(self, y, state), level = "trace", fields(
        has_token = y.is_some(),
        num_layers = self.num_layers,
        pred_hidden = self.pred_hidden
    ))]
    pub fn predict_one(&self, y: Option<&Array>, state: &LstmState) -> Result<(Array, LstmState)> {
        let _span = trace_span!("lstm_predict_one").entered();
        let batch_size = state.hidden.shape()[1] as usize;

        trace!(batch_size, state_hidden_shape = ?state.hidden.shape(), "LSTM single-step prediction");

        // Get embedding for input token
        let embedding = if let Some(y) = y {
            // y: [batch, 1] -> embedding: [batch, 1, pred_hidden]
            trace!(token_shape = ?y.shape(), "Embedding input token");
            self.embed(y)?
        } else {
            // Use blank token embedding
            trace!(blank_idx = self.blank_idx, "Embedding blank token");
            let fill_value = Array::from_int(self.blank_idx as i32);
            let blank_idx = Array::full::<i32>(&[batch_size as i32, 1], &fill_value)?;
            self.embed(&blank_idx)?
        };

        trace!(embedding_shape = ?embedding.shape(), "Embedding computed");

        // Run through LSTM layers
        // embedding: [batch, 1, pred_hidden]
        let mut x = embedding;
        let mut new_hiddens = Vec::with_capacity(self.num_layers);
        let mut new_cells = Vec::with_capacity(self.num_layers);

        for layer_idx in 0..self.num_layers {
            let _layer_span = trace_span!("lstm_layer", layer = layer_idx).entered();

            // Get layer state: [batch, hidden_size]
            let h = state.hidden.index((layer_idx as i32, .., ..));
            let c = state.cell.index((layer_idx as i32, .., ..));

            trace!(layer = layer_idx, h_shape = ?h.shape(), c_shape = ?c.shape(), "Processing LSTM layer");

            // Run LSTM cell for this layer
            let (new_h, new_c) = self.lstm_cell(layer_idx, &x.squeeze_axes(&[1])?, &h, &c)?;

            trace!(layer = layer_idx, new_h_shape = ?new_h.shape(), "LSTM layer complete");

            // Store new states
            new_hiddens.push(new_h.clone());
            new_cells.push(new_c.clone());

            // Output becomes input to next layer (add sequence dim back)
            x = ops::expand_dims(&new_h, 1)?;
        }

        // Stack states: [num_layers, batch, hidden_size]
        let new_hidden = ops::stack(&new_hiddens.iter().collect::<Vec<_>>(), 0)?;
        let new_cell = ops::stack(&new_cells.iter().collect::<Vec<_>>(), 0)?;

        let new_state = LstmState {
            hidden: new_hidden,
            cell: new_cell,
        };

        trace!(
            output_shape = ?x.shape(),
            new_hidden_shape = ?new_state.hidden.shape(),
            "LSTM prediction complete"
        );

        Ok((x, new_state))
    }

    /// Run LSTM cell for a single layer.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the LSTM layer
    /// * `x` - Input tensor [batch, hidden_size]
    /// * `h` - Previous hidden state [batch, hidden_size]
    /// * `c` - Previous cell state [batch, hidden_size]
    ///
    /// # Returns
    /// * `(new_h, new_c)` - New hidden and cell states
    fn lstm_cell(
        &self,
        layer_idx: usize,
        x: &Array,
        h: &Array,
        c: &Array,
    ) -> Result<(Array, Array)> {
        let _span = trace_span!("lstm_cell", layer = layer_idx).entered();
        let layer = &self.lstm_layers[layer_idx];

        trace!(
            layer = layer_idx,
            x_shape = ?x.shape(),
            h_shape = ?h.shape(),
            c_shape = ?c.shape(),
            weight_ih_shape = ?layer.weight_ih.shape(),
            "Computing LSTM cell gates"
        );

        // gates = [x, h] @ [weight_ih, weight_hh]^T + (bias_ih + bias_hh), split into i, f, g, o
        let xh = ops::concatenate(&[x, h], -1)?;
        let gates = ops::addmm(&layer.fused_bias, &xh, &layer.fused_weight_t, None, None)?;
        let parts = gates.split_equal(4, -1)?;
        let i = ops::sigmoid(&parts[0])?;
        let f = ops::sigmoid(&parts[1])?;
        let g = ops::tanh(&parts[2])?;
        let o = ops::sigmoid(&parts[3])?;

        // New cell state: c_new = f * c + i * g
        let new_c = ops::add(&ops::multiply(&f, c)?, &ops::multiply(&i, &g)?)?;

        // New hidden state: h_new = o * tanh(c_new)
        let new_h = ops::multiply(&o, &ops::tanh(&new_c)?)?;

        trace!(
            layer = layer_idx,
            new_h_shape = ?new_h.shape(),
            new_c_shape = ?new_c.shape(),
            "LSTM cell complete"
        );

        Ok((new_h, new_c))
    }

    /// Get embeddings for token indices.
    fn embed(&self, indices: &Array) -> Result<Array> {
        // indices: [batch, seq_len] -> [batch, seq_len, pred_hidden]
        // Use EmbeddingLayer::lookup which handles both standard and quantized embeddings
        self.embedding.lookup(indices)
    }

    /// Load weights from a checkpoint.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `prefix` - Key prefix for this layer
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // NeMo/PyTorch keys:
        // decoder.prediction.embed.weight: [vocab_size + 1, pred_hidden]
        // decoder.prediction.dec_rnn.lstm.weight_ih_l0: [4 * hidden, input_size]
        // decoder.prediction.dec_rnn.lstm.weight_hh_l0: [4 * hidden, hidden_size]
        // decoder.prediction.dec_rnn.lstm.bias_ih_l0: [4 * hidden]
        // decoder.prediction.dec_rnn.lstm.bias_hh_l0: [4 * hidden]
        // (and l1 for second layer, etc.)
        //
        // MLX keys (alternative naming):
        // decoder.prediction.dec_rnn.lstm.0.Wx: [4 * hidden, input_size] (= weight_ih)
        // decoder.prediction.dec_rnn.lstm.0.Wh: [4 * hidden, hidden_size] (= weight_hh)
        // decoder.prediction.dec_rnn.lstm.0.bias: [4 * hidden] (combined bias)
        // (and .1 for second layer, etc.)

        // Load embedding (may be quantized)
        if let Some(layer) =
            weights.fetch_embedding_any(&[format!("{}.prediction.embed", prefix)], quant)?
        {
            self.embedding = layer;
        }

        // Load LSTM layers, in PyTorch or MLX naming
        let lstm = format!("{}.prediction.dec_rnn.lstm", prefix);
        for (i, layer) in self.lstm_layers.iter_mut().enumerate() {
            for (param, keys) in [
                (
                    &mut layer.weight_ih,
                    [format!("{lstm}.weight_ih_l{i}"), format!("{lstm}.{i}.Wx")],
                ),
                (
                    &mut layer.weight_hh,
                    [format!("{lstm}.weight_hh_l{i}"), format!("{lstm}.{i}.Wh")],
                ),
            ] {
                if let Some(value) = weights.fetch_any(&keys)? {
                    *param = value.clone();
                }
            }

            // PyTorch keeps separate input and hidden biases. MLX stores their
            // sum, the only thing the LSTM cell uses (gates = x @ W_ih^T + b_ih
            // + h @ W_hh^T + b_hh), so it becomes b_ih with a zero b_hh.
            match weights.get(&format!("{lstm}.{i}.bias")) {
                Some(bias) => {
                    layer.bias_ih = bias.clone();
                    layer.bias_hh = Array::zeros::<f32>(&[bias.dim(0)])?;
                }
                None => {
                    if let Some(bias) = weights.fetch(&format!("{lstm}.bias_ih_l{i}"))? {
                        layer.bias_ih = bias.clone();
                    }
                    if let Some(bias) = weights.fetch(&format!("{lstm}.bias_hh_l{i}"))? {
                        layer.bias_hh = bias.clone();
                    }
                }
            }
            layer.fuse()?;
        }

        Ok(())
    }
}

/// Activation function for joint network.
#[derive(Debug, Clone, Copy)]
pub enum JointActivation {
    /// `max(x, 0)`
    Relu,
    /// `tanh(x)`
    Tanh,
    /// `1 / (1 + e^-x)`
    Sigmoid,
}

impl JointActivation {
    /// Parse activation from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "tanh" => Self::Tanh,
            "sigmoid" => Self::Sigmoid,
            _ => Self::Relu, // Default to ReLU
        }
    }

    /// Apply activation function.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Relu => mlx_rs::ops::maximum(x, &Array::from_f32(0.0))
                .map_err(|e| Error::model(format!("ReLU failed: {}", e))),
            Self::Tanh => {
                mlx_rs::ops::tanh(x).map_err(|e| Error::model(format!("Tanh failed: {}", e)))
            }
            Self::Sigmoid => {
                mlx_rs::ops::sigmoid(x).map_err(|e| Error::model(format!("Sigmoid failed: {}", e)))
            }
        }
    }
}

/// RNN-T Joint Network.
///
/// Combines encoder output (acoustic representation) and prediction network output
/// (label representation) to produce output logits.
///
/// The joint network computes:
/// ```text
/// z = activation(encoder_proj(h) + pred_proj(g))
/// logits = output_proj(z)
/// ```
///
/// where h is encoder output and g is prediction network output.
#[derive(Debug)]
pub struct RNNTJoint {
    /// Language-ID conditioning applied to encoder frames before `encoder_proj`
    /// (prompt-conditioned models only)
    prompt: Option<LanguagePrompt>,
    /// Projection from encoder hidden to joint hidden (supports quantized weights)
    encoder_proj: LinearLayer,
    /// Projection from prediction hidden to joint hidden (supports quantized weights)
    pred_proj: LinearLayer,
    /// Output projection from joint hidden to vocabulary (supports quantized weights)
    joint_proj: LinearLayer,
    /// Activation function
    activation: JointActivation,
    /// Joint hidden dimension
    joint_hidden: usize,
    /// Vocabulary size including blank
    vocab_size: usize,
}

impl RNNTJoint {
    /// Create a new joint network.
    ///
    /// # Arguments
    /// * `config` - Joint network configuration
    /// * `encoder_hidden` - Encoder output dimension
    /// * `pred_hidden` - Prediction network output dimension
    /// * `vocab_size` - Vocabulary size (including blank token)
    pub fn new(
        config: &JointConfig,
        encoder_hidden: usize,
        pred_hidden: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let joint_hidden = config.joint_hidden;
        let activation = JointActivation::from_str(&config.activation);
        tracing::debug!(
            "RNNTJoint::new encoder_hidden {} pred_hidden {} vocab_size {} activation {:?}",
            encoder_hidden,
            pred_hidden,
            vocab_size,
            activation
        );

        // Create zero-initialized linear layers (will be loaded from checkpoint)
        let create_linear = |out_dim: usize, in_dim: usize| -> Result<LinearLayer> {
            let weight = Array::zeros::<f32>(&[out_dim as i32, in_dim as i32])?;
            let bias = Some(Array::zeros::<f32>(&[out_dim as i32])?);
            Ok(LinearLayer::standard(weight, bias))
        };

        let encoder_proj = create_linear(joint_hidden, encoder_hidden)?;
        let pred_proj = create_linear(joint_hidden, pred_hidden)?;
        let joint_proj = create_linear(vocab_size, joint_hidden)?;

        Ok(Self {
            prompt: None,
            encoder_proj,
            pred_proj,
            joint_proj,
            activation,
            joint_hidden,
            vocab_size,
        })
    }

    /// Get vocabulary size (including blank).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Condition every encoder frame on a language prompt before projecting it.
    pub fn set_prompt(&mut self, prompt: Option<LanguagePrompt>) {
        self.prompt = prompt;
    }

    /// The language prompt, for prompt-conditioned models.
    pub fn prompt(&self) -> Option<&LanguagePrompt> {
        self.prompt.as_ref()
    }

    /// Mutable language prompt, e.g. to change the target language.
    pub fn prompt_mut(&mut self) -> Option<&mut LanguagePrompt> {
        self.prompt.as_mut()
    }

    /// Joint forward pass.
    ///
    /// # Arguments
    /// * `encoder_output` - Encoder output [batch, time, encoder_hidden]
    /// * `pred_output` - Prediction network output [batch, labels, pred_hidden]
    ///
    /// # Returns
    /// * Log probabilities [batch, time, labels, vocab_size]
    #[instrument(skip(self, encoder_output, pred_output), fields(
        encoder_shape = ?encoder_output.shape(),
        pred_shape = ?pred_output.shape(),
        joint_hidden = self.joint_hidden,
        vocab_size = self.vocab_size
    ))]
    pub fn forward(&self, encoder_output: &Array, pred_output: &Array) -> Result<Array> {
        let _span = trace_span!("rnnt_joint_forward").entered();

        // encoder_output: [B, T, H_enc]
        // pred_output: [B, U, H_pred]

        let batch = encoder_output.shape()[0];
        let time = encoder_output.shape()[1];
        let labels = pred_output.shape()[1];

        debug!(
            batch,
            time,
            labels,
            joint_hidden = self.joint_hidden,
            vocab_size = self.vocab_size,
            "Joint forward pass"
        );

        // Project encoder: [B, T, joint_hidden]
        trace!("Projecting encoder output");
        let enc_proj = self.project_encoder(encoder_output)?;
        trace!(enc_proj_shape = ?enc_proj.shape(), "Encoder projection complete");

        // Project prediction: [B, U, joint_hidden]
        trace!("Projecting prediction output");
        let pred_proj = self.pred_proj.forward(pred_output)?;
        trace!(pred_proj_shape = ?pred_proj.shape(), "Prediction projection complete");

        // Broadcast add: [B, T, 1, H] + [B, 1, U, H] -> [B, T, U, H]
        // Expand dims for broadcasting
        let enc_expanded = ops::expand_dims(&enc_proj, 2)?; // [B, T, 1, H]
        let pred_expanded = ops::expand_dims(&pred_proj, 1)?; // [B, 1, U, H]

        // Add with broadcasting
        trace!("Broadcasting encoder and prediction projections");
        let joint = ops::add(&enc_expanded, &pred_expanded)?; // [B, T, U, H]
        trace!(joint_shape = ?joint.shape(), "Joint tensor computed");

        // Apply activation
        trace!(activation = ?self.activation, "Applying activation");
        let activated = self.activation.apply(&joint)?;

        // Output projection: [B, T, U, vocab_size]
        // Need to reshape for linear, then reshape back
        trace!("Computing output projection");
        let joint_flat = ops::reshape(
            &activated,
            &[batch * time * labels, self.joint_hidden as i32],
        )?;

        let logits_flat = self.joint_proj.forward(&joint_flat)?;

        let logits = ops::reshape(&logits_flat, &[batch, time, labels, self.vocab_size as i32])?;
        trace!(logits_shape = ?logits.shape(), "Logits computed");

        // Log softmax over vocabulary dimension
        trace!("Applying log softmax");
        let log_probs = mlx_rs::nn::log_softmax(&logits, -1)?;

        debug!(output_shape = ?log_probs.shape(), "Joint forward complete");

        Ok(log_probs)
    }

    /// Single-step joint forward for decoding.
    ///
    /// # Arguments
    /// * `encoder_output` - Single encoder frame [batch, 1, encoder_hidden]
    /// * `pred_output` - Single prediction output [batch, 1, pred_hidden]
    ///
    /// # Returns
    /// * Log probabilities [batch, vocab_size]
    pub fn forward_one(&self, encoder_output: &Array, pred_output: &Array) -> Result<Array> {
        // encoder_output: [B, 1, H_enc] or [B, H_enc]
        // pred_output: [B, 1, H_pred] or [B, H_pred]
        let enc = if encoder_output.ndim() == 3 {
            encoder_output.squeeze_axes(&[1])?
        } else {
            encoder_output.clone()
        };
        let pred = if pred_output.ndim() == 3 {
            pred_output.squeeze_axes(&[1])?
        } else {
            pred_output.clone()
        };

        let logits = self.logits_from_projections(
            &self.project_encoder(&enc)?,
            &self.project_prediction(&pred)?,
        )?;
        Ok(mlx_rs::nn::log_softmax(&logits, -1)?)
    }

    /// Project encoder output into the joint space: [..., encoder_hidden] -> [..., joint_hidden].
    ///
    /// Greedy decoding projects every frame once up front instead of per step.
    /// Prompt-conditioned models apply the language prompt first.
    pub fn project_encoder(&self, encoder_output: &Array) -> Result<Array> {
        match &self.prompt {
            Some(prompt) => self.encoder_proj.forward(&prompt.forward(encoder_output)?),
            None => self.encoder_proj.forward(encoder_output),
        }
    }

    /// Project prediction output into the joint space: [..., pred_hidden] -> [..., joint_hidden].
    pub fn project_prediction(&self, pred_output: &Array) -> Result<Array> {
        self.pred_proj.forward(pred_output)
    }

    /// Logits (not normalized) from projected encoder and prediction outputs.
    ///
    /// The projections broadcast against each other, e.g. [T, H] + [1, H] -> [T, vocab_size].
    pub fn logits_from_projections(&self, enc_proj: &Array, pred_proj: &Array) -> Result<Array> {
        let joint = ops::add(enc_proj, pred_proj)?;
        let activated = self.activation.apply(&joint)?;
        self.joint_proj.forward(&activated)
    }

    /// Load weights from a checkpoint.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `prefix` - Key prefix for this layer
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &Weights,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // NeMo keys:
        // joint.enc.weight, joint.enc.bias (or joint.encoder.weight)
        // joint.pred.weight, joint.pred.bias (or joint.prediction.weight)
        // joint.joint_net.0.weight, joint.joint_net.0.bias (linear)
        // joint.joint_net.2.weight, joint.joint_net.2.bias (output)

        // Each layer under the first of several key patterns the checkpoint has
        for (layer, names) in [
            (&mut self.encoder_proj, ["enc", "encoder", "joint_net.0"]),
            (&mut self.pred_proj, ["pred", "prediction", "joint_net.1"]),
            (
                &mut self.joint_proj,
                ["joint_net.2", "joint_net.output", "out"],
            ),
        ] {
            let prefixes = names.map(|n| format!("{prefix}.{n}"));
            if let Some(loaded) = weights.fetch_linear_any(&prefixes, quant)? {
                *layer = loaded;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::config::PredNetConfig;

    #[test]
    fn test_stateless_decoder_init() {
        let config = DecoderConfig {
            prednet: Some(PredNetConfig {
                pred_hidden: 256,
                pred_rnn_layers: None,
                t_max: None,
                dropout: None,
            }),
            context_size: Some(2),
            ..Default::default()
        };

        let decoder = StatelessTransducerDecoder::new(&config, 1000).unwrap();
        assert_eq!(decoder.pred_hidden(), 256);
        assert_eq!(decoder.context_size(), 2);
        assert_eq!(decoder.vocab_size(), 1000);
        assert_eq!(decoder.blank_idx(), 1000);
    }

    #[test]
    fn test_stateless_decoder_init_state() {
        let config = DecoderConfig {
            prednet: Some(PredNetConfig {
                pred_hidden: 256,
                pred_rnn_layers: None,
                t_max: None,
                dropout: None,
            }),
            context_size: Some(2),
            ..Default::default()
        };

        let decoder = StatelessTransducerDecoder::new(&config, 1000).unwrap();
        let state = decoder.initialize_state(4).unwrap();

        assert_eq!(state.shape(), &[4, 1]); // context_size - 1 = 1
    }

    #[test]
    fn test_joint_init() {
        let config = JointConfig {
            joint_hidden: 512,
            activation: "relu".to_string(),
            ..Default::default()
        };

        let joint = RNNTJoint::new(&config, 640, 256, 1001).unwrap();
        assert_eq!(joint.vocab_size(), 1001);
        assert_eq!(joint.joint_hidden, 512);
    }
}
