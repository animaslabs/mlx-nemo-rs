//! RNN-T decoder components.
//!
//! This module implements the Transducer decoder components:
//! - `StatelessTransducerDecoder`: The prediction network (label model)
//! - `RNNTJoint`: The joint network that combines encoder and prediction outputs

use std::collections::HashMap;

use mlx_rs::ops::{self, indexing, indexing::IndexOp};
use mlx_rs::Array;

use crate::asr::config::{DecoderConfig, JointConfig};
use crate::core::error::{Error, Result};

/// Stateless Transducer Decoder (Prediction Network).
///
/// A prediction network that uses embeddings of previous tokens (context)
/// to predict the next token. Unlike LSTM-based decoders, this is stateless
/// and simply concatenates/sums embeddings of the context window.
///
/// This is the decoder used by Parakeet RNNT models.
pub struct StatelessTransducerDecoder {
    /// Embedding layer: vocab_size + 1 (blank) -> pred_hidden
    embedding_weight: Array,
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

        Ok(Self {
            embedding_weight,
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
    pub fn initialize_state(&self, batch_size: usize) -> Result<Array> {
        // State contains context_size - 1 previous tokens (the current token completes it)
        let state_len = self.context_size.saturating_sub(1).max(1);
        // Create state filled with blank_idx
        let fill_value = Array::from_int(self.blank_idx as i32);
        let state = Array::full::<i32>(&[batch_size as i32, state_len as i32], &fill_value)?;
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

        if let Some(y) = y {
            // y: [B, U] -> embeddings: [B, U, H]
            self.predict_with_tokens(y, state, add_sos)
        } else {
            // No input tokens - return zero embedding (like pad/blank)
            let zeros = Array::zeros::<f32>(&[batch as i32, 1, self.pred_hidden as i32])?;
            let new_state = if let Some(s) = state {
                s.clone()
            } else {
                self.initialize_state(batch)?
            };

            if add_sos {
                // Prepend SOS (zero vector)
                let sos = Array::zeros::<f32>(&[batch as i32, 1, self.pred_hidden as i32])?;
                let output = ops::concatenate_axis(&[&sos, &zeros], 1)?;
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
        let context = ops::concatenate_axis(&[&prev_state, &y.as_type::<i32>()?], 1)?;

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
        let output = ops::stack_axis(&outputs.iter().collect::<Vec<_>>(), 1)?;

        // Update state: last context_size - 1 tokens from context
        let state_start = seq_len;
        let state_end = seq_len + self.context_size - 1;
        let new_state = self.slice_context(&context, state_start, state_end)?;

        if add_sos {
            // Prepend SOS (zero vector)
            let sos = Array::zeros::<f32>(&[batch as i32, 1, self.pred_hidden as i32])?;
            let output = ops::concatenate_axis(&[&sos, &output], 1)?;
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
    pub fn predict_one(&self, y: Option<&Array>, state: &Array) -> Result<(Array, Array)> {
        let batch = state.shape()[0] as usize;

        if let Some(y) = y {
            // Build full context: [state, y]
            let y_i32 = y.as_type::<i32>()?;
            let context = ops::concatenate_axis(&[state, &y_i32], 1)?;

            // Get embeddings of last context_size tokens
            let ctx_start = 0;
            let ctx_end = self.context_size;
            let window = self.slice_context(&context, ctx_start, ctx_end)?;

            // Sum embeddings
            let emb = self.embed(&window)?;
            let output = ops::sum_axis(&emb, 1, true)?; // [batch, 1, pred_hidden]

            // New state: last context_size - 1 tokens
            let new_state = self.slice_context(&context, 1, self.context_size)?;

            Ok((output, new_state))
        } else {
            // No token - return blank embedding
            let fill_value = Array::from_int(self.blank_idx as i32);
            let blank_idx = Array::full::<i32>(&[batch as i32, 1], &fill_value)?;
            let blank_emb = self.embed(&blank_idx)?;

            // State stays the same when outputting blank
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
        let batch = indices.shape()[0];
        let seq_len = indices.shape()[1];

        // Flatten indices to [batch * seq_len]
        let flat_indices = ops::reshape(indices, &[-1])?;

        // Do embedding lookup using take_axis on axis 0
        // embedding_weight: [vocab_size, pred_hidden]
        // flat_indices: [batch * seq_len]
        // result: [batch * seq_len, pred_hidden]
        let flat_emb = indexing::take_axis(&self.embedding_weight, &flat_indices, 0)?;

        // Reshape back to [batch, seq_len, pred_hidden]
        let emb = ops::reshape(&flat_emb, &[batch, seq_len, self.pred_hidden as i32])?;

        Ok(emb)
    }

    /// Load weights from a checkpoint.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        // NeMo key: decoder.prediction.embed.weight
        let embed_key = format!("{}.prediction.embed.weight", prefix);

        if let Some(weight) = weights.get(&embed_key) {
            self.embedding_weight = weight.clone();
            tracing::debug!("Loaded prediction embedding: {:?}", weight.shape());
        } else {
            // Try alternative key patterns
            let alt_key = format!("{}.embed.weight", prefix);
            if let Some(weight) = weights.get(&alt_key) {
                self.embedding_weight = weight.clone();
                tracing::debug!("Loaded prediction embedding (alt): {:?}", weight.shape());
            } else {
                tracing::warn!(
                    "Prediction embedding not found at {} or {}",
                    embed_key,
                    alt_key
                );
            }
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
pub struct RNNTDecoder {
    /// Embedding layer: vocab_size + 1 (blank) -> pred_hidden
    embedding_weight: Array,
    /// LSTM layers - list of (weight_ih, weight_hh, bias_ih, bias_hh)
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
struct LstmLayer {
    /// Input-hidden weight: [4 * hidden_size, input_size]
    weight_ih: Array,
    /// Hidden-hidden weight: [4 * hidden_size, hidden_size]
    weight_hh: Array,
    /// Input-hidden bias: [4 * hidden_size]
    bias_ih: Array,
    /// Hidden-hidden bias: [4 * hidden_size]
    bias_hh: Array,
}

/// LSTM decoder state for a single batch item.
/// Contains hidden and cell states for each layer.
#[derive(Clone)]
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

        // Initialize LSTM layers
        let mut lstm_layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // All layers have the same input/hidden size for NeMo models
            let gate_size = 4 * pred_hidden;
            lstm_layers.push(LstmLayer {
                weight_ih: Array::zeros::<f32>(&[gate_size as i32, pred_hidden as i32])?,
                weight_hh: Array::zeros::<f32>(&[gate_size as i32, pred_hidden as i32])?,
                bias_ih: Array::zeros::<f32>(&[gate_size as i32])?,
                bias_hh: Array::zeros::<f32>(&[gate_size as i32])?,
            });
        }

        Ok(Self {
            embedding_weight,
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
    pub fn predict_one(&self, y: Option<&Array>, state: &LstmState) -> Result<(Array, LstmState)> {
        let batch_size = state.hidden.shape()[1] as usize;

        // Get embedding for input token
        let embedding = if let Some(y) = y {
            // y: [batch, 1] -> embedding: [batch, 1, pred_hidden]
            self.embed(y)?
        } else {
            // Use blank token embedding
            let fill_value = Array::from_int(self.blank_idx as i32);
            let blank_idx = Array::full::<i32>(&[batch_size as i32, 1], &fill_value)?;
            self.embed(&blank_idx)?
        };

        // Run through LSTM layers
        // embedding: [batch, 1, pred_hidden]
        let mut x = embedding;
        let mut new_hiddens = Vec::with_capacity(self.num_layers);
        let mut new_cells = Vec::with_capacity(self.num_layers);

        for layer_idx in 0..self.num_layers {
            // Get layer state: [batch, hidden_size]
            let h = state.hidden.index((layer_idx as i32, .., ..));
            let c = state.cell.index((layer_idx as i32, .., ..));

            // Run LSTM cell for this layer
            let (new_h, new_c) = self.lstm_cell(layer_idx, &x.squeeze_axes(&[1])?, &h, &c)?;

            // Store new states
            new_hiddens.push(new_h.clone());
            new_cells.push(new_c.clone());

            // Output becomes input to next layer (add sequence dim back)
            x = ops::expand_dims(&new_h, 1)?;
        }

        // Stack states: [num_layers, batch, hidden_size]
        let new_hidden = ops::stack_axis(&new_hiddens.iter().collect::<Vec<_>>(), 0)?;
        let new_cell = ops::stack_axis(&new_cells.iter().collect::<Vec<_>>(), 0)?;

        let new_state = LstmState {
            hidden: new_hidden,
            cell: new_cell,
        };

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
        let layer = &self.lstm_layers[layer_idx];

        // Compute gates: i, f, g, o
        // gates = x @ weight_ih^T + bias_ih + h @ weight_hh^T + bias_hh
        let weight_ih_t = layer.weight_ih.transpose()?;
        let weight_hh_t = layer.weight_hh.transpose()?;

        let gates_ih = ops::add(&ops::matmul(x, &weight_ih_t)?, &layer.bias_ih)?;
        let gates_hh = ops::add(&ops::matmul(h, &weight_hh_t)?, &layer.bias_hh)?;
        let gates = ops::add(&gates_ih, &gates_hh)?;

        // Split into 4 gates: [batch, hidden_size] each
        let hidden_size = self.pred_hidden as i32;
        let i = ops::sigmoid(&gates.index((.., 0..hidden_size)))?;
        let f = ops::sigmoid(&gates.index((.., hidden_size..(2 * hidden_size))))?;
        let g = ops::tanh(&gates.index((.., (2 * hidden_size)..(3 * hidden_size))))?;
        let o = ops::sigmoid(&gates.index((.., (3 * hidden_size)..)))?;

        // New cell state: c_new = f * c + i * g
        let new_c = ops::add(&ops::multiply(&f, c)?, &ops::multiply(&i, &g)?)?;

        // New hidden state: h_new = o * tanh(c_new)
        let new_h = ops::multiply(&o, &ops::tanh(&new_c)?)?;

        Ok((new_h, new_c))
    }

    /// Get embeddings for token indices.
    fn embed(&self, indices: &Array) -> Result<Array> {
        // indices: [batch, seq_len] -> [batch, seq_len, pred_hidden]
        let batch = indices.shape()[0];
        let seq_len = indices.shape()[1];

        // Flatten indices to [batch * seq_len]
        let flat_indices = ops::reshape(indices, &[-1])?;

        // Do embedding lookup using take_axis on axis 0
        // embedding_weight: [vocab_size, pred_hidden]
        // flat_indices: [batch * seq_len]
        // result: [batch * seq_len, pred_hidden]
        let flat_emb = indexing::take_axis(&self.embedding_weight, &flat_indices, 0)?;

        // Reshape back to [batch, seq_len, pred_hidden]
        let emb = ops::reshape(&flat_emb, &[batch, seq_len, self.pred_hidden as i32])?;

        Ok(emb)
    }

    /// Load weights from a checkpoint.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        // NeMo keys:
        // decoder.prediction.embed.weight: [vocab_size + 1, pred_hidden]
        // decoder.prediction.dec_rnn.lstm.weight_ih_l0: [4 * hidden, input_size]
        // decoder.prediction.dec_rnn.lstm.weight_hh_l0: [4 * hidden, hidden_size]
        // decoder.prediction.dec_rnn.lstm.bias_ih_l0: [4 * hidden]
        // decoder.prediction.dec_rnn.lstm.bias_hh_l0: [4 * hidden]
        // (and l1 for second layer, etc.)

        // Load embedding
        let embed_key = format!("{}.prediction.embed.weight", prefix);
        if let Some(weight) = weights.get(&embed_key) {
            self.embedding_weight = weight.clone();
            tracing::debug!("Loaded prediction embedding: {:?}", weight.shape());
        } else {
            tracing::warn!("Prediction embedding not found at {}", embed_key);
        }

        // Load LSTM layers
        for layer_idx in 0..self.num_layers {
            let weight_ih_key = format!(
                "{}.prediction.dec_rnn.lstm.weight_ih_l{}",
                prefix, layer_idx
            );
            let weight_hh_key = format!(
                "{}.prediction.dec_rnn.lstm.weight_hh_l{}",
                prefix, layer_idx
            );
            let bias_ih_key = format!("{}.prediction.dec_rnn.lstm.bias_ih_l{}", prefix, layer_idx);
            let bias_hh_key = format!("{}.prediction.dec_rnn.lstm.bias_hh_l{}", prefix, layer_idx);

            if let Some(w) = weights.get(&weight_ih_key) {
                self.lstm_layers[layer_idx].weight_ih = w.clone();
                tracing::debug!("Loaded LSTM layer {} weight_ih: {:?}", layer_idx, w.shape());
            } else {
                tracing::warn!("LSTM weight_ih not found at {}", weight_ih_key);
            }

            if let Some(w) = weights.get(&weight_hh_key) {
                self.lstm_layers[layer_idx].weight_hh = w.clone();
                tracing::debug!("Loaded LSTM layer {} weight_hh: {:?}", layer_idx, w.shape());
            } else {
                tracing::warn!("LSTM weight_hh not found at {}", weight_hh_key);
            }

            if let Some(b) = weights.get(&bias_ih_key) {
                self.lstm_layers[layer_idx].bias_ih = b.clone();
                tracing::debug!("Loaded LSTM layer {} bias_ih: {:?}", layer_idx, b.shape());
            } else {
                tracing::warn!("LSTM bias_ih not found at {}", bias_ih_key);
            }

            if let Some(b) = weights.get(&bias_hh_key) {
                self.lstm_layers[layer_idx].bias_hh = b.clone();
                tracing::debug!("Loaded LSTM layer {} bias_hh: {:?}", layer_idx, b.shape());
            } else {
                tracing::warn!("LSTM bias_hh not found at {}", bias_hh_key);
            }
        }

        Ok(())
    }
}

/// Activation function for joint network.
#[derive(Debug, Clone, Copy)]
pub enum JointActivation {
    Relu,
    Tanh,
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
pub struct RNNTJoint {
    /// Projection from encoder hidden to joint hidden
    encoder_proj_weight: Array,
    encoder_proj_bias: Option<Array>,
    /// Projection from prediction hidden to joint hidden
    pred_proj_weight: Array,
    pred_proj_bias: Option<Array>,
    /// Output projection from joint hidden to vocabulary
    joint_proj_weight: Array,
    joint_proj_bias: Option<Array>,
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

        // Initialize weights with zeros (will be loaded from checkpoint)
        let encoder_proj_weight =
            Array::zeros::<f32>(&[joint_hidden as i32, encoder_hidden as i32])?;
        let encoder_proj_bias = Some(Array::zeros::<f32>(&[joint_hidden as i32])?);

        let pred_proj_weight = Array::zeros::<f32>(&[joint_hidden as i32, pred_hidden as i32])?;
        let pred_proj_bias = Some(Array::zeros::<f32>(&[joint_hidden as i32])?);

        let joint_proj_weight = Array::zeros::<f32>(&[vocab_size as i32, joint_hidden as i32])?;
        let joint_proj_bias = Some(Array::zeros::<f32>(&[vocab_size as i32])?);

        Ok(Self {
            encoder_proj_weight,
            encoder_proj_bias,
            pred_proj_weight,
            pred_proj_bias,
            joint_proj_weight,
            joint_proj_bias,
            activation,
            joint_hidden,
            vocab_size,
        })
    }

    /// Get vocabulary size (including blank).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Joint forward pass.
    ///
    /// # Arguments
    /// * `encoder_output` - Encoder output [batch, time, encoder_hidden]
    /// * `pred_output` - Prediction network output [batch, labels, pred_hidden]
    ///
    /// # Returns
    /// * Log probabilities [batch, time, labels, vocab_size]
    pub fn forward(&self, encoder_output: &Array, pred_output: &Array) -> Result<Array> {
        // encoder_output: [B, T, H_enc]
        // pred_output: [B, U, H_pred]

        let batch = encoder_output.shape()[0];
        let time = encoder_output.shape()[1];
        let labels = pred_output.shape()[1];

        // Project encoder: [B, T, joint_hidden]
        let enc_proj = self.linear(
            encoder_output,
            &self.encoder_proj_weight,
            self.encoder_proj_bias.as_ref(),
        )?;

        // Project prediction: [B, U, joint_hidden]
        let pred_proj = self.linear(
            pred_output,
            &self.pred_proj_weight,
            self.pred_proj_bias.as_ref(),
        )?;

        // Broadcast add: [B, T, 1, H] + [B, 1, U, H] -> [B, T, U, H]
        // Expand dims for broadcasting
        let enc_expanded = ops::expand_dims(&enc_proj, 2)?; // [B, T, 1, H]
        let pred_expanded = ops::expand_dims(&pred_proj, 1)?; // [B, 1, U, H]

        // Add with broadcasting
        let joint = ops::add(&enc_expanded, &pred_expanded)?; // [B, T, U, H]

        // Apply activation
        let activated = self.activation.apply(&joint)?;

        // Output projection: [B, T, U, vocab_size]
        // Need to reshape for linear, then reshape back
        let joint_flat = ops::reshape(
            &activated,
            &[batch * time * labels, self.joint_hidden as i32],
        )?;

        let logits_flat = self.linear(
            &joint_flat,
            &self.joint_proj_weight,
            self.joint_proj_bias.as_ref(),
        )?;

        let logits = ops::reshape(&logits_flat, &[batch, time, labels, self.vocab_size as i32])?;

        // Log softmax over vocabulary dimension
        let log_probs = mlx_rs::nn::log_softmax(&logits, -1)?;

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

        // Ensure 2D for linear
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

        // Project encoder: [B, joint_hidden]
        let enc_proj = self.linear(
            &enc,
            &self.encoder_proj_weight,
            self.encoder_proj_bias.as_ref(),
        )?;

        // Project prediction: [B, joint_hidden]
        let pred_proj = self.linear(&pred, &self.pred_proj_weight, self.pred_proj_bias.as_ref())?;

        // Add
        let joint = ops::add(&enc_proj, &pred_proj)?;

        // Apply activation
        let activated = self.activation.apply(&joint)?;

        // Output projection: [B, vocab_size]
        let logits = self.linear(
            &activated,
            &self.joint_proj_weight,
            self.joint_proj_bias.as_ref(),
        )?;

        // Log softmax
        let log_probs = mlx_rs::nn::log_softmax(&logits, -1)?;

        Ok(log_probs)
    }

    /// Linear transformation: y = xW^T + b
    fn linear(&self, x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
        // x: [..., in_features], weight: [out_features, in_features]
        // result: [..., out_features]
        // Transpose weight: [out, in] -> [in, out]
        let weight_t = weight.transpose()?;
        let mut result = ops::matmul(x, &weight_t)?;

        if let Some(b) = bias {
            result = ops::add(&result, b)?;
        }

        Ok(result)
    }

    /// Load weights from a checkpoint.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        // NeMo keys:
        // joint.enc.weight, joint.enc.bias (or joint.encoder.weight)
        // joint.pred.weight, joint.pred.bias (or joint.prediction.weight)
        // joint.joint_net.0.weight, joint.joint_net.0.bias (linear)
        // joint.joint_net.2.weight, joint.joint_net.2.bias (output)

        // Try different key patterns
        let enc_weight_keys = [
            format!("{}.enc.weight", prefix),
            format!("{}.encoder.weight", prefix),
            format!("{}.joint_net.0.weight", prefix),
        ];
        let enc_bias_keys = [
            format!("{}.enc.bias", prefix),
            format!("{}.encoder.bias", prefix),
            format!("{}.joint_net.0.bias", prefix),
        ];

        for key in &enc_weight_keys {
            if let Some(w) = weights.get(key) {
                self.encoder_proj_weight = w.clone();
                tracing::debug!("Loaded joint encoder weight from {}: {:?}", key, w.shape());
                break;
            }
        }

        for key in &enc_bias_keys {
            if let Some(b) = weights.get(key) {
                self.encoder_proj_bias = Some(b.clone());
                tracing::debug!("Loaded joint encoder bias from {}", key);
                break;
            }
        }

        let pred_weight_keys = [
            format!("{}.pred.weight", prefix),
            format!("{}.prediction.weight", prefix),
            format!("{}.joint_net.1.weight", prefix),
        ];
        let pred_bias_keys = [
            format!("{}.pred.bias", prefix),
            format!("{}.prediction.bias", prefix),
            format!("{}.joint_net.1.bias", prefix),
        ];

        for key in &pred_weight_keys {
            if let Some(w) = weights.get(key) {
                self.pred_proj_weight = w.clone();
                tracing::debug!("Loaded joint pred weight from {}: {:?}", key, w.shape());
                break;
            }
        }

        for key in &pred_bias_keys {
            if let Some(b) = weights.get(key) {
                self.pred_proj_bias = Some(b.clone());
                tracing::debug!("Loaded joint pred bias from {}", key);
                break;
            }
        }

        // Output projection - try multiple patterns
        let out_weight_keys = [
            format!("{}.joint_net.2.weight", prefix),
            format!("{}.joint_net.output.weight", prefix),
            format!("{}.out.weight", prefix),
        ];
        let out_bias_keys = [
            format!("{}.joint_net.2.bias", prefix),
            format!("{}.joint_net.output.bias", prefix),
            format!("{}.out.bias", prefix),
        ];

        for key in &out_weight_keys {
            if let Some(w) = weights.get(key) {
                self.joint_proj_weight = w.clone();
                tracing::debug!("Loaded joint output weight from {}: {:?}", key, w.shape());
                break;
            }
        }

        for key in &out_bias_keys {
            if let Some(b) = weights.get(key) {
                self.joint_proj_bias = Some(b.clone());
                tracing::debug!("Loaded joint output bias from {}", key);
                break;
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
