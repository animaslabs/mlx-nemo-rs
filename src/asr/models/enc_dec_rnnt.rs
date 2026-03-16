//! Encoder-Decoder RNN-T ASR model.
//!
//! This model implements the RNN-T (Transducer) ASR architecture used by
//! NeMo's EncDecRNNTModel and EncDecRNNTBPEModel (Parakeet RNNT models).
//!
//! # Streaming Support
//!
//! For streaming-capable models, this module provides cache-aware streaming:
//!
//! ```rust,ignore
//! let mut model = EncDecRNNTModel::restore("model.nemo")?;
//!
//! // Initialize streaming
//! let mut session = model.init_streaming(0, 1)?;
//!
//! // Process audio chunks
//! while let Some(chunk) = audio_source.next_chunk() {
//!     let output = model.process_chunk(&mut session, &chunk)?;
//!     print!("{}", output.text[0]);
//! }
//!
//! // Finalize
//! let final_output = model.finalize_streaming(session)?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
use mlx_rs::Array;
use sentencepiece::SentencePieceProcessor;

use crate::asr::config::{AsrModelConfig, JointConfig};
use crate::asr::modules::{
    ConformerEncoder, LstmState, MelSpectrogram, RNNTDecoder, RNNTJoint, StatelessTransducerDecoder,
};
use crate::asr::streaming::{
    compute_streaming_config, get_streaming_modes, EncoderCache, StreamingConfig,
    StreamingMelBuffer, StreamingModeInfo, StreamingOutput,
};
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::config::from_yaml_str;
use crate::core::error::{Error, Result};

/// Decoding strategy for RNNT models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnntDecodingStrategy {
    /// Greedy decoding (fast, lower quality)
    Greedy,
    /// Beam search decoding (slower, higher quality)
    BeamSearch { beam_size: usize },
}

impl Default for RnntDecodingStrategy {
    fn default() -> Self {
        Self::Greedy
    }
}

/// Prediction network type (stateless or LSTM-based).
pub enum PredictionNetwork {
    /// Stateless decoder (embedding-based)
    Stateless(StatelessTransducerDecoder),
    /// LSTM-based decoder
    Lstm(RNNTDecoder),
}

/// Decoder state type.
pub enum DecoderState {
    /// Stateless decoder state (context buffer)
    Stateless(Array),
    /// LSTM decoder state (hidden + cell)
    Lstm(LstmState),
}

impl PredictionNetwork {
    /// Get the prediction network hidden dimension.
    pub fn pred_hidden(&self) -> usize {
        match self {
            Self::Stateless(d) => d.pred_hidden(),
            Self::Lstm(d) => d.pred_hidden(),
        }
    }

    /// Get the blank token index.
    pub fn blank_idx(&self) -> usize {
        match self {
            Self::Stateless(d) => d.blank_idx(),
            Self::Lstm(d) => d.blank_idx(),
        }
    }

    /// Get the vocabulary size (excluding blank).
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Stateless(d) => d.vocab_size(),
            Self::Lstm(d) => d.vocab_size(),
        }
    }

    /// Initialize decoder state for a batch.
    pub fn initialize_state(&self, batch_size: usize) -> Result<DecoderState> {
        match self {
            Self::Stateless(d) => Ok(DecoderState::Stateless(d.initialize_state(batch_size)?)),
            Self::Lstm(d) => Ok(DecoderState::Lstm(d.initialize_state(batch_size)?)),
        }
    }

    /// Initialize decoder state with warmup, matching NeMo's add_sos=True behavior.
    ///
    /// NeMo's decoder.predict(y=None, state=None, add_sos=True) runs the LSTM twice
    /// with blank inputs to "warm up" the state before decoding. This method
    /// replicates that behavior by:
    /// 1. Initialize state with zeros
    /// 2. Run predict_one(None) to get first output and updated state
    /// 3. Run predict_one(None) again to get the warmed up state
    ///
    /// Returns the prediction output and state after the warmup.
    pub fn initialize_state_with_warmup(&self, batch_size: usize) -> Result<(Array, DecoderState)> {
        // Initialize with zeros
        let state = self.initialize_state(batch_size)?;

        // First pass: blank input with zeros state
        let (_, state1) = self.predict_one(None, &state)?;

        // Second pass: blank input with updated state (matches NeMo's add_sos=True output[-1])
        let (output, state2) = self.predict_one(None, &state1)?;

        Ok((output, state2))
    }

    /// Run a single-step prediction for decoding.
    ///
    /// # Arguments
    /// * `y` - Single token index [batch, 1] or None for blank/SOS
    /// * `state` - Previous decoder state
    ///
    /// # Returns
    /// * `(output, new_state)` where output is [batch, 1, pred_hidden]
    pub fn predict_one(
        &self,
        y: Option<&Array>,
        state: &DecoderState,
    ) -> Result<(Array, DecoderState)> {
        match (self, state) {
            (Self::Stateless(decoder), DecoderState::Stateless(s)) => {
                let (output, new_state) = decoder.predict_one(y, s)?;
                Ok((output, DecoderState::Stateless(new_state)))
            }
            (Self::Lstm(decoder), DecoderState::Lstm(s)) => {
                let (output, new_state) = decoder.predict_one(y, s)?;
                Ok((output, DecoderState::Lstm(new_state)))
            }
            _ => Err(Error::model("Decoder and state type mismatch".to_string())),
        }
    }

    /// Load weights from a checkpoint.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) -> Result<()> {
        match self {
            Self::Stateless(d) => d.load_weights(weights, prefix),
            Self::Lstm(d) => d.load_weights(weights, prefix),
        }
    }

    /// Check if this is an LSTM-based decoder.
    pub fn is_lstm(&self) -> bool {
        matches!(self, Self::Lstm(_))
    }
}

/// Encoder-Decoder RNN-T ASR Model.
///
/// This model consists of:
/// 1. Audio preprocessor (Mel spectrogram)
/// 2. Conformer/FastConformer encoder
/// 3. Prediction network (stateless or LSTM-based decoder)
/// 4. Joint network
///
/// # Example
///
/// ```rust,ignore
/// use mlx_nemo::asr::EncDecRNNTModel;
///
/// // Load from .nemo checkpoint
/// let model = EncDecRNNTModel::restore("parakeet-rnnt-0.6b.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// ```
pub struct EncDecRNNTModel {
    config: AsrModelConfig,
    preprocessor: MelSpectrogram,
    encoder: ConformerEncoder,
    decoder: PredictionNetwork,
    joint: RNNTJoint,
    tokenizer: Option<SentencePieceProcessor>,
    /// Blank token ID (typically vocab_size, last token)
    blank_id: usize,
    /// Decoding strategy
    decoding_strategy: RnntDecodingStrategy,
    chunk_count: usize,
}

impl EncDecRNNTModel {
    /// Create a new model from configuration.
    pub fn new(config: AsrModelConfig) -> Result<Self> {
        let preprocessor = MelSpectrogram::new(config.preprocessor.clone().into())?;
        let encoder = ConformerEncoder::new(config.encoder.clone())?;

        // Determine vocabulary size from config or default
        let vocab_size = if config.decoder.num_classes > 0 {
            config.decoder.num_classes as usize
        } else {
            // Will be set when loading tokenizer
            1024 // Default placeholder
        };

        // Detect decoder type from config
        let is_lstm_decoder = config
            .decoder
            .prednet
            .as_ref()
            .map(|p| p.is_lstm_based())
            .unwrap_or(false);

        // Create prediction network (decoder) based on config
        let decoder = if is_lstm_decoder {
            tracing::info!("Using LSTM-based prediction network");
            PredictionNetwork::Lstm(RNNTDecoder::new(&config.decoder, vocab_size)?)
        } else {
            tracing::info!("Using stateless prediction network");
            PredictionNetwork::Stateless(StatelessTransducerDecoder::new(
                &config.decoder,
                vocab_size,
            )?)
        };

        // Create joint network using effective config values
        let joint_config = config.joint.clone().unwrap_or_default();

        let encoder_hidden = encoder.output_dim();
        let pred_hidden = decoder.pred_hidden();

        // Use effective values from jointnet if present
        let joint_hidden = joint_config.effective_joint_hidden();
        let activation = joint_config.effective_activation().to_string();

        let effective_joint_config = JointConfig {
            joint_hidden,
            activation,
            ..joint_config
        };

        let joint = RNNTJoint::new(
            &effective_joint_config,
            encoder_hidden,
            pred_hidden,
            vocab_size + 1,
        )?;

        // Blank is typically the last token (vocab_size)
        let blank_id = vocab_size;

        // Determine decoding strategy from config
        let decoding_strategy = if let Some(ref decoding) = config.decoding {
            match decoding.strategy.as_str() {
                "beam" | "beam_search" => RnntDecodingStrategy::BeamSearch {
                    beam_size: decoding.beam_size.unwrap_or(4),
                },
                _ => RnntDecodingStrategy::Greedy,
            }
        } else {
            RnntDecodingStrategy::Greedy
        };

        Ok(Self {
            config,
            preprocessor,
            encoder,
            decoder,
            joint,
            tokenizer: None,
            blank_id,
            decoding_strategy,
            chunk_count: 0,
        })
    }

    /// Restore a model from a .nemo checkpoint file.
    pub fn restore(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading RNNT model from: {}", path.display());

        // Open the checkpoint
        let checkpoint = NemoCheckpoint::open(path).or_else(|_| {
            // Try gzip format for older checkpoints
            NemoCheckpoint::open_gzip(path)
        })?;

        // Parse the configuration
        let config: AsrModelConfig = from_yaml_str(&checkpoint.config_yaml)?;

        // Verify this is an RNNT model
        if !config.is_rnnt() {
            tracing::warn!(
                "Config does not appear to be an RNNT model. Target: {:?}",
                config.target
            );
        }

        tracing::debug!(
            "Model config loaded, encoder layers: {}",
            config.encoder.n_layers
        );

        // Create the model
        let mut model = Self::new(config)?;

        // Load tokenizer if available
        if let Some(tokenizer_path) = checkpoint.find_tokenizer() {
            tracing::debug!("Loading tokenizer from: {}", tokenizer_path.display());
            model.load_tokenizer(&tokenizer_path)?;
        }

        // Load weights
        if checkpoint.weights_path.is_some() {
            match checkpoint.load_weights() {
                Ok(weights) => {
                    tracing::debug!("Loading {} weight tensors", weights.len());
                    model.load_weights(&weights)?;
                }
                Err(e) => {
                    tracing::warn!("Could not load weights: {}", e);
                }
            }
        }

        Ok(model)
    }

    /// Load a tokenizer from a SentencePiece model file.
    pub fn load_tokenizer(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tokenizer = SentencePieceProcessor::open(path)
            .map_err(|e| Error::tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        // Update vocabulary size and blank token
        let vocab_size = tokenizer.len();
        self.blank_id = vocab_size; // Blank is typically vocab_size (last index)

        // Recreate decoder with correct vocab size
        let is_lstm_decoder = self.decoder.is_lstm();
        self.decoder = if is_lstm_decoder {
            PredictionNetwork::Lstm(RNNTDecoder::new(&self.config.decoder, vocab_size)?)
        } else {
            PredictionNetwork::Stateless(StatelessTransducerDecoder::new(
                &self.config.decoder,
                vocab_size,
            )?)
        };

        // Recreate joint network with correct vocab size
        let joint_config = self.config.joint.clone().unwrap_or_default();
        let encoder_hidden = self.encoder.output_dim();
        let pred_hidden = self.decoder.pred_hidden();
        let joint_hidden = joint_config.effective_joint_hidden();
        let activation = joint_config.effective_activation().to_string();

        let effective_joint_config = JointConfig {
            joint_hidden,
            activation,
            ..joint_config
        };

        self.joint = RNNTJoint::new(
            &effective_joint_config,
            encoder_hidden,
            pred_hidden,
            vocab_size + 1,
        )?;

        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Load weights from a HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<()> {
        // Load encoder weights
        self.encoder.load_weights(weights)?;

        // Load decoder (prediction network) weights
        self.decoder.load_weights(weights, "decoder")?;

        // Load joint network weights
        self.joint.load_weights(weights, "joint")?;

        Ok(())
    }

    /// Set the decoding strategy.
    pub fn set_decoding_strategy(&mut self, strategy: RnntDecodingStrategy) {
        self.decoding_strategy = strategy;
    }

    /// Get the current decoding strategy.
    pub fn decoding_strategy(&self) -> RnntDecodingStrategy {
        self.decoding_strategy
    }

    /// Transcribe audio files.
    ///
    /// # Arguments
    /// * `paths` - Paths to audio files (WAV format)
    ///
    /// # Returns
    /// * Vector of transcribed text strings
    pub fn transcribe(&mut self, paths: &[impl AsRef<Path>]) -> Result<Vec<String>> {
        // Load and preprocess audio
        let audio_batch = self.load_audio_batch(paths)?;

        // Run forward pass to get encoder output
        let (encoder_output, encoder_lengths) =
            self.forward(&audio_batch.0, Some(&audio_batch.1))?;

        // Decode using greedy decoding (now supports both decoder types)
        let token_sequences = self.greedy_decode(&encoder_output, &encoder_lengths)?;

        // Convert tokens to text
        let texts = self.decode_tokens(&token_sequences)?;

        Ok(texts)
    }

    /// Perform greedy decoding using the model's decoder.
    fn greedy_decode(
        &self,
        encoder_output: &Array,
        encoder_lengths: &Array,
    ) -> Result<Vec<Vec<usize>>> {
        use mlx_rs::ops::indexing::{argmax_axis, IndexOp};

        const MAX_SYMBOLS_PER_STEP: usize = 10;

        encoder_output.eval()?;
        encoder_lengths.eval()?;

        let batch_size = encoder_output.shape()[0] as usize;

        // Get encoder lengths as a vector
        let lengths_i32 = encoder_lengths.as_type::<i32>()?;
        lengths_i32.eval()?;
        let lengths_data: Vec<i32> = lengths_i32.as_slice::<i32>().to_vec();

        let mut results = Vec::with_capacity(batch_size);

        // Process each sequence in the batch independently
        for b in 0..batch_size {
            let seq_len = lengths_data[b] as usize;
            let mut tokens = Vec::new();

            // Initialize decoder state for this sequence
            let mut state = self.decoder.initialize_state(1)?;

            // Process each encoder frame
            let mut t = 0;
            while t < seq_len {
                // Get encoder output for current frame: [1, 1, encoder_hidden]
                let enc_frame =
                    encoder_output.index((b as i32..b as i32 + 1, t as i32..t as i32 + 1, ..));

                let mut symbols_emitted = 0;

                // Emit symbols until blank or max symbols reached
                loop {
                    // Get prediction network output
                    let last_token = if tokens.is_empty() {
                        None
                    } else {
                        let last = *tokens.last().unwrap() as i32;
                        Some(Array::from_slice(&[last], &[1, 1]))
                    };
                    let (pred_output, new_state) =
                        self.decoder.predict_one(last_token.as_ref(), &state)?;

                    // Compute joint output: [1, vocab_size]
                    let log_probs = self.joint.forward_one(&enc_frame, &pred_output)?;
                    log_probs.eval()?;

                    // Get argmax
                    let prediction = argmax_axis(&log_probs, -1, false)?;
                    prediction.eval()?;
                    let pred_token = prediction.as_slice::<u32>()[0] as usize;

                    if pred_token == self.blank_id {
                        // Blank: advance to next encoder frame
                        break;
                    }

                    // Non-blank: emit token and update state
                    tokens.push(pred_token);
                    state = new_state;
                    symbols_emitted += 1;

                    if symbols_emitted >= MAX_SYMBOLS_PER_STEP {
                        // Safety limit reached, advance frame
                        break;
                    }
                }

                t += 1;
            }

            results.push(tokens);
        }

        Ok(results)
    }

    /// Run forward pass on preprocessed audio.
    ///
    /// # Arguments
    /// * `audio` - Raw audio waveform [batch, samples]
    /// * `lengths` - Optional audio lengths [batch]
    ///
    /// # Returns
    /// * Encoder output [batch, time, encoder_hidden]
    /// * Encoder output lengths [batch]
    pub fn forward(&mut self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Preprocess audio to mel spectrogram
        let (features, feature_lengths) = self.preprocessor.forward(audio, lengths)?;

        // Debug: print mel spectrogram stats
        features.eval()?;
        feature_lengths.eval()?;
        tracing::debug!(
            "Mel spectrogram shape: {:?}, length: {:?}",
            features.shape(),
            feature_lengths.as_slice::<i32>()
        );
        let features_flat = features.reshape(&[-1])?;
        features_flat.eval()?;
        let f_slice: &[f32] = features_flat.as_slice();
        let f_min = f_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_max = f_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let f_mean: f32 = f_slice.iter().sum::<f32>() / f_slice.len() as f32;
        tracing::debug!(
            "Mel min: {:.4}, max: {:.4}, mean: {:.4}",
            f_min,
            f_max,
            f_mean
        );

        // Encode
        let (encoded, encoded_lengths) = self.encoder.forward(&features, &feature_lengths)?;

        // Debug: print encoder output stats
        encoded.eval()?;
        encoded_lengths.eval()?;
        tracing::debug!(
            "Encoder output shape: {:?}, length: {:?}",
            encoded.shape(),
            encoded_lengths.as_slice::<i32>()
        );
        let encoded_flat = encoded.reshape(&[-1])?;
        encoded_flat.eval()?;
        let e_slice: &[f32] = encoded_flat.as_slice();
        let e_min = e_slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let e_max = e_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e_mean: f32 = e_slice.iter().sum::<f32>() / e_slice.len() as f32;
        tracing::debug!(
            "Encoder min: {:.4}, max: {:.4}, mean: {:.4}",
            e_min,
            e_max,
            e_mean
        );

        // For RNNT, we return encoder output directly (decoding happens separately)
        // Transpose from [B, D, T] to [B, T, D]
        let encoded_transposed = mlx_rs::ops::transpose_axes(&encoded, &[0, 2, 1])?;

        Ok((encoded_transposed, encoded_lengths))
    }

    /// Load audio files and return batched tensors.
    fn load_audio_batch(&self, paths: &[impl AsRef<Path>]) -> Result<(Array, Array)> {
        let sample_rate = self.config.sample_rate;
        let mut audio_samples = Vec::new();
        let mut lengths = Vec::new();

        for path in paths {
            let path = path.as_ref();
            let (samples, file_sample_rate) = load_wav(path)?;

            // Resample if necessary
            let samples = if file_sample_rate != sample_rate {
                resample(&samples, file_sample_rate, sample_rate)?
            } else {
                samples
            };

            lengths.push(samples.len());
            audio_samples.push(samples);
        }

        // Pad to max length and create batch
        let max_len = lengths.iter().max().copied().unwrap_or(0);
        let batch_size = audio_samples.len();

        let mut padded_data = vec![0.0f32; batch_size * max_len];
        for (i, samples) in audio_samples.iter().enumerate() {
            let offset = i * max_len;
            padded_data[offset..offset + samples.len()].copy_from_slice(samples);
        }

        let audio = Array::from_slice(&padded_data, &[batch_size as i32, max_len as i32]);
        let lengths_arr = Array::from_slice(
            &lengths.iter().map(|&l| l as i32).collect::<Vec<_>>(),
            &[batch_size as i32],
        );

        Ok((audio, lengths_arr))
    }

    /// Decode token sequences to text.
    fn decode_tokens(&self, token_sequences: &[Vec<usize>]) -> Result<Vec<String>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::tokenizer("No tokenizer loaded. Call load_tokenizer first."))?;

        let mut texts = Vec::with_capacity(token_sequences.len());

        for tokens in token_sequences {
            let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let text = tokenizer
                .decode_piece_ids(&token_ids)
                .map_err(|e| Error::tokenizer(format!("Failed to decode tokens: {}", e)))?;
            texts.push(text);
        }

        Ok(texts)
    }

    /// Get the model configuration.
    pub fn config(&self) -> &AsrModelConfig {
        &self.config
    }

    /// Get the sample rate expected by the model.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the blank token ID.
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Get a reference to the preprocessor.
    pub fn preprocessor(&self) -> &MelSpectrogram {
        &self.preprocessor
    }

    /// Get a reference to the encoder.
    pub fn encoder(&self) -> &ConformerEncoder {
        &self.encoder
    }

    /// Get a reference to the prediction network.
    pub fn prediction_network(&self) -> &PredictionNetwork {
        &self.decoder
    }

    /// Get a reference to the joint network.
    pub fn joint(&self) -> &RNNTJoint {
        &self.joint
    }

    /// Check if this model uses an LSTM-based decoder.
    pub fn is_lstm_decoder(&self) -> bool {
        self.decoder.is_lstm()
    }

    // ==================== Streaming API ====================

    /// Check if this model supports streaming inference.
    pub fn supports_streaming(&self) -> bool {
        self.encoder.supports_streaming()
    }

    /// Get available streaming modes with latency information.
    pub fn streaming_modes(&self) -> Vec<StreamingModeInfo> {
        let frame_duration_ms = self.config.preprocessor.window_stride
            * 1000.0
            * self.encoder.subsampling_factor() as f32;
        get_streaming_modes(self.encoder.config(), frame_duration_ms)
    }

    /// Initialize a streaming session.
    ///
    /// # Arguments
    /// * `mode_idx` - Streaming mode index
    /// * `batch_size` - Number of independent audio streams
    pub fn init_streaming(
        &mut self,
        mode_idx: usize,
        batch_size: usize,
    ) -> Result<RnntStreamingSession> {
        if !self.supports_streaming() {
            return Err(Error::config(
                "Model does not support streaming. Use a streaming-capable model.",
            ));
        }

        // Reset attention and encoder debug counters for fresh streaming session
        crate::asr::modules::encoders::attention::reset_attention_counter();
        crate::asr::modules::encoders::reset_encoder_counter();

        // Compute streaming configuration
        let config = compute_streaming_config(self.encoder.config(), mode_idx, 10000)?;

        tracing::debug!(
            mode_idx,
            chunk_size_first = config.chunk_size.0,
            chunk_size_subsequent = config.chunk_size.1,
            shift_size_first = config.shift_size.0,
            shift_size_subsequent = config.shift_size.1,
            pre_encode_cache_first = config.pre_encode_cache_size.0,
            pre_encode_cache_subsequent = config.pre_encode_cache_size.1,
            valid_out_len = config.valid_out_len,
            drop_extra_pre_encoded = config.drop_extra_pre_encoded,
            cache_drop_size = config.cache_drop_size,
            last_channel_cache_size = config.last_channel_cache_size,
            last_time_cache_size = config.last_time_cache_size,
            "init_streaming configuration"
        );

        // Setup encoder
        self.encoder.setup_streaming(config.cache_drop_size);

        // Reset preprocessor streaming state (remainder buffer, preemphasis state)
        self.preprocessor.reset();

        // Create encoder cache
        let encoder_cache = EncoderCache::new(
            self.encoder.num_layers(),
            batch_size,
            config.last_channel_cache_size,
            config.last_time_cache_size,
            self.encoder.d_model(),
        )?;

        // Create mel buffer
        let preprocessor_config = self.preprocessor.config();
        let n_mels = preprocessor_config.features;
        let mel_buffer = StreamingMelBuffer::new(
            preprocessor_config.sample_rate,
            preprocessor_config.window_size,
            preprocessor_config.window_stride,
            n_mels,
            preprocessor_config.n_fft,
        );

        tracing::debug!(
            window_samples = mel_buffer.window_samples,
            hop_samples = mel_buffer.hop_samples,
            n_mels,
            subsampling_factor = self.encoder.subsampling_factor(),
            "mel_buffer configuration"
        );

        // Initialize decoder state for each batch sample
        let mut decoder_states = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            decoder_states.push(self.decoder.initialize_state(1)?);
        }

        Ok(RnntStreamingSession {
            config,
            encoder_cache,
            mel_buffer,
            decoder_states,
            last_tokens: vec![None; batch_size],
            accumulated_tokens: vec![Vec::new(); batch_size],
            is_first_chunk: true,
            blank_id: self.blank_id,
            n_mels,
        })
    }

    /// Process an audio chunk and return partial transcription.
    ///
    /// This follows NeMo's streaming approach:
    /// 1. Convert audio to mel spectrogram using the streaming mel API
    /// 2. Append mel frames to the buffer for chunked extraction
    /// 3. Extract mel chunks with pre_encode_cache overlap
    /// 4. Process through encoder and decoder
    ///
    /// The preprocessor maintains preemphasis state and audio remainder buffer
    /// internally for seamless chunk boundaries.
    pub fn process_chunk(
        &mut self,
        session: &mut RnntStreamingSession,
        audio_chunk: &[f32],
    ) -> Result<StreamingOutput> {
        let batch_size = session.encoder_cache.batch_size();

        // Step 1: Use the streaming mel spectrogram API
        // This handles preemphasis state and remainder buffer internally
        let mel_features = self.preprocessor.process_chunk(audio_chunk)?;

        tracing::debug!(
            "PROCESS CHUNK #{} len({}) Got mel_features {:?}",
            self.chunk_count,
            audio_chunk.len(),
            mel_features.shape()
        );
        self.chunk_count += 1;

        // Track total audio samples for logging
        session.mel_buffer.total_audio_samples += audio_chunk.len();

        // Check if we got any frames
        let n_frames = mel_features.dim(1) as usize;
        if n_frames > 0 {
            // Step 2: Append mel frames to the buffer
            // mel_features is [n_mels, time], append directly as Array
            session.mel_buffer.append_mel(&mel_features)?;

            tracing::debug!(
                frames_appended = n_frames,
                audio_samples = audio_chunk.len(),
                total_frames = session.mel_buffer.total_frames,
                "appended mel frames from audio"
            );
        } else if audio_chunk.is_empty() {
            // Empty chunk is ok, just process any buffered mel frames
        } else {
            tracing::trace!(
                audio_samples = audio_chunk.len(),
                "not enough audio for mel frames yet"
            );
        }

        // Step 3: Process all complete mel chunks
        let mut all_new_tokens: Vec<Vec<usize>> = vec![Vec::new(); batch_size];
        let mut chunks_processed = 0;
        let mut total_encoder_frames_this_call = 0usize;

        loop {
            // Get chunk parameters for current step
            let chunk_size = session.config.chunk_size_for_step(session.is_first_chunk);
            let shift_size = session.config.shift_size_for_step(session.is_first_chunk);
            let pre_encode_cache = session
                .config
                .pre_encode_cache_for_step(session.is_first_chunk);
            // NeMo applies drop_extra_pre_encoded on subsequent chunks only (not the first).
            // The Python streaming script passes drop_extra_pre_encoded=0 for step_num=0.
            // This is because the first chunk doesn't have pre_encode_cache overlap.
            let drop_extra = if session.is_first_chunk {
                0 // Don't drop on first chunk
            } else {
                session.config.drop_extra_pre_encoded
            };

            // Check if we have enough mel frames
            let remaining = session.mel_buffer.remaining_frames();
            if remaining < chunk_size {
                if chunks_processed == 0 {
                    tracing::trace!(
                        chunk_size_needed = chunk_size,
                        frames_remaining = remaining,
                        "need more mel frames for chunk"
                    );
                }
                break;
            }
            tracing::debug!(
                chunk_size,
                shift_size,
                pre_encode_cache,
                drop_extra,
                buffer_idx = session.mel_buffer.buffer_idx,
                total_frames = session.mel_buffer.total_frames,
                "extracting mel chunk"
            );

            // Extract mel chunk with pre_encode_cache overlap
            // Returns (mel_array, total_frames, valid_frames)
            let (mel_chunk, _total_frames, chunk_len) = session
                .mel_buffer
                .extract_chunk(chunk_size, shift_size, pre_encode_cache)
                .ok_or_else(|| crate::core::error::Error::model("Failed to extract mel chunk"))?;

            {
                // Flatten the mel chunk to a Vec for debug statistics
                let mel_flat: Vec<f32> = mel_chunk
                    .flatten(0, -1)
                    .ok()
                    .and_then(|a| a.as_slice::<f32>().to_vec().into())
                    .unwrap_or_default();
                let mel_min = mel_flat.iter().cloned().fold(f32::INFINITY, f32::min);
                let mel_max = mel_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mel_mean: f32 = if !mel_flat.is_empty() {
                    mel_flat.iter().sum::<f32>() / mel_flat.len() as f32
                } else {
                    0.0
                };
                tracing::trace!(
                    frames_extracted = chunk_len,
                    chunk_size,
                    cache_frames = chunk_len.saturating_sub(chunk_size),
                    mel_min,
                    mel_max,
                    mel_mean,
                    "extracted mel chunk"
                );
                // Show first few mel frames at trace level
                // mel_chunk is [n_mels, time] format, so frame t, freq f = mel_flat[f * chunk_len + t]
                for t in 0..5.min(chunk_len) {
                    let f0 = mel_flat.get(t).copied().unwrap_or(0.0);
                    let f1 = mel_flat.get(chunk_len + t).copied().unwrap_or(0.0);
                    let f2 = mel_flat.get(2 * chunk_len + t).copied().unwrap_or(0.0);
                    tracing::trace!(
                        frame = t,
                        freq_bin_0 = f0,
                        freq_bin_1 = f1,
                        freq_bin_2 = f2,
                        "mel frame values"
                    );
                }
            }

            // Add batch dimension: [n_mels, time] -> [1, n_mels, time]
            let features = mel_chunk.reshape(&[1, session.n_mels as i32, chunk_len as i32])?;

            // Get lengths for encoder
            let lengths = Array::from_slice(&[chunk_len as i32], &[1]);

            // Run encoder with cache
            let cache_len_before = session.encoder_cache.last_channel_len.as_slice::<i64>()[0];

            let (encoded, encoded_lengths, cache_channel_next, cache_time_next, cache_len_next) =
                self.encoder.forward_streaming(
                    &features,
                    &lengths,
                    &session.encoder_cache.last_channel,
                    &session.encoder_cache.last_time,
                    &session.encoder_cache.last_channel_len,
                    drop_extra,
                )?;

            // Update encoder cache
            session.encoder_cache.last_channel = cache_channel_next;
            session.encoder_cache.last_time = cache_time_next;
            session.encoder_cache.last_channel_len = cache_len_next;

            // Transpose encoder output from [B, D, T] to [B, T, D]
            let encoded = mlx_rs::ops::transpose_axes(&encoded, &[0, 2, 1])?;
            encoded.eval()?;
            encoded_lengths.eval()?;

            let raw_enc_len = encoded_lengths.as_slice::<i32>()[0] as usize;
            let cache_len_after = session.encoder_cache.last_channel_len.as_slice::<i64>()[0];

            {
                // Show encoder output stats
                let enc_flat = encoded.reshape(&[-1]).unwrap();
                enc_flat.eval().unwrap();
                let enc_slice: &[f32] = enc_flat.as_slice();
                let enc_min = enc_slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let enc_max = enc_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let enc_mean: f32 = enc_slice.iter().sum::<f32>() / enc_slice.len() as f32;
                let enc_var: f32 = enc_slice
                    .iter()
                    .map(|&v| (v - enc_mean).powi(2))
                    .sum::<f32>()
                    / enc_slice.len() as f32;
                let enc_std = enc_var.sqrt();
                let first_10: Vec<f32> = enc_slice.iter().take(10).cloned().collect();
                // Log at debug level for chunks around problematic areas
                tracing::debug!(
                    chunk_idx = chunks_processed,
                    output_frames = raw_enc_len,
                    cache_len_before,
                    cache_len_after,
                    enc_min,
                    enc_max,
                    enc_mean,
                    enc_std,
                    ?first_10,
                    "encoder output stats"
                );
            }

            // Truncate to valid_out_len, but also respect actual tensor shape
            let valid_out_len = session.config.valid_out_len;
            let actual_tensor_frames = encoded.shape()[1] as usize;
            // Use the minimum of: what encoder reports, valid_out_len, and actual tensor frames
            let enc_seq_len = raw_enc_len.min(valid_out_len).min(actual_tensor_frames);
            total_encoder_frames_this_call += enc_seq_len;

            tracing::trace!(
                valid_out_len,
                raw_enc_len,
                actual_tensor_frames,
                frames_to_decode = enc_seq_len,
                "decoding encoder frames"
            );

            // Decode
            const MAX_SYMBOLS_PER_STEP: usize = 10;
            let mut tokens_this_chunk = Vec::new();
            let mut blank_count = 0;
            let mut non_blank_count = 0;

            for b in 0..batch_size {
                for t in 0..enc_seq_len {
                    let enc_frame =
                        encoded.index((b as i32..b as i32 + 1, t as i32..t as i32 + 1, ..));

                    let mut symbols_emitted = 0;

                    loop {
                        let last_token = session.last_tokens[b]
                            .map(|tok| Array::from_slice(&[tok as i32], &[1, 1]));
                        let (pred_output, new_state) = self
                            .decoder
                            .predict_one(last_token.as_ref(), &session.decoder_states[b])?;

                        let log_probs = self.joint.forward_one(&enc_frame, &pred_output)?;
                        log_probs.eval()?;

                        let prediction = argmax_axis(&log_probs, -1, false)?;
                        prediction.eval()?;
                        let pred_token = prediction.as_slice::<u32>()[0] as usize;

                        // Debug: log top predictions when we get a blank at first symbol
                        if pred_token == session.blank_id && symbols_emitted == 0 && t == 0 {
                            let lp_slice: &[f32] = log_probs.as_slice();
                            // Get top 5 predictions
                            let mut indexed: Vec<(usize, f32)> =
                                lp_slice.iter().cloned().enumerate().collect();
                            indexed.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
                            let blank_prob = lp_slice
                                .get(session.blank_id)
                                .copied()
                                .unwrap_or(f32::NEG_INFINITY);
                            tracing::debug!(
                                ?top5,
                                blank_prob,
                                blank_id = session.blank_id,
                                "first frame blank prediction"
                            );
                        }

                        if pred_token == session.blank_id {
                            blank_count += 1;
                            break;
                        }

                        non_blank_count += 1;
                        all_new_tokens[b].push(pred_token);
                        session.accumulated_tokens[b].push(pred_token);
                        session.last_tokens[b] = Some(pred_token);
                        session.decoder_states[b] = new_state;
                        symbols_emitted += 1;
                        tokens_this_chunk.push(pred_token);

                        if symbols_emitted >= MAX_SYMBOLS_PER_STEP {
                            break;
                        }
                    }
                }
            }

            {
                let total_tokens = session
                    .accumulated_tokens
                    .get(0)
                    .map(|t| t.len())
                    .unwrap_or(0);
                let token_text = if let Some(tokenizer) = &self.tokenizer {
                    let ids: Vec<u32> = tokens_this_chunk.iter().map(|&t| t as u32).collect();
                    tokenizer.decode_piece_ids(&ids).unwrap_or_default()
                } else {
                    format!("{:?}", tokens_this_chunk)
                };
                // Log token IDs for comparison with Python
                let token_ids_str: String = tokens_this_chunk
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                tracing::debug!(
                    encoder_frames = enc_seq_len,
                    blank_emissions = blank_count,
                    non_blank_tokens = non_blank_count,
                    total_accumulated = total_tokens,
                    tokens_this_chunk = tokens_this_chunk.len(),
                    token_ids = %token_ids_str,
                    decoded_text = %token_text,
                    "decoding results"
                );
            }

            session.is_first_chunk = false;
            chunks_processed += 1;

            // Trim old mel frames to bound memory usage.
            // Keep enough frames for the largest pre_encode_cache we might need.
            let max_pre_encode_cache = session.config.pre_encode_cache_size.1;
            let _ = session.mel_buffer.trim_old_frames(max_pre_encode_cache);
        }

        if chunks_processed > 0 {
            let (_audio_bytes, mel_bytes) = session.mel_buffer.memory_stats();
            tracing::trace!(
                chunks_processed,
                encoder_frames = total_encoder_frames_this_call,
                mel_bytes,
                accumulated_tokens = session
                    .accumulated_tokens
                    .get(0)
                    .map(|t| t.len())
                    .unwrap_or(0),
                "processed mel chunks"
            );
        }

        // High-level debug: show progress summary
        {
            let total_audio_sec = session.mel_buffer.total_audio_samples as f32
                / session.mel_buffer.sample_rate as f32;
            let total_mel_frames = session.mel_buffer.total_frames;
            let mel_idx = session.mel_buffer.buffer_idx;
            let tokens = session
                .accumulated_tokens
                .get(0)
                .map(|t| t.len())
                .unwrap_or(0);
            tracing::debug!(
                audio_samples = session.mel_buffer.total_audio_samples,
                total_mel_frames,
                mel_idx,
                remaining_frames = total_mel_frames.saturating_sub(mel_idx),
                tokens,
                "streaming progress summary"
            );
        }

        // Build output texts from accumulated tokens
        let mut texts = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let text = if let Some(tokenizer) = &self.tokenizer {
                let token_ids: Vec<u32> = session.accumulated_tokens[b]
                    .iter()
                    .map(|&t| t as u32)
                    .collect();
                tokenizer.decode_piece_ids(&token_ids).unwrap_or_default()
            } else {
                String::new()
            };
            texts.push(text);
        }

        Ok(StreamingOutput {
            log_probs: None,
            tokens: all_new_tokens,
            text: texts,
            is_final: false,
        })
    }

    /// Finalize streaming and get final transcription.
    ///
    /// Flushes any remaining audio in the preprocessor's remainder buffer
    /// and processes any remaining mel frames.
    pub fn finalize_streaming(
        &mut self,
        mut session: RnntStreamingSession,
    ) -> Result<StreamingOutput> {
        // Flush any remaining audio in the preprocessor's remainder buffer
        let final_mel = self.preprocessor.flush()?;
        final_mel.eval()?;

        let n_frames = final_mel.dim(1) as usize;
        if n_frames > 0 {
            // Append final mel frames to buffer (final_mel is already [n_mels, time])
            session.mel_buffer.append_mel(&final_mel)?;

            tracing::debug!(
                final_mel_frames = n_frames,
                "finalize: flushed remaining audio to mel frames"
            );
        }

        // Process any remaining mel frames
        // If there are remaining frames but not enough for a full chunk, pad with zeros
        let chunk_size = session.config.chunk_size_for_step(session.is_first_chunk);
        let remaining = session.mel_buffer.remaining_frames();

        if remaining > 0 && remaining < chunk_size {
            // Pad mel buffer with zeros to make a full chunk
            let frames_to_pad = chunk_size - remaining;
            let n_mels = session.n_mels;

            tracing::debug!(
                frames_to_pad,
                chunk_size,
                "finalize: padding mel frames for full chunk"
            );

            // Create zero padding Array in [n_mels, time] format
            let padding = mlx_rs::ops::zeros::<f32>(&[n_mels as i32, frames_to_pad as i32])?;
            session.mel_buffer.append_mel(&padding)?;
        }

        while !session.mel_buffer.is_empty() {
            let remaining = session.mel_buffer.remaining_frames();
            if remaining == 0 {
                break;
            }

            tracing::debug!(
                mel_frames_remaining = remaining,
                "finalize: processing remaining mel frames"
            );

            // Try to process remaining frames
            let result = self.process_chunk(&mut session, &[])?;

            // Break if no progress was made
            if result.tokens.iter().all(|t| t.is_empty())
                && session.mel_buffer.remaining_frames() == remaining
            {
                break;
            }
        }

        // Final output with all accumulated tokens
        let batch_size = session.accumulated_tokens.len();
        let mut final_tokens = Vec::with_capacity(batch_size);
        let mut final_texts = Vec::with_capacity(batch_size);

        for tokens in &session.accumulated_tokens {
            let text = if let Some(tokenizer) = &self.tokenizer {
                let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                tokenizer.decode_piece_ids(&token_ids).unwrap_or_default()
            } else {
                String::new()
            };

            final_tokens.push(tokens.clone());
            final_texts.push(text);
        }

        Ok(StreamingOutput {
            log_probs: None,
            tokens: final_tokens,
            text: final_texts,
            is_final: true,
        })
    }
}

/// Streaming session state for RNNT models.
///
/// This session manages the streaming state following NeMo's approach:
/// 1. Audio is incrementally converted to mel frames
/// 2. Mel frames are stored in a buffer
/// 3. Chunks are extracted with pre_encode_cache overlap
pub struct RnntStreamingSession {
    /// Streaming configuration
    pub config: StreamingConfig,

    /// Encoder cache state
    pub encoder_cache: EncoderCache,

    /// Mel spectrogram buffer (stores pre-computed mel frames)
    pub mel_buffer: StreamingMelBuffer,

    /// Decoder states for each batch sample
    pub decoder_states: Vec<DecoderState>,

    /// Last emitted token for each batch sample
    pub last_tokens: Vec<Option<usize>>,

    /// All accumulated tokens for each batch sample
    pub accumulated_tokens: Vec<Vec<usize>>,

    /// Whether this is the first chunk (determines chunk_size and pre_encode_cache)
    pub is_first_chunk: bool,

    /// Blank token ID
    blank_id: usize,

    /// Number of mel features
    n_mels: usize,
}

/// Load a WAV file and return samples and sample rate.
#[cfg(feature = "asr")]
fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    use hound::WavReader;

    let reader = WavReader::open(path)
        .map_err(|e| Error::audio(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::audio(format!("Failed to read samples: {}", e)))?;

            // Convert stereo to mono by averaging
            if channels > 1 {
                samples
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect()
            } else {
                samples
            }
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1 << (bits - 1)) as f32;

            let samples: Vec<i32> = reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| Error::audio(format!("Failed to read samples: {}", e)))?;

            // Convert to float and handle channels
            if channels > 1 {
                samples
                    .chunks(channels)
                    .map(|chunk| {
                        chunk.iter().map(|&s| s as f32).sum::<f32>() / (channels as f32 * max_val)
                    })
                    .collect()
            } else {
                samples.iter().map(|&s| s as f32 / max_val).collect()
            }
        }
    };

    Ok((samples, sample_rate))
}

#[cfg(not(feature = "asr"))]
fn load_wav(_path: &Path) -> Result<(Vec<f32>, u32)> {
    Err(Error::not_implemented(
        "Audio loading requires the 'asr' feature",
    ))
}

/// Resample audio to a target sample rate.
#[cfg(feature = "asr")]
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| Error::audio(format!("Failed to create resampler: {}", e)))?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| Error::audio(format!("Resampling failed: {}", e)))?;

    Ok(waves_out.into_iter().next().unwrap_or_default())
}

#[cfg(not(feature = "asr"))]
fn resample(_samples: &[f32], _from_rate: u32, _to_rate: u32) -> Result<Vec<f32>> {
    Err(Error::not_implemented(
        "Audio resampling requires the 'asr' feature",
    ))
}
