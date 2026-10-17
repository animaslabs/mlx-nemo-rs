//! Encoder-Decoder RNN-T ASR model.
//!
//! This model implements the RNN-T (Transducer) ASR architecture used by
//! NeMo's EncDecRNNTModel and EncDecRNNTBPEModel (Parakeet RNNT models).
//!
//! # Streaming Support
//!
//! For streaming-capable models, this module provides cache-aware streaming:
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # use mlx_voice::asr::EncDecRNNTModel;
//! # let audio_chunks: Vec<Vec<f32>> = Vec::new();
//! let mut model = EncDecRNNTModel::restore("model.nemo")?;
//!
//! // Initialize streaming
//! let mut session = model.init_streaming(0, 1)?;
//!
//! // Process audio chunks
//! for chunk in &audio_chunks {
//!     let output = model.process_chunk(&mut session, chunk)?;
//!     print!("{}", output.text[0]);
//! }
//!
//! // Finalize
//! let final_output = model.finalize_streaming(session)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use sentencepiece::SentencePieceProcessor;

use crate::asr::config::{AsrModelConfig, JointConfig};
use crate::asr::decoding::{RnntStreamingDecoder, decode_text, lang_tag_table, last_lang_tag};
use crate::asr::modules::{
    ConformerEncoder, LanguagePrompt, LstmState, MelSpectrogram, MelSpectrogramConfig, RNNTDecoder,
    RNNTJoint, StatelessTransducerDecoder,
};
use crate::asr::streaming::{
    EncoderCache, StreamingConfig, StreamingMelBuffer, StreamingModeInfo, StreamingOutput,
    compute_streaming_config, get_streaming_modes,
};
use crate::audio::resample;
use crate::core::ArrayExt;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::config::from_yaml_str;
use crate::core::error::{Error, Result};
use crate::core::quantization::QuantizationConfig;

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

impl DecoderState {
    /// Compute a summary norm of the decoder state for debugging.
    pub fn compute_norm(&self) -> f32 {
        match self {
            DecoderState::Stateless(arr) => {
                arr.eval().ok();
                let data = arr.to_vec_row_major::<f32>().unwrap_or_default();
                if !data.is_empty() {
                    // Sample first 100 values for efficiency
                    let sample_len = data.len().min(100);
                    data[..sample_len].iter().map(|v| v * v).sum::<f32>().sqrt()
                } else {
                    0.0
                }
            }
            DecoderState::Lstm(lstm_state) => {
                // Compute norm of hidden state
                lstm_state.hidden.eval().ok();
                let data = lstm_state
                    .hidden
                    .to_vec_row_major::<f32>()
                    .unwrap_or_default();
                if !data.is_empty() {
                    let sample_len = data.len().min(100);
                    data[..sample_len].iter().map(|v| v * v).sum::<f32>().sqrt()
                } else {
                    0.0
                }
            }
        }
    }
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
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `prefix` - Key prefix for this layer
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        prefix: &str,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        match self {
            Self::Stateless(d) => d.load_weights(weights, prefix, quant),
            Self::Lstm(d) => d.load_weights(weights, prefix, quant),
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
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// use mlx_voice::asr::EncDecRNNTModel;
///
/// // Load from .nemo checkpoint
/// let mut model = EncDecRNNTModel::restore("parakeet-rnnt-0.6b.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// # Ok(())
/// # }
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
    /// For each token id, whether it is a language tag such as `<en-US>`
    lang_tags: Vec<bool>,
    /// Leave language tags out of transcribed text
    strip_lang_tags: bool,
}

impl EncDecRNNTModel {
    /// Create a new model from configuration.
    pub fn new(config: AsrModelConfig) -> Result<Self> {
        config.validate()?;
        // NeMo only dithers during training; at inference it would make the
        // output random
        let mut preproc_config: MelSpectrogramConfig = config.preprocessor.clone().into();
        preproc_config.dither = 0.0;
        let preprocessor = MelSpectrogram::new(preproc_config)?;
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
            lang_tags: Vec::new(),
            strip_lang_tags: true,
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

        // Load weights (.nemo files are not quantized)
        if checkpoint.weights_path.is_some() {
            match checkpoint.load_weights() {
                Ok(weights) => {
                    tracing::debug!("Loading {} weight tensors", weights.len());
                    model.load_weights(&weights, None)?;
                }
                Err(e) => {
                    tracing::warn!("Could not load weights: {}", e);
                }
            }
        }

        Ok(model)
    }

    /// Restore a model from standalone safetensors files.
    ///
    /// This method loads a model from separate files (weights, config, tokenizer)
    /// rather than a bundled .nemo archive. This is useful for:
    /// - Models exported to safetensors format
    /// - HuggingFace/mlx-community model directories
    /// - Custom model deployments
    ///
    /// # Arguments
    /// * `weights_path` - Path to the safetensors weights file
    /// * `config_path` - Path to the config file (YAML or JSON with NeMo AsrModelConfig schema)
    /// * `tokenizer_path` - Optional path to the SentencePiece tokenizer file
    /// * `format` - Optional weight format. If None, auto-detects from safetensors metadata
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::{asr::EncDecRNNTModel, WeightFormat};
    /// // Load with explicit format
    /// let model = EncDecRNNTModel::restore_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     Some(WeightFormat::Mlx),
    /// )?;
    ///
    /// // Load with auto-detected format
    /// let model = EncDecRNNTModel::restore_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     None,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn restore_safetensors(
        weights_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        tokenizer_path: Option<impl AsRef<Path>>,
        format: Option<crate::core::WeightFormat>,
    ) -> Result<Self> {
        use crate::core::checkpoint::SafetensorsLoader;

        let weights_path = weights_path.as_ref();
        let config_path = config_path.as_ref();
        tracing::info!(
            "Loading RNNT model from safetensors: weights={}, config={}",
            weights_path.display(),
            config_path.display()
        );

        // Create loader
        let mut loader = SafetensorsLoader::new(
            weights_path,
            config_path,
            tokenizer_path.as_ref().map(|p| p.as_ref()),
        )?;

        // Override format if specified
        if let Some(fmt) = format {
            loader.set_weight_format(fmt);
        }

        // Load config
        let config: AsrModelConfig = loader.load_config()?;

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

        // Create model
        let mut model = Self::new(config)?;

        // Load tokenizer if available
        if let Some(tok_path) = loader.tokenizer_path() {
            tracing::debug!("Loading tokenizer from: {}", tok_path.display());
            model.load_tokenizer(tok_path)?;
        } else if let Some(tok_path) = tokenizer_path {
            tracing::debug!("Loading tokenizer from: {}", tok_path.as_ref().display());
            model.load_tokenizer(tok_path)?;
        }

        // Load weights (may be quantized)
        let weights = loader.load_weights()?;
        let quant = loader.quantization();
        tracing::debug!(
            "Loading {} weight tensors (format: {:?}, quantized: {})",
            weights.len(),
            loader.weight_format,
            quant.is_some()
        );
        model.load_weights(&weights, quant)?;

        Ok(model)
    }

    /// Restore a model from a directory containing safetensors files.
    ///
    /// Auto-detects file locations within the directory:
    /// - Weights: model.safetensors, weights.safetensors, model_weights.safetensors
    /// - Config: config.yaml, config.json, model_config.yaml
    /// - Tokenizer: tokenizer.model, *.spm, sentencepiece.bpe.model
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::asr::EncDecRNNTModel;
    /// // Load from directory structure:
    /// // my_model/
    /// // ├── model.safetensors
    /// // ├── config.yaml
    /// // └── tokenizer.model
    /// let model = EncDecRNNTModel::restore_from_directory("my_model/")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn restore_from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        use crate::core::checkpoint::SafetensorsLoader;

        let dir = dir.as_ref();
        tracing::info!("Loading RNNT model from directory: {}", dir.display());

        let loader = SafetensorsLoader::from_directory(dir)?;

        // Load config
        let config: AsrModelConfig = loader.load_config()?;

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

        // Create model
        let mut model = Self::new(config)?;

        // Load tokenizer if found
        if let Some(tok_path) = loader.tokenizer_path() {
            tracing::debug!("Loading tokenizer from: {}", tok_path.display());
            model.load_tokenizer(tok_path)?;
        }

        // Load weights (may be quantized)
        let weights = loader.load_weights()?;
        let quant = loader.quantization();
        tracing::debug!(
            "Loading {} weight tensors (format: {:?}, quantized: {})",
            weights.len(),
            loader.weight_format,
            quant.is_some()
        );
        model.load_weights(&weights, quant)?;

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

        self.lang_tags = lang_tag_table(&tokenizer);
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Load weights from a HashMap.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        // Load encoder weights (linear layers may be quantized)
        self.encoder.load_weights(weights, quant)?;

        // Load decoder (prediction network) weights (embeddings may be quantized)
        self.decoder.load_weights(weights, "decoder", quant)?;

        // Load joint network weights (linear layers may be quantized)
        self.joint.load_weights(weights, "joint", quant)?;

        // Language-ID prompt (prompt-conditioned models only)
        self.joint
            .set_prompt(LanguagePrompt::from_config(&self.config, weights, quant)?);

        Ok(())
    }

    /// Whether the model takes a target language (prompt-conditioned models
    /// such as nemotron-3.5-asr-streaming-0.6b).
    pub fn is_multilingual(&self) -> bool {
        self.joint.prompt().is_some()
    }

    /// Select the target language, e.g. `"en-US"`, `"de"`, or `"auto"` to let
    /// the model detect it (the default).
    ///
    /// Applies to offline transcription and to streaming chunks processed
    /// afterwards. Fails for models without a language prompt.
    pub fn set_language(&mut self, language: &str) -> Result<()> {
        self.joint
            .prompt_mut()
            .ok_or_else(|| Error::config("this model has no language prompt"))?
            .set_language(language)
    }

    /// The selected target language, for prompt-conditioned models.
    pub fn language(&self) -> Option<&str> {
        self.joint.prompt().map(|p| p.language())
    }

    /// Every accepted target language key (empty for models without a prompt).
    pub fn languages(&self) -> Vec<&str> {
        self.joint
            .prompt()
            .map(|p| p.languages().collect())
            .unwrap_or_default()
    }

    /// Keep or strip the language tags (e.g. `<en-US>`) that prompt-conditioned
    /// models emit after each utterance's terminal punctuation.
    ///
    /// Tags are stripped from text by default; token ids always keep them.
    pub fn set_strip_lang_tags(&mut self, strip: bool) {
        self.strip_lang_tags = strip;
    }

    /// The language named by the last language tag in `tokens`, e.g. `"en-US"`.
    pub fn detected_language(&self, tokens: &[usize]) -> Option<String> {
        last_lang_tag(self.tokenizer.as_ref()?, &self.lang_tags, tokens)
    }

    /// Text of `tokens`, without language tags unless they are kept.
    fn tokens_to_text(&self, tokenizer: &SentencePieceProcessor, tokens: &[usize]) -> String {
        decode_text(tokenizer, tokens, self.strip_lang_tags)
    }

    /// Set the decoding strategy.
    pub fn set_decoding_strategy(&mut self, strategy: RnntDecodingStrategy) {
        self.decoding_strategy = strategy;
    }

    /// Run the encoder in `dtype` instead of float32, e.g. `Dtype::Bfloat16`.
    ///
    /// Roughly halves encoder time on Apple Silicon for a small accuracy
    /// cost. Mel features, the decoder and the joint network stay float32.
    /// Applies to [`transcribe`](Self::transcribe) and
    /// [`forward`](Self::forward), not the streaming session.
    pub fn set_encoder_dtype(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        self.encoder.set_dtype(dtype)
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
        let lengths: Vec<i32> = encoder_lengths
            .as_dtype(mlx_rs::Dtype::Int32)?
            .to_vec_row_major()?;
        let greedy = RnntStreamingDecoder::new(self.blank_id);

        // Each sequence decodes independently from a fresh decoder state
        let mut results = Vec::with_capacity(lengths.len());
        for (b, &len) in lengths.iter().enumerate() {
            let seq_len = (len.max(0) as usize).min(encoder_output.dim(1) as usize);
            let frames = encoder_output.index((b as i32, ..seq_len as i32, ..));
            let enc_proj = self.joint.project_encoder(&frames)?;
            let mut state = self.decoder.initialize_state(1)?;
            let mut last_token = None;
            let output = greedy.decode_projected(
                &enc_proj,
                seq_len,
                &self.decoder,
                &mut state,
                &self.joint,
                &mut last_token,
                None,
            )?;
            results.push(output.new_tokens);
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

        // Encode
        let (encoded, encoded_lengths) = self.encoder.forward(&features, &feature_lengths)?;
        tracing::debug!(encoded_shape = ?encoded.shape(), "encoder output");

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
            texts.push(self.tokens_to_text(tokenizer, tokens));
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

        // Note: preprocessor is stateless, no reset needed

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
        use mlx_rs::ops::indexing::IndexOp;
        let batch_size = session.encoder_cache.batch_size();

        // Step 1: Use the stateless mel spectrogram API
        // Only process if we have audio samples
        let _n_frames = if !audio_chunk.is_empty() {
            let audio_array = Array::from_slice(audio_chunk, &[1, audio_chunk.len() as i32]);
            let (mel_features, _) = self.preprocessor.forward(&audio_array, None)?;
            // mel_features is [batch=1, n_mels, frames]
            // Remove batch dimension: [n_mels, frames]
            let mel_features = mel_features.index(0);

            tracing::debug!(
                "PROCESS CHUNK #{} len({}) Got mel_features {:?}",
                self.chunk_count,
                audio_chunk.len(),
                mel_features.shape()
            );
            self.chunk_count += 1;

            // Track total audio samples for logging
            session.mel_buffer.total_audio_samples += audio_chunk.len();

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
            }
            n_frames
        } else {
            // Empty chunk is ok, just process any buffered mel frames
            0
        };

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
            // NeMo drops drop_extra_pre_encoded frames on subsequent chunks only
            let drop_extra = session.config.drop_extra_for_step(session.is_first_chunk);

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

            // Add batch dimension: [n_mels, time] -> [1, n_mels, time]
            let features = mel_chunk.reshape(&[1, session.n_mels as i32, chunk_len as i32])?;

            // Get lengths for encoder
            let lengths = Array::from_slice(&[chunk_len as i32], &[1]);

            // Run encoder with cache
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
            // A host array, so this doesn't wait on the GPU
            let raw_enc_len = encoded_lengths.to_vec_row_major::<i32>()?[0].max(0) as usize;

            // Truncate to valid_out_len, but also respect actual tensor shape
            let actual_tensor_frames = encoded.shape()[1] as usize;
            let enc_seq_len = raw_enc_len
                .min(session.config.valid_out_len)
                .min(actual_tensor_frames);
            total_encoder_frames_this_call += enc_seq_len;

            let greedy = RnntStreamingDecoder::new(session.blank_id);
            for b in 0..batch_size {
                let row = encoded.index((b as i32..b as i32 + 1, .., ..));
                let output = greedy.decode_frames(
                    &row,
                    enc_seq_len,
                    &self.decoder,
                    &mut session.decoder_states[b],
                    &self.joint,
                    &mut session.last_tokens[b],
                    None,
                )?;
                session.accumulated_tokens[b].extend(&output.new_tokens);
                all_new_tokens[b].extend(output.new_tokens);
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
                self.tokens_to_text(tokenizer, &session.accumulated_tokens[b])
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
    /// Processes any remaining mel frames in the buffer.
    pub fn finalize_streaming(
        &mut self,
        mut session: RnntStreamingSession,
    ) -> Result<StreamingOutput> {
        // Note: preprocessor is stateless, no flush needed
        // All audio has been processed through process_chunk()

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
                self.tokens_to_text(tokenizer, tokens)
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
