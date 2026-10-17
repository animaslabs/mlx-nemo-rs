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
//! let mut model = EncDecRNNTModel::from_pretrained("model.nemo")?;
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

use crate::asr::Tokenizer;
use crate::asr::config::{AsrModelConfig, JointConfig};
use crate::asr::decoding::TimedToken;
use crate::asr::decoding::{RnntStreamingDecoder, rnnt_beam_decode};
use crate::asr::modules::{
    ConformerEncoder, DecoderState, LanguagePrompt, MelConfig, MelProcessor, PredictionNetwork,
    RNNTDecoder, RNNTJoint, StatelessTransducerDecoder,
};
use crate::asr::streaming::{
    EncoderCache, StreamingConfig, StreamingMelBuffer, StreamingModeInfo, StreamingOutput,
    compute_streaming_config, get_streaming_modes,
};
use crate::asr::transcript::{AudioInput, DecodingStrategy, TranscribeOptions, Transcript};
use crate::core::ArrayExt;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::error::{Error, Result};
use crate::core::loading::{LoadOptions, LoadReport, Weights};
use crate::core::quantization::QuantizationConfig;
use crate::core::source::{LocalModel, ModelSource};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

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
/// let mut model = EncDecRNNTModel::from_pretrained("parakeet-rnnt-0.6b.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// # Ok(())
/// # }
/// ```
pub struct EncDecRNNTModel {
    config: AsrModelConfig,
    preprocessor: MelProcessor,
    encoder: ConformerEncoder,
    decoder: PredictionNetwork,
    joint: RNNTJoint,
    tokenizer: Option<Tokenizer>,
    /// Blank token ID (typically vocab_size, last token)
    blank_id: usize,
    /// Decoding strategy
    decoding_strategy: DecodingStrategy,
    chunk_count: usize,
    /// Leave language tags out of transcribed text
    strip_lang_tags: bool,
}

impl RnntStreamingSession {
    /// The session's streaming configuration (chunk and cache sizes).
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

impl std::fmt::Debug for EncDecRNNTModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncDecRNNTModel")
            .field("sample_rate", &self.sample_rate())
            .field("encoder_layers", &self.encoder.num_layers())
            .field("lstm_decoder", &self.is_lstm_decoder())
            .field("blank_id", &self.blank_id)
            .field("decoding_strategy", &self.decoding_strategy)
            .field("tokenizer", &self.tokenizer)
            .field("language", &self.language())
            .finish_non_exhaustive()
    }
}

impl EncDecRNNTModel {
    /// Create a new model from configuration.
    pub fn new(config: AsrModelConfig) -> Result<Self> {
        config.validate()?;
        // NeMo only dithers during training; at inference it would make the
        // output random
        let mut preproc_config: MelConfig = config.preprocessor.clone().into();
        preproc_config.dither = 0.0;
        let preprocessor = MelProcessor::new(preproc_config)?;
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
                "beam" | "beam_search" => DecodingStrategy::BeamSearch {
                    beam_size: decoding.beam_size.unwrap_or(4),
                },
                _ => DecodingStrategy::Greedy,
            }
        } else {
            DecodingStrategy::Greedy
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
            strip_lang_tags: true,
        })
    }

    /// Load a model from a `.nemo` archive, a model directory or a Hugging
    /// Face Hub repository (see [`ModelSource`]).
    ///
    /// A model directory holds safetensors weights (`model.safetensors`), a
    /// config (`config.json`, `config.yaml` or `model_config.yaml`) and a
    /// SentencePiece tokenizer (`tokenizer.model`). Fails if the checkpoint
    /// lacks a tokenizer or any weight the model has;
    /// [`from_pretrained_with`](Self::from_pretrained_with) can allow that.
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// use mlx_voice::asr::EncDecRNNTModel;
    ///
    /// let mut model = EncDecRNNTModel::from_pretrained("model.nemo")?;
    /// // Downloads with the `hub` feature
    /// let mut model = EncDecRNNTModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?;
    /// let texts = model.transcribe(&["audio.wav"])?;
    /// # Ok(())
    /// # }
    /// ```
    #[doc(alias = "restore")]
    #[doc(alias = "restore_from")]
    pub fn from_pretrained(source: impl Into<ModelSource>) -> Result<Self> {
        Self::from_pretrained_with(source, LoadOptions::default())
    }

    /// Like [`from_pretrained`](Self::from_pretrained), loading with `options`.
    pub fn from_pretrained_with(
        source: impl Into<ModelSource>,
        options: LoadOptions,
    ) -> Result<Self> {
        match source.into().resolve()? {
            LocalModel::Nemo(path) => Self::from_nemo(&path, options),
            LocalModel::Directory(dir) => Self::from_directory(&dir, options),
        }
    }

    /// Load from a `.nemo` archive.
    fn from_nemo(path: &Path, options: LoadOptions) -> Result<Self> {
        tracing::info!("Loading RNNT model from: {}", path.display());

        // Open the checkpoint
        let checkpoint = NemoCheckpoint::open(path)?;

        // Parse the configuration
        let config: AsrModelConfig = checkpoint.load_config()?;

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

        match checkpoint.find_tokenizer() {
            Some(tokenizer_path) => {
                tracing::debug!("Loading tokenizer from: {}", tokenizer_path.display());
                model.load_tokenizer(&tokenizer_path)?;
            }
            None => options.missing_tokenizer(path)?,
        }

        // Load weights (.nemo files are not quantized)
        let weights = checkpoint.load_weights()?;
        tracing::debug!("Loading {} weight tensors", weights.len());
        model.load_weights_with(&weights, None, options)?;

        Ok(model)
    }

    /// Load a model from standalone safetensors files.
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
    /// let model = EncDecRNNTModel::from_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     Some(WeightFormat::Mlx),
    /// )?;
    ///
    /// // Load with auto-detected format
    /// let model = EncDecRNNTModel::from_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     None,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_safetensors(
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
            loader.weight_format(),
            quant.is_some()
        );
        model.load_weights(&weights, quant)?;

        Ok(model)
    }

    /// Load from a directory of safetensors weights, config and tokenizer.
    fn from_directory(dir: &Path, options: LoadOptions) -> Result<Self> {
        use crate::core::checkpoint::SafetensorsLoader;

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

        match loader.tokenizer_path() {
            Some(tok_path) => {
                tracing::debug!("Loading tokenizer from: {}", tok_path.display());
                model.load_tokenizer(tok_path)?;
            }
            None => options.missing_tokenizer(dir)?,
        }

        // Load weights (may be quantized)
        let weights = loader.load_weights()?;
        let quant = loader.quantization();
        tracing::debug!(
            "Loading {} weight tensors (format: {:?}, quantized: {})",
            weights.len(),
            loader.weight_format(),
            quant.is_some()
        );
        model.load_weights_with(&weights, quant, options)?;

        Ok(model)
    }

    /// Load a tokenizer from a SentencePiece model file.
    pub fn load_tokenizer(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::open(path)?;

        // Update vocabulary size and blank token
        let vocab_size = tokenizer.vocab_size();
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

    /// The model's tokenizer, for turning token ids into text (`None` only
    /// if a non-strict load found no tokenizer).
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Load weights from a checkpoint's tensors, failing if any tensor the
    /// model has is missing.
    ///
    /// # Arguments
    /// * `weights` - Weight dictionary
    /// * `quant` - Optional quantization config for quantized models
    pub fn load_weights(
        &mut self,
        weights: &HashMap<String, Array>,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        self.load_weights_with(weights, quant, LoadOptions::default())
            .map(|_| ())
    }

    /// Load weights from a checkpoint's tensors with `options`, returning the
    /// tensors a non-strict load had to leave at their initial values.
    pub fn load_weights_with(
        &mut self,
        weights: &HashMap<String, Array>,
        quant: Option<&QuantizationConfig>,
        options: LoadOptions,
    ) -> Result<LoadReport> {
        let weights = Weights::new(weights, options);

        // Load encoder weights (linear layers may be quantized)
        self.encoder.load_weights(&weights, quant)?;

        // Load decoder (prediction network) weights (embeddings may be quantized)
        self.decoder.load_weights(&weights, "decoder", quant)?;

        // Load joint network weights (linear layers may be quantized)
        self.joint.load_weights(&weights, "joint", quant)?;

        // Language-ID prompt (prompt-conditioned models only)
        self.joint
            .set_prompt(LanguagePrompt::from_config(&self.config, &weights, quant)?);

        Ok(weights.report())
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

    /// The language named by the last language tag in `tokens`, e.g. `"en-US"`.
    pub fn detected_language(&self, tokens: &[usize]) -> Option<String> {
        self.tokenizer.as_ref()?.detected_language(tokens)
    }

    /// Text of `tokens`, without language tags unless they are kept.
    fn tokens_to_text(&self, tokenizer: &Tokenizer, tokens: &[usize]) -> String {
        tokenizer.decode(tokens, self.strip_lang_tags)
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

    /// The decoding strategy from the model's config, used when
    /// [`TranscribeOptions::decoding`] is `None`.
    pub fn decoding_strategy(&self) -> DecodingStrategy {
        self.decoding_strategy
    }

    /// Transcribe WAV files or in-memory audio (see [`AudioInput`]) with the
    /// default [`TranscribeOptions`].
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::asr::EncDecRNNTModel;
    /// let mut model = EncDecRNNTModel::from_pretrained("model.nemo")?;
    /// for transcript in model.transcribe(["a.wav", "b.wav"])? {
    ///     println!("{transcript}");
    ///     for word in &transcript.words {
    ///         println!("{:6.2}-{:6.2} {}", word.start, word.end, word.text);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn transcribe<'a, I>(&mut self, inputs: I) -> Result<Vec<Transcript>>
    where
        I: IntoIterator,
        I::Item: Into<AudioInput<'a>>,
    {
        self.transcribe_with(inputs, &TranscribeOptions::default())
    }

    /// Transcribe one WAV file or in-memory clip.
    pub fn transcribe_one<'a>(&mut self, input: impl Into<AudioInput<'a>>) -> Result<Transcript> {
        Ok(self.transcribe([input.into()])?.remove(0))
    }

    /// Transcribe with explicit options (decoding strategy, language, batch
    /// size).
    pub fn transcribe_with<'a, I>(
        &mut self,
        inputs: I,
        options: &TranscribeOptions,
    ) -> Result<Vec<Transcript>>
    where
        I: IntoIterator,
        I::Item: Into<AudioInput<'a>>,
    {
        if self.tokenizer.is_none() {
            return Err(Error::tokenizer(
                "No tokenizer loaded. Call load_tokenizer first.",
            ));
        }
        let decoding = options.decoding.unwrap_or(self.decoding_strategy);

        // Switch language for this call only
        let previous_language = match &options.language {
            Some(language) => {
                let previous = self.language().map(str::to_owned);
                self.set_language(language)?;
                previous
            }
            None => None,
        };
        let sample_rate = self.sample_rate();
        let decoded = super::decode_in_batches(inputs, sample_rate, options.batch_size, |clips| {
            self.decode_clips(clips, decoding)
        });
        if let Some(language) = previous_language {
            self.set_language(&language)?;
        }

        let frame_sec = self.frame_sec();
        let tokenizer = self.tokenizer.as_ref().expect("checked above");
        Ok(decoded?
            .iter()
            .map(|tokens| {
                Transcript::from_tokens(tokenizer, tokens, frame_sec, options.strip_lang_tags)
            })
            .collect())
    }

    /// Seconds per encoder frame.
    fn frame_sec(&self) -> f32 {
        self.config.preprocessor.window_stride * self.encoder.subsampling_factor() as f32
    }

    /// Decode clips already at the model's sample rate.
    fn decode_clips(
        &mut self,
        clips: &[Vec<f32>],
        decoding: DecodingStrategy,
    ) -> Result<Vec<Vec<TimedToken>>> {
        let (audio, lengths) = super::pad_batch(clips);
        let (encoder_output, encoder_lengths) = self.forward(&audio, Some(&lengths))?;

        match decoding {
            DecodingStrategy::Greedy => self.greedy_decode(&encoder_output, &encoder_lengths),
            DecodingStrategy::BeamSearch { beam_size } => rnnt_beam_decode(
                &encoder_output,
                &encoder_lengths,
                &self.decoder,
                &self.joint,
                self.blank_id,
                beam_size,
            ),
        }
    }

    /// Perform greedy decoding using the model's decoder.
    fn greedy_decode(
        &self,
        encoder_output: &Array,
        encoder_lengths: &Array,
    ) -> Result<Vec<Vec<TimedToken>>> {
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
            results.push(
                output
                    .new_tokens
                    .iter()
                    .zip(&output.timesteps)
                    .map(|(&id, &t)| TimedToken {
                        id,
                        start_frame: t,
                        end_frame: t + 1,
                    })
                    .collect(),
            );
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
    pub fn preprocessor(&self) -> &MelProcessor {
        &self.preprocessor
    }

    /// Get a reference to the encoder.
    #[cfg(feature = "internals")]
    pub fn encoder(&self) -> &ConformerEncoder {
        &self.encoder
    }

    /// Get a reference to the prediction network.
    #[cfg(feature = "internals")]
    pub fn prediction_network(&self) -> &PredictionNetwork {
        &self.decoder
    }

    /// Get a reference to the joint network.
    #[cfg(feature = "internals")]
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
    /// * `mode_idx` - Streaming mode index into the checkpoint's `att_context_size` list (see
    ///   `streaming_modes()` for each mode's latency). NeMo checkpoints list modes
    ///   from most to least lookahead, so mode 0, the model's default, has the
    ///   highest latency and accuracy
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
#[derive(Debug)]
pub struct RnntStreamingSession {
    /// Streaming configuration
    config: StreamingConfig,

    /// Encoder cache state
    encoder_cache: EncoderCache,

    /// Mel spectrogram buffer (stores pre-computed mel frames)
    mel_buffer: StreamingMelBuffer,

    /// Decoder states for each batch sample
    decoder_states: Vec<DecoderState>,

    /// Last emitted token for each batch sample
    last_tokens: Vec<Option<usize>>,

    /// All accumulated tokens for each batch sample
    accumulated_tokens: Vec<Vec<usize>>,

    /// Whether this is the first chunk (determines chunk_size and pre_encode_cache)
    is_first_chunk: bool,

    /// Blank token ID
    blank_id: usize,

    /// Number of mel features
    n_mels: usize,
}
