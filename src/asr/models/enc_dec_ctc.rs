//! Encoder-Decoder CTC ASR model.
//!
//! This model implements the CTC-based ASR architecture used by
//! NeMo's EncDecCTCModel and EncDecCTCModelBPE.
//!
//! # Streaming Support
//!
//! For streaming-capable models (e.g., Nemotron), this module provides
//! cache-aware streaming inference:
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # use mlx_voice::asr::EncDecCTCModel;
//! # let audio_chunks: Vec<Vec<f32>> = Vec::new();
//! let mut model = EncDecCTCModel::from_pretrained("model.nemo")?;
//!
//! // Check streaming support and modes
//! if model.supports_streaming() {
//!     for mode in model.streaming_modes() {
//!         println!("Mode {}: {}ms latency", mode.mode_idx, mode.latency_ms);
//!     }
//!
//!     // Initialize streaming session
//!     let mut session = model.init_streaming(0, 1)?;
//!
//!     // Process audio chunks
//!     for chunk in &audio_chunks {
//!         let output = model.process_chunk(&mut session, chunk)?;
//!         print!("{}", output.text[0]);
//!     }
//!
//!     // Finalize
//!     let final_output = model.finalize_streaming(session)?;
//!     println!("{}", final_output.text[0]);
//! }
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use mlx_rs::Array;

use crate::asr::Tokenizer;
use crate::asr::config::AsrModelConfig;
use crate::asr::decoding::{greedy_ctc_merge, greedy_decode, lcs_merge_buffer};
use crate::asr::modules::{CTCDecoder, ConformerEncoder, MelProcessor};
use crate::asr::streaming::{
    EncoderCache, StreamingConfig, StreamingModeInfo, StreamingOutput, compute_streaming_config,
    get_streaming_modes,
};
use crate::asr::transcript::{AudioInput, DecodingStrategy, TranscribeOptions, Transcript};
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::error::{Error, Result};
use crate::core::loading::{LoadOptions, LoadReport, Weights};
use crate::core::quantization::QuantizationConfig;
use crate::core::source::{LocalModel, ModelSource};

/// Encoder-Decoder CTC ASR Model.
///
/// This model consists of:
/// 1. Audio preprocessor (Mel spectrogram)
/// 2. Conformer/FastConformer encoder
/// 3. CTC decoder
///
/// # Example
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// use mlx_voice::asr::EncDecCTCModel;
///
/// // Load from .nemo checkpoint
/// let mut model = EncDecCTCModel::from_pretrained("model.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// # Ok(())
/// # }
/// ```
pub struct EncDecCTCModel {
    config: AsrModelConfig,
    preprocessor: MelProcessor,
    encoder: ConformerEncoder,
    decoder: CTCDecoder,
    tokenizer: Option<Tokenizer>,
    /// Blank token ID for CTC (typically last token in vocab)
    blank_id: usize,
    chunk_count: usize,
}

impl CtcStreamingSession {
    /// The session's streaming configuration (chunk and cache sizes).
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

impl std::fmt::Debug for EncDecCTCModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncDecCTCModel")
            .field("sample_rate", &self.sample_rate())
            .field("encoder_layers", &self.encoder.num_layers())
            .field("blank_id", &self.blank_id)
            .field("tokenizer", &self.tokenizer)
            .finish_non_exhaustive()
    }
}

impl EncDecCTCModel {
    /// Create a new model from configuration.
    pub fn new(config: AsrModelConfig) -> Result<Self> {
        config.validate()?;
        // NeMo only dithers during training
        let mut mel_config: crate::asr::modules::MelConfig = config.preprocessor.clone().into();
        mel_config.dither = 0.0;
        let preprocessor = MelProcessor::new(mel_config)?;
        let encoder = ConformerEncoder::new(config.encoder.clone())?;

        // Determine vocabulary size from config or default
        let vocab_size = if config.decoder.num_classes > 0 {
            config.decoder.num_classes as usize
        } else {
            // Will be set when loading tokenizer
            1024 // Default placeholder
        };

        let decoder = CTCDecoder::new(config.decoder.clone(), encoder.output_dim(), vocab_size)?;

        // Blank is typically the last token
        let blank_id = vocab_size - 1;

        Ok(Self {
            config,
            preprocessor,
            encoder,
            decoder,
            tokenizer: None,
            blank_id,
            chunk_count: 0,
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
    /// use mlx_voice::asr::EncDecCTCModel;
    ///
    /// let mut model = EncDecCTCModel::from_pretrained("model.nemo")?;
    /// // Downloads with the `hub` feature
    /// let mut model = EncDecCTCModel::from_pretrained("nvidia/parakeet-ctc-0.6b")?;
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
        tracing::info!("Loading model from: {}", path.display());

        // Open the checkpoint
        let checkpoint = NemoCheckpoint::open(path)?;

        // Parse the configuration
        let config: AsrModelConfig = checkpoint.load_config()?;

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
    /// # use mlx_voice::{asr::EncDecCTCModel, WeightFormat};
    /// // Load with explicit format
    /// let model = EncDecCTCModel::from_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     Some(WeightFormat::Mlx),
    /// )?;
    ///
    /// // Load with auto-detected format
    /// let model = EncDecCTCModel::from_safetensors(
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
            "Loading model from safetensors: weights={}, config={}",
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

        tracing::info!("Loading model from directory: {}", dir.display());

        let loader = SafetensorsLoader::from_directory(dir)?;

        // Load config
        let config: AsrModelConfig = loader.load_config()?;
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

        // Blank comes after the vocabulary
        self.blank_id = tokenizer.vocab_size();

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
        weights: &std::collections::HashMap<String, Array>,
        quant: Option<&QuantizationConfig>,
    ) -> Result<()> {
        self.load_weights_with(weights, quant, LoadOptions::default())
            .map(|_| ())
    }

    /// Load weights from a checkpoint's tensors with `options`, returning the
    /// tensors a non-strict load had to leave at their initial values.
    pub fn load_weights_with(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
        quant: Option<&QuantizationConfig>,
        options: LoadOptions,
    ) -> Result<LoadReport> {
        let weights = Weights::new(weights, options);
        self.encoder.load_weights(&weights, quant)?;
        self.decoder.load_weights(&weights, "decoder", quant)?;
        Ok(weights.report())
    }

    /// Transcribe WAV files or in-memory audio (see [`AudioInput`]) with the
    /// default [`TranscribeOptions`].
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

    /// Transcribe with explicit options. CTC models decode greedily and have
    /// no language prompt, so beam search and a language are errors.
    pub fn transcribe_with<'a, I>(
        &mut self,
        inputs: I,
        options: &TranscribeOptions,
    ) -> Result<Vec<Transcript>>
    where
        I: IntoIterator,
        I::Item: Into<AudioInput<'a>>,
    {
        if !matches!(options.decoding, None | Some(DecodingStrategy::Greedy)) {
            return Err(Error::not_implemented("beam search for CTC models"));
        }
        if options.language.is_some() {
            return Err(Error::config("this model has no language prompt"));
        }
        if self.tokenizer.is_none() {
            return Err(Error::tokenizer(
                "No tokenizer loaded. Call load_tokenizer first.",
            ));
        }

        let sample_rate = self.sample_rate();
        let decoded = super::decode_in_batches(inputs, sample_rate, options.batch_size, |clips| {
            let (audio, lengths) = super::pad_batch(clips);
            let (log_probs, lengths) = self.forward(&audio, Some(&lengths))?;
            greedy_decode(&log_probs, &lengths, self.blank_id)
        })?;

        let frame_sec =
            self.config.preprocessor.window_stride * self.encoder.subsampling_factor() as f32;
        let tokenizer = self.tokenizer.as_ref().expect("checked above");
        Ok(decoded
            .iter()
            .map(|tokens| {
                Transcript::from_tokens(tokenizer, tokens, frame_sec, options.strip_lang_tags)
            })
            .collect())
    }

    /// Run forward pass on preprocessed audio.
    ///
    /// # Arguments
    /// * `audio` - Raw audio waveform [batch, samples]
    /// * `lengths` - Optional audio lengths [batch]
    ///
    /// # Returns
    /// * Log probabilities [batch, time, vocab]
    /// * Output lengths [batch]
    pub fn forward(&mut self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Preprocess audio to mel spectrogram
        let (features, feature_lengths) = self.preprocessor.forward(audio, lengths)?;

        // Encode
        let (encoded, encoded_lengths) = self.encoder.forward(&features, &feature_lengths)?;

        // Decode to log probabilities
        let log_probs = self.decoder.forward(&encoded)?;

        Ok((log_probs, encoded_lengths))
    }

    /// Get the model configuration.
    pub fn config(&self) -> &AsrModelConfig {
        &self.config
    }

    /// Get the sample rate expected by the model.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get a reference to the preprocessor (for debugging).
    pub fn preprocessor(&self) -> &MelProcessor {
        &self.preprocessor
    }

    // ==================== Streaming API ====================

    /// Check if this model supports streaming inference.
    ///
    /// Returns true if the encoder has streaming configuration
    /// (chunked_limited attention style).
    pub fn supports_streaming(&self) -> bool {
        self.encoder.supports_streaming()
    }

    /// Get available streaming modes with latency information.
    ///
    /// Each mode represents a different latency/accuracy tradeoff.
    /// Lower mode indices typically have higher latency but better accuracy.
    pub fn streaming_modes(&self) -> Vec<StreamingModeInfo> {
        // Frame duration in ms (after subsampling)
        let frame_duration_ms = self.config.preprocessor.window_stride
            * 1000.0
            * self.encoder.subsampling_factor() as f32;

        get_streaming_modes(self.encoder.config(), frame_duration_ms)
    }

    /// Initialize a streaming session for the given mode.
    ///
    /// # Arguments
    /// * `mode_idx` - Streaming mode index into the checkpoint's `att_context_size` list (see
    ///   `streaming_modes()` for each mode's latency). NeMo checkpoints list modes
    ///   from most to least lookahead, so mode 0, the model's default, has the
    ///   highest latency and accuracy
    /// * `batch_size` - Number of independent audio streams to process
    ///
    /// # Returns
    /// A `CtcStreamingSession` containing the initial cache state
    pub fn init_streaming(
        &mut self,
        mode_idx: usize,
        batch_size: usize,
    ) -> Result<CtcStreamingSession> {
        if !self.supports_streaming() {
            return Err(Error::config(
                "Model does not support streaming. Use a streaming-capable model like Nemotron.",
            ));
        }

        // Compute streaming configuration
        let config = compute_streaming_config(self.encoder.config(), mode_idx, 10000)?;

        // Setup encoder with cache drop size
        self.encoder.setup_streaming(config.cache_drop_size);

        // Create initial encoder cache
        let encoder_cache = EncoderCache::new(
            self.encoder.num_layers(),
            batch_size,
            config.last_channel_cache_size,
            config.last_time_cache_size,
            self.encoder.d_model(),
        )?;

        // Note: preprocessor is stateless, no reset needed

        Ok(CtcStreamingSession {
            config,
            encoder_cache,
            prev_tokens: vec![Vec::new(); batch_size],
            is_first_chunk: true,
            blank_id: self.blank_id,
        })
    }

    /// Process an audio chunk and return partial transcription.
    ///
    /// Uses the streaming mel spectrogram API which maintains preemphasis
    /// state and audio remainder buffer internally for seamless chunk boundaries.
    ///
    /// # Arguments
    /// * `session` - Mutable reference to the streaming session
    /// * `audio_chunk` - Audio samples [batch, samples] or [samples] for single stream
    ///
    /// # Returns
    /// Streaming output with partial tokens and text
    pub fn process_chunk(
        &mut self,
        session: &mut CtcStreamingSession,
        audio_chunk: &[f32],
    ) -> Result<StreamingOutput> {
        let batch_size = session.encoder_cache.batch_size();

        // Use the stateless mel spectrogram API
        let audio_array = Array::from_slice(audio_chunk, &[1, audio_chunk.len() as i32]);
        let (mel_features, _) = self.preprocessor.forward(&audio_array, None)?;
        // mel_features is [batch=1, n_mels, frames]

        tracing::debug!(
            "PROCESS CHUNK #{}: Got mel_features size {:?}",
            self.chunk_count,
            mel_features.shape()
        );
        self.chunk_count += 1;

        // Check if we got any frames
        let n_frames = mel_features.dim(2) as usize;
        if n_frames == 0 {
            return Ok(StreamingOutput::default());
        }

        // Already in correct shape [batch=1, n_mels, frames] for encoder
        let features = mel_features;

        // Get lengths
        let seq_len = features.dim(2);
        let lengths = Array::from_slice(&[seq_len as i32], &[1]);

        // NeMo always applies drop_extra_pre_encoded when cache is provided,
        // even on the first chunk (when cache is empty but not None).
        // The streaming mode means cache is always passed, so always drop.
        let drop_extra = session.config.drop_extra_pre_encoded;

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

        // Update cache
        session.encoder_cache.last_channel = cache_channel_next;
        session.encoder_cache.last_time = cache_time_next;
        session.encoder_cache.last_channel_len = cache_len_next;

        // Decode to log probabilities
        let log_probs = self.decoder.forward(&encoded)?;

        // Greedy decode
        let token_sequences: Vec<Vec<usize>> =
            greedy_decode(&log_probs, &encoded_lengths, self.blank_id)?
                .into_iter()
                .map(|tokens| tokens.into_iter().map(|t| t.id).collect())
                .collect();
        tracing::debug!(?token_sequences, "decoded token sequences");

        // Merge with previous tokens using LCS
        let delay = session.config.lookahead_frames;
        let mut merged_tokens = Vec::with_capacity(batch_size);
        let mut texts = Vec::with_capacity(batch_size);

        for (i, new_tokens) in token_sequences.into_iter().enumerate() {
            let merged = lcs_merge_buffer(&session.prev_tokens[i], &new_tokens, delay, 5, 1);

            // Collapse CTC output
            let collapsed = greedy_ctc_merge(&merged, session.blank_id);

            // Decode to text
            let text = self
                .tokenizer
                .as_ref()
                .map(|tokenizer| tokenizer.decode(&collapsed, false))
                .unwrap_or_default();

            session.prev_tokens[i] = merged;
            merged_tokens.push(collapsed);
            texts.push(text);
        }

        session.is_first_chunk = false;

        Ok(StreamingOutput {
            log_probs: Some(log_probs),
            tokens: merged_tokens,
            text: texts,
            is_final: false,
        })
    }

    /// Finalize streaming and get final transcription.
    ///
    /// Returns the complete transcription from accumulated tokens.
    pub fn finalize_streaming(&mut self, session: CtcStreamingSession) -> Result<StreamingOutput> {
        // Note: preprocessor is stateless, no flush needed
        // All audio has been processed through process_chunk()

        // Final decode of accumulated tokens
        let batch_size = session.prev_tokens.len();
        let mut final_tokens = Vec::with_capacity(batch_size);
        let mut final_texts = Vec::with_capacity(batch_size);

        for tokens in &session.prev_tokens {
            let collapsed = greedy_ctc_merge(tokens, session.blank_id);

            let text = self
                .tokenizer
                .as_ref()
                .map(|tokenizer| tokenizer.decode(&collapsed, false))
                .unwrap_or_default();

            final_tokens.push(collapsed);
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

/// Streaming session state for CTC models.
///
/// Contains all state needed for incremental streaming inference.
#[derive(Debug)]
pub struct CtcStreamingSession {
    /// Streaming configuration (chunk sizes, cache sizes, etc.)
    config: StreamingConfig,

    /// Encoder cache state (attention + convolution)
    encoder_cache: EncoderCache,

    /// Previous tokens for each batch sample (for LCS merge)
    prev_tokens: Vec<Vec<usize>>,

    /// Whether this is the first chunk (affects chunk size)
    is_first_chunk: bool,

    /// Blank token ID for CTC decoding
    blank_id: usize,
}
