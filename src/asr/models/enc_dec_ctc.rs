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
//! let mut model = EncDecCTCModel::restore("model.nemo")?;
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
use crate::asr::modules::{CTCDecoder, ConformerEncoder, MelSpectrogram};
use crate::asr::streaming::{
    EncoderCache, StreamingConfig, StreamingModeInfo, StreamingOutput, compute_streaming_config,
    get_streaming_modes,
};
use crate::audio::resample;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::config::from_yaml_str;
use crate::core::error::{Error, Result};
use crate::core::loading::{LoadOptions, LoadReport, Weights};
use crate::core::quantization::QuantizationConfig;

/// Debug output containing all intermediate tensors from forward pass.
#[derive(Debug)]
pub struct DebugOutput {
    /// Mel spectrogram features BEFORE normalization [batch, n_mels, frames]
    pub mel_pre_norm: Array,
    /// Mel spectrogram features AFTER normalization [batch, n_mels, frames]
    pub mel_spectrogram: Array,
    /// Mel spectrogram lengths [batch]
    pub mel_lengths: Array,
    /// Encoder output [batch, d_model, time']
    pub encoder_output: Array,
    /// Encoder output lengths [batch]
    pub encoder_lengths: Array,
    /// Log probabilities [batch, time', vocab_size]
    pub log_probs: Array,
}

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
/// let mut model = EncDecCTCModel::restore("model.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// # Ok(())
/// # }
/// ```
pub struct EncDecCTCModel {
    config: AsrModelConfig,
    preprocessor: MelSpectrogram,
    encoder: ConformerEncoder,
    decoder: CTCDecoder,
    tokenizer: Option<Tokenizer>,
    /// Blank token ID for CTC (typically last token in vocab)
    blank_id: usize,
    chunk_count: usize,
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
        let preprocessor = MelSpectrogram::new(mel_config)?;
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

    /// Restore a model from a .nemo checkpoint file.
    ///
    /// Fails if the checkpoint lacks a tokenizer or any weight the model has;
    /// [`restore_with`](Self::restore_with) can allow that.
    pub fn restore(path: impl AsRef<Path>) -> Result<Self> {
        Self::restore_with(path, LoadOptions::default())
    }

    /// Restore a model from a .nemo checkpoint file, loading with `options`.
    pub fn restore_with(path: impl AsRef<Path>, options: LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading model from: {}", path.display());

        // Open the checkpoint
        let checkpoint = NemoCheckpoint::open(path).or_else(|_| {
            // Try gzip format for older checkpoints
            NemoCheckpoint::open_gzip(path)
        })?;

        // Parse the configuration
        let config: AsrModelConfig = from_yaml_str(&checkpoint.config_yaml)?;

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
    /// # use mlx_voice::{asr::EncDecCTCModel, WeightFormat};
    /// // Load with explicit format
    /// let model = EncDecCTCModel::restore_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     Some(WeightFormat::Mlx),
    /// )?;
    ///
    /// // Load with auto-detected format
    /// let model = EncDecCTCModel::restore_safetensors(
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
    /// # use mlx_voice::asr::EncDecCTCModel;
    /// // Load from directory structure:
    /// // my_model/
    /// // ├── model.safetensors
    /// // ├── config.yaml
    /// // └── tokenizer.model
    /// let model = EncDecCTCModel::restore_from_directory("my_model/")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn restore_from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        Self::restore_from_directory_with(dir, LoadOptions::default())
    }

    /// Like [`restore_from_directory`](Self::restore_from_directory), loading
    /// with `options`.
    pub fn restore_from_directory_with(
        dir: impl AsRef<Path>,
        options: LoadOptions,
    ) -> Result<Self> {
        use crate::core::checkpoint::SafetensorsLoader;

        let dir = dir.as_ref();
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
            loader.weight_format,
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

    /// Transcribe audio files.
    ///
    /// # Arguments
    /// * `paths` - Paths to audio files (WAV format)
    ///
    /// # Returns
    /// * Vector of transcribed text strings
    pub fn transcribe(&mut self, paths: &[impl AsRef<Path>]) -> Result<Vec<String>> {
        let clips = super::load_clips(paths, self.sample_rate())?;
        self.transcribe_clips(&clips)
    }

    /// Transcribe mono audio samples (in `[-1, 1]`) recorded at
    /// `sample_rate`, resampling them to the model's rate if needed.
    pub fn transcribe_samples(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let clip = resample(samples, sample_rate, self.sample_rate())?;
        Ok(self.transcribe_clips(&[clip])?.remove(0))
    }

    /// Transcribe clips already at the model's sample rate.
    fn transcribe_clips(&mut self, clips: &[Vec<f32>]) -> Result<Vec<String>> {
        let (audio, lengths) = super::pad_batch(clips);

        // Run inference
        let (log_probs, lengths) = self.forward(&audio, Some(&lengths))?;

        // Decode
        let token_sequences = greedy_decode(&log_probs, &lengths, self.blank_id)?;

        // Convert tokens to text
        let texts = self.decode_tokens(&token_sequences)?;

        Ok(texts)
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

    /// Decode token sequences to text.
    fn decode_tokens(&self, token_sequences: &[Vec<usize>]) -> Result<Vec<String>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::tokenizer("No tokenizer loaded. Call load_tokenizer first."))?;

        Ok(token_sequences
            .iter()
            .map(|tokens| tokenizer.decode(tokens, false))
            .collect())
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
    pub fn preprocessor(&self) -> &MelSpectrogram {
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
        let token_sequences = greedy_decode(&log_probs, &encoded_lengths, self.blank_id)?;
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
    pub config: StreamingConfig,

    /// Encoder cache state (attention + convolution)
    pub encoder_cache: EncoderCache,

    /// Previous tokens for each batch sample (for LCS merge)
    pub prev_tokens: Vec<Vec<usize>>,

    /// Whether this is the first chunk (affects chunk size)
    pub is_first_chunk: bool,

    /// Blank token ID for CTC decoding
    blank_id: usize,
}
