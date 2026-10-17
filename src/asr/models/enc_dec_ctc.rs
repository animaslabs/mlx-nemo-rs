//! Encoder-Decoder CTC ASR model.
//!
//! This model implements the CTC-based ASR architecture used by
//! NeMo's EncDecCTCModel and EncDecCTCModelBPE.
//!
//! NVIDIA's cache-aware streaming models with a CTC decoder are hybrid
//! RNN-T/CTC models: load those with
//! [`EncDecRnntModel`](crate::asr::EncDecRnntModel) and pick the CTC decoder
//! with [`DecoderType::Ctc`](crate::asr::DecoderType::Ctc), offline or in a
//! [`StreamingSession`](crate::asr::StreamingSession).

use std::path::Path;

use mlx_rs::Array;

use crate::asr::Tokenizer;
use crate::asr::config::AsrModelConfig;
use crate::asr::decoding::greedy_decode;
use crate::asr::modules::{ConformerEncoder, CtcDecoder, MelProcessor};
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
/// use mlx_voice::asr::EncDecCtcModel;
///
/// // Load from .nemo checkpoint
/// let mut model = EncDecCtcModel::from_pretrained("model.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// # Ok(())
/// # }
/// ```
#[doc(alias = "EncDecCTCModel")]
pub struct EncDecCtcModel {
    config: AsrModelConfig,
    preprocessor: MelProcessor,
    encoder: ConformerEncoder,
    decoder: CtcDecoder,
    tokenizer: Option<Tokenizer>,
    /// Blank token ID for CTC (typically last token in vocab)
    blank_id: usize,
}

impl std::fmt::Debug for EncDecCtcModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncDecCtcModel")
            .field("sample_rate", &self.sample_rate())
            .field("encoder_layers", &self.encoder.num_layers())
            .field("blank_id", &self.blank_id)
            .field("tokenizer", &self.tokenizer)
            .finish_non_exhaustive()
    }
}

impl EncDecCtcModel {
    /// Create a new model from configuration.
    pub fn new(config: AsrModelConfig) -> Result<Self> {
        config.validate()?;
        // NeMo only dithers during training
        let mut mel_config: crate::asr::modules::MelConfig = config.preprocessor.clone().into();
        mel_config.dither = 0.0;
        let preprocessor = MelProcessor::new(mel_config)?;
        let encoder = ConformerEncoder::new(config.encoder.clone())?;

        // Determine vocabulary size from config or default
        // Set properly when the tokenizer loads
        let vocab_size = config.decoder.num_classes.unwrap_or(1024);

        let decoder = CtcDecoder::new(config.decoder.clone(), encoder.output_dim(), vocab_size)?;

        // Blank is typically the last token
        let blank_id = vocab_size - 1;

        Ok(Self {
            config,
            preprocessor,
            encoder,
            decoder,
            tokenizer: None,
            blank_id,
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
    /// use mlx_voice::asr::EncDecCtcModel;
    ///
    /// let mut model = EncDecCtcModel::from_pretrained("model.nemo")?;
    /// // Downloads with the `hub` feature
    /// let mut model = EncDecCtcModel::from_pretrained("nvidia/parakeet-ctc-0.6b")?;
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
    /// # use mlx_voice::{asr::EncDecCtcModel, WeightFormat};
    /// // Load with explicit format
    /// let model = EncDecCtcModel::from_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     Some(WeightFormat::Mlx),
    /// )?;
    ///
    /// // Load with auto-detected format
    /// let model = EncDecCtcModel::from_safetensors(
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
}
