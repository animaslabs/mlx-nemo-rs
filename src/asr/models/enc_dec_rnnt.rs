//! Encoder-Decoder RNN-T ASR model.
//!
//! This model implements the RNN-T (Transducer) ASR architecture used by
//! NeMo's EncDecRNNTModel and EncDecRNNTBPEModel (Parakeet RNNT models),
//! including TDT (token-and-duration transducer) models such as
//! parakeet-tdt-0.6b-v3, which decode greedily offline.
//!
//! # Streaming
//!
//! Streaming-capable models transcribe incrementally through a
//! [`StreamingSession`](crate::asr::StreamingSession):
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # use mlx_voice::asr::{EncDecRnntModel, StreamingOptions};
//! # let audio_chunks: Vec<Vec<f32>> = Vec::new();
//! let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
//! let mut stream = model.stream(StreamingOptions::default())?;
//! for chunk in &audio_chunks {
//!     stream.push_audio(chunk, 16_000)?;
//!     while let Some(update) = stream.next_update()? {
//!         print!("{}", update.delta);
//!     }
//! }
//! let transcript = stream.finish()?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::asr::Tokenizer;
use crate::asr::config::{AsrModelConfig, DecodingAlgorithm, JointConfig};
use crate::asr::decoding::TimedToken;
use crate::asr::decoding::{
    RnntStreamingDecoder, TdtGreedyDecoder, greedy_decode, refine_punctuation_timestamps,
    rnnt_beam_decode,
};
use crate::asr::modules::{
    ConformerEncoder, CtcDecoder, LanguagePrompt, MelConfig, MelProcessor, PredictionNetwork,
    RnntDecoder, RnntJoint, StatelessTransducerDecoder,
};
use crate::asr::pipeline::{StreamingOptions, StreamingSession};
use crate::asr::streaming::{StreamingModeInfo, get_streaming_modes};
use crate::asr::transcript::{
    AudioInput, DecoderType, DecodingStrategy, TranscribeOptions, Transcript,
};
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
/// use mlx_voice::asr::EncDecRnntModel;
///
/// // Load from .nemo checkpoint
/// let mut model = EncDecRnntModel::from_pretrained("parakeet-rnnt-0.6b.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// # Ok(())
/// # }
/// ```
#[doc(alias = "EncDecRNNTModel")]
pub struct EncDecRnntModel {
    pub(crate) config: AsrModelConfig,
    pub(crate) preprocessor: MelProcessor,
    pub(crate) encoder: ConformerEncoder,
    pub(crate) decoder: PredictionNetwork,
    pub(crate) joint: RnntJoint,
    /// The auxiliary CTC decoder of a hybrid RNN-T/CTC model
    pub(crate) ctc_decoder: Option<CtcDecoder>,
    pub(crate) tokenizer: Option<Tokenizer>,
    /// Blank token ID (typically vocab_size, last token)
    pub(crate) blank_id: usize,
    /// Decoding strategy from the config
    decoding_strategy: DecodingStrategy,
    /// Frames each duration logit advances (TDT models only)
    tdt_durations: Option<Vec<usize>>,
}

impl std::fmt::Debug for EncDecRnntModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncDecRnntModel")
            .field("sample_rate", &self.sample_rate())
            .field("encoder_layers", &self.encoder.num_layers())
            .field("lstm_decoder", &self.is_lstm_decoder())
            .field("ctc_decoder", &self.has_ctc_decoder())
            .field("tdt_durations", &self.tdt_durations)
            .field("blank_id", &self.blank_id)
            .field("decoding_strategy", &self.decoding_strategy)
            .field("tokenizer", &self.tokenizer)
            .field("language", &self.language())
            .finish_non_exhaustive()
    }
}

impl EncDecRnntModel {
    internal_fn! {
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
            // Set properly when the tokenizer loads
            let vocab_size = config.decoder.num_classes.unwrap_or(1024);

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
                PredictionNetwork::Lstm(RnntDecoder::new(&config.decoder, vocab_size)?)
            } else {
                tracing::info!("Using stateless prediction network");
                PredictionNetwork::Stateless(StatelessTransducerDecoder::new(
                    &config.decoder,
                    vocab_size,
                )?)
            };

            let encoder_hidden = encoder.output_dim();
            let joint = Self::new_joint(&config, encoder_hidden, decoder.pred_hidden(), vocab_size)?;

            let ctc_decoder = Self::new_ctc_decoder(&config, encoder_hidden, vocab_size)?;

            // Blank is typically the last token (vocab_size)
            let blank_id = vocab_size;

            // Determine decoding strategy from config
            let decoding_strategy = if let Some(ref decoding) = config.decoding {
                match decoding.strategy {
                    DecodingAlgorithm::Beam => DecodingStrategy::BeamSearch {
                        beam_size: decoding.beam_size.unwrap_or(4),
                    },
                    _ => DecodingStrategy::Greedy,
                }
            } else {
                DecodingStrategy::Greedy
            };

            let tdt_durations = config.tdt_durations();

            Ok(Self {
                config,
                preprocessor,
                encoder,
                decoder,
                joint,
                ctc_decoder,
                tokenizer: None,
                blank_id,
                decoding_strategy,
                tdt_durations,
            })
        }
    }

    /// The joint network, with `vocab_size` tokens plus blank (and one
    /// duration logit per TDT duration), using effective jointnet values.
    fn new_joint(
        config: &AsrModelConfig,
        encoder_hidden: usize,
        pred_hidden: usize,
        vocab_size: usize,
    ) -> Result<RnntJoint> {
        let joint_config = config.joint.clone().unwrap_or_default();
        let effective_joint_config = JointConfig {
            joint_hidden: joint_config.effective_joint_hidden(),
            activation: joint_config.effective_activation().clone(),
            num_extra_outputs: config.tdt_durations().map(|d| d.len()),
            ..joint_config
        };
        RnntJoint::new(
            &effective_joint_config,
            encoder_hidden,
            pred_hidden,
            vocab_size + 1,
        )
    }

    /// The auxiliary CTC decoder of a hybrid model, with `vocab_size` tokens
    /// plus blank.
    fn new_ctc_decoder(
        config: &AsrModelConfig,
        encoder_hidden: usize,
        vocab_size: usize,
    ) -> Result<Option<CtcDecoder>> {
        config
            .aux_ctc
            .as_ref()
            .filter(|_| config.is_hybrid())
            .map(|aux| CtcDecoder::new(aux.decoder.clone(), encoder_hidden, vocab_size + 1))
            .transpose()
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
    /// use mlx_voice::asr::EncDecRnntModel;
    ///
    /// let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
    /// // Downloads with the `hub` feature
    /// let mut model = EncDecRnntModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?;
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

    internal_fn! {
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
                PredictionNetwork::Lstm(RnntDecoder::new(&self.config.decoder, vocab_size)?)
            } else {
                PredictionNetwork::Stateless(StatelessTransducerDecoder::new(
                    &self.config.decoder,
                    vocab_size,
                )?)
            };

            // Recreate joint network with correct vocab size
            let encoder_hidden = self.encoder.output_dim();
            self.joint = Self::new_joint(
                &self.config,
                encoder_hidden,
                self.decoder.pred_hidden(),
                vocab_size,
            )?;
            self.ctc_decoder = Self::new_ctc_decoder(&self.config, encoder_hidden, vocab_size)?;

            self.tokenizer = Some(tokenizer);
            Ok(())
        }
    }

    /// The model's tokenizer, for turning token ids into text (`None` only
    /// if a non-strict load found no tokenizer).
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    internal_fn! {
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
    }

    internal_fn! {
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

            // Auxiliary CTC decoder (hybrid models only)
            if let Some(ctc_decoder) = &mut self.ctc_decoder {
                ctc_decoder.load_weights(&weights, "ctc_decoder", quant)?;
            }

            // Language-ID prompt (prompt-conditioned models only)
            self.joint
                .set_prompt(LanguagePrompt::from_config(&self.config, &weights, quant)?);

            Ok(weights.report())
        }
    }

    /// Whether this is a TDT (token-and-duration transducer) model, whose
    /// joint also predicts how many encoder frames each step advances.
    ///
    /// TDT models decode greedily and offline: beam search and streaming
    /// sessions are not implemented for them.
    pub fn is_tdt(&self) -> bool {
        self.tdt_durations.is_some()
    }

    /// Whether this is a hybrid RNN-T/CTC model with an auxiliary CTC decoder,
    /// selected with [`DecoderType::Ctc`].
    pub fn has_ctc_decoder(&self) -> bool {
        self.ctc_decoder.is_some()
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

    /// Run the encoder in `dtype` instead of float32, e.g. `Dtype::Bfloat16`.
    ///
    /// Roughly halves encoder time on Apple Silicon for a small accuracy
    /// cost. Mel features, the decoder and the joint network stay float32.
    /// Applies to offline transcription ([`transcribe`](Self::transcribe)
    /// and the other `transcribe*` methods), not the streaming session.
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
    /// # use mlx_voice::asr::EncDecRnntModel;
    /// let mut model = EncDecRnntModel::from_pretrained("model.nemo")?;
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

    /// Transcribe with explicit options (decoding strategy, decoder type,
    /// language, batch size).
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::asr::{DecoderType, EncDecRnntModel, TranscribeOptions};
    /// // A hybrid RNN-T/CTC model can decode with either decoder
    /// let mut model = EncDecRnntModel::from_pretrained("hybrid.nemo")?;
    /// let options = TranscribeOptions::default().decoder_type(DecoderType::Ctc);
    /// let transcripts = model.transcribe_with(["a.wav"], &options)?;
    /// # Ok(())
    /// # }
    /// ```
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
        let decoder_type = self.resolve_decoder_type(options.decoder_type)?;
        let decoding = match decoder_type {
            DecoderType::Ctc => match options.decoding {
                None | Some(DecodingStrategy::Greedy) => DecodingStrategy::Greedy,
                Some(_) => return Err(Error::not_implemented("beam search for CTC decoding")),
            },
            DecoderType::Rnnt => options.decoding.unwrap_or(self.decoding_strategy),
        };

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
            self.decode_clips(clips, decoder_type, decoding)
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

    /// The decoder `requested`, or RNN-T; fails for CTC without a CTC decoder.
    pub(crate) fn resolve_decoder_type(
        &self,
        requested: Option<DecoderType>,
    ) -> Result<DecoderType> {
        let decoder_type = requested.unwrap_or_default();
        if decoder_type == DecoderType::Ctc && !self.has_ctc_decoder() {
            return Err(Error::config(
                "this model has no CTC decoder (only hybrid RNN-T/CTC models do)",
            ));
        }
        Ok(decoder_type)
    }

    /// Seconds per encoder frame.
    fn frame_sec(&self) -> f32 {
        self.config.preprocessor.window_stride * self.encoder.subsampling_factor() as f32
    }

    /// Decode clips already at the model's sample rate.
    fn decode_clips(
        &mut self,
        clips: &[Vec<f32>],
        decoder_type: DecoderType,
        decoding: DecodingStrategy,
    ) -> Result<Vec<Vec<TimedToken>>> {
        let (audio, lengths) = super::pad_batch(clips);
        let (encoder_output, encoder_lengths) = self.forward(&audio, Some(&lengths))?;

        if let (DecoderType::Ctc, Some(ctc_decoder)) = (decoder_type, &self.ctc_decoder) {
            let log_probs = ctc_decoder.log_probs(&encoder_output)?;
            return greedy_decode(&log_probs, &encoder_lengths, self.blank_id);
        }
        match decoding {
            DecodingStrategy::Greedy => self.greedy_decode(&encoder_output, &encoder_lengths),
            DecodingStrategy::BeamSearch { .. } if self.is_tdt() => Err(Error::not_implemented(
                "beam search for TDT models (use greedy decoding)",
            )),
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
        let tdt = match &self.tdt_durations {
            Some(durations) => {
                let decoder = TdtGreedyDecoder::new(self.blank_id, durations.clone())?;
                Some(match self.config.greedy_max_symbols() {
                    Some(max_symbols) => decoder.with_max_symbols(max_symbols),
                    None => decoder,
                })
            }
            None => None,
        };

        // Each sequence decodes independently from a fresh decoder state
        let mut results = Vec::with_capacity(lengths.len());
        for (b, &len) in lengths.iter().enumerate() {
            let seq_len = (len.max(0) as usize).min(encoder_output.dim(1) as usize);
            let frames = encoder_output.index((b as i32, ..seq_len as i32, ..));
            let enc_proj = self.joint.project_encoder(&frames)?;
            if let Some(tdt) = &tdt {
                let mut tokens =
                    tdt.decode_projected(&enc_proj, seq_len, &self.decoder, &self.joint)?;
                if let Some(tokenizer) = &self.tokenizer {
                    refine_punctuation_timestamps(&mut tokens, |id| tokenizer.is_punctuation(id));
                }
                results.push(tokens);
                continue;
            }
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

    internal_fn! {
        /// Run forward pass on preprocessed audio.
        ///
        /// # Arguments
        /// * `audio` - Raw audio waveform [batch, samples]
        /// * `lengths` - Optional audio lengths `[batch]`
        ///
        /// # Returns
        /// * Encoder output [batch, time, encoder_hidden]
        /// * Encoder output lengths `[batch]`
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
    pub fn joint(&self) -> &RnntJoint {
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

    /// Start a [`StreamingSession`] that borrows this model.
    ///
    /// Use [`StreamingSession::new`] with the model itself for a session that
    /// owns it. Offline [`transcribe`](Self::transcribe) is unaffected by the
    /// session's streaming mode.
    pub fn stream(&mut self, options: StreamingOptions) -> Result<StreamingSession<&mut Self>> {
        StreamingSession::new(self, options)
    }

    /// Transcribe a whole WAV file or clip by streaming it, as a live stream
    /// would see it (e.g. to measure streaming accuracy).
    pub fn transcribe_streaming<'a>(
        &mut self,
        input: impl Into<AudioInput<'a>>,
        options: StreamingOptions,
    ) -> Result<Transcript> {
        let sample_rate = self.sample_rate();
        let samples = input.into().load(sample_rate)?;
        let mut stream = self.stream(options)?;
        stream.push_audio(&samples, sample_rate)?;
        stream.finish()
    }
}
