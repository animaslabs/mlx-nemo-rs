//! Cache-aware RNN-T streaming pipeline.
//!
//! This module provides the main streaming pipeline for RNN-T models,
//! mirroring NeMo's `CacheAwareRNNTPipeline`.

use std::path::Path;

use mlx_rs::Array;
use sentencepiece::SentencePieceProcessor;

use crate::asr::config::AsrModelConfig;

/// Find token IDs that represent punctuation in the tokenizer vocabulary.
///
/// Returns IDs of tokens where the decoded piece consists entirely of
/// punctuation and/or whitespace, with at least one punctuation character.
fn find_punctuation_tokens(tokenizer: &SentencePieceProcessor) -> Vec<usize> {
    (0..tokenizer.len())
        .filter(|&id| {
            let piece = tokenizer.decode_piece_ids(&[id as u32]).unwrap_or_default();
            // Must have at least one punctuation char and only contain punctuation/whitespace
            piece.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace())
                && piece.chars().any(|c| c.is_ascii_punctuation())
        })
        .collect()
}
use crate::asr::decoding::{word_start_table, IncrementalDetokenizer, RnntStreamingDecoder};
use crate::asr::models::{DecoderState, PredictionNetwork};
use crate::asr::modules::{ConformerEncoder, MelProcessor, StreamingMel, StreamingMelConfig};
use crate::asr::streaming::{
    compute_streaming_config, get_streaming_modes, EncoderCache, FeatureBuffer, StreamingConfig,
    StreamingModeInfo,
};
use crate::core::ArrayExt;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::error::{Error, Result};

// Type alias for backwards compatibility
use crate::asr::modules::MelProcessor as MelSpectrogram;

/// Output from a transcription step.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOutput {
    /// Transcribed text (accumulated since stream start)
    pub text: String,
    /// New tokens emitted in this step
    pub new_tokens: Vec<usize>,
    /// All accumulated token IDs (only populated when `is_final` is true to avoid O(N²) memory)
    pub tokens: Vec<usize>,
    /// Total number of accumulated tokens (available on every step)
    pub token_count: usize,
    /// Whether this is the final output
    pub is_final: bool,
}

impl TranscribeOutput {
    /// Check if any new tokens were emitted.
    pub fn has_new_tokens(&self) -> bool {
        !self.new_tokens.is_empty()
    }
}

/// Streaming state for the pipeline.
struct StreamingState {
    /// Decoder state
    decoder_state: DecoderState,
    /// Last emitted token
    last_token: Option<usize>,
    /// All accumulated tokens
    accumulated_tokens: Vec<usize>,
    /// Whether first chunk has been processed
    is_first_chunk: bool,
    /// Step counter for logging
    step_count: usize,
    /// Cumulative statistics
    cumulative_stats: CumulativeStats,
    /// Text of the accumulated tokens
    detokenizer: IncrementalDetokenizer,
}

/// Cumulative statistics for debugging alignment.
#[derive(Debug, Clone, Default)]
struct CumulativeStats {
    /// Total audio samples processed
    total_audio_samples: usize,
    /// Total mel frames processed
    total_mel_frames: usize,
    /// Total encoder frames produced
    total_encoder_frames: usize,
    /// Total blank predictions
    total_blanks: usize,
    /// Total non-blank tokens emitted
    total_tokens: usize,
}

/// Check if audio chunk is silent based on RMS energy.
fn is_silent(audio: &[f32], threshold: f32) -> bool {
    if threshold <= 0.0 || audio.is_empty() {
        return false;
    }
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    rms < threshold
}

/// Load a WAV file and return samples and sample rate.
#[cfg(feature = "asr")]
fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    use hound::WavReader;

    tracing::debug!(path = %path.display(), "Loading WAV file");

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

    tracing::debug!(
        num_samples = samples.len(),
        sample_rate,
        duration_secs = samples.len() as f32 / sample_rate as f32,
        "WAV file loaded"
    );

    Ok((samples, sample_rate))
}

#[cfg(not(feature = "asr"))]
fn load_wav(_path: &Path) -> Result<(Vec<f32>, u32)> {
    Err(Error::not_implemented(
        "Audio loading requires the 'asr' feature",
    ))
}

/// Resample audio to target sample rate.
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate != to_rate {
        tracing::debug!(from_rate, to_rate, "Resampling audio");
    }
    crate::audio::resample(samples, from_rate, to_rate)
}

/// Cache-aware RNN-T streaming pipeline.
///
/// This pipeline handles:
/// - Audio preprocessing (mel spectrogram)
/// - Cache-aware encoder streaming
/// - Greedy RNN-T decoding
/// - Token accumulation and text generation
///
/// # Example
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// # use mlx_voice::asr::CacheAwareRNNTPipeline;
/// # let samples: Vec<f32> = Vec::new();
/// let mut pipeline = CacheAwareRNNTPipeline::restore("model.nemo", 0)?;
///
/// // File mode
/// pipeline.set_audio_file("audio.wav")?;
/// let text = pipeline.run()?;
///
/// // Live streaming
/// pipeline.reset();
/// pipeline.add_audio(&samples);
/// while let Some(output) = pipeline.transcribe_step()? {
///     print!("{}", output.text);
/// }
/// # Ok(())
/// # }
/// ```
pub struct CacheAwareRNNTPipeline {
    // Model components
    config: AsrModelConfig,
    preprocessor: MelProcessor,
    pub(super) encoder: ConformerEncoder,
    pub(super) decoder: PredictionNetwork,
    pub(super) joint: crate::asr::modules::RNNTJoint,
    pub(super) tokenizer: Option<SentencePieceProcessor>,

    // Streaming components
    pub(super) streaming_decoder: RnntStreamingDecoder,
    pub(super) streaming_config: StreamingConfig,
    encoder_cache: EncoderCache,

    // Mel streaming
    pub(super) mel_stream_config: StreamingMelConfig,
    streaming_mel: StreamingMel,
    is_first_chunk: bool,

    // Audio input state
    file_samples: Option<Vec<f32>>,
    file_position: usize,
    live_buffer: Vec<f32>,
    stream_ended: bool,
    is_done: bool,
    mel_flushed: bool,

    // State
    state: StreamingState,
    blank_id: usize,
    pub(super) n_mels: usize,

    // Punctuation suppression
    pub(super) suppress_punctuation: bool,
    pub(super) punctuation_token_ids: Vec<usize>,
    /// For each token id, whether it starts a new word
    pub(super) word_start: Vec<bool>,

    // Blank threshold
    blank_threshold: f32,

    // Silence detection
    silence_threshold: f32,
}

impl CacheAwareRNNTPipeline {
    /// Restore a pipeline from a .nemo checkpoint.
    ///
    /// # Arguments
    /// * `path` - Path to the .nemo checkpoint file
    /// * `mode_idx` - Streaming mode index (0 = highest latency)
    pub fn restore(path: impl AsRef<Path>, mode_idx: usize) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading pipeline from: {}", path.display());

        // Load checkpoint
        let checkpoint = NemoCheckpoint::open(path)?;
        let config: AsrModelConfig = checkpoint.load_config()?;
        config.validate()?;

        // Create preprocessor with dither disabled for inference
        // NeMo only applies dither during training, not inference
        let mut preproc_config: crate::asr::modules::MelSpectrogramConfig =
            config.preprocessor.clone().into();
        preproc_config.dither = 0.0;
        let preprocessor = MelSpectrogram::new(preproc_config.clone())?;

        // Create encoder
        let mut encoder = ConformerEncoder::new(config.encoder.clone())?;

        // Load tokenizer
        let tokenizer = checkpoint.find_tokenizer().and_then(|path| {
            SentencePieceProcessor::open(&path)
                .map_err(|e| tracing::warn!("Failed to load tokenizer: {}", e))
                .ok()
        });

        // Determine vocabulary size
        let vocab_size = tokenizer
            .as_ref()
            .map(|t| t.len())
            .unwrap_or(config.decoder.num_classes as usize);

        // Create decoder (prediction network)
        let is_lstm_decoder = config
            .decoder
            .prednet
            .as_ref()
            .map(|p| p.is_lstm_based())
            .unwrap_or(false);

        let mut decoder = if is_lstm_decoder {
            tracing::info!("Using LSTM-based prediction network");
            PredictionNetwork::Lstm(crate::asr::modules::RNNTDecoder::new(
                &config.decoder,
                vocab_size,
            )?)
        } else {
            tracing::info!("Using stateless prediction network");
            PredictionNetwork::Stateless(crate::asr::modules::StatelessTransducerDecoder::new(
                &config.decoder,
                vocab_size,
            )?)
        };

        // Create joint network
        let joint_config = config.joint.clone().unwrap_or_default();
        let encoder_hidden = encoder.output_dim();
        let pred_hidden = decoder.pred_hidden();
        let joint_hidden = joint_config.effective_joint_hidden();
        let activation = joint_config.effective_activation().to_string();

        let effective_joint_config = crate::asr::config::JointConfig {
            joint_hidden,
            activation,
            ..joint_config
        };

        let mut joint = crate::asr::modules::RNNTJoint::new(
            &effective_joint_config,
            encoder_hidden,
            pred_hidden,
            vocab_size + 1,
        )?;

        // Load weights (.nemo files are not quantized)
        let weights = checkpoint.load_weights()?;
        // Note: MelSpectrogram doesn't need weight loading (uses fixed mel filterbanks)
        encoder.load_weights(&weights, None)?;
        decoder.load_weights(&weights, "decoder", None)?;
        joint.load_weights(&weights, "joint", None)?;

        // Set encoder's attention context for the selected streaming mode
        // This ensures the encoder uses the correct left/right context for attention masking
        encoder.set_streaming_mode(mode_idx);

        // Compute streaming configuration
        let streaming_config = compute_streaming_config(encoder.config(), mode_idx, 10000)?;

        // Setup encoder for streaming
        encoder.setup_streaming(streaming_config.cache_drop_size);

        // Create encoder cache
        let encoder_cache = EncoderCache::new(
            encoder.num_layers(),
            1, // batch_size = 1 for single-stream
            streaming_config.last_channel_cache_size,
            streaming_config.last_time_cache_size,
            encoder.d_model(),
        )?;

        // Create mel streaming components
        let n_mels = preproc_config.features;
        let mel_stream_config = StreamingMelConfig::new(
            preproc_config.clone(),
            &streaming_config,
            encoder.subsampling_factor(),
        );
        let streaming_mel = StreamingMel::new(preproc_config)?;

        // Create streaming decoder
        let blank_id = vocab_size;
        let streaming_decoder = RnntStreamingDecoder::new(blank_id);

        // Find punctuation tokens from tokenizer
        let punctuation_token_ids = tokenizer
            .as_ref()
            .map(|t| find_punctuation_tokens(t))
            .unwrap_or_default();
        tracing::debug!(
            num_punctuation_tokens = punctuation_token_ids.len(),
            "identified punctuation tokens"
        );

        // Initialize decoder state with zeros (matches NeMo's add_sos=False behavior)
        // NeMo's greedy decoder passes state=None on first step, which initializes to zeros
        let decoder_state = decoder.initialize_state(1)?;
        let state = StreamingState {
            decoder_state,
            last_token: None,
            accumulated_tokens: Vec::new(),
            is_first_chunk: true,
            step_count: 0,
            cumulative_stats: CumulativeStats::default(),
            detokenizer: IncrementalDetokenizer::default(),
        };

        let word_start = tokenizer.as_ref().map(word_start_table).unwrap_or_default();

        Ok(Self {
            config,
            preprocessor,
            encoder,
            decoder,
            joint,
            tokenizer,
            streaming_decoder,
            streaming_config,
            encoder_cache,
            mel_stream_config,
            streaming_mel,
            is_first_chunk: true,
            file_samples: None,
            file_position: 0,
            live_buffer: Vec::new(),
            stream_ended: false,
            is_done: false,
            mel_flushed: false,
            state,
            blank_id,
            n_mels,
            suppress_punctuation: false,
            punctuation_token_ids,
            word_start,
            blank_threshold: 0.0,
            silence_threshold: 0.0,
        })
    }

    /// Restore a pipeline from standalone safetensors files.
    ///
    /// This method loads a pipeline from separate files (weights, config, tokenizer)
    /// rather than a bundled .nemo archive.
    ///
    /// # Arguments
    /// * `weights_path` - Path to the safetensors weights file
    /// * `config_path` - Path to the config file (YAML or JSON with NeMo AsrModelConfig schema)
    /// * `tokenizer_path` - Optional path to the SentencePiece tokenizer file
    /// * `mode_idx` - Streaming mode index (0 = highest latency)
    /// * `format` - Optional weight format. If None, auto-detects from safetensors metadata
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::{asr::CacheAwareRNNTPipeline, WeightFormat};
    /// let pipeline = CacheAwareRNNTPipeline::restore_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     0,
    ///     Some(WeightFormat::Mlx),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn restore_safetensors(
        weights_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        tokenizer_path: Option<impl AsRef<Path>>,
        mode_idx: usize,
        format: Option<crate::core::WeightFormat>,
    ) -> Result<Self> {
        use crate::core::checkpoint::SafetensorsLoader;

        let weights_path = weights_path.as_ref();
        let config_path = config_path.as_ref();
        tracing::info!(
            "Loading pipeline from safetensors: weights={}, config={}",
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

        // Build the pipeline from components
        Self::build_from_config_and_weights(config, loader, mode_idx)
    }

    /// Restore a pipeline from a directory containing safetensors files.
    ///
    /// Auto-detects file locations within the directory.
    ///
    /// # Arguments
    /// * `dir` - Path to the directory containing model files
    /// * `mode_idx` - Streaming mode index (0 = highest latency)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::asr::CacheAwareRNNTPipeline;
    /// let pipeline = CacheAwareRNNTPipeline::restore_from_directory("my_model/", 0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn restore_from_directory(dir: impl AsRef<Path>, mode_idx: usize) -> Result<Self> {
        use crate::core::checkpoint::SafetensorsLoader;

        let dir = dir.as_ref();
        tracing::info!("Loading pipeline from directory: {}", dir.display());

        let loader = SafetensorsLoader::from_directory(dir)?;
        let config: AsrModelConfig = loader.load_config()?;

        Self::build_from_config_and_weights(config, loader, mode_idx)
    }

    /// Internal helper to build pipeline from config and a SafetensorsLoader.
    fn build_from_config_and_weights(
        config: AsrModelConfig,
        loader: crate::core::checkpoint::SafetensorsLoader,
        mode_idx: usize,
    ) -> Result<Self> {
        config.validate()?;

        // Create preprocessor with dither disabled for inference
        let mut preproc_config: crate::asr::modules::MelSpectrogramConfig =
            config.preprocessor.clone().into();
        preproc_config.dither = 0.0;
        let preprocessor = MelSpectrogram::new(preproc_config.clone())?;

        // Create encoder
        let mut encoder = ConformerEncoder::new(config.encoder.clone())?;

        // Load tokenizer
        let tokenizer = loader.tokenizer_path().and_then(|path| {
            SentencePieceProcessor::open(path)
                .map_err(|e| tracing::warn!("Failed to load tokenizer: {}", e))
                .ok()
        });

        // Determine vocabulary size
        let vocab_size = tokenizer
            .as_ref()
            .map(|t| t.len())
            .unwrap_or(config.decoder.num_classes as usize);

        // Create decoder (prediction network)
        let is_lstm_decoder = config
            .decoder
            .prednet
            .as_ref()
            .map(|p| p.is_lstm_based())
            .unwrap_or(false);

        let mut decoder = if is_lstm_decoder {
            tracing::info!("Using LSTM-based prediction network");
            PredictionNetwork::Lstm(crate::asr::modules::RNNTDecoder::new(
                &config.decoder,
                vocab_size,
            )?)
        } else {
            tracing::info!("Using stateless prediction network");
            PredictionNetwork::Stateless(crate::asr::modules::StatelessTransducerDecoder::new(
                &config.decoder,
                vocab_size,
            )?)
        };

        // Create joint network
        let joint_config = config.joint.clone().unwrap_or_default();
        let encoder_hidden = encoder.output_dim();
        let pred_hidden = decoder.pred_hidden();
        let joint_hidden = joint_config.effective_joint_hidden();
        let activation = joint_config.effective_activation().to_string();

        let effective_joint_config = crate::asr::config::JointConfig {
            joint_hidden,
            activation,
            ..joint_config
        };

        let mut joint = crate::asr::modules::RNNTJoint::new(
            &effective_joint_config,
            encoder_hidden,
            pred_hidden,
            vocab_size + 1,
        )?;

        // Load weights (may be quantized)
        let weights = loader.load_weights()?;
        let quant = loader.quantization();
        tracing::debug!(
            "Loading {} weight tensors (format: {:?}, quantized: {})",
            weights.len(),
            loader.weight_format,
            quant.is_some()
        );
        encoder.load_weights(&weights, quant)?;
        decoder.load_weights(&weights, "decoder", quant)?;
        joint.load_weights(&weights, "joint", quant)?;

        // Set encoder's attention context for the selected streaming mode
        encoder.set_streaming_mode(mode_idx);

        // Compute streaming configuration
        let streaming_config = compute_streaming_config(encoder.config(), mode_idx, 10000)?;

        // Setup encoder for streaming
        encoder.setup_streaming(streaming_config.cache_drop_size);

        // Create encoder cache
        let encoder_cache = EncoderCache::new(
            encoder.num_layers(),
            1, // batch_size = 1 for single-stream
            streaming_config.last_channel_cache_size,
            streaming_config.last_time_cache_size,
            encoder.d_model(),
        )?;

        // Create mel streaming components
        let n_mels = preproc_config.features;
        let mel_stream_config = StreamingMelConfig::new(
            preproc_config.clone(),
            &streaming_config,
            encoder.subsampling_factor(),
        );
        let streaming_mel = StreamingMel::new(preproc_config)?;

        // Create streaming decoder
        let blank_id = vocab_size;
        let streaming_decoder = RnntStreamingDecoder::new(blank_id);

        // Find punctuation tokens from tokenizer
        let punctuation_token_ids = tokenizer
            .as_ref()
            .map(|t| find_punctuation_tokens(t))
            .unwrap_or_default();
        tracing::debug!(
            num_punctuation_tokens = punctuation_token_ids.len(),
            "identified punctuation tokens"
        );

        // Initialize decoder state
        let decoder_state = decoder.initialize_state(1)?;
        let state = StreamingState {
            decoder_state,
            last_token: None,
            accumulated_tokens: Vec::new(),
            is_first_chunk: true,
            step_count: 0,
            cumulative_stats: CumulativeStats::default(),
            detokenizer: IncrementalDetokenizer::default(),
        };

        let word_start = tokenizer.as_ref().map(word_start_table).unwrap_or_default();

        Ok(Self {
            config,
            preprocessor,
            encoder,
            decoder,
            joint,
            tokenizer,
            streaming_decoder,
            streaming_config,
            encoder_cache,
            mel_stream_config,
            streaming_mel,
            is_first_chunk: true,
            file_samples: None,
            file_position: 0,
            live_buffer: Vec::new(),
            stream_ended: false,
            is_done: false,
            mel_flushed: false,
            state,
            blank_id,
            n_mels,
            suppress_punctuation: false,
            punctuation_token_ids,
            word_start,
            blank_threshold: 0.0,
            silence_threshold: 0.0,
        })
    }

    /// Get the sample rate expected by this pipeline.
    pub fn sample_rate(&self) -> u32 {
        self.config.preprocessor.sample_rate
    }

    /// Get available streaming modes.
    pub fn streaming_modes(&self) -> Vec<StreamingModeInfo> {
        let frame_duration_ms = self.config.preprocessor.window_stride
            * 1000.0
            * self.encoder.subsampling_factor() as f32;
        get_streaming_modes(self.encoder.config(), frame_duration_ms)
    }

    /// Get the current streaming configuration.
    pub fn streaming_config(&self) -> &StreamingConfig {
        &self.streaming_config
    }

    /// Enable or disable punctuation suppression during decoding.
    ///
    /// When enabled, punctuation tokens are masked out (set to -inf) before
    /// argmax selection, preventing them from being generated.
    pub fn set_suppress_punctuation(&mut self, suppress: bool) {
        self.suppress_punctuation = suppress;
        if suppress {
            tracing::info!(
                num_masked_tokens = self.punctuation_token_ids.len(),
                "punctuation suppression enabled"
            );
        }
    }

    /// Check if punctuation suppression is enabled.
    pub fn suppress_punctuation(&self) -> bool {
        self.suppress_punctuation
    }

    /// Get the list of punctuation token IDs.
    pub fn punctuation_token_ids(&self) -> &[usize] {
        &self.punctuation_token_ids
    }

    /// Set the blank threshold for decoding.
    ///
    /// When > 0, blank is only emitted if blank_log_prob - best_non_blank_log_prob > threshold.
    /// This helps prevent the model from over-predicting blanks during streaming,
    /// which can cause sentences to be missed.
    ///
    /// Recommended values:
    /// - 0.0 = disabled (default argmax behavior)
    /// - 0.5 = blank must be ~1.6x more probable than best non-blank (mild)
    /// - 1.0 = blank must be ~2.7x more probable (moderate)
    /// - 2.0 = blank must be ~7.4x more probable (aggressive)
    pub fn set_blank_threshold(&mut self, threshold: f32) {
        self.blank_threshold = threshold;
        // Rebuild streaming decoder with new threshold
        self.streaming_decoder =
            RnntStreamingDecoder::new(self.blank_id).with_blank_threshold(threshold);
        if threshold > 0.0 {
            tracing::info!(
                threshold = %format!("{:.2}", threshold),
                "blank threshold enabled"
            );
        }
    }

    /// Get the current blank threshold.
    pub fn blank_threshold(&self) -> f32 {
        self.blank_threshold
    }

    /// Set silence detection threshold.
    ///
    /// When > 0, audio chunks with RMS energy below threshold are skipped.
    /// This prevents the model from processing silent frames that could
    /// trigger stuck behavior in the decoder.
    ///
    /// # Arguments
    /// * `threshold` - RMS energy threshold. 0.0 = disabled (default).
    ///
    /// # Recommended values
    /// - 0.0 = disabled (default)
    /// - 0.005-0.01 = quiet environments
    /// - 0.01-0.02 = typical environments
    /// - 0.02-0.05 = noisy environments
    pub fn set_silence_threshold(&mut self, threshold: f32) {
        self.silence_threshold = threshold;
        if threshold > 0.0 {
            tracing::info!(
                threshold = %format!("{:.4}", threshold),
                "silence detection enabled"
            );
        }
    }

    /// Get the current silence detection threshold.
    pub fn silence_threshold(&self) -> f32 {
        self.silence_threshold
    }

    /// Set audio from a file path.
    pub fn set_audio_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.reset();

        let (samples, file_sample_rate) = load_wav(path.as_ref())?;

        // Resample if needed
        let samples = if file_sample_rate != self.sample_rate() {
            tracing::info!(
                from = file_sample_rate,
                to = self.sample_rate(),
                "Resampling audio"
            );
            resample_audio(&samples, file_sample_rate, self.sample_rate())?
        } else {
            samples
        };

        let num_samples = samples.len();
        let duration_secs = num_samples as f32 / self.sample_rate() as f32;
        self.file_samples = Some(samples);
        self.file_position = 0;

        tracing::info!(
            num_samples,
            duration_secs,
            sample_rate = self.sample_rate(),
            "Loaded audio file"
        );

        Ok(())
    }

    /// Set audio samples directly.
    pub fn set_audio_samples(&mut self, samples: Vec<f32>) {
        self.reset();

        let num_samples = samples.len();
        let duration_secs = num_samples as f32 / self.sample_rate() as f32;

        tracing::info!(
            num_samples,
            duration_secs,
            sample_rate = self.sample_rate(),
            "Set audio samples directly"
        );

        self.file_samples = Some(samples);
        self.file_position = 0;
    }

    /// Add live audio samples.
    pub fn add_audio(&mut self, samples: &[f32]) {
        let buffer_before = self.live_buffer.len();
        self.live_buffer.extend_from_slice(samples);
        tracing::debug!(
            added_samples = samples.len(),
            buffer_before,
            buffer_after = self.live_buffer.len(),
            "add_audio: added samples to live buffer"
        );
    }

    /// Signal the end of the audio stream.
    pub fn end_stream(&mut self) {
        tracing::debug!(
            live_buffer_remaining = self.live_buffer.len(),
            mel_frames_remaining = self.streaming_mel.total_frames(),
            chunk_size = self.mel_stream_config.chunk_size,
            shift_size = self.mel_stream_config.shift_size,
            pre_encode_cache = self.mel_stream_config.pre_encode_cache,
            "Stream ended - remaining audio will be processed in subsequent transcribe_step() calls"
        );
        self.stream_ended = true;
    }

    /// Check if the pipeline has finished processing.
    pub fn is_done(&self) -> bool {
        self.is_done
    }

    /// Reset the pipeline for a new audio stream.
    pub fn reset(&mut self) {
        // Reset audio state
        self.file_samples = None;
        self.file_position = 0;
        self.live_buffer.clear();
        self.stream_ended = false;
        self.is_done = false;
        self.mel_flushed = false;

        // Reset mel streaming state
        self.streaming_mel.reset();
        self.is_first_chunk = true;

        // Reset encoder cache
        self.encoder_cache = EncoderCache::new(
            self.encoder.num_layers(),
            1,
            self.streaming_config.last_channel_cache_size,
            self.streaming_config.last_time_cache_size,
            self.encoder.d_model(),
        )
        .expect("Failed to create encoder cache");

        // Reset decoder state with zeros (matches NeMo's add_sos=False behavior)
        let decoder_state = self
            .decoder
            .initialize_state(1)
            .expect("Failed to initialize decoder state");
        self.state.decoder_state = decoder_state;
        self.state.last_token = None;
        self.state.accumulated_tokens.clear();
        self.state.is_first_chunk = true;
        self.state.step_count = 0;
        self.state.cumulative_stats = CumulativeStats::default();
        self.state.detokenizer.reset();
    }

    /// Get audio chunk size in samples for the current streaming config.
    fn audio_chunk_size(&self) -> usize {
        self.mel_stream_config
            .audio_chunk_size(self.preprocessor.hop_samples())
    }

    /// Process an audio chunk through the streaming mel processor.
    fn process_audio_chunk(&mut self, audio_chunk: &[f32]) -> Result<()> {
        if audio_chunk.is_empty() {
            tracing::trace!("process_audio_chunk: empty chunk, skipping");
            return Ok(());
        }

        let frames_before = self.streaming_mel.total_frames();
        let new_frames = self.streaming_mel.add_audio(audio_chunk)?;

        tracing::debug!(
            audio_chunk_len = audio_chunk.len(),
            frames_before,
            new_frames,
            total_frames = self.streaming_mel.total_frames(),
            "process_audio_chunk: complete"
        );

        Ok(())
    }

    /// Process pending audio from file or live buffer.
    fn process_pending_audio(&mut self) -> Result<()> {
        let chunk_audio_size = self.audio_chunk_size();

        tracing::trace!(
            chunk_audio_size,
            live_buffer_len = self.live_buffer.len(),
            stream_ended = self.stream_ended,
            has_file_samples = self.file_samples.is_some(),
            "process_pending_audio: starting"
        );

        // Process file audio
        if let Some(ref samples) = self.file_samples.clone() {
            while self.file_position < samples.len() {
                let end = (self.file_position + chunk_audio_size).min(samples.len());
                let audio_chunk = &samples[self.file_position..end];

                // Skip silent chunks
                if is_silent(audio_chunk, self.silence_threshold) {
                    tracing::trace!(
                        threshold = self.silence_threshold,
                        chunk_samples = audio_chunk.len(),
                        "Skipping silent audio chunk (file)"
                    );
                    self.file_position = end;

                    // Check if we've reached end of file
                    if end >= samples.len() && !self.stream_ended {
                        self.stream_ended = true;
                    }
                    continue;
                }

                // Process audio chunk
                self.process_audio_chunk(audio_chunk)?;
                self.file_position = end;

                // Check if we've reached end of file
                if end >= samples.len() && !self.stream_ended {
                    self.stream_ended = true;
                }

                // Only process one chunk at a time to allow interleaved extraction
                break;
            }
        }

        // Process live audio
        while self.live_buffer.len() >= chunk_audio_size {
            let audio_chunk: Vec<f32> = self.live_buffer.drain(..chunk_audio_size).collect();

            // Skip silent chunks
            if is_silent(&audio_chunk, self.silence_threshold) {
                tracing::trace!(
                    threshold = self.silence_threshold,
                    "Skipping silent audio chunk (live)"
                );
                continue;
            }

            // Process audio chunk
            self.process_audio_chunk(&audio_chunk)?;

            // Only process one chunk at a time
            break;
        }

        // Process remaining live audio when stream ended
        if self.stream_ended && !self.live_buffer.is_empty() {
            let remaining_samples = self.live_buffer.len();
            tracing::debug!(
                remaining_samples,
                "Processing remaining audio at stream end"
            );

            let remaining = std::mem::take(&mut self.live_buffer);
            self.process_audio_chunk(&remaining)?;

            tracing::debug!(
                total_frames = self.streaming_mel.total_frames(),
                "After processing remaining audio"
            );
        }

        // Flush final mel frames from audio lookback buffer
        if self.stream_ended && self.live_buffer.is_empty() && !self.mel_flushed {
            self.mel_flushed = true;
            let flushed = self.streaming_mel.flush()?;
            if flushed > 0 {
                tracing::debug!(
                    flushed_frames = flushed,
                    total_frames = self.streaming_mel.total_frames(),
                    "Flushed final mel frames from audio lookback"
                );
            }
        }

        Ok(())
    }

    /// Try to extract a mel chunk from the streaming mel buffer.
    ///
    /// Mirrors NeMo's `CacheAwareStreamingAudioBuffer`: the first chunk is
    /// `chunk_size_first` frames with no pre-encode cache; every later chunk is
    /// the `pre_encode_cache` frames before it plus `chunk_size` new frames.
    /// The last chunk is shorter rather than padded.
    fn try_extract_chunk(&mut self) -> Result<Option<FeatureBuffer>> {
        let is_first = self.is_first_chunk;
        let config = &self.mel_stream_config;
        let chunk_size = config.chunk_size_for_chunk(is_first);
        let pre_encode_cache = config.pre_encode_cache_for_chunk(is_first);
        let shift_size = config.shift_size_for_chunk(is_first);
        let next_pre_encode_cache = config.pre_encode_cache;
        let min_frames = config.min_frames_for_chunk(is_first);
        let min_frames_next = config.min_frames_for_chunk(false);

        // The buffer starts with this chunk's pre-encode cache
        let available = self.streaming_mel.total_frames();
        let required = pre_encode_cache + chunk_size;
        if available < required {
            if !self.stream_ended {
                return Ok(None);
            }
            // NeMo stops once a chunk has too few new frames for one encoder frame
            if available.saturating_sub(pre_encode_cache) < min_frames {
                self.is_done = true;
                return Ok(None);
            }
        }

        let total = available.min(required);
        let mel_chunk = self.streaming_mel.get_frames(0, total)?;
        let new_frames = total - pre_encode_cache;

        // Advance to the next chunk, keeping its pre-encode cache
        self.streaming_mel
            .trim(pre_encode_cache + shift_size - next_pre_encode_cache);
        self.is_first_chunk = false;

        let next_new_frames = self
            .streaming_mel
            .total_frames()
            .saturating_sub(next_pre_encode_cache);
        let is_last = self.stream_ended && next_new_frames < min_frames_next;
        if is_last {
            self.is_done = true;
        }

        tracing::debug!(
            is_first,
            is_last,
            total,
            pre_encode_cache,
            new_frames,
            "try_extract_chunk: extracted chunk"
        );

        Ok(Some(
            FeatureBuffer::new(mel_chunk, new_frames)
                .with_first(is_first)
                .with_last(is_last)
                .with_pre_encode_cache(pre_encode_cache),
        ))
    }

    /// Try to extract the next feature buffer.
    pub(super) fn try_extract_next(&mut self) -> Result<Option<FeatureBuffer>> {
        tracing::trace!(
            is_done = self.is_done,
            stream_ended = self.stream_ended,
            live_buffer_len = self.live_buffer.len(),
            total_frames = self.streaming_mel.total_frames(),
            "try_extract_next: starting"
        );

        // If we've already finished processing, don't try to extract more
        if self.is_done {
            tracing::trace!("try_extract_next: already done, returning None");
            return Ok(None);
        }

        // Process any pending audio from file or live buffer
        self.process_pending_audio()?;

        // Try to extract a chunk
        if let Some(buffer) = self.try_extract_chunk()? {
            tracing::debug!(
                is_first = buffer.is_first,
                is_last = buffer.is_last,
                valid_size = buffer.valid_size,
                pre_encode_cache = buffer.pre_encode_cache,
                "try_extract_next: returning chunk"
            );
            if buffer.is_last {
                self.is_done = true;
            }
            return Ok(Some(buffer));
        }

        // No chunk available
        let remaining = self.streaming_mel.total_frames();
        tracing::trace!(
            remaining,
            stream_ended = self.stream_ended,
            "try_extract_next: no chunk available"
        );
        if self.stream_ended && remaining == 0 {
            self.is_done = true;
        }

        Ok(None)
    }

    /// Process the next available request and return transcription output.
    ///
    /// Returns `None` if no data is available to process yet.
    pub fn transcribe_step(&mut self) -> Result<Option<TranscribeOutput>> {
        // Get next feature buffer
        let feature_buffer = match self.try_extract_next()? {
            Some(fb) => fb,
            None => return Ok(None),
        };

        // Increment and log step counter
        self.state.step_count += 1;
        tracing::info!(step = self.state.step_count, "transcribe_step");

        self.process_feature_buffer(feature_buffer)
    }

    /// Process a feature buffer and return transcription output.
    fn process_feature_buffer(&mut self, fb: FeatureBuffer) -> Result<Option<TranscribeOutput>> {
        let is_last = fb.is_last;

        // NeMo drops the extra pre-encoded frames on every chunk but the first
        let drop_extra = self.streaming_config.drop_extra_for_step(fb.is_first);

        // Add batch dimension: [n_mels, time] -> [1, n_mels, time]
        let chunk_len = fb.features.dim(1) as usize;
        let features = fb
            .features
            .reshape(&[1, self.n_mels as i32, chunk_len as i32])?;
        // Like NeMo, the length covers the pre-encode cache plus the new frames
        let lengths = Array::from_slice(&[chunk_len as i32], &[1]);

        let hop_length = self.preprocessor.hop_samples();
        let audio_samples_this_chunk = fb.valid_size * hop_length;

        tracing::debug!(
            step = self.state.step_count,
            is_first = fb.is_first,
            is_last,
            pre_encode_cache = fb.pre_encode_cache,
            chunk_len,
            drop_extra,
            "PIPELINE: received feature buffer"
        );

        // Run encoder with cache
        let (encoded, encoded_lengths, cache_channel_next, cache_time_next, cache_len_next) =
            self.encoder.forward_streaming(
                &features,
                &lengths,
                &self.encoder_cache.last_channel,
                &self.encoder_cache.last_time,
                &self.encoder_cache.last_channel_len,
                drop_extra,
            )?;

        self.encoder_cache.last_channel = cache_channel_next;
        self.encoder_cache.last_time = cache_time_next;
        self.encoder_cache.last_channel_len = cache_len_next;

        // Transpose encoder output from [B, D, T] to [B, T, D]
        let encoded = mlx_rs::ops::transpose_axes(&encoded, &[0, 2, 1])?;

        // chunked_limited keeps every output (NeMo's keep_all_outputs); regular
        // style clips to valid_out_len except on the last chunk
        let actual_tensor_frames = encoded.shape()[1] as usize;
        // A host array, so this doesn't wait on the GPU
        let encoded_lengths = encoded_lengths.to_vec_row_major::<i32>()?[0].max(0) as usize;
        let enc_seq_len = if self.encoder.is_chunked_limited() || is_last {
            encoded_lengths.min(actual_tensor_frames)
        } else {
            encoded_lengths
                .min(self.streaming_config.valid_out_len)
                .min(actual_tensor_frames)
        };

        // Decode with optional punctuation masking
        let masked_ids = if self.suppress_punctuation {
            Some(self.punctuation_token_ids.as_slice())
        } else {
            None
        };
        let decode_output = self.streaming_decoder.decode_frames(
            &encoded,
            enc_seq_len,
            &self.decoder,
            &mut self.state.decoder_state,
            &self.joint,
            &mut self.state.last_token,
            masked_ids,
        )?;

        // Update cumulative statistics
        let stats = &mut self.state.cumulative_stats;
        stats.total_audio_samples += audio_samples_this_chunk;
        stats.total_mel_frames += fb.valid_size;
        stats.total_encoder_frames += enc_seq_len;
        stats.total_blanks += decode_output.blank_count;
        stats.total_tokens += decode_output.new_tokens.len();

        tracing::debug!(
            step = self.state.step_count,
            frames_decoded = enc_seq_len,
            blanks = decode_output.blank_count,
            new_tokens = decode_output.new_tokens.len(),
            cum_audio_samples = stats.total_audio_samples,
            cum_encoder_frames = stats.total_encoder_frames,
            cum_tokens = stats.total_tokens,
            "PIPELINE: decode complete"
        );

        self.state
            .accumulated_tokens
            .extend(&decode_output.new_tokens);
        self.state.is_first_chunk = false;

        let text = match &self.tokenizer {
            Some(tokenizer) => self
                .state
                .detokenizer
                .update(tokenizer, &self.word_start, &self.state.accumulated_tokens),
            None => String::new(),
        };

        // Only clone accumulated_tokens on final output to avoid O(N²) memory usage
        let token_count = self.state.accumulated_tokens.len();
        let tokens = if is_last {
            self.state.accumulated_tokens.clone()
        } else {
            Vec::new()
        };

        Ok(Some(TranscribeOutput {
            text,
            new_tokens: decode_output.new_tokens,
            tokens,
            token_count,
            is_final: is_last,
        }))
    }

    /// Run the pipeline to completion on the loaded audio.
    ///
    /// Returns the final transcription text.
    pub fn run(&mut self) -> Result<String> {
        let mut final_text = String::new();

        while let Some(output) = self.transcribe_step()? {
            if output.is_final {
                final_text = output.text;
            } else if !output.text.is_empty() {
                final_text = output.text;
            }
        }

        Ok(final_text)
    }

    /// Get the blank token ID.
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Get a reference to the encoder.
    pub fn encoder(&self) -> &ConformerEncoder {
        &self.encoder
    }

    /// Get a reference to the preprocessor.
    pub fn preprocessor(&self) -> &MelSpectrogram {
        &self.preprocessor
    }

    /// Get a reference to the prediction network (decoder).
    pub fn prediction_network(&self) -> &PredictionNetwork {
        &self.decoder
    }

    /// Get a reference to the joint network.
    pub fn joint(&self) -> &crate::asr::modules::RNNTJoint {
        &self.joint
    }

    /// Get a reference to the streaming MEL processor.
    pub fn streaming_mel(&self) -> &StreamingMel {
        &self.streaming_mel
    }

    /// Set a spectrum transform hook to modify the power spectrum before mel filterbank.
    pub fn set_spectrum_transform(
        &mut self,
        transform: std::sync::Arc<dyn crate::asr::SpectrumTransform>,
    ) {
        self.streaming_mel.set_spectrum_transform(transform);
    }

    /// Process any pending audio in the live buffer through the streaming MEL
    /// processor, even if there isn't a full chunk yet.
    ///
    /// Normally `transcribe_step()` only feeds audio to the MEL processor in
    /// full `audio_chunk_size` batches. This method flushes whatever is
    /// buffered so that `streaming_mel().latest_frame_bins()` reflects the
    /// most recent audio — useful for real-time visualization without
    /// duplicating the MEL processor.
    pub fn flush_audio_to_mel(&mut self) -> Result<()> {
        if self.live_buffer.is_empty() {
            return Ok(());
        }
        let audio = std::mem::take(&mut self.live_buffer);
        self.process_audio_chunk(&audio)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcribe_output_default() {
        let output = TranscribeOutput::default();
        assert!(output.text.is_empty());
        assert!(!output.has_new_tokens());
        assert!(!output.is_final);
    }
}
