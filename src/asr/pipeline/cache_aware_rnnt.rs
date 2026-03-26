//! Cache-aware RNN-T streaming pipeline.
//!
//! This module provides the main streaming pipeline for RNN-T models,
//! mirroring NeMo's `CacheAwareRNNTPipeline`.

use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;
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
use crate::asr::decoding::RnntStreamingDecoder;
use crate::asr::models::{DecoderState, PredictionNetwork};
use crate::asr::modules::{
    process_audio_streaming, ConformerEncoder, MelProcessor, MelStreamState, StreamingMelConfig,
};
use crate::asr::streaming::{
    compute_streaming_config, get_streaming_modes, EncoderCache, FeatureBuffer, StreamingConfig,
    StreamingModeInfo,
};
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
    /// Last extracted mel chunk's overlap region (for boundary verification)
    /// Stores the last `pre_encode_cache` frames of the previous extraction
    last_overlap_norms: Vec<f32>,
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
#[cfg(feature = "asr")]
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    tracing::debug!(from_rate, to_rate, "Resampling audio");

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
fn resample_audio(_samples: &[f32], _from_rate: u32, _to_rate: u32) -> Result<Vec<f32>> {
    Err(Error::not_implemented(
        "Audio resampling requires the 'asr' feature",
    ))
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
/// ```rust,ignore
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
/// ```
pub struct CacheAwareRNNTPipeline {
    // Model components
    config: AsrModelConfig,
    preprocessor: MelProcessor,
    encoder: ConformerEncoder,
    decoder: PredictionNetwork,
    joint: crate::asr::modules::RNNTJoint,
    tokenizer: Option<SentencePieceProcessor>,

    // Streaming components
    streaming_decoder: RnntStreamingDecoder,
    streaming_config: StreamingConfig,
    encoder_cache: EncoderCache,

    // Stateless mel streaming (replaces StreamingMelProcessor)
    mel_stream_config: StreamingMelConfig,
    mel_stream_state: MelStreamState,
    audio_buffer: Vec<f32>,
    mel_buffer: Option<Array>,
    mel_total_frames: usize,
    mel_buffer_idx: usize,
    mel_buffer_offset: usize,
    is_first_chunk: bool,
    mel_initialized: bool,

    // Audio input state
    file_samples: Option<Vec<f32>>,
    file_position: usize,
    live_buffer: Vec<f32>,
    stream_ended: bool,
    is_done: bool,

    // State
    state: StreamingState,
    blank_id: usize,
    n_mels: usize,

    // Punctuation suppression
    suppress_punctuation: bool,
    punctuation_token_ids: Vec<usize>,

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

        // Create stateless mel streaming components
        let n_mels = preproc_config.features;
        let mel_stream_config = StreamingMelConfig::new(preproc_config, &streaming_config);
        let mel_stream_state = MelStreamState::new(&preprocessor);

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
            last_overlap_norms: Vec::new(),
        };

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
            mel_stream_state,
            audio_buffer: Vec::new(),
            mel_buffer: None,
            mel_total_frames: 0,
            mel_buffer_idx: 0,
            mel_buffer_offset: 0,
            is_first_chunk: true,
            mel_initialized: false,
            file_samples: None,
            file_position: 0,
            live_buffer: Vec::new(),
            stream_ended: false,
            is_done: false,
            state,
            blank_id,
            n_mels,
            suppress_punctuation: false,
            punctuation_token_ids,
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
    /// ```rust,ignore
    /// let pipeline = CacheAwareRNNTPipeline::restore_safetensors(
    ///     "weights.safetensors",
    ///     "config.yaml",
    ///     Some("tokenizer.model"),
    ///     0,
    ///     Some(WeightFormat::Mlx),
    /// )?;
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
    /// ```rust,ignore
    /// let pipeline = CacheAwareRNNTPipeline::restore_from_directory("my_model/", 0)?;
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

        // Create stateless mel streaming components
        let n_mels = preproc_config.features;
        let mel_stream_config = StreamingMelConfig::new(preproc_config, &streaming_config);
        let mel_stream_state = MelStreamState::new(&preprocessor);

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
            last_overlap_norms: Vec::new(),
        };

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
            mel_stream_state,
            audio_buffer: Vec::new(),
            mel_buffer: None,
            mel_total_frames: 0,
            mel_buffer_idx: 0,
            mel_buffer_offset: 0,
            is_first_chunk: true,
            mel_initialized: false,
            file_samples: None,
            file_position: 0,
            live_buffer: Vec::new(),
            stream_ended: false,
            is_done: false,
            state,
            blank_id,
            n_mels,
            suppress_punctuation: false,
            punctuation_token_ids,
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
        let remaining_frames = self.mel_total_frames.saturating_sub(self.mel_buffer_idx);
        let overlap_samples = self.mel_stream_state.overlap.len();
        tracing::debug!(
            live_buffer_remaining = self.live_buffer.len(),
            overlap_samples,
            mel_frames_remaining = remaining_frames,
            mel_total_frames = self.mel_total_frames,
            mel_buffer_idx = self.mel_buffer_idx,
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

        // Reset stateless mel streaming state
        self.mel_stream_state = MelStreamState::new(&self.preprocessor);
        self.audio_buffer.clear();
        self.mel_buffer = None;
        self.mel_total_frames = 0;
        self.mel_buffer_idx = 0;
        self.mel_buffer_offset = 0;
        self.is_first_chunk = true;
        self.mel_initialized = false;

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
        self.state.last_overlap_norms.clear();
    }

    /// Get audio chunk size in samples for the current streaming config.
    fn audio_chunk_size(&self) -> usize {
        self.mel_stream_config
            .audio_chunk_size(self.preprocessor.hop_samples())
    }

    /// Initialize mel buffer with silence for pre_encode_cache.
    fn initialize_mel_with_silence(&mut self) -> Result<()> {
        // For the first chunk, use pre_encode_cache_first (which may be 0)
        let pre_encode_cache = self.mel_stream_config.pre_encode_cache_first;

        tracing::debug!(
            pre_encode_cache_first = pre_encode_cache,
            pre_encode_cache_subsequent = self.mel_stream_config.pre_encode_cache,
            chunk_size = self.mel_stream_config.chunk_size,
            shift_size = self.mel_stream_config.shift_size,
            n_mels = self.n_mels,
            "initialize_mel_with_silence: starting"
        );

        if pre_encode_cache == 0 {
            tracing::debug!("initialize_mel_with_silence: no pre_encode_cache needed for first chunk");
            return Ok(());
        }

        // LOG_MEL_ZERO = ln(2^-24) ≈ -16.6355 (matches NeMo's log_zero_guard)
        const LOG_MEL_ZERO: f32 = -16.635532;

        let silence = mlx_rs::ops::full::<f32>(
            &[self.n_mels as i32, pre_encode_cache as i32],
            &Array::from_f32(LOG_MEL_ZERO),
        )?;

        self.mel_buffer = Some(silence);
        self.mel_total_frames = pre_encode_cache;
        self.mel_buffer_idx = 0;
        self.mel_buffer_offset = 0;

        tracing::debug!(
            pre_encode_cache,
            mel_total_frames = self.mel_total_frames,
            mel_buffer_idx = self.mel_buffer_idx,
            "initialize_mel_with_silence: complete"
        );

        Ok(())
    }

    /// Add mel frames to the mel buffer.
    fn append_mel_frames(&mut self, mel: Array) -> Result<()> {
        let n_frames = mel.dim(1) as usize;
        if n_frames == 0 {
            tracing::trace!("append_mel_frames: no frames to append");
            return Ok(());
        }

        let mel_total_before = self.mel_total_frames;
        let mel_buffer_size_before = self.mel_buffer.as_ref().map(|b| b.dim(1) as usize).unwrap_or(0);

        // Python's approach: trim to expected length, take last chunk_size frames
        let feature_chunk_len = self.mel_stream_config.chunk_size;
        let expected_len = feature_chunk_len + 1;

        let mel_trimmed = if n_frames > expected_len {
            tracing::debug!(
                n_frames,
                expected_len,
                trimming_to = expected_len,
                "append_mel_frames: trimming excess frames"
            );
            mel.index((.., ..expected_len as i32))
        } else {
            mel
        };
        let trimmed_frames = mel_trimmed.dim(1) as usize;

        let mel_to_append = if trimmed_frames > feature_chunk_len {
            let start = (trimmed_frames - feature_chunk_len) as i32;
            tracing::debug!(
                trimmed_frames,
                feature_chunk_len,
                keeping_from = start,
                "append_mel_frames: taking last chunk_size frames"
            );
            mel_trimmed.index((.., start..))
        } else {
            mel_trimmed
        };

        let appended_frames = mel_to_append.dim(1) as usize;

        // Append to mel buffer
        self.mel_buffer = match &self.mel_buffer {
            Some(existing) => {
                Some(mlx_rs::ops::concatenate_axis(&[existing, &mel_to_append], 1)?)
            }
            None => Some(mel_to_append),
        };
        self.mel_total_frames += appended_frames;

        let mel_buffer_size_after = self.mel_buffer.as_ref().map(|b| b.dim(1) as usize).unwrap_or(0);

        tracing::debug!(
            mel_computed = n_frames,
            mel_appended = appended_frames,
            mel_total_before,
            mel_total_after = self.mel_total_frames,
            mel_buffer_size_before,
            mel_buffer_size_after,
            mel_buffer_idx = self.mel_buffer_idx,
            remaining_frames = self.remaining_frames(),
            "append_mel_frames: complete"
        );

        Ok(())
    }

    /// Process an audio chunk through the stateless mel streaming API.
    fn process_audio_chunk(&mut self, audio_chunk: &[f32]) -> Result<()> {
        if audio_chunk.is_empty() {
            tracing::trace!("process_audio_chunk: empty chunk, skipping");
            return Ok(());
        }

        let audio_len = audio_chunk.len();
        let overlap_before = self.mel_stream_state.overlap.len();

        // Add to audio buffer
        self.audio_buffer.extend_from_slice(audio_chunk);
        let buffer_len = self.audio_buffer.len();

        tracing::debug!(
            audio_chunk_len = audio_len,
            overlap_before,
            audio_buffer_len = buffer_len,
            total_available = buffer_len + overlap_before,
            "process_audio_chunk: starting"
        );

        // Process audio using stateless API
        let (mel_frames, new_state) = process_audio_streaming(
            &self.preprocessor,
            &self.audio_buffer,
            &self.mel_stream_state,
        )?;

        let overlap_after = new_state.overlap.len();
        let frames_produced = mel_frames.as_ref().map(|m| m.dim(1) as usize).unwrap_or(0);

        tracing::debug!(
            frames_produced,
            overlap_after,
            "process_audio_chunk: mel processing complete"
        );

        // Update state
        self.mel_stream_state = new_state;

        // Clear audio buffer since stateless API consumed what it needed
        // and the new state.overlap contains the overlap for next call
        self.audio_buffer.clear();

        // Append mel frames if any were produced
        if let Some(mel) = mel_frames {
            self.append_mel_frames(mel)?;
        }

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
            let overlap_samples = self.mel_stream_state.overlap.len();
            tracing::debug!(
                remaining_samples,
                overlap_samples,
                total = remaining_samples + overlap_samples,
                "Processing remaining audio at stream end"
            );

            let remaining = std::mem::take(&mut self.live_buffer);
            self.process_audio_chunk(&remaining)?;

            tracing::debug!(
                mel_total_frames = self.mel_total_frames,
                mel_buffer_idx = self.mel_buffer_idx,
                remaining_frames = self.remaining_frames(),
                "After processing remaining audio"
            );
        }

        Ok(())
    }

    /// Get remaining mel frames to be extracted.
    fn remaining_frames(&self) -> usize {
        self.mel_total_frames.saturating_sub(self.mel_buffer_idx)
    }

    /// Pad mel buffer with zeros.
    fn pad_mel_buffer(&mut self, frames: usize) -> Result<()> {
        if frames == 0 {
            tracing::trace!("pad_mel_buffer: no padding needed");
            return Ok(());
        }

        let mel_total_before = self.mel_total_frames;
        let mel_buffer_size_before = self.mel_buffer.as_ref().map(|b| b.dim(1) as usize).unwrap_or(0);

        let padding = Array::zeros::<f32>(&[self.n_mels as i32, frames as i32])?;

        self.mel_buffer = match &self.mel_buffer {
            Some(existing) => Some(mlx_rs::ops::concatenate_axis(&[existing, &padding], 1)?),
            None => Some(padding),
        };
        self.mel_total_frames += frames;

        let mel_buffer_size_after = self.mel_buffer.as_ref().map(|b| b.dim(1) as usize).unwrap_or(0);

        tracing::debug!(
            frames_padded = frames,
            mel_total_before,
            mel_total_after = self.mel_total_frames,
            mel_buffer_size_before,
            mel_buffer_size_after,
            mel_buffer_idx = self.mel_buffer_idx,
            remaining_frames = self.remaining_frames(),
            "pad_mel_buffer: complete"
        );

        Ok(())
    }

    /// Trim old mel frames that are no longer needed.
    fn trim_old_frames(&mut self, keep_frames: usize) -> Result<()> {
        let mel_buffer = match &self.mel_buffer {
            Some(buf) => buf,
            None => return Ok(()),
        };

        let buffer_size_before = mel_buffer.dim(1) as usize;
        let local_idx = self.mel_buffer_idx.saturating_sub(self.mel_buffer_offset);
        let can_trim = local_idx.saturating_sub(keep_frames);

        if can_trim > 0 {
            let trimmed = mel_buffer.index((.., can_trim as i32..));
            let buffer_size_after = trimmed.dim(1) as usize;
            self.mel_buffer = Some(trimmed);
            self.mel_buffer_offset += can_trim;

            tracing::trace!(
                keep_frames,
                local_idx,
                can_trim,
                buffer_size_before,
                buffer_size_after,
                new_offset = self.mel_buffer_offset,
                "trim_old_frames: trimmed buffer"
            );
        }

        Ok(())
    }

    /// Try to extract a mel chunk.
    fn try_extract_chunk(&mut self) -> Option<FeatureBuffer> {
        let chunk_size = self.mel_stream_config.chunk_size;
        let shift_size = self.mel_stream_config.shift_size;
        // Use different pre_encode_cache for first vs subsequent chunks
        let pre_encode_cache = self
            .mel_stream_config
            .pre_encode_cache_for_chunk(self.is_first_chunk);

        let remaining = self.remaining_frames();
        let required = chunk_size + pre_encode_cache;

        tracing::debug!(
            remaining,
            required,
            chunk_size,
            shift_size,
            pre_encode_cache,
            mel_total_frames = self.mel_total_frames,
            mel_buffer_idx = self.mel_buffer_idx,
            mel_buffer_offset = self.mel_buffer_offset,
            is_first_chunk = self.is_first_chunk,
            stream_ended = self.stream_ended,
            is_done = self.is_done,
            "try_extract_chunk: checking state"
        );

        // Not enough frames yet
        if remaining < required {
            if self.is_first_chunk && remaining > 0 {
                // Pad first chunk if needed
                let frames_to_pad = required - remaining;
                tracing::debug!(remaining, required, frames_to_pad, "Padding first chunk");

                if let Err(e) = self.pad_mel_buffer(frames_to_pad) {
                    tracing::error!("Failed to pad: {}", e);
                    return None;
                }
            } else if self.stream_ended && remaining > 0 {
                // Pad final chunk
                let frames_to_pad = required - remaining;
                tracing::debug!(remaining, required, frames_to_pad, "Padding final chunk");

                if let Err(e) = self.pad_mel_buffer(frames_to_pad) {
                    tracing::error!("Failed to pad: {}", e);
                    return None;
                }
            } else {
                // No frames or waiting for more
                tracing::debug!(
                    remaining,
                    stream_ended = self.stream_ended,
                    "try_extract_chunk: not enough frames, returning None"
                );
                if self.stream_ended && remaining == 0 {
                    self.is_done = true;
                }
                return None;
            }
        }

        // Extract chunk from mel buffer
        let mel_buffer = self.mel_buffer.as_ref()?;
        let buffer_size = mel_buffer.dim(1) as usize;

        let extract_start = self.mel_buffer_idx;
        let extract_end = self.mel_total_frames.min(extract_start + chunk_size + pre_encode_cache);
        let total_extract = extract_end - extract_start;

        // Convert to local indices
        let local_start = extract_start.saturating_sub(self.mel_buffer_offset);
        let local_end = local_start + total_extract;

        tracing::debug!(
            extract_start,
            extract_end,
            total_extract,
            mel_buffer_offset = self.mel_buffer_offset,
            local_start,
            local_end,
            buffer_size,
            "try_extract_chunk: extracting frames"
        );

        if local_end > buffer_size {
            tracing::error!(
                local_end,
                buffer_size,
                "try_extract_chunk: local_end exceeds buffer size!"
            );
        }

        let mel_chunk = mel_buffer.index((.., local_start as i32..local_end as i32));

        // Debug: Log mel frame values at chunk boundaries for continuity verification
        // and compare with previous chunk's overlap region
        {
            mel_chunk.eval().ok();
            let chunk_data = mel_chunk.as_slice::<f32>();
            let n_mels = self.n_mels;
            let n_frames = total_extract;

            // Compute norms for all frames in the overlap region (first pre_encode_cache frames)
            let overlap_norms: Vec<f32> = (0..pre_encode_cache)
                .map(|f| {
                    (0..n_mels)
                        .map(|m| {
                            let v = chunk_data[m * n_frames + f];
                            v * v
                        })
                        .sum::<f32>()
                        .sqrt()
                })
                .collect();

            // First frame of chunk (should match last frame of previous chunk's overlap region)
            let frame0_norm = overlap_norms.first().copied().unwrap_or(0.0);

            // Pre-encode cache boundary (frame at index pre_encode_cache-1, last context frame)
            let cache_boundary_norm = overlap_norms.last().copied().unwrap_or(0.0);

            // First "new" frame (frame at index pre_encode_cache)
            let first_new_norm: f32 = (0..n_mels)
                .map(|m| {
                    let v = chunk_data[m * n_frames + pre_encode_cache];
                    v * v
                })
                .sum::<f32>()
                .sqrt();

            // Last frame of chunk
            let last_frame_idx = n_frames - 1;
            let last_frame_norm: f32 = (0..n_mels)
                .map(|m| {
                    let v = chunk_data[m * n_frames + last_frame_idx];
                    v * v
                })
                .sum::<f32>()
                .sqrt();

            // Compute norms for the overlap region that will be used by the NEXT chunk
            // The next chunk will use pre_encode_cache (subsequent value) as its overlap
            let next_pre_encode_cache = self.mel_stream_config.pre_encode_cache;
            let next_overlap_norms: Vec<f32> = (0..next_pre_encode_cache)
                .map(|f| {
                    // Position in mel buffer that will be frame f of next chunk's overlap
                    // After this extraction, mel_buffer_idx will be at extract_start + shift_size
                    // The next chunk will extract starting from that position
                    // So the overlap frames are at indices shift_size + f within current chunk
                    let frame_idx = shift_size + f;
                    if frame_idx < n_frames {
                        (0..n_mels)
                            .map(|m| {
                                let v = chunk_data[m * n_frames + frame_idx];
                                v * v
                            })
                            .sum::<f32>()
                            .sqrt()
                    } else {
                        // Frame not available in this chunk (will be computed later)
                        // Mark as NaN to indicate invalid data
                        f32::NAN
                    }
                })
                .collect();

            // Compare current overlap with what we stored from previous chunk
            // Skip comparison if:
            // 1. No previous overlap data (first chunk)
            // 2. Previous overlap data contains NaN (frames weren't available)
            // 3. Overlap sizes don't match
            let has_valid_prev = !self.state.last_overlap_norms.is_empty()
                && self.state.last_overlap_norms.len() == overlap_norms.len()
                && !self.state.last_overlap_norms.iter().any(|v| v.is_nan());

            let overlap_match = if has_valid_prev {
                // Compare frame-by-frame
                let diffs: Vec<f32> = overlap_norms
                    .iter()
                    .zip(self.state.last_overlap_norms.iter())
                    .map(|(a, b)| (a - b).abs())
                    .collect();
                let max_diff = diffs.iter().cloned().fold(0.0f32, f32::max);
                let avg_diff = diffs.iter().sum::<f32>() / diffs.len().max(1) as f32;

                tracing::debug!(
                    extract_start,
                    overlap_max_diff = %format!("{:.6}", max_diff),
                    overlap_avg_diff = %format!("{:.6}", avg_diff),
                    prev_overlap_first = %format!("{:.4}", self.state.last_overlap_norms.first().copied().unwrap_or(0.0)),
                    curr_overlap_first = %format!("{:.4}", overlap_norms.first().copied().unwrap_or(0.0)),
                    prev_overlap_last = %format!("{:.4}", self.state.last_overlap_norms.last().copied().unwrap_or(0.0)),
                    curr_overlap_last = %format!("{:.4}", overlap_norms.last().copied().unwrap_or(0.0)),
                    "try_extract_chunk: OVERLAP CONTINUITY CHECK"
                );

                // Allow up to 5% relative difference due to numerical precision in FFT/mel computation
                // across chunk boundaries (STFT overlap handling may introduce small variations)
                let avg_norm = (overlap_norms.iter().sum::<f32>()
                    + self.state.last_overlap_norms.iter().sum::<f32>())
                    / (2.0 * overlap_norms.len() as f32);

                // For near-silence regions (avg_norm < 50), use absolute threshold instead
                // because relative differences become meaningless when signal is near zero
                if avg_norm < 50.0 {
                    max_diff < 10.0 // Absolute threshold for silence
                } else {
                    let relative_diff = max_diff / avg_norm;
                    relative_diff < 0.05 // 5% relative tolerance
                }
            } else {
                // Skip comparison - either first chunk or previous data was incomplete
                if !self.state.last_overlap_norms.is_empty() {
                    tracing::debug!(
                        extract_start,
                        prev_overlap_len = self.state.last_overlap_norms.len(),
                        curr_overlap_len = overlap_norms.len(),
                        prev_has_nan = self.state.last_overlap_norms.iter().any(|v| v.is_nan()),
                        "try_extract_chunk: skipping overlap comparison (incomplete data)"
                    );
                }
                true
            };

            if !overlap_match {
                // Downgrade to trace level since small discontinuities don't significantly impact
                // transcription quality - the encoder handles minor variations gracefully
                tracing::trace!(
                    extract_start,
                    "try_extract_chunk: overlap mismatch detected (minor discontinuity)"
                );
            }

            // Store the overlap region that will be used for the NEXT chunk's comparison
            self.state.last_overlap_norms = next_overlap_norms;

            tracing::debug!(
                extract_start,
                frame0_norm = %format!("{:.4}", frame0_norm),
                cache_boundary_norm = %format!("{:.4}", cache_boundary_norm),
                first_new_norm = %format!("{:.4}", first_new_norm),
                last_frame_norm = %format!("{:.4}", last_frame_norm),
                "try_extract_chunk: mel frame norms at boundaries"
            );
        }

        // Advance buffer index by shift_size
        let old_buffer_idx = self.mel_buffer_idx;
        if self.is_first_chunk {
            self.mel_buffer_idx += shift_size + pre_encode_cache;
        } else {
            self.mel_buffer_idx += shift_size;
        }

        // Trim old frames to bound memory
        // Keep enough frames for the NEXT chunk's overlap (which uses subsequent pre_encode_cache)
        let keep_for_next = self.mel_stream_config.pre_encode_cache;
        if let Err(e) = self.trim_old_frames(keep_for_next) {
            tracing::error!("Failed to trim: {}", e);
        }

        let is_first = self.is_first_chunk;
        self.is_first_chunk = false;

        let remaining_after = self.remaining_frames();
        let is_last = self.stream_ended && remaining_after < chunk_size;
        if is_last {
            self.is_done = true;
            tracing::info!(
                remaining_after,
                chunk_size,
                "Final chunk extracted - stream complete"
            );
        }

        tracing::debug!(
            is_first,
            is_last,
            valid_frames = total_extract,
            pre_encode_cache,
            mel_buffer_idx_before = old_buffer_idx,
            mel_buffer_idx_after = self.mel_buffer_idx,
            shift_size,
            remaining_after,
            mel_total_frames = self.mel_total_frames,
            "try_extract_chunk: extraction complete"
        );

        Some(
            FeatureBuffer::new(mel_chunk, total_extract)
                .with_first(is_first)
                .with_last(is_last)
                .with_pre_encode_cache(pre_encode_cache),
        )
    }

    /// Try to extract the next feature buffer.
    fn try_extract_next(&mut self) -> Result<Option<FeatureBuffer>> {
        tracing::trace!(
            is_done = self.is_done,
            mel_initialized = self.mel_initialized,
            stream_ended = self.stream_ended,
            live_buffer_len = self.live_buffer.len(),
            mel_total_frames = self.mel_total_frames,
            mel_buffer_idx = self.mel_buffer_idx,
            remaining_frames = self.remaining_frames(),
            "try_extract_next: starting"
        );

        // If we've already finished processing, don't try to extract more
        if self.is_done {
            tracing::trace!("try_extract_next: already done, returning None");
            return Ok(None);
        }

        // Initialize with silence if not done yet
        if !self.mel_initialized {
            self.initialize_mel_with_silence()?;
            self.mel_initialized = true;
        }

        // Process any pending audio from file or live buffer
        self.process_pending_audio()?;

        // Try to extract a chunk
        if let Some(buffer) = self.try_extract_chunk() {
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
        let remaining = self.remaining_frames();
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
        let is_first = fb.is_first || self.state.is_first_chunk;
        let is_last = fb.is_last;

        // Always apply drop_extra_pre_encoded in streaming mode.
        // NeMo applies this whenever cache_last_channel is not None, which in
        // streaming mode is always true (cache is pre-initialized to zeros).
        // This corrects for extra frames produced by strided subsampling at chunk boundaries.
        let drop_extra = self.streaming_config.drop_extra_pre_encoded;

        // Add batch dimension: [n_mels, time] -> [1, n_mels, time]
        let chunk_len = fb.features.dim(1) as usize;
        let valid_len = fb.valid_size;
        let features = fb
            .features
            .reshape(&[1, self.n_mels as i32, chunk_len as i32])?;
        // Python uses mel_len (valid_size) for the lengths parameter, not tensor dimension.
        // The tensor has chunk_len frames, but only valid_len are considered valid.
        // The encoder uses lengths for attention masking.
        let lengths = Array::from_slice(&[valid_len as i32], &[1]);

        // Calculate audio time position based on mel frames processed
        // Each mel frame = hop_length samples = 160 samples at 16kHz = 10ms
        let hop_length = 160; // TODO: get from config
        let sample_rate = self.sample_rate() as f32;
        let audio_samples_this_chunk = chunk_len * hop_length;
        let audio_time_start =
            self.state.cumulative_stats.total_audio_samples as f32 / sample_rate;
        let audio_time_end = (self.state.cumulative_stats.total_audio_samples
            + audio_samples_this_chunk) as f32
            / sample_rate;

        // Debug: Log feature buffer details with audio position
        tracing::debug!(
            step = self.state.step_count,
            is_first,
            is_last,
            pre_encode_cache = fb.pre_encode_cache,
            chunk_len,
            valid_len,
            audio_time_start = %format!("{:.3}s", audio_time_start),
            audio_time_end = %format!("{:.3}s", audio_time_end),
            "PIPELINE: received feature buffer"
        );

        // Log encoder cache state BEFORE processing
        {
            self.encoder_cache.last_channel_len.eval().ok();
            let cache_len_before = self.encoder_cache.last_channel_len.as_slice::<i64>()[0];

            // Compute cache channel norm (sample first layer)
            let cache_channel_norm: f32 = {
                self.encoder_cache.last_channel.eval().ok();
                let data = self.encoder_cache.last_channel.as_slice::<f32>();
                if !data.is_empty() {
                    // Just sample first 100 values for efficiency
                    let sample_len = data.len().min(100);
                    data[..sample_len].iter().map(|v| v * v).sum::<f32>().sqrt()
                } else {
                    0.0
                }
            };

            tracing::debug!(
                step = self.state.step_count,
                cache_len_before,
                cache_channel_norm = %format!("{:.6}", cache_channel_norm),
                "ENCODER CACHE: state before forward"
            );
        }

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

        // Log encoder cache state AFTER processing (before updating)
        {
            cache_len_next.eval().ok();
            let cache_len_after = cache_len_next.as_slice::<i64>()[0];

            // Compute new cache channel norm (sample first layer)
            let new_cache_channel_norm: f32 = {
                cache_channel_next.eval().ok();
                let data = cache_channel_next.as_slice::<f32>();
                if !data.is_empty() {
                    let sample_len = data.len().min(100);
                    data[..sample_len].iter().map(|v| v * v).sum::<f32>().sqrt()
                } else {
                    0.0
                }
            };

            tracing::debug!(
                step = self.state.step_count,
                cache_len_after,
                new_cache_channel_norm = %format!("{:.6}", new_cache_channel_norm),
                "ENCODER CACHE: state after forward"
            );
        }

        // Update encoder cache
        self.encoder_cache.last_channel = cache_channel_next;
        self.encoder_cache.last_time = cache_time_next;
        self.encoder_cache.last_channel_len = cache_len_next;

        // Transpose encoder output from [B, D, T] to [B, T, D]
        let encoded = mlx_rs::ops::transpose_axes(&encoded, &[0, 2, 1])?;
        encoded.eval()?;
        encoded_lengths.eval()?;

        // Get valid output length
        // For chunked_limited style, keep all outputs (matches NeMo's keep_all_outputs=True)
        // For regular style, clip to valid_out_len
        let raw_enc_len = encoded_lengths.as_slice::<i32>()[0] as usize;
        let actual_tensor_frames = encoded.shape()[1] as usize;
        let valid_out_len = self.streaming_config.valid_out_len;
        let enc_seq_len = if self.encoder.is_chunked_limited() {
            // chunked_limited with keep_all_outputs=True: use all encoder outputs
            raw_enc_len.min(actual_tensor_frames)
        } else {
            // regular style: clip to valid_out_len
            raw_enc_len.min(valid_out_len).min(actual_tensor_frames)
        };

        // Log mel input for comparison with Python
        // Tensor layout is [1, n_mels, time] in row-major order
        // To match Python's mel[0, :5, 0], we need indices at time=0 for mel bins 0-4
        // Index formula: mel[0, i, 0] = i * time_len + 0
        {
            // Force evaluation to ensure we have the actual computed values
            let features_contiguous =
                mlx_rs::ops::as_strided(&features, features.shape(), None, None)?;
            features_contiguous.eval()?;
            let mel_data = features_contiguous.as_slice::<f32>();
            let n_mels = self.n_mels;
            let time_len = chunk_len;

            // Get first 5 mel bins at time frame 0
            let mel_frame0_slice: Vec<f32> = (0..5.min(n_mels))
                .map(|i| {
                    let idx = i * time_len; // mel[0, i, 0]
                    (mel_data[idx] * 1_000_000.0).round() / 1_000_000.0
                })
                .collect();

            // L2 norm of all mel bins at time frame 0
            let mel_frame0_norm: f32 = (0..n_mels)
                .map(|i| {
                    let v = mel_data[i * time_len]; // mel[0, i, 0]
                    v * v
                })
                .sum::<f32>()
                .sqrt();

            tracing::trace!(
                ?mel_frame0_slice,
                mel_frame0_norm = %format!("{:.6}", mel_frame0_norm),
                "mel frame 0"
            );
        }

        // Debug encoder output
        self.encoder_cache.last_channel_len.eval()?;
        let cache_len = self.encoder_cache.last_channel_len.as_slice::<i64>()[0];

        // Detailed debug for alignment diagnosis
        tracing::debug!(
            step = self.state.step_count,
            mel_chunk_len = chunk_len,
            mel_valid_len = valid_len,
            drop_extra,
            enc_raw_len = raw_enc_len,
            enc_tensor_frames = actual_tensor_frames,
            enc_valid_out_config = valid_out_len,
            enc_seq_len_to_decode = enc_seq_len,
            enc_cache_len = cache_len,
            "PIPELINE: encoder output"
        );

        tracing::trace!(
            is_first,
            is_last,
            chunk_len,
            valid_len,
            drop_extra,
            raw_enc_len,
            actual_tensor_frames,
            valid_out_len,
            enc_seq_len,
            cache_len,
            "processing feature buffer"
        );

        // Log encoder frame values for comparison with Python
        if actual_tensor_frames > 0 {
            let enc_data = encoded.as_slice::<f32>();
            let d_model = encoded.shape()[2] as usize;

            // Frame 0
            let frame0_start = 0;
            let frame0_slice: Vec<f32> = enc_data[frame0_start..frame0_start + 10.min(d_model)]
                .iter()
                .map(|&v| (v * 1_000_000.0).round() / 1_000_000.0)
                .collect();
            let frame0_norm: f32 = enc_data[frame0_start..frame0_start + d_model]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            tracing::trace!(
                ?frame0_slice,
                frame0_norm = %format!("{:.6}", frame0_norm),
                "encoder frame 0"
            );

            // Frame 1 if available
            if actual_tensor_frames > 1 {
                let frame1_start = d_model;
                let frame1_slice: Vec<f32> = enc_data[frame1_start..frame1_start + 10.min(d_model)]
                    .iter()
                    .map(|&v| (v * 1_000_000.0).round() / 1_000_000.0)
                    .collect();
                let frame1_norm: f32 = enc_data[frame1_start..frame1_start + d_model]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();
                tracing::trace!(
                    ?frame1_slice,
                    frame1_norm = %format!("{:.6}", frame1_norm),
                    "encoder frame 1"
                );
            }
        }

        // Log decoder state before decoding
        {
            let decoder_state_norm = self.state.decoder_state.compute_norm();
            tracing::debug!(
                step = self.state.step_count,
                decoder_state_norm = %format!("{:.6}", decoder_state_norm),
                last_token = ?self.state.last_token,
                "DECODER STATE: before decoding"
            );
        }

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

        // Log decoder state after decoding
        {
            let decoder_state_norm = self.state.decoder_state.compute_norm();
            tracing::debug!(
                step = self.state.step_count,
                decoder_state_norm = %format!("{:.6}", decoder_state_norm),
                last_token = ?self.state.last_token,
                new_tokens = decode_output.new_tokens.len(),
                "DECODER STATE: after decoding"
            );
        }

        // Update cumulative statistics
        self.state.cumulative_stats.total_audio_samples += audio_samples_this_chunk;
        self.state.cumulative_stats.total_mel_frames += valid_len;
        self.state.cumulative_stats.total_encoder_frames += enc_seq_len;
        self.state.cumulative_stats.total_blanks += decode_output.blank_count;
        self.state.cumulative_stats.total_tokens += decode_output.new_tokens.len();

        // Detailed decode summary
        tracing::debug!(
            step = self.state.step_count,
            frames_decoded = enc_seq_len,
            blanks = decode_output.blank_count,
            new_tokens = decode_output.new_tokens.len(),
            total_tokens = self.state.accumulated_tokens.len() + decode_output.new_tokens.len(),
            last_token = ?self.state.last_token,
            "PIPELINE: decode complete"
        );

        // Cumulative statistics
        tracing::debug!(
            step = self.state.step_count,
            cum_audio_samples = self.state.cumulative_stats.total_audio_samples,
            cum_audio_time = %format!("{:.3}s", self.state.cumulative_stats.total_audio_samples as f32 / sample_rate),
            cum_mel_frames = self.state.cumulative_stats.total_mel_frames,
            cum_encoder_frames = self.state.cumulative_stats.total_encoder_frames,
            cum_blanks = self.state.cumulative_stats.total_blanks,
            cum_tokens = self.state.cumulative_stats.total_tokens,
            "PIPELINE: cumulative stats"
        );

        // Log when all frames produce blank (potential stuck state)
        if decode_output.new_tokens.is_empty() && decode_output.blank_count > 0 {
            tracing::trace!(
                step = self.state.step_count,
                last_token = ?self.state.last_token,
                blank_count = decode_output.blank_count,
                total_tokens = self.state.accumulated_tokens.len(),
                "all frames produced blank"
            );
        }

        // Accumulate tokens
        self.state
            .accumulated_tokens
            .extend(&decode_output.new_tokens);
        self.state.is_first_chunk = false;

        // Generate text
        let text = if let Some(tokenizer) = &self.tokenizer {
            let token_ids: Vec<u32> = self
                .state
                .accumulated_tokens
                .iter()
                .map(|&t| t as u32)
                .collect();
            tokenizer.decode_piece_ids(&token_ids).unwrap_or_default()
        } else {
            String::new()
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
