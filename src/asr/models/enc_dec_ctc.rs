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
//! ```rust,ignore
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
//!     while let Some(chunk) = audio_source.next_chunk() {
//!         let output = model.process_chunk(&mut session, &chunk)?;
//!         print!("{}", output.text[0]);
//!     }
//!
//!     // Finalize
//!     let final_output = model.finalize_streaming(session)?;
//!     println!("{}", final_output.text[0]);
//! }
//! ```

use std::path::Path;

use mlx_rs::Array;
use sentencepiece::SentencePieceProcessor;

use crate::asr::config::AsrModelConfig;
use crate::asr::decoding::{greedy_ctc_merge, greedy_decode, lcs_merge_buffer};
use crate::asr::modules::{CTCDecoder, ConformerEncoder, MelSpectrogram};
use crate::asr::streaming::{
    compute_streaming_config, get_streaming_modes, EncoderCache, MelBuffer, StreamingConfig,
    StreamingModeInfo, StreamingOutput,
};
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::config::from_yaml_str;
use crate::core::error::{Error, Result};

/// Debug output containing all intermediate tensors from forward pass.
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
/// ```rust,ignore
/// use mlx_nemo::asr::EncDecCTCModel;
///
/// // Load from .nemo checkpoint
/// let model = EncDecCTCModel::restore("model.nemo")?;
///
/// // Transcribe audio files
/// let texts = model.transcribe(&["audio.wav"])?;
/// ```
pub struct EncDecCTCModel {
    config: AsrModelConfig,
    preprocessor: MelSpectrogram,
    encoder: ConformerEncoder,
    decoder: CTCDecoder,
    tokenizer: Option<SentencePieceProcessor>,
    /// Blank token ID for CTC (typically last token in vocab)
    blank_id: usize,
}

impl EncDecCTCModel {
    /// Create a new model from configuration.
    pub fn new(config: AsrModelConfig) -> Result<Self> {
        let preprocessor = MelSpectrogram::new(config.preprocessor.clone())?;
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
        })
    }

    /// Restore a model from a .nemo checkpoint file.
    pub fn restore(path: impl AsRef<Path>) -> Result<Self> {
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

        // Load tokenizer if available
        if let Some(tokenizer_path) = checkpoint.find_tokenizer() {
            tracing::debug!("Loading tokenizer from: {}", tokenizer_path.display());
            model.load_tokenizer(&tokenizer_path)?;
        }

        // Load weights
        // Note: For now, we support safetensors format
        // PyTorch .ckpt support requires additional work
        if checkpoint.weights_path.is_some() {
            match checkpoint.load_weights() {
                Ok(weights) => {
                    tracing::debug!("Loading {} weight tensors", weights.len());
                    model.load_weights(&weights)?;
                }
                Err(e) => {
                    tracing::warn!("Could not load weights: {}", e);
                    // Continue without weights for now (useful for testing structure)
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
        let vocab_size = tokenizer.len() + 1; // +1 for blank token
        self.blank_id = vocab_size - 1;

        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Load weights from a HashMap.
    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<()> {
        self.encoder.load_weights(weights)?;
        self.decoder.load_weights(weights, "decoder")?;
        Ok(())
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

        // Run inference
        let (log_probs, lengths) = self.forward(&audio_batch.0, Some(&audio_batch.1))?;

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

    /// Run forward pass with debug output of intermediate values.
    ///
    /// Returns all intermediate tensors for debugging/verification.
    pub fn forward_debug(&mut self, audio: &Array, lengths: Option<&Array>) -> Result<DebugOutput> {
        // Preprocess audio to mel spectrogram (with pre-norm values)
        let (pre_norm, features, feature_lengths) =
            self.preprocessor.forward_debug(audio, lengths)?;
        pre_norm.eval()?;
        features.eval()?;
        feature_lengths.eval()?;

        // Encode
        let (encoded, encoded_lengths) = self.encoder.forward(&features, &feature_lengths)?;
        encoded.eval()?;
        encoded_lengths.eval()?;

        // Decode to log probabilities
        let log_probs = self.decoder.forward(&encoded)?;
        log_probs.eval()?;

        Ok(DebugOutput {
            mel_pre_norm: pre_norm,
            mel_spectrogram: features,
            mel_lengths: feature_lengths,
            encoder_output: encoded,
            encoder_lengths: encoded_lengths,
            log_probs,
        })
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
    /// * `mode_idx` - Streaming mode index (0 = default, typically lowest latency)
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

        // Create mel buffer
        let preprocessor_config = self.preprocessor.config();
        let mel_buffer = MelBuffer::new(
            preprocessor_config.sample_rate,
            preprocessor_config.window_size,
            preprocessor_config.window_stride,
            preprocessor_config.features,
        );

        Ok(CtcStreamingSession {
            config,
            encoder_cache,
            mel_buffer,
            prev_tokens: vec![Vec::new(); batch_size],
            is_first_chunk: true,
            blank_id: self.blank_id,
        })
    }

    /// Process an audio chunk and return partial transcription.
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

        // Add audio to buffer
        session.mel_buffer.add_audio(audio_chunk);

        // Check if we have enough audio for a chunk
        let chunk_frames = session.config.chunk_size_for_step(session.is_first_chunk);
        let shift_frames = session.config.shift_size_for_step(session.is_first_chunk);
        let hop_length = self.preprocessor.hop_length();
        let window_length = self.preprocessor.window_length();
        let chunk_samples = window_length + chunk_frames.saturating_sub(1) * hop_length;

        if session.mel_buffer.audio_buffer.len() < chunk_samples {
            // Not enough audio yet
            return Ok(StreamingOutput::default());
        }

        // Extract audio for this chunk
        let audio_data = session
            .mel_buffer
            .consume_frames(chunk_frames, shift_frames);

        // Convert to Array
        let audio = Array::from_slice(&audio_data, &[1, audio_data.len() as i32]);

        // Compute mel features (without normalization for streaming)
        let features = self.preprocessor.forward_unnormalized(&audio)?;

        // For streaming, we use buffer-based normalization
        let (mean, std) = session.mel_buffer.get_norm_stats();
        let mean_arr = Array::from_slice(&mean, &[session.mel_buffer.n_mels as i32]);
        let std_arr = Array::from_slice(&std, &[session.mel_buffer.n_mels as i32]);

        // Update running stats with new features
        features.eval()?;
        let features_flat: Vec<f32> = features.as_slice().to_vec();
        session.mel_buffer.update_stats(&features_flat);

        // Normalize with running stats (or skip if not enough data)
        let features = if session.mel_buffer.frames_processed > 10 {
            self.preprocessor
                .normalize_with_stats(&features, &mean_arr, &std_arr)?
        } else {
            features
        };

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
        println!("token_sequences: {:?}", token_sequences);

        // Merge with previous tokens using LCS
        let delay = session.config.lookahead_frames;
        let mut merged_tokens = Vec::with_capacity(batch_size);
        let mut texts = Vec::with_capacity(batch_size);

        for (i, new_tokens) in token_sequences.into_iter().enumerate() {
            let merged = lcs_merge_buffer(&session.prev_tokens[i], &new_tokens, delay, 5, 1);

            // Collapse CTC output
            let collapsed = greedy_ctc_merge(&merged, session.blank_id);

            // Decode to text
            let text = if let Some(tokenizer) = &self.tokenizer {
                let token_ids: Vec<u32> = collapsed.iter().map(|&t| t as u32).collect();
                tokenizer.decode_piece_ids(&token_ids).unwrap_or_default()
            } else {
                String::new()
            };

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
    /// Processes any remaining audio in the buffer and returns
    /// the complete transcription.
    pub fn finalize_streaming(
        &mut self,
        mut session: CtcStreamingSession,
    ) -> Result<StreamingOutput> {
        // Process any remaining audio in the buffer
        if !session.mel_buffer.audio_buffer.is_empty() {
            // Pad to minimum chunk size if needed
            let min_samples = self.preprocessor.window_length();
            if session.mel_buffer.audio_buffer.len() < min_samples {
                session.mel_buffer.audio_buffer.resize(min_samples, 0.0);
            }

            // Process final chunk
            let remaining: Vec<f32> = session.mel_buffer.audio_buffer.drain(..).collect();
            let _ = self.process_chunk(&mut session, &remaining)?;
        }

        // Final decode of accumulated tokens
        let batch_size = session.prev_tokens.len();
        let mut final_tokens = Vec::with_capacity(batch_size);
        let mut final_texts = Vec::with_capacity(batch_size);

        for tokens in &session.prev_tokens {
            let collapsed = greedy_ctc_merge(tokens, session.blank_id);

            let text = if let Some(tokenizer) = &self.tokenizer {
                let token_ids: Vec<u32> = collapsed.iter().map(|&t| t as u32).collect();
                tokenizer.decode_piece_ids(&token_ids).unwrap_or_default()
            } else {
                String::new()
            };

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
pub struct CtcStreamingSession {
    /// Streaming configuration (chunk sizes, cache sizes, etc.)
    pub config: StreamingConfig,

    /// Encoder cache state (attention + convolution)
    pub encoder_cache: EncoderCache,

    /// Audio/feature buffer for mel spectrogram computation
    pub mel_buffer: MelBuffer,

    /// Previous tokens for each batch sample (for LCS merge)
    pub prev_tokens: Vec<Vec<usize>>,

    /// Whether this is the first chunk (affects chunk size)
    pub is_first_chunk: bool,

    /// Blank token ID for CTC decoding
    blank_id: usize,
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
