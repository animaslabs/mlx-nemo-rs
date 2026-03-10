//! Encoder-Decoder RNN-T ASR model.
//!
//! This model implements the RNN-T (Transducer) ASR architecture used by
//! NeMo's EncDecRNNTModel and EncDecRNNTBPEModel (Parakeet RNNT models).

use std::path::Path;

use mlx_rs::Array;
use sentencepiece::SentencePieceProcessor;

use crate::asr::config::{AsrModelConfig, JointConfig};
use crate::asr::decoding::{rnnt_beam_decode, rnnt_greedy_decode};
use crate::asr::modules::{
    ConformerEncoder, MelSpectrogram, RNNTJoint, StatelessTransducerDecoder,
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

/// Encoder-Decoder RNN-T ASR Model.
///
/// This model consists of:
/// 1. Audio preprocessor (Mel spectrogram)
/// 2. Conformer/FastConformer encoder
/// 3. Stateless prediction network (decoder)
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
    decoder: StatelessTransducerDecoder,
    joint: RNNTJoint,
    tokenizer: Option<SentencePieceProcessor>,
    /// Blank token ID (typically vocab_size, last token)
    blank_id: usize,
    /// Decoding strategy
    decoding_strategy: RnntDecodingStrategy,
}

impl EncDecRNNTModel {
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

        // Create prediction network (decoder)
        let decoder = StatelessTransducerDecoder::new(&config.decoder, vocab_size)?;

        // Create joint network
        let joint_config = config.joint.clone().unwrap_or_else(|| JointConfig {
            joint_hidden: 640,
            activation: "relu".to_string(),
            ..Default::default()
        });

        let encoder_hidden = encoder.output_dim();
        let pred_hidden = decoder.pred_hidden();
        let joint = RNNTJoint::new(&joint_config, encoder_hidden, pred_hidden, vocab_size + 1)?;

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

        // Recreate decoder and joint with correct vocab size
        self.decoder = StatelessTransducerDecoder::new(&self.config.decoder, vocab_size)?;

        let joint_config = self.config.joint.clone().unwrap_or_else(|| JointConfig {
            joint_hidden: 640,
            activation: "relu".to_string(),
            ..Default::default()
        });
        let encoder_hidden = self.encoder.output_dim();
        let pred_hidden = self.decoder.pred_hidden();
        self.joint = RNNTJoint::new(&joint_config, encoder_hidden, pred_hidden, vocab_size + 1)?;

        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Load weights from a HashMap.
    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<()> {
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

        // Decode based on strategy
        let token_sequences = match self.decoding_strategy {
            RnntDecodingStrategy::Greedy => rnnt_greedy_decode(
                &encoder_output,
                &encoder_lengths,
                &self.decoder,
                &self.joint,
                self.blank_id,
                None,
            )?,
            RnntDecodingStrategy::BeamSearch { beam_size } => rnnt_beam_decode(
                &encoder_output,
                &encoder_lengths,
                &self.decoder,
                &self.joint,
                self.blank_id,
                beam_size,
                None,
            )?,
        };

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
    /// * Encoder output [batch, time, encoder_hidden]
    /// * Encoder output lengths [batch]
    pub fn forward(&mut self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Preprocess audio to mel spectrogram
        let (features, feature_lengths) = self.preprocessor.forward(audio, lengths)?;

        // Encode
        let (encoded, encoded_lengths) = self.encoder.forward(&features, &feature_lengths)?;

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

    /// Get a reference to the decoder (prediction network).
    pub fn decoder(&self) -> &StatelessTransducerDecoder {
        &self.decoder
    }

    /// Get a reference to the joint network.
    pub fn joint(&self) -> &RNNTJoint {
        &self.joint
    }
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
