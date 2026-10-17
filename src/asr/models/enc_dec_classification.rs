//! Encoder-Decoder Classification model.
//!
//! This model implements the classification architecture used by
//! NeMo's EncDecClassificationModel, e.g., MarbleNet for VAD.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! use mlx_voice::asr::EncDecClassificationModel;
//!
//! let mut model = EncDecClassificationModel::restore("vad_model.nemo")?;
//! let probs = model.infer(&["audio.wav"])?;
//! // probs[0] contains per-frame speech probabilities
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use mlx_rs::ops;

use mlx_rs::Array;

use crate::asr::config::ClassificationModelConfig;
use crate::asr::modules::MelSpectrogram;
use crate::asr::modules::encoders::JasperEncoder;
use crate::audio::resample;
use crate::core::ArrayExt;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::config::from_yaml_str;
use crate::core::error::{Error, Result};
use crate::core::quantization::LinearLayer;

/// Encoder-Decoder Classification Model.
///
/// Consists of:
/// 1. Audio preprocessor (Mel spectrogram)
/// 2. Jasper/ConvASR encoder
/// 3. Linear decoder for frame-level classification
pub struct EncDecClassificationModel {
    config: ClassificationModelConfig,
    preprocessor: MelSpectrogram,
    encoder: JasperEncoder,
    decoder: LinearLayer,
    num_classes: usize,
}

impl EncDecClassificationModel {
    /// Create a new model from configuration.
    pub fn new(config: ClassificationModelConfig) -> Result<Self> {
        // Disable dithering for inference to ensure deterministic mel output
        let mut mel_config: crate::asr::modules::MelConfig = config.preprocessor.clone().into();
        mel_config.dither = 0.0;
        let preprocessor = MelSpectrogram::new(mel_config)?;
        let encoder = JasperEncoder::new(&config.encoder)?;

        let feat_in = encoder.output_dim();
        let num_classes = config.decoder.num_classes;

        // Initialize decoder linear layer with zeros (will be loaded from weights)
        let weight = ops::zeros::<f32>(&[num_classes as i32, feat_in as i32])?;
        let bias = Some(ops::zeros::<f32>(&[num_classes as i32])?);
        let decoder = LinearLayer::standard(weight, bias);

        Ok(Self {
            config,
            preprocessor,
            encoder,
            decoder,
            num_classes,
        })
    }

    /// Restore a model from a .nemo checkpoint file.
    pub fn restore(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading classification model from: {}", path.display());

        let checkpoint = NemoCheckpoint::open(path).or_else(|_| NemoCheckpoint::open_gzip(path))?;

        let config: ClassificationModelConfig = from_yaml_str(&checkpoint.config_yaml)?;

        tracing::debug!(
            "Classification model config loaded, {} jasper blocks",
            config.encoder.jasper.len()
        );

        let mut model = Self::new(config)?;

        let weights = checkpoint.load_weights()?;
        tracing::debug!("Loading {} weight tensors", weights.len());
        model.load_weights(&weights)?;

        Ok(model)
    }

    /// Load weights from a HashMap.
    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Array>,
    ) -> Result<()> {
        // Load preprocessor filterbank from checkpoint if available.
        // This ensures exact match with the model's training filterbank.
        if let Some(fb) = weights.get("preprocessor.featurizer.fb") {
            // Checkpoint fb shape: [1, n_mels, n_fft/2+1], need [n_mels, n_fft/2+1]
            let fb = if fb.ndim() == 3 {
                fb.squeeze_axes(&[0]).unwrap_or_else(|_| fb.clone())
            } else {
                fb.clone()
            };
            tracing::debug!("Loaded preprocessor filterbank: {:?}", fb.shape());
            self.preprocessor.set_filterbank(fb);
        }

        self.encoder.load_weights(weights)?;

        // Try decoder weight paths in order of likelihood
        let decoder_key_pairs = [
            // ConvASRDecoderClassification (NeMo EncDecClassificationModel)
            (
                "decoder.decoder_layers.0.weight",
                "decoder.decoder_layers.0.bias",
            ),
            // MultiLayerPerceptron decoder
            ("decoder._layer0.weight", "decoder._layer0.bias"),
            // Simple linear decoder (Frame_VAD_Multilingual_MarbleNet)
            ("decoder.layer0.weight", "decoder.layer0.bias"),
        ];

        for (weight_key, bias_key) in &decoder_key_pairs {
            if let Some(weight) = weights.get(*weight_key) {
                // NeMo may store as Conv1d [out, in, 1], squeeze if needed
                let weight = if weight.ndim() == 3 {
                    weight
                        .squeeze_axes(&[-1])
                        .unwrap_or_else(|_| weight.clone())
                } else {
                    weight.clone()
                };
                let bias = weights.get(*bias_key).cloned();
                self.decoder = LinearLayer::standard(weight, bias);
                tracing::debug!("Loaded decoder from {}", weight_key);
                return Ok(());
            }
        }

        Err(Error::model(
            "Could not find decoder weights in checkpoint (tried decoder.decoder_layers.0.*, decoder._layer0.*, decoder.layer0.*)",
        ))
    }

    /// Run forward pass on preprocessed audio.
    ///
    /// # Arguments
    /// * `audio` - Raw audio waveform [batch, samples]
    /// * `lengths` - Optional audio lengths [batch]
    ///
    /// # Returns
    /// * Logits [batch, time, num_classes]
    pub fn forward(&mut self, audio: &Array, lengths: Option<&Array>) -> Result<(Array, Array)> {
        // Preprocess audio to mel spectrogram
        let (features, feature_lengths) = self.preprocessor.forward(audio, lengths)?;

        // MLX Conv1d expects (N, L, C_in), but preprocessor outputs (N, C, L)
        // Transpose to (N, L, C)
        let features = ops::transpose_axes(&features, &[0, 2, 1])?;

        // Encode
        let (encoded, encoded_lengths) = self.encoder.forward(&features, &feature_lengths)?;

        // Decoder: linear projection [B, T, D] -> [B, T, num_classes]
        let logits = self.decoder.forward(&encoded)?;

        Ok((logits, encoded_lengths))
    }

    /// Run inference on audio files.
    ///
    /// # Arguments
    /// * `paths` - Paths to audio files (WAV format)
    ///
    /// # Returns
    /// * Vector of per-frame speech probabilities for each file.
    ///   Each inner Vec<f32> contains the probability of class 1 (speech)
    ///   for each frame.
    pub fn infer(&mut self, paths: &[impl AsRef<Path>]) -> Result<Vec<Vec<f32>>> {
        let audio_batch = self.load_audio_batch(paths)?;
        let (logits, lengths) = self.forward(&audio_batch.0, Some(&audio_batch.1))?;

        let probs = mlx_rs::ops::softmax_axis(&logits, -1, None)?;
        probs.eval()?;
        lengths.eval()?;

        let batch_size = probs.dim(0) as usize;
        let time_dim = probs.dim(1) as usize;
        let num_classes = self.num_classes;
        let lengths_data: Vec<i32> = lengths.to_vec_row_major()?;
        let probs_data: Vec<f32> = probs.to_vec_row_major()?;

        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let valid_len = lengths_data[b] as usize;
            let batch_offset = b * time_dim * num_classes;
            let mut frame_probs = Vec::with_capacity(valid_len);
            for f in 0..valid_len {
                let idx = batch_offset + f * num_classes + 1;
                frame_probs.push(probs_data[idx]);
            }
            results.push(frame_probs);
        }

        Ok(results)
    }

    /// Run inference on raw audio samples.
    ///
    /// # Arguments
    /// * `samples` - Raw audio samples (f32, single channel, at model sample rate)
    ///
    /// # Returns
    /// * Per-frame speech probabilities (class 1).
    pub fn infer_samples(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let audio = Array::from_slice(samples, &[1, samples.len() as i32]);
        let lengths = Array::from_slice(&[samples.len() as i32], &[1]);

        let (logits, out_lengths) = self.forward(&audio, Some(&lengths))?;

        let probs = mlx_rs::ops::softmax_axis(&logits, -1, None)?;
        probs.eval()?;
        out_lengths.eval()?;

        let valid_len = out_lengths.to_vec_row_major::<i32>()?[0] as usize;
        let probs_data: Vec<f32> = probs.to_vec_row_major()?;
        let num_classes = self.num_classes;
        let mut result = Vec::with_capacity(valid_len);
        for f in 0..valid_len {
            let idx = f * num_classes + 1;
            result.push(probs_data[idx]);
        }
        Ok(result)
    }

    /// Get the frame duration in seconds, accounting for encoder stride.
    ///
    /// For MarbleNet with 10ms hop and stride-2 first block: 0.02s per frame.
    pub fn frame_duration(&self) -> f32 {
        self.config.preprocessor.window_stride * self.encoder.total_stride() as f32
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ClassificationModelConfig {
        &self.config
    }

    /// Get the sample rate expected by the model.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of output classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Load audio files and return batched tensors.
    fn load_audio_batch(&self, paths: &[impl AsRef<Path>]) -> Result<(Array, Array)> {
        let sample_rate = self.config.sample_rate;
        let mut audio_samples = Vec::new();
        let mut lengths = Vec::new();

        for path in paths {
            let path = path.as_ref();
            let (samples, file_sample_rate) = load_wav(path)?;

            let samples = if file_sample_rate != sample_rate {
                resample(&samples, file_sample_rate, sample_rate)?
            } else {
                samples
            };

            lengths.push(samples.len());
            audio_samples.push(samples);
        }

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

    Ok((samples, sample_rate))
}

#[cfg(not(feature = "asr"))]
fn load_wav(_path: &Path) -> Result<(Vec<f32>, u32)> {
    Err(Error::not_implemented(
        "Audio loading requires the 'asr' feature",
    ))
}
