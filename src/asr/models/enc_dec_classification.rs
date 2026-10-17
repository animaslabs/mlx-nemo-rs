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
//! let mut model = EncDecClassificationModel::from_pretrained("vad_model.nemo")?;
//! let probs = model.infer_one("audio.wav")?;
//! // probs[i] is the speech probability of frame i
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use mlx_rs::ops;

use mlx_rs::Array;

use crate::asr::config::ClassificationModelConfig;
use crate::asr::modules::MelProcessor;
use crate::asr::modules::encoders::JasperEncoder;
use crate::asr::transcript::AudioInput;
use crate::core::ArrayExt;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::error::{Error, Result};
use crate::core::loading::{LoadOptions, LoadReport, Weights};
use crate::core::quantization::LinearLayer;
use crate::core::source::{LocalModel, ModelSource};

/// Clips run through the model at once by [`EncDecClassificationModel::infer`].
const INFER_BATCH_SIZE: usize = 4;

/// Encoder-Decoder Classification Model.
///
/// Consists of:
/// 1. Audio preprocessor (Mel spectrogram)
/// 2. Jasper/ConvASR encoder
/// 3. Linear decoder for frame-level classification
pub struct EncDecClassificationModel {
    config: ClassificationModelConfig,
    preprocessor: MelProcessor,
    encoder: JasperEncoder,
    decoder: LinearLayer,
    num_classes: usize,
}

impl std::fmt::Debug for EncDecClassificationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncDecClassificationModel")
            .field("sample_rate", &self.sample_rate())
            .field("num_classes", &self.num_classes)
            .finish_non_exhaustive()
    }
}

impl EncDecClassificationModel {
    internal_fn! {
        /// Create a new model from configuration.
        pub fn new(config: ClassificationModelConfig) -> Result<Self> {
            // Disable dithering for inference to ensure deterministic mel output
            let mut mel_config: crate::asr::modules::MelConfig = config.preprocessor.clone().into();
            mel_config.dither = 0.0;
            let preprocessor = MelProcessor::new(mel_config)?;
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
    }

    /// Load a model from a `.nemo` archive, local or on the Hugging Face Hub
    /// (see [`ModelSource`]).
    ///
    /// Fails if the checkpoint lacks any weight the model has;
    /// [`from_pretrained_with`](Self::from_pretrained_with) can allow that.
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
            LocalModel::Directory(dir) => Err(Error::not_implemented(format!(
                "loading a classification model from a model directory ({}); use a .nemo checkpoint",
                dir.display()
            ))),
        }
    }

    /// Load from a `.nemo` archive.
    fn from_nemo(path: &Path, options: LoadOptions) -> Result<Self> {
        tracing::info!("Loading classification model from: {}", path.display());

        let checkpoint = NemoCheckpoint::open(path)?;

        let config: ClassificationModelConfig = checkpoint.load_config()?;

        tracing::debug!(
            "Classification model config loaded, {} jasper blocks",
            config.encoder.jasper.len()
        );

        let mut model = Self::new(config)?;

        let weights = checkpoint.load_weights()?;
        tracing::debug!("Loading {} weight tensors", weights.len());
        model.load_weights_with(&weights, options)?;

        Ok(model)
    }

    internal_fn! {
        /// Load weights from a checkpoint's tensors, failing if any tensor the
        /// model has is missing.
        pub fn load_weights(
            &mut self,
            weights: &std::collections::HashMap<String, Array>,
        ) -> Result<()> {
            self.load_weights_with(weights, LoadOptions::default())
                .map(|_| ())
        }
    }

    internal_fn! {
        /// Load weights from a checkpoint's tensors with `options`, returning the
        /// tensors a non-strict load had to leave at their initial values.
        pub fn load_weights_with(
            &mut self,
            weights: &std::collections::HashMap<String, Array>,
            options: LoadOptions,
        ) -> Result<LoadReport> {
            let weights = Weights::new(weights, options);
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

            self.encoder.load_weights(&weights)?;

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
                    return Ok(weights.report());
                }
            }

            let keys = decoder_key_pairs.map(|(weight_key, _)| weight_key.to_string());
            weights.report_missing(&keys)?;
            Ok(weights.report())
        }
    }

    internal_fn! {
        /// Run forward pass on preprocessed audio.
        ///
        /// # Arguments
        /// * `audio` - Raw audio waveform [batch, samples]
        /// * `lengths` - Optional audio lengths `[batch]`
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
    }

    /// Per-frame speech probabilities for WAV files or in-memory audio (see
    /// [`AudioInput`]), one `Vec` per input. Audio is resampled to the
    /// model's rate, and each frame lasts
    /// [`frame_duration`](Self::frame_duration) seconds.
    ///
    /// ```no_run
    /// # fn main() -> mlx_voice::Result<()> {
    /// # use mlx_voice::asr::EncDecClassificationModel;
    /// # let samples = vec![0.0f32; 48_000];
    /// let mut model = EncDecClassificationModel::from_pretrained("vad_model.nemo")?;
    /// let from_files = model.infer(["a.wav", "b.wav"])?;
    /// let from_memory = model.infer_one((&samples[..], 48_000))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer<'a, I>(&mut self, inputs: I) -> Result<Vec<Vec<f32>>>
    where
        I: IntoIterator,
        I::Item: Into<AudioInput<'a>>,
    {
        let sample_rate = self.sample_rate();
        let inputs: Vec<AudioInput<'a>> = inputs.into_iter().map(Into::into).collect();
        let mut results = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(INFER_BATCH_SIZE) {
            let clips = batch
                .iter()
                .map(|input| input.load(sample_rate))
                .collect::<Result<Vec<_>>>()?;
            results.extend(self.infer_clips(&clips)?);
        }
        Ok(results)
    }

    /// Per-frame speech probabilities for one WAV file or in-memory clip.
    pub fn infer_one<'a>(&mut self, input: impl Into<AudioInput<'a>>) -> Result<Vec<f32>> {
        Ok(self.infer([input.into()])?.remove(0))
    }

    /// Speech probabilities of clips already at the model's sample rate.
    /// Empty clips give no frames without running the model.
    fn infer_clips(&mut self, clips: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let nonempty: Vec<usize> = (0..clips.len()).filter(|&i| !clips[i].is_empty()).collect();
        let mut results = vec![Vec::new(); clips.len()];
        if nonempty.is_empty() {
            return Ok(results);
        }
        let to_run: Vec<Vec<f32>> = nonempty.iter().map(|&i| clips[i].clone()).collect();
        let (audio, lengths) = super::pad_batch(&to_run);
        let (logits, lengths) = self.forward(&audio, Some(&lengths))?;

        let probs = mlx_rs::ops::softmax_axis(&logits, -1, None)?;
        let time_dim = probs.dim(1) as usize;
        let num_classes = self.num_classes;
        let lengths: Vec<i32> = lengths.to_vec_row_major()?;
        let probs: Vec<f32> = probs.to_vec_row_major()?;

        for (b, &i) in nonempty.iter().enumerate() {
            let valid_len = lengths[b].max(0) as usize;
            let offset = b * time_dim * num_classes;
            // Class 1 is speech
            results[i] = (0..valid_len)
                .map(|f| probs[offset + f * num_classes + 1])
                .collect();
        }
        Ok(results)
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
}
