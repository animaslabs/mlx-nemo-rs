//! SortFormer encoder-label diarization model.
//!
//! This implements NVIDIA's SortFormer diarization model which identifies
//! who is speaking when (up to 4 speakers). The model produces per-frame
//! speaker activity probabilities.
//!
//! # Architecture
//!
//! ```text
//! audio → MelProcessor → FastConformer → encoder_proj → TransformerEncoder
//!       → SortformerModules (speaker sigmoid head) → [B, T, num_spks]
//! ```
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # use mlx_voice::{asr::SortformerEncLabelModel, Array};
//! # let audio_array = Array::from_slice(&[0.0f32; 16000], &[1, 16000]);
//! # let lengths = Array::from_slice(&[16000i32], &[1]);
//! let mut model = SortformerEncLabelModel::from_pretrained("diar_model.nemo")?;
//! let preds = model.forward(&audio_array, &lengths)?;
//! // preds: [1, T, 4] - per-frame speaker probabilities
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;

use crate::asr::config::{SortformerConfig, SortformerModulesConfig};
use crate::asr::modules::encoders::transformer::TransformerEncoder;
use crate::asr::modules::sortformer::SortformerModules;
use crate::asr::modules::{ConformerEncoder, MelConfig, MelProcessor};
use crate::asr::streaming::StreamingSortformerState;
use crate::core::ArrayExt;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::error::{Error, Result};
use crate::core::loading::{LoadOptions, LoadReport, Weights};
use crate::core::source::{LocalModel, ModelSource};

/// SortFormer diarization model.
pub struct SortformerEncLabelModel {
    /// Audio preprocessor
    preprocessor: MelProcessor,
    /// FastConformer encoder
    encoder: ConformerEncoder,
    /// Transformer encoder for speaker modeling
    transformer_encoder: TransformerEncoder,
    /// Speaker head modules
    sortformer_modules: SortformerModules,
    /// Model configuration
    config: SortformerConfig,
}

impl std::fmt::Debug for SortformerEncLabelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SortformerEncLabelModel")
            .field("max_num_spks", &self.max_num_spks())
            .field("encoder_layers", &self.encoder.num_layers())
            .finish_non_exhaustive()
    }
}

impl SortformerEncLabelModel {
    /// Create a new SortFormer model from configuration.
    pub fn new(config: SortformerConfig) -> Result<Self> {
        // Create preprocessor from PreprocessorConfig → MelConfig
        let mut mel_config = MelConfig::from(config.preprocessor.clone());
        // NeMo only dithers during training
        mel_config.dither = 0.0;
        let preprocessor = MelProcessor::new(mel_config)?;

        // Create conformer encoder
        let encoder = ConformerEncoder::new(config.encoder.clone())?;

        // Create transformer encoder
        let transformer_encoder = TransformerEncoder::new(config.transformer_encoder.clone())?;

        // Create sortformer modules
        let sortformer_modules = SortformerModules::new(&config.sortformer_modules)?;

        Ok(Self {
            preprocessor,
            encoder,
            transformer_encoder,
            sortformer_modules,
            config,
        })
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
                "loading a Sortformer model from a model directory ({}); use a .nemo checkpoint",
                dir.display()
            ))),
        }
    }

    /// Load from a `.nemo` archive.
    fn from_nemo(path: &Path, options: LoadOptions) -> Result<Self> {
        tracing::info!(
            "Loading SortFormer diarization model from: {}",
            path.display()
        );

        // Open the checkpoint
        let checkpoint = NemoCheckpoint::open(path)?;

        // Parse the configuration
        let config: SortformerConfig = checkpoint.load_config()?;

        tracing::debug!(
            "SortFormer config: max_spks={}, encoder_layers={}, tf_layers={}",
            config.max_num_of_spks,
            config.encoder.n_layers,
            config.transformer_encoder.num_layers
        );

        // Create the model
        let mut model = Self::new(config)?;

        // Load weights
        let weights = checkpoint.load_weights()?;
        tracing::debug!("Loading {} weight tensors", weights.len());
        model.load_weights_with(&weights, options)?;

        Ok(model)
    }

    /// Load weights from a checkpoint's tensors, failing if any tensor the
    /// model has is missing.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<()> {
        self.load_weights_with(weights, LoadOptions::default())
            .map(|_| ())
    }

    /// Load weights from a checkpoint's tensors with `options`, returning the
    /// tensors a non-strict load had to leave at their initial values.
    pub fn load_weights_with(
        &mut self,
        weights: &HashMap<String, Array>,
        options: LoadOptions,
    ) -> Result<LoadReport> {
        let weights = Weights::new(weights, options);

        // Load conformer encoder weights
        self.encoder.load_weights(&weights, None)?;

        // Load transformer encoder weights
        self.transformer_encoder
            .load_weights(&weights, "transformer_encoder", None)?;

        // Load sortformer modules weights
        self.sortformer_modules
            .load_weights(&weights, "sortformer_modules", None)?;

        tracing::info!("SortFormer model weights loaded");
        Ok(weights.report())
    }

    /// Non-streaming forward pass.
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples [batch, samples]
    /// * `audio_lengths` - Number of valid samples per batch `[batch]`
    ///
    /// # Returns
    /// * Speaker activity probabilities [batch, time, num_spks]
    pub fn forward(&mut self, audio: &Array, audio_lengths: &Array) -> Result<Array> {
        // Preprocessor: audio → mel features [B, n_mels, T]
        let (mel, mel_lengths) = self.preprocessor.forward(audio, Some(audio_lengths))?;

        // Conformer encoder: mel → embeddings [B, D, T']
        let (enc_out, _enc_lengths) = self.encoder.forward(&mel, &mel_lengths)?;

        // Transpose to [B, T', D]
        let emb_seq = ops::transpose_axes(&enc_out, &[0, 2, 1])?;

        // Project from fc_d_model to tf_d_model
        let projected = self.sortformer_modules.project_encoder_output(&emb_seq)?;

        // Transformer encoder: [B, T', tf_d_model] → [B, T', tf_d_model]
        let hidden_out = self.transformer_encoder.forward(&projected, None)?;

        // Speaker sigmoids: [B, T', num_spks]
        self.sortformer_modules
            .forward_speaker_sigmoids(&hidden_out)
    }

    /// One streaming diarization step (port of NeMo's `forward_streaming_step`).
    ///
    /// The chunk is pre-encoded, concatenated after the speaker cache and FIFO
    /// (which hold pre-encode embeddings), and run through the full conformer
    /// and transformer. Only the central chunk frames are returned.
    ///
    /// # Arguments
    /// * `mel_chunk` - Mel features [1, n_mels, frames], including any left/right context
    /// * `mel_length` - Number of valid mel frames
    /// * `state` - Streaming state (updated in place)
    /// * `drop_extra_pre_encoded` - Leading pre-encoded frames to discard (to match a
    ///   cache-aware ASR chunk that includes its pre-encode cache)
    /// * `left_offset`, `right_offset` - Left/right context in mel frames
    ///
    /// # Returns
    /// * Speaker activity for the central chunk frames [1, chunk_frames, num_spks]
    pub(crate) fn forward_streaming_step(
        &mut self,
        mel_chunk: &Array,
        mel_length: usize,
        state: &mut StreamingSortformerState,
        drop_extra_pre_encoded: usize,
        left_offset: usize,
        right_offset: usize,
    ) -> Result<Array> {
        // Pre-encode (subsampling only): [1, n_mels, T] -> [1, T', d_model]
        let mel_btf = ops::transpose_axes(mel_chunk, &[0, 2, 1])?;
        let mel_lengths = Array::from_slice(&[mel_length as i32], &[1]);
        let (mut chunk_embs, chunk_lengths) = self
            .encoder
            .forward_subsampling_only(&mel_btf, &mel_lengths)?;
        let mut chunk_valid = chunk_lengths.to_vec_row_major::<i32>()?[0].max(0) as usize;

        if drop_extra_pre_encoded > 0 {
            chunk_embs = chunk_embs.index((.., drop_extra_pre_encoded as i32.., ..));
            chunk_valid = chunk_valid.saturating_sub(drop_extra_pre_encoded);
        }
        let chunk_valid = chunk_valid.min(chunk_embs.dim(1) as usize);
        tracing::trace!(
            mel_length,
            chunk_frames = chunk_embs.dim(1),
            chunk_valid,
            "SortFormer pre-encoded chunk"
        );

        // Conformer over [spkcache, fifo, chunk]
        let full_embs = state.concat_with_chunk(&chunk_embs)?;
        let total_len = full_embs.dim(1);
        let full_lengths = Array::from_slice(&[total_len], &[1]);
        let (enc_out, _) = self
            .encoder
            .forward_from_embeddings(&full_embs, &full_lengths)?;
        let emb_seq = ops::transpose_axes(&enc_out, &[0, 2, 1])?;

        // Projection, transformer and speaker head
        let projected = self.sortformer_modules.project_encoder_output(&emb_seq)?;
        let hidden_out = self.transformer_encoder.forward(&projected, None)?;
        let preds = self
            .sortformer_modules
            .forward_speaker_sigmoids(&hidden_out)?;

        // Zero predictions past the valid chunk frames
        let valid_len = (state.spkcache_len() + state.fifo_len() + chunk_valid) as i32;
        let preds = if valid_len < total_len {
            let positions = ops::arange::<_, i32>(0, total_len, 1)?.reshape(&[1, total_len, 1])?;
            let valid = ops::lt(&positions, Array::from_int(valid_len))?;
            ops::select(&valid, &preds, Array::from_f32(0.0))?
        } else {
            preds
        };

        let subsampling = self.subsampling_factor() as f32;
        let lc = (left_offset as f32 / subsampling).round() as usize;
        let rc = (right_offset as f32 / subsampling).ceil() as usize;
        state.update(&chunk_embs, &preds, lc, rc)
    }

    /// Diarize a whole recording with the streaming algorithm.
    ///
    /// Port of NeMo's `forward_streaming`: the mel features are split into
    /// `chunk_len` encoder-frame chunks with left/right context, and each is
    /// run through one streaming step with the speaker cache and FIFO.
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples [1, samples]
    /// * `audio_lengths` - Number of valid samples `[1]`
    ///
    /// # Returns
    /// * Speaker activity probabilities [1, frames, num_spks]
    pub fn diarize(&mut self, audio: &Array, audio_lengths: &Array) -> Result<Array> {
        let (mel, mel_lengths) = self.preprocessor.forward(audio, Some(audio_lengths))?;
        let feat_len = mel.dim(2) as usize;
        let valid_len = (mel_lengths.to_vec_row_major::<i32>()?[0].max(0) as usize).min(feat_len);

        let params = &self.config.sortformer_modules;
        let sub = self.subsampling_factor();
        let chunk_frames = params.chunk_len * sub;
        let (left_ctx, right_ctx) = (
            params.chunk_left_context * sub,
            params.chunk_right_context * sub,
        );

        let mut state = self.create_streaming_state()?;
        let mut chunk_preds = Vec::new();
        let mut stt = 0usize;
        while stt < feat_len {
            let left_offset = left_ctx.min(stt);
            let end = (stt + chunk_frames).min(feat_len);
            let right_offset = right_ctx.min(feat_len - end);

            let chunk = mel.index((
                ..,
                ..,
                (stt - left_offset) as i32..(end + right_offset) as i32,
            ));
            let chunk_valid = (valid_len + left_offset)
                .saturating_sub(stt)
                .min(chunk.dim(2) as usize);

            let preds = self.forward_streaming_step(
                &chunk,
                chunk_valid,
                &mut state,
                0,
                left_offset,
                right_offset,
            )?;
            preds.eval()?;
            chunk_preds.push(preds);
            stt = end;
        }

        let preds = ops::concatenate(&chunk_preds.iter().collect::<Vec<_>>(), 1)?;
        let output_frames = feat_len.div_ceil(sub) as i32;
        Ok(preds.index((.., ..output_frames.min(preds.dim(1)), ..)))
    }

    /// Create a new streaming state for this model.
    pub(crate) fn create_streaming_state(&self) -> Result<StreamingSortformerState> {
        StreamingSortformerState::new(
            self.config.sortformer_modules.clone(),
            self.encoder.d_model(),
        )
    }

    /// Override streaming parameters (speaker cache, FIFO, chunk geometry).
    ///
    /// Takes effect for states created afterwards.
    pub(crate) fn streaming_params_mut(&mut self) -> &mut SortformerModulesConfig {
        &mut self.config.sortformer_modules
    }

    /// Get the encoder's subsampling factor.
    pub fn subsampling_factor(&self) -> usize {
        self.encoder.subsampling_factor()
    }

    /// Get the number of speakers this model supports.
    pub fn max_num_spks(&self) -> usize {
        self.config.max_num_of_spks
    }

    /// Get the model configuration.
    pub fn config(&self) -> &SortformerConfig {
        &self.config
    }

    /// Get the encoder's output dimension.
    pub fn encoder_dim(&self) -> usize {
        self.encoder.output_dim()
    }
}
