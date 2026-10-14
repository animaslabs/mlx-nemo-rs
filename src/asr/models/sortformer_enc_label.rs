//! SortFormer encoder-label diarization model.
//!
//! This implements NVIDIA's SortFormer diarization model which identifies
//! who is speaking when (up to 4 speakers). The model produces per-frame
//! speaker activity probabilities.
//!
//! # Architecture
//!
//! ```text
//! audio → MelSpectrogram → FastConformer → encoder_proj → TransformerEncoder
//!       → SortformerModules (speaker sigmoid head) → [B, T, num_spks]
//! ```
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> mlx_nemo::Result<()> {
//! # use mlx_nemo::{asr::SortformerEncLabelModel, Array};
//! # let audio_array = Array::from_slice(&[0.0f32; 16000], &[1, 16000]);
//! # let lengths = Array::from_slice(&[16000i32], &[1]);
//! let mut model = SortformerEncLabelModel::restore("diar_model.nemo")?;
//! let preds = model.forward(&audio_array, &lengths)?;
//! // preds: [1, T, 4] - per-frame speaker probabilities
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use crate::asr::config::SortformerConfig;
use crate::asr::modules::encoders::transformer::TransformerEncoder;
use crate::asr::modules::sortformer::SortformerModules;
use crate::asr::modules::{ConformerEncoder, MelConfig, MelProcessor};
use crate::asr::streaming::StreamingSortformerState;
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::config::from_yaml_str;
use crate::core::error::Result;

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

impl SortformerEncLabelModel {
    /// Create a new SortFormer model from configuration.
    pub fn new(config: SortformerConfig) -> Result<Self> {
        // Create preprocessor from PreprocessorConfig → MelConfig
        let mel_config = MelConfig::from(config.preprocessor.clone());
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

    /// Restore a model from a .nemo checkpoint file.
    pub fn restore(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        tracing::info!("Loading SortFormer diarization model from: {}", path.display());

        // Open the checkpoint
        let checkpoint = NemoCheckpoint::open(path).or_else(|_| {
            NemoCheckpoint::open_gzip(path)
        })?;

        // Parse the configuration
        let config: SortformerConfig = from_yaml_str(&checkpoint.config_yaml)?;

        tracing::debug!(
            "SortFormer config: max_spks={}, encoder_layers={}, tf_layers={}",
            config.max_num_of_spks,
            config.encoder.n_layers,
            config.transformer_encoder.num_layers
        );

        // Create the model
        let mut model = Self::new(config)?;

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

    /// Load weights from a state dict.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) -> Result<()> {
        // Load conformer encoder weights
        self.encoder.load_weights(weights, None)?;

        // Load transformer encoder weights
        self.transformer_encoder.load_weights(weights, "transformer_encoder", None)?;

        // Load sortformer modules weights
        self.sortformer_modules.load_weights(weights, "sortformer_modules", None)?;

        tracing::info!("SortFormer model weights loaded successfully");
        Ok(())
    }

    /// Non-streaming forward pass.
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples [batch, samples]
    /// * `audio_lengths` - Number of valid samples per batch [batch]
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
        self.sortformer_modules.forward_speaker_sigmoids(&hidden_out)
    }

    /// Streaming forward pass for a single chunk.
    ///
    /// # Arguments
    /// * `mel_chunk` - Mel features for this chunk [batch, n_mels, chunk_frames]
    /// * `mel_lengths` - Valid mel frame count [batch]
    /// * `state` - Streaming state (updated in place)
    ///
    /// # Returns
    /// * Chunk-level speaker activity [batch, chunk_time, num_spks]
    pub fn forward_streaming(
        &mut self,
        mel_chunk: &Array,
        mel_lengths: &Array,
        state: &mut StreamingSortformerState,
    ) -> Result<Array> {
        // Step 1: Pre-encode chunk (subsampling only)
        let mel_btf = ops::transpose_axes(mel_chunk, &[0, 2, 1])?;
        let (chunk_embs, _chunk_lengths) = self.encoder.forward_subsampling_only(&mel_btf, mel_lengths)?;

        let chunk_len = chunk_embs.dim(1) as usize;

        // Step 2: Concatenate [context, chunk_embs]
        let context = state.get_context()?;
        let context_len = state.context_len();

        let full_embs = if context_len > 0 {
            ops::concatenate_axis(&[&context, &chunk_embs], 1)?
        } else {
            chunk_embs.clone()
        };

        let total_len = full_embs.dim(1);
        let full_lengths = Array::from_slice(&[total_len], &[1]);

        // Step 3: Run conformer from embeddings (skip subsampling)
        let (enc_out, _) = self.encoder.forward_from_embeddings(&full_embs, &full_lengths)?;

        // Transpose to [B, T, D]
        let emb_seq = ops::transpose_axes(&enc_out, &[0, 2, 1])?;

        // Step 4: Project and run transformer
        let projected = self.sortformer_modules.project_encoder_output(&emb_seq)?;
        let hidden_out = self.transformer_encoder.forward(&projected, None)?;

        // Step 5: Speaker sigmoids for all frames
        let all_preds = self.sortformer_modules.forward_speaker_sigmoids(&hidden_out)?;

        // Step 6: Extract chunk predictions (last chunk_len frames)
        let start = (total_len - chunk_len as i32) as i32;
        let chunk_preds = all_preds.index((.., start.., ..));

        // Step 7: Extract chunk embeddings from the conformer output for state update
        let chunk_enc_embs = emb_seq.index((.., start.., ..));

        // Step 8: Update streaming state
        state.update(&chunk_enc_embs, &chunk_preds)?;

        Ok(chunk_preds)
    }

    /// Create a new streaming state for this model.
    pub fn create_streaming_state(&self, batch_size: usize) -> Result<StreamingSortformerState> {
        StreamingSortformerState::new(
            batch_size,
            self.config.sortformer_modules.fc_d_model,
            self.config.max_num_of_spks,
            self.config.spkcache_len,
            self.config.fifo_len,
        )
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
