//! Cache-aware RNN-T streaming pipeline.
//!
//! This module provides the main streaming pipeline for RNN-T models,
//! mirroring NeMo's `CacheAwareRNNTPipeline`.

use std::path::Path;

use mlx_rs::Array;
use sentencepiece::SentencePieceProcessor;

use crate::asr::config::AsrModelConfig;
use crate::asr::decoding::RnntStreamingDecoder;
use crate::asr::models::{DecoderState, PredictionNetwork};
use crate::asr::modules::{ConformerEncoder, MelSpectrogram, RNNTJoint};
use crate::asr::streaming::{
    compute_streaming_config, get_streaming_modes, EncoderCache, FeatureBuffer, RequestStreamer,
    StreamingConfig, StreamingModeInfo,
};
use crate::core::checkpoint::NemoCheckpoint;
use crate::core::error::Result;

/// Output from a transcription step.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOutput {
    /// Transcribed text (accumulated since stream start)
    pub text: String,
    /// New tokens emitted in this step
    pub new_tokens: Vec<usize>,
    /// All accumulated token IDs
    pub tokens: Vec<usize>,
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
    preprocessor: MelSpectrogram,
    encoder: ConformerEncoder,
    decoder: PredictionNetwork,
    joint: RNNTJoint,
    tokenizer: Option<SentencePieceProcessor>,

    // Streaming components
    streaming_decoder: RnntStreamingDecoder,
    streaming_config: StreamingConfig,
    encoder_cache: EncoderCache,
    streamer: RequestStreamer,

    // State
    state: StreamingState,
    blank_id: usize,
    n_mels: usize,
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

        let mut joint = RNNTJoint::new(
            &effective_joint_config,
            encoder_hidden,
            pred_hidden,
            vocab_size + 1,
        )?;

        // Load weights
        let weights = checkpoint.load_weights()?;
        // Note: MelSpectrogram doesn't need weight loading (uses fixed mel filterbanks)
        encoder.load_weights(&weights)?;
        decoder.load_weights(&weights, "decoder")?;
        joint.load_weights(&weights, "joint")?;

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

        // Create request streamer with dither disabled (same as preprocessor)
        let n_mels = preproc_config.features;
        let streamer = RequestStreamer::new(
            MelSpectrogram::new(preproc_config)?,
            streaming_config.clone(),
        );

        // Create streaming decoder
        let blank_id = vocab_size;
        let streaming_decoder = RnntStreamingDecoder::new(blank_id);

        // Initialize decoder state with zeros (matches NeMo's add_sos=False behavior)
        // NeMo's greedy decoder passes state=None on first step, which initializes to zeros
        let decoder_state = decoder.initialize_state(1)?;
        let state = StreamingState {
            decoder_state,
            last_token: None,
            accumulated_tokens: Vec::new(),
            is_first_chunk: true,
            step_count: 0,
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
            streamer,
            state,
            blank_id,
            n_mels,
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

    /// Set audio from a file path.
    pub fn set_audio_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.reset();
        self.streamer.set_audio_file(path)
    }

    /// Set audio samples directly.
    pub fn set_audio_samples(&mut self, samples: Vec<f32>) {
        self.reset();
        self.streamer.set_audio_samples(samples);
    }

    /// Add live audio samples.
    pub fn add_audio(&mut self, samples: &[f32]) {
        self.streamer.add_audio(samples);
    }

    /// Signal the end of the audio stream.
    pub fn end_stream(&mut self) {
        self.streamer.end_stream();
    }

    /// Check if the pipeline has finished processing.
    pub fn is_done(&self) -> bool {
        self.streamer.is_done()
    }

    /// Reset the pipeline for a new audio stream.
    pub fn reset(&mut self) {
        // Reset streamer
        self.streamer.reset();
        self.preprocessor.reset();

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
    }

    /// Process the next available request and return transcription output.
    ///
    /// Returns `None` if no data is available to process yet.
    pub fn transcribe_step(&mut self) -> Result<Option<TranscribeOutput>> {
        // Get next feature buffer from streamer
        let feature_buffer = match self.streamer.next() {
            Some(Ok(fb)) => fb,
            Some(Err(e)) => return Err(e),
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
            let features_contiguous = mlx_rs::ops::as_strided(&features, features.shape(), None, None)?;
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

        // Decode
        let decode_output = self.streaming_decoder.decode_frames(
            &encoded,
            enc_seq_len,
            &self.decoder,
            &mut self.state.decoder_state,
            &self.joint,
            &mut self.state.last_token,
        )?;

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

        Ok(Some(TranscribeOutput {
            text,
            new_tokens: decode_output.new_tokens,
            tokens: self.state.accumulated_tokens.clone(),
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
    pub fn joint(&self) -> &RNNTJoint {
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
