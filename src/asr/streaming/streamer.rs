//! Request streamer for streaming ASR inference.
//!
//! This module provides a request generator that handles audio chunking,
//! mel computation, and buffer management for streaming pipelines.
//! It mirrors NeMo's `ContinuousBatchedRequestStreamer`.

use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;

use crate::asr::modules::MelSpectrogram;
use crate::asr::streaming::{FeatureBuffer, StreamingConfig, StreamingMelBuffer};
use crate::core::error::{Error, Result};

/// Check if audio chunk is silent based on RMS energy.
///
/// Returns `true` if the RMS energy is below the threshold.
/// Returns `false` if threshold is <= 0 (disabled) or audio is empty.
fn is_silent(audio: &[f32], threshold: f32) -> bool {
    if threshold <= 0.0 || audio.is_empty() {
        return false; // Disabled or no audio
    }
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    rms < threshold
}

/// Compute RMS energy of audio samples.
fn compute_rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt()
}

/// Request streamer for streaming ASR inference.
///
/// This streamer handles:
/// - Audio input from files or live streams
/// - Incremental mel spectrogram computation
/// - Chunk extraction with proper overlap (pre_encode_cache)
///
/// The streamer mirrors NeMo's `AudioBufferer` approach:
/// - Maintains a rolling audio buffer of size `pre_encode_cache + chunk`
/// - First chunk is padded with zeros (silence) for pre_encode_cache
/// - Subsequent chunks get overlap from the previous chunk
///
/// # Example
///
/// ```rust,ignore
/// let mut streamer = RequestStreamer::new(preprocessor, config);
///
/// // File mode
/// streamer.set_audio_file("audio.wav")?;
/// while let Some(request) = streamer.next() {
///     // process request
/// }
///
/// // Live mode
/// streamer.add_audio(&samples);
/// while let Some(request) = streamer.next() {
///     // process request
/// }
/// streamer.end_stream();
/// ```
pub struct RequestStreamer {
    /// Mel spectrogram preprocessor
    preprocessor: MelSpectrogram,
    /// Streaming configuration
    config: StreamingConfig,
    /// Mel buffer for chunk extraction
    mel_buffer: StreamingMelBuffer,
    /// Sample rate
    sample_rate: u32,

    // Audio source state
    /// Audio samples from file (pre-loaded)
    file_samples: Option<Vec<f32>>,
    /// Current position in file samples
    file_position: usize,
    /// Live audio buffer (accumulated samples)
    live_buffer: Vec<f32>,

    // Stream state
    /// Whether this is the first chunk
    is_first_chunk: bool,
    /// Whether end of stream has been signaled
    stream_ended: bool,
    /// Whether all data has been processed
    is_done: bool,
    /// Number of mel features
    n_mels: usize,

    // Audio overlap buffer for pre_encode_cache
    /// Audio samples to prepend to the next chunk (for overlap/pre_encode_cache)
    audio_overlap: Vec<f32>,

    // Progress tracking
    /// Counter for live audio chunks processed
    live_chunk_counter: usize,
    /// Total audio samples processed from live buffer
    live_samples_processed: usize,

    // Silence detection
    /// RMS energy threshold below which audio is considered silent.
    /// 0.0 = disabled (default), typical values: 0.005-0.02
    silence_threshold: f32,
}

impl RequestStreamer {
    /// Create a new request streamer.
    pub fn new(preprocessor: MelSpectrogram, config: StreamingConfig) -> Self {
        let preproc_config = preprocessor.config();
        let n_mels = preproc_config.features;
        let sample_rate = preproc_config.sample_rate;

        let mut mel_buffer = StreamingMelBuffer::new(
            sample_rate,
            preproc_config.window_size,
            preproc_config.window_stride,
            n_mels,
            preproc_config.n_fft,
        );

        // Note: We don't initialize mel buffer with silence here.
        // Python takes (chunk_size + pre_encode_cache) audio samples per chunk
        // and computes mel from all of it. The encoder's internal caching
        // handles the context between chunks.

        tracing::info!(
            sample_rate,
            n_mels,
            chunk_size = config.chunk_size.1,
            pre_encode_cache = config.pre_encode_cache_size.1,
            "Created RequestStreamer"
        );

        Self {
            preprocessor,
            config,
            mel_buffer,
            sample_rate,
            file_samples: None,
            file_position: 0,
            live_buffer: Vec::new(),
            is_first_chunk: true,
            stream_ended: false,
            is_done: false,
            n_mels,
            audio_overlap: Vec::new(),
            live_chunk_counter: 0,
            live_samples_processed: 0,
            silence_threshold: 0.0,
        }
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the streaming configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Set audio from a file path.
    ///
    /// Loads the entire file into memory and prepares for streaming.
    pub fn set_audio_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let (samples, file_sample_rate) = load_wav(path.as_ref())?;

        // Resample if needed
        let samples = if file_sample_rate != self.sample_rate {
            tracing::info!(
                from = file_sample_rate,
                to = self.sample_rate,
                "Resampling audio"
            );
            resample_audio(&samples, file_sample_rate, self.sample_rate)?
        } else {
            samples
        };

        let num_samples = samples.len();
        let duration_secs = num_samples as f32 / self.sample_rate as f32;
        self.file_samples = Some(samples);
        self.file_position = 0;
        self.reset_stream_state();

        tracing::info!(
            num_samples,
            duration_secs,
            sample_rate = self.sample_rate,
            "Loaded audio file"
        );
        Ok(())
    }

    /// Set audio samples directly (for pre-loaded audio).
    pub fn set_audio_samples(&mut self, samples: Vec<f32>) {
        let num_samples = samples.len();
        let duration_secs = num_samples as f32 / self.sample_rate as f32;

        tracing::info!(
            num_samples,
            duration_secs,
            sample_rate = self.sample_rate,
            "Set audio samples directly"
        );

        self.file_samples = Some(samples);
        self.file_position = 0;
        self.reset_stream_state();
    }

    /// Add live audio samples to the buffer.
    ///
    /// Use this for real-time streaming where audio arrives incrementally.
    pub fn add_audio(&mut self, samples: &[f32]) {
        tracing::trace!(
            added_samples = samples.len(),
            buffer_before = self.live_buffer.len(),
            buffer_after = self.live_buffer.len() + samples.len(),
            first_five = ?samples.iter().take(5).collect::<Vec<_>>(),
            "Adding live audio samples"
        );
        self.live_buffer.extend_from_slice(samples);
    }

    /// Signal the end of the audio stream.
    ///
    /// After calling this, the streamer will:
    /// 1. Flush remaining audio through the preprocessor
    /// 2. Process any remaining mel frames
    /// 3. Mark the final request with `is_last = true`
    pub fn end_stream(&mut self) {
        tracing::debug!(
            live_buffer_remaining = self.live_buffer.len(),
            mel_frames_remaining = self.mel_buffer.remaining_frames(),
            "Stream ended"
        );
        self.stream_ended = true;
    }

    /// Check if the streamer is done (all data processed).
    pub fn is_done(&self) -> bool {
        self.is_done
    }

    /// Check if there's more data to process.
    pub fn has_more(&self) -> bool {
        !self.is_done
    }

    /// Reset the stream state for a new audio stream.
    pub fn reset(&mut self) {
        tracing::debug!("Resetting RequestStreamer for new audio stream");
        self.file_samples = None;
        self.file_position = 0;
        self.live_buffer.clear();
        self.mel_buffer.reset();
        self.preprocessor.reset();
        self.audio_overlap.clear();
        self.reset_stream_state();
    }

    fn reset_stream_state(&mut self) {
        self.is_first_chunk = true;
        self.stream_ended = false;
        self.is_done = false;
        self.audio_overlap.clear();
        self.live_chunk_counter = 0;
        self.live_samples_processed = 0;
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
    }

    /// Get the current silence detection threshold.
    pub fn silence_threshold(&self) -> f32 {
        self.silence_threshold
    }

    /// Get audio chunk size in samples for the current streaming config.
    ///
    /// This calculates how many audio samples are consumed per chunk.
    /// This matches Python NeMo's approach:
    ///   buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    ///   audio_chunk_samples = buffer_mel_frames * hop_length
    ///
    /// The audio advances by this full amount each step - overlap is handled
    /// by the encoder's internal caching, not by audio-level overlap.
    pub fn audio_chunk_size(&self) -> usize {
        // Include pre_encode_cache to match Python's audio chunk calculation
        // Python: buffer_mel_frames = model_chunk_size + pre_encode_cache_size
        // let chunk_mel_frames = self.config.chunk_size.1 + self.config.pre_encode_cache_size.1;
        let chunk_mel_frames = self.config.chunk_size.1;
        let hop_samples = self.mel_buffer.hop_samples;
        chunk_mel_frames * hop_samples
    }

    /// Get the full audio buffer size in samples (chunk + pre_encode_cache).
    ///
    /// This is the size of the buffer passed to mel computation.
    fn audio_buffer_size(&self) -> usize {
        let buffer_mel_frames = self.config.chunk_size.1 + self.config.pre_encode_cache_size.1;
        let hop_samples = self.mel_buffer.hop_samples;
        buffer_mel_frames * hop_samples
    }

    /// Get the pre-encode cache size in audio samples.
    ///
    /// This is the number of silence/overlap samples to prepend to each audio chunk
    /// to match Python NeMo's AudioBufferer behavior.
    fn pre_encode_cache_samples(&self) -> usize {
        let pre_encode_cache_frames = self.config.pre_encode_cache_size.1;
        let hop_samples = self.mel_buffer.hop_samples;
        pre_encode_cache_frames * hop_samples
    }

    /// Process available audio and try to extract the next request.
    fn try_extract_next(&mut self) -> Result<Option<FeatureBuffer>> {
        // Calculate audio position info for debugging
        let audio_position_samples = if self.file_samples.is_some() {
            self.file_position
        } else {
            self.live_samples_processed
        };
        let audio_position_secs = audio_position_samples as f32 / self.sample_rate as f32;

        tracing::trace!(
            is_first_chunk = self.is_first_chunk,
            stream_ended = self.stream_ended,
            mel_frames_available = self.mel_buffer.remaining_frames(),
            audio_position_samples,
            audio_position_secs = %format!("{:.3}s", audio_position_secs),
            "try_extract_next called"
        );

        // Step 1: Initialize mel buffer with silence for pre_encode_cache (first time only)
        // This matches Python's BatchedCacheFeatureBufferer which initializes
        // the feature_buffer to LOG_MEL_ZERO for the first pre_encode_cache frames.
        if self.is_first_chunk && self.mel_buffer.total_frames == 0 {
            let pre_encode_cache = self.config.pre_encode_cache_size.1;
            if pre_encode_cache > 0 {
                self.mel_buffer.initialize_with_silence(pre_encode_cache)?;
                tracing::debug!(
                    pre_encode_cache,
                    "Initialized mel buffer with silence for pre_encode_cache"
                );
            }
        }

        // Step 2: Process any pending audio into mel frames
        self.process_pending_audio()?;

        // Step 3: Try to extract a mel chunk
        // Python's feature buffer works like this:
        // - Buffer has 121 frames: [9 pre_encode_cache] + [112 chunk]
        // - Each step: shift left by 112, keep 9 old frames, append 112 new
        // - The 9 old frames are PRESERVED from previous step (not recomputed)
        let chunk_size = self.config.chunk_size.1; // 112
        let shift_size = self.config.shift_size.1; // 112
        let pre_encode_cache = self.config.pre_encode_cache_size.1; // 9

        // Check if we have enough frames to extract a full chunk.
        // We need chunk_size + pre_encode_cache total frames for proper extraction.
        let remaining = self.mel_buffer.remaining_frames();
        let required_frames = chunk_size + pre_encode_cache;

        if remaining < required_frames {
            if self.is_first_chunk {
                // For the first chunk, pad with zeros to reach required_frames.
                // The silence initialization provides pre_encode_cache frames,
                // but we may not have enough audio frames yet.
                let frames_to_pad = required_frames - remaining;
                tracing::debug!(
                    remaining,
                    required_frames,
                    frames_to_pad,
                    "Padding first chunk with zeros"
                );
                let padding =
                    mlx_rs::ops::zeros::<f32>(&[self.n_mels as i32, frames_to_pad as i32])?;
                self.mel_buffer.append_mel(&padding)?;
            } else if self.stream_ended {
                // Handle final partial chunk
                return self.handle_final_chunk();
            } else {
                // Not enough frames yet, wait for more audio
                return Ok(None);
            }
        }

        // Extract mel chunk: pre_encode_cache + chunk_size frames, shift by shift_size
        // This leaves pre_encode_cache frames of overlap for the next chunk
        let (mel_chunk, _total_frames, valid_frames) = self
            .mel_buffer
            .extract_chunk(chunk_size, shift_size, pre_encode_cache)
            .ok_or_else(|| Error::model("Failed to extract mel chunk"))?;

        // Trim old mel frames to bound memory usage.
        // Keep enough frames for the pre_encode_cache overlap needed for next chunk.
        self.mel_buffer.trim_old_frames(pre_encode_cache)?;

        // Calculate expected encoder output frames
        // subsampling_factor = 8, so encoder_frames = ceil(mel_frames / 8)
        let subsampling_factor = 8;
        let expected_encoder_frames = (valid_frames + subsampling_factor - 1) / subsampling_factor;

        tracing::debug!(
            chunk_size,
            shift_size,
            pre_encode_cache,
            mel_valid_frames = valid_frames,
            expected_encoder_frames,
            buffer_idx = self.mel_buffer.buffer_idx,
            total_frames = self.mel_buffer.total_frames,
            remaining_after = self.mel_buffer.remaining_frames(),
            is_first = self.is_first_chunk,
            "STREAMER: extracted mel chunk"
        );

        // DO NOT reset mel buffer - we need to preserve overlap frames for pre_encode_cache!
        // Python's BatchedCacheFeatureBufferer keeps the last pre_encode_cache frames
        // by shifting the buffer left, not by resetting it.
        // NOTE: trim_old_frames above handles memory cleanup while preserving the overlap.

        let is_first = self.is_first_chunk;
        self.is_first_chunk = false;

        // Check if this is the last chunk
        let is_last = self.stream_ended && self.mel_buffer.remaining_frames() < chunk_size;
        if is_last {
            tracing::debug!("This is the final chunk");
            self.is_done = true;
        }

        tracing::trace!(
            is_first,
            is_last,
            valid_frames,
            pre_encode_cache,
            "Returning FeatureBuffer"
        );

        Ok(Some(
            FeatureBuffer::new(mel_chunk, valid_frames)
                .with_first(is_first)
                .with_last(is_last)
                .with_pre_encode_cache(pre_encode_cache),
        ))
    }

    /// Get the number of lookback samples for mel computation continuity.
    ///
    /// Python uses 1 hop (160 samples) of lookback for mel continuity.
    /// This ensures proper windowing at chunk boundaries.
    fn mel_lookback_samples(&self) -> usize {
        // Python: n_chunk_look_back = int(timestep_duration * sample_rate) = 160
        // JJK save multiple hops because if the window size is 512, hop is 160, lookback is 352
        // self.mel_buffer.window_samples - self.mel_buffer.hop_samples
        self.mel_buffer.hop_samples
    }

    /// Build the audio buffer for mel computation.
    ///
    /// This matches Python NeMo's BatchedCacheFeatureBufferer approach:
    /// - Takes chunk_size + 1 hop of lookback for mel continuity
    /// - Feature-level caching is handled by the mel buffer, not audio overlap
    fn build_audio_buffer(&mut self, audio_chunk: &[f32]) -> Vec<f32> {
        let lookback_samples = self.mel_lookback_samples();

        // For first chunk, prepend zeros for lookback
        // For subsequent chunks, prepend last lookback_samples from previous chunk
        if self.audio_overlap.is_empty() {
            self.audio_overlap = vec![0.0f32; lookback_samples];
        }

        // Prepend lookback, then append current chunk
        let mut buffer = Vec::with_capacity(lookback_samples + audio_chunk.len());
        buffer.extend_from_slice(&self.audio_overlap);
        buffer.extend_from_slice(audio_chunk);

        // Save last lookback_samples from audio_chunk for next iteration
        if audio_chunk.len() >= lookback_samples {
            let start_idx = audio_chunk.len() - lookback_samples;
            self.audio_overlap = audio_chunk[start_idx..].to_vec();
        } else {
            // Chunk is shorter than lookback_samples (final chunk case)
            let shift = audio_chunk.len();
            self.audio_overlap.drain(..shift);
            self.audio_overlap.extend_from_slice(audio_chunk);
        }

        tracing::trace!(
            lookback_samples,
            audio_chunk_len = audio_chunk.len(),
            buffer_len = buffer.len(),
            "Built audio buffer with lookback"
        );

        buffer
    }

    /// Update the audio overlap buffer without building the full audio buffer.
    ///
    /// This is used when skipping silent chunks to maintain continuity
    /// in the overlap buffer for when speech returns.
    fn update_audio_overlap(&mut self, audio_chunk: &[f32]) {
        let lookback_samples = self.mel_lookback_samples();

        // Initialize overlap if empty
        if self.audio_overlap.is_empty() {
            self.audio_overlap = vec![0.0f32; lookback_samples];
        }

        // Save last lookback_samples from audio_chunk for next iteration
        if audio_chunk.len() >= lookback_samples {
            let start_idx = audio_chunk.len() - lookback_samples;
            self.audio_overlap = audio_chunk[start_idx..].to_vec();
        } else {
            // Chunk is shorter than lookback_samples
            let shift = audio_chunk.len();
            self.audio_overlap.drain(..shift);
            self.audio_overlap.extend_from_slice(audio_chunk);
        }
    }

    fn process_pending_audio(&mut self) -> Result<()> {
        let chunk_audio_size = self.audio_chunk_size();

        tracing::trace!(
            chunk_audio_size,
            file_position = self.file_position,
            live_buffer_len = self.live_buffer.len(),
            "Processing pending audio"
        );

        // Process file audio
        // First, check if we have file samples and extract the chunk to avoid borrow issues
        let (audio_chunk, new_position, reached_end) = if let Some(ref samples) = self.file_samples
        {
            if self.file_position < samples.len() {
                let end = (self.file_position + chunk_audio_size).min(samples.len());
                let chunk = samples[self.file_position..end].to_vec();
                let reached = end >= samples.len();
                (Some(chunk), end, reached)
            } else {
                (None, self.file_position, false)
            }
        } else {
            (None, self.file_position, false)
        };

        // Python's BatchedCacheFeatureBufferer approach:
        // 1. Take audio chunk + small lookback for mel continuity
        // 2. Compute mel, trim to expected_feat_len, then take last feature_chunk_len frames
        // 3. Append to feature buffer (which maintains pre_encode_cache overlap)
        let feature_chunk_len = self.config.chunk_size.1; // 112 frames
        // Python: expected_feat_len = feature_chunk_len + plus_one, where plus_one = 1
        // This matches Python's trimming before taking the last chunk
        let expected_feat_len = feature_chunk_len + 1; // 113 frames

        // Now process the chunk if we have one
        if let Some(audio_chunk) = audio_chunk {
            if !audio_chunk.is_empty() {
                // Skip silent chunks - don't process through mel/encoder
                if is_silent(&audio_chunk, self.silence_threshold) {
                    tracing::trace!(
                        rms = %format!("{:.6}", compute_rms(&audio_chunk)),
                        threshold = self.silence_threshold,
                        chunk_samples = audio_chunk.len(),
                        "Skipping silent audio chunk (file)"
                    );
                    // Update position but don't produce mel frames
                    self.file_position = new_position;
                    // Still update overlap buffer for continuity when speech returns
                    self.update_audio_overlap(&audio_chunk);

                    // If we've reached the end of file, signal stream end
                    if reached_end && !self.stream_ended {
                        self.stream_ended = true;
                    }
                    return Ok(());
                }

                // Build the audio buffer with small lookback for mel continuity
                let audio_buffer = self.build_audio_buffer(&audio_chunk);

                let audio_array =
                    mlx_rs::Array::from_slice(&audio_buffer, &[audio_buffer.len() as i32]);
                let (mel, _lengths) = self.preprocessor.forward(&audio_array, None)?;
                // Remove batch dim: [batch, n_mels, frames] -> [n_mels, frames]
                use mlx_rs::ops::indexing::IndexOp;
                let mel = mel.index(0);
                mel.eval()?;
                let n_frames = mel.dim(1) as usize;

                if n_frames > 0 {
                    // Python's approach:
                    // 1. First trim to expected_feat_len (113)
                    // 2. Then take last feature_chunk_len (112) frames
                    // This ensures we take frames 1-112, not frames 2-113
                    let mel_trimmed = if n_frames > expected_feat_len {
                        mel.index((.., ..expected_feat_len as i32))
                    } else {
                        mel
                    };
                    let trimmed_frames = mel_trimmed.dim(1) as usize;

                    let mel_to_append = if trimmed_frames > feature_chunk_len {
                        let start = (trimmed_frames - feature_chunk_len) as i32;
                        mel_trimmed.index((.., start..))
                    } else {
                        mel_trimmed
                    };
                    let appended_frames = mel_to_append.dim(1) as usize;
                    self.mel_buffer.append_mel(&mel_to_append)?;

                    let audio_position_secs = self.file_position as f32 / self.sample_rate as f32;
                    tracing::debug!(
                        audio_chunk_samples = audio_chunk.len(),
                        audio_buffer_samples = audio_buffer.len(),
                        audio_position_secs = %format!("{:.3}s", audio_position_secs),
                        mel_computed = n_frames,
                        mel_trimmed = trimmed_frames,
                        mel_appended = appended_frames,
                        mel_buffer_total = self.mel_buffer.total_frames,
                        "STREAMER: processed audio -> mel"
                    );
                }
                self.file_position = new_position;
            }

            // If we've reached the end of file, signal stream end
            if reached_end && !self.stream_ended {
                self.stream_ended = true;
            }
        }

        // Process live audio in streaming-chunk-sized blocks
        // Python's BatchedCacheFeatureBufferer approach:
        // 1. Take audio chunk + small lookback for mel continuity
        // 2. Compute mel, trim to expected_feat_len, then take last feature_chunk_len frames
        // 3. Append to feature buffer (which maintains pre_encode_cache overlap)

        while self.live_buffer.len() >= chunk_audio_size {
            // Take exactly chunk_audio_size samples
            let audio_chunk: Vec<f32> = self.live_buffer.drain(..chunk_audio_size).collect();

            // Update progress counters
            self.live_chunk_counter += 1;
            self.live_samples_processed += audio_chunk.len();

            // Skip silent chunks - don't process through mel/encoder
            if is_silent(&audio_chunk, self.silence_threshold) {
                tracing::trace!(
                    rms = %format!("{:.6}", compute_rms(&audio_chunk)),
                    threshold = self.silence_threshold,
                    chunk_num = self.live_chunk_counter,
                    "Skipping silent audio chunk (live)"
                );
                // Still update overlap buffer for continuity when speech returns
                self.update_audio_overlap(&audio_chunk);
                continue;
            }

            // Build the audio buffer with small lookback for mel continuity
            let audio_buffer = self.build_audio_buffer(&audio_chunk);

            let audio_array =
                mlx_rs::Array::from_slice(&audio_buffer, &[audio_buffer.len() as i32]);
            let (mel, _lengths) = self.preprocessor.forward(&audio_array, None)?;
            // Remove batch dim: [batch, n_mels, frames] -> [n_mels, frames]
            let mel = mel.index(0);
            mel.eval()?;
            let n_frames = mel.dim(1) as usize;

            if n_frames > 0 {
                // Python's approach:
                // 1. First trim to expected_feat_len (113)
                // 2. Then take last feature_chunk_len (112) frames
                // This ensures we take frames 1-112, not frames 2-113
                let mel_trimmed = if n_frames > expected_feat_len {
                    mel.index((.., ..expected_feat_len as i32))
                } else {
                    mel
                };
                let trimmed_frames = mel_trimmed.dim(1) as usize;

                let mel_to_append = if trimmed_frames > feature_chunk_len {
                    let start = (trimmed_frames - feature_chunk_len) as i32;
                    mel_trimmed.index((.., start..))
                } else {
                    mel_trimmed
                };
                let appended_frames = mel_to_append.dim(1) as usize;
                self.mel_buffer.append_mel(&mel_to_append)?;

                tracing::trace!(
                    chunk_num = self.live_chunk_counter,
                    samples_processed = self.live_samples_processed,
                    audio_chunk_len = audio_chunk.len(),
                    audio_buffer_len = audio_buffer.len(),
                    mel_computed = n_frames,
                    mel_trimmed = trimmed_frames,
                    mel_appended = appended_frames,
                    buffer_total = self.mel_buffer.total_frames,
                    live_buffer_remaining = self.live_buffer.len(),
                    "Processed live audio chunk"
                );
            }
        }

        // Process remaining audio only when stream has ended
        if self.stream_ended && !self.live_buffer.is_empty() {
            tracing::debug!(
                remaining_samples = self.live_buffer.len(),
                "Processing remaining audio at stream end"
            );

            // Reset preprocessor state
            // self.preprocessor.reset();

            // Build the audio buffer
            let remaining = self.live_buffer.clone();
            let audio_buffer = self.build_audio_buffer(&remaining);

            let audio_array =
                mlx_rs::Array::from_slice(&audio_buffer, &[audio_buffer.len() as i32]);
            let (mel, _lengths) = self.preprocessor.forward(&audio_array, None)?;
            // Remove batch dim: [batch, n_mels, frames] -> [n_mels, frames]
            let mel = mel.index(0);
            mel.eval()?;
            let n_frames = mel.dim(1) as usize;

            if n_frames > 0 {
                // Keep all mel frames
                self.mel_buffer.append_mel(&mel)?;
            }
            self.live_buffer.clear();

            tracing::trace!(
                mel_frames = n_frames,
                "Processed final remaining audio"
            );
        }

        Ok(())
    }

    fn handle_final_chunk(&mut self) -> Result<Option<FeatureBuffer>> {
        tracing::debug!(
            mel_frames_before_flush = self.mel_buffer.remaining_frames(),
            "Handling final chunk"
        );

        // Flush any remaining audio in preprocessor
        let final_mel = self.preprocessor.flush()?;
        final_mel.eval()?;
        let n_frames = final_mel.dim(1) as usize;
        if n_frames > 0 {
            tracing::trace!(flushed_frames = n_frames, "Flushed preprocessor");
            self.mel_buffer.append_mel(&final_mel)?;
        }

        let remaining = self.mel_buffer.remaining_frames();
        if remaining == 0 {
            self.is_done = true;
            return Ok(None);
        }

        // Use consistent chunk parameters (always use subsequent values like Python)
        let chunk_size = self.config.chunk_size.1;
        let shift_size = self.config.shift_size.1;
        let pre_encode_cache = self.config.pre_encode_cache_size.1;

        // Pad to make a full chunk if needed.
        // We need chunk_size + pre_encode_cache frames total for extraction:
        // - pre_encode_cache frames from overlap (context from previous chunk)
        // - chunk_size frames of new content
        let required_frames = chunk_size + pre_encode_cache;
        if remaining < required_frames {
            let frames_to_pad = required_frames - remaining;
            tracing::trace!(
                remaining,
                required_frames,
                frames_to_pad,
                "Padding final chunk"
            );
            let padding = mlx_rs::ops::zeros::<f32>(&[self.n_mels as i32, frames_to_pad as i32])?;
            self.mel_buffer.append_mel(&padding)?;
        }

        // Extract with proper parameters: chunk_size new frames, shift by shift_size,
        // pre_encode_cache frames from overlap
        let (mel_chunk, _total_frames, valid_frames) = self
            .mel_buffer
            .extract_chunk(chunk_size, shift_size, pre_encode_cache)
            .ok_or_else(|| Error::model("Failed to extract final mel chunk"))?;

        let is_first = self.is_first_chunk;
        self.is_first_chunk = false;
        self.is_done = true;

        tracing::debug!(
            is_first,
            valid_frames,
            pre_encode_cache,
            "Returning final FeatureBuffer"
        );

        Ok(Some(
            FeatureBuffer::new(mel_chunk, valid_frames)
                .with_first(is_first)
                .with_last(true)
                .with_pre_encode_cache(pre_encode_cache),
        ))
    }
}

impl Iterator for RequestStreamer {
    type Item = Result<FeatureBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            return None;
        }

        match self.try_extract_next() {
            Ok(Some(buffer)) => Some(Ok(buffer)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
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

    tracing::trace!(
        sample_rate,
        channels,
        bits_per_sample = spec.bits_per_sample,
        sample_format = ?spec.sample_format,
        "WAV file spec"
    );

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
        tracing::trace!("No resampling needed (rates match)");
        return Ok(samples.to_vec());
    }

    tracing::debug!(
        from_rate,
        to_rate,
        input_samples = samples.len(),
        "Resampling audio"
    );

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

    let output = waves_out.into_iter().next().unwrap_or_default();
    tracing::trace!(
        input_samples = samples.len(),
        output_samples = output.len(),
        "Resampling complete"
    );
    Ok(output)
}

#[cfg(not(feature = "asr"))]
fn resample_audio(_samples: &[f32], _from_rate: u32, _to_rate: u32) -> Result<Vec<f32>> {
    Err(Error::not_implemented(
        "Audio resampling requires the 'asr' feature",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require model/preprocessor setup
    // These are basic structure tests

    #[test]
    fn test_request_streamer_state() {
        // Can't fully test without preprocessor, but can test state logic
        assert!(true); // Placeholder
    }

    #[test]
    fn test_compute_rms() {
        // Silence
        let silence = vec![0.0f32; 100];
        assert_eq!(compute_rms(&silence), 0.0);

        // Constant signal: RMS of [1.0, 1.0, ...] = 1.0
        let constant = vec![1.0f32; 100];
        assert!((compute_rms(&constant) - 1.0).abs() < 1e-6);

        // Alternating signal: RMS of [-1.0, 1.0, ...] = 1.0
        let alternating: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((compute_rms(&alternating) - 1.0).abs() < 1e-6);

        // Small signal
        let small = vec![0.01f32; 100];
        assert!((compute_rms(&small) - 0.01).abs() < 1e-6);

        // Empty
        assert_eq!(compute_rms(&[]), 0.0);
    }

    #[test]
    fn test_is_silent() {
        // Threshold disabled (0.0)
        let audio = vec![0.0f32; 100];
        assert!(!is_silent(&audio, 0.0));

        // Silent audio with threshold enabled
        let silence = vec![0.0f32; 100];
        assert!(is_silent(&silence, 0.01));

        // Loud audio above threshold
        let loud = vec![0.1f32; 100];
        assert!(!is_silent(&loud, 0.01));

        // Quiet audio below threshold
        let quiet = vec![0.005f32; 100];
        assert!(is_silent(&quiet, 0.01));

        // Audio just above threshold
        let borderline = vec![0.015f32; 100];
        assert!(!is_silent(&borderline, 0.01));

        // Empty audio
        assert!(!is_silent(&[], 0.01));

        // Negative threshold (disabled)
        assert!(!is_silent(&silence, -1.0));
    }
}
