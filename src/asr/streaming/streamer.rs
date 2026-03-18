//! Request streamer for streaming ASR inference.
//!
//! This module provides a request generator that handles audio chunking,
//! mel computation, and buffer management for streaming pipelines.
//! It mirrors NeMo's `ContinuousBatchedRequestStreamer`.

use std::path::Path;

use crate::asr::modules::MelSpectrogram;
use crate::asr::streaming::{FeatureBuffer, StreamingConfig, StreamingMelBuffer};
use crate::core::error::{Error, Result};

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

        // Initialize mel buffer with pre_encode_cache silence frames.
        // This matches Python's BatchedCacheFeatureBufferer which initializes
        // feature_buffer to LOG_MEL_ZERO. The first pre_encode_cache frames
        // provide context for the encoder's subsampling layer.
        let pre_encode_cache = config.pre_encode_cache_size.1;
        if let Err(e) = mel_buffer.initialize_with_silence(pre_encode_cache) {
            tracing::warn!("Failed to initialize mel buffer with silence: {}", e);
        }

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

        self.file_samples = Some(samples);
        self.file_position = 0;
        self.reset_stream_state();
        Ok(())
    }

    /// Set audio samples directly (for pre-loaded audio).
    pub fn set_audio_samples(&mut self, samples: Vec<f32>) {
        self.file_samples = Some(samples);
        self.file_position = 0;
        self.reset_stream_state();
    }

    /// Add live audio samples to the buffer.
    ///
    /// Use this for real-time streaming where audio arrives incrementally.
    pub fn add_audio(&mut self, samples: &[f32]) {
        self.live_buffer.extend_from_slice(samples);
    }

    /// Signal the end of the audio stream.
    ///
    /// After calling this, the streamer will:
    /// 1. Flush remaining audio through the preprocessor
    /// 2. Process any remaining mel frames
    /// 3. Mark the final request with `is_last = true`
    pub fn end_stream(&mut self) {
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
        self.file_samples = None;
        self.file_position = 0;
        self.live_buffer.clear();
        self.mel_buffer.reset();
        self.preprocessor.reset();
        self.audio_overlap.clear();
        self.reset_stream_state();

        // Re-initialize mel buffer with pre_encode_cache silence frames
        let pre_encode_cache = self.config.pre_encode_cache_size.1;
        if let Err(e) = self.mel_buffer.initialize_with_silence(pre_encode_cache) {
            tracing::warn!("Failed to initialize mel buffer with silence: {}", e);
        }
    }

    fn reset_stream_state(&mut self) {
        self.is_first_chunk = true;
        self.stream_ended = false;
        self.is_done = false;
        self.audio_overlap.clear();
    }

    /// Get audio chunk size in samples for the current streaming config.
    ///
    /// This calculates how many NEW audio samples are consumed per chunk.
    /// This matches Python's MonoStream frame_size = chunk_size_in_secs * sample_rate.
    ///
    /// Note: The mel buffer receives more samples per chunk (chunk + pre_encode_cache)
    /// because we prepend overlap/silence. But this method returns just the NEW audio
    /// consumed from the source per iteration.
    pub fn audio_chunk_size(&self) -> usize {
        // Just the chunk size, NOT including pre_encode_cache
        // This matches Python's MonoStream frame_size
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
        // Step 1: Process any pending audio into mel frames
        self.process_pending_audio()?;

        // Step 2: Try to extract a mel chunk
        // NeMo's CacheAwareRNNTPipeline uses the "subsequent" values (index [1]) for ALL chunks:
        //   model_chunk_size = model_chunk_size[1]
        //   pre_encode_cache_size = pre_encode_cache_size[1]
        // This differs from the encoder's setup_streaming_params which uses different
        // values for first vs subsequent chunks.
        let chunk_size = self.config.chunk_size.1;
        let shift_size = self.config.shift_size.1;
        let pre_encode_cache = self.config.pre_encode_cache_size.1;

        // Check if we have enough mel frames
        let remaining = self.mel_buffer.remaining_frames();
        if remaining < chunk_size {
            // Not enough frames yet
            if self.stream_ended {
                // Handle final partial chunk
                return self.handle_final_chunk();
            }
            return Ok(None);
        }

        // Extract mel chunk
        // Returns (mel_array, total_frames, valid_frames)
        // valid_frames matches Python's mel_len (total - 1)
        let (mel_chunk, _total_frames, valid_frames) = self
            .mel_buffer
            .extract_chunk(chunk_size, shift_size, pre_encode_cache)
            .ok_or_else(|| Error::model("Failed to extract mel chunk"))?;

        let is_first = self.is_first_chunk;
        self.is_first_chunk = false;

        // Check if this is the last chunk
        let is_last = self.stream_ended && self.mel_buffer.is_empty();
        if is_last {
            self.is_done = true;
        }

        Ok(Some(
            FeatureBuffer::new(mel_chunk, valid_frames)
                .with_first(is_first)
                .with_last(is_last)
                .with_pre_encode_cache(pre_encode_cache),
        ))
    }

    /// Build the audio buffer for mel computation.
    ///
    /// This mirrors Python's AudioBufferer behavior:
    /// - First chunk: buffer = [zeros (pre_encode) | audio (chunk)]
    /// - Subsequent: buffer = [overlap from prev | audio (chunk)]
    ///
    /// The overlap (pre_encode_cache samples from the end of each chunk)
    /// is saved for the next chunk.
    fn build_audio_buffer(&mut self, audio_chunk: &[f32]) -> Vec<f32> {
        let pre_encode_samples = self.pre_encode_cache_samples();
        let buffer_size = self.audio_buffer_size();

        let mut buffer = Vec::with_capacity(buffer_size);

        if self.is_first_chunk {
            // First chunk: prepend silence (zeros) for pre_encode_cache
            tracing::debug!(
                pre_encode_samples,
                audio_chunk_len = audio_chunk.len(),
                "First chunk: prepending silence"
            );
            buffer.resize(pre_encode_samples, 0.0f32);
            buffer.extend_from_slice(audio_chunk);
        } else {
            // Subsequent chunks: prepend overlap from previous chunk
            if self.audio_overlap.len() >= pre_encode_samples {
                buffer.extend_from_slice(&self.audio_overlap[..pre_encode_samples]);
            } else {
                // Pad with zeros if overlap is too short (shouldn't happen normally)
                let zeros_needed = pre_encode_samples - self.audio_overlap.len();
                buffer.resize(zeros_needed, 0.0f32);
                buffer.extend_from_slice(&self.audio_overlap);
            }
            buffer.extend_from_slice(audio_chunk);
        }

        // Pad to full buffer size if needed (for final short chunk)
        if buffer.len() < buffer_size {
            buffer.resize(buffer_size, 0.0f32);
        }

        // Save the end of this chunk as overlap for the next chunk.
        // We need pre_encode_cache samples from the end of the NEW audio
        // (not from the buffer which includes previous overlap).
        self.audio_overlap.clear();
        if audio_chunk.len() >= pre_encode_samples {
            // Take last pre_encode_samples from the audio chunk
            self.audio_overlap
                .extend_from_slice(&audio_chunk[audio_chunk.len() - pre_encode_samples..]);
        } else {
            // Audio chunk is shorter than pre_encode_samples
            // This can happen for final chunks - include what we have
            // and pad the rest from the beginning of the chunk
            if !self.audio_overlap.is_empty() && buffer.len() >= pre_encode_samples {
                // Take from the end of the buffer we just built
                self.audio_overlap
                    .extend_from_slice(&buffer[buffer.len() - pre_encode_samples..]);
            } else {
                self.audio_overlap.extend_from_slice(audio_chunk);
            }
        }

        buffer
    }

    fn process_pending_audio(&mut self) -> Result<()> {
        let chunk_audio_size = self.audio_chunk_size();

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

        // Now process the chunk if we have one
        if let Some(audio_chunk) = audio_chunk {
            if !audio_chunk.is_empty() {
                // Reset preprocessor state before each chunk
                self.preprocessor.reset();

                // Build the full audio buffer with overlap/silence prefix
                let audio_buffer = self.build_audio_buffer(&audio_chunk);

                let mel = self.preprocessor.process_chunk(&audio_buffer)?;
                mel.eval()?;
                let n_frames = mel.dim(1) as usize;
                if n_frames > 0 {
                    // Take only the LAST chunk_size mel frames, matching Python's
                    // BatchedCacheFeatureBufferer which does:
                    //   feat_chunk=features[:, :, -self.feature_chunk_len:]
                    // This discards the overlap/zeros portion at the start and keeps
                    // only the frames corresponding to the new audio.
                    let chunk_size = self.config.chunk_size.1;
                    let fresh_mel = if n_frames > chunk_size {
                        // Take only the LAST chunk_size frames
                        use mlx_rs::ops::indexing::IndexOp;
                        let start = (n_frames - chunk_size) as i32;
                        mel.index((.., start..))
                    } else {
                        // Not enough frames: keep all
                        mel
                    };
                    let fresh_n_frames = fresh_mel.dim(1) as usize;
                    if fresh_n_frames > 0 {
                        self.mel_buffer.append_mel(&fresh_mel)?;
                    }
                }
                self.file_position = new_position;
            }

            // If we've reached the end of file, signal stream end
            if reached_end && !self.stream_ended {
                self.stream_ended = true;
            }
        }

        // Process live audio in streaming-chunk-sized blocks
        while self.live_buffer.len() >= chunk_audio_size {
            // Take exactly chunk_audio_size samples
            let audio_chunk: Vec<f32> = self.live_buffer.drain(..chunk_audio_size).collect();

            // Reset preprocessor state before each chunk
            self.preprocessor.reset();

            // Build the full audio buffer with overlap/silence prefix
            let audio_buffer = self.build_audio_buffer(&audio_chunk);

            let mel = self.preprocessor.process_chunk(&audio_buffer)?;
            mel.eval()?;
            let n_frames = mel.dim(1) as usize;
            if n_frames > 0 {
                // Take only the LAST chunk_size mel frames (see comment above)
                let chunk_size = self.config.chunk_size.1;
                let fresh_mel = if n_frames > chunk_size {
                    use mlx_rs::ops::indexing::IndexOp;
                    let start = (n_frames - chunk_size) as i32;
                    mel.index((.., start..))
                } else {
                    mel
                };
                let fresh_n_frames = fresh_mel.dim(1) as usize;
                if fresh_n_frames > 0 {
                    self.mel_buffer.append_mel(&fresh_mel)?;
                }
            }
        }

        // Process remaining audio only when stream has ended
        if self.stream_ended && !self.live_buffer.is_empty() {
            // Reset preprocessor state
            self.preprocessor.reset();

            // Build the full audio buffer with overlap/silence prefix
            let remaining = self.live_buffer.clone();
            let audio_buffer = self.build_audio_buffer(&remaining);

            let mel = self.preprocessor.process_chunk(&audio_buffer)?;
            mel.eval()?;
            let n_frames = mel.dim(1) as usize;
            if n_frames > 0 {
                // Take only the LAST chunk_size mel frames (see comment above)
                let chunk_size = self.config.chunk_size.1;
                let fresh_mel = if n_frames > chunk_size {
                    use mlx_rs::ops::indexing::IndexOp;
                    let start = (n_frames - chunk_size) as i32;
                    mel.index((.., start..))
                } else {
                    mel
                };
                let fresh_n_frames = fresh_mel.dim(1) as usize;
                if fresh_n_frames > 0 {
                    self.mel_buffer.append_mel(&fresh_mel)?;
                }
            }
            self.live_buffer.clear();
        }

        Ok(())
    }

    fn handle_final_chunk(&mut self) -> Result<Option<FeatureBuffer>> {
        // Flush any remaining audio in preprocessor
        let final_mel = self.preprocessor.flush()?;
        final_mel.eval()?;
        let n_frames = final_mel.dim(1) as usize;
        if n_frames > 0 {
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

        // Pad to make a full chunk if needed
        if remaining < chunk_size {
            let frames_to_pad = chunk_size - remaining;
            let padding = mlx_rs::ops::zeros::<f32>(&[self.n_mels as i32, frames_to_pad as i32])?;
            self.mel_buffer.append_mel(&padding)?;
        }

        let (mel_chunk, _total_frames, valid_frames) = self
            .mel_buffer
            .extract_chunk(chunk_size, shift_size, pre_encode_cache)
            .ok_or_else(|| Error::model("Failed to extract final mel chunk"))?;

        let is_first = self.is_first_chunk;
        self.is_first_chunk = false;
        self.is_done = true;

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

/// Resample audio to target sample rate.
#[cfg(feature = "asr")]
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
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
}
