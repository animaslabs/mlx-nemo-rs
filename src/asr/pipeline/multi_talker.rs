//! Multi-talker streaming pipeline.
//!
//! Combines a streaming ASR pipeline with a SortFormer diarization model
//! to produce speaker-tagged transcriptions. Uses a serial approach:
//! single ASR instance + diarization tagging.
//!
//! # Example
//!
//! ```rust,ignore
//! let mut pipeline = MultiTalkerPipeline::restore(
//!     "nemotron-speech-streaming-en-0.6b.nemo",
//!     "diar_streaming_sortformer_4spk-v2.1.nemo",
//!     0, // streaming mode
//! )?;
//!
//! pipeline.set_audio_file("meeting.wav")?;
//! while let Some(output) = pipeline.transcribe_step()? {
//!     for segment in &output.segments {
//!         println!("[Speaker {}] {}", segment.speaker, segment.text);
//!     }
//! }
//! ```

use std::path::Path;

use mlx_rs::Array;

use super::cache_aware_rnnt::{CacheAwareRNNTPipeline, TranscribeOutput};
use crate::asr::models::SortformerEncLabelModel;
use crate::asr::streaming::StreamingSortformerState;
use crate::core::ArrayExt;
use crate::core::error::Result;

/// A segment of text attributed to a specific speaker.
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    /// Speaker index (0-based, up to max_num_spks - 1)
    pub speaker: usize,
    /// Text content for this segment
    pub text: String,
    /// Start time in seconds (approximate)
    pub start_time: f32,
    /// End time in seconds (approximate)
    pub end_time: f32,
}

/// Output from a multi-talker transcription step.
#[derive(Debug, Clone)]
pub struct MultiTalkerOutput {
    /// Speaker-tagged text segments
    pub segments: Vec<SpeakerSegment>,
    /// Raw ASR output (without speaker tagging)
    pub asr_output: TranscribeOutput,
    /// Whether this is the final output
    pub is_final: bool,
}

/// Multi-talker streaming pipeline.
///
/// Combines `CacheAwareRNNTPipeline` (ASR) with `SortformerEncLabelModel` (diarization)
/// to produce speaker-attributed transcription.
pub struct MultiTalkerPipeline {
    /// ASR streaming pipeline
    asr_pipeline: CacheAwareRNNTPipeline,
    /// Diarization model
    diar_model: SortformerEncLabelModel,
    /// Diarization streaming state
    diar_state: StreamingSortformerState,
    /// Accumulated diarization predictions (per-frame speaker probs)
    diar_preds: Vec<Vec<f32>>,
    /// Number of speakers
    max_num_spks: usize,
    /// Seconds per diarization frame (after encoder subsampling)
    diar_frame_sec: f32,
    /// Sample rate
    sample_rate: u32,
    /// Audio samples accumulated for diarization
    diar_audio_buffer: Vec<f32>,
    /// Diarization chunk size in samples
    diar_chunk_samples: usize,
    /// Total diarization frames processed
    total_diar_frames: usize,
}

impl MultiTalkerPipeline {
    /// Create a multi-talker pipeline from ASR and diarization model paths.
    ///
    /// # Arguments
    /// * `asr_path` - Path to the streaming ASR .nemo model
    /// * `diar_path` - Path to the SortFormer diarization .nemo model
    /// * `mode_idx` - Streaming mode index for the ASR model
    pub fn restore(
        asr_path: impl AsRef<Path>,
        diar_path: impl AsRef<Path>,
        mode_idx: usize,
    ) -> Result<Self> {
        // Load ASR pipeline
        let asr_pipeline = CacheAwareRNNTPipeline::restore(asr_path, mode_idx)?;

        // Load diarization model
        let diar_model = SortformerEncLabelModel::restore(diar_path)?;

        let sample_rate = asr_pipeline.sample_rate();
        let max_num_spks = diar_model.max_num_spks();

        // Create diarization streaming state
        let diar_state = diar_model.create_streaming_state(1)?;

        // Compute diarization frame duration
        // The conformer subsamples by subsampling_factor, with window_stride of 0.01s
        let subsampling_factor = diar_model.subsampling_factor();
        let window_stride = 0.01; // Default NeMo window_stride
        let diar_frame_sec = window_stride * subsampling_factor as f32;

        // Diarization chunk size: use the config chunk_len * subsampling_factor * stride * sample_rate
        let chunk_len = diar_model.config().chunk_len;
        let diar_chunk_samples = (chunk_len as f32 * diar_frame_sec * sample_rate as f32) as usize;

        tracing::info!(
            max_num_spks,
            subsampling_factor,
            diar_frame_sec,
            diar_chunk_samples,
            "Multi-talker pipeline initialized"
        );

        Ok(Self {
            asr_pipeline,
            diar_model,
            diar_state,
            diar_preds: Vec::new(),
            max_num_spks,
            diar_frame_sec,
            sample_rate,
            diar_audio_buffer: Vec::new(),
            diar_chunk_samples,
            total_diar_frames: 0,
        })
    }

    /// Set audio from a WAV file.
    pub fn set_audio_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.asr_pipeline.set_audio_file(&path)?;

        // Also load audio for diarization
        let (samples, file_sr) = crate::audio::load_wav(path.as_ref())?;
        let samples = if file_sr != self.sample_rate {
            crate::audio::resample(&samples, file_sr, self.sample_rate)?
        } else {
            samples
        };
        self.diar_audio_buffer = samples;

        Ok(())
    }

    /// Add live audio samples.
    pub fn add_audio(&mut self, samples: &[f32]) {
        self.asr_pipeline.add_audio(samples);
        self.diar_audio_buffer.extend_from_slice(samples);
    }

    /// Signal end of audio stream.
    pub fn end_stream(&mut self) {
        self.asr_pipeline.end_stream();
    }

    /// Reset the pipeline for new audio.
    pub fn reset(&mut self) {
        self.asr_pipeline.reset();
        self.diar_preds.clear();
        self.diar_audio_buffer.clear();
        self.total_diar_frames = 0;

        // Recreate streaming state
        if let Ok(state) = self.diar_model.create_streaming_state(1) {
            self.diar_state = state;
        }
    }

    /// Process one transcription step, returning speaker-tagged output.
    pub fn transcribe_step(&mut self) -> Result<Option<MultiTalkerOutput>> {
        // Run ASR step
        let asr_output = match self.asr_pipeline.transcribe_step()? {
            Some(out) => out,
            None => return Ok(None),
        };

        // Process any pending diarization audio
        self.process_diarization()?;

        // Tag the ASR output with speaker information
        let segments = self.tag_with_speakers(&asr_output);
        let is_final = asr_output.is_final;

        Ok(Some(MultiTalkerOutput {
            segments,
            asr_output,
            is_final,
        }))
    }

    /// Run full transcription on loaded audio.
    pub fn run(&mut self) -> Result<Vec<SpeakerSegment>> {
        let mut all_segments = Vec::new();

        // Run diarization on all available audio first
        self.run_diarization()?;

        // Run ASR
        while let Some(output) = self.transcribe_step()? {
            all_segments.extend(output.segments);
            if output.is_final {
                break;
            }
        }

        // Merge consecutive segments from same speaker
        Ok(merge_segments(all_segments))
    }

    /// Run diarization on all buffered audio (batch mode).
    pub fn run_diarization(&mut self) -> Result<()> {
        if self.diar_audio_buffer.is_empty() {
            return Ok(());
        }

        let audio = std::mem::take(&mut self.diar_audio_buffer);

        // Process audio → mel → diarization
        let audio_arr = Array::from_slice(&audio, &[1, audio.len() as i32]);
        let audio_len = Array::from_slice(&[audio.len() as i32], &[1]);

        let preds = self.diar_model.forward(&audio_arr, &audio_len)?;
        preds.eval()?;

        // Store per-frame predictions
        let n_frames = preds.dim(1) as usize;
        let preds_flat = preds.reshape(&[-1])?;
        preds_flat.eval()?;
        let preds_data: Vec<f32> = preds_flat.to_vec_row_major()?;

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.max_num_spks;
            let end = start + self.max_num_spks;
            self.diar_preds.push(preds_data[start..end].to_vec());
        }
        self.total_diar_frames += n_frames;

        Ok(())
    }

    /// Process pending diarization audio in chunks.
    fn process_diarization(&mut self) -> Result<()> {
        // Process complete chunks of audio for diarization using batch forward
        while self.diar_audio_buffer.len() >= self.diar_chunk_samples {
            let chunk: Vec<f32> = self.diar_audio_buffer.drain(..self.diar_chunk_samples).collect();

            let audio_arr = Array::from_slice(&chunk, &[1, chunk.len() as i32]);
            let audio_len = Array::from_slice(&[chunk.len() as i32], &[1]);

            // Run diarization on chunk (non-streaming for now)
            let preds = self.diar_model.forward(&audio_arr, &audio_len)?;
            preds.eval()?;

            // Store predictions
            let n_frames = preds.dim(1) as usize;
            let preds_flat = preds.reshape(&[-1])?;
            preds_flat.eval()?;
            let preds_data: Vec<f32> = preds_flat.to_vec_row_major()?;

            for frame_idx in 0..n_frames {
                let start = frame_idx * self.max_num_spks;
                let end = start + self.max_num_spks;
                self.diar_preds.push(preds_data[start..end].to_vec());
            }
            self.total_diar_frames += n_frames;
        }

        Ok(())
    }

    /// Tag ASR output tokens with speaker IDs based on diarization predictions.
    fn tag_with_speakers(&self, asr_output: &TranscribeOutput) -> Vec<SpeakerSegment> {
        if asr_output.text.is_empty() || self.diar_preds.is_empty() {
            if !asr_output.text.is_empty() {
                return vec![SpeakerSegment {
                    speaker: 0,
                    text: asr_output.text.clone(),
                    start_time: 0.0,
                    end_time: 0.0,
                }];
            }
            return Vec::new();
        }

        // Simple approach: split text into words, assign each word to a speaker
        // based on its approximate timestamp in the diarization output
        let words: Vec<&str> = asr_output.text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let total_diar_time = self.total_diar_frames as f32 * self.diar_frame_sec;

        let mut segments = Vec::new();
        let mut current_speaker = None;
        let mut current_text = String::new();
        let mut segment_start = 0.0f32;

        for (word_idx, word) in words.iter().enumerate() {
            // Estimate time position for this word
            let word_time = (word_idx as f32 / words.len() as f32) * total_diar_time;

            // Find the corresponding diarization frame
            let diar_frame_idx = (word_time / self.diar_frame_sec) as usize;
            let diar_frame_idx = diar_frame_idx.min(self.diar_preds.len().saturating_sub(1));

            // Get the dominant speaker for this frame
            let speaker = if diar_frame_idx < self.diar_preds.len() {
                let probs = &self.diar_preds[diar_frame_idx];
                probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                0
            };

            // Check if speaker changed
            if current_speaker != Some(speaker) {
                // Emit previous segment if any
                if !current_text.is_empty() {
                    segments.push(SpeakerSegment {
                        speaker: current_speaker.unwrap_or(0),
                        text: current_text.trim().to_string(),
                        start_time: segment_start,
                        end_time: word_time,
                    });
                }
                current_speaker = Some(speaker);
                current_text = String::new();
                segment_start = word_time;
            }

            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(word);
        }

        // Emit final segment
        if !current_text.is_empty() {
            segments.push(SpeakerSegment {
                speaker: current_speaker.unwrap_or(0),
                text: current_text.trim().to_string(),
                start_time: segment_start,
                end_time: total_diar_time,
            });
        }

        segments
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Merge consecutive segments from the same speaker.
fn merge_segments(segments: Vec<SpeakerSegment>) -> Vec<SpeakerSegment> {
    if segments.is_empty() {
        return segments;
    }

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for segment in segments.into_iter().skip(1) {
        if segment.speaker == current.speaker {
            // Same speaker - merge
            current.text.push(' ');
            current.text.push_str(&segment.text);
            current.end_time = segment.end_time;
        } else {
            // Different speaker - emit and start new
            merged.push(current);
            current = segment;
        }
    }
    merged.push(current);

    merged
}
