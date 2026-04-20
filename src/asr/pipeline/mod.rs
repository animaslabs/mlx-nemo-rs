//! Streaming ASR pipelines.
//!
//! This module provides high-level pipeline abstractions for streaming
//! ASR inference, mirroring NeMo's `CacheAwareRNNTPipeline`.
//!
//! # Example
//!
//! ```rust,ignore
//! use mlx_nemo::asr::pipeline::CacheAwareRNNTPipeline;
//!
//! // Load model and create pipeline
//! let mut pipeline = CacheAwareRNNTPipeline::restore("model.nemo", 0)?;
//!
//! // File mode
//! pipeline.set_audio_file("audio.wav")?;
//! let text = pipeline.run()?;
//! println!("Transcription: {}", text);
//!
//! // Live streaming mode
//! pipeline.reset();
//! pipeline.add_audio(&chunk1);
//! while let Some(output) = pipeline.transcribe_step()? {
//!     print!("{}", output.text);
//! }
//! pipeline.end_stream();
//! while let Some(output) = pipeline.transcribe_step()? {
//!     print!("{}", output.text);
//! }
//! ```

mod cache_aware_rnnt;
mod multi_talker;

pub use cache_aware_rnnt::{CacheAwareRNNTPipeline, TranscribeOutput};
pub use multi_talker::{MultiTalkerOutput, MultiTalkerPipeline, SpeakerSegment};
