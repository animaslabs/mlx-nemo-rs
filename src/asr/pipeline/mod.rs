//! Streaming ASR pipelines.
//!
//! This module provides high-level pipeline abstractions for streaming
//! ASR inference, mirroring NeMo's `CacheAwareRNNTPipeline`.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # let chunk1: Vec<f32> = Vec::new();
//! use mlx_voice::asr::pipeline::CacheAwareRNNTPipeline;
//!
//! // Load model and create pipeline
//! let mut pipeline = CacheAwareRNNTPipeline::from_pretrained("model.nemo", 0)?;
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
//! # Ok(())
//! # }
//! ```

mod cache_aware_rnnt;
mod multi_talker;

pub use cache_aware_rnnt::{CacheAwareRNNTPipeline, TranscribeOutput};
pub use multi_talker::{
    MultiTalkerConfig, MultiTalkerMode, MultiTalkerOutput, MultiTalkerPipeline, SpeakerSegment,
    SpeakerWord,
};
