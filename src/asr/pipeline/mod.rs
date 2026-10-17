//! Streaming transcription.
//!
//! [`StreamingSession`] transcribes one audio stream incrementally with a
//! cache-aware RNN-T model, mirroring NeMo's cache-aware streaming inference.
//! [`MultiTalkerPipeline`] adds a streaming diarizer to attribute words to
//! speakers.
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! # let chunk: Vec<f32> = Vec::new();
//! use mlx_voice::asr::{EncDecRNNTModel, StreamingOptions};
//!
//! let mut model = EncDecRNNTModel::from_pretrained("model.nemo")?;
//!
//! // A whole file, streamed
//! let transcript = model.transcribe_streaming("audio.wav", StreamingOptions::default())?;
//!
//! // Live audio
//! let mut stream = model.stream(StreamingOptions::default())?;
//! stream.push_audio(&chunk, 16_000)?;
//! while let Some(update) = stream.next_update()? {
//!     print!("{}", update.delta);
//! }
//! let transcript = stream.finish()?;
//! # Ok(())
//! # }
//! ```

mod multi_talker;
mod session;

pub use multi_talker::{
    MultiTalkerConfig, MultiTalkerMode, MultiTalkerOutput, MultiTalkerPipeline, SpeakerSegment,
    SpeakerWord,
};
pub use session::{StreamUpdate, StreamingOptions, StreamingSession};
