//! # mlx-voice
//!
//! A Rust implementation of NVIDIA NeMo for Apple Silicon, built on MLX.
//!
//! It loads NeMo speech recognition checkpoints (`.nemo` archives or MLX
//! safetensors) and runs them on the GPU through Metal: offline and
//! cache-aware streaming transcription, voice activity detection and
//! speaker diarization. Text-to-speech is planned.
//!
//! ## Quick Start
//!
//! ```no_run
//! # fn main() -> mlx_voice::Result<()> {
//! use mlx_voice::asr::EncDecCtcModel;
//!
//! // Load a pretrained model from a .nemo file
//! let mut model = EncDecCtcModel::from_pretrained("path/to/model.nemo")?;
//!
//! // Transcribe audio
//! let transcriptions = model.transcribe(&["audio.wav"])?;
//! println!("{}", transcriptions[0]);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs, missing_debug_implementations)]

/// Declare a module that is public with the `internals` feature and
/// crate-private otherwise.
///
/// Internal modules hold building blocks (encoder layers, decoders, streaming
/// caches, checkpoint parsing) whose API may change in any release.
macro_rules! internal_mod {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[cfg(feature = "internals")]
        pub mod $name;
        $(#[$attr])*
        #[cfg(not(feature = "internals"))]
        #[allow(dead_code, unused_imports)]
        pub(crate) mod $name;
    };
}

/// Declare a method that is public with the `internals` feature and
/// crate-private otherwise, e.g. a model's low-level constructor or forward
/// pass.
macro_rules! internal_fn {
    ($(#[$attr:meta])* pub fn $($rest:tt)*) => {
        $(#[$attr])*
        #[cfg(feature = "internals")]
        pub fn $($rest)*

        $(#[$attr])*
        #[cfg(not(feature = "internals"))]
        #[allow(dead_code)]
        pub(crate) fn $($rest)*
    };
}

pub mod core;

#[cfg(feature = "asr")]
pub mod asr;

pub mod audio;

// Re-export commonly used types
pub use core::error::{Error, Result};
pub use core::loading::{LoadOptions, LoadReport};
pub use core::source::{HubRepo, ModelSource};
#[doc(no_inline)]
pub use mlx_rs::Array;

/// The `mlx-rs` version this crate is built against, for working with [`Array`]
/// directly (e.g. `mlx_voice::mlx_rs::ops`) without a separate, possibly
/// mismatched, `mlx-rs` dependency.
#[doc(no_inline)]
pub use mlx_rs;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::core::error::{Error, Result};

    #[cfg(feature = "asr")]
    pub use crate::asr::prelude::*;
}
