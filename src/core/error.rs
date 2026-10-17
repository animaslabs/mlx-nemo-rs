//! Error types for mlx-voice.

use std::path::{Path, PathBuf};
use thiserror::Error;

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for mlx-voice operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing configuration files.
    #[error("Configuration error: {message}")]
    Config {
        /// What went wrong
        message: String,
        /// The underlying error, if any
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Error loading or saving checkpoints.
    #[error("Checkpoint error: {message}")]
    Checkpoint {
        /// What went wrong
        message: String,
        /// The checkpoint file or directory, if known
        path: Option<PathBuf>,
    },

    /// Model-related errors.
    #[error("Model error: {0}")]
    Model(String),

    /// Tensor shape mismatch.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<i32>,
        /// Actual shape
        actual: Vec<i32>,
    },

    /// Type mismatch error.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type
        expected: String,
        /// Actual type
        actual: String,
    },

    /// MLX operation error.
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// MLX I/O error.
    #[error("MLX I/O error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),

    /// Missing required field in configuration or data.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Tokenizer-related errors.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Audio processing errors.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Decoding errors during inference.
    #[error("Decoding error: {0}")]
    Decoding(String),

    /// Archive (TAR) error.
    #[error("Archive error: {0}")]
    Archive(String),

    /// Pickle parsing error.
    #[error("Pickle error: {0}")]
    Pickle(String),

    /// Feature not implemented.
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl Error {
    /// Create a configuration error with a message.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }

    /// Create a configuration error with a message and source error.
    pub fn config_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Config {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a checkpoint error.
    pub fn checkpoint(message: impl Into<String>) -> Self {
        Self::Checkpoint {
            message: message.into(),
            path: None,
        }
    }

    /// Create a checkpoint error with a path.
    pub fn checkpoint_at(message: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self::Checkpoint {
            message: message.into(),
            path: Some(path.into()),
        }
    }

    /// Create a checkpoint error for a weight tensor the model needs but the
    /// checkpoint lacks.
    pub(crate) fn missing_weight(key: &str) -> Self {
        Self::checkpoint(format!("Missing weight `{key}`"))
    }

    /// Create a checkpoint error for a checkpoint or model directory without a
    /// SentencePiece tokenizer.
    pub(crate) fn missing_tokenizer(path: &Path) -> Self {
        Self::checkpoint_at(
            format!("No SentencePiece tokenizer found in {}", path.display()),
            path,
        )
    }

    /// Create a model error.
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model(message.into())
    }

    /// Create a tokenizer error.
    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer(message.into())
    }

    /// Create an audio processing error.
    pub fn audio(message: impl Into<String>) -> Self {
        Self::Audio(message.into())
    }

    /// Create a decoding error.
    pub fn decoding(message: impl Into<String>) -> Self {
        Self::Decoding(message.into())
    }

    /// Create an archive error.
    pub fn archive(message: impl Into<String>) -> Self {
        Self::Archive(message.into())
    }

    /// Create a pickle parsing error.
    pub fn pickle(message: impl Into<String>) -> Self {
        Self::Pickle(message.into())
    }

    /// Create a not-implemented error.
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented(feature.into())
    }
}
