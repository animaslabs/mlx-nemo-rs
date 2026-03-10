//! Error types for mlx-nemo.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for mlx-nemo operations.
#[derive(Debug, Error)]
pub enum Error {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing configuration files.
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Error loading or saving checkpoints.
    #[error("Checkpoint error: {message}")]
    Checkpoint {
        message: String,
        path: Option<PathBuf>,
    },

    /// Model-related errors.
    #[error("Model error: {0}")]
    Model(String),

    /// Tensor shape mismatch.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    /// Type mismatch error.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

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

    /// YAML parsing error.
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Archive (TAR) error.
    #[error("Archive error: {0}")]
    Archive(String),

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

    /// Create a not-implemented error.
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented(feature.into())
    }
}
