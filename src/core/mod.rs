//! Core abstractions and utilities for mlx-nemo.
//!
//! This module provides the foundational types and traits used across
//! all collections (ASR, TTS, LLM, etc.).

pub mod checkpoint;
pub mod config;
pub mod error;

pub use error::{Error, Result};
