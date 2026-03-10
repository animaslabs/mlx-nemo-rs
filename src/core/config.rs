//! Configuration loading and parsing utilities.
//!
//! This module provides traits and utilities for loading model configurations
//! from YAML files, similar to NeMo's Hydra/OmegaConf-based configuration system.

use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::core::error::{Error, Result};

/// Trait for types that can be instantiated from a configuration.
pub trait FromConfig: Sized {
    /// The configuration type.
    type Config: DeserializeOwned + Serialize + Clone;

    /// Create an instance from configuration.
    fn from_config(config: Self::Config) -> Result<Self>;
}

/// Load configuration from a YAML string.
pub fn from_yaml_str<T: DeserializeOwned>(yaml: &str) -> Result<T> {
    serde_yaml::from_str(yaml).map_err(|e| Error::config_with_source("Failed to parse YAML", e))
}

/// Load configuration from a YAML file.
pub fn from_yaml_file<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path)?;
    from_yaml_str(&content)
}

/// Serialize configuration to a YAML string.
pub fn to_yaml_string<T: Serialize>(config: &T) -> Result<String> {
    serde_yaml::to_string(config)
        .map_err(|e| Error::config_with_source("Failed to serialize YAML", e))
}

/// Helper to extract a nested value from a YAML structure.
///
/// This is useful for handling NeMo's `_target_` pattern where the type
/// is specified in the configuration.
pub fn extract_target(config: &serde_yaml::Value) -> Option<&str> {
    config.get("_target_").and_then(|v| v.as_str())
}

/// Get the class name from a NeMo target string.
///
/// For example, `"nemo.collections.asr.modules.ConformerEncoder"` returns `"ConformerEncoder"`.
pub fn target_class_name(target: &str) -> &str {
    target.rsplit('.').next().unwrap_or(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_class_name() {
        assert_eq!(
            target_class_name("nemo.collections.asr.modules.ConformerEncoder"),
            "ConformerEncoder"
        );
        assert_eq!(target_class_name("SimpleClass"), "SimpleClass");
    }
}
