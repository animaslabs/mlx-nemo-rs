//! Checkpoint loading and saving utilities.
//!
//! This module handles reading and writing NeMo `.nemo` checkpoint files,
//! which are TAR archives containing model configuration, weights, and artifacts.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use mlx_rs::Array;
use tar::Archive;

use crate::core::error::{Error, Result};

/// Contents extracted from a .nemo checkpoint file.
#[derive(Debug)]
pub struct NemoCheckpoint {
    /// Raw YAML content of the model configuration.
    pub config_yaml: String,

    /// Model weights as a map from parameter name to array.
    /// Note: These are loaded lazily or from safetensors/pickle format.
    pub weights_path: Option<PathBuf>,

    /// Extracted artifact files (tokenizers, etc.)
    pub artifacts: HashMap<String, Vec<u8>>,

    /// Temporary directory containing extracted files (if any).
    temp_dir: Option<tempfile::TempDir>,
}

impl NemoCheckpoint {
    /// Open and extract a .nemo checkpoint file.
    ///
    /// NeMo files are TAR archives (uncompressed since NeMo 1.7.0).
    /// This function extracts the contents to a temporary directory
    /// and returns the parsed checkpoint.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::checkpoint_at(
                format!("File not found: {}", path.display()),
                path,
            ));
        }

        let file = File::open(path)?;

        // Try to open as uncompressed TAR first (NeMo >= 1.7.0)
        // Fall back to gzip-compressed TAR for older files
        let mut archive = Archive::new(file);

        // Create a temporary directory to extract files
        let temp_dir = tempfile::tempdir()?;
        let temp_path = temp_dir.path();

        // Extract all files
        archive.unpack(temp_path).map_err(|e| {
            // Try gzip fallback
            Error::archive(format!(
                "Failed to extract .nemo archive (tried uncompressed TAR): {}",
                e
            ))
        })?;

        // Find and read model_config.yaml
        let config_yaml = Self::find_and_read_file(temp_path, "model_config.yaml")?;

        // Find weights file (could be .ckpt or .safetensors)
        let weights_path = Self::find_weights_file(temp_path);

        // Collect artifact files (tokenizers, vocab files, etc.)
        let artifacts = Self::collect_artifacts(temp_path)?;

        Ok(Self {
            config_yaml,
            weights_path,
            artifacts,
            temp_dir: Some(temp_dir),
        })
    }

    /// Open a gzip-compressed .nemo file (older format).
    pub fn open_gzip(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::checkpoint_at(
                format!("File not found: {}", path.display()),
                path,
            ));
        }

        let file = File::open(path)?;
        let decoder = flate2::read::GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        let temp_dir = tempfile::tempdir()?;
        let temp_path = temp_dir.path();

        archive
            .unpack(temp_path)
            .map_err(|e| Error::archive(format!("Failed to extract gzip .nemo archive: {}", e)))?;

        let config_yaml = Self::find_and_read_file(temp_path, "model_config.yaml")?;
        let weights_path = Self::find_weights_file(temp_path);
        let artifacts = Self::collect_artifacts(temp_path)?;

        Ok(Self {
            config_yaml,
            weights_path,
            artifacts,
            temp_dir: Some(temp_dir),
        })
    }

    /// Get the path to an extracted artifact file.
    pub fn artifact_path(&self, name: &str) -> Option<PathBuf> {
        self.temp_dir.as_ref().and_then(|dir| {
            let path = dir.path().join(name);
            if path.exists() {
                Some(path)
            } else {
                // Try to find by suffix
                Self::find_file_by_suffix(dir.path(), name).ok()
            }
        })
    }

    /// Get the content of an artifact file.
    pub fn artifact(&self, name: &str) -> Option<&[u8]> {
        self.artifacts.get(name).map(|v| v.as_slice())
    }

    /// Find a tokenizer model file in the artifacts.
    pub fn find_tokenizer(&self) -> Option<PathBuf> {
        // Look for SentencePiece model files
        for name in self.artifacts.keys() {
            if name.ends_with("_tokenizer.model") || name.ends_with(".model") {
                if let Some(dir) = &self.temp_dir {
                    let path = dir.path().join(name);
                    if path.exists() {
                        return Some(path);
                    }
                }
            }
        }

        // Also check the temp directory directly
        if let Some(dir) = &self.temp_dir {
            if let Ok(entries) = std::fs::read_dir(dir.path()) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.ends_with("_tokenizer.model") || name == "tokenizer.model" {
                            return Some(path);
                        }
                    }
                }
            }
        }

        None
    }

    /// Load weights from the checkpoint.
    ///
    /// This loads the weights from either a safetensors or PyTorch checkpoint file.
    pub fn load_weights(&self) -> Result<HashMap<String, Array>> {
        let weights_path = self
            .weights_path
            .as_ref()
            .ok_or_else(|| Error::checkpoint("No weights file found in checkpoint"))?;

        let extension = weights_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension {
            "safetensors" => {
                // Load using mlx-rs safetensors support
                let tensors = Array::load_safetensors(weights_path)?;
                Ok(tensors)
            }
            "ckpt" => {
                // PyTorch checkpoint format - needs conversion
                // For now, return an error suggesting to use safetensors
                Err(Error::not_implemented(
                    "PyTorch .ckpt format not yet supported. Please convert to safetensors format.",
                ))
            }
            _ => Err(Error::checkpoint(format!(
                "Unknown weights format: {}",
                extension
            ))),
        }
    }

    // Helper functions

    fn find_and_read_file(dir: &Path, filename: &str) -> Result<String> {
        let path = Self::find_file_by_suffix(dir, filename)?;
        let mut content = String::new();
        File::open(&path)?.read_to_string(&mut content)?;
        Ok(content)
    }

    fn find_file_by_suffix(dir: &Path, suffix: &str) -> Result<PathBuf> {
        // First, check direct path
        let direct_path = dir.join(suffix);
        if direct_path.exists() {
            return Ok(direct_path);
        }

        // Search recursively
        fn search_dir(dir: &Path, suffix: &str) -> Option<PathBuf> {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(found) = search_dir(&path, suffix) {
                            return Some(found);
                        }
                    } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name == suffix || name.ends_with(suffix) {
                            return Some(path);
                        }
                    }
                }
            }
            None
        }

        search_dir(dir, suffix)
            .ok_or_else(|| Error::checkpoint(format!("File '{}' not found in archive", suffix)))
    }

    fn find_weights_file(dir: &Path) -> Option<PathBuf> {
        // Prefer safetensors over .ckpt
        let safetensors_path = dir.join("model_weights.safetensors");
        if safetensors_path.exists() {
            return Some(safetensors_path);
        }

        let ckpt_path = dir.join("model_weights.ckpt");
        if ckpt_path.exists() {
            return Some(ckpt_path);
        }

        // Search for any weights file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(".safetensors") || name.ends_with(".ckpt") {
                        return Some(path);
                    }
                }
            }
        }

        None
    }

    fn collect_artifacts(dir: &Path) -> Result<HashMap<String, Vec<u8>>> {
        let mut artifacts = HashMap::new();

        fn collect_from_dir(
            dir: &Path,
            base: &Path,
            artifacts: &mut HashMap<String, Vec<u8>>,
        ) -> Result<()> {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        collect_from_dir(&path, base, artifacts)?;
                    } else {
                        let relative = path.strip_prefix(base).unwrap_or(&path);
                        let name = relative.to_string_lossy().to_string();

                        // Skip config and weights files
                        if name == "model_config.yaml"
                            || name.ends_with(".ckpt")
                            || name.ends_with(".safetensors")
                        {
                            continue;
                        }

                        let mut content = Vec::new();
                        File::open(&path)?.read_to_end(&mut content)?;
                        artifacts.insert(name, content);
                    }
                }
            }
            Ok(())
        }

        collect_from_dir(dir, dir, &mut artifacts)?;
        Ok(artifacts)
    }
}

/// Builder for creating .nemo checkpoint files.
pub struct NemoCheckpointWriter {
    config_yaml: Option<String>,
    weights: Option<HashMap<String, Array>>,
    artifacts: HashMap<String, Vec<u8>>,
}

impl NemoCheckpointWriter {
    /// Create a new checkpoint writer.
    pub fn new() -> Self {
        Self {
            config_yaml: None,
            weights: None,
            artifacts: HashMap::new(),
        }
    }

    /// Set the model configuration.
    pub fn config<T: serde::Serialize>(mut self, config: &T) -> Result<Self> {
        self.config_yaml = Some(serde_yaml::to_string(config)?);
        Ok(self)
    }

    /// Set the model configuration from raw YAML.
    pub fn config_yaml(mut self, yaml: String) -> Self {
        self.config_yaml = Some(yaml);
        self
    }

    /// Set the model weights.
    pub fn weights(mut self, weights: HashMap<String, Array>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Add an artifact file.
    pub fn artifact(mut self, name: impl Into<String>, content: Vec<u8>) -> Self {
        self.artifacts.insert(name.into(), content);
        self
    }

    /// Write the checkpoint to a file.
    pub fn write(self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)?;
        let mut archive = tar::Builder::new(file);

        // Write config
        if let Some(config) = &self.config_yaml {
            let data = config.as_bytes();
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, "model_config.yaml", data)?;
        }

        // Write weights as safetensors
        if let Some(weights) = &self.weights {
            let temp_dir = tempfile::tempdir()?;
            let weights_path = temp_dir.path().join("model_weights.safetensors");
            Array::save_safetensors(weights.clone(), None, &weights_path)?;

            let mut weights_file = File::open(&weights_path)?;
            archive.append_file("model_weights.safetensors", &mut weights_file)?;
        }

        // Write artifacts
        for (name, content) in &self.artifacts {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, content.as_slice())?;
        }

        archive.finish()?;
        Ok(())
    }
}

impl Default for NemoCheckpointWriter {
    fn default() -> Self {
        Self::new()
    }
}
