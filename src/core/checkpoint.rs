//! Checkpoint loading and saving utilities.
//!
//! This module handles reading and writing NeMo `.nemo` checkpoint files,
//! which are TAR archives containing model configuration, weights, and artifacts.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use mlx_rs::Array;
use tar::Archive;

use crate::core::error::{Error, Result};

/// Contents extracted from a .nemo checkpoint file.
#[derive(Debug)]
pub struct NemoCheckpoint {
    /// Raw YAML content of the model configuration.
    config_yaml: String,

    /// Model weights as a map from parameter name to array.
    /// Note: These are loaded lazily or from safetensors/pickle format.
    weights_path: Option<PathBuf>,

    /// Extracted artifact files (tokenizers, etc.)
    artifacts: HashMap<String, Vec<u8>>,

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

        let mut file = File::open(path)?;

        // Most .nemo files are uncompressed TAR (NeMo >= 1.7.0), but some
        // (e.g. nemotron-3.5-asr-streaming-0.6b) are still gzip-compressed
        let mut magic = [0u8; 2];
        let is_gzip = file.read_exact(&mut magic).is_ok() && magic == [0x1f, 0x8b];
        if is_gzip {
            return Self::open_gzip(path);
        }
        file.seek(SeekFrom::Start(0))?;
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
    fn open_gzip(path: impl AsRef<Path>) -> Result<Self> {
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

    /// Raw YAML content of the model configuration.
    pub fn config_yaml(&self) -> &str {
        &self.config_yaml
    }

    /// Path of the extracted weights file, if the archive has one.
    pub fn weights_path(&self) -> Option<&Path> {
        self.weights_path.as_deref()
    }

    /// Names of the extracted artifact files (tokenizers, vocabularies, etc.).
    pub fn artifact_names(&self) -> impl Iterator<Item = &str> {
        self.artifacts.keys().map(String::as_str)
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

    /// Load the model configuration from the checkpoint.
    ///
    /// This parses the config_yaml into the specified type.
    pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        yaml_serde::from_str(&self.config_yaml)
            .map_err(|e| Error::config(format!("Failed to parse model config: {}", e)))
    }

    /// Load weights from the checkpoint.
    ///
    /// This loads the weights from either a safetensors or PyTorch checkpoint file.
    /// Weights are transformed from PyTorch format to MLX format (Conv1d/Conv2d transposition).
    pub fn load_weights(&self) -> Result<HashMap<String, Array>> {
        let weights_path = self
            .weights_path
            .as_ref()
            .ok_or_else(|| Error::checkpoint("No weights file found in checkpoint"))?;

        let extension = weights_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let raw_weights = match extension {
            "safetensors" => {
                // Load using mlx-rs safetensors support
                Array::load_safetensors(weights_path)?
            }
            "ckpt" => {
                // Load using PyTorch checkpoint loader
                crate::core::pytorch::load(weights_path)?
            }
            _ => {
                return Err(Error::checkpoint(format!(
                    "Unknown weights format: '{}'. Supported: .safetensors, .ckpt",
                    extension
                )));
            }
        };

        // Transform weights from PyTorch format to MLX format
        // .nemo files always contain PyTorch-format weights
        crate::core::weights::transform_weights_to_mlx(raw_weights)
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
#[derive(Debug)]
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
        self.config_yaml = Some(crate::core::config::to_yaml_string(config)?);
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

/// Loader for standalone safetensors files (not within .nemo archives).
///
/// This loader supports loading models from directory structures commonly
/// used by HuggingFace and mlx-community:
///
/// ```text
/// my_model/
/// ├── model.safetensors          (or weights.safetensors, model_weights.safetensors)
/// ├── config.yaml                (or config.json, model_config.yaml)
/// └── tokenizer.model            (or *.spm, sentencepiece.bpe.model)
/// ```
///
/// # Example
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// # use mlx_voice::SafetensorsLoader;
/// # use mlx_voice::asr::AsrModelConfig;
/// // Load from explicit paths
/// let loader = SafetensorsLoader::new(
///     "weights.safetensors",
///     "config.yaml",
///     Some("tokenizer.model"),
/// )?;
///
/// // Load from directory (auto-detect files)
/// let loader = SafetensorsLoader::from_directory("my_model/")?;
///
/// // Access components
/// let weights = loader.load_weights()?;
/// let config: AsrModelConfig = loader.load_config()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SafetensorsLoader {
    /// Path to the safetensors weights file.
    weights_path: PathBuf,

    /// Raw config content (YAML or JSON).
    config_content: String,

    /// Path to tokenizer file, if found.
    tokenizer_path: Option<PathBuf>,

    /// Detected or specified weight format.
    /// If None, format will be auto-detected from weight shapes during loading.
    weight_format: Option<crate::core::weights::WeightFormat>,

    /// Quantization configuration, if the model is quantized.
    quantization: Option<crate::core::quantization::QuantizationConfig>,
}

impl SafetensorsLoader {
    /// Create a loader from explicit file paths.
    ///
    /// # Arguments
    /// * `weights_path` - Path to the safetensors file
    /// * `config_path` - Path to the config file (YAML or JSON)
    /// * `tokenizer_path` - Optional path to tokenizer file
    pub fn new(
        weights_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        tokenizer_path: Option<impl AsRef<Path>>,
    ) -> Result<Self> {
        let weights_path = weights_path.as_ref().to_path_buf();
        let config_path = config_path.as_ref();

        if !weights_path.exists() {
            return Err(Error::checkpoint_at(
                format!("Weights file not found: {}", weights_path.display()),
                &weights_path,
            ));
        }

        if !config_path.exists() {
            return Err(Error::checkpoint_at(
                format!("Config file not found: {}", config_path.display()),
                config_path,
            ));
        }

        // Read config content
        let config_content = std::fs::read_to_string(config_path)?;

        // Try to detect weight format from safetensors metadata
        // If metadata indicates MLX, use that. Otherwise, leave as None for shape-based detection.
        let weight_format = crate::core::weights::read_safetensors_metadata(&weights_path)
            .ok()
            .and_then(|m| {
                let fmt = crate::core::weights::WeightFormat::from_safetensors_metadata(&m);
                if fmt == crate::core::weights::WeightFormat::Mlx {
                    Some(fmt)
                } else {
                    // Don't default to PyTorch; let shape-based detection decide
                    None
                }
            });

        let tokenizer_path = tokenizer_path.map(|p| p.as_ref().to_path_buf());

        // Try to parse quantization config from JSON config
        let quantization = Self::parse_quantization_config(&config_content);

        Ok(Self {
            weights_path,
            config_content,
            tokenizer_path,
            weight_format,
            quantization,
        })
    }

    /// Parse quantization configuration from config content.
    ///
    /// Looks for `"quantization": { "bits": N, "group_size": M }` in JSON config.
    fn parse_quantization_config(
        config_content: &str,
    ) -> Option<crate::core::quantization::QuantizationConfig> {
        // Try parsing as JSON first
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(config_content) {
            if let Some(quant) = json.get("quantization") {
                let bits = quant.get("bits").and_then(|v| v.as_i64()).unwrap_or(8) as i32;
                let group_size = quant
                    .get("group_size")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(64) as i32;

                tracing::info!(bits, group_size, "detected quantized model");
                return Some(crate::core::quantization::QuantizationConfig::new(
                    bits, group_size,
                ));
            }
        }

        // Try parsing as YAML
        if let Ok(yaml) = yaml_serde::from_str::<yaml_serde::Value>(config_content) {
            if let Some(quant) = yaml.get("quantization") {
                let bits = quant.get("bits").and_then(|v| v.as_i64()).unwrap_or(8) as i32;
                let group_size = quant
                    .get("group_size")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(64) as i32;

                tracing::info!(bits, group_size, "detected quantized model");
                return Some(crate::core::quantization::QuantizationConfig::new(
                    bits, group_size,
                ));
            }
        }

        None
    }

    /// Create a loader from explicit paths with a specific weight format.
    pub fn with_format(
        weights_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        tokenizer_path: Option<impl AsRef<Path>>,
        format: crate::core::weights::WeightFormat,
    ) -> Result<Self> {
        let mut loader = Self::new(weights_path, config_path, tokenizer_path)?;
        loader.weight_format = Some(format);
        Ok(loader)
    }

    /// Load from a directory, auto-detecting file locations.
    ///
    /// Searches for:
    /// - Weights: model.safetensors, weights.safetensors, model_weights.safetensors
    /// - Config: config.yaml, config.json, model_config.yaml, model_config.json
    /// - Tokenizer: *.model, *.spm, tokenizer.model, sentencepiece.bpe.model
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        if !dir.is_dir() {
            return Err(Error::checkpoint_at(
                format!("Not a directory: {}", dir.display()),
                dir,
            ));
        }

        // Find weights file
        let weights_path = Self::find_weights_in_dir(dir)?;

        // Find config file
        let config_path = Self::find_config_in_dir(dir)?;

        // Find tokenizer (optional)
        let tokenizer_path = Self::find_tokenizer_in_dir(dir);

        let mut loader = Self::new(weights_path, config_path, tokenizer_path)?;

        // If quantization wasn't found in the main config (e.g., model_config.yaml),
        // also check config.json which is where MLX stores quantization info
        if loader.quantization.is_none() {
            let config_json = dir.join("config.json");
            if config_json.exists() {
                if let Ok(content) = std::fs::read_to_string(&config_json) {
                    loader.quantization = Self::parse_quantization_config(&content);
                }
            }
        }

        Ok(loader)
    }

    /// Load weights from the safetensors file.
    ///
    /// Applies weight transformations based on the detected/specified format.
    /// If format is None, uses shape-based detection to determine the format.
    pub fn load_weights(&self) -> Result<HashMap<String, Array>> {
        crate::core::weights::load_safetensors_weights(&self.weights_path, self.weight_format)
    }

    /// Load weights without any transformation.
    ///
    /// Use this if you want to handle format conversion yourself.
    pub fn load_weights_raw(&self) -> Result<HashMap<String, Array>> {
        Array::load_safetensors(&self.weights_path).map_err(|e| e.into())
    }

    /// Load and parse the model configuration.
    ///
    /// Automatically detects YAML vs JSON format.
    pub fn load_config<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        // Try YAML first, then JSON
        if let Ok(config) = yaml_serde::from_str(&self.config_content) {
            return Ok(config);
        }

        serde_json::from_str(&self.config_content).map_err(|e| {
            Error::config(format!(
                "Failed to parse config (tried YAML and JSON): {}",
                e
            ))
        })
    }

    /// Path to the safetensors weights file.
    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }

    /// The weight layout, if known (from metadata or set explicitly).
    pub fn weight_format(&self) -> Option<crate::core::weights::WeightFormat> {
        self.weight_format
    }

    /// Get the raw config content.
    pub fn config_content(&self) -> &str {
        &self.config_content
    }

    /// Get the tokenizer path if found.
    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer_path.as_deref()
    }

    /// Set the weight format explicitly.
    pub fn set_weight_format(&mut self, format: crate::core::weights::WeightFormat) {
        self.weight_format = Some(format);
    }

    /// Get the quantization config if the model is quantized.
    pub fn quantization(&self) -> Option<&crate::core::quantization::QuantizationConfig> {
        self.quantization.as_ref()
    }

    /// Check if this model is quantized.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }

    // Helper: Find weights file in directory
    fn find_weights_in_dir(dir: &Path) -> Result<PathBuf> {
        // Priority order for weights files
        let candidates = [
            "model.safetensors",
            "weights.safetensors",
            "model_weights.safetensors",
        ];

        for name in candidates {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }

        // Try subdirectory patterns (e.g., weights/model.safetensors)
        let subdir_path = dir.join("weights").join("model.safetensors");
        if subdir_path.exists() {
            return Ok(subdir_path);
        }

        // Search for any .safetensors file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        return Ok(path);
                    }
                }
            }
        }

        Err(Error::checkpoint_at(
            "No safetensors file found in directory",
            dir,
        ))
    }

    // Helper: Find config file in directory
    fn find_config_in_dir(dir: &Path) -> Result<PathBuf> {
        // Priority order for config files
        let candidates = [
            "config.yaml",
            "model_config.yaml",
            "config.json",
            "model_config.json",
        ];

        for name in candidates {
            let path = dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(Error::checkpoint_at(
            "No config file found in directory",
            dir,
        ))
    }

    // Helper: Find tokenizer file in directory
    fn find_tokenizer_in_dir(dir: &Path) -> Option<PathBuf> {
        // Priority order for tokenizer files
        let candidates = [
            "tokenizer.model",
            "tokenizer.spm",
            "sentencepiece.bpe.model",
        ];

        for name in candidates {
            let path = dir.join(name);
            if path.exists() {
                return Some(path);
            }
        }

        // Search for any .model or .spm file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.ends_with(".model") || name.ends_with(".spm") {
                        return Some(path);
                    }
                }
            }
        }

        None
    }
}
