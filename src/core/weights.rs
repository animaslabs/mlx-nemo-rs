//! Weight format handling and transformation utilities.
//!
//! This module provides utilities for handling different weight formats
//! (PyTorch vs MLX) and transforming weights between them.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use crate::core::error::{Error, Result};

/// Weight format indicating how tensor data is laid out.
///
/// Different frameworks use different conventions for tensor layouts.
/// This enum allows us to handle weights from various sources correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WeightFormat {
    /// PyTorch format - Conv1d weights are (O, I, K) and need transposition to (O, K, I).
    #[default]
    PyTorch,
    /// MLX format - weights are already in the correct layout for MLX operations.
    Mlx,
}

impl WeightFormat {
    /// Detect weight format from safetensors metadata.
    ///
    /// Looks for a "format" key in the metadata. If it contains "mlx" (case-insensitive),
    /// returns `WeightFormat::Mlx`. Otherwise, defaults to `WeightFormat::PyTorch`.
    pub fn from_safetensors_metadata(metadata: &HashMap<String, String>) -> Self {
        if let Some(format) = metadata.get("format") {
            if format.to_lowercase().contains("mlx") {
                return Self::Mlx;
            }
        }
        // Default to PyTorch format (most common)
        Self::PyTorch
    }

    /// Detect weight format by analyzing Conv2d weight shapes.
    ///
    /// For NeMo models, the first subsampling conv has 1 input channel:
    /// - PyTorch format: (O, I, H, W) = (256, 1, 3, 3) -> shape[1] = 1
    /// - MLX format: (O, H, W, I) = (256, 3, 3, 1) -> shape[3] = 1
    ///
    /// This heuristic looks for Conv2d weights where one dimension is 1 (the input channel)
    /// and uses its position to infer the format.
    pub fn detect_from_weights(weights: &HashMap<String, Array>) -> Self {
        // Look for the first subsampling conv (most reliable indicator)
        // In NeMo: encoder.pre_encode.conv.0.weight
        for (name, weight) in weights {
            let name_lower = name.to_lowercase();

            // Look for first conv2d weight (4D tensor with "conv" in name)
            if weight.ndim() == 4
                && name_lower.contains("conv")
                && name_lower.ends_with(".weight")
                && (name_lower.contains("pre_encode") || name_lower.contains("subsampling"))
            {
                let shape = weight.shape();
                // For first conv: in_channels = 1
                // PyTorch: (O, 1, H, W) -> shape[1] = 1
                // MLX: (O, H, W, 1) -> shape[3] = 1
                if shape[1] == 1 && shape[2] > 1 && shape[3] > 1 {
                    tracing::debug!(
                        "Detected PyTorch format from {}: shape {:?} (in_channels at dim 1)",
                        name,
                        shape
                    );
                    return Self::PyTorch;
                }
                if shape[3] == 1 && shape[1] > 1 && shape[2] > 1 {
                    tracing::debug!(
                        "Detected MLX format from {}: shape {:?} (in_channels at dim 3)",
                        name,
                        shape
                    );
                    return Self::Mlx;
                }
            }
        }

        // Fallback: check any Conv2d weight with asymmetric dimensions
        for (name, weight) in weights {
            let name_lower = name.to_lowercase();
            if weight.ndim() == 4 && name_lower.contains("conv") && name_lower.ends_with(".weight")
            {
                let shape = weight.shape();
                // For pointwise (1x1) conv: kernel = 1x1
                // PyTorch: (O, I, 1, 1)
                // MLX: (O, 1, 1, I)
                // If dim 1 and 2 are both 1, it's MLX format
                if shape[1] == 1 && shape[2] == 1 && shape[3] > 1 {
                    tracing::debug!(
                        "Detected MLX format from {}: shape {:?} (pointwise conv)",
                        name,
                        shape
                    );
                    return Self::Mlx;
                }
                // If dim 2 and 3 are both 1, it's PyTorch format
                if shape[2] == 1 && shape[3] == 1 && shape[1] > 1 {
                    tracing::debug!(
                        "Detected PyTorch format from {}: shape {:?} (pointwise conv)",
                        name,
                        shape
                    );
                    return Self::PyTorch;
                }
            }
        }

        // Default to PyTorch if we can't determine
        tracing::debug!("Could not detect format from weight shapes, defaulting to PyTorch");
        Self::PyTorch
    }

    /// Check if this format requires Conv1d weight transposition.
    pub fn needs_conv1d_transpose(&self) -> bool {
        matches!(self, Self::PyTorch)
    }
}

/// Transform weights from PyTorch format to MLX format.
///
/// This function transforms weights at load time based on their names:
/// - Conv1d weights (*.conv*.weight with 3 dims): transpose from (O, I, K) to (O, K, I)
/// - Conv2d weights (*.conv*.weight with 4 dims): transpose from (O, I, H, W) to (O, H, W, I)
/// - Other weights: unchanged
///
/// # Arguments
/// * `weights` - HashMap of weight name to Array
///
/// # Returns
/// * Transformed weights HashMap
pub fn transform_weights_to_mlx(weights: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut transformed = HashMap::with_capacity(weights.len());

    for (name, weight) in weights {
        let transformed_weight = if should_transpose_conv1d(&name, &weight) {
            // Transpose Conv1d weight from PyTorch (O, I, K) to MLX (O, K, I)
            weight.transpose_axes(&[0, 2, 1]).map_err(|e| {
                Error::checkpoint(format!(
                    "Failed to transpose Conv1d weight '{}': {}",
                    name, e
                ))
            })?
        } else if should_transpose_conv2d(&name, &weight) {
            // Transpose Conv2d weight from PyTorch (O, I, H, W) to MLX (O, H, W, I)
            weight.transpose_axes(&[0, 2, 3, 1]).map_err(|e| {
                Error::checkpoint(format!(
                    "Failed to transpose Conv2d weight '{}': {}",
                    name, e
                ))
            })?
        } else {
            weight
        };
        transformed.insert(name, transformed_weight);
    }

    Ok(transformed)
}

/// Check if a weight tensor should be transposed as a Conv1d weight.
///
/// Returns true if:
/// - The name contains "conv" (case-insensitive) and ends with ".weight"
/// - OR the name ends with "_conv.weight" or "pointwise_conv.weight" etc.
/// - AND the tensor has exactly 3 dimensions
fn should_transpose_conv1d(name: &str, weight: &Array) -> bool {
    let name_lower = name.to_lowercase();
    let is_conv_weight = name_lower.ends_with(".weight")
        && (name_lower.contains("conv")
            || name_lower.contains("depthwise")
            || name_lower.contains("pointwise"));
    let has_3_dims = weight.ndim() == 3;

    is_conv_weight && has_3_dims
}

/// Check if a weight tensor should be transposed as a Conv2d weight.
///
/// Returns true if:
/// - The name contains "conv" (case-insensitive) and ends with ".weight"
/// - AND the tensor has exactly 4 dimensions
fn should_transpose_conv2d(name: &str, weight: &Array) -> bool {
    let name_lower = name.to_lowercase();
    let is_conv_weight = name_lower.ends_with(".weight")
        && (name_lower.contains("conv")
            || name_lower.contains("depthwise")
            || name_lower.contains("pointwise"));
    let has_4_dims = weight.ndim() == 4;

    is_conv_weight && has_4_dims
}

/// Load safetensors weights from a file with optional format detection.
///
/// # Arguments
/// * `path` - Path to the safetensors file
/// * `format` - Optional weight format. If None, attempts to detect from metadata.
///
/// # Returns
/// * Loaded and potentially transformed weights
pub fn load_safetensors_weights(
    path: impl AsRef<Path>,
    format: Option<WeightFormat>,
) -> Result<HashMap<String, Array>> {
    let path = path.as_ref();

    // Load weights using mlx-rs
    let weights = Array::load_safetensors(path)?;

    // Determine format
    let format = format.unwrap_or_else(|| {
        // Try to detect from metadata first
        if let Ok(metadata) = read_safetensors_metadata(path) {
            let from_metadata = WeightFormat::from_safetensors_metadata(&metadata);
            if from_metadata == WeightFormat::Mlx {
                return from_metadata;
            }
        }
        // Fall back to shape-based detection
        WeightFormat::detect_from_weights(&weights)
    });

    tracing::info!("Using weight format: {:?}", format);

    // Transform if needed
    if format.needs_conv1d_transpose() {
        transform_weights_to_mlx(weights)
    } else {
        Ok(weights)
    }
}

/// Read metadata from a safetensors file.
///
/// Safetensors format:
/// - First 8 bytes: little-endian u64 header size
/// - Next N bytes: JSON header containing tensor info and optional "__metadata__"
pub fn read_safetensors_metadata(path: impl AsRef<Path>) -> Result<HashMap<String, String>> {
    use std::fs::File;
    use std::io::Read;

    let path = path.as_ref();
    let mut file = File::open(path)?;

    // Read header size (8 bytes, little-endian u64)
    let mut size_bytes = [0u8; 8];
    file.read_exact(&mut size_bytes)?;
    let header_size = u64::from_le_bytes(size_bytes) as usize;

    // Sanity check header size
    if header_size > 100_000_000 {
        return Err(Error::checkpoint(format!(
            "Safetensors header size too large: {} bytes",
            header_size
        )));
    }

    // Read header JSON
    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes)?;

    // Parse as JSON
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| Error::checkpoint(format!("Invalid safetensors header: {}", e)))?;

    // Extract metadata if present
    if let Some(metadata) = header.get("__metadata__") {
        if let Some(obj) = metadata.as_object() {
            let mut result = HashMap::new();
            for (key, value) in obj {
                if let Some(s) = value.as_str() {
                    result.insert(key.clone(), s.to_string());
                }
            }
            return Ok(result);
        }
    }

    Ok(HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_format_default() {
        assert_eq!(WeightFormat::default(), WeightFormat::PyTorch);
    }

    #[test]
    fn test_weight_format_from_metadata() {
        let mut metadata = HashMap::new();
        assert_eq!(
            WeightFormat::from_safetensors_metadata(&metadata),
            WeightFormat::PyTorch
        );

        metadata.insert("format".to_string(), "mlx".to_string());
        assert_eq!(
            WeightFormat::from_safetensors_metadata(&metadata),
            WeightFormat::Mlx
        );

        metadata.insert("format".to_string(), "MLX".to_string());
        assert_eq!(
            WeightFormat::from_safetensors_metadata(&metadata),
            WeightFormat::Mlx
        );

        metadata.insert("format".to_string(), "pytorch".to_string());
        assert_eq!(
            WeightFormat::from_safetensors_metadata(&metadata),
            WeightFormat::PyTorch
        );
    }

    #[test]
    fn test_needs_conv1d_transpose() {
        assert!(WeightFormat::PyTorch.needs_conv1d_transpose());
        assert!(!WeightFormat::Mlx.needs_conv1d_transpose());
    }

    #[test]
    fn test_should_transpose_conv1d() {
        // 3D tensor that should be transposed
        let weight_3d = Array::from_slice(&[0.0f32; 24], &[2, 3, 4]);

        assert!(should_transpose_conv1d("encoder.conv.weight", &weight_3d));
        assert!(should_transpose_conv1d("encoder.conv1.weight", &weight_3d));
        assert!(should_transpose_conv1d("depthwise_conv.weight", &weight_3d));
        assert!(should_transpose_conv1d("pointwise_conv.weight", &weight_3d));
        assert!(should_transpose_conv1d("CONV.weight", &weight_3d)); // Case insensitive

        // Should NOT transpose
        assert!(!should_transpose_conv1d(
            "encoder.linear.weight",
            &weight_3d
        )); // Not conv
        assert!(!should_transpose_conv1d("encoder.conv.bias", &weight_3d)); // Not .weight

        // 2D tensor (e.g., Linear) should NOT be transposed even if named conv
        let weight_2d = Array::from_slice(&[0.0f32; 12], &[3, 4]);
        assert!(!should_transpose_conv1d("encoder.conv.weight", &weight_2d));

        // 4D tensor (e.g., Conv2d) should NOT be transposed by this function
        let weight_4d = Array::from_slice(&[0.0f32; 48], &[2, 3, 4, 2]);
        assert!(!should_transpose_conv1d(
            "encoder.conv2d.weight",
            &weight_4d
        ));
    }

    #[test]
    fn test_should_transpose_conv2d() {
        // 4D tensor that should be transposed
        let weight_4d = Array::from_slice(&[0.0f32; 48], &[2, 3, 4, 2]);

        assert!(should_transpose_conv2d("encoder.conv.weight", &weight_4d));
        assert!(should_transpose_conv2d("encoder.conv2d.weight", &weight_4d));
        assert!(should_transpose_conv2d("depthwise_conv.weight", &weight_4d));
        assert!(should_transpose_conv2d("pointwise_conv.weight", &weight_4d));
        assert!(should_transpose_conv2d("CONV.weight", &weight_4d)); // Case insensitive

        // Should NOT transpose
        assert!(!should_transpose_conv2d(
            "encoder.linear.weight",
            &weight_4d
        )); // Not conv
        assert!(!should_transpose_conv2d("encoder.conv.bias", &weight_4d)); // Not .weight

        // 3D tensor (e.g., Conv1d) should NOT be transposed by this function
        let weight_3d = Array::from_slice(&[0.0f32; 24], &[2, 3, 4]);
        assert!(!should_transpose_conv2d("encoder.conv.weight", &weight_3d));

        // 2D tensor (e.g., Linear) should NOT be transposed
        let weight_2d = Array::from_slice(&[0.0f32; 12], &[3, 4]);
        assert!(!should_transpose_conv2d("encoder.conv.weight", &weight_2d));
    }
}
