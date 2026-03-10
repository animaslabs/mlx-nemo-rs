//! Example usage and basic tests for mlx-nemo-rs.

use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test that the library compiles and basic imports work
        assert!(true);
    }

    #[test]
    fn test_error_handling() {
        use mlx_nemo::core::error::{Error, Result};

        // Test error creation
        let error = Error::model("Test error");
        assert!(error.to_string().contains("Model error"));

        // Test result type
        let result: Result<()> = Err(Error::not_implemented("Test"));
        assert!(result.is_err());
    }

    #[test]
    fn test_config_parsing() {
        use mlx_nemo::core::config::{extract_target, from_yaml_str};

        let yaml = r#"
_target_: nemo.collections.asr.models.EncDecCTCModel
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  n_layers: 16
"#;

        // Test YAML parsing
        let value: serde_yaml::Value = from_yaml_str(yaml).unwrap();
        assert!(extract_target(&value).is_some());

        // Test target extraction
        let target = extract_target(&value).unwrap();
        assert!(target.contains("EncDecCTCModel"));
    }
}

/// Example of loading and using an ASR model.
pub fn example_asr_usage() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "asr")]
    {
        use mlx_nemo::asr::EncDecCTCModel;

        // Load a model from a .nemo file
        let model = EncDecCTCModel::restore("path/to/model.nemo")?;

        // Transcribe audio files
        let audio_files = vec!["audio1.wav", "audio2.wav"];
        let transcriptions = model.transcribe(&audio_files)?;

        // Print results
        for (file, text) in audio_files.iter().zip(transcriptions.iter()) {
            println!("{}: {}", file, text);
        }
    }

    #[cfg(not(feature = "asr"))]
    {
        println!("ASR feature not enabled");
    }

    Ok(())
}

/// Example of MLX operations.
pub fn example_mlx_operations() -> Result<(), Box<dyn std::error::Error>> {
    use mlx_nemo::mlx::{self, nn, tensor};

    // Initialize MLX
    mlx::init()?;

    // Create tensors
    let a = tensor::random(&[2, 3], -1.0, 1.0)?;
    let b = tensor::ones(&[2, 3])?;

    // Basic operations
    let c = mlx_rs::add(&a, &b)?;
    println!("Tensor shape: {:?}", c.shape());

    // Neural network layer
    let layer = nn::Linear::new(3, 5, true)?;
    let output = layer.forward(&c)?;
    println!("Linear layer output shape: {:?}", output.shape());

    // Activation functions
    let relu_output = nn::relu(&output)?;
    let softmax_output = nn::softmax(&relu_output, -1)?;

    println!("Final output shape: {:?}", softmax_output.shape());

    Ok(())
}

/// Example of checkpoint operations.
pub fn example_checkpoint_operations() -> Result<(), Box<dyn std::error::Error>> {
    use mlx_nemo::core::checkpoint::{NemoCheckpoint, NemoCheckpointWriter};
    use mlx_nemo::mlx::tensor;

    // Create a simple checkpoint
    let mut weights = std::collections::HashMap::new();
    weights.insert("weight1".to_string(), tensor::random(&[10, 5], -0.1, 0.1)?);
    weights.insert("bias1".to_string(), tensor::zeros(&[5])?);

    // Write checkpoint
    let writer = NemoCheckpointWriter::new()
        .config(&serde_json::json!({
            "model_type": "test",
            "version": "1.0"
        }))?
        .weights(weights);

    writer.write("test_model.nemo")?;
    println!("Checkpoint written to test_model.nemo");

    // Read checkpoint back
    let checkpoint = NemoCheckpoint::open("test_model.nemo")?;
    println!("Loaded checkpoint with config: {}", checkpoint.config_yaml);

    Ok(())
}

/// Main example function.
pub fn main() {
    println!("mlx-nemo-rs Examples");
    println!("====================");

    // Run examples
    if let Err(e) = example_asr_usage() {
        println!("ASR example failed: {}", e);
    }

    if let Err(e) = example_mlx_operations() {
        println!("MLX example failed: {}", e);
    }

    if let Err(e) = example_checkpoint_operations() {
        println!("Checkpoint example failed: {}", e);
    }
}
