//! Strict and non-strict weight loading against a real checkpoint.

use std::path::Path;

use mlx_voice::LoadOptions;
use mlx_voice::asr::{AsrModelConfig, EncDecRnntModel};
use mlx_voice::core::checkpoint::NemoCheckpoint;

const NEMOTRON: &str =
    "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";
const DROPPED: &str = "encoder.layers.3.feed_forward1.linear1.weight";

#[test]
#[ignore] // Requires the Nemotron model file to be present
fn test_missing_tensor_is_an_error_unless_non_strict() {
    let path = Path::new(NEMOTRON);
    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let checkpoint = NemoCheckpoint::open(path).unwrap();
    let config: AsrModelConfig = checkpoint.load_config().unwrap();
    let mut weights = checkpoint.load_weights().unwrap();
    assert!(
        weights.remove(DROPPED).is_some(),
        "{DROPPED} not in checkpoint"
    );

    let mut model = EncDecRnntModel::new(config.clone()).unwrap();
    let err = model.load_weights(&weights, None).unwrap_err().to_string();
    assert!(err.contains(DROPPED), "unexpected error: {err}");

    let mut model = EncDecRnntModel::new(config).unwrap();
    let report = model
        .load_weights_with(&weights, None, LoadOptions::default().strict(false))
        .unwrap();
    assert_eq!(report.missing, vec![DROPPED.to_string()]);
}

#[test]
#[ignore] // Requires the Nemotron model file to be present
fn test_complete_checkpoint_loads_strictly() {
    let path = Path::new(NEMOTRON);
    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let checkpoint = NemoCheckpoint::open(path).unwrap();
    let config: AsrModelConfig = checkpoint.load_config().unwrap();
    let weights = checkpoint.load_weights().unwrap();
    let mut model = EncDecRnntModel::new(config).unwrap();
    let report = model
        .load_weights_with(&weights, None, LoadOptions::default())
        .unwrap();
    assert!(report.is_complete());
}
