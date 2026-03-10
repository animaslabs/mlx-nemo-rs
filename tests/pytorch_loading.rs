//! Integration tests for PyTorch checkpoint loading.

use std::path::Path;

#[test]
#[ignore] // Requires model file to be present
fn test_load_parakeet_rnnt_checkpoint() {
    let path = Path::new("models/nvidia/parakeet-rnnt-0.6b/model_weights.ckpt");

    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let weights = mlx_nemo::core::pytorch::load(path).expect("Failed to load checkpoint");

    // Should have loaded tensors
    assert!(!weights.is_empty(), "No tensors loaded");
    println!("Loaded {} tensors", weights.len());

    // Check some expected tensor names exist
    assert!(
        weights.contains_key("encoder.pre_encode.out.weight"),
        "Missing encoder.pre_encode.out.weight"
    );
    assert!(
        weights.contains_key("preprocessor.featurizer.window"),
        "Missing preprocessor.featurizer.window"
    );

    // Verify tensor shapes for known tensors
    let window = weights.get("preprocessor.featurizer.window").unwrap();
    assert_eq!(
        window.shape(),
        &[400],
        "Unexpected shape for featurizer window"
    );

    let fb = weights.get("preprocessor.featurizer.fb").unwrap();
    assert_eq!(
        fb.shape(),
        &[1, 80, 257],
        "Unexpected shape for featurizer fb"
    );
}

#[test]
#[ignore] // Requires model file to be present
fn test_load_ctc_nemo() {
    // Load weights directly from the .nemo file to verify PyTorch checkpoint loading
    use mlx_nemo::core::checkpoint::NemoCheckpoint;

    let path = Path::new("models/nvidia/parakeet-ctc-0.6b/parakeet-ctc-0.6b.nemo");

    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let checkpoint = NemoCheckpoint::open(path).expect("Failed to open checkpoint");
    let weights = checkpoint.load_weights().expect("Failed to load weights");

    // Verify we loaded expected number of tensors
    assert!(
        weights.len() > 900,
        "Expected >900 tensors, got {}",
        weights.len()
    );

    // Verify key weight tensors exist with correct shapes
    assert!(weights.contains_key("encoder.pre_encode.out.weight"));
    assert!(weights.contains_key("encoder.layers.0.self_attn.linear_q.weight"));
    assert!(weights.contains_key("decoder.decoder_layers.0.weight"));
}

#[test]
fn test_is_pytorch_zip_nonexistent() {
    assert!(!mlx_nemo::core::pytorch::is_pytorch_zip(
        "/nonexistent/path.ckpt"
    ));
}

#[test]
fn test_storage_type_parsing() {
    use mlx_nemo::core::pytorch::StorageType;

    assert_eq!(
        StorageType::from_class_name("FloatStorage"),
        Some(StorageType::Float32)
    );
    assert_eq!(
        StorageType::from_class_name("LongStorage"),
        Some(StorageType::Int64)
    );
    assert_eq!(
        StorageType::from_class_name("HalfStorage"),
        Some(StorageType::Float16)
    );
    assert_eq!(StorageType::from_class_name("UnknownStorage"), None);
}
