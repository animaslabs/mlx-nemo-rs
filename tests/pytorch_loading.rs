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

#[test]
#[ignore] // Requires Nemotron model file to be present
fn test_load_nemotron_checkpoint() {
    use mlx_nemo::core::checkpoint::NemoCheckpoint;

    let path = Path::new(
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
    );

    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let checkpoint = NemoCheckpoint::open(path).expect("Failed to open checkpoint");
    let weights = checkpoint.load_weights().expect("Failed to load weights");

    // Nemotron has more layers (24 vs 17) and larger model
    assert!(
        weights.len() > 600,
        "Expected >600 tensors, got {}",
        weights.len()
    );
    println!("Loaded {} tensors from Nemotron checkpoint", weights.len());

    // Verify key weight tensors exist
    assert!(
        weights.contains_key("encoder.pre_encode.out.weight"),
        "Missing encoder.pre_encode.out.weight"
    );
    assert!(
        weights.contains_key("encoder.layers.0.self_attn.linear_q.weight"),
        "Missing encoder.layers.0.self_attn.linear_q.weight"
    );

    // Find joint network key pattern
    let joint_keys: Vec<_> = weights.keys().filter(|k| k.contains("joint")).collect();
    assert!(
        !joint_keys.is_empty(),
        "Expected joint network weights to be present"
    );
    println!("Found {} joint-related tensors", joint_keys.len());

    // Verify Nemotron-specific tensor shapes
    // d_model = 1024 (vs 512 in Parakeet)
    let linear_q = weights
        .get("encoder.layers.0.self_attn.linear_q.weight")
        .unwrap();
    assert_eq!(
        linear_q.shape(),
        &[1024, 1024],
        "Unexpected shape for linear_q.weight - expected [1024, 1024] for Nemotron"
    );
}

#[test]
#[ignore] // Requires Nemotron model file to be present
fn test_nemotron_weight_shapes() {
    use mlx_nemo::core::checkpoint::NemoCheckpoint;

    let path = Path::new(
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
    );

    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let checkpoint = NemoCheckpoint::open(path).expect("Failed to open checkpoint");
    let weights = checkpoint.load_weights().expect("Failed to load weights");

    // Check pre_encode (subsampling) output projection
    let out_weight = weights
        .get("encoder.pre_encode.out.weight")
        .expect("Missing encoder.pre_encode.out.weight");
    println!("encoder.pre_encode.out.weight: {:?}", out_weight.shape());

    // This should tell us what the expected input dimension is
    // Shape should be [d_model, linear_in] = [1024, conv_channels * feat_out]
    assert_eq!(out_weight.shape()[0], 1024, "d_model should be 1024");

    let linear_in = out_weight.shape()[1];
    println!("Expected linear_in from weights: {}", linear_in);

    // With 256 channels: linear_in / 256 = feat_out
    let feat_out = linear_in / 256;
    println!("Implied feat_out: {} (= {} / 256)", feat_out, linear_in);

    // Check first conv weight
    if let Some(w) = weights.get("encoder.pre_encode.conv.0.weight") {
        println!("encoder.pre_encode.conv.0.weight: {:?}", w.shape());
    }
}

#[test]
#[ignore] // Requires Nemotron model file to be present
fn test_parse_nemotron_config() {
    use mlx_nemo::asr::config::AsrModelConfig;
    use mlx_nemo::core::checkpoint::NemoCheckpoint;

    let path = Path::new(
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
    );

    if !path.exists() {
        eprintln!("Skipping test: model file not found at {}", path.display());
        return;
    }

    let checkpoint = NemoCheckpoint::open(path).expect("Failed to open checkpoint");
    let config: AsrModelConfig = checkpoint.load_config().expect("Failed to load config");

    // Verify Nemotron-specific config values
    assert_eq!(
        config.encoder.feat_in, 128,
        "Nemotron uses 128 mel features"
    );
    assert_eq!(
        config.encoder.d_model, 1024,
        "Nemotron d_model should be 1024"
    );
    assert_eq!(
        config.encoder.n_layers, 24,
        "Nemotron has 24 encoder layers"
    );
    assert_eq!(
        config.encoder.conv_kernel_size, 9,
        "Nemotron conv kernel is 9"
    );
    assert_eq!(
        config.encoder.conv_norm_type, "layer_norm",
        "Nemotron uses layer_norm in convolution"
    );

    // Verify streaming config
    assert!(
        config.encoder.is_causal_conv(),
        "Nemotron should have causal convolution"
    );
    assert!(
        config.encoder.is_causal_downsampling(),
        "Nemotron should have causal downsampling"
    );
    assert!(
        config.encoder.is_streaming(),
        "Nemotron should be streaming-capable"
    );

    // Verify att_context_size is nested (multiple streaming modes)
    let att_ctx = config
        .encoder
        .att_context_size
        .as_ref()
        .expect("Should have att_context_size");
    assert!(
        att_ctx.num_modes() >= 4,
        "Nemotron should have at least 4 streaming modes, got {}",
        att_ctx.num_modes()
    );

    // First mode should be [70, 13] (1.12s latency)
    let (left, right) = att_ctx.get_default().expect("Should have default context");
    assert_eq!(left, 70, "Default left context should be 70");
    assert_eq!(right, 13, "Default right context should be 13");

    println!("Nemotron config parsed successfully:");
    println!("  feat_in: {}", config.encoder.feat_in);
    println!("  d_model: {}", config.encoder.d_model);
    println!("  n_layers: {}", config.encoder.n_layers);
    println!("  streaming modes: {}", att_ctx.num_modes());
}
