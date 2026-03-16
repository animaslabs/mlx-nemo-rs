//! Debug decoder step by step with known inputs.
//!
//! This tests the LSTM decoder in isolation to compare with Python:
//! 1. Blank embedding through LSTM with zeros state
//! 2. Joint network with known encoder input
//!
//! Run with: cargo run --example debug_decoder_isolated

use mlx_nemo::asr::CacheAwareRNNTPipeline;
use mlx_rs::{Array, ops};

fn main() -> mlx_nemo::Result<()> {
    let model_path = "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";

    println!("Loading pipeline from: {}", model_path);
    let pipeline = CacheAwareRNNTPipeline::restore(model_path, 2)?;

    println!("\n=== Pipeline Loaded ===");
    println!("Sample rate: {} Hz", pipeline.sample_rate());
    println!("Blank ID: {}", pipeline.blank_id());

    // Access the decoder and joint network
    let decoder = pipeline.prediction_network();
    let joint = pipeline.joint();

    println!("\n=== Decoder Configuration ===");
    println!("Pred hidden: {}", decoder.pred_hidden());
    println!("Blank idx: {}", decoder.blank_idx());
    println!("Vocab size (excl blank): {}", decoder.vocab_size());
    println!("Is LSTM: {}", decoder.is_lstm());

    // Test 1: LSTM with blank token (zeros embedding) and zeros state
    // Initialize state (zeros)
    let state = decoder.initialize_state(1)?;
    println!("\n=== Test 1: LSTM with blank input and zeros state ===");
    println!("Input: blank token (embedding should be zeros)");
    println!("State: h=zeros, c=zeros");

    // Predict without a token (uses blank embedding)
    let (pred_out, _new_state) = decoder.predict_one(None, &state)?;
    pred_out.eval()?;

    // Print prediction output
    let pred_flat = pred_out.reshape(&[-1])?;
    pred_flat.eval()?;
    let pred_slice: &[f32] = pred_flat.as_slice();

    println!("\nPred output (first 10 values):");
    let first_10: Vec<f32> = pred_slice.iter().take(10).copied().collect();
    println!("  {:?}", first_10);

    println!("\nExpected from Python:");
    println!("  [-0.00382, -0.04477, 0.06805, 0.03125, -0.06471, 0.00307, 0.03036, 0.10786, -0.04205, 0.11384]");

    // Check if values match (within tolerance)
    let expected = [-0.003824716666713357, -0.044768139719963074, 0.06804525852203369,
                    0.03125034272670746, -0.06471093744039536, 0.003071495098993182,
                    0.03035961091518402, 0.10786266624927521, -0.04204889014363289,
                    0.11384092271327972];

    let mut max_diff: f32 = 0.0;
    for (i, (&actual, &exp)) in pred_slice.iter().take(10).zip(expected.iter()).enumerate() {
        let diff = (actual - exp).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.001 {
            println!("  [{}] MISMATCH: actual={:.6}, expected={:.6}, diff={:.6}", i, actual, exp, diff);
        }
    }
    println!("\nMax difference (first 10): {:.6e}", max_diff);
    if max_diff < 1e-5 {
        println!("✓ LSTM output matches Python!");
    } else if max_diff < 1e-3 {
        println!("✓ LSTM output close to Python (within 1e-3)");
    } else {
        println!("✗ LSTM output differs from Python!");
    }

    // Test 2: Joint network with zeros encoder input
    println!("\n=== Test 2: Joint network with zeros encoder input ===");

    // Create zeros encoder frame [1, 1, 1024]
    let enc_frame = Array::zeros::<f32>(&[1, 1, 1024])?;

    // Use the prediction output from Test 1
    let log_probs = joint.forward_one(&enc_frame, &pred_out)?;
    log_probs.eval()?;

    let lp_flat = log_probs.reshape(&[-1])?;
    lp_flat.eval()?;
    let lp_slice: &[f32] = lp_flat.as_slice();

    println!("Log probs (first 5 values):");
    let first_5: Vec<f32> = lp_slice.iter().take(5).copied().collect();
    println!("  {:?}", first_5);

    // Get blank probability
    let blank_id = decoder.blank_idx();
    let blank_prob = lp_slice.get(blank_id).copied().unwrap_or(f32::NEG_INFINITY);
    println!("\nBlank (id={}) log prob: {:.6}", blank_id, blank_prob);

    // Get argmax
    let best_token = lp_slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_prob = lp_slice[best_token];
    println!("Best token: {}, log prob: {:.6}", best_token, best_prob);

    println!("\nExpected from Python:");
    println!("  Best token: 1024 (blank), log prob: -0.001873");

    if best_token == blank_id {
        println!("\n✓ Joint network predicts blank token - matches Python!");
    } else {
        println!("\n✗ Joint network does NOT predict blank - differs from Python!");
        println!("  Rust predicted token: {}", best_token);
    }

    // Test 3: Joint network with sample encoder values (like Python uses)
    println!("\n=== Test 3: Joint network with sample encoder input ===");

    // Create encoder frame with sample values matching Python test
    let mut enc_data = vec![0.0f32; 1024];
    enc_data[0] = -0.04;
    enc_data[1] = -0.03;
    enc_data[2] = 0.06;
    enc_data[3] = -0.02;
    enc_data[4] = -0.04;

    let enc_frame = Array::from_slice(&enc_data, &[1, 1, 1024]);

    let log_probs = joint.forward_one(&enc_frame, &pred_out)?;
    log_probs.eval()?;

    let lp_flat = log_probs.reshape(&[-1])?;
    lp_flat.eval()?;
    let lp_slice: &[f32] = lp_flat.as_slice();

    let best_token = lp_slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_prob = lp_slice[best_token];
    let blank_prob = lp_slice.get(blank_id).copied().unwrap_or(f32::NEG_INFINITY);

    println!("Best token: {}, log prob: {:.6}", best_token, best_prob);
    println!("Blank (id={}) log prob: {:.6}", blank_id, blank_prob);

    println!("\nExpected from Python:");
    println!("  Best token: 1024 (blank), log prob: -0.001873");

    if best_token == blank_id {
        println!("\n✓ Matches Python!");
    } else {
        println!("\n✗ Differs from Python - expected blank (1024)");
    }

    // Test 4: Warmup initialization (matches NeMo add_sos=True)
    println!("\n=== Test 4: Warmup initialization (add_sos=True equivalent) ===");

    // Initialize with warmup - runs LSTM twice with blank inputs
    let (warmup_pred, _warmup_state) = decoder.initialize_state_with_warmup(1)?;
    warmup_pred.eval()?;

    let warmup_flat = warmup_pred.reshape(&[-1])?;
    warmup_flat.eval()?;
    let warmup_slice: &[f32] = warmup_flat.as_slice();

    println!("Warmup pred output (first 10 values):");
    let warmup_first_10: Vec<f32> = warmup_slice.iter().take(10).copied().collect();
    println!("  {:?}", warmup_first_10);

    // Expected from Python (output[0,1] from add_sos=True):
    // [0.0011212689569219947, 0.0056362031027674675, 0.023998918011784554, ...]
    let expected_warmup = [0.0011212689569219947, 0.0056362031027674675, 0.023998918011784554,
                          0.04137953370809555, -0.01906406320631504, 0.00014715063199400902,
                          0.0006426768959499896, 0.021843202784657478, -0.04307417199015617,
                          0.03243801742792129];

    println!("\nExpected from Python (add_sos=True output[-1]):");
    println!("  {:?}", &expected_warmup[..10]);

    let mut warmup_max_diff: f32 = 0.0;
    for (i, (&actual, &exp)) in warmup_slice.iter().take(10).zip(expected_warmup.iter()).enumerate() {
        let diff = (actual - exp).abs();
        if diff > warmup_max_diff {
            warmup_max_diff = diff;
        }
        if diff > 0.001 {
            println!("  [{}] MISMATCH: actual={:.6}, expected={:.6}, diff={:.6}", i, actual, exp, diff);
        }
    }
    println!("\nMax difference (first 10): {:.6e}", warmup_max_diff);

    if warmup_max_diff < 1e-5 {
        println!("✓ Warmup output matches Python!");
    } else if warmup_max_diff < 1e-3 {
        println!("✓ Warmup output close to Python (within 1e-3)");
    } else {
        println!("✗ Warmup output differs from Python!");
    }

    println!("\n=== Done ===");

    Ok(())
}
