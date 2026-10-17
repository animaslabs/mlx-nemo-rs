//! Loading models from the Hugging Face Hub.
//!
//! Ignored by default: they need network access (files already in the
//! Hugging Face cache are not downloaded again). Run with
//! `cargo test --release --features hub --test hub -- --ignored --test-threads=1`.

use mlx_voice::HubRepo;
use mlx_voice::asr::{EncDecClassificationModel, EncDecRnntModel};

const TEST_AUDIO: &str = "samples/2086-149220-0033.wav";

#[test]
#[ignore]
fn test_hub_nemo_archive() {
    // A repository with a single .nemo archive
    let mut model =
        EncDecClassificationModel::from_pretrained("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0")
            .expect("Failed to load VAD model from the Hub");
    let probs = model.infer(&[TEST_AUDIO]).expect("Inference failed");
    let speech = probs[0].iter().filter(|&&p| p > 0.5).count();
    assert!(speech * 2 > probs[0].len(), "expected mostly speech");
}

#[test]
#[ignore]
fn test_hub_safetensors_directory() {
    // A repository with MLX safetensors weights, config and tokenizer
    let mut model = EncDecRnntModel::from_pretrained(
        HubRepo::new("animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit").revision("main"),
    )
    .expect("Failed to load RNN-T model from the Hub");
    let transcript = model
        .transcribe_one(TEST_AUDIO)
        .expect("Transcription failed");
    assert!(
        transcript.text.to_lowercase().contains("observed phoebe"),
        "unexpected transcript: {transcript}"
    );
}

#[test]
#[ignore]
fn test_missing_repository_is_an_error() {
    let err = EncDecRnntModel::from_pretrained("animaslabs/does-not-exist-xyz").unwrap_err();
    assert!(err.to_string().contains("does-not-exist-xyz"), "{err}");
}
