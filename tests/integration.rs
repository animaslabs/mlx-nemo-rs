//! End-to-end integration tests for ASR pipeline and VAD.
//!
//! All tests are `#[ignore]` by default — they require model files that are
//! downloaded separately.  Run with:
//!
//!   cargo test --test integration -- --ignored --test-threads=1

use std::path::Path;

use mlx_voice::asr::{
    CacheAwareRNNTPipeline, EncDecClassificationModel, EncDecRNNTModel, VadPostProcessConfig,
    VadPostProcessor,
};

// ---------------------------------------------------------------------------
// Test audio
// ---------------------------------------------------------------------------

const TEST_AUDIO: &str = "samples/2086-149220-0033.wav";

const EXPECTED_TEXT: &str = "well i don't wish to see it any more observed phoebe \
    turning away her eyes it is certainly very like the old portrait";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return `true` if the model path exists; otherwise print a skip message and
/// return `false`.
fn require_model(path: &str) -> bool {
    if Path::new(path).exists() {
        true
    } else {
        eprintln!("Skipping test: model not found at {}", path);
        false
    }
}

/// Lowercase, strip non-alphanumeric (keep spaces), collapse whitespace, trim.
fn normalize_text(s: &str) -> String {
    let lower = s.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Word-level edit distance divided by reference length.
fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    let n = ref_words.len();
    let m = hyp_words.len();

    if n == 0 {
        return if m == 0 { 0.0 } else { 1.0 };
    }

    // Standard Levenshtein DP (word level)
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if ref_words[i - 1] == hyp_words[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[n][m] as f32 / n as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_full_file_transcription_rnnt() {
    let model_dir = "models/animaslabs/parakeet-rnnt-0.6b-mlx";
    if !require_model(model_dir) {
        return;
    }

    let mut model =
        EncDecRNNTModel::restore_from_directory(model_dir).expect("Failed to load RNNT model");

    let transcriptions = model
        .transcribe(&[TEST_AUDIO])
        .expect("Transcription failed");

    assert_eq!(transcriptions.len(), 1);
    let hypothesis = normalize_text(&transcriptions[0]);
    let reference = normalize_text(EXPECTED_TEXT);
    let wer = word_error_rate(&reference, &hypothesis);

    println!("Reference:  {}", reference);
    println!("Hypothesis: {}", hypothesis);
    println!("WER:        {:.1}%", wer * 100.0);

    assert!(
        wer < 0.10,
        "WER {:.1}% exceeds 10% threshold\n  ref: {}\n  hyp: {}",
        wer * 100.0,
        reference,
        hypothesis,
    );
}

#[test]
#[ignore]
fn test_streaming_transcription() {
    let model_dir = "models/animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit";
    if !require_model(model_dir) {
        return;
    }

    let mut pipeline = CacheAwareRNNTPipeline::restore_from_directory(model_dir, 0)
        .expect("Failed to load streaming pipeline");

    pipeline
        .set_audio_file(TEST_AUDIO)
        .expect("Failed to load audio file");

    let mut final_text = String::new();
    loop {
        match pipeline.transcribe_step().expect("transcribe_step failed") {
            Some(output) => {
                if !output.text.is_empty() {
                    final_text = output.text.clone();
                }
                if output.is_final {
                    break;
                }
            }
            None => break,
        }
    }

    let hypothesis = normalize_text(&final_text);
    let reference = normalize_text(EXPECTED_TEXT);
    let wer = word_error_rate(&reference, &hypothesis);

    println!("Reference:  {}", reference);
    println!("Hypothesis: {}", hypothesis);
    println!("WER:        {:.1}%", wer * 100.0);

    assert!(
        wer < 0.15,
        "WER {:.1}% exceeds 15% threshold\n  ref: {}\n  hyp: {}",
        wer * 100.0,
        reference,
        hypothesis,
    );
}

#[test]
#[ignore]
fn test_vad_speech_detection() {
    let model_path =
        "models/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0/frame_vad_multilingual_marblenet_v2.0.nemo";
    if !require_model(model_path) {
        return;
    }

    let mut model =
        EncDecClassificationModel::restore(model_path).expect("Failed to load VAD model");

    let all_probs = model.infer(&[TEST_AUDIO]).expect("VAD inference failed");
    assert_eq!(all_probs.len(), 1);

    let probs = &all_probs[0];
    let frame_duration = model.frame_duration();
    let audio_duration = probs.len() as f32 * frame_duration;

    println!("Frames: {}", probs.len());
    println!("Frame duration: {:.4}s", frame_duration);
    println!("Audio duration:  {:.2}s", audio_duration);

    let processor = VadPostProcessor::new(VadPostProcessConfig::default());
    let segments = processor.process(probs, frame_duration, audio_duration);

    println!("Speech segments: {}", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  [{:2}] {:.3}s - {:.3}s  ({:.3}s)",
            i,
            seg.start,
            seg.end,
            seg.duration()
        );
    }

    assert!(
        !segments.is_empty(),
        "Expected at least one speech segment but got none"
    );

    let total_speech: f32 = segments.iter().map(|s| s.duration()).sum();
    let speech_ratio = total_speech / audio_duration;
    println!("Speech ratio: {:.1}%", speech_ratio * 100.0);

    assert!(
        speech_ratio > 0.50,
        "Expected >50% speech coverage but got {:.1}%",
        speech_ratio * 100.0,
    );
}

// ---------------------------------------------------------------------------
// Unit tests for helpers (always run)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_normalize_text_basic() {
        assert_eq!(
            normalize_text("Hello, World!  How are you?"),
            "hello world how are you"
        );
    }

    #[test]
    fn test_normalize_text_punctuation() {
        assert_eq!(normalize_text("It's a test—really."), "it s a test really");
    }

    #[test]
    fn test_wer_identical() {
        assert_eq!(word_error_rate("hello world", "hello world"), 0.0);
    }

    #[test]
    fn test_wer_completely_wrong() {
        assert_eq!(word_error_rate("hello world", "foo bar"), 1.0);
    }

    #[test]
    fn test_wer_partial() {
        // 1 substitution out of 4 words = 0.25
        let wer = word_error_rate("the quick brown fox", "the quick brown dog");
        assert!((wer - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_wer_empty_reference() {
        assert_eq!(word_error_rate("", ""), 0.0);
        assert_eq!(word_error_rate("", "extra words"), 1.0);
    }
}
