//! End-to-end integration tests for ASR pipeline and VAD.
//!
//! All tests are `#[ignore]` by default — they require model files that are
//! downloaded separately.  Run with:
//!
//!   cargo test --test integration -- --ignored --test-threads=1

use std::path::Path;

use mlx_voice::asr::{
    AudioInput, EncDecClassificationModel, EncDecRNNTModel, StreamingOptions, StreamingSession,
    TranscribeOptions, Transcript, VadPostProcessConfig, VadPostProcessor,
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

    let mut model = EncDecRNNTModel::from_pretrained(model_dir).expect("Failed to load RNNT model");

    let transcriptions = model
        .transcribe(&[TEST_AUDIO])
        .expect("Transcription failed");

    assert_eq!(transcriptions.len(), 1);
    let hypothesis = normalize_text(&transcriptions[0].text);
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
fn test_in_memory_audio_matches_file() {
    let model_dir = "models/animaslabs/parakeet-rnnt-0.6b-mlx";
    if !require_model(model_dir) {
        return;
    }
    let mut model = EncDecRNNTModel::from_pretrained(model_dir).expect("Failed to load RNNT model");

    let from_file = model
        .transcribe_one(TEST_AUDIO)
        .expect("Transcription failed");
    let (samples, sample_rate) = mlx_voice::audio::load_wav(TEST_AUDIO).unwrap();
    let from_samples = model
        .transcribe_one((&samples, sample_rate))
        .expect("Transcription failed");
    assert_eq!(from_samples, from_file);

    // A different input rate is resampled to the model's
    let resampled = mlx_voice::audio::resample(&samples, sample_rate, 48_000).unwrap();
    let from_48k = model.transcribe_one((&resampled, 48_000)).unwrap();
    let wer = word_error_rate(
        &normalize_text(EXPECTED_TEXT),
        &normalize_text(&from_48k.text),
    );
    assert!(
        wer < 0.10,
        "WER {:.1}% at 48 kHz: {}",
        wer * 100.0,
        from_48k
    );

    // Mixed inputs in small batches keep their order; empty audio is empty
    let empty: Vec<f32> = Vec::new();
    let inputs = [
        AudioInput::from(TEST_AUDIO),
        AudioInput::samples(&empty, sample_rate),
        AudioInput::samples(&samples, sample_rate),
    ];
    let batched = model
        .transcribe_with(inputs, &TranscribeOptions::default().batch_size(2))
        .unwrap();
    assert_eq!(batched.len(), 3);
    assert_eq!(batched[0].text, from_file.text);
    assert_eq!(batched[1], Transcript::default());
    assert_eq!(batched[2].text, from_file.text);
}

#[test]
#[ignore]
fn test_word_timestamps() {
    let model_dir = "models/animaslabs/parakeet-rnnt-0.6b-mlx";
    if !require_model(model_dir) {
        return;
    }
    let mut model = EncDecRNNTModel::from_pretrained(model_dir).expect("Failed to load RNNT model");
    let (samples, sample_rate) = mlx_voice::audio::load_wav(TEST_AUDIO).unwrap();
    let duration = samples.len() as f32 / sample_rate as f32;

    for options in [
        TranscribeOptions::default(),
        TranscribeOptions::default().beam_search(4),
    ] {
        let transcript = model
            .transcribe_with([TEST_AUDIO], &options)
            .unwrap()
            .remove(0);
        let words: Vec<&str> = transcript.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(words.join(" "), transcript.text.trim(), "{options:?}");
        assert_eq!(
            transcript.tokens.len(),
            transcript.tokens.iter().filter(|t| t.end > t.start).count()
        );
        for pair in transcript.words.windows(2) {
            assert!(pair[0].start <= pair[1].start, "{:?}", pair);
        }
        let first = transcript.words.first().unwrap();
        let last = transcript.words.last().unwrap();
        assert!(first.start < 2.0, "first word at {}", first.start);
        assert!(
            last.end <= duration + 0.1 && last.end > duration - 3.0,
            "last word ends at {} of {duration}",
            last.end
        );
    }
}

#[test]
#[ignore]
fn test_tokenizer_round_trip() {
    let model_dir = "models/animaslabs/parakeet-rnnt-0.6b-mlx";
    if !require_model(model_dir) {
        return;
    }
    let model = EncDecRNNTModel::from_pretrained(model_dir).expect("Failed to load RNNT model");
    let tokenizer = model.tokenizer().expect("model has a tokenizer");

    assert_eq!(tokenizer.vocab_size(), model.blank_id());
    let ids = tokenizer.encode("well i don't wish to see it").unwrap();
    assert!(tokenizer.is_word_start(ids[0]));
    assert_eq!(tokenizer.decode(&ids, true), "well i don't wish to see it");
    assert_eq!(tokenizer.piece_to_id("▁well"), Some(ids[0]));
    assert_eq!(tokenizer.detected_language(&ids), None);
}

#[test]
#[ignore]
fn test_multilingual_offline_transcription() {
    let model_path =
        "models/nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo";
    if !require_model(model_path) {
        return;
    }
    let mut model = EncDecRNNTModel::from_pretrained(model_path).expect("Failed to load model");

    // Auto mode detects the language and keeps the tag out of text and words
    let auto = model.transcribe_one(TEST_AUDIO).unwrap();
    assert_eq!(auto.language.as_deref(), Some("en-US"));
    assert!(!auto.text.contains('<'), "{auto}");
    assert!(auto.words.iter().all(|w| !w.text.contains('<')));
    assert!(auto.tokens.iter().any(|t| t.text == "<en-US>"));

    // A per-call language leaves the model's selection unchanged
    let options = TranscribeOptions::default()
        .language("en-US")
        .strip_lang_tags(false);
    let english = model
        .transcribe_with([TEST_AUDIO], &options)
        .unwrap()
        .remove(0);
    assert!(english.text.contains("<en-US>"), "{english}");
    assert_eq!(model.language(), Some("auto"));
    assert!(
        model
            .transcribe_with(
                [TEST_AUDIO],
                &TranscribeOptions::default().language("xx-YY")
            )
            .is_err()
    );
    assert_eq!(model.language(), Some("auto"));
}

#[test]
#[ignore]
fn test_multilingual_streaming_transcription() {
    let model_path =
        "models/nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo";
    if !require_model(model_path) {
        return;
    }

    let mut model = EncDecRNNTModel::from_pretrained(model_path).expect("Failed to load model");
    assert_eq!(model.language(), Some("auto"));
    assert!(model.languages().contains(&"de-DE"));
    assert!(model.set_language("xx-YY").is_err());

    // Auto mode: the model tags the language, and the tag stays out of the text
    let options = StreamingOptions::default().mode(3);
    let auto = model
        .transcribe_streaming(TEST_AUDIO, options.clone())
        .expect("streaming failed");
    assert_eq!(auto.language.as_deref(), Some("en-US"));
    assert!(!auto.text.contains('<'), "language tag in text: {auto}");
    let wer = word_error_rate(&normalize_text(EXPECTED_TEXT), &normalize_text(&auto.text));
    println!("auto:  {} (WER {:.1}%)", auto, wer * 100.0);
    assert!(wer < 0.15, "WER {:.1}%: {}", wer * 100.0, auto);

    // An explicit target language
    model.set_language("en-US").unwrap();
    let english = model.transcribe_streaming(TEST_AUDIO, options).unwrap();
    let wer = word_error_rate(
        &normalize_text(EXPECTED_TEXT),
        &normalize_text(&english.text),
    );
    println!("en-US: {} (WER {:.1}%)", english, wer * 100.0);
    assert!(wer < 0.15, "WER {:.1}%: {}", wer * 100.0, english);
}

#[test]
#[ignore]
fn test_streaming_transcription() {
    let model_dir = "models/animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit";
    if !require_model(model_dir) {
        return;
    }
    let mut model = EncDecRNNTModel::from_pretrained(model_dir).expect("Failed to load model");

    // Live-style input: 100 ms chunks at 48 kHz, resampled by the session
    let (samples, sample_rate) = mlx_voice::audio::load_wav(TEST_AUDIO).unwrap();
    let samples = mlx_voice::audio::resample(&samples, sample_rate, 48_000).unwrap();
    let mut stream = model.stream(StreamingOptions::default()).unwrap();
    let mut deltas = String::new();
    let mut last_end = 0.0f32;
    for chunk in samples.chunks(4_800) {
        stream.push_audio(chunk, 48_000).unwrap();
        while let Some(update) = stream.next_update().unwrap() {
            deltas.push_str(&update.delta);
            for token in &update.tokens {
                assert!(
                    token.start >= last_end - 0.081,
                    "{token:?} before {last_end}"
                );
                last_end = token.end;
            }
        }
    }
    assert!(
        stream.push_audio(&[], 16_000).is_err(),
        "rate change accepted"
    );
    stream.end_audio().unwrap();
    while let Some(update) = stream.next_update().unwrap() {
        deltas.push_str(&update.delta);
    }
    let transcript = stream.finish().unwrap();
    assert!(stream.is_done());
    assert!(
        stream.push_audio(&[0.0], 48_000).is_err(),
        "push after end accepted"
    );
    assert_eq!(deltas, transcript.text);

    let hypothesis = normalize_text(&transcript.text);
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
    let last_word = transcript.words.last().unwrap();
    let duration = samples.len() as f32 / 48_000.0;
    assert!(
        last_word.end <= duration + 0.2,
        "{last_word:?} past {duration}"
    );

    // The session is reusable after a reset. Audio pushed before decoding is
    // all buffered (at 16 kHz, less what the resampler still holds)
    stream.reset().unwrap();
    assert_eq!(stream.buffered_samples(), 0);
    stream.push_audio(&samples, 48_000).unwrap();
    let expected = samples.len() / 3;
    let buffered = stream.buffered_samples();
    assert!(
        buffered <= expected && buffered + 1_100 >= expected,
        "buffered {buffered} of {expected}"
    );
    stream.next_update().unwrap().expect("a chunk is ready");
    assert!(stream.buffered_samples() < buffered);
    assert_eq!(stream.finish().unwrap(), transcript);
    assert!(
        stream.buffered_samples() < 1_600,
        "{}",
        stream.buffered_samples()
    );
}

#[test]
#[ignore]
fn test_streaming_modes_are_independent() {
    let model_dir = "models/animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit";
    if !require_model(model_dir) {
        return;
    }
    let mut model = EncDecRNNTModel::from_pretrained(model_dir).expect("Failed to load model");
    let offline = model.transcribe_one(TEST_AUDIO).unwrap();

    // Streaming in another mode leaves offline transcription unchanged
    let mode_2 = model
        .transcribe_streaming(TEST_AUDIO, StreamingOptions::default().mode(2))
        .unwrap();
    assert_eq!(model.transcribe_one(TEST_AUDIO).unwrap(), offline);

    // Switching modes on one model matches a fresh model in that mode
    let mode_0 = model
        .transcribe_streaming(TEST_AUDIO, StreamingOptions::default())
        .unwrap();
    let mut fresh = EncDecRNNTModel::from_pretrained(model_dir).unwrap();
    let fresh_0 = fresh
        .transcribe_streaming(TEST_AUDIO, StreamingOptions::default())
        .unwrap();
    assert_eq!(mode_0, fresh_0);
    println!("mode 0: {mode_0}\nmode 2: {mode_2}");

    // An owned session hands its model back
    let session = StreamingSession::new(fresh, StreamingOptions::default()).unwrap();
    let _model: EncDecRNNTModel = session.into_model();
}

#[test]
#[ignore]
fn test_vad_speech_detection() {
    let model_path = "models/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0/frame_vad_multilingual_marblenet_v2.0.nemo";
    if !require_model(model_path) {
        return;
    }

    let mut model =
        EncDecClassificationModel::from_pretrained(model_path).expect("Failed to load VAD model");

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
