//! Integration tests for streaming diarization and multi-talker ASR.
//!
//! All tests are `#[ignore]` by default — they require model files that are
//! downloaded separately:
//!
//!   cargo xtask download nvidia/diar_streaming_sortformer_4spk-v2.1
//!   cargo xtask download nvidia/multitalker-parakeet-streaming-0.6b-v1
//!
//! Run with:
//!
//!   cargo test --release --test multitalker -- --ignored --test-threads=1 --nocapture

use std::path::Path;

use mlx_voice::asr::{
    MultiTalkerMode, MultiTalkerPipeline, SortformerEncLabelModel, SpeakerSegment,
};
use mlx_voice::Array;

const DIAR_MODEL: &str =
    "models/nvidia/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo";
const MT_ASR_MODEL: &str =
    "models/nvidia/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo";

const SINGLE_SPEAKER_AUDIO: &str = "samples/Speaker26_004_16k.wav";
const SINGLE_SPEAKER_TRANSCRIPT: &str = "samples/Speaker26_004_16k-transcript.txt";

const SECOND_SPEAKER_AUDIO: &str = "samples/2086-149220-0033.wav";
const SECOND_SPEAKER_TEXT: &str = "well i don't wish to see it any more observed phoebe \
    turning away her eyes it is certainly very like the old portrait";

/// A single-talker streaming model, for serial mode
const SERIAL_ASR_MODEL: &str = "models/animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit";

fn require(paths: &[&str]) -> bool {
    match paths.iter().find(|p| !Path::new(p).exists()) {
        Some(missing) => {
            eprintln!("Skipping test: {} not found", missing);
            false
        }
        None => true,
    }
}

/// Lowercase, strip non-alphanumeric (keep spaces), collapse whitespace.
fn normalize_text(s: &str) -> String {
    let cleaned: String = s
        .to_lowercase()
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
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hypothesis.split_whitespace().collect();
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut cur = vec![i; h.len() + 1];
        for j in 1..=h.len() {
            let cost = usize::from(r[i - 1] != h[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[h.len()] as f32 / r.len().max(1) as f32
}

fn load_audio(path: &str) -> Vec<f32> {
    let (samples, sr) = mlx_voice::audio::load_wav(Path::new(path)).expect("load wav");
    assert_eq!(sr, 16000, "test audio must be 16 kHz");
    samples
}

/// Per-frame speaker probabilities from streaming diarization of a file.
fn diarize(path: &str) -> Vec<Vec<f32>> {
    let mut model = SortformerEncLabelModel::restore(DIAR_MODEL).expect("load diarizer");
    let audio = load_audio(path);
    let audio_arr = Array::from_slice(&audio, &[1, audio.len() as i32]);
    let lengths = Array::from_slice(&[audio.len() as i32], &[1]);
    let preds = model.diarize(&audio_arr, &lengths).expect("diarize");
    let n_spk = preds.dim(2) as usize;
    let flat: Vec<f32> = preds.reshape(&[-1]).unwrap().as_slice::<f32>().to_vec();
    flat.chunks(n_spk).map(|c| c.to_vec()).collect()
}

#[test]
#[ignore]
fn test_streaming_diarization_single_speaker() {
    if !require(&[DIAR_MODEL, SINGLE_SPEAKER_AUDIO]) {
        return;
    }
    let preds = diarize(SINGLE_SPEAKER_AUDIO);

    // ~60 s at 80 ms per frame
    assert!(preds.len() > 700, "got {} frames", preds.len());

    // Print activity per second for inspection
    for (sec, frames) in preds.chunks(25).enumerate() {
        let means: Vec<String> = (0..frames[0].len())
            .map(|s| {
                format!(
                    "{:.2}",
                    frames.iter().map(|f| f[s]).sum::<f32>() / frames.len() as f32
                )
            })
            .collect();
        println!("{:3}s {}", sec * 2, means.join(" "));
    }

    let speech: Vec<&Vec<f32>> = preds
        .iter()
        .filter(|f| f.iter().any(|&p| p > 0.5))
        .collect();
    let spk0 = speech.iter().filter(|f| f[0] > 0.5).count();
    assert!(
        spk0 as f32 >= 0.95 * speech.len() as f32,
        "speaker 0 active in {}/{} speech frames",
        spk0,
        speech.len()
    );
}

#[test]
#[ignore]
fn test_multitalker_single_speaker_wer() {
    if !require(&[DIAR_MODEL, MT_ASR_MODEL, SINGLE_SPEAKER_AUDIO]) {
        return;
    }
    let mut pipeline = MultiTalkerPipeline::restore(MT_ASR_MODEL, DIAR_MODEL, 0).expect("load");
    assert_eq!(pipeline.mode(), MultiTalkerMode::Parallel);

    pipeline
        .set_audio_file(SINGLE_SPEAKER_AUDIO)
        .expect("audio");
    let segments = pipeline.run().expect("run");

    assert!(
        segments.iter().all(|s| s.speaker == 0),
        "expected only speaker 0"
    );
    let hypothesis: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
    let hypothesis = normalize_text(&hypothesis.join(" "));
    let reference = normalize_text(&std::fs::read_to_string(SINGLE_SPEAKER_TRANSCRIPT).unwrap());

    let wer = word_error_rate(&reference, &hypothesis);
    println!("WER: {:.3}\nHYP: {}", wer, hypothesis);
    assert!(wer < 0.15, "WER too high: {:.3}", wer);
}

/// Mix the first 20 s of one speaker with a different speaker overlaid from 8 s.
fn overlap_mix() -> Vec<f32> {
    let a = load_audio(SINGLE_SPEAKER_AUDIO);
    let b = load_audio(SECOND_SPEAKER_AUDIO);
    let offset = 8 * 16000;
    let mut mix = a[..20 * 16000].to_vec();
    mix.resize(mix.len().max(offset + b.len()), 0.0);
    for (i, v) in b.iter().enumerate() {
        mix[offset + i] += v;
    }
    mix.iter().map(|v| (v * 0.7).clamp(-1.0, 1.0)).collect()
}

fn transcribe_samples(
    asr_model: &str,
    samples: Vec<f32>,
) -> (MultiTalkerMode, Vec<SpeakerSegment>) {
    let mut pipeline = MultiTalkerPipeline::restore(asr_model, DIAR_MODEL, 0).expect("load");
    pipeline.add_audio(&samples);
    pipeline.end_stream();
    let segments = pipeline.run().expect("run");
    for s in &segments {
        println!(
            "[{:6.2}-{:6.2}] speaker_{}: {}",
            s.start_time, s.end_time, s.speaker, s.text
        );
    }
    (pipeline.mode(), segments)
}

/// Text per speaker, normalized.
fn speaker_texts(segments: &[SpeakerSegment]) -> Vec<String> {
    let n = segments.iter().map(|s| s.speaker + 1).max().unwrap_or(0);
    (0..n)
        .map(|spk| {
            let texts: Vec<&str> = segments
                .iter()
                .filter(|s| s.speaker == spk)
                .map(|s| s.text.as_str())
                .collect();
            normalize_text(&texts.join(" "))
        })
        .collect()
}

#[test]
#[ignore]
fn test_multitalker_overlapping_speakers() {
    if !require(&[
        DIAR_MODEL,
        MT_ASR_MODEL,
        SINGLE_SPEAKER_AUDIO,
        SECOND_SPEAKER_AUDIO,
    ]) {
        return;
    }
    let (mode, segments) = transcribe_samples(MT_ASR_MODEL, overlap_mix());
    assert_eq!(mode, MultiTalkerMode::Parallel);

    let texts = speaker_texts(&segments);
    assert_eq!(texts.len(), 2, "expected two speakers, got {:?}", texts);

    // The second speaker is fully overlapped by the first and must still be
    // transcribed on its own
    let second_wer = word_error_rate(&normalize_text(SECOND_SPEAKER_TEXT), &texts[1]);
    println!("second speaker WER: {:.3}", second_wer);
    assert!(
        second_wer < 0.25,
        "second speaker WER too high: {:.3}",
        second_wer
    );
    assert!(texts[0].contains("raised one of his dishes towards his mouth"));
    assert!(
        !texts[0].contains("phoebe"),
        "second speaker leaked into the first"
    );
}

#[test]
#[ignore]
fn test_serial_mode_attributes_speakers() {
    if !require(&[
        DIAR_MODEL,
        SERIAL_ASR_MODEL,
        SINGLE_SPEAKER_AUDIO,
        SECOND_SPEAKER_AUDIO,
    ]) {
        return;
    }
    let (mode, segments) = transcribe_samples(SERIAL_ASR_MODEL, overlap_mix());
    assert_eq!(mode, MultiTalkerMode::Serial);
    // A single-talker model can't separate overlap, but both speakers should
    // be attributed some words
    assert!(segments.iter().any(|s| s.speaker == 0));
    assert!(segments.iter().any(|s| s.speaker == 1));
}
