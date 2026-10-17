//! Offline transcription inputs, options and results.

use std::fmt;
use std::path::Path;

use crate::asr::Tokenizer;
use crate::asr::decoding::TimedToken;
use crate::audio::{load_wav, resample};
use crate::core::error::Result;

/// Audio to transcribe: a WAV file, or mono samples in `[-1, 1]` at a given
/// sample rate. Either is resampled to the model's rate.
///
/// Paths and `(samples, sample_rate)` pairs convert into it:
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// # use mlx_voice::asr::EncDecRNNTModel;
/// # let mut model = EncDecRNNTModel::from_pretrained("model.nemo")?;
/// # let samples = vec![0.0f32; 48_000];
/// let from_files = model.transcribe(["a.wav", "b.wav"])?;
/// let from_memory = model.transcribe_one((&samples[..], 48_000))?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AudioInput<'a> {
    /// A WAV file
    Path(&'a Path),
    /// Mono samples in `[-1, 1]`
    Samples {
        /// The samples
        samples: &'a [f32],
        /// Their sample rate in Hz
        sample_rate: u32,
    },
}

impl<'a> AudioInput<'a> {
    /// Mono `samples` in `[-1, 1]` recorded at `sample_rate`.
    pub fn samples(samples: &'a [f32], sample_rate: u32) -> Self {
        Self::Samples {
            samples,
            sample_rate,
        }
    }

    /// The audio as mono samples at `sample_rate`.
    pub(crate) fn load(&self, sample_rate: u32) -> Result<Vec<f32>> {
        let (samples, rate) = match *self {
            Self::Path(path) => {
                let (samples, rate) = load_wav(path)?;
                if rate == sample_rate {
                    return Ok(samples);
                }
                return resample(&samples, rate, sample_rate);
            }
            Self::Samples {
                samples,
                sample_rate,
            } => (samples, sample_rate),
        };
        resample(samples, rate, sample_rate)
    }
}

impl<'a, P: AsRef<Path> + ?Sized> From<&'a P> for AudioInput<'a> {
    fn from(path: &'a P) -> Self {
        Self::Path(path.as_ref())
    }
}

impl<'a> From<(&'a [f32], u32)> for AudioInput<'a> {
    fn from((samples, sample_rate): (&'a [f32], u32)) -> Self {
        Self::samples(samples, sample_rate)
    }
}

impl<'a> From<(&'a Vec<f32>, u32)> for AudioInput<'a> {
    fn from((samples, sample_rate): (&'a Vec<f32>, u32)) -> Self {
        Self::samples(samples, sample_rate)
    }
}

/// How to decode encoder output into tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DecodingStrategy {
    /// Greedy decoding (fast)
    #[default]
    Greedy,
    /// Beam search decoding, NeMo's default algorithm (slower, usually
    /// slightly more accurate). RNN-T models only.
    BeamSearch {
        /// Number of hypotheses kept per frame (at least 1)
        beam_size: usize,
    },
}

/// Options for offline transcription.
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// # use mlx_voice::asr::{EncDecRNNTModel, TranscribeOptions};
/// # let mut model = EncDecRNNTModel::from_pretrained("model.nemo")?;
/// let options = TranscribeOptions::default().beam_search(4).batch_size(8);
/// let transcripts = model.transcribe_with(["a.wav", "b.wav"], &options)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TranscribeOptions {
    /// Decoding strategy; `None` uses the model's configured strategy
    /// (`decoding.strategy` in its config).
    pub decoding: Option<DecodingStrategy>,
    /// Target language of a multilingual (prompt-conditioned) model, e.g.
    /// `"de-DE"`, `"de"` or `"auto"`; `None` keeps the model's current one.
    pub language: Option<String>,
    /// Leave language tags such as `<en-US>` out of text (the default).
    /// Token ids always keep them.
    pub strip_lang_tags: bool,
    /// Files decoded in one batch. Larger batches are faster but use more
    /// memory, since every file is padded to the longest one.
    pub batch_size: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            decoding: None,
            language: None,
            strip_lang_tags: true,
            batch_size: 4,
        }
    }
}

impl TranscribeOptions {
    /// Use `decoding` in place of the model's configured strategy.
    pub fn decoding(mut self, decoding: DecodingStrategy) -> Self {
        self.decoding = Some(decoding);
        self
    }

    /// Use beam search with `beam_size` hypotheses.
    pub fn beam_search(self, beam_size: usize) -> Self {
        self.decoding(DecodingStrategy::BeamSearch { beam_size })
    }

    /// Transcribe in `language` (multilingual models only).
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Keep or strip language tags in text.
    pub fn strip_lang_tags(mut self, strip: bool) -> Self {
        self.strip_lang_tags = strip;
        self
    }

    /// Decode `batch_size` files at a time (at least 1).
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }
}

/// A transcription with token and word timings.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Transcript {
    /// The full text
    pub text: String,
    /// Every decoded token, language tags included
    pub tokens: Vec<Token>,
    /// Words, built from tokens that start with a word boundary (`▁`).
    /// Language tags are left out when they are stripped from the text.
    pub words: Vec<Word>,
    /// The language named by the last language tag, e.g. `"en-US"`, for
    /// multilingual models
    pub language: Option<String>,
}

/// A decoded token.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Token {
    /// Token id in the model's [`Tokenizer`]
    pub id: usize,
    /// Text of the token on its own, without its leading space
    pub text: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
}

/// A word of a transcript.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Word {
    /// The word's text
    pub text: String,
    /// Start time in seconds (of its first token)
    pub start: f32,
    /// End time in seconds (of its last token)
    pub end: f32,
}

impl fmt::Display for Transcript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl Transcript {
    /// Build a transcript from decoded tokens, where each encoder frame lasts
    /// `frame_sec` seconds.
    pub(crate) fn from_tokens(
        tokenizer: &Tokenizer,
        timed: &[TimedToken],
        frame_sec: f32,
        strip_lang_tags: bool,
    ) -> Self {
        let ids: Vec<usize> = timed.iter().map(|t| t.id).collect();
        let tokens: Vec<Token> = timed
            .iter()
            .map(|t| Token {
                id: t.id,
                text: tokenizer.decode(&[t.id], false),
                start: t.start_frame as f32 * frame_sec,
                end: t.end_frame as f32 * frame_sec,
            })
            .collect();

        // Group tokens into words at word starts
        let mut words = Vec::new();
        let mut word: Vec<&Token> = Vec::new();
        let mut flush = |word: &mut Vec<&Token>| {
            if let (Some(first), Some(last)) = (word.first(), word.last()) {
                let ids: Vec<usize> = word.iter().map(|t| t.id).collect();
                let text = tokenizer.decode(&ids, strip_lang_tags).trim().to_string();
                if !text.is_empty() {
                    words.push(Word {
                        text,
                        start: first.start,
                        end: last.end,
                    });
                }
            }
            word.clear();
        };
        for token in &tokens {
            if strip_lang_tags && tokenizer.is_lang_tag(token.id) {
                continue;
            }
            if tokenizer.is_word_start(token.id) {
                flush(&mut word);
            }
            word.push(token);
        }
        flush(&mut word);

        Self {
            text: tokenizer.decode(&ids, strip_lang_tags),
            language: tokenizer.detected_language(&ids),
            tokens,
            words,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_input_conversions() {
        let samples = vec![0.5f32; 4];
        assert_eq!(
            AudioInput::from("a.wav"),
            AudioInput::Path(Path::new("a.wav"))
        );
        assert_eq!(
            AudioInput::from(&String::from("a.wav")),
            AudioInput::Path(Path::new("a.wav"))
        );
        assert_eq!(
            AudioInput::from((&samples, 16_000)),
            AudioInput::samples(&samples, 16_000)
        );
        // Arrays of paths iterate as references to references
        let paths = ["a.wav", "b.wav"];
        let inputs: Vec<AudioInput> = paths.iter().map(AudioInput::from).collect();
        assert_eq!(inputs[1], AudioInput::Path(Path::new("b.wav")));
    }

    #[test]
    fn test_samples_are_resampled() {
        let samples = vec![0.0f32; 48_000];
        let loaded = AudioInput::samples(&samples, 48_000).load(16_000).unwrap();
        assert!(
            (loaded.len() as i64 - 16_000).abs() < 100,
            "{}",
            loaded.len()
        );
        let same = AudioInput::samples(&samples, 16_000).load(16_000).unwrap();
        assert_eq!(same.len(), 48_000);
    }

    #[test]
    fn test_options_builder() {
        let options = TranscribeOptions::default()
            .beam_search(4)
            .language("de")
            .batch_size(0);
        assert_eq!(
            options.decoding,
            Some(DecodingStrategy::BeamSearch { beam_size: 4 })
        );
        assert_eq!(options.language.as_deref(), Some("de"));
        assert_eq!(options.batch_size, 1);
        assert!(options.strip_lang_tags);
    }
}
