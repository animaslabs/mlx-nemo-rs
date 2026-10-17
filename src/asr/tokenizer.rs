//! SentencePiece tokenizer used by the ASR models.

use std::fmt;
use std::path::Path;

use sentencepiece::SentencePieceProcessor;

use crate::asr::decoding::{decode_text, lang_tag_table, last_lang_tag, word_start_table};
use crate::core::error::{Error, Result};

/// A model's SentencePiece tokenizer, for turning token ids into text.
///
/// Token ids are the ones the models emit, e.g. in
/// [`Transcript::tokens`](crate::asr::Transcript::tokens). The
/// blank token is not part of the vocabulary: its id is
/// [`vocab_size`](Self::vocab_size) for RNN-T models.
pub struct Tokenizer {
    processor: SentencePieceProcessor,
    /// For each token id, whether its piece starts a new word
    word_start: Vec<bool>,
    /// For each token id, whether it is a language tag such as `<en-US>`
    lang_tags: Vec<bool>,
}

impl Tokenizer {
    /// Load a SentencePiece model file (e.g. `tokenizer.model`).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let processor = SentencePieceProcessor::open(path)
            .map_err(|e| Error::tokenizer(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self {
            word_start: word_start_table(&processor),
            lang_tags: lang_tag_table(&processor),
            processor,
        })
    }

    /// Number of tokens in the vocabulary, not counting blank.
    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    /// Text of `ids`. With `strip_lang_tags`, language tags such as `<en-US>`
    /// and the whitespace before them are left out.
    ///
    /// Ids outside the vocabulary (such as blank) decode to an empty string.
    pub fn decode(&self, ids: &[usize], strip_lang_tags: bool) -> String {
        decode_text(&self.processor, ids, strip_lang_tags)
    }

    /// Token ids of `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let pieces = self
            .processor
            .encode(text)
            .map_err(|e| Error::tokenizer(format!("Failed to encode text: {}", e)))?;
        Ok(pieces.into_iter().map(|p| p.id as usize).collect())
    }

    /// Id of a vocabulary piece, e.g. `"▁the"`.
    pub fn piece_to_id(&self, piece: &str) -> Option<usize> {
        self.processor
            .piece_to_id(piece)
            .ok()
            .flatten()
            .map(|id| id as usize)
    }

    /// Whether token `id` starts a new word (its piece has a leading `▁`).
    pub fn is_word_start(&self, id: usize) -> bool {
        self.word_start.get(id).copied().unwrap_or(false)
    }

    /// Whether token `id` is a language tag such as `<en-US>`, which
    /// prompt-conditioned models emit after each utterance.
    pub fn is_lang_tag(&self, id: usize) -> bool {
        self.lang_tags.get(id).copied().unwrap_or(false)
    }

    /// The language named by the last language tag in `ids`, e.g. `"en-US"`.
    pub fn detected_language(&self, ids: &[usize]) -> Option<String> {
        last_lang_tag(&self.processor, &self.lang_tags, ids)
    }

    /// The underlying SentencePiece processor.
    pub(crate) fn processor(&self) -> &SentencePieceProcessor {
        &self.processor
    }

    /// For each token id, whether it starts a new word.
    pub(crate) fn word_start_table(&self) -> &[bool] {
        &self.word_start
    }
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}
