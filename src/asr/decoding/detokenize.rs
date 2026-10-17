//! Incremental SentencePiece detokenization for streaming.
//!
//! Re-decoding every accumulated token on each streaming step costs O(N) per
//! step and O(N²) over a long file. Words never change once the next word has
//! started, so only the tokens of the last word are re-decoded.

use sentencepiece::SentencePieceProcessor;

/// For each token id, whether its piece starts a new word (a leading "▁").
pub fn word_start_table(tokenizer: &SentencePieceProcessor) -> Vec<bool> {
    (0..tokenizer.len())
        .map(|id| {
            // A piece with a leading "▁" decodes with a space between copies
            let id = id as u32;
            let one = tokenizer.decode_piece_ids(&[id]).unwrap_or_default();
            let two = tokenizer.decode_piece_ids(&[id, id]).unwrap_or_default();
            two.len() > 2 * one.len()
        })
        .collect()
}

/// Decodes a growing token sequence, re-decoding only its last word.
#[derive(Debug, Clone, Default)]
pub struct IncrementalDetokenizer {
    /// Text of every token before `tail_start`
    committed: String,
    /// Index of the first token not yet in `committed` (a word start)
    tail_start: usize,
}

impl IncrementalDetokenizer {
    /// Forget all decoded text.
    pub fn reset(&mut self) {
        self.committed.clear();
        self.tail_start = 0;
    }

    /// Text of `tokens`, which must extend the tokens of the previous call.
    ///
    /// `word_start` comes from [`word_start_table`].
    pub fn update(
        &mut self,
        tokenizer: &SentencePieceProcessor,
        word_start: &[bool],
        tokens: &[usize],
    ) -> String {
        let starts_word = |token: usize| word_start.get(token).copied().unwrap_or(false);
        let decode = |tokens: &[usize]| {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            tokenizer.decode_piece_ids(&ids).unwrap_or_default()
        };
        // SentencePiece drops the space a leading "▁" decodes to, so put it back
        // between segments.
        let append = |text: &mut String, segment: &[usize]| {
            if !text.is_empty() && starts_word(segment[0]) {
                text.push(' ');
            }
            text.push_str(&decode(segment));
        };

        // Commit every word before the last word start
        if let Some(last_start) = (self.tail_start + 1..tokens.len())
            .rev()
            .find(|&i| starts_word(tokens[i]))
        {
            append(&mut self.committed, &tokens[self.tail_start..last_start]);
            self.tail_start = last_start;
        }

        let mut text = self.committed.clone();
        if self.tail_start < tokens.len() {
            append(&mut text, &tokens[self.tail_start..]);
        }
        text
    }
}
