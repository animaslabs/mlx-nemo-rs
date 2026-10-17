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

/// Whether `c` is a punctuation mark (Unicode general category P): ASCII,
/// Latin-1, General Punctuation and CJK punctuation.
pub fn is_punctuation_char(c: char) -> bool {
    match c {
        '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~' => false,
        _ if c.is_ascii_punctuation() => true,
        '¡' | '§' | '«' | '¶' | '·' | '»' | '¿' => true,
        '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{2043}' | '\u{2045}'..='\u{2051}' => true,
        '\u{2053}'..='\u{205E}' | '\u{3001}'..='\u{3003}' | '\u{3008}'..='\u{3011}' => true,
        _ => false,
    }
}

/// For each token id, whether it decodes to a single punctuation mark (see
/// [`is_punctuation_char`]).
pub fn punctuation_table(tokenizer: &SentencePieceProcessor) -> Vec<bool> {
    (0..tokenizer.len())
        .map(|id| {
            let text = tokenizer.decode_piece_ids(&[id as u32]).unwrap_or_default();
            let mut chars = text.chars();
            matches!((chars.next(), chars.next()), (Some(c), None) if is_punctuation_char(c))
        })
        .collect()
}

/// Whether `piece` is a language tag such as `<en-US>`, which prompt-conditioned
/// models emit after the terminal punctuation (NeMo's `<[a-z]{2}-[A-Z]{2}>`).
pub fn is_lang_tag(piece: &str) -> bool {
    let b = piece.as_bytes();
    b.len() == 7
        && b[0] == b'<'
        && b[1..3].iter().all(u8::is_ascii_lowercase)
        && b[3] == b'-'
        && b[4..6].iter().all(u8::is_ascii_uppercase)
        && b[6] == b'>'
}

/// For each token id, whether it is a language tag (see [`is_lang_tag`]).
pub fn lang_tag_table(tokenizer: &SentencePieceProcessor) -> Vec<bool> {
    (0..tokenizer.len())
        .map(|id| is_lang_tag(&tokenizer.decode_piece_ids(&[id as u32]).unwrap_or_default()))
        .collect()
}

/// The language named by the last language tag in `tokens`, e.g. `"en-US"`.
///
/// `lang_tags` comes from [`lang_tag_table`].
pub fn last_lang_tag(
    tokenizer: &SentencePieceProcessor,
    lang_tags: &[bool],
    tokens: &[usize],
) -> Option<String> {
    let &tag = tokens
        .iter()
        .rev()
        .find(|&&t| lang_tags.get(t).copied().unwrap_or(false))?;
    let piece = tokenizer.decode_piece_ids(&[tag as u32]).ok()?;
    Some(piece[1..piece.len() - 1].to_string())
}

/// `text` without language tags or the whitespace before them, like NeMo's
/// `strip_lang_tags` (`re.sub(r'\s*<[a-z]{2}-[A-Z]{2}>', '', text)`).
pub fn strip_lang_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find('<') {
        let tag_end = pos + 7;
        if rest.get(pos..tag_end).is_some_and(is_lang_tag) {
            out.push_str(rest[..pos].trim_end());
            rest = &rest[tag_end..];
        } else {
            out.push_str(&rest[..pos + 1]);
            rest = &rest[pos + 1..];
        }
    }
    out.push_str(rest);
    out
}

/// Text of `tokens`, optionally without language tags (see [`strip_lang_tags`]).
pub fn decode_text(
    tokenizer: &SentencePieceProcessor,
    tokens: &[usize],
    strip_tags: bool,
) -> String {
    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode_piece_ids(&ids).unwrap_or_default();
    if strip_tags {
        strip_lang_tags(&text)
    } else {
        text
    }
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
    /// `word_start` comes from [`word_start_table`]. With `strip_tags`,
    /// language tags are left out (see [`strip_lang_tags`]).
    pub fn update(
        &mut self,
        tokenizer: &SentencePieceProcessor,
        word_start: &[bool],
        strip_tags: bool,
        tokens: &[usize],
    ) -> String {
        let starts_word = |token: usize| word_start.get(token).copied().unwrap_or(false);
        // SentencePiece drops the space a leading "▁" decodes to, so put it back
        // between segments.
        // A tag never starts a word, so it lands in the same segment as the
        // space before it and both strip together.
        let append = |text: &mut String, segment: &[usize]| {
            let mut decoded = String::new();
            if !text.is_empty() && starts_word(segment[0]) {
                decoded.push(' ');
            }
            decoded.push_str(&decode_text(tokenizer, segment, false));
            if strip_tags {
                decoded = strip_lang_tags(&decoded);
            }
            text.push_str(&decoded);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_lang_tags() {
        assert_eq!(
            strip_lang_tags("Hello world. <en-US> Next one.<de-DE>"),
            "Hello world. Next one."
        );
        assert_eq!(strip_lang_tags(" <en-US>"), "");
        assert_eq!(strip_lang_tags("a <b> <unk> 1<2"), "a <b> <unk> 1<2");
        assert_eq!(strip_lang_tags("trailing <"), "trailing <");
    }

    #[test]
    fn test_is_punctuation_char() {
        for c in [',', '.', '?', '!', '\'', '-', '%', '/', '¿', '¡', '«', '…', '—', '。'] {
            assert!(is_punctuation_char(c), "{c}");
        }
        for c in ['a', 'é', '1', ' ', '$', '+', '<', '~', '€'] {
            assert!(!is_punctuation_char(c), "{c}");
        }
    }

    #[test]
    fn test_is_lang_tag() {
        for tag in ["<en-US>", "<de-DE>", "<bg-BG>"] {
            assert!(is_lang_tag(tag), "{tag}");
        }
        for piece in ["<unk>", "<en-us>", "en-US", "<eng-US>", "▁<en-US>", "<"] {
            assert!(!is_lang_tag(piece), "{piece}");
        }
    }
}
