//! WordPiece tokenizer (BERT-family subword tokenisation).
//!
//! WordPiece differs from byte-pair encoding ([`crate::tokenizer::bpe`]) in two
//! important ways:
//!
//! 1. **Greedy longest-match-first**: instead of iteratively merging the
//!    lowest-rank pair, WordPiece walks left-to-right over a *pre-tokenised
//!    word* and, at each position, consumes the **longest** vocabulary entry
//!    that matches a prefix of the remaining characters.  Continuation pieces
//!    (every piece after the first in a word) are prefixed with a continuation
//!    marker (`##` in the original BERT vocabulary).
//! 2. **`[UNK]` fallback**: if no vocabulary prefix matches at some position,
//!    or the word is longer than `max_input_chars_per_word`, the *entire word*
//!    is replaced by a single unknown token.  This is the exact behaviour of
//!    HuggingFace `WordpieceTokenizer`.
//!
//! Reference: Wu et al. 2016, "Google's Neural Machine Translation System"
//! (§4.1, the WordPiece model) and the `transformers`
//! `BertTokenizer`/`WordpieceTokenizer` implementation.
//!
//! # Pre-tokenisation
//!
//! WordPiece operates on already-whitespace-split *words*.  This module
//! provides [`WordPieceTokenizer::encode`] which performs a simple
//! whitespace + ASCII-punctuation split (matching BERT's basic tokeniser for
//! the common case) and applies the WordPiece model to each word, as well as
//! [`WordPieceTokenizer::encode_word`] for callers that have already performed
//! their own pre-tokenisation.
//!
//! # Vocabulary layout
//!
//! The vocabulary is supplied as an ordered list of `(piece, id)` entries.
//! Whole-word pieces appear verbatim (`"play"`); continuation pieces carry the
//! continuation marker (`"##ing"`).  Special tokens such as `[UNK]`, `[CLS]`,
//! `[SEP]`, `[PAD]` and `[MASK]` are ordinary vocabulary entries; the unknown
//! token id is recorded separately so the matcher can emit it on failure.

use std::collections::HashMap;

use crate::error::{LmError, LmResult};

// ─── Configuration ───────────────────────────────────────────────────────────

/// The default continuation marker used by BERT-family vocabularies.
pub const DEFAULT_CONTINUATION_PREFIX: &str = "##";

/// The default unknown-token surface form used by BERT-family vocabularies.
pub const DEFAULT_UNK_TOKEN: &str = "[UNK]";

/// The default cap on the number of characters in a single word.  Words longer
/// than this are emitted as a single `[UNK]` (matching HuggingFace's default).
pub const DEFAULT_MAX_INPUT_CHARS_PER_WORD: usize = 100;

// ─── WordPieceTokenizer ──────────────────────────────────────────────────────

/// A WordPiece subword tokenizer.
///
/// Construct one with [`WordPieceTokenizer::new`] (full control) or
/// [`WordPieceTokenizer::from_pieces`] (convenience for tests).
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    /// Surface form → token id.  Continuation pieces include the marker.
    vocab: HashMap<String, u32>,
    /// Reverse map for decoding.
    id_to_piece: Vec<String>,
    /// Continuation prefix (e.g. `"##"`).
    continuation_prefix: String,
    /// The token id emitted when a word cannot be tokenised.
    unk_id: u32,
    /// Words longer (in `char`s) than this become a single `[UNK]`.
    max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Build a tokenizer from an ordered vocabulary.
    ///
    /// `pieces[i]` is the surface form of token id `i`.  Continuation pieces
    /// **must already carry** the `continuation_prefix`.  `unk_token` is the
    /// surface form of the unknown token and must be present in `pieces`.
    ///
    /// # Errors
    ///
    /// * [`LmError::EmptyInput`] if `pieces` is empty.
    /// * [`LmError::InvalidConfig`] if a piece is duplicated, if
    ///   `unk_token` is absent, or if `continuation_prefix` is empty.
    pub fn new(
        pieces: Vec<String>,
        continuation_prefix: impl Into<String>,
        unk_token: &str,
        max_input_chars_per_word: usize,
    ) -> LmResult<Self> {
        if pieces.is_empty() {
            return Err(LmError::EmptyInput {
                context: "WordPieceTokenizer::new pieces",
            });
        }
        let continuation_prefix = continuation_prefix.into();
        if continuation_prefix.is_empty() {
            return Err(LmError::InvalidConfig {
                msg: "WordPiece continuation prefix must be non-empty".into(),
            });
        }
        if max_input_chars_per_word == 0 {
            return Err(LmError::InvalidConfig {
                msg: "max_input_chars_per_word must be > 0".into(),
            });
        }

        let mut vocab = HashMap::with_capacity(pieces.len());
        for (id, piece) in pieces.iter().enumerate() {
            if vocab.insert(piece.clone(), id as u32).is_some() {
                return Err(LmError::InvalidConfig {
                    msg: format!("duplicate WordPiece surface form at id {id}: {piece:?}"),
                });
            }
        }

        let unk_id = *vocab.get(unk_token).ok_or_else(|| LmError::InvalidConfig {
            msg: format!("unknown token {unk_token:?} missing from WordPiece vocabulary"),
        })?;

        Ok(Self {
            vocab,
            id_to_piece: pieces,
            continuation_prefix,
            unk_id,
            max_input_chars_per_word,
        })
    }

    /// Convenience constructor using the BERT defaults (`"##"` continuation,
    /// `"[UNK]"` unknown token, 100-char word cap).
    ///
    /// `pieces` must contain `"[UNK]"`.
    pub fn from_pieces(pieces: Vec<String>) -> LmResult<Self> {
        Self::new(
            pieces,
            DEFAULT_CONTINUATION_PREFIX,
            DEFAULT_UNK_TOKEN,
            DEFAULT_MAX_INPUT_CHARS_PER_WORD,
        )
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_piece.len()
    }

    /// The unknown-token id.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Look up a surface form's id (continuation pieces must include the marker).
    pub fn piece_to_id(&self, piece: &str) -> Option<u32> {
        self.vocab.get(piece).copied()
    }

    // ── Core matcher ─────────────────────────────────────────────────────

    /// Tokenise a **single pre-tokenised word** into WordPiece ids.
    ///
    /// Implements greedy longest-match-first: at each cursor position the
    /// longest vocabulary prefix is consumed.  The first piece is matched
    /// verbatim; every subsequent piece is matched against the continuation
    /// form (`prefix ++ substring`).  If matching fails at any position, the
    /// whole word collapses to a single `[UNK]` id.
    ///
    /// An empty word yields an empty token list.
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }
        // WordPiece operates over Unicode scalar values, not bytes, so that a
        // multi-byte character is never split mid-way.
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return vec![self.unk_id];
        }

        let mut out = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            // Find the longest prefix [start, end) present in the vocabulary.
            let mut end = chars.len();
            let mut matched_id: Option<u32> = None;
            while start < end {
                // Build the candidate surface form.
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start == 0 {
                    substr
                } else {
                    let mut s =
                        String::with_capacity(self.continuation_prefix.len() + substr.len());
                    s.push_str(&self.continuation_prefix);
                    s.push_str(&substr);
                    s
                };
                if let Some(&id) = self.vocab.get(&candidate) {
                    matched_id = Some(id);
                    break;
                }
                end -= 1;
            }

            match matched_id {
                // No prefix matched at this position → whole word is unknown.
                None => return vec![self.unk_id],
                Some(id) => {
                    out.push(id);
                    start = end;
                }
            }
        }
        out
    }

    /// Tokenise free text: split on whitespace and ASCII punctuation, then run
    /// the WordPiece model on each resulting word.
    ///
    /// Punctuation characters become their own single-character words (the
    /// behaviour of BERT's basic tokeniser), so `"don't!"` is pre-split into
    /// `["don", "'", "t", "!"]` before WordPiece matching.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in basic_pretokenize(text) {
            out.extend(self.encode_word(&word));
        }
        out
    }

    // ── Decode ───────────────────────────────────────────────────────────

    /// Decode token ids back to a string.
    ///
    /// Continuation pieces (those carrying the marker) are concatenated onto
    /// the previous token with the marker stripped; whole-word pieces are
    /// separated by a single space.  This reverses [`Self::encode`] up to the
    /// loss of original whitespace (the standard WordPiece detokenisation).
    ///
    /// # Errors
    ///
    /// [`LmError::OutOfVocab`] if any id is outside the vocabulary range.
    pub fn decode(&self, ids: &[u32]) -> LmResult<String> {
        let mut out = String::new();
        for (i, &id) in ids.iter().enumerate() {
            let piece = self
                .id_to_piece
                .get(id as usize)
                .ok_or(LmError::OutOfVocab { token: id })?;
            if let Some(rest) = piece.strip_prefix(&self.continuation_prefix) {
                out.push_str(rest);
            } else {
                if i != 0 {
                    out.push(' ');
                }
                out.push_str(piece);
            }
        }
        Ok(out)
    }
}

// ─── Pre-tokenisation helper ─────────────────────────────────────────────────

/// Split text into words on whitespace, additionally breaking out runs of
/// ASCII punctuation as individual single-character words.
///
/// This mirrors the whitespace + punctuation behaviour of BERT's
/// `BasicTokenizer` for ASCII input (it does not perform lower-casing,
/// accent-stripping or CJK handling, which are orthogonal normalisation steps).
pub fn basic_pretokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        } else if is_punctuation(ch) {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            words.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

/// Whether `ch` is treated as a standalone punctuation token.
///
/// Matches HuggingFace `_is_punctuation`: all ASCII non-alphanumeric printable
/// characters plus any Unicode code point in a punctuation category.
fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;
    // ASCII ranges that are punctuation in BERT's tokeniser.
    if (33..=47).contains(&cp)      // ! " # $ % & ' ( ) * + , - . /
        || (58..=64).contains(&cp)  // : ; < = > ? @
        || (91..=96).contains(&cp)  // [ \ ] ^ _ `
        || (123..=126).contains(&cp)
    {
        return true;
    }
    ch.is_ascii_punctuation()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// BERT-style vocabulary: a handful of whole words and continuation pieces
    /// plus the `[UNK]` token.
    fn bert_like() -> WordPieceTokenizer {
        // ids assigned by order.
        let pieces: Vec<String> = [
            "[UNK]",  // 0
            "[CLS]",  // 1
            "[SEP]",  // 2
            "play",   // 3
            "##ing",  // 4
            "##ed",   // 5
            "##er",   // 6
            "un",     // 7
            "##aff",  // 8
            "##able", // 9
            "!",      // 10
            "'",      // 11
            "s",      // 12
            "t",      // 13
            "don",    // 14
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        WordPieceTokenizer::from_pieces(pieces).expect("bert-like vocab should build")
    }

    #[test]
    fn builds_and_reports_unk() {
        let t = bert_like();
        assert_eq!(t.vocab_size(), 15);
        assert_eq!(t.unk_id(), 0);
        assert_eq!(t.piece_to_id("play"), Some(3));
        assert_eq!(t.piece_to_id("##ing"), Some(4));
    }

    #[test]
    fn empty_vocab_rejected() {
        assert!(matches!(
            WordPieceTokenizer::from_pieces(vec![]),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn missing_unk_rejected() {
        let pieces = vec!["a".to_string(), "b".to_string()];
        assert!(matches!(
            WordPieceTokenizer::from_pieces(pieces),
            Err(LmError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn duplicate_piece_rejected() {
        let pieces = vec!["[UNK]".to_string(), "a".to_string(), "a".to_string()];
        assert!(matches!(
            WordPieceTokenizer::from_pieces(pieces),
            Err(LmError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn greedy_longest_match_single_word() {
        // "playing" = "play" + "##ing".
        let t = bert_like();
        assert_eq!(t.encode_word("playing"), vec![3, 4]);
    }

    #[test]
    fn greedy_prefers_longer_prefix() {
        // "unaffable" = "un" + "##aff" + "##able".
        let t = bert_like();
        assert_eq!(t.encode_word("unaffable"), vec![7, 8, 9]);
    }

    #[test]
    fn whole_word_exact_match() {
        let t = bert_like();
        assert_eq!(t.encode_word("play"), vec![3]);
    }

    #[test]
    fn unknown_word_collapses_to_unk() {
        // "xyz" has no matching prefix → single [UNK].
        let t = bert_like();
        assert_eq!(t.encode_word("xyz"), vec![0]);
    }

    #[test]
    fn partial_failure_collapses_whole_word_to_unk() {
        // "playz": "play" matches, but "z" continuation fails → the *whole*
        // word is [UNK] (canonical WordPiece behaviour, not [play, UNK]).
        let t = bert_like();
        assert_eq!(t.encode_word("playz"), vec![0]);
    }

    #[test]
    fn empty_word_is_empty() {
        let t = bert_like();
        assert!(t.encode_word("").is_empty());
    }

    #[test]
    fn over_long_word_is_unk() {
        // Include the continuation form "##a" so that a multi-'a' word is
        // genuinely tokenisable (first 'a' whole-word, the rest continuation).
        let pieces = vec![
            "[UNK]".to_string(), // 0
            "a".to_string(),     // 1
            "##a".to_string(),   // 2
        ];
        let t = WordPieceTokenizer::new(pieces, "##", "[UNK]", 4)
            .expect("short-cap vocab should build");
        // 5 'a's exceeds cap=4 → single [UNK] even though it is otherwise
        // fully tokenisable.
        assert_eq!(t.encode_word("aaaaa"), vec![0]);
        // 4 'a's is within cap → 'a' then three '##a' continuation pieces.
        assert_eq!(t.encode_word("aaaa"), vec![1, 2, 2, 2]);
    }

    #[test]
    fn encode_splits_on_whitespace_and_punctuation() {
        // "don't!" → ["don", "'", "t", "!"] → [14, 11, 13, 10].
        let t = bert_like();
        assert_eq!(t.encode("don't!"), vec![14, 11, 13, 10]);
    }

    #[test]
    fn encode_multiword() {
        // "playing play" → [play,##ing] then [play] → [3,4,3].
        let t = bert_like();
        assert_eq!(t.encode("playing play"), vec![3, 4, 3]);
    }

    #[test]
    fn decode_joins_continuations() {
        let t = bert_like();
        // [play, ##ing] → "playing".
        assert_eq!(t.decode(&[3, 4]).expect("decode should succeed"), "playing");
        // [play, play] → "play play" (whole words separated by space).
        assert_eq!(
            t.decode(&[3, 3]).expect("decode should succeed"),
            "play play"
        );
    }

    #[test]
    fn decode_out_of_range_errors() {
        let t = bert_like();
        assert!(matches!(
            t.decode(&[999]),
            Err(LmError::OutOfVocab { token: 999 })
        ));
    }

    #[test]
    fn encode_decode_roundtrip_known_words() {
        let t = bert_like();
        let ids = t.encode("playing unaffable");
        let text = t.decode(&ids).expect("decode should succeed");
        assert_eq!(text, "playing unaffable");
    }

    #[test]
    fn unicode_word_not_split_midcharacter() {
        // A multi-byte char ('é') as a whole-word piece: it must match as a
        // single scalar, never as a partial byte.
        let pieces = vec!["[UNK]".to_string(), "é".to_string()];
        let t = WordPieceTokenizer::from_pieces(pieces).expect("unicode vocab should build");
        assert_eq!(t.encode_word("é"), vec![1]);
    }
}
