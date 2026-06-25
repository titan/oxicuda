//! Unigram language-model subword tokenizer (SentencePiece "unigram" model).
//!
//! Unlike byte-pair encoding ([`crate::tokenizer::bpe`]) and WordPiece
//! ([`crate::tokenizer::wordpiece`]), the Unigram model does **not** apply a
//! sequence of merge rules.  Instead every vocabulary piece carries a
//! log-probability, and the tokenisation of a string is defined as the
//! **segmentation that maximises the sum of piece log-probabilities** — the
//! single most likely way to cover the input with vocabulary pieces.
//!
//! Finding that segmentation is an instance of the shortest-path /
//! maximum-likelihood-path problem over the lattice of all possible pieces and
//! is solved here exactly with the **Viterbi forward dynamic program**:
//!
//! ```text
//!   best[0]      = 0
//!   best[j]      = max over pieces p ending at j of  best[j - len(p)] + logp(p)
//!   backptr[j]   = the (start, piece_id) achieving that maximum
//! ```
//!
//! After the forward pass, the optimal pieces are recovered by following
//! `backptr` from the end of the string back to the start.  This is the same
//! algorithm SentencePiece uses at inference time (its `Viterbi` encoder).
//!
//! # Unknown characters
//!
//! Any single input character not covered by *any* vocabulary piece would make
//! the lattice unreachable.  To keep the model total, every position can always
//! emit the configured **unknown piece** (covering exactly one character) at a
//! fixed penalty.  This mirrors SentencePiece's `<unk>` handling, where the
//! unknown surface form receives `unk_score = unk_log_prob`.
//!
//! Reference: Kudo 2018, "Subword Regularization: Improving Neural Network
//! Translation Models with Multiple Subword Candidates" (the unigram LM), and
//! the SentencePiece reference implementation.

use std::collections::HashMap;

use crate::error::{LmError, LmResult};

// ─── UnigramTokenizer ────────────────────────────────────────────────────────

/// A Unigram-LM subword tokenizer.
///
/// Construct with [`UnigramTokenizer::new`].  Each vocabulary entry is a
/// `(surface_form, log_probability)` pair; larger (less negative)
/// log-probabilities make a piece more likely to be chosen.
#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    /// `id_to_piece[i]` — surface form of token `i`.
    id_to_piece: Vec<String>,
    /// `id_to_score[i]` — log-probability of token `i`.
    id_to_score: Vec<f64>,
    /// Surface form → id.
    piece_to_id: HashMap<String, u32>,
    /// Id of the unknown piece (covers a single unknown character).
    unk_id: u32,
    /// Maximum piece length in `char`s — bounds the inner Viterbi loop.
    max_piece_chars: usize,
}

impl UnigramTokenizer {
    // ── Constructor ──────────────────────────────────────────────────────

    /// Build a Unigram tokenizer from a scored vocabulary.
    ///
    /// `pieces[i] = (surface, log_prob)`.  `unk_piece` is the surface form of
    /// the unknown token (typically `"<unk>"`) and must be present in `pieces`.
    /// Its score is used as the per-character penalty for uncoverable input.
    ///
    /// # Errors
    ///
    /// * [`LmError::EmptyInput`] if `pieces` is empty.
    /// * [`LmError::InvalidConfig`] if a surface form is duplicated, if any
    ///   score is non-finite, or if `unk_piece` is absent.
    pub fn new(pieces: Vec<(String, f64)>, unk_piece: &str) -> LmResult<Self> {
        if pieces.is_empty() {
            return Err(LmError::EmptyInput {
                context: "UnigramTokenizer::new pieces",
            });
        }

        let mut id_to_piece = Vec::with_capacity(pieces.len());
        let mut id_to_score = Vec::with_capacity(pieces.len());
        let mut piece_to_id = HashMap::with_capacity(pieces.len());
        let mut max_piece_chars = 0usize;

        for (id, (surface, score)) in pieces.into_iter().enumerate() {
            if !score.is_finite() {
                return Err(LmError::InvalidConfig {
                    msg: format!("non-finite Unigram score for piece {surface:?}"),
                });
            }
            if surface.is_empty() {
                return Err(LmError::InvalidConfig {
                    msg: "Unigram vocabulary contains an empty surface form".into(),
                });
            }
            let n_chars = surface.chars().count();
            max_piece_chars = max_piece_chars.max(n_chars);
            if piece_to_id.insert(surface.clone(), id as u32).is_some() {
                return Err(LmError::InvalidConfig {
                    msg: format!("duplicate Unigram surface form at id {id}: {surface:?}"),
                });
            }
            id_to_piece.push(surface);
            id_to_score.push(score);
        }

        let unk_id = *piece_to_id
            .get(unk_piece)
            .ok_or_else(|| LmError::InvalidConfig {
                msg: format!("unknown piece {unk_piece:?} missing from Unigram vocabulary"),
            })?;

        Ok(Self {
            id_to_piece,
            id_to_score,
            piece_to_id,
            unk_id,
            max_piece_chars,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_piece.len()
    }

    /// The unknown-piece id.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Log-probability of a token id, or `None` if out of range.
    pub fn score(&self, id: u32) -> Option<f64> {
        self.id_to_score.get(id as usize).copied()
    }

    /// Surface form → id lookup.
    pub fn piece_to_id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    // ── Viterbi encode ───────────────────────────────────────────────────

    /// Encode `text` into the maximum-likelihood token sequence.
    ///
    /// Runs the Viterbi forward DP over the lattice of all vocabulary pieces
    /// matching substrings of `text`, with a single-character `<unk>` fallback
    /// at every position so the lattice is always reachable.  Returns the
    /// token ids of the optimal segmentation.
    ///
    /// An empty string yields an empty token list.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let (ids, _score) = self.encode_with_score(text);
        ids
    }

    /// Like [`Self::encode`] but also returns the total log-probability of the
    /// chosen segmentation (the Viterbi path score).  Useful for ranking and
    /// for tests that assert a specific path was selected.
    ///
    /// For the empty string the score is `0.0`.
    pub fn encode_with_score(&self, text: &str) -> (Vec<u32>, f64) {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return (Vec::new(), 0.0);
        }

        // Byte offset of each char boundary so we can slice substrings cheaply
        // without re-walking the string. `char_byte[i]` is the byte index where
        // `chars[i]` begins; `char_byte[n]` is `text.len()`.
        let mut char_byte = Vec::with_capacity(n + 1);
        let mut acc = 0usize;
        char_byte.push(0);
        for ch in &chars {
            acc += ch.len_utf8();
            char_byte.push(acc);
        }

        // best[j] = best total log-prob to cover chars[0..j].
        // back[j]  = (start index, token id) of the last piece ending at j.
        let neg_inf = f64::NEG_INFINITY;
        let mut best = vec![neg_inf; n + 1];
        let mut back: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best[0] = 0.0;

        let unk_score = self.id_to_score[self.unk_id as usize];

        for end in 1..=n {
            // The longest piece we need to consider ends at `end` and starts no
            // earlier than `end - max_piece_chars`.
            let earliest = end.saturating_sub(self.max_piece_chars.max(1));
            for start in earliest..end {
                if best[start] == neg_inf {
                    continue; // unreachable prefix
                }
                let substr = &text[char_byte[start]..char_byte[end]];
                if let Some(&id) = self.piece_to_id.get(substr) {
                    let cand = best[start] + self.id_to_score[id as usize];
                    if cand > best[end] {
                        best[end] = cand;
                        back[end] = Some((start, id));
                    }
                }
            }

            // Single-character <unk> fallback: always available, covers
            // chars[end-1..end].
            let start = end - 1;
            if best[start] != neg_inf {
                let cand = best[start] + unk_score;
                if cand > best[end] {
                    best[end] = cand;
                    back[end] = Some((start, self.unk_id));
                }
            }
        }

        // Backtrack from n to 0 to recover the pieces, then reverse.
        let mut ids_rev = Vec::new();
        let mut pos = n;
        while pos > 0 {
            match back[pos] {
                Some((start, id)) => {
                    ids_rev.push(id);
                    pos = start;
                }
                // Should be unreachable thanks to the <unk> fallback, but keep
                // the loop total: emit <unk> for one char and continue.
                None => {
                    ids_rev.push(self.unk_id);
                    pos -= 1;
                }
            }
        }
        ids_rev.reverse();
        (ids_rev, best[n])
    }

    // ── Decode ───────────────────────────────────────────────────────────

    /// Decode token ids back to a string by concatenating their surface forms.
    ///
    /// # Errors
    ///
    /// [`LmError::OutOfVocab`] if any id is outside the vocabulary range.
    pub fn decode(&self, ids: &[u32]) -> LmResult<String> {
        let mut out = String::new();
        for &id in ids {
            let piece = self
                .id_to_piece
                .get(id as usize)
                .ok_or(LmError::OutOfVocab { token: id })?;
            out.push_str(piece);
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny scored vocabulary over the alphabet {a,b,c} plus some multi-char
    /// pieces.  Scores are log-probabilities (higher = more likely).
    fn toy() -> UnigramTokenizer {
        // Single chars are cheap-ish; multi-char pieces are made more probable
        // so the Viterbi path prefers them when they fit.
        let pieces = vec![
            ("<unk>".to_string(), -10.0), // 0
            ("a".to_string(), -2.0),      // 1
            ("b".to_string(), -2.0),      // 2
            ("c".to_string(), -2.0),      // 3
            ("ab".to_string(), -1.0),     // 4  (cheaper than a+b = -4)
            ("bc".to_string(), -1.0),     // 5
            ("abc".to_string(), -0.5),    // 6  (cheapest of all)
        ];
        UnigramTokenizer::new(pieces, "<unk>").expect("toy unigram vocab should build")
    }

    #[test]
    fn builds_and_reports() {
        let t = toy();
        assert_eq!(t.vocab_size(), 7);
        assert_eq!(t.unk_id(), 0);
        assert_eq!(t.piece_to_id("abc"), Some(6));
        assert_eq!(t.score(6), Some(-0.5));
    }

    #[test]
    fn empty_vocab_rejected() {
        assert!(matches!(
            UnigramTokenizer::new(vec![], "<unk>"),
            Err(LmError::EmptyInput { .. })
        ));
    }

    #[test]
    fn missing_unk_rejected() {
        let pieces = vec![("a".to_string(), -1.0)];
        assert!(matches!(
            UnigramTokenizer::new(pieces, "<unk>"),
            Err(LmError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn non_finite_score_rejected() {
        let pieces = vec![("<unk>".to_string(), -10.0), ("a".to_string(), f64::NAN)];
        assert!(matches!(
            UnigramTokenizer::new(pieces, "<unk>"),
            Err(LmError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn duplicate_piece_rejected() {
        let pieces = vec![
            ("<unk>".to_string(), -10.0),
            ("a".to_string(), -1.0),
            ("a".to_string(), -2.0),
        ];
        assert!(matches!(
            UnigramTokenizer::new(pieces, "<unk>"),
            Err(LmError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn empty_string_encodes_empty() {
        let t = toy();
        let (ids, score) = t.encode_with_score("");
        assert!(ids.is_empty());
        assert_eq!(score, 0.0);
    }

    #[test]
    fn viterbi_picks_single_longest_piece() {
        // "abc": options include [a,b,c]=-6, [ab,c]=-3, [a,bc]=-3, [abc]=-0.5.
        // The single "abc" piece (-0.5) is optimal.
        let t = toy();
        let (ids, score) = t.encode_with_score("abc");
        assert_eq!(ids, vec![6]);
        assert!((score - (-0.5)).abs() < 1e-9, "score={score}");
    }

    #[test]
    fn viterbi_chooses_best_of_two_paths() {
        // Build a vocab where "ab" then "cd" beats other splits.
        let pieces = vec![
            ("<unk>".to_string(), -20.0),
            ("a".to_string(), -5.0),
            ("b".to_string(), -5.0),
            ("c".to_string(), -5.0),
            ("d".to_string(), -5.0),
            ("ab".to_string(), -1.0),
            ("cd".to_string(), -1.0),
            ("abc".to_string(), -3.0),
        ];
        let t = UnigramTokenizer::new(pieces, "<unk>").expect("vocab should build");
        // "abcd": [ab,cd]=-2 ; [abc,d]=-8 ; [a,b,c,d]=-20. Optimal = [ab,cd].
        let (ids, score) = t.encode_with_score("abcd");
        assert_eq!(
            ids,
            vec![
                t.piece_to_id("ab").expect("ab id"),
                t.piece_to_id("cd").expect("cd id"),
            ]
        );
        assert!((score - (-2.0)).abs() < 1e-9, "score={score}");
    }

    #[test]
    fn falls_back_to_unk_for_uncovered_char() {
        // 'z' is not in the vocab → must use <unk> for it.
        let t = toy();
        let (ids, _score) = t.encode_with_score("az");
        assert_eq!(ids, vec![1, 0]); // 'a', <unk>
    }

    #[test]
    fn unk_only_when_nothing_matches() {
        let t = toy();
        let ids = t.encode("zz");
        assert_eq!(ids, vec![0, 0]); // two <unk>
    }

    #[test]
    fn prefers_two_pieces_when_cheaper_than_one_unk() {
        // With "ab" cheap (-1) vs two singles (-4) vs is there a 3-char? not for "ab".
        let t = toy();
        let ids = t.encode("ab");
        assert_eq!(ids, vec![4]); // single "ab" piece
    }

    #[test]
    fn decode_concatenates_surfaces() {
        let t = toy();
        assert_eq!(t.decode(&[4, 3]).expect("decode"), "abc"); // "ab"+"c"
        assert_eq!(t.decode(&[6]).expect("decode"), "abc");
    }

    #[test]
    fn decode_out_of_range_errors() {
        let t = toy();
        assert!(matches!(
            t.decode(&[999]),
            Err(LmError::OutOfVocab { token: 999 })
        ));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let t = toy();
        for text in &["a", "ab", "abc", "abcabc", "bca"] {
            let ids = t.encode(text);
            let decoded = t.decode(&ids).expect("decode should succeed");
            assert_eq!(&decoded, text, "roundtrip failed for {text:?}");
        }
    }

    #[test]
    fn unicode_multibyte_pieces() {
        // Ensure byte-offset slicing handles multi-byte characters correctly.
        let pieces = vec![
            ("<unk>".to_string(), -10.0),
            ("é".to_string(), -1.0),
            ("à".to_string(), -1.0),
            ("éà".to_string(), -0.5),
        ];
        let t = UnigramTokenizer::new(pieces, "<unk>").expect("unicode vocab should build");
        // "éà" → single "éà" piece (-0.5) beats "é"+"à" (-2.0).
        let (ids, score) = t.encode_with_score("éà");
        assert_eq!(ids, vec![3]);
        assert!((score - (-0.5)).abs() < 1e-9);
    }
}
