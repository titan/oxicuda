//! # Prompt-lookup decoding (n-gram drafting) + no-repeat-ngram blocking.
//!
//! **Prompt-lookup decoding** (Saxena 2023; also "n-gram speculative decoding")
//! is a model-free draft strategy for speculative decoding: instead of a small
//! draft *model*, candidate continuation tokens are found by matching the most
//! recent `n`-gram suffix of the generated sequence against earlier occurrences
//! in the prompt/context, then copying the tokens that followed. It is highly
//! effective for input-grounded tasks (summarisation, code editing, RAG) where
//! the output reuses spans verbatim from the input.
//!
//! This module provides:
//!
//! * [`PromptLookupDecoder`] — finds a draft continuation of up to
//!   `num_pred_tokens` by searching the context for the latest matching n-gram
//!   suffix (longest match preferred, most-recent occurrence as tie-break).
//! * [`no_repeat_ngram_banned`] — the classic *no-repeat n-gram* constraint
//!   (Paulus 2017): returns the set of token ids that would complete a
//!   previously-seen `n`-gram, so the caller can mask their logits to
//!   `−∞` and prevent verbatim n-gram repetition.
//!
//! Both operate on `&[u32]` token-id slices; no model evaluation is required.

use crate::error::{InferError, InferResult};

// ─── PromptLookupDecoder ─────────────────────────────────────────────────────

/// N-gram prompt-lookup drafter.
#[derive(Debug, Clone, Copy)]
pub struct PromptLookupDecoder {
    /// Size `n` of the suffix n-gram used to find a match (must be ≥ 1).
    pub ngram_size: usize,
    /// Maximum number of draft tokens to copy after a match (must be ≥ 1).
    pub num_pred_tokens: usize,
}

impl PromptLookupDecoder {
    /// Create a new decoder.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `ngram_size == 0` or
    ///   `num_pred_tokens == 0`.
    pub fn new(ngram_size: usize, num_pred_tokens: usize) -> InferResult<Self> {
        if ngram_size == 0 {
            return Err(InferError::InvalidConfig("ngram_size must be >= 1"));
        }
        if num_pred_tokens == 0 {
            return Err(InferError::InvalidConfig("num_pred_tokens must be >= 1"));
        }
        Ok(Self {
            ngram_size,
            num_pred_tokens,
        })
    }

    /// Propose draft tokens by matching the suffix n-gram of `tokens` against
    /// earlier occurrences within the same sequence.
    ///
    /// The search tries the full `ngram_size` first; the *most recent* prior
    /// occurrence of the suffix is used (it tends to predict the immediate
    /// continuation best). Returns up to `num_pred_tokens` tokens that followed
    /// that occurrence, or an empty `Vec` if no match exists.
    ///
    /// # Arguments
    /// * `tokens` — the full token sequence generated so far (prompt + output).
    ///
    /// # Errors
    /// Never errors for valid construction; returns `Ok(vec![])` when no draft
    /// is available.
    pub fn propose(&self, tokens: &[u32]) -> InferResult<Vec<u32>> {
        let n = self.ngram_size;
        let len = tokens.len();
        if len <= n {
            return Ok(Vec::new());
        }

        // Suffix n-gram to match (the last n tokens).
        let suffix = &tokens[len - n..];

        // Search earlier positions for an occurrence of `suffix`, scanning from
        // the most recent candidate start backward. The match must end strictly
        // before the current suffix start so we copy *forward* tokens.
        // Valid match start range: 0 ..= (len - n - 1).
        let suffix_start = len - n; // exclusive upper bound for a non-suffix match
        for start in (0..suffix_start).rev() {
            if &tokens[start..start + n] == suffix {
                // Tokens that followed this occurrence. Cap the copy so it never
                // extends into the current suffix region (which would re-emit
                // the suffix itself), and never exceeds num_pred_tokens.
                let copy_from = start + n;
                let copy_to = (copy_from + self.num_pred_tokens).min(suffix_start);
                if copy_from < copy_to {
                    return Ok(tokens[copy_from..copy_to].to_vec());
                }
                // This occurrence yields no usable continuation; keep searching
                // for an earlier match that does.
            }
        }
        Ok(Vec::new())
    }
}

// ─── No-repeat n-gram blocking ───────────────────────────────────────────────

/// Compute the set of token ids that must be banned to enforce the *no-repeat
/// n-gram* constraint at the current step (Paulus et al. 2017).
///
/// Given the generated `tokens` and an n-gram size, the banned tokens are
/// exactly those `v` such that the `(n−1)`-gram suffix of `tokens` followed by
/// `v` has already appeared as an `n`-gram earlier in the sequence. Masking
/// these prevents the model from repeating any `n`-gram verbatim.
///
/// Returns a sorted, de-duplicated `Vec<usize>` of banned token ids (suitable
/// for setting `logits[id] = −∞`).
///
/// # Arguments
/// * `tokens`     — token sequence generated so far.
/// * `ngram_size` — `n` (must be ≥ 1).
///
/// # Errors
/// * [`InferError::InvalidConfig`] if `ngram_size == 0`.
pub fn no_repeat_ngram_banned(tokens: &[u32], ngram_size: usize) -> InferResult<Vec<usize>> {
    if ngram_size == 0 {
        return Err(InferError::InvalidConfig("ngram_size must be >= 1"));
    }
    let n = ngram_size;
    let len = tokens.len();
    // Need at least n-1 tokens of history to form a prefix, plus one prior
    // n-gram to match against.
    if len + 1 < n {
        return Ok(Vec::new());
    }
    // n == 1 bans every token that has already appeared (no 1-gram may repeat).
    if n == 1 {
        let mut banned: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        banned.sort_unstable();
        banned.dedup();
        return Ok(banned);
    }

    // Current (n−1)-gram prefix is the last n-1 tokens.
    if len < n - 1 {
        return Ok(Vec::new());
    }
    let prefix = &tokens[len - (n - 1)..];

    let mut banned = Vec::new();
    // Scan all complete n-grams in the history: starts 0 ..= len - n.
    if len >= n {
        for start in 0..=(len - n) {
            let gram = &tokens[start..start + n];
            if &gram[..n - 1] == prefix {
                banned.push(gram[n - 1] as usize);
            }
        }
    }
    banned.sort_unstable();
    banned.dedup();
    Ok(banned)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_ok() {
        let d = PromptLookupDecoder::new(2, 3).expect("ok");
        assert_eq!(d.ngram_size, 2);
        assert_eq!(d.num_pred_tokens, 3);
    }

    #[test]
    fn propose_copies_continuation() {
        // "a b c ... a b" → suffix "a b" matched at index 0 → copy "c".
        let d = PromptLookupDecoder::new(2, 2).expect("ok");
        let tokens = vec![1_u32, 2, 3, 4, 5, 1, 2];
        let draft = d.propose(&tokens).expect("ok");
        // earliest/most-recent "1 2" before the suffix is at index 0 → follows 3.
        assert_eq!(draft, vec![3, 4], "should copy tokens after matched n-gram");
    }

    #[test]
    fn propose_prefers_most_recent_match() {
        // Two occurrences of "1 2": at idx 0 (→3) and idx 4 (→9).
        // The suffix at the end is "1 2"; most-recent prior match is idx 4 → 9.
        let d = PromptLookupDecoder::new(2, 1).expect("ok");
        let tokens = vec![1_u32, 2, 3, 0, 1, 2, 9, 0, 1, 2];
        let draft = d.propose(&tokens).expect("ok");
        assert_eq!(draft, vec![9], "most-recent occurrence should be preferred");
    }

    #[test]
    fn propose_respects_num_pred_tokens_cap() {
        let d = PromptLookupDecoder::new(1, 2).expect("ok");
        // suffix "7"; earlier "7" at idx 0 followed by 8,9,10 → cap at 2.
        let tokens = vec![7_u32, 8, 9, 10, 11, 7];
        let draft = d.propose(&tokens).expect("ok");
        assert_eq!(draft.len(), 2, "draft should be capped at num_pred_tokens");
        assert_eq!(draft, vec![8, 9]);
    }

    #[test]
    fn propose_no_match_returns_empty() {
        let d = PromptLookupDecoder::new(2, 3).expect("ok");
        let tokens = vec![1_u32, 2, 3, 4, 5, 6]; // suffix "5 6" never seen before
        let draft = d.propose(&tokens).expect("ok");
        assert!(draft.is_empty(), "no matching n-gram ⇒ empty draft");
    }

    #[test]
    fn propose_short_sequence_empty() {
        let d = PromptLookupDecoder::new(3, 2).expect("ok");
        let tokens = vec![1_u32, 2]; // len <= n
        let draft = d.propose(&tokens).expect("ok");
        assert!(draft.is_empty());
    }

    #[test]
    fn propose_does_not_copy_beyond_end() {
        // Match near the end shouldn't read past the slice.
        let d = PromptLookupDecoder::new(2, 5).expect("ok");
        let tokens = vec![3_u32, 4, 9, 3, 4]; // suffix "3 4"; match idx 0 → copy "9"
        let draft = d.propose(&tokens).expect("ok");
        assert_eq!(draft, vec![9]);
    }

    #[test]
    fn no_repeat_blocks_completing_token() {
        // history: 1 2 3 1 2  with n=3.
        // prefix (last n-1) = "1 2"; earlier "1 2 3" exists ⇒ ban 3.
        let banned = no_repeat_ngram_banned(&[1, 2, 3, 1, 2], 3).expect("ok");
        assert_eq!(
            banned,
            vec![3],
            "should ban the token completing the 3-gram"
        );
    }

    #[test]
    fn no_repeat_no_match_empty() {
        // prefix "4 5" never started a prior 3-gram.
        let banned = no_repeat_ngram_banned(&[1, 2, 3, 4, 5], 3).expect("ok");
        assert!(banned.is_empty());
    }

    #[test]
    fn no_repeat_bigram() {
        // n=2: prefix is last 1 token "1"; earlier bigrams starting "1": "1 2".
        let banned = no_repeat_ngram_banned(&[1, 2, 3, 1], 2).expect("ok");
        assert_eq!(banned, vec![2]);
    }

    #[test]
    fn no_repeat_unigram_bans_all_seen() {
        // n=1 bans every previously seen token.
        let banned = no_repeat_ngram_banned(&[5, 2, 5, 7], 1).expect("ok");
        assert_eq!(banned, vec![2, 5, 7]);
    }

    #[test]
    fn no_repeat_multiple_completions() {
        // "1 2" appears before with two different following tokens (3 and 9).
        let banned = no_repeat_ngram_banned(&[1, 2, 3, 0, 1, 2, 9, 0, 1, 2], 3).expect("ok");
        assert_eq!(banned, vec![3, 9], "both prior completions must be banned");
    }

    #[test]
    fn no_repeat_dedup_sorted() {
        // Repeated identical completion appears once, sorted.
        let banned = no_repeat_ngram_banned(&[1, 2, 5, 0, 1, 2, 5, 0, 1, 2], 3).expect("ok");
        assert_eq!(banned, vec![5]);
    }

    #[test]
    fn no_repeat_short_history_empty() {
        let banned = no_repeat_ngram_banned(&[1], 3).expect("ok");
        assert!(banned.is_empty());
    }

    #[test]
    fn err_decoder_zero_ngram() {
        assert!(matches!(
            PromptLookupDecoder::new(0, 3),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_decoder_zero_pred_tokens() {
        assert!(matches!(
            PromptLookupDecoder::new(2, 0),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_no_repeat_zero_ngram() {
        assert!(matches!(
            no_repeat_ngram_banned(&[1, 2, 3], 0),
            Err(InferError::InvalidConfig(_))
        ));
    }
}
