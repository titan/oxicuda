//! Translation Edit Rate (TER).
//!
//! Reference: Snover, M., Dorr, B., Schwartz, R., Micciulla, L. & Makhoul, J.
//! (2006). *A Study of Translation Edit Rate with Targeted Human Annotation*.
//! AMTA 2006. <https://aclanthology.org/2006.amta-papers.25/>.
//!
//! # Metric
//!
//! TER is the minimum number of word-level edits needed to turn a hypothesis into
//! a reference, normalised by the reference length:
//!
//! ```text
//! TER = (num_edits + num_shifts) / ref_length
//! ```
//!
//! The edits are the usual insert / delete / substitute of Levenshtein distance,
//! **plus block shifts**: a contiguous run of hypothesis words may be moved to a
//! different position as a single edit. Finding the optimal set of shifts is
//! NP-hard, so TER uses a **greedy shift heuristic** (Snover et al. §2.2):
//!
//! 1. Compute the current edit distance between the (possibly shifted) hypothesis
//!    and the reference.
//! 2. Search every contiguous block of hypothesis words that also occurs in the
//!    reference and whose two endpoints are currently *mis-aligned* (so moving the
//!    block could help). For each candidate destination, tentatively apply the
//!    shift and recompute the edit distance.
//! 3. Apply the single shift that **reduces** the edit distance the most (each
//!    shift counts as one edit). Repeat until no shift helps.
//! 4. Add the remaining insert/delete/substitute edit distance.
//!
//! This module reuses [`crate::metrics::edit_distance::align`] for the word-level
//! edit distance and alignment backtrace. Production code never panics: all
//! fallible paths return [`SeqError`] (notably an empty reference is rejected).

use crate::error::{SeqError, SeqResult};
use crate::metrics::edit_distance::{EditOp, align};

/// Result of a Translation Edit Rate computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerResult {
    /// TER score = `(num_edits + num_shifts) / ref_len`.
    pub score: f64,
    /// Number of insert/delete/substitute edits after all shifts were applied.
    pub num_edits: usize,
    /// Number of block shifts the greedy heuristic applied.
    pub num_shifts: usize,
    /// Reference length used for normalisation.
    pub ref_len: usize,
}

/// Maximum block length considered by the greedy shift search.
///
/// Snover et al. cap shifts at 10 words; longer matching blocks are rare and the
/// cap bounds the search to keep the heuristic tractable.
const MAX_SHIFT_LEN: usize = 10;

/// Number of insert/delete/substitute edits between two slices.
fn edit_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    align(a, b).counts.distance()
}

/// Mark, for each hypothesis position, whether it is currently aligned to an
/// equal reference word (a "match" in the optimal alignment).
///
/// Positions that are *not* matched are the ones a shift might usefully relocate.
fn aligned_mask<T: Eq>(hyp: &[T], ref_: &[T]) -> Vec<bool> {
    let mut mask = vec![false; hyp.len()];
    for op in align(hyp, ref_).ops {
        if let EditOp::Match { src, .. } = op {
            if src < mask.len() {
                mask[src] = true;
            }
        }
    }
    mask
}

/// Apply a block shift: move `len` words starting at `from` so the block begins at
/// `to` (index expressed in the *post-removal* sequence). Returns the new vector.
fn apply_shift<T: Clone>(seq: &[T], from: usize, len: usize, to: usize) -> Vec<T> {
    let mut block: Vec<T> = seq[from..from + len].to_vec();
    let mut rest: Vec<T> = Vec::with_capacity(seq.len() - len);
    rest.extend_from_slice(&seq[..from]);
    rest.extend_from_slice(&seq[from + len..]);
    let mut out = Vec::with_capacity(seq.len());
    out.extend_from_slice(&rest[..to]);
    out.append(&mut block);
    out.extend_from_slice(&rest[to..]);
    out
}

/// Find the single best shift that reduces the edit distance the most.
///
/// Returns `Some((from, len, to, new_distance))` for the most-reducing shift, or
/// `None` if no shift strictly lowers the current distance.
fn best_shift<T: Eq + Clone>(
    hyp: &[T],
    ref_: &[T],
    current: usize,
) -> Option<(usize, usize, usize, usize)> {
    let h = hyp.len();
    if h == 0 {
        return None;
    }
    let mask = aligned_mask(hyp, ref_);

    let mut best: Option<(usize, usize, usize, usize)> = None;
    // For every candidate block [from, from+len) in the hypothesis …
    let max_len = MAX_SHIFT_LEN.min(h);
    for len in 1..=max_len {
        for from in 0..=h - len {
            // Only consider blocks that are not already fully aligned in place
            // (shifting an already-matched block cannot help and wastes an edit).
            let block_aligned = (from..from + len).all(|p| mask[p]);
            if block_aligned {
                continue;
            }
            // The block must occur in the reference for a shift to plausibly help.
            if !occurs_in(ref_, &hyp[from..from + len]) {
                continue;
            }
            // Try every destination in the post-removal sequence.
            let rest_len = h - len;
            for to in 0..=rest_len {
                // Skip the no-op shift (putting the block back where it was).
                if to == from {
                    continue;
                }
                let shifted = apply_shift(hyp, from, len, to);
                let dist = edit_distance(&shifted, ref_);
                // One shift costs one edit; only keep it if it nets a reduction.
                if dist + 1 < current {
                    let better = match best {
                        None => true,
                        Some((_, _, _, bd)) => dist < bd,
                    };
                    if better {
                        best = Some((from, len, to, dist));
                    }
                }
            }
        }
    }
    best
}

/// Whether `needle` appears as a contiguous sub-slice of `haystack`.
fn occurs_in<T: Eq>(haystack: &[T], needle: &[T]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    let last = haystack.len() - needle.len();
    for start in 0..=last {
        if haystack[start..start + needle.len()] == *needle {
            return true;
        }
    }
    false
}

/// Core TER computation over generic comparable tokens.
///
/// Greedily applies block shifts (each counting as one edit) while they reduce the
/// edit distance, then adds the residual insert/delete/substitute distance and
/// normalises by `ref_.len()`. Returns [`SeqError::EmptyInput`] for an empty
/// reference (the metric is undefined when `ref_len == 0`).
fn ter_tokens<T: Eq + Clone>(hyp: &[T], ref_: &[T]) -> SeqResult<TerResult> {
    let ref_len = ref_.len();
    if ref_len == 0 {
        return Err(SeqError::EmptyInput);
    }

    let mut current_hyp: Vec<T> = hyp.to_vec();
    let mut num_shifts = 0usize;
    let mut current_dist = edit_distance(&current_hyp, ref_);

    // Greedy shift loop: keep applying the most-reducing shift until none helps.
    // The loop must terminate because each accepted shift strictly lowers the
    // total (distance + shifts) lower bound: distance drops by at least 2 while
    // shifts rise by 1, so the bound `current_dist + num_shifts` strictly
    // decreases and is bounded below by 0.
    loop {
        if current_dist == 0 {
            break;
        }
        match best_shift(&current_hyp, ref_, current_dist) {
            Some((from, len, to, new_dist)) => {
                current_hyp = apply_shift(&current_hyp, from, len, to);
                current_dist = new_dist;
                num_shifts += 1;
            }
            None => break,
        }
    }

    let num_edits = current_dist;
    let score = (num_edits + num_shifts) as f64 / ref_len as f64;
    Ok(TerResult {
        score,
        num_edits,
        num_shifts,
        ref_len,
    })
}

/// Translation Edit Rate between a hypothesis and a reference (word strings).
///
/// `TER = (num_edits + num_shifts) / ref_length`. Returns
/// [`SeqError::EmptyInput`] if the reference is empty.
pub fn ter(hyp: &[&str], ref_: &[&str]) -> SeqResult<TerResult> {
    ter_tokens(hyp, ref_)
}

/// Token-id variant of [`ter`] for already-tokenised integer sequences.
pub fn ter_ids(hyp: &[usize], ref_: &[usize]) -> SeqResult<TerResult> {
    ter_tokens(hyp, ref_)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<&str> {
        s.split_whitespace().collect()
    }

    #[test]
    fn identical_sentences_score_zero() {
        let h = words("the cat sat on the mat");
        let r = words("the cat sat on the mat");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_edits, 0);
        assert_eq!(res.num_shifts, 0);
        assert!(res.score.abs() < 1e-12, "score={}", res.score);
        assert_eq!(res.ref_len, 6);
    }

    #[test]
    fn pure_substitutions_no_shifts() {
        // Two words substituted, none movable → 2 edits, 0 shifts, 2/5.
        let r = words("a b c d e");
        let h = words("a x c y e");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_shifts, 0);
        assert_eq!(res.num_edits, 2);
        assert!((res.score - 2.0 / 5.0).abs() < 1e-12, "score={}", res.score);
    }

    #[test]
    fn block_reordering_finds_shift_and_lowers_score() {
        // Reference: A B C D E F ; hypothesis has the block "E F" moved up front.
        // A pure edit distance must delete "E F" at the front and insert them at
        // the end (or substitute), whereas one block shift fixes it cheaply.
        let r = words("A B C D E F");
        let h = words("E F A B C D");
        let no_shift = align(&h, &r).counts.distance() as f64 / r.len() as f64;
        let res = ter(&h, &r).expect("ter");
        assert!(
            res.num_shifts >= 1,
            "expected a shift, got {}",
            res.num_shifts
        );
        assert!(
            res.score < no_shift - 1e-12,
            "shifted score {} should beat no-shift {}",
            res.score,
            no_shift
        );
        // After the optimal shift the sentences are identical: 1 shift, 0 edits.
        assert_eq!(res.num_edits, 0);
        assert_eq!(res.num_shifts, 1);
        assert!((res.score - 1.0 / 6.0).abs() < 1e-12, "score={}", res.score);
    }

    #[test]
    fn single_swap_is_one_shift() {
        // Swapping a single adjacent pair: "b a" → "a b" is one shift.
        let r = words("a b c");
        let h = words("b a c");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_edits, 0);
        assert_eq!(res.num_shifts, 1);
        assert!((res.score - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn insertions_counted() {
        // Hypothesis has one extra word → one deletion edit, no shifts.
        let r = words("the quick fox");
        let h = words("the quick brown fox");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_shifts, 0);
        assert_eq!(res.num_edits, 1);
        assert!((res.score - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn deletions_counted() {
        // Hypothesis is missing one word → one insertion edit, no shifts.
        let r = words("the quick brown fox");
        let h = words("the quick fox");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_shifts, 0);
        assert_eq!(res.num_edits, 1);
        assert!((res.score - 1.0 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn normalisation_by_reference_length() {
        // Identical-content but different reference lengths scale the score.
        let r_short = words("a b");
        let h_short = words("a x");
        let res_short = ter(&h_short, &r_short).expect("ter");
        assert!((res_short.score - 1.0 / 2.0).abs() < 1e-12);

        let r_long = words("a b c d");
        let h_long = words("a x c d");
        let res_long = ter(&h_long, &r_long).expect("ter");
        assert!((res_long.score - 1.0 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn empty_reference_is_error() {
        let h = words("a b c");
        let r: Vec<&str> = Vec::new();
        assert!(ter(&h, &r).is_err());
    }

    #[test]
    fn empty_hypothesis_against_reference() {
        // Empty hypothesis: all reference words are insertions, no shifts.
        let h: Vec<&str> = Vec::new();
        let r = words("a b c");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_shifts, 0);
        assert_eq!(res.num_edits, 3);
        assert!((res.score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn token_id_variant_matches_string_variant() {
        // Same structural reordering expressed as token ids.
        let h_ids = vec![4usize, 5, 0, 1, 2, 3];
        let r_ids = vec![0usize, 1, 2, 3, 4, 5];
        let res = ter_ids(&h_ids, &r_ids).expect("ter");
        assert_eq!(res.num_edits, 0);
        assert_eq!(res.num_shifts, 1);
        assert!((res.score - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn shift_never_increases_total_cost() {
        // The greedy heuristic must never produce a TER above the plain
        // edit-distance / ref-len (shifts are only taken when they reduce cost).
        let cases = [
            ("the cat sat", "the sat cat"),
            ("one two three four", "four three two one"),
            ("a b c d e", "b c d e a"),
            ("hello world foo bar", "foo bar hello world"),
        ];
        for (hs, rs) in cases {
            let h = words(hs);
            let r = words(rs);
            let baseline = align(&h, &r).counts.distance() as f64 / r.len() as f64;
            let res = ter(&h, &r).expect("ter");
            assert!(
                res.score <= baseline + 1e-12,
                "case ({hs} | {rs}): ter {} > baseline {}",
                res.score,
                baseline
            );
        }
    }

    #[test]
    fn far_block_move_is_single_shift() {
        // Move a 2-word block across a long sentence; still one shift, zero edits.
        let r = words("w x a b c d y z");
        let h = words("a b w x c d y z");
        let res = ter(&h, &r).expect("ter");
        assert_eq!(res.num_edits, 0);
        assert_eq!(res.num_shifts, 1);
        assert!((res.score - 1.0 / 8.0).abs() < 1e-12, "score={}", res.score);
    }
}
