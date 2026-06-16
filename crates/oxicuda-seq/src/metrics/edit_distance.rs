//! Levenshtein edit distance with alignment backtrace.
//!
//! Unlike the scalar [`crate::metrics::metrics::edit_distance`], which returns
//! only the minimum number of edits, this module reconstructs the full
//! **optimal edit script** by backtracing through the dynamic-programming
//! table. The script is a sequence of [`EditOp`] values that transforms the
//! source sequence into the target with the fewest insertions, deletions, and
//! substitutions, and is the basis for ASR Word/Character Error Rate breakdowns
//! and alignment visualisation.
//!
//! The standard Wagner-Fischer recurrence is used:
//! ```text
//! D[i][0] = i,  D[0][j] = j
//! D[i][j] = min( D[i-1][j]   + 1,                 (deletion)
//!                D[i][j-1]   + 1,                 (insertion)
//!                D[i-1][j-1] + (a_i ≠ b_j) )       (match / substitution)
//! ```
//! Ties are broken deterministically with the priority
//! match/substitute → delete → insert, which yields a stable, left-aligned
//! script.

use crate::error::SeqResult;

/// A single elementary edit in an alignment between a source and target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditOp {
    /// Source and target symbols are equal — kept as-is. Holds `(src, tgt)` indices.
    Match { src: usize, tgt: usize },
    /// Source symbol replaced by a different target symbol. Holds `(src, tgt)`.
    Substitute { src: usize, tgt: usize },
    /// Source symbol deleted (present in source, absent in target). Holds `src`.
    Delete { src: usize },
    /// Target symbol inserted (absent in source, present in target). Holds `tgt`.
    Insert { tgt: usize },
}

/// Counts of each operation kind in an edit script, plus the total distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EditCounts {
    /// Number of matched (unchanged) symbols.
    pub matches: usize,
    /// Number of substitutions.
    pub substitutions: usize,
    /// Number of deletions.
    pub deletions: usize,
    /// Number of insertions.
    pub insertions: usize,
}

impl EditCounts {
    /// Levenshtein distance = substitutions + deletions + insertions.
    #[must_use]
    pub fn distance(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

/// Result of an edit-distance alignment: the optimal script and its counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditAlignment {
    /// The optimal edit script, ordered source-left to source-right.
    pub ops: Vec<EditOp>,
    /// Aggregate operation counts.
    pub counts: EditCounts,
}

/// Compute the Levenshtein distance and an optimal alignment between `a` and `b`.
///
/// Works for any sequence of `Eq` elements (characters, tokens, words …). The
/// returned [`EditAlignment`] contains both the operation list and the counts.
pub fn align<T: Eq>(a: &[T], b: &[T]) -> EditAlignment {
    let m = a.len();
    let n = b.len();
    let cols = n + 1;
    // dp[i][j] in flat layout.
    let mut dp = vec![0usize; (m + 1) * cols];
    for i in 0..=m {
        dp[i * cols] = i;
    }
    for j in 0..=n {
        dp[j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = dp[(i - 1) * cols + j] + 1;
            let ins = dp[i * cols + (j - 1)] + 1;
            let sub = dp[(i - 1) * cols + (j - 1)] + cost;
            dp[i * cols + j] = del.min(ins).min(sub);
        }
    }

    // Backtrace from (m, n) to (0, 0).
    let mut ops_rev: Vec<EditOp> = Vec::new();
    let mut counts = EditCounts::default();
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        let here = dp[i * cols + j];
        // Priority: diagonal (match/sub) → delete → insert.
        if i > 0 && j > 0 {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            if here == dp[(i - 1) * cols + (j - 1)] + cost {
                if cost == 0 {
                    ops_rev.push(EditOp::Match {
                        src: i - 1,
                        tgt: j - 1,
                    });
                    counts.matches += 1;
                } else {
                    ops_rev.push(EditOp::Substitute {
                        src: i - 1,
                        tgt: j - 1,
                    });
                    counts.substitutions += 1;
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && here == dp[(i - 1) * cols + j] + 1 {
            ops_rev.push(EditOp::Delete { src: i - 1 });
            counts.deletions += 1;
            i -= 1;
            continue;
        }
        // Remaining case: insertion (j > 0 must hold here).
        ops_rev.push(EditOp::Insert { tgt: j - 1 });
        counts.insertions += 1;
        j -= 1;
    }

    ops_rev.reverse();
    EditAlignment {
        ops: ops_rev,
        counts,
    }
}

/// Levenshtein distance via the backtracing aligner (kept for symmetry).
///
/// Equivalent in value to [`crate::metrics::metrics::edit_distance`].
pub fn edit_distance_aligned<T: Eq>(a: &[T], b: &[T]) -> usize {
    align(a, b).counts.distance()
}

/// Word Error Rate: `(S + D + I) / max(N_ref, 1)` where `N_ref = reference.len()`.
///
/// `reference` and `hypothesis` are token sequences (e.g. words). Returns `0.0`
/// for two empty inputs and `f64::from(hyp.len())` semantics scaled by the
/// reference length otherwise.
pub fn word_error_rate<T: Eq>(reference: &[T], hypothesis: &[T]) -> SeqResult<f64> {
    let counts = align(reference, hypothesis).counts;
    let n = reference.len();
    if n == 0 {
        // No reference tokens: rate is 0 if hypothesis also empty, else the
        // number of spurious insertions (conventionally normalised by 1).
        return Ok(counts.distance() as f64);
    }
    Ok(counts.distance() as f64 / n as f64)
}

/// Character Error Rate over `char` sequences derived from `&str` inputs.
pub fn character_error_rate(reference: &str, hypothesis: &str) -> SeqResult<f64> {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    word_error_rate(&r, &h)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn kitten_to_sitting_distance_three() {
        let a = chars("kitten");
        let b = chars("sitting");
        let al = align(&a, &b);
        assert_eq!(al.counts.distance(), 3);
    }

    #[test]
    fn kitten_to_sitting_op_breakdown() {
        // kitten → sitting: substitute k→s, substitute e→i, insert g.
        let a = chars("kitten");
        let b = chars("sitting");
        let al = align(&a, &b);
        assert_eq!(al.counts.substitutions, 2);
        assert_eq!(al.counts.insertions, 1);
        assert_eq!(al.counts.deletions, 0);
        assert_eq!(al.counts.matches, 4); // i, t, t, n
    }

    #[test]
    fn identical_sequences_all_matches() {
        let a = chars("hello");
        let al = align(&a, &a);
        assert_eq!(al.counts.distance(), 0);
        assert_eq!(al.counts.matches, 5);
        assert!(al.ops.iter().all(|op| matches!(op, EditOp::Match { .. })));
    }

    #[test]
    fn empty_source_is_all_insertions() {
        let a: Vec<char> = Vec::new();
        let b = chars("abc");
        let al = align(&a, &b);
        assert_eq!(al.counts.insertions, 3);
        assert_eq!(al.counts.distance(), 3);
        assert_eq!(al.ops.len(), 3);
    }

    #[test]
    fn empty_target_is_all_deletions() {
        let a = chars("abc");
        let b: Vec<char> = Vec::new();
        let al = align(&a, &b);
        assert_eq!(al.counts.deletions, 3);
        assert_eq!(al.counts.distance(), 3);
    }

    #[test]
    fn both_empty_is_no_ops() {
        let a: Vec<char> = Vec::new();
        let b: Vec<char> = Vec::new();
        let al = align(&a, &b);
        assert!(al.ops.is_empty());
        assert_eq!(al.counts.distance(), 0);
    }

    #[test]
    fn ops_reconstruct_target() {
        // Applying the script to the source must reproduce the target exactly.
        let a = chars("intention");
        let b = chars("execution");
        let al = align(&a, &b);
        let mut rebuilt: Vec<char> = Vec::new();
        for op in &al.ops {
            match *op {
                EditOp::Match { tgt, .. } | EditOp::Substitute { tgt, .. } => rebuilt.push(b[tgt]),
                EditOp::Insert { tgt } => rebuilt.push(b[tgt]),
                EditOp::Delete { .. } => {}
            }
        }
        assert_eq!(rebuilt, b);
    }

    #[test]
    fn ops_consume_source_in_order() {
        // The source indices touched by Match/Substitute/Delete must be 0..m in order.
        let a = chars("abcdef");
        let b = chars("azced");
        let al = align(&a, &b);
        let mut consumed: Vec<usize> = Vec::new();
        for op in &al.ops {
            match *op {
                EditOp::Match { src, .. }
                | EditOp::Substitute { src, .. }
                | EditOp::Delete { src } => consumed.push(src),
                EditOp::Insert { .. } => {}
            }
        }
        let expected: Vec<usize> = (0..a.len()).collect();
        assert_eq!(consumed, expected);
    }

    #[test]
    fn distance_matches_scalar_reference() {
        // Agreement with the existing scalar edit_distance on several pairs.
        let pairs = [
            ("flaw", "lawn"),
            ("gumbo", "gambol"),
            ("book", "back"),
            ("", "nonempty"),
            ("same", "same"),
        ];
        for (x, y) in pairs {
            let a = chars(x);
            let b = chars(y);
            let via_align = edit_distance_aligned(&a, &b);
            let via_scalar = crate::metrics::metrics::edit_distance(&a, &b);
            assert_eq!(via_align, via_scalar, "{x} vs {y}");
        }
    }

    #[test]
    fn op_count_equals_alignment_length_invariant() {
        // matches + subs is the number of aligned columns where both advance.
        let a = chars("alignment");
        let b = chars("assignment");
        let al = align(&a, &b);
        let c = al.counts;
        // Source length = matches + subs + deletions.
        assert_eq!(c.matches + c.substitutions + c.deletions, a.len());
        // Target length = matches + subs + insertions.
        assert_eq!(c.matches + c.substitutions + c.insertions, b.len());
    }

    #[test]
    fn word_error_rate_basic() {
        // ref: the cat sat ; hyp: the cat sit  → 1 substitution / 3 = 0.333…
        let r = vec!["the", "cat", "sat"];
        let h = vec!["the", "cat", "sit"];
        let wer = word_error_rate(&r, &h).expect("wer");
        assert!((wer - 1.0 / 3.0).abs() < 1e-9, "wer={wer}");
    }

    #[test]
    fn word_error_rate_perfect_is_zero() {
        let r = vec!["a", "b", "c"];
        let wer = word_error_rate(&r, &r).expect("wer");
        assert!(wer.abs() < 1e-12);
    }

    #[test]
    fn word_error_rate_empty_reference() {
        let r: Vec<&str> = Vec::new();
        let h = vec!["x", "y"];
        let wer = word_error_rate(&r, &h).expect("wer");
        assert!((wer - 2.0).abs() < 1e-12);
    }

    #[test]
    fn character_error_rate_string_api() {
        let cer = character_error_rate("kitten", "sitting").expect("cer");
        assert!((cer - 3.0 / 6.0).abs() < 1e-9, "cer={cer}");
    }

    #[test]
    fn works_on_token_ids() {
        let a = vec![1usize, 2, 3, 4];
        let b = vec![1usize, 3, 4];
        let al = align(&a, &b);
        // Deleting the token `2` is the single optimal edit.
        assert_eq!(al.counts.distance(), 1);
        assert_eq!(al.counts.deletions, 1);
    }
}
