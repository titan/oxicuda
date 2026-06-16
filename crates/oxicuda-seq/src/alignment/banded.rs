//! Banded global (Needleman–Wunsch) alignment.
//!
//! Restricts the Needleman–Wunsch dynamic program to a diagonal band of
//! half-width `band` — only cells with `|i − j| ≤ band` are evaluated. For two
//! sequences that are near-identical (small edit distance) this finds the
//! optimal *banded* global alignment by touching only the `O(n · band)` cells
//! inside the band; every cell outside the band is held at `−∞` and can never
//! lie on an optimal path. The score is therefore `≤` the unrestricted
//! Needleman–Wunsch score, and equals it whenever the band is wide enough to
//! contain an optimal path.

use super::needleman_wunsch::{Alignment, ScoringMatrix};
use crate::error::{SeqError, SeqResult};

/// Sentinel for unreachable / off-band DP cells.
const NEG_INF: i32 = i32::MIN / 2;

/// Run banded global alignment of `a` against `b`.
///
/// The scoring scheme is the linear-gap [`ScoringMatrix`] shared with
/// [`needleman_wunsch`](super::needleman_wunsch::needleman_wunsch). Only the
/// diagonal band `|i − j| ≤ band` is computed; the returned [`Alignment`] holds
/// the optimal score and traceback subject to that restriction.
///
/// # Errors
/// * [`SeqError::EmptyInput`] if either sequence is empty.
/// * [`SeqError::InvalidConfiguration`] if `band < |a.len() − b.len()|`: the end
///   cell `(m, n)` then lies outside the band, so no global alignment fits and
///   the request is ill-posed.
pub fn banded_align<T: PartialEq>(
    a: &[T],
    b: &[T],
    score: &ScoringMatrix,
    band: usize,
) -> SeqResult<Alignment> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(SeqError::EmptyInput);
    }
    let diff = m.abs_diff(n);
    if band < diff {
        return Err(SeqError::InvalidConfiguration(format!(
            "band {band} is narrower than the length difference {diff}; no global alignment fits"
        )));
    }

    let cols = n + 1;
    let mut dp = vec![NEG_INF; (m + 1) * cols];
    let mut trace = vec![0u8; (m + 1) * cols];
    // trace codes: 0 = diag, 1 = up (gap in b), 2 = left (gap in a).

    dp[0] = 0;
    trace[0] = 0;
    // First column inside the band (gaps in b).
    let col_limit = m.min(band);
    for i in 1..=col_limit {
        dp[i * cols] = score.gap.saturating_mul(i as i32);
        trace[i * cols] = 1;
    }
    // First row inside the band (gaps in a).
    let row_limit = n.min(band);
    for j in 1..=row_limit {
        dp[j] = score.gap.saturating_mul(j as i32);
        trace[j] = 2;
    }

    for i in 1..=m {
        let lo = i.saturating_sub(band).max(1);
        let hi = (i + band).min(n);
        for j in lo..=hi {
            let s = if a[i - 1] == b[j - 1] {
                score.match_score
            } else {
                score.mismatch
            };
            let diag = dp[(i - 1) * cols + (j - 1)].saturating_add(s);
            let up = dp[(i - 1) * cols + j].saturating_add(score.gap);
            let left = dp[i * cols + (j - 1)].saturating_add(score.gap);
            let (best, dir) = if diag >= up && diag >= left {
                (diag, 0u8)
            } else if up >= left {
                (up, 1u8)
            } else {
                (left, 2u8)
            };
            dp[i * cols + j] = best;
            trace[i * cols + j] = dir;
        }
    }

    let final_score = dp[m * cols + n];

    // Trace back from (m, n) following recorded directions.
    let mut a_align = Vec::new();
    let mut b_align = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        match trace[i * cols + j] {
            0 if i > 0 && j > 0 => {
                a_align.push(Some(i - 1));
                b_align.push(Some(j - 1));
                i -= 1;
                j -= 1;
            }
            1 if i > 0 => {
                a_align.push(Some(i - 1));
                b_align.push(None);
                i -= 1;
            }
            2 if j > 0 => {
                a_align.push(None);
                b_align.push(Some(j - 1));
                j -= 1;
            }
            _ => {
                // Boundary safety: prefer up if rows remain, else left.
                if i > 0 {
                    a_align.push(Some(i - 1));
                    b_align.push(None);
                    i -= 1;
                } else if j > 0 {
                    a_align.push(None);
                    b_align.push(Some(j - 1));
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }
    a_align.reverse();
    b_align.reverse();
    Ok(Alignment {
        a_aligned: a_align,
        b_aligned: b_align,
        score: final_score,
    })
}

#[cfg(test)]
mod tests {
    use super::super::needleman_wunsch::needleman_wunsch;
    use super::*;

    #[test]
    fn banded_identical_matches_full_nw() {
        let a = b"ACGTACGT";
        let s = ScoringMatrix::default();
        let full = needleman_wunsch(a, a, &s).expect("nw");
        // diff = 0, so any band ≥ 0 admits the all-match diagonal.
        let banded = banded_align(a, a, &s, 2).expect("banded");
        assert_eq!(banded.score, full.score);
        assert_eq!(banded.score, a.len() as i32 * s.match_score);
        assert_eq!(banded.a_aligned.len(), banded.b_aligned.len());
    }

    #[test]
    fn banded_score_le_full_nw() {
        let a = b"ACGTACGTAC";
        let b = b"ACGTCGTACG";
        let s = ScoringMatrix::default();
        let full = needleman_wunsch(a, b, &s).expect("nw");
        let banded = banded_align(a, b, &s, 1).expect("banded");
        assert!(
            banded.score <= full.score,
            "banded {} should be ≤ full NW {}",
            banded.score,
            full.score
        );
    }

    #[test]
    fn wide_band_reproduces_full_nw() {
        let a = b"GATTACA";
        let b = b"GCATGCT";
        let s = ScoringMatrix::default();
        let full = needleman_wunsch(a, b, &s).expect("nw");
        // band ≥ max(m, n) covers the whole grid → identical to full NW.
        let banded = banded_align(a, b, &s, a.len().max(b.len())).expect("banded");
        assert_eq!(banded.score, full.score);
    }

    #[test]
    fn one_substitution_recovered_with_narrow_band() {
        let a = b"ACGTACGT";
        let b = b"ACGTTCGT"; // single substitution at index 4 (A → T)
        let s = ScoringMatrix::default();
        let banded = banded_align(a, b, &s, 1).expect("banded");
        let expected = 7 * s.match_score + s.mismatch;
        assert_eq!(banded.score, expected);
        let full = needleman_wunsch(a, b, &s).expect("nw");
        assert_eq!(banded.score, full.score);
    }

    #[test]
    fn band_too_small_for_length_diff_errors() {
        let a = b"ACGTA"; // len 5
        let b = b"ACGTACGTA"; // len 9, diff = 4
        let s = ScoringMatrix::default();
        assert!(banded_align(a, b, &s, 2).is_err());
        // A band that just covers the difference succeeds.
        assert!(banded_align(a, b, &s, 4).is_ok());
    }

    #[test]
    fn empty_input_errors() {
        let s = ScoringMatrix::default();
        let empty: &[u8] = b"";
        assert!(banded_align(empty, b"ACGT", &s, 3).is_err());
        assert!(banded_align(b"ACGT", empty, &s, 3).is_err());
    }
}
