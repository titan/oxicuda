//! Smith-Waterman local alignment.

use super::needleman_wunsch::ScoringMatrix;
use crate::error::{SeqError, SeqResult};

/// Local alignment result: aligned slices + score + start indices.
#[derive(Debug, Clone)]
pub struct LocalAlignment {
    pub a_aligned: Vec<Option<usize>>,
    pub b_aligned: Vec<Option<usize>>,
    pub score: i32,
    pub start_a: usize,
    pub start_b: usize,
    pub end_a: usize,
    pub end_b: usize,
}

/// Run Smith-Waterman local alignment.
pub fn smith_waterman(a: &[u8], b: &[u8], score: &ScoringMatrix) -> SeqResult<LocalAlignment> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(SeqError::EmptyInput);
    }
    let cols = n + 1;
    let mut dp = vec![0i32; (m + 1) * cols];
    let mut trace = vec![0u8; (m + 1) * cols];
    let mut best = 0i32;
    let mut best_i = 0usize;
    let mut best_j = 0usize;

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] {
                score.match_score
            } else {
                score.mismatch
            };
            let diag = dp[(i - 1) * cols + (j - 1)] + s;
            let up = dp[(i - 1) * cols + j] + score.gap;
            let left = dp[i * cols + (j - 1)] + score.gap;
            let mut cand = diag.max(up).max(left).max(0);
            let dir = if cand == 0 {
                3u8
            } else if cand == diag {
                0u8
            } else if cand == up {
                1u8
            } else {
                2u8
            };
            if cand < 0 {
                cand = 0;
            }
            dp[i * cols + j] = cand;
            trace[i * cols + j] = dir;
            if cand > best {
                best = cand;
                best_i = i;
                best_j = j;
            }
        }
    }

    // Trace back from (best_i, best_j) until score 0.
    let mut a_align = Vec::new();
    let mut b_align = Vec::new();
    let mut i = best_i;
    let mut j = best_j;
    while i > 0 && j > 0 && dp[i * cols + j] > 0 {
        match trace[i * cols + j] {
            0 => {
                a_align.push(Some(i - 1));
                b_align.push(Some(j - 1));
                i -= 1;
                j -= 1;
            }
            1 => {
                a_align.push(Some(i - 1));
                b_align.push(None);
                i -= 1;
            }
            2 => {
                a_align.push(None);
                b_align.push(Some(j - 1));
                j -= 1;
            }
            _ => break,
        }
    }
    let start_a = i;
    let start_b = j;
    a_align.reverse();
    b_align.reverse();
    Ok(LocalAlignment {
        a_aligned: a_align,
        b_aligned: b_align,
        score: best,
        start_a,
        start_b,
        end_a: best_i,
        end_b: best_j,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sw_finds_embedded_substring() {
        let a = b"XXXACGTYYY";
        let b = b"AACGTGG";
        let s = ScoringMatrix::default();
        let r = smith_waterman(a, b, &s).expect("ok");
        // The longest common substring "ACGT" has score 4 with match=1.
        assert!(r.score >= 4, "got score {}", r.score);
    }

    #[test]
    fn sw_no_match_zero_score() {
        let a = b"AAAA";
        let b = b"CCCC";
        // Mismatch penalty -1, gap -2; best local score should be 0 (empty alignment).
        let s = ScoringMatrix::default();
        let r = smith_waterman(a, b, &s).expect("ok");
        assert_eq!(r.score, 0);
    }
}
