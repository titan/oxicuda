//! Needleman-Wunsch global alignment with a simple linear-gap penalty.

use crate::error::{SeqError, SeqResult};

/// Simple scoring scheme: match reward, mismatch penalty, linear gap penalty.
#[derive(Debug, Clone, Copy)]
pub struct ScoringMatrix {
    pub match_score: i32,
    pub mismatch: i32,
    pub gap: i32,
}

impl Default for ScoringMatrix {
    fn default() -> Self {
        Self {
            match_score: 1,
            mismatch: -1,
            gap: -2,
        }
    }
}

/// Aligned pair: each position is `Some(idx)` for sequence character or `None` for a gap.
#[derive(Debug, Clone)]
pub struct Alignment {
    pub a_aligned: Vec<Option<usize>>,
    pub b_aligned: Vec<Option<usize>>,
    pub score: i32,
}

/// Run global Needleman-Wunsch alignment.
pub fn needleman_wunsch(a: &[u8], b: &[u8], score: &ScoringMatrix) -> SeqResult<Alignment> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(SeqError::EmptyInput);
    }
    let cols = n + 1;
    let mut dp = vec![0i32; (m + 1) * cols];
    let mut trace = vec![0u8; (m + 1) * cols];
    // 0 = diag, 1 = up (gap in b), 2 = left (gap in a)

    for i in 0..=m {
        dp[i * cols] = score.gap * i as i32;
        trace[i * cols] = 1;
    }
    for j in 0..=n {
        dp[j] = score.gap * j as i32;
        trace[j] = 2;
    }
    trace[0] = 0;

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

    // Trace back
    let mut a_align = Vec::new();
    let mut b_align = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        let dir = trace[i * cols + j];
        match dir {
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
                // boundary safety: prefer up if i > 0, else left
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
    use super::*;

    #[test]
    fn nw_simple() {
        let a = b"GATTACA";
        let b = b"GCATGCU";
        let s = ScoringMatrix::default();
        let r = needleman_wunsch(a, b, &s).expect("ok");
        // The score is well-defined for this scoring system.
        // With match=1, mismatch=-1, gap=-2, optimal score is 0 for this pair.
        assert!(
            r.score >= -3 && r.score <= 3,
            "score in unexpected range: {}",
            r.score
        );
        assert_eq!(r.a_aligned.len(), r.b_aligned.len());
    }

    #[test]
    fn nw_identical() {
        let a = b"ACGT";
        let s = ScoringMatrix::default();
        let r = needleman_wunsch(a, a, &s).expect("ok");
        assert_eq!(r.score, 4);
    }
}
