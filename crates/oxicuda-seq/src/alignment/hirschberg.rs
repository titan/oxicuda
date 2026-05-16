//! Hirschberg's O(min(m, n)) memory global alignment.
//!
//! Implements the classical divide-and-conquer reduction of NW.

use super::needleman_wunsch::ScoringMatrix;
use crate::error::{SeqError, SeqResult};

/// Result of Hirschberg alignment.
#[derive(Debug, Clone)]
pub struct HirschbergAlignment {
    pub a_aligned: Vec<Option<usize>>,
    pub b_aligned: Vec<Option<usize>>,
    pub score: i32,
}

/// Compute the NW score row using O(n) memory; offsets keep the original index space.
fn nw_score(a: &[u8], b: &[u8], sc: &ScoringMatrix) -> Vec<i32> {
    let n = b.len();
    let mut prev = vec![0i32; n + 1];
    let mut cur = vec![0i32; n + 1];
    for j in 0..=n {
        prev[j] = sc.gap * j as i32;
    }
    for i in 1..=a.len() {
        cur[0] = sc.gap * i as i32;
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] {
                sc.match_score
            } else {
                sc.mismatch
            };
            cur[j] = (prev[j - 1] + s)
                .max(prev[j] + sc.gap)
                .max(cur[j - 1] + sc.gap);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev
}

/// Run Hirschberg global alignment.  Returns the same alignment shape as NW.
pub fn hirschberg_align(a: &[u8], b: &[u8], sc: &ScoringMatrix) -> SeqResult<HirschbergAlignment> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(SeqError::EmptyInput);
    }
    // Use absolute indices via offsets to recover original positions.
    let mut a_align: Vec<Option<usize>> = Vec::new();
    let mut b_align: Vec<Option<usize>> = Vec::new();
    hirschberg_rec(a, b, 0, 0, sc, &mut a_align, &mut b_align);

    // Compute the score by walking the alignment with the scoring matrix.
    let mut score = 0i32;
    for (ai, bi) in a_align.iter().zip(b_align.iter()) {
        match (ai, bi) {
            (Some(i), Some(j)) => {
                score += if a[*i] == b[*j] {
                    sc.match_score
                } else {
                    sc.mismatch
                };
            }
            _ => score += sc.gap,
        }
    }
    Ok(HirschbergAlignment {
        a_aligned: a_align,
        b_aligned: b_align,
        score,
    })
}

/// Append a vertical gap-in-b run (only `a` characters with `b = None`).
fn append_a_only(
    a_off: usize,
    len: usize,
    a_align: &mut Vec<Option<usize>>,
    b_align: &mut Vec<Option<usize>>,
) {
    for k in 0..len {
        a_align.push(Some(a_off + k));
        b_align.push(None);
    }
}

/// Append a horizontal gap-in-a run.
fn append_b_only(
    b_off: usize,
    len: usize,
    a_align: &mut Vec<Option<usize>>,
    b_align: &mut Vec<Option<usize>>,
) {
    for k in 0..len {
        a_align.push(None);
        b_align.push(Some(b_off + k));
    }
}

/// Direct NW alignment for small sub-problems (m == 0 or 1).
fn nw_base(
    a: &[u8],
    b: &[u8],
    a_off: usize,
    b_off: usize,
    sc: &ScoringMatrix,
    a_align: &mut Vec<Option<usize>>,
    b_align: &mut Vec<Option<usize>>,
) {
    if a.is_empty() {
        append_b_only(b_off, b.len(), a_align, b_align);
        return;
    }
    if a.len() == 1 {
        // Try matching `a[0]` at every position of b; pick the highest-scoring placement.
        let n = b.len();
        let mut best_j: Option<usize> = None;
        let mut best_score = i32::MIN;
        for j in 0..n {
            let mut s = 0i32;
            s += sc.gap * j as i32; // initial gaps in a
            s += if a[0] == b[j] {
                sc.match_score
            } else {
                sc.mismatch
            };
            s += sc.gap * (n - j - 1) as i32; // trailing gaps in a
            if s > best_score {
                best_score = s;
                best_j = Some(j);
            }
        }
        // Also consider the all-gaps option (a[0] as a gap, b entirely gaps in a)
        let all_gaps_score = sc.gap + sc.gap * n as i32;
        if all_gaps_score > best_score {
            append_a_only(a_off, 1, a_align, b_align);
            append_b_only(b_off, n, a_align, b_align);
            return;
        }
        if let Some(j) = best_j {
            append_b_only(b_off, j, a_align, b_align);
            a_align.push(Some(a_off));
            b_align.push(Some(b_off + j));
            append_b_only(b_off + j + 1, n - j - 1, a_align, b_align);
        } else {
            append_a_only(a_off, 1, a_align, b_align);
            append_b_only(b_off, n, a_align, b_align);
        }
        return;
    }
    // Should not happen — recursion handles it.
    append_a_only(a_off, a.len(), a_align, b_align);
    append_b_only(b_off, b.len(), a_align, b_align);
}

fn hirschberg_rec(
    a: &[u8],
    b: &[u8],
    a_off: usize,
    b_off: usize,
    sc: &ScoringMatrix,
    a_align: &mut Vec<Option<usize>>,
    b_align: &mut Vec<Option<usize>>,
) {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        append_b_only(b_off, n, a_align, b_align);
        return;
    }
    if n == 0 {
        append_a_only(a_off, m, a_align, b_align);
        return;
    }
    if m == 1 {
        nw_base(a, b, a_off, b_off, sc, a_align, b_align);
        return;
    }
    let i_mid = m / 2;
    let score_l = nw_score(&a[..i_mid], b, sc);
    let a_rev: Vec<u8> = a[i_mid..].iter().rev().cloned().collect();
    let b_rev: Vec<u8> = b.iter().rev().cloned().collect();
    let score_r = nw_score(&a_rev, &b_rev, sc);
    // Find j* maximising score_l[j] + score_r[n - j]
    let mut best_j = 0usize;
    let mut best_v = i32::MIN;
    for j in 0..=n {
        let v = score_l[j] + score_r[n - j];
        if v > best_v {
            best_v = v;
            best_j = j;
        }
    }
    hirschberg_rec(
        &a[..i_mid],
        &b[..best_j],
        a_off,
        b_off,
        sc,
        a_align,
        b_align,
    );
    hirschberg_rec(
        &a[i_mid..],
        &b[best_j..],
        a_off + i_mid,
        b_off + best_j,
        sc,
        a_align,
        b_align,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::needleman_wunsch::needleman_wunsch;

    #[test]
    fn hirschberg_matches_nw_score() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGT", b"ACGGACGT"),
            (b"A", b"AC"),
            (b"AC", b"A"),
        ];
        let sc = ScoringMatrix::default();
        for (a, b) in cases {
            let r1 = needleman_wunsch(a, b, &sc).expect("ok");
            let r2 = hirschberg_align(a, b, &sc).expect("ok");
            assert_eq!(r1.score, r2.score, "score mismatch on {:?}/{:?}", a, b);
        }
    }
}
