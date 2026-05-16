//! Gotoh affine-gap global alignment (3 DP matrices: match M, gap-in-x X, gap-in-y Y).

use crate::error::{SeqError, SeqResult};

/// Affine-gap scoring scheme.
#[derive(Debug, Clone, Copy)]
pub struct GotohScoring {
    pub match_score: i32,
    pub mismatch: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
}

impl Default for GotohScoring {
    fn default() -> Self {
        Self {
            match_score: 2,
            mismatch: -1,
            gap_open: -5,
            gap_extend: -1,
        }
    }
}

/// Result of Gotoh affine-gap global alignment.
#[derive(Debug, Clone)]
pub struct GotohAlignment {
    pub a_aligned: Vec<Option<usize>>,
    pub b_aligned: Vec<Option<usize>>,
    pub score: i32,
}

/// Run Gotoh's affine-gap global alignment.
pub fn gotoh_align(a: &[u8], b: &[u8], sc: &GotohScoring) -> SeqResult<GotohAlignment> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(SeqError::EmptyInput);
    }
    let cols = n + 1;
    let neg_inf = i32::MIN / 2;
    let mut mm = vec![neg_inf; (m + 1) * cols];
    let mut x = vec![neg_inf; (m + 1) * cols];
    let mut y = vec![neg_inf; (m + 1) * cols];

    mm[0] = 0;
    for i in 1..=m {
        x[i * cols] = sc.gap_open + (i as i32 - 1) * sc.gap_extend;
    }
    for j in 1..=n {
        y[j] = sc.gap_open + (j as i32 - 1) * sc.gap_extend;
    }

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] {
                sc.match_score
            } else {
                sc.mismatch
            };
            let prev_diag = [
                mm[(i - 1) * cols + (j - 1)],
                x[(i - 1) * cols + (j - 1)],
                y[(i - 1) * cols + (j - 1)],
            ];
            mm[i * cols + j] = prev_diag.iter().cloned().fold(neg_inf, i32::max) + s;
            x[i * cols + j] =
                (mm[(i - 1) * cols + j] + sc.gap_open).max(x[(i - 1) * cols + j] + sc.gap_extend);
            y[i * cols + j] =
                (mm[i * cols + (j - 1)] + sc.gap_open).max(y[i * cols + (j - 1)] + sc.gap_extend);
        }
    }
    let score = [mm[m * cols + n], x[m * cols + n], y[m * cols + n]]
        .iter()
        .cloned()
        .fold(neg_inf, i32::max);

    // Trace back picking the layer with highest score.
    let mut a_align = Vec::new();
    let mut b_align = Vec::new();
    let mut i = m;
    let mut j = n;
    let mut layer = if mm[i * cols + j] >= x[i * cols + j] && mm[i * cols + j] >= y[i * cols + j] {
        0
    } else if x[i * cols + j] >= y[i * cols + j] {
        1
    } else {
        2
    };

    while i > 0 || j > 0 {
        match layer {
            0 => {
                if i > 0 && j > 0 {
                    a_align.push(Some(i - 1));
                    b_align.push(Some(j - 1));
                    // Pick which layer mm came from
                    let prev = [
                        mm[(i - 1) * cols + (j - 1)],
                        x[(i - 1) * cols + (j - 1)],
                        y[(i - 1) * cols + (j - 1)],
                    ];
                    let max_prev = prev.iter().cloned().fold(neg_inf, i32::max);
                    layer = if prev[0] == max_prev {
                        0
                    } else if prev[1] == max_prev {
                        1
                    } else {
                        2
                    };
                    i -= 1;
                    j -= 1;
                } else if i > 0 {
                    layer = 1;
                } else {
                    layer = 2;
                }
            }
            1 => {
                if i > 0 {
                    a_align.push(Some(i - 1));
                    b_align.push(None);
                    let open = mm[(i - 1) * cols + j] + sc.gap_open;
                    let ext = x[(i - 1) * cols + j] + sc.gap_extend;
                    layer = if open >= ext { 0 } else { 1 };
                    i -= 1;
                } else {
                    layer = 2;
                }
            }
            _ => {
                if j > 0 {
                    a_align.push(None);
                    b_align.push(Some(j - 1));
                    let open = mm[i * cols + (j - 1)] + sc.gap_open;
                    let ext = y[i * cols + (j - 1)] + sc.gap_extend;
                    layer = if open >= ext { 0 } else { 2 };
                    j -= 1;
                } else {
                    layer = 1;
                }
            }
        }
    }
    a_align.reverse();
    b_align.reverse();
    Ok(GotohAlignment {
        a_aligned: a_align,
        b_aligned: b_align,
        score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gotoh_identical_is_match_count() {
        let a = b"AAAA";
        let r = gotoh_align(a, a, &GotohScoring::default()).expect("ok");
        assert_eq!(r.score, 2 * a.len() as i32);
    }

    #[test]
    fn gotoh_handles_long_gap() {
        let a = b"AAAA";
        let b = b"AAAAGGGGAAAA";
        let sc = GotohScoring {
            match_score: 2,
            mismatch: -1,
            gap_open: -5,
            gap_extend: -1,
        };
        let r = gotoh_align(a, b, &sc).expect("ok");
        // Should achieve a finite score; not -inf.
        assert!(r.score > i32::MIN / 4);
    }
}
