//! GoDec — semi-soft GoDec via bilateral random projections (Zhou & Tao 2011).
//!
//! Decompose `M = L + S + N` with `rank(L) ≤ r`, `||S||_0 ≤ k`. Iterates:
//!   - Estimate `L` from `M − S` via top-r truncated SVD.
//!   - Estimate `S` by hard-thresholding `M − L` to the top-`k` entries by magnitude.

use crate::error::{CsError, CsResult};
use crate::linalg::jacobi_svd::jacobi_svd_thin;
use crate::robust_pca::RobustPcaResult;
use crate::thresholding::iht::hard_threshold_k;

/// GoDec decomposition.
pub fn godec(
    m: &[f64],
    h: usize,
    w: usize,
    rank: usize,
    card: usize,
    max_iter: usize,
    tol: f64,
) -> CsResult<RobustPcaResult> {
    if m.len() != h * w {
        return Err(CsError::ShapeMismatch {
            expected: vec![h, w],
            got: vec![m.len()],
        });
    }
    if rank == 0 || rank > h.min(w) {
        return Err(CsError::InvalidRank(rank));
    }
    if card > h * w {
        return Err(CsError::SupportTooLarge {
            requested: card,
            max: h * w,
        });
    }
    let mut l = vec![0.0_f64; h * w];
    let mut s = vec![0.0_f64; h * w];
    let r_dim = h.min(w);
    let mut iter = 0usize;
    let mut last = f64::INFINITY;
    for _ in 0..max_iter {
        // L update.
        let mut x_for_l = vec![0.0_f64; h * w];
        for k in 0..(h * w) {
            x_for_l[k] = m[k] - s[k];
        }
        // Truncated SVD to rank `rank`.
        if h >= w {
            let (uu, ss, vv) = jacobi_svd_thin(&x_for_l, h, w)?;
            let mut ss_trunc = ss.clone();
            for i in rank..ss_trunc.len() {
                ss_trunc[i] = 0.0;
            }
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..rank.min(w) {
                        acc += uu[i * w + k] * ss_trunc[k] * vv[j * w + k];
                    }
                    l[i * w + j] = acc;
                }
            }
        } else {
            let mut t = vec![0.0_f64; w * h];
            for i in 0..h {
                for j in 0..w {
                    t[j * h + i] = x_for_l[i * w + j];
                }
            }
            let (uu, ss, vv) = jacobi_svd_thin(&t, w, h)?;
            let mut ss_trunc = ss.clone();
            for i in rank..ss_trunc.len() {
                ss_trunc[i] = 0.0;
            }
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..rank.min(r_dim) {
                        acc += vv[i * r_dim + k] * ss_trunc[k] * uu[j * r_dim + k];
                    }
                    l[i * w + j] = acc;
                }
            }
        }
        // S update: hard-threshold top-`card` of M - L.
        let mut diff = vec![0.0_f64; h * w];
        for k in 0..(h * w) {
            diff[k] = m[k] - l[k];
        }
        let (s_new, _supp) = hard_threshold_k(&diff, card)?;
        let mut delta = 0.0_f64;
        for k in 0..(h * w) {
            let d = s_new[k] - s[k];
            delta += d * d;
        }
        s = s_new;
        iter += 1;
        let cur = delta.sqrt();
        if (last - cur).abs() < tol && cur < tol {
            break;
        }
        last = cur;
    }
    Ok(RobustPcaResult {
        low_rank: l,
        sparse: s,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn godec_runs() {
        let mut m = vec![0.0_f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                m[i * 4 + j] = ((i + 1) as f64) * ((j + 1) as f64);
            }
        }
        m[5] += 10.0;
        let r = godec(&m, 4, 4, 1, 2, 50, 1.0e-7).expect("ok");
        assert!(r.iterations > 0);
    }
}
