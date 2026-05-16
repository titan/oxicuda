//! Principal Component Pursuit (Candès-Li-Ma-Wright 2011).
//!
//! `min ||L||_* + λ ||S||_1 s.t. L + S = M` via ADMM.
//!
//! At each iteration:
//!   - L = D_{1/μ}( M − S + Y/μ ) — singular-value soft-threshold
//!   - S = soft_threshold( M − L + Y/μ, λ/μ )
//!   - Y ← Y + μ (M − L − S)

use crate::error::{CsError, CsResult};
use crate::linalg::jacobi_svd::jacobi_svd_thin;
use crate::robust_pca::RobustPcaResult;
use crate::thresholding::iht::soft_threshold;

/// Run PCP on input `m` of shape `h × w` (row-major). Returns `(L, S)`.
pub fn robust_pca_pcp(
    m: &[f64],
    h: usize,
    w: usize,
    lambda: Option<f64>,
    mu: Option<f64>,
    max_iter: usize,
    tol: f64,
) -> CsResult<RobustPcaResult> {
    if m.len() != h * w {
        return Err(CsError::ShapeMismatch {
            expected: vec![h, w],
            got: vec![m.len()],
        });
    }
    let n_max = h.max(w);
    let lam = lambda.unwrap_or(1.0_f64 / (n_max as f64).sqrt());
    if lam <= 0.0 {
        return Err(CsError::InvalidParameter("lambda must be > 0".into()));
    }
    // Estimate mu by ||M||_2 (largest singular value).
    let mu_val = match mu {
        Some(v) => {
            if v <= 0.0 {
                return Err(CsError::InvalidParameter("mu must be > 0".into()));
            }
            v
        }
        None => {
            // Power-method on m m^T.
            let mut v = vec![1.0_f64 / (w as f64).sqrt(); w];
            let mut lam_est = 1.0_f64;
            for _ in 0..20 {
                // m v
                let mut mv = vec![0.0_f64; h];
                for i in 0..h {
                    let mut s = 0.0_f64;
                    for j in 0..w {
                        s += m[i * w + j] * v[j];
                    }
                    mv[i] = s;
                }
                // m^T mv
                let mut mtmv = vec![0.0_f64; w];
                for i in 0..h {
                    for j in 0..w {
                        mtmv[j] += m[i * w + j] * mv[i];
                    }
                }
                let nrm = mtmv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0e-300);
                lam_est = nrm.sqrt();
                for j in 0..w {
                    v[j] = mtmv[j] / nrm;
                }
            }
            (1.25_f64 / lam_est).max(1.0e-6)
        }
    };
    let mut l = vec![0.0_f64; h * w];
    let mut s = vec![0.0_f64; h * w];
    let mut y = vec![0.0_f64; h * w];
    let r_dim = h.min(w);
    let mut iter = 0usize;
    let mut last = f64::INFINITY;
    for _ in 0..max_iter {
        // L update.
        let mut x_for_l = vec![0.0_f64; h * w];
        for k in 0..(h * w) {
            x_for_l[k] = m[k] - s[k] + y[k] / mu_val;
        }
        // SVD soft-threshold.
        let tau = 1.0 / mu_val;
        if h >= w {
            let (uu, ss, vv) = jacobi_svd_thin(&x_for_l, h, w)?;
            let mut ss_new = vec![0.0_f64; ss.len()];
            for (i, &si) in ss.iter().enumerate() {
                ss_new[i] = (si - tau).max(0.0);
            }
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..w {
                        acc += uu[i * w + k] * ss_new[k] * vv[j * w + k];
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
            let mut ss_new = vec![0.0_f64; ss.len()];
            for (i, &si) in ss.iter().enumerate() {
                ss_new[i] = (si - tau).max(0.0);
            }
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..r_dim {
                        acc += vv[i * r_dim + k] * ss_new[k] * uu[j * r_dim + k];
                    }
                    l[i * w + j] = acc;
                }
            }
        }
        // S update.
        let mut x_for_s = vec![0.0_f64; h * w];
        for k in 0..(h * w) {
            x_for_s[k] = m[k] - l[k] + y[k] / mu_val;
        }
        s = soft_threshold(&x_for_s, lam / mu_val);
        // Y update.
        let mut resid_sq = 0.0_f64;
        for k in 0..(h * w) {
            let r = m[k] - l[k] - s[k];
            y[k] += mu_val * r;
            resid_sq += r * r;
        }
        iter += 1;
        let cur = resid_sq.sqrt();
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
    fn pcp_runs() {
        // Rank-1 + sparse: M = u v^T + small outlier.
        let mut m = vec![0.0_f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                m[i * 4 + j] = ((i + 1) as f64) * ((j + 1) as f64);
            }
        }
        m[0] += 10.0;
        let r = robust_pca_pcp(&m, 4, 4, Some(0.3), Some(0.5), 100, 1.0e-6).expect("ok");
        // The sparse component should have a large value at (0,0).
        assert!(r.sparse[0].abs() > 1.0);
    }
}
