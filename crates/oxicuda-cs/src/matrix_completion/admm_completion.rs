//! ADMM matrix completion using consensus splitting.
//!
//! Splits the problem as `min ||X||_* + I_{P_Ω(Z) = P_Ω(M)}(Z) s.t. X = Z`.
//! Augmented Lagrangian leads to alternation:
//!   - X-step: SVD soft-threshold of Z - U with threshold `1/rho`.
//!   - Z-step: project (X + U) onto the observed-entries constraint.
//!   - U-step: dual update.

use crate::error::{CsError, CsResult};
use crate::linalg::jacobi_svd::jacobi_svd_thin;
use crate::matrix_completion::CompletionResult;

/// ADMM for nuclear-norm matrix completion.
pub fn admm_matrix_completion(
    m: &[f64],
    mask: &[bool],
    h: usize,
    w: usize,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<CompletionResult> {
    if m.len() != h * w {
        return Err(CsError::ShapeMismatch {
            expected: vec![h, w],
            got: vec![m.len()],
        });
    }
    if mask.len() != h * w {
        return Err(CsError::DimensionMismatch {
            a: mask.len(),
            b: h * w,
        });
    }
    if rho <= 0.0 {
        return Err(CsError::InvalidParameter("rho must be > 0".into()));
    }
    let mut x = vec![0.0_f64; h * w];
    let mut z = vec![0.0_f64; h * w];
    for k in 0..(h * w) {
        if mask[k] {
            z[k] = m[k];
        }
    }
    let mut u = vec![0.0_f64; h * w];
    let r_dim = h.min(w);
    let mut iter = 0usize;
    let mut last_resid = f64::INFINITY;
    for _ in 0..max_iter {
        // X-step: SVD soft-threshold of (z - u).
        let mut zmu = vec![0.0_f64; h * w];
        for k in 0..(h * w) {
            zmu[k] = z[k] - u[k];
        }
        let tau = 1.0 / rho;
        if h >= w {
            let (uu, ss, vv) = jacobi_svd_thin(&zmu, h, w)?;
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
                    x[i * w + j] = acc;
                }
            }
        } else {
            let mut t = vec![0.0_f64; w * h];
            for i in 0..h {
                for j in 0..w {
                    t[j * h + i] = zmu[i * w + j];
                }
            }
            let (uu, ss, vv) = jacobi_svd_thin(&t, w, h)?;
            let mut ss_new = vec![0.0_f64; ss.len()];
            for (i, &si) in ss.iter().enumerate() {
                ss_new[i] = (si - tau).max(0.0);
            }
            // x = (uu ss_new vv^T)^T => x[i, j] = sum_k vv[i*h+k] * ss_new[k] * uu[j*h+k]
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..r_dim {
                        acc += vv[i * r_dim + k] * ss_new[k] * uu[j * r_dim + k];
                    }
                    x[i * w + j] = acc;
                }
            }
        }
        // Z-step: project x + u onto observation set.
        let mut resid_sq = 0.0_f64;
        let mut obs_norm_sq = 0.0_f64;
        for k in 0..(h * w) {
            if mask[k] {
                z[k] = m[k];
                let d = z[k] - x[k];
                resid_sq += d * d;
                obs_norm_sq += m[k] * m[k];
            } else {
                z[k] = x[k] + u[k];
            }
        }
        // U-step.
        for k in 0..(h * w) {
            u[k] += x[k] - z[k];
        }
        iter += 1;
        let cur = resid_sq.sqrt() / obs_norm_sq.sqrt().max(1.0e-300);
        if (last_resid - cur).abs() < tol && cur < tol {
            break;
        }
        last_resid = cur;
    }
    Ok(CompletionResult {
        x,
        residual: last_resid,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admm_completion_runs() {
        let m = vec![1.0, 2.0, 2.0, 4.0];
        let mask = vec![true, true, true, false];
        let r = admm_matrix_completion(&m, &mask, 2, 2, 1.0, 200, 1.0e-7).expect("ok");
        assert!(r.iterations > 0);
    }
}
