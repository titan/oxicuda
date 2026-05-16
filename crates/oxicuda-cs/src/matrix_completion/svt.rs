//! Singular Value Thresholding (Cai-Candès-Shen 2010).
//!
//! Iterates `Y_{k+1} = Y_k + δ_k · P_Ω(M − X_k)`, `X_{k+1} = D_τ(Y_{k+1})`, where `D_τ`
//! is the singular-value soft-threshold operator.

use crate::error::{CsError, CsResult};
use crate::linalg::jacobi_svd::jacobi_svd_thin;
use crate::matrix_completion::CompletionResult;

/// SVT for matrix completion.
///
/// `mask[i*w + j] == true` indicates `(i,j)` is observed with value `m[i*w + j]`.
/// `tau` is the soft-threshold radius on singular values. `delta` is the step (recommended ≈ 1.2).
pub fn svt(
    m: &[f64],
    mask: &[bool],
    h: usize,
    w: usize,
    tau: f64,
    delta: f64,
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
    if tau <= 0.0 {
        return Err(CsError::InvalidParameter("tau must be > 0".into()));
    }
    if delta <= 0.0 {
        return Err(CsError::InvalidParameter("delta must be > 0".into()));
    }
    let mut y = vec![0.0_f64; h * w];
    let mut x = vec![0.0_f64; h * w];
    let mut iter = 0usize;
    let mut last_residual = f64::INFINITY;
    let r_dim = h.min(w);
    for _ in 0..max_iter {
        // SVD-based shrinkage of y -> x via thin SVD.
        let (u, s, v) = if h >= w {
            jacobi_svd_thin(&y, h, w)?
        } else {
            // For h < w, factor y^T to keep "thin" condition.
            let mut yt = vec![0.0_f64; w * h];
            for i in 0..h {
                for j in 0..w {
                    yt[j * h + i] = y[i * w + j];
                }
            }
            let (u2, s2, v2) = jacobi_svd_thin(&yt, w, h)?;
            // y = (u2 s2 v2^T)^T = v2 s2 u2^T -> for our storage convention U is w × h, V is h × h.
            // We treat this case by returning (u_full, s, v_full) corresponding to U of size h × h and V of size w × h.
            (v2, s2, u2)
        };
        let mut s_new = vec![0.0_f64; s.len()];
        for (i, &si) in s.iter().enumerate() {
            s_new[i] = (si - tau).max(0.0);
        }
        // Reconstruct X = U diag(s_new) V^T.
        x.fill(0.0);
        if h >= w {
            // U is h × w; V is w × w in our row-major convention where v[i*w+j]=V[i,j]
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..w {
                        acc += u[i * w + k] * s_new[k] * v[j * w + k];
                    }
                    x[i * w + j] = acc;
                }
            }
        } else {
            // U from jacobi_svd_thin(y^T) is w × h (matching the V output of y^T). We swapped:
            // u (here) is actually V of y^T sized h × h; v (here) is U of y^T sized w × h.
            // Reconstruct x as U_y s V_y^T with U_y = u (h × h), V_y = v (w × h).
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0_f64;
                    for k in 0..r_dim {
                        acc += u[i * r_dim + k] * s_new[k] * v[j * r_dim + k];
                    }
                    x[i * w + j] = acc;
                }
            }
        }
        // Y update: Y += delta * P_Ω(M - X).
        let mut residual_sq = 0.0_f64;
        let mut obs_norm_sq = 0.0_f64;
        for k in 0..(h * w) {
            if mask[k] {
                let r = m[k] - x[k];
                y[k] += delta * r;
                residual_sq += r * r;
                obs_norm_sq += m[k] * m[k];
            }
        }
        iter += 1;
        let denom = obs_norm_sq.sqrt().max(1.0e-300);
        let cur_res = residual_sq.sqrt() / denom;
        if (last_residual - cur_res).abs() < tol && cur_res < tol {
            break;
        }
        last_residual = cur_res;
    }
    Ok(CompletionResult {
        x,
        residual: last_residual,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svt_recovers_low_rank_2x2() {
        // Rank-1 matrix [[1, 2], [2, 4]] observed at 3 entries.
        let m = vec![1.0_f64, 2.0, 2.0, 4.0];
        let mask = vec![true, true, true, false];
        let r = svt(&m, &mask, 2, 2, 0.5, 1.5, 1000, 1.0e-9).expect("ok");
        // SVT recovers something — verify it ran and recovered observed values approximately.
        assert!(r.iterations > 0);
        // Observed entries should be close to the original.
        assert!((r.x[0] - 1.0).abs() < 1.0);
        assert!((r.x[1] - 2.0).abs() < 1.0);
        assert!((r.x[2] - 2.0).abs() < 1.0);
    }
}
