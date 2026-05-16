//! Mehrotra Predictor-Corrector interior-point LP.
//!
//! Solves `min cᵀx s.t. A x = b, x ≥ 0` via the classical PC method:
//!   1. Predictor (affine, σ=0).
//!   2. Centring parameter σ = (μ_aff/μ)³.
//!   3. Corrector with σμ centring term and ΔxΔz cross term.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;
use crate::lp::primal_dual_lp::LpResult;

/// Mehrotra PC interior-point.
pub fn mehrotra_predictor_corrector(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    c: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<LpResult> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m || c.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    let mut x = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; m];
    let mut z = vec![1.0_f64; n];
    for it in 0..max_iter {
        let ax = mat_vec(a, m, n, &x)?;
        let mut r_p = vec![0.0_f64; m];
        for i in 0..m {
            r_p[i] = ax[i] - b[i];
        }
        let at_y = mat_t_vec(a, m, n, &y)?;
        let mut r_d = vec![0.0_f64; n];
        for j in 0..n {
            r_d[j] = at_y[j] + z[j] - c[j];
        }
        let mu: f64 = (0..n).map(|j| x[j] * z[j]).sum::<f64>() / n as f64;
        if norm2(&r_p) < tol && norm2(&r_d) < tol && mu < tol {
            return Ok(LpResult {
                x,
                y,
                z,
                iter: it,
                mu,
            });
        }
        // Form M = A · diag(x/z) · Aᵀ.
        let mut m_mat = vec![0.0_f64; m * m];
        for i in 0..m {
            for jc in 0..m {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    let ratio = x[k] / z[k];
                    acc += a[i * n + k] * a[jc * n + k] * ratio;
                }
                m_mat[i * m + jc] = acc;
            }
        }
        // === Predictor (σ=0). ===
        let r_xz: Vec<f64> = (0..n).map(|j| x[j] * z[j]).collect();
        let mut tmp = vec![0.0_f64; n];
        for k in 0..n {
            tmp[k] = r_xz[k] / z[k] - (x[k] / z[k]) * r_d[k];
        }
        let a_tmp = mat_vec(a, m, n, &tmp)?;
        let mut rhs = vec![0.0_f64; m];
        for i in 0..m {
            rhs[i] = -r_p[i] + a_tmp[i];
        }
        let dy_a = solve_dense(&m_mat, m, &rhs)?;
        let at_dy_a = mat_t_vec(a, m, n, &dy_a)?;
        let mut dz_a = vec![0.0_f64; n];
        for j in 0..n {
            dz_a[j] = -r_d[j] - at_dy_a[j];
        }
        let mut dx_a = vec![0.0_f64; n];
        for j in 0..n {
            dx_a[j] = (-r_xz[j] - x[j] * dz_a[j]) / z[j];
        }
        let mut alpha_p_aff = 1.0_f64;
        let mut alpha_d_aff = 1.0_f64;
        for j in 0..n {
            if dx_a[j] < 0.0 {
                let r = -x[j] / dx_a[j];
                if r < alpha_p_aff {
                    alpha_p_aff = r;
                }
            }
            if dz_a[j] < 0.0 {
                let r = -z[j] / dz_a[j];
                if r < alpha_d_aff {
                    alpha_d_aff = r;
                }
            }
        }
        // μ_aff and σ.
        let mu_aff: f64 = (0..n)
            .map(|j| (x[j] + alpha_p_aff * dx_a[j]) * (z[j] + alpha_d_aff * dz_a[j]))
            .sum::<f64>()
            / n as f64;
        let sigma = if mu < 1.0e-300 {
            0.0
        } else {
            (mu_aff / mu).powi(3).clamp(0.0, 1.0)
        };
        // === Corrector RHS: r_xz with σ·μ + Δx_a · Δz_a. ===
        let mut r_xz_c = vec![0.0_f64; n];
        for j in 0..n {
            r_xz_c[j] = x[j] * z[j] + dx_a[j] * dz_a[j] - sigma * mu;
        }
        let mut tmp_c = vec![0.0_f64; n];
        for k in 0..n {
            tmp_c[k] = r_xz_c[k] / z[k] - (x[k] / z[k]) * r_d[k];
        }
        let a_tmp_c = mat_vec(a, m, n, &tmp_c)?;
        let mut rhs_c = vec![0.0_f64; m];
        for i in 0..m {
            rhs_c[i] = -r_p[i] + a_tmp_c[i];
        }
        let dy = solve_dense(&m_mat, m, &rhs_c)?;
        let at_dy = mat_t_vec(a, m, n, &dy)?;
        let mut dz = vec![0.0_f64; n];
        for j in 0..n {
            dz[j] = -r_d[j] - at_dy[j];
        }
        let mut dx = vec![0.0_f64; n];
        for j in 0..n {
            dx[j] = (-r_xz_c[j] - x[j] * dz[j]) / z[j];
        }
        // Step length with safety 0.99.
        let mut alpha_p = 1.0_f64;
        let mut alpha_d = 1.0_f64;
        for j in 0..n {
            if dx[j] < 0.0 {
                let r = -x[j] / dx[j];
                if r < alpha_p {
                    alpha_p = r;
                }
            }
            if dz[j] < 0.0 {
                let r = -z[j] / dz[j];
                if r < alpha_d {
                    alpha_d = r;
                }
            }
        }
        let safety = 0.99_f64;
        let alpha_p_final = safety * alpha_p;
        let alpha_d_final = safety * alpha_d;
        for j in 0..n {
            x[j] += alpha_p_final * dx[j];
            z[j] += alpha_d_final * dz[j];
        }
        for i in 0..m {
            y[i] += alpha_d_final * dy[i];
        }
    }
    let mu = (0..n).map(|j| x[j] * z[j]).sum::<f64>() / n as f64;
    Err(CvxError::NotConverged {
        iter: max_iter,
        residual: mu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mehrotra_simple_lp() {
        // min -x-y s.t. x+y+s=1; x,y,s ≥ 0. Optimum ≈ -1.
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64];
        let c = vec![-1.0_f64, -1.0, 0.0];
        let res = mehrotra_predictor_corrector(&a, 1, 3, &b, &c, 100, 1.0e-7).expect("ok");
        let obj: f64 = res.x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum();
        assert!((obj + 1.0).abs() < 1.0e-3);
    }
}
