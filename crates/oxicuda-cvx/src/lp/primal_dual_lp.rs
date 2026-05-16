//! Primal-dual interior-point LP (simple long-step central-path follower).
//!
//! Solves `min cᵀx  s.t. A x = b, x ≥ 0`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// Long-step primal-dual interior-point.  Returns `(x, y, z)`.
pub fn primal_dual_lp(
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
    // Initialise interior point.
    let mut x = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; m];
    let mut z = vec![1.0_f64; n];
    let mut iters = 0usize;
    for it in 0..max_iter {
        // Residuals.
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
        let primal_inf = norm2(&r_p);
        let dual_inf = norm2(&r_d);
        if primal_inf < tol && dual_inf < tol && mu < tol {
            return Ok(LpResult {
                x,
                y,
                z,
                iter: it,
                mu,
            });
        }
        // Newton step with σ = 0.1 (centring).
        // KKT residuals: r_p = Ax-b, r_d = A^T y + z - c, r_xz = XZe - σμe.
        // Reduced system: (A X Z^{-1} A^T) dy = -r_p + A Z^{-1} r_xz - A X Z^{-1} r_d.
        let sigma = 0.1_f64;
        let r_xz: Vec<f64> = (0..n).map(|j| x[j] * z[j] - sigma * mu).collect();
        // Form M = A X Z⁻¹ Aᵀ (symmetric m × m positive semidefinite).
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
        // Build RHS.
        let mut tmp = vec![0.0_f64; n];
        for k in 0..n {
            tmp[k] = r_xz[k] / z[k] - (x[k] / z[k]) * r_d[k];
        }
        let a_tmp = mat_vec(a, m, n, &tmp)?;
        let mut rhs = vec![0.0_f64; m];
        for i in 0..m {
            rhs[i] = -r_p[i] + a_tmp[i];
        }
        let dy = solve_dense(&m_mat, m, &rhs)?;
        // dz = -r_d - Aᵀ dy
        let at_dy = mat_t_vec(a, m, n, &dy)?;
        let mut dz = vec![0.0_f64; n];
        for j in 0..n {
            dz[j] = -r_d[j] - at_dy[j];
        }
        // dx = (-r_xz - x dz) / z
        let mut dx = vec![0.0_f64; n];
        for j in 0..n {
            dx[j] = (-r_xz[j] - x[j] * dz[j]) / z[j];
        }
        // Step length.
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
        let alpha = safety * alpha_p.min(alpha_d);
        for j in 0..n {
            x[j] += alpha * dx[j];
            z[j] += alpha * dz[j];
        }
        for i in 0..m {
            y[i] += alpha * dy[i];
        }
        iters = it + 1;
    }
    let mu = (0..n).map(|j| x[j] * z[j]).sum::<f64>() / n as f64;
    Err(CvxError::NotConverged {
        iter: iters,
        residual: mu,
    })
}

/// LP IP result.
#[derive(Debug, Clone)]
pub struct LpResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub iter: usize,
    pub mu: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdlp_simple() {
        // min -x - y s.t. x + y + s = 1; x, y, s ≥ 0.  Optimum value: -1.
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64];
        let c = vec![-1.0_f64, -1.0, 0.0];
        let res = primal_dual_lp(&a, 1, 3, &b, &c, 100, 1.0e-7).expect("ok");
        let obj: f64 = res.x.iter().zip(c.iter()).map(|(xi, ci)| xi * ci).sum();
        assert!((obj + 1.0).abs() < 1.0e-3);
    }
}
