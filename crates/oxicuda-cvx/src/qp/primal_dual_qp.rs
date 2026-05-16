//! Primal-dual interior-point QP for `min ½ x^T P x + q^T x  s.t. A x = b, x ≥ 0`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;

/// QP primal-dual IP result.
#[derive(Debug, Clone)]
pub struct QpIpResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub iter: usize,
    pub mu: f64,
}

/// Long-step primal-dual interior-point QP.
pub fn primal_dual_qp(
    p_mat: &[f64],
    n: usize,
    q: &[f64],
    a: &[f64],
    m: usize,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<QpIpResult> {
    if p_mat.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p_mat.len()],
        });
    }
    if q.len() != n {
        return Err(CvxError::DimensionMismatch { a: q.len(), b: n });
    }
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    let mut x = vec![1.0_f64; n];
    let mut y = vec![0.0_f64; m];
    let mut z = vec![1.0_f64; n];
    for it in 0..max_iter {
        // Residuals.
        let p_x = mat_vec(p_mat, n, n, &x)?;
        let at_y = mat_t_vec(a, m, n, &y)?;
        let mut r_d = vec![0.0_f64; n];
        for j in 0..n {
            r_d[j] = p_x[j] + q[j] - at_y[j] - z[j];
        }
        let ax = mat_vec(a, m, n, &x)?;
        let mut r_p = vec![0.0_f64; m];
        for i in 0..m {
            r_p[i] = ax[i] - b[i];
        }
        let mu: f64 = (0..n).map(|j| x[j] * z[j]).sum::<f64>() / n as f64;
        if norm2(&r_p) < tol && norm2(&r_d) < tol && mu < tol {
            return Ok(QpIpResult {
                x,
                y,
                z,
                iter: it,
                mu,
            });
        }
        // Newton system: solve reduced via Schur.
        // Equations: P dx − A^T dy − dz = -r_d
        //            A dx = -r_p
        //            Z dx + X dz = -r_xz   where r_xz = x ⊙ z − σ μ
        // From row 3: dz = (-r_xz - Z dx) / X.
        // Substitute into row 1: (P + Z X⁻¹) dx − A^T dy = -r_d + (r_xz / X) = -r_d + Z⁻¹ (...).
        // Actually: dz = (-r_xz − z dx) / x → P dx − A^T dy − (-r_xz − z dx)/x = -r_d.
        //   (P + Z X⁻¹) dx − A^T dy = -r_d − r_xz / x.  with Z X⁻¹ = diag(z/x).
        let sigma = 0.1_f64;
        let r_xz: Vec<f64> = (0..n).map(|j| x[j] * z[j] - sigma * mu).collect();
        let mut m_mat = vec![0.0_f64; (n + m) * (n + m)];
        for i in 0..n {
            for j in 0..n {
                m_mat[i * (n + m) + j] = p_mat[i * n + j];
            }
            m_mat[i * (n + m) + i] += z[i] / x[i];
            for k in 0..m {
                m_mat[i * (n + m) + n + k] = -a[k * n + i];
            }
        }
        for k in 0..m {
            for j in 0..n {
                m_mat[(n + k) * (n + m) + j] = a[k * n + j];
            }
        }
        let mut rhs = vec![0.0_f64; n + m];
        for j in 0..n {
            rhs[j] = -r_d[j] - r_xz[j] / x[j];
        }
        for k in 0..m {
            rhs[n + k] = -r_p[k];
        }
        let sol = solve_dense(&m_mat, n + m, &rhs)?;
        let dx: Vec<f64> = sol[..n].to_vec();
        let dy: Vec<f64> = sol[n..].to_vec();
        let mut dz = vec![0.0_f64; n];
        for j in 0..n {
            dz[j] = (-r_xz[j] - z[j] * dx[j]) / x[j];
        }
        // Step lengths.
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
    fn pdqp_simple() {
        // min 0.5 (x²) s.t. x + s = 1, x, s ≥ 0; optimum at x=0.
        let p_mat = vec![1.0_f64, 0.0, 0.0, 0.0];
        let q = vec![0.0_f64, 0.0];
        let a = vec![1.0_f64, 1.0];
        let b = vec![1.0_f64];
        let res = primal_dual_qp(&p_mat, 2, &q, &a, 1, &b, 100, 1.0e-7).expect("ok");
        // Solution: x=0, s=1 gives obj 0; alternative x=1, s=0 gives 0.5.
        assert!(res.x[0].abs() < 1.0e-3);
    }
}
