//! Symmetric Positive-Definite (SPD) cone with affine-invariant metric.
//!
//! For `P, Q in SPD(n)` and tangent `X` (symmetric):
//! - inner product: `g_P(X, Y) = tr(P^{-1} X P^{-1} Y)`
//! - exp_P(X) = P^{1/2} exp(P^{-1/2} X P^{-1/2}) P^{1/2}
//! - log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
//! - distance d(P, Q) = ||log(P^{-1/2} Q P^{-1/2})||_F

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;

/// Project a matrix onto the symmetric subspace.
pub fn spd_project_symmetric(m: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if m.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![m.len()],
        });
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (m[i * n + j] + m[j * n + i]);
        }
    }
    Ok(out)
}

/// Compute `M^{1/2}` and `M^{-1/2}` for a symmetric matrix `M`.
fn sym_sqrt_pair(m: &[f64], n: usize) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    let (w, v) = jacobi_eigh(m, n)?;
    for wi in &w {
        if *wi < 0.0 && wi.abs() > 1e-6 {
            return Err(ManifoldError::ManifoldConstraint(
                "spd: matrix has non-trivial negative eigenvalue".into(),
            ));
        }
    }
    let mut sq = vec![0.0; n * n];
    let mut inv_sq = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc_p = 0.0;
            let mut acc_n = 0.0;
            for k in 0..n {
                let lam = w[k].max(1e-14);
                let s = lam.sqrt();
                acc_p += v[i * n + k] * v[j * n + k] * s;
                acc_n += v[i * n + k] * v[j * n + k] / s;
            }
            sq[i * n + j] = acc_p;
            inv_sq[i * n + j] = acc_n;
        }
    }
    Ok((sq, inv_sq))
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// SPD exponential map at `P` of tangent vector `x` (symmetric).
pub fn spd_exp(p_mat: &[f64], x: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if p_mat.len() != n * n || x.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p_mat.len()],
        });
    }
    let (sq, inv_sq) = sym_sqrt_pair(p_mat, n)?;
    let mid = matmul(&matmul(&inv_sq, x, n), &inv_sq, n);
    // Compute matrix exponential of symmetric mid via eigendecomposition
    let mid_sym = spd_project_symmetric(&mid, n)?;
    let (w, v) = jacobi_eigh(&mid_sym, n)?;
    let mut exp_mid = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += v[i * n + k] * v[j * n + k] * w[k].exp();
            }
            exp_mid[i * n + j] = acc;
        }
    }
    let out = matmul(&matmul(&sq, &exp_mid, n), &sq, n);
    Ok(out)
}

/// SPD logarithmic map at `P` of point `Q`.
pub fn spd_log(p_mat: &[f64], q_mat: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if p_mat.len() != n * n || q_mat.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![p_mat.len()],
        });
    }
    let (sq, inv_sq) = sym_sqrt_pair(p_mat, n)?;
    let mid = matmul(&matmul(&inv_sq, q_mat, n), &inv_sq, n);
    let mid_sym = spd_project_symmetric(&mid, n)?;
    let (w, v) = jacobi_eigh(&mid_sym, n)?;
    let mut log_mid = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                let lam = w[k].max(1e-14);
                acc += v[i * n + k] * v[j * n + k] * lam.ln();
            }
            log_mid[i * n + j] = acc;
        }
    }
    Ok(matmul(&matmul(&sq, &log_mid, n), &sq, n))
}

/// Affine-invariant SPD distance.
pub fn spd_distance(p_mat: &[f64], q_mat: &[f64], n: usize) -> ManifoldResult<f64> {
    let (_sq, inv_sq) = sym_sqrt_pair(p_mat, n)?;
    let mid = matmul(&matmul(&inv_sq, q_mat, n), &inv_sq, n);
    let mid_sym = spd_project_symmetric(&mid, n)?;
    let (w, _v) = jacobi_eigh(&mid_sym, n)?;
    let mut s = 0.0;
    for lam in &w {
        let l = lam.max(1e-14).ln();
        s += l * l;
    }
    Ok(s.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spd_distance_identity_zero() {
        let n = 3;
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            p[i * n + i] = 1.0;
        }
        let d = spd_distance(&p, &p, n).expect("ok");
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn spd_exp_log_inverse() {
        let n = 2;
        // P = diag(2, 3)
        let p = vec![2.0, 0.0, 0.0, 3.0];
        // X (symmetric small)
        let x = vec![0.1, 0.0, 0.0, -0.1];
        let q = spd_exp(&p, &x, n).expect("ok");
        let xrec = spd_log(&p, &q, n).expect("ok");
        for (a, b) in x.iter().zip(&xrec) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
