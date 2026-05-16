//! Householder QR decomposition for `m × n` matrices (m ≥ n).

use crate::error::{CvxError, CvxResult};

/// Householder QR: A (m × n, m ≥ n) is decomposed as `A = Q R`.
///
/// Returns `(q, r)` both row-major: `Q` is `m × m`, `R` is `m × n` (upper-triangular block).
pub fn householder_qr(a: &[f64], m: usize, n: usize) -> CvxResult<(Vec<f64>, Vec<f64>)> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if m < n {
        return Err(CvxError::InvalidParameter(format!(
            "QR requires m≥n, got m={m}, n={n}"
        )));
    }
    let mut r = a.to_vec();
    // Initialise Q as identity m × m.
    let mut q = vec![0.0_f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    let p = n.min(m);
    for k in 0..p {
        // Form v from column k below row k.
        let mut sigma_sq = 0.0_f64;
        for i in k..m {
            sigma_sq += r[i * n + k] * r[i * n + k];
        }
        if sigma_sq < 1.0e-300 {
            continue;
        }
        let sigma = sigma_sq.sqrt();
        let alpha = if r[k * n + k] >= 0.0 { -sigma } else { sigma };
        let mut v = vec![0.0_f64; m - k];
        v[0] = r[k * n + k] - alpha;
        for i in 1..(m - k) {
            v[i] = r[(k + i) * n + k];
        }
        let mut beta_sq = 0.0_f64;
        for &vi in &v {
            beta_sq += vi * vi;
        }
        if beta_sq < 1.0e-300 {
            continue;
        }
        let inv_half_beta = 2.0 / beta_sq;
        // Apply H to R from the left for columns k..n.
        for j in k..n {
            let mut s = 0.0_f64;
            for i in 0..(m - k) {
                s += v[i] * r[(k + i) * n + j];
            }
            s *= inv_half_beta;
            for i in 0..(m - k) {
                r[(k + i) * n + j] -= s * v[i];
            }
        }
        // Apply H to Q from the right: Q_new = Q (I - 2 v v^T / v^Tv).
        // Equivalent to: Q_new = Q - (Q v) (2 v^T / v^Tv).
        for i in 0..m {
            let mut s = 0.0_f64;
            for jj in 0..(m - k) {
                s += q[i * m + (k + jj)] * v[jj];
            }
            s *= inv_half_beta;
            for jj in 0..(m - k) {
                q[i * m + (k + jj)] -= s * v[jj];
            }
        }
    }
    // Zero-out below diag.
    for i in 1..m {
        let upto = i.min(n);
        for j in 0..upto {
            r[i * n + j] = 0.0;
        }
    }
    Ok((q, r))
}

/// Solve `A x = b` for full-rank `m × n` `A` (m ≥ n) via QR. Least squares if `m > n`.
pub fn qr_solve(a: &[f64], m: usize, n: usize, b: &[f64]) -> CvxResult<Vec<f64>> {
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    let (q, r) = householder_qr(a, m, n)?;
    // y = Q^T b (length m).
    let mut y = vec![0.0_f64; m];
    for j in 0..m {
        let mut s = 0.0_f64;
        for i in 0..m {
            s += q[i * m + j] * b[i];
        }
        y[j] = s;
    }
    // Solve R x = y (top n × n block).
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= r[i * n + j] * x[j];
        }
        let d = r[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CvxError::SingularMatrix(format!("zero R[{i},{i}]")));
        }
        x[i] = s / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (q, r) = householder_qr(&a, 3, 3).expect("ok");
        // Q should be orthogonal — Q^T Q = I.
        let mut qtq = vec![0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0_f64;
                for k in 0..3 {
                    s += q[k * 3 + i] * q[k * 3 + j];
                }
                qtq[i * 3 + j] = s;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[i * 3 + j] - exp).abs() < 1.0e-10);
            }
        }
        // R upper triangular.
        for i in 1..3 {
            for j in 0..i {
                assert!(r[i * 3 + j].abs() < 1.0e-10);
            }
        }
    }

    #[test]
    fn qr_solve_square() {
        // A = [[1,2],[3,4]], b = [5, 11]; x = [1, 2].
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 11.0];
        let x = qr_solve(&a, 2, 2, &b).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-9);
        assert!((x[1] - 2.0).abs() < 1.0e-9);
    }

    #[test]
    fn qr_solve_least_squares() {
        // Overdetermined: A = [[1,0],[0,1],[1,1]], b = [1, 1, 2]; exact x = [1,1].
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 2.0];
        let x = qr_solve(&a, 3, 2, &b).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-9);
        assert!((x[1] - 1.0).abs() < 1.0e-9);
    }
}
