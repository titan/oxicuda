//! Cholesky factorisation `A = L L^T` for symmetric positive-definite matrices.

use crate::error::{CsError, CsResult};

/// Factor SPD matrix A (n × n row-major) into lower-triangular L.
/// Returns L (row-major, n × n) with upper triangle zeroed.
pub fn cholesky_factor(a: &[f64], n: usize) -> CsResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(CsError::SingularMatrix(format!(
                        "Cholesky: non-positive pivot {s} at row {i}"
                    )));
                }
                l[i * n + j] = s.sqrt();
            } else {
                let d = l[j * n + j];
                if d.abs() < 1.0e-300 {
                    return Err(CsError::SingularMatrix(format!(
                        "Cholesky: zero diagonal at {j}"
                    )));
                }
                l[i * n + j] = s / d;
            }
        }
    }
    Ok(l)
}

/// Solve A x = b given L = chol(A) (lower-triangular row-major).
pub fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> CsResult<Vec<f64>> {
    if l.len() != n * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![l.len()],
        });
    }
    if b.len() != n {
        return Err(CsError::DimensionMismatch { a: b.len(), b: n });
    }
    // L y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let d = l[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CsError::SingularMatrix(format!("zero L[{i},{i}]")));
        }
        y[i] = s / d;
    }
    // L^T x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let d = l[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CsError::SingularMatrix(format!("zero L[{i},{i}]")));
        }
        x[i] = s / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let l = cholesky_factor(&a, 3).expect("ok");
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!((l[i * 3 + j] - exp).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn cholesky_solve_basic() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]], known SPD.
        let a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let b = vec![1.0, 2.0, 3.0];
        let l = cholesky_factor(&a, 3).expect("ok");
        let x = cholesky_solve(&l, 3, &b).expect("ok");
        // Verify A x ≈ b.
        let mut ax = vec![0.0_f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                ax[i] += a[i * 3 + j] * x[j];
            }
        }
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1.0e-9);
        }
    }

    #[test]
    fn cholesky_rejects_nonpd() {
        let a = vec![1.0, 2.0, 2.0, 1.0]; // indefinite
        assert!(cholesky_factor(&a, 2).is_err());
    }
}
