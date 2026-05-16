//! Matrix inverse via Gauss-Jordan elimination with partial pivoting.

use crate::error::{SurvivalError, SurvivalResult};

/// Compute the inverse of an n×n row-major matrix using Gauss-Jordan elimination.
pub(crate) fn gauss_jordan_inverse(a: &[f64], n: usize) -> SurvivalResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n * n],
            got: vec![a.len()],
        });
    }
    // Augmented matrix: [A | I], size n × 2n
    let m = 2 * n;
    let mut aug = vec![0.0_f64; n * m];
    for i in 0..n {
        for j in 0..n {
            aug[i * m + j] = a[i * n + j];
        }
        aug[i * m + n + i] = 1.0;
    }
    // Forward elimination with partial pivoting
    for k in 0..n {
        // pivot row
        let mut max_row = k;
        let mut max_val = aug[k * m + k].abs();
        for i in (k + 1)..n {
            let v = aug[i * m + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1.0e-14 {
            return Err(SurvivalError::SingularMatrix);
        }
        if max_row != k {
            for j in 0..m {
                aug.swap(k * m + j, max_row * m + j);
            }
        }
        // Normalise pivot row
        let pivot = aug[k * m + k];
        for j in 0..m {
            aug[k * m + j] /= pivot;
        }
        // Eliminate other rows
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = aug[i * m + k];
            if factor == 0.0 {
                continue;
            }
            for j in 0..m {
                aug[i * m + j] -= factor * aug[k * m + j];
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * m + n + j];
        }
    }
    Ok(inv)
}

/// Determinant of an n×n matrix via LU with partial pivoting.
pub(crate) fn determinant(a: &[f64], n: usize) -> SurvivalResult<f64> {
    if a.len() != n * n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n * n],
            got: vec![a.len()],
        });
    }
    let mut m = a.to_vec();
    let mut det = 1.0_f64;
    for k in 0..n {
        let mut max_row = k;
        let mut max_val = m[k * n + k].abs();
        for i in (k + 1)..n {
            let v = m[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1.0e-14 {
            return Ok(0.0);
        }
        if max_row != k {
            for j in 0..n {
                m.swap(k * n + j, max_row * n + j);
            }
            det = -det;
        }
        let pivot = m[k * n + k];
        det *= pivot;
        for i in (k + 1)..n {
            let factor = m[i * n + k] / pivot;
            for j in k..n {
                m[i * n + j] -= factor * m[k * n + j];
            }
        }
    }
    Ok(det)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_identity() {
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let inv = gauss_jordan_inverse(&i, 2).expect("ok");
        assert_eq!(inv, i);
    }

    #[test]
    fn inverse_2x2() {
        // A = [[1,2],[3,4]]; det = -2; A^-1 = [[-2, 1], [1.5, -0.5]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let inv = gauss_jordan_inverse(&a, 2).expect("ok");
        assert!((inv[0] + 2.0).abs() < 1.0e-10);
        assert!((inv[1] - 1.0).abs() < 1.0e-10);
        assert!((inv[2] - 1.5).abs() < 1.0e-10);
        assert!((inv[3] + 0.5).abs() < 1.0e-10);
    }

    #[test]
    fn det_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let d = determinant(&a, 2).expect("ok");
        assert!((d + 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn det_singular_zero() {
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let d = determinant(&a, 2).expect("ok");
        assert!(d.abs() < 1.0e-10);
    }
}
