//! Cholesky factorisation `A = L L^T` for a symmetric positive-definite matrix.

use crate::error::{SurvivalError, SurvivalResult};

/// Compute the lower-triangular Cholesky factor `L` (row-major) such that `A = L L^T`.
/// Adds a small jitter `ridge` to the diagonal for numerical stability.
pub(crate) fn cholesky(a: &[f64], n: usize, ridge: f64) -> SurvivalResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n * n],
            got: vec![a.len()],
        });
    }
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            if i == j {
                s += ridge;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(SurvivalError::SingularMatrix);
                }
                l[i * n + j] = s.sqrt();
            } else {
                let denom = l[j * n + j];
                if denom.abs() < f64::EPSILON {
                    return Err(SurvivalError::SingularMatrix);
                }
                l[i * n + j] = s / denom;
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_identity() {
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let l = cholesky(&i, 2, 0.0).expect("ok");
        assert_eq!(l, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn cholesky_spd_2x2() {
        // A = [[4, 2], [2, 5]]; L = [[2, 0], [1, 2]]
        let a = vec![4.0, 2.0, 2.0, 5.0];
        let l = cholesky(&a, 2, 0.0).expect("ok");
        assert!((l[0] - 2.0).abs() < 1.0e-10);
        assert!((l[1] - 0.0).abs() < 1.0e-10);
        assert!((l[2] - 1.0).abs() < 1.0e-10);
        assert!((l[3] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let a = vec![-1.0, 0.0, 0.0, -1.0];
        assert!(cholesky(&a, 2, 0.0).is_err());
    }
}
