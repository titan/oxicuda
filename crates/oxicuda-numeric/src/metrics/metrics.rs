//! Error norms, condition numbers, and residuals.

use crate::error::{NumericError, NumericResult};

/// Absolute error `|a - b|`.
pub fn absolute_error(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

/// Relative error `|a - b| / max(|b|, ε)`.
pub fn relative_error(a: f64, b: f64) -> f64 {
    let denom = b.abs().max(1.0e-300);
    (a - b).abs() / denom
}

/// Max-norm of a vector.
pub fn max_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

/// Compute residual norm `||A x - b||_2` for an `m × n` matrix `A` in row-major.
pub fn residual_norm(a: &[f64], m: usize, n: usize, x: &[f64], b: &[f64]) -> NumericResult<f64> {
    if a.len() != m * n {
        return Err(NumericError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if x.len() != n {
        return Err(NumericError::DimensionMismatch { a: x.len(), b: n });
    }
    if b.len() != m {
        return Err(NumericError::DimensionMismatch { a: b.len(), b: m });
    }
    let mut acc = 0.0_f64;
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        let r = s - b[i];
        acc += r * r;
    }
    Ok(acc.sqrt())
}

/// 2-norm condition number of a 2×2 matrix.
pub fn condition_number_two_by_two(a: &[f64]) -> NumericResult<f64> {
    if a.len() != 4 {
        return Err(NumericError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![a.len()],
        });
    }
    let (a11, a12, a21, a22) = (a[0], a[1], a[2], a[3]);
    let trace = (a11 * a11 + a12 * a12) + (a21 * a21 + a22 * a22);
    let det = a11 * a22 - a12 * a21;
    let disc = trace * trace - 4.0 * det * det;
    if disc < 0.0 {
        return Err(NumericError::NumericalInstability(
            "negative discriminant (singular?)".into(),
        ));
    }
    let s = disc.sqrt();
    let sigma_max_sq = 0.5 * (trace + s);
    let sigma_min_sq = 0.5 * (trace - s);
    if sigma_min_sq <= 0.0 {
        return Err(NumericError::SingularMatrix(
            "smallest singular value is zero".into(),
        ));
    }
    Ok((sigma_max_sq / sigma_min_sq).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_err_basic() {
        assert!((absolute_error(1.0, 2.0) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn rel_err_basic() {
        assert!((relative_error(2.0, 1.0) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn max_norm_works() {
        let v = vec![1.0_f64, -3.0, 2.0];
        assert!((max_norm(&v) - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn residual_zero_at_solution() {
        let a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = vec![1.0_f64, 2.0];
        let x = vec![1.0_f64, 2.0];
        let r = residual_norm(&a, 2, 2, &x, &b).expect("ok");
        assert!(r < 1.0e-12);
    }

    #[test]
    fn cond_identity() {
        let a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let c = condition_number_two_by_two(&a).expect("ok");
        assert!((c - 1.0).abs() < 1.0e-10);
    }
}
