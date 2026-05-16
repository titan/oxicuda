//! Polynomial deflation: divide `p(x)` by `(x - r)` once a root `r` is known.
//!
//! Synthetic division: given `p(x) = a_n x^n + … + a_0`, the deflated polynomial
//! `q(x) = p(x)/(x − r)` has coefficients `b` with `b_{n-1} = a_n`, `b_{k-1} = a_k + r · b_k`.

use crate::error::{NumericError, NumericResult};

/// Deflate the polynomial `p` (indexed by power) by the simple root `r`.
/// Returns the `n-1`-degree quotient. The remainder (≈ `p(r)`) is discarded.
pub fn deflate_polynomial(coeffs: &[f64], r: f64) -> NumericResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let n = coeffs.len();
    if n == 1 {
        return Err(NumericError::DegreeTooHigh {
            degree: 0,
            limit: 1,
        });
    }
    let mut b = vec![0.0_f64; n - 1];
    b[n - 2] = coeffs[n - 1];
    for k in (1..(n - 1)).rev() {
        b[k - 1] = coeffs[k] + r * b[k];
    }
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflate_quadratic_via_root_one() {
        // (x - 1)(x - 2) = x² - 3x + 2  ⇒  coeffs = [2, -3, 1]
        let p = vec![2.0_f64, -3.0, 1.0];
        let q = deflate_polynomial(&p, 1.0).expect("ok");
        // expected: x - 2 ⇒ [-2, 1]
        assert_eq!(q.len(), 2);
        assert!((q[0] + 2.0).abs() < 1.0e-12);
        assert!((q[1] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn deflate_cubic_three_roots() {
        // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6  → coeffs = [-6, 11, -6, 1]
        let p = vec![-6.0_f64, 11.0, -6.0, 1.0];
        let q = deflate_polynomial(&p, 1.0).expect("ok");
        // expected (x-2)(x-3) = x² - 5x + 6 ⇒ [6, -5, 1]
        assert!((q[0] - 6.0).abs() < 1.0e-10);
        assert!((q[1] + 5.0).abs() < 1.0e-10);
        assert!((q[2] - 1.0).abs() < 1.0e-10);
    }
}
