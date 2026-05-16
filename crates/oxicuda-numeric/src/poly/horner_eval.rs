//! Horner's nested polynomial evaluation.
//!
//! `p(x) = ((a_n · x + a_{n-1}) · x + a_{n-2}) · x + … + a_0`.
//! Coefficients indexed by power: `coeffs[i] = a_i`.

use crate::error::{NumericError, NumericResult};

/// Evaluate `p(x)` via Horner. Empty `coeffs` is an error.
pub fn horner(coeffs: &[f64], x: f64) -> NumericResult<f64> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let n = coeffs.len();
    let mut acc = coeffs[n - 1];
    for i in (0..(n - 1)).rev() {
        acc = acc * x + coeffs[i];
    }
    Ok(acc)
}

/// Simultaneously evaluate `p(x)` and `p'(x)` via Horner.
pub fn horner_with_deriv(coeffs: &[f64], x: f64) -> NumericResult<(f64, f64)> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let n = coeffs.len();
    let mut p = coeffs[n - 1];
    let mut dp = 0.0_f64;
    for i in (0..(n - 1)).rev() {
        dp = dp * x + p;
        p = p * x + coeffs[i];
    }
    Ok((p, dp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horner_constant() {
        let p = vec![3.5_f64];
        assert!((horner(&p, 7.0).expect("ok") - 3.5).abs() < 1.0e-12);
    }

    #[test]
    fn horner_linear() {
        // p(x) = 2 + 3 x
        let p = vec![2.0_f64, 3.0];
        assert!((horner(&p, 4.0).expect("ok") - 14.0).abs() < 1.0e-12);
    }

    #[test]
    fn horner_cubic() {
        // p(x) = 1 + 2x + 3x² + 4x³  at x=2 → 1+4+12+32 = 49
        let p = vec![1.0_f64, 2.0, 3.0, 4.0];
        assert!((horner(&p, 2.0).expect("ok") - 49.0).abs() < 1.0e-12);
    }

    #[test]
    fn horner_deriv() {
        // p(x)=x²+2x+1; p'(x)=2x+2; at x=3 → p=16, p'=8
        let p = vec![1.0_f64, 2.0, 1.0];
        let (v, d) = horner_with_deriv(&p, 3.0).expect("ok");
        assert!((v - 16.0).abs() < 1.0e-12);
        assert!((d - 8.0).abs() < 1.0e-12);
    }

    #[test]
    fn horner_empty_err() {
        let r = horner(&[] as &[f64], 1.0);
        assert!(matches!(r, Err(NumericError::EmptyInput)));
    }
}
