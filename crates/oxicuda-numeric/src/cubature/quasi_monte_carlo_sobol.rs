//! Quasi-Monte Carlo with a simple low-discrepancy sequence.
//!
//! Implements the van der Corput sequence (Sobol base-2 first dimension) for each
//! dimension using direction numbers derived from primitive polynomials. For simplicity
//! we use the radical-inverse in bases (2, 3, 5, 7, …) — formally a Halton sequence,
//! which shares the dispersion properties of Sobol for small dimensions.

use crate::error::{NumericError, NumericResult};

const PRIMES: [u32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

fn radical_inverse(i: u32, base: u32) -> f64 {
    let mut n = i;
    let mut f = 1.0_f64 / base as f64;
    let mut r = 0.0_f64;
    while n > 0 {
        r += (n % base) as f64 * f;
        n /= base;
        f /= base as f64;
    }
    r
}

/// Return the `i`-th Halton/Sobol-base-2 point in `[0, 1)^d`.
pub fn sobol_point(i: u32, d: usize) -> NumericResult<Vec<f64>> {
    if d == 0 || d > PRIMES.len() {
        return Err(NumericError::InvalidParameter(format!(
            "dimension d must satisfy 1 ≤ d ≤ {}",
            PRIMES.len()
        )));
    }
    let mut out = Vec::with_capacity(d);
    for &p in PRIMES.iter().take(d) {
        out.push(radical_inverse(i + 1, p));
    }
    Ok(out)
}

/// Quasi-Monte Carlo integration of `f` over an axis-aligned box.
pub fn sobol_integrate<F>(f: F, lo: &[f64], hi: &[f64], n_samples: usize) -> NumericResult<f64>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    if lo.len() != hi.len() {
        return Err(NumericError::DimensionMismatch {
            a: lo.len(),
            b: hi.len(),
        });
    }
    if lo.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let d = lo.len();
    let mut vol = 1.0_f64;
    for i in 0..d {
        vol *= hi[i] - lo[i];
    }
    let mut sum = 0.0_f64;
    for i in 0..n_samples {
        let u = sobol_point(i as u32, d)?;
        let x: Vec<f64> = (0..d).map(|k| lo[k] + u[k] * (hi[k] - lo[k])).collect();
        sum += f(&x)?;
    }
    Ok(vol * sum / n_samples as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobol_first_dim_van_der_corput() {
        // i = 1 (one-indexed): radical_inverse(1, 2) = 0.5
        let p = sobol_point(0, 1).expect("ok");
        assert!((p[0] - 0.5).abs() < 1.0e-12);
        let p = sobol_point(1, 1).expect("ok");
        // i=2 → 0.25
        assert!((p[0] - 0.25).abs() < 1.0e-12);
        let p = sobol_point(2, 1).expect("ok");
        // i=3 → 0.75
        assert!((p[0] - 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn sobol_constant_integration() {
        let f = |_x: &[f64]| -> NumericResult<f64> { Ok(1.0) };
        let v = sobol_integrate(f, &[0.0, 0.0], &[1.0, 1.0], 256).expect("ok");
        assert!((v - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn sobol_polynomial_2d() {
        // ∫_0^1 ∫_0^1 (x + y) dx dy = 1
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0] + x[1]) };
        let v = sobol_integrate(f, &[0.0, 0.0], &[1.0, 1.0], 2048).expect("ok");
        assert!((v - 1.0).abs() < 1.0e-2);
    }
}
