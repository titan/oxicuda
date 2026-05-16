//! Aberth-Ehrlich method — all complex roots of a polynomial simultaneously.
//!
//! For polynomial `p(z)` of degree `n`:
//! 1. Initialise `z_i` evenly spaced on a circle of radius `R ≈ 1 + max|a_i|/|a_n|`.
//! 2. Iterate `z_i ← z_i - w_i` where
//!    `w_i = (p(z_i)/p'(z_i)) / (1 - (p(z_i)/p'(z_i)) · Σ_{j≠i} 1/(z_i - z_j))`.
//!
//! Convergence: cubic when roots are simple.

#![allow(clippy::should_implement_trait)]

use crate::error::{NumericError, NumericResult};

/// Lightweight complex number used by this module (mirrors `std::ops::*` semantics).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    pub fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }

    pub fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    pub fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

fn horner_complex(coeffs: &[f64], z: Complex64) -> Complex64 {
    let n = coeffs.len();
    if n == 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mut acc = Complex64::new(coeffs[n - 1], 0.0);
    for i in (0..(n - 1)).rev() {
        acc = acc.mul(z).add(Complex64::new(coeffs[i], 0.0));
    }
    acc
}

fn horner_complex_with_deriv(coeffs: &[f64], z: Complex64) -> (Complex64, Complex64) {
    let n = coeffs.len();
    if n == 0 {
        return (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    }
    let mut p = Complex64::new(coeffs[n - 1], 0.0);
    let mut dp = Complex64::new(0.0, 0.0);
    for i in (0..(n - 1)).rev() {
        dp = dp.mul(z).add(p);
        p = p.mul(z).add(Complex64::new(coeffs[i], 0.0));
    }
    (p, dp)
}

/// Compute all complex roots of polynomial `coeffs[0] + coeffs[1] z + … + coeffs[n] z^n`.
pub fn aberth_all_roots(
    coeffs: &[f64],
    tol: f64,
    max_iter: usize,
) -> NumericResult<Vec<Complex64>> {
    if coeffs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    if coeffs[coeffs.len() - 1].abs() < 1.0e-300 {
        return Err(NumericError::InvalidParameter(
            "leading coefficient is zero".into(),
        ));
    }
    let n = coeffs.len() - 1;
    if n == 0 {
        return Ok(vec![]);
    }
    let an = coeffs[n];
    let cauchy_radius = 1.0
        + coeffs[..n]
            .iter()
            .map(|c| (c / an).abs())
            .fold(0.0_f64, f64::max);
    let mut zs: Vec<Complex64> = (0..n)
        .map(|i| {
            let theta = std::f64::consts::TAU * (i as f64) / (n as f64) + 0.4;
            Complex64::from_polar(cauchy_radius, theta)
        })
        .collect();
    for _ in 0..max_iter {
        let mut max_step = 0.0_f64;
        let mut new_zs = zs.clone();
        for i in 0..n {
            let (p, dp) = horner_complex_with_deriv(coeffs, zs[i]);
            if dp.abs() < 1.0e-300 {
                new_zs[i] = Complex64::new(zs[i].re + 1.0e-6, zs[i].im + 1.0e-6);
                continue;
            }
            let ratio = p.div(dp);
            let mut sum_inv = Complex64::new(0.0, 0.0);
            for (j, zj) in zs.iter().enumerate() {
                if j == i {
                    continue;
                }
                let diff = zs[i].sub(*zj);
                if diff.abs() < 1.0e-300 {
                    continue;
                }
                sum_inv = sum_inv.add(Complex64::new(1.0, 0.0).div(diff));
            }
            let denom = Complex64::new(1.0, 0.0).sub(ratio.mul(sum_inv));
            if denom.abs() < 1.0e-300 {
                continue;
            }
            let w = ratio.div(denom);
            new_zs[i] = zs[i].sub(w);
            let step = w.abs();
            if step > max_step {
                max_step = step;
            }
        }
        zs = new_zs;
        if max_step < tol {
            return Ok(zs);
        }
    }
    let residuals: f64 = zs.iter().map(|z| horner_complex(coeffs, *z).abs()).sum();
    if residuals < tol * 100.0 {
        return Ok(zs);
    }
    Err(NumericError::NotConverged {
        iter: max_iter,
        residual: residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aberth_quadratic() {
        // x^2 - 1 = 0 → roots ±1
        let coeffs = vec![-1.0_f64, 0.0, 1.0];
        let roots = aberth_all_roots(&coeffs, 1.0e-12, 200).expect("ok");
        let mut reals: Vec<f64> = roots.iter().map(|z| z.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        assert!((reals[0] + 1.0).abs() < 1.0e-8);
        assert!((reals[1] - 1.0).abs() < 1.0e-8);
    }

    #[test]
    fn aberth_cubic() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let coeffs = vec![-6.0_f64, 11.0, -6.0, 1.0];
        let roots = aberth_all_roots(&coeffs, 1.0e-10, 300).expect("ok");
        let mut reals: Vec<f64> = roots.iter().map(|z| z.re).collect();
        reals.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        for (got, expected) in reals.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert!((got - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn aberth_complex_roots() {
        // x^2 + 1 = 0 → ±i
        let coeffs = vec![1.0_f64, 0.0, 1.0];
        let roots = aberth_all_roots(&coeffs, 1.0e-12, 200).expect("ok");
        let mut imags: Vec<f64> = roots.iter().map(|z| z.im).collect();
        imags.sort_by(|a, b| a.partial_cmp(b).expect("ord"));
        assert!((imags[0] + 1.0).abs() < 1.0e-6);
        assert!((imags[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn aberth_empty_err() {
        let res = aberth_all_roots(&[] as &[f64], 1.0e-12, 50);
        assert!(matches!(res, Err(NumericError::EmptyInput)));
    }
}
