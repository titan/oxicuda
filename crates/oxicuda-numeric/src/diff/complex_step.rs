//! Complex-step differentiation `f'(x) ≈ Im(f(x + i h)) / h`.
//!
//! Achieves machine-precision accuracy without subtractive cancellation, provided
//! `f` admits analytic continuation. The user must supply a "complex-step" friendly
//! function: one that performs the necessary algebraic operations on a tiny `(re, im)` pair.

#![allow(clippy::should_implement_trait)]

use crate::error::{NumericError, NumericResult};

/// A minimal complex type for the complex-step user function.
#[derive(Debug, Clone, Copy)]
pub struct CDual {
    pub re: f64,
    pub im: f64,
}

impl CDual {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
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

    pub fn sin(self) -> Self {
        // sin(a+bi) = sin a cosh b + i cos a sinh b
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    pub fn exp(self) -> Self {
        let ex = self.re.exp();
        Self {
            re: ex * self.im.cos(),
            im: ex * self.im.sin(),
        }
    }
}

/// Complex-step derivative of a user analytic function `f: CDual -> CDual` at real `x`.
pub fn complex_step_derivative<F>(f: F, x: f64, h: f64) -> NumericResult<f64>
where
    F: Fn(CDual) -> NumericResult<CDual>,
{
    if !h.is_finite() || h <= 0.0 {
        return Err(NumericError::InvalidStepSize { step: h });
    }
    let val = f(CDual::new(x, h))?;
    Ok(val.im / h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cstep_sin() {
        // d/dx sin x = cos x; at x = 1 → cos 1
        let f = |z: CDual| -> NumericResult<CDual> { Ok(z.sin()) };
        let d = complex_step_derivative(f, 1.0, 1.0e-30).expect("ok");
        assert!((d - 1.0_f64.cos()).abs() < 1.0e-12);
    }

    #[test]
    fn cstep_exp() {
        let f = |z: CDual| -> NumericResult<CDual> { Ok(z.exp()) };
        let d = complex_step_derivative(f, 0.5, 1.0e-30).expect("ok");
        assert!((d - 0.5_f64.exp()).abs() < 1.0e-12);
    }

    #[test]
    fn cstep_polynomial() {
        // f(z) = z²
        let f = |z: CDual| -> NumericResult<CDual> { Ok(z.mul(z)) };
        let d = complex_step_derivative(f, 3.0, 1.0e-30).expect("ok");
        assert!((d - 6.0).abs() < 1.0e-12);
    }
}
