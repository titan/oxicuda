//! Forward-mode automatic differentiation via dual numbers.
//!
//! A *dual number* `a + b ε` carries a value `a` (`real`) and a single
//! directional derivative `b` (`dual`), with the defining relation `ε² = 0`.
//! Propagating a dual number through a computation evaluates the function and its
//! derivative simultaneously and to machine precision (no truncation error and no
//! subtractive cancellation, unlike finite differences). For any differentiable
//! `g`, the chain rule gives
//!
//! ```text
//! g(a + b ε) = g(a) + g'(a) · b · ε,
//! ```
//!
//! so every elementary function is implemented by setting the new `real` to
//! `g(a)` and the new `dual` to `g'(a) · b`.
//!
//! This module provides the core single-direction [`Dual`] type with the full
//! complement of arithmetic operators (`Add`, `Sub`, `Mul`, `Div`, `Neg` and
//! their `*Assign` forms) and elementary functions (`exp`, `ln`, `sqrt`, `powf`,
//! `powi`, `recip`, `abs`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `asin`,
//! `acos`, `atan`), all composition-safe under nesting. On top of it sit the
//! forward-mode utilities:
//!
//! * [`derivative`] — `f'(x)` for a scalar function `f: Dual → Dual`.
//! * [`gradient`] — `∇f` of `f: &[Dual] → Dual` by `n` directional sweeps.
//! * [`jacobian`] — the Jacobian of `f: &[Dual] → Vec<Dual>` by `n` sweeps.
//! * [`jacobian_vector_product`] — `J · v` in a **single** sweep by seeding the
//!   dual parts with the direction `v`.
//!
//! Domain errors (e.g. `ln` of a non-positive argument, `asin` out of
//! `[-1, 1]`) are reported through [`NumericError`]; the utilities never panic.

#![allow(clippy::should_implement_trait)]

use crate::error::{NumericError, NumericResult};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A forward-mode dual number `real + dual · ε` (with `ε² = 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// The primal value.
    pub real: f64,
    /// The first-order derivative component (the `ε` coefficient).
    pub dual: f64,
}

impl Dual {
    /// Create a dual number with explicit primal and derivative parts.
    #[must_use]
    pub const fn new(real: f64, dual: f64) -> Self {
        Self { real, dual }
    }

    /// A constant: derivative part zero. Use for parameters held fixed.
    #[must_use]
    pub const fn constant(x: f64) -> Self {
        Self { real: x, dual: 0.0 }
    }

    /// An independent variable seeded for differentiation (derivative part one).
    #[must_use]
    pub const fn variable(x: f64) -> Self {
        Self { real: x, dual: 1.0 }
    }

    /// A variable seeded with an arbitrary direction (used for JVPs).
    #[must_use]
    pub const fn seeded(x: f64, direction: f64) -> Self {
        Self {
            real: x,
            dual: direction,
        }
    }

    /// `e^self`.
    #[must_use]
    pub fn exp(self) -> Self {
        let e = self.real.exp();
        Self {
            real: e,
            dual: e * self.dual,
        }
    }

    /// Natural logarithm `ln(self)`.
    ///
    /// # Errors
    /// [`NumericError::OutOfDomain`] if `real ≤ 0`.
    pub fn ln(self) -> NumericResult<Self> {
        if self.real <= 0.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "ln".into(),
            });
        }
        Ok(Self {
            real: self.real.ln(),
            dual: self.dual / self.real,
        })
    }

    /// Square root `sqrt(self)`.
    ///
    /// # Errors
    /// [`NumericError::OutOfDomain`] if `real < 0`.
    pub fn sqrt(self) -> NumericResult<Self> {
        if self.real < 0.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "sqrt".into(),
            });
        }
        let s = self.real.sqrt();
        // d/dx sqrt(x) = 1/(2 sqrt(x)); at x = 0 the derivative is +∞, which we
        // surface as an infinite dual component rather than panicking.
        let d = if s == 0.0 {
            if self.dual == 0.0 {
                0.0
            } else {
                f64::INFINITY * self.dual.signum()
            }
        } else {
            self.dual / (2.0 * s)
        };
        Ok(Self { real: s, dual: d })
    }

    /// Raise to a real power `self^p` (with `p` a plain `f64` constant).
    ///
    /// `d/dx x^p = p · x^{p-1}`.
    ///
    /// # Errors
    /// [`NumericError::OutOfDomain`] if `real < 0` and `p` is not an integer (the
    /// result would be complex), or if `real == 0` and `p < 1` with a non-zero
    /// derivative seed (the derivative would be singular).
    pub fn powf(self, p: f64) -> NumericResult<Self> {
        if self.real < 0.0 && p.fract() != 0.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "powf (negative base, non-integer exponent)".into(),
            });
        }
        let value = self.real.powf(p);
        let deriv = if self.dual == 0.0 {
            0.0
        } else if self.real == 0.0 {
            // x^p with x → 0: derivative p x^{p-1}.
            if p == 1.0 {
                p * self.dual
            } else if p > 1.0 {
                0.0
            } else {
                return Err(NumericError::OutOfDomain {
                    value: self.real,
                    function: "powf (zero base, exponent < 1 with derivative)".into(),
                });
            }
        } else {
            p * self.real.powf(p - 1.0) * self.dual
        };
        Ok(Self {
            real: value,
            dual: deriv,
        })
    }

    /// Raise to an integer power `self^n` (total, no domain restriction).
    ///
    /// `d/dx x^n = n · x^{n-1}` (with the `n = 0` derivative being zero).
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        let value = self.real.powi(n);
        let deriv = if n == 0 {
            0.0
        } else {
            (n as f64) * self.real.powi(n - 1) * self.dual
        };
        Self {
            real: value,
            dual: deriv,
        }
    }

    /// Reciprocal `1 / self`.
    ///
    /// # Errors
    /// [`NumericError::OutOfDomain`] if `real == 0`.
    pub fn recip(self) -> NumericResult<Self> {
        if self.real == 0.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "recip".into(),
            });
        }
        let r = self.real.recip();
        Ok(Self {
            real: r,
            dual: -self.dual * r * r,
        })
    }

    /// Absolute value `|self|`.
    ///
    /// `d/dx |x| = sign(x)`; the derivative at `x = 0` is taken as `0`
    /// (sub-gradient convention).
    #[must_use]
    pub fn abs(self) -> Self {
        let s = if self.real > 0.0 {
            1.0
        } else if self.real < 0.0 {
            -1.0
        } else {
            0.0
        };
        Self {
            real: self.real.abs(),
            dual: s * self.dual,
        }
    }

    /// Sine.
    #[must_use]
    pub fn sin(self) -> Self {
        Self {
            real: self.real.sin(),
            dual: self.real.cos() * self.dual,
        }
    }

    /// Cosine.
    #[must_use]
    pub fn cos(self) -> Self {
        Self {
            real: self.real.cos(),
            dual: -self.real.sin() * self.dual,
        }
    }

    /// Tangent.
    ///
    /// `d/dx tan(x) = sec²(x) = 1 + tan²(x)`. Total, mirroring [`f64::tan`]: at an
    /// odd multiple of `π/2` (never exactly representable) the result is the large
    /// finite `f64` value, with a correspondingly large derivative.
    #[must_use]
    pub fn tan(self) -> Self {
        let t = self.real.tan();
        Self {
            real: t,
            dual: (1.0 + t * t) * self.dual,
        }
    }

    /// Hyperbolic sine.
    #[must_use]
    pub fn sinh(self) -> Self {
        Self {
            real: self.real.sinh(),
            dual: self.real.cosh() * self.dual,
        }
    }

    /// Hyperbolic cosine.
    #[must_use]
    pub fn cosh(self) -> Self {
        Self {
            real: self.real.cosh(),
            dual: self.real.sinh() * self.dual,
        }
    }

    /// Hyperbolic tangent.
    ///
    /// `d/dx tanh(x) = 1 − tanh²(x)`.
    #[must_use]
    pub fn tanh(self) -> Self {
        let t = self.real.tanh();
        Self {
            real: t,
            dual: (1.0 - t * t) * self.dual,
        }
    }

    /// Inverse sine `asin(self)`.
    ///
    /// `d/dx asin(x) = 1 / sqrt(1 − x²)`.
    ///
    /// # Errors
    /// [`NumericError::OutOfDomain`] if `|real| > 1`, or `|real| == 1` with a
    /// non-zero derivative seed (the derivative is singular there).
    pub fn asin(self) -> NumericResult<Self> {
        if self.real.abs() > 1.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "asin".into(),
            });
        }
        let denom = 1.0 - self.real * self.real;
        let deriv = if self.dual == 0.0 {
            0.0
        } else if denom <= 0.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "asin (derivative singular at ±1)".into(),
            });
        } else {
            self.dual / denom.sqrt()
        };
        Ok(Self {
            real: self.real.asin(),
            dual: deriv,
        })
    }

    /// Inverse cosine `acos(self)`.
    ///
    /// `d/dx acos(x) = −1 / sqrt(1 − x²)`.
    ///
    /// # Errors
    /// [`NumericError::OutOfDomain`] if `|real| > 1`, or `|real| == 1` with a
    /// non-zero derivative seed.
    pub fn acos(self) -> NumericResult<Self> {
        if self.real.abs() > 1.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "acos".into(),
            });
        }
        let denom = 1.0 - self.real * self.real;
        let deriv = if self.dual == 0.0 {
            0.0
        } else if denom <= 0.0 {
            return Err(NumericError::OutOfDomain {
                value: self.real,
                function: "acos (derivative singular at ±1)".into(),
            });
        } else {
            -self.dual / denom.sqrt()
        };
        Ok(Self {
            real: self.real.acos(),
            dual: deriv,
        })
    }

    /// Inverse tangent `atan(self)`.
    ///
    /// `d/dx atan(x) = 1 / (1 + x²)`.
    #[must_use]
    pub fn atan(self) -> Self {
        Self {
            real: self.real.atan(),
            dual: self.dual / (1.0 + self.real * self.real),
        }
    }
}

impl From<f64> for Dual {
    fn from(x: f64) -> Self {
        Self::constant(x)
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // Leibniz: (a + b ε)(c + d ε) = ac + (ad + bc) ε.
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // Quotient rule; if rhs.real == 0 the result carries ±∞/NaN rather than
        // panicking (the fallible API is `recip`/explicit guards by the caller).
        let inv = 1.0 / rhs.real;
        Self {
            real: self.real * inv,
            dual: (self.dual * rhs.real - self.real * rhs.dual) * inv * inv,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

impl AddAssign for Dual {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Dual {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Dual {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for Dual {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Scalar (`f64`) mixed operators for ergonomic expressions like `2.0 * x`.
impl Add<f64> for Dual {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        self + Dual::constant(rhs)
    }
}
impl Add<Dual> for f64 {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual::constant(self) + rhs
    }
}
impl Sub<f64> for Dual {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        self - Dual::constant(rhs)
    }
}
impl Sub<Dual> for f64 {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual::constant(self) - rhs
    }
}
impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self * Dual::constant(rhs)
    }
}
impl Mul<Dual> for f64 {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual::constant(self) * rhs
    }
}
impl Div<f64> for Dual {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self / Dual::constant(rhs)
    }
}
impl Div<Dual> for f64 {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        Dual::constant(self) / rhs
    }
}

/// Forward-mode derivative `f'(x)` of a scalar function `f: Dual → Dual`.
///
/// Seeds `x` as an independent variable and reads off the derivative component.
///
/// # Errors
/// Propagates any error returned by `f`.
pub fn derivative<F>(f: F, x: f64) -> NumericResult<f64>
where
    F: Fn(Dual) -> NumericResult<Dual>,
{
    Ok(f(Dual::variable(x))?.dual)
}

/// Forward-mode value and derivative `(f(x), f'(x))` in one sweep.
///
/// # Errors
/// Propagates any error returned by `f`.
pub fn value_and_derivative<F>(f: F, x: f64) -> NumericResult<(f64, f64)>
where
    F: Fn(Dual) -> NumericResult<Dual>,
{
    let r = f(Dual::variable(x))?;
    Ok((r.real, r.dual))
}

/// Forward-mode gradient `∇f` of `f: &[Dual] → Dual` by `n` directional sweeps.
///
/// Sweep `j` seeds coordinate `j` with derivative `1` and the rest with `0`,
/// recovering `∂f/∂x_j`.
///
/// # Errors
/// [`NumericError::EmptyInput`] if `x` is empty, or any error returned by `f`.
pub fn gradient<F>(f: F, x: &[f64]) -> NumericResult<Vec<f64>>
where
    F: Fn(&[Dual]) -> NumericResult<Dual>,
{
    let n = x.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut grad = vec![0.0_f64; n];
    let mut input: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
    for j in 0..n {
        input[j].dual = 1.0;
        grad[j] = f(&input)?.dual;
        input[j].dual = 0.0;
    }
    Ok(grad)
}

/// Forward-mode value and gradient `(f(x), ∇f(x))`.
///
/// # Errors
/// [`NumericError::EmptyInput`] if `x` is empty, or any error returned by `f`.
pub fn value_and_gradient<F>(f: F, x: &[f64]) -> NumericResult<(f64, Vec<f64>)>
where
    F: Fn(&[Dual]) -> NumericResult<Dual>,
{
    let n = x.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut grad = vec![0.0_f64; n];
    let mut input: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();
    let mut value = 0.0_f64;
    for j in 0..n {
        input[j].dual = 1.0;
        let r = f(&input)?;
        if j == 0 {
            value = r.real;
        }
        grad[j] = r.dual;
        input[j].dual = 0.0;
    }
    Ok((value, grad))
}

/// Forward-mode Jacobian of `f: &[Dual] → Vec<Dual>` (`m` outputs, `n` inputs).
///
/// Performs `n` sweeps (one per input column); returns a row-major `m × n`
/// matrix where `jac[i * n + j] = ∂f_i/∂x_j`.
///
/// # Errors
/// [`NumericError::EmptyInput`] if `x` is empty,
/// [`NumericError::ShapeMismatch`] if `f` returns a different output length
/// across sweeps, or any error returned by `f`.
pub fn jacobian<F>(f: F, x: &[f64]) -> NumericResult<(usize, usize, Vec<f64>)>
where
    F: Fn(&[Dual]) -> NumericResult<Vec<Dual>>,
{
    let n = x.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut input: Vec<Dual> = x.iter().map(|&v| Dual::constant(v)).collect();

    // First sweep establishes the output dimension `m`.
    input[0].dual = 1.0;
    let first = f(&input)?;
    input[0].dual = 0.0;
    let m = first.len();
    if m == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut jac = vec![0.0_f64; m * n];
    for i in 0..m {
        jac[i * n] = first[i].dual;
    }

    for j in 1..n {
        input[j].dual = 1.0;
        let col = f(&input)?;
        input[j].dual = 0.0;
        if col.len() != m {
            return Err(NumericError::ShapeMismatch {
                expected: vec![m],
                got: vec![col.len()],
            });
        }
        for i in 0..m {
            jac[i * n + j] = col[i].dual;
        }
    }
    Ok((m, n, jac))
}

/// Jacobian-vector product `J(x) · v` in a **single** forward sweep.
///
/// Seeds each input `x_k` with direction `v_k`; by linearity of the dual
/// propagation, the derivative component of output `i` equals
/// `Σ_k (∂f_i/∂x_k) v_k = (J v)_i`. This is the cheap, matrix-free directional
/// derivative that makes forward mode attractive for tall Jacobians.
///
/// # Errors
/// [`NumericError::EmptyInput`] if `x` is empty,
/// [`NumericError::DimensionMismatch`] if `x` and `v` differ in length, or any
/// error returned by `f`.
pub fn jacobian_vector_product<F>(f: F, x: &[f64], v: &[f64]) -> NumericResult<Vec<f64>>
where
    F: Fn(&[Dual]) -> NumericResult<Vec<Dual>>,
{
    if x.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    if x.len() != v.len() {
        return Err(NumericError::DimensionMismatch {
            a: x.len(),
            b: v.len(),
        });
    }
    let input: Vec<Dual> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Dual::seeded(xi, vi))
        .collect();
    let out = f(&input)?;
    Ok(out.iter().map(|d| d.dual).collect())
}

/// Directional derivative of a scalar field `f: &[Dual] → Dual` along `v`.
///
/// Equivalent to `∇f · v` computed in a single sweep.
///
/// # Errors
/// [`NumericError::EmptyInput`] if `x` is empty,
/// [`NumericError::DimensionMismatch`] if `x` and `v` differ in length, or any
/// error returned by `f`.
pub fn directional_derivative<F>(f: F, x: &[f64], v: &[f64]) -> NumericResult<f64>
where
    F: Fn(&[Dual]) -> NumericResult<Dual>,
{
    if x.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    if x.len() != v.len() {
        return Err(NumericError::DimensionMismatch {
            a: x.len(),
            b: v.len(),
        });
    }
    let input: Vec<Dual> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| Dual::seeded(xi, vi))
        .collect();
    Ok(f(&input)?.dual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::central_difference::central_difference;
    use crate::diff::complex_step::{CDual, complex_step_derivative};

    const TOL: f64 = 1.0e-12;

    #[test]
    fn arithmetic_operators_and_assign() {
        let x = Dual::variable(3.0); // x, dx=1
        let y = Dual::constant(2.0);
        // (x + y) → value 5, deriv 1
        let a = x + y;
        assert!((a.real - 5.0).abs() < TOL && (a.dual - 1.0).abs() < TOL);
        // (x - y) → 1, deriv 1
        let b = x - y;
        assert!((b.real - 1.0).abs() < TOL && (b.dual - 1.0).abs() < TOL);
        // (x * y) → 6, deriv = y = 2
        let c = x * y;
        assert!((c.real - 6.0).abs() < TOL && (c.dual - 2.0).abs() < TOL);
        // (x / y) → 1.5, deriv = 1/y = 0.5
        let d = x / y;
        assert!((d.real - 1.5).abs() < TOL && (d.dual - 0.5).abs() < TOL);
        // -x → -3, deriv -1
        let e = -x;
        assert!((e.real + 3.0).abs() < TOL && (e.dual + 1.0).abs() < TOL);
        // assign variants
        let mut m = x;
        m += y;
        m -= Dual::constant(1.0);
        m *= Dual::constant(2.0);
        m /= Dual::constant(4.0);
        // ((3+2-1)*2)/4 = 2 ; derivative: x had dx=1, the +/- consts don't change
        // it, *2 → 2, /4 → 0.5
        assert!((m.real - 2.0).abs() < TOL, "m.real={}", m.real);
        assert!((m.dual - 0.5).abs() < TOL, "m.dual={}", m.dual);
        // scalar mixed operators
        let s = 2.0 * x + 1.0;
        assert!((s.real - 7.0).abs() < TOL && (s.dual - 2.0).abs() < TOL);
    }

    #[test]
    fn product_and_quotient_rule_match_analytic() {
        // f(x) = (x² + 1)(x − 3); f'(x) = 2x(x−3) + (x²+1) = 3x² − 6x + 1.
        let f = |x: Dual| -> NumericResult<Dual> { Ok((x.powi(2) + 1.0) * (x - 3.0)) };
        for &xv in &[-2.0, 0.5, 1.0, 4.0] {
            let d = derivative(f, xv).expect("ok");
            let want = 3.0 * xv * xv - 6.0 * xv + 1.0;
            assert!((d - want).abs() < 1.0e-10, "x={xv}: {d} vs {want}");
        }
        // g(x) = (x − 1)/(x² + 2); quotient rule.
        let g = |x: Dual| -> NumericResult<Dual> { Ok((x - 1.0) / (x.powi(2) + 2.0)) };
        for &xv in &[-1.0, 0.0, 2.5] {
            let d = derivative(g, xv).expect("ok");
            let num = xv * xv + 2.0;
            let want = (num - (xv - 1.0) * 2.0 * xv) / (num * num);
            assert!((d - want).abs() < 1.0e-10, "x={xv}: {d} vs {want}");
        }
    }

    #[test]
    fn elementary_functions_against_analytic_derivatives() {
        // (function, point, analytic derivative)
        let checks: Vec<(&str, f64, f64, f64)> = vec![
            ("exp", 0.7, 0.7_f64.exp(), 0.7_f64.exp()),
            ("sin", 1.1, 1.1_f64.sin(), 1.1_f64.cos()),
            ("cos", 1.1, 1.1_f64.cos(), -1.1_f64.sin()),
            ("sinh", 0.4, 0.4_f64.sinh(), 0.4_f64.cosh()),
            ("cosh", 0.4, 0.4_f64.cosh(), 0.4_f64.sinh()),
            ("tanh", 0.9, 0.9_f64.tanh(), 1.0 - 0.9_f64.tanh().powi(2)),
            ("atan", 0.3, 0.3_f64.atan(), 1.0 / (1.0 + 0.09)),
            ("abs", -2.0, 2.0, -1.0),
        ];
        for (name, x, fv, dv) in checks {
            let z = Dual::variable(x);
            let r = match name {
                "exp" => z.exp(),
                "sin" => z.sin(),
                "cos" => z.cos(),
                "sinh" => z.sinh(),
                "cosh" => z.cosh(),
                "tanh" => z.tanh(),
                "atan" => z.atan(),
                "abs" => z.abs(),
                _ => unreachable!(),
            };
            assert!((r.real - fv).abs() < 1.0e-12, "{name} value");
            assert!(
                (r.dual - dv).abs() < 1.0e-12,
                "{name} deriv: {} vs {dv}",
                r.dual
            );
        }
    }

    #[test]
    fn fallible_elementary_functions() {
        let x = Dual::variable(2.0);
        // ln 2 → value ln2, deriv 1/2
        let l = x.ln().expect("ln");
        assert!((l.real - 2.0_f64.ln()).abs() < TOL && (l.dual - 0.5).abs() < TOL);
        // sqrt 2 → deriv 1/(2√2)
        let s = x.sqrt().expect("sqrt");
        assert!((s.dual - 1.0 / (2.0 * 2.0_f64.sqrt())).abs() < TOL);
        // recip 2 → 0.5, deriv -1/4
        let r = x.recip().expect("recip");
        assert!((r.real - 0.5).abs() < TOL && (r.dual + 0.25).abs() < TOL);
        // tan 0.5 → deriv sec² = 1 + tan²
        let t = Dual::variable(0.5).tan();
        let tt = 0.5_f64.tan();
        assert!((t.dual - (1.0 + tt * tt)).abs() < TOL);
        // powf: x^2.5 → deriv 2.5 x^1.5
        let p = x.powf(2.5).expect("powf");
        assert!((p.dual - 2.5 * 2.0_f64.powf(1.5)).abs() < 1.0e-10);
        // asin / acos at 0.5
        let half = Dual::variable(0.5);
        let asn = half.asin().expect("asin");
        assert!((asn.dual - 1.0 / (1.0_f64 - 0.25).sqrt()).abs() < TOL);
        let acs = half.acos().expect("acos");
        assert!((acs.dual + 1.0 / (1.0_f64 - 0.25).sqrt()).abs() < TOL);
    }

    #[test]
    fn domain_errors_are_reported() {
        assert!(matches!(
            Dual::variable(-1.0).ln(),
            Err(NumericError::OutOfDomain { .. })
        ));
        assert!(matches!(
            Dual::variable(-1.0).sqrt(),
            Err(NumericError::OutOfDomain { .. })
        ));
        assert!(matches!(
            Dual::variable(0.0).recip(),
            Err(NumericError::OutOfDomain { .. })
        ));
        assert!(matches!(
            Dual::variable(2.0).asin(),
            Err(NumericError::OutOfDomain { .. })
        ));
        assert!(matches!(
            Dual::variable(-2.0).acos(),
            Err(NumericError::OutOfDomain { .. })
        ));
        // powf negative base, non-integer exponent
        assert!(matches!(
            Dual::variable(-1.5).powf(0.5),
            Err(NumericError::OutOfDomain { .. })
        ));
        // powf zero base with exponent < 1 and a live derivative seed is singular.
        assert!(matches!(
            Dual::variable(0.0).powf(0.5),
            Err(NumericError::OutOfDomain { .. })
        ));
    }

    #[test]
    fn composition_safe_nested_functions() {
        // h(x) = sin(exp(x²)); h'(x) = cos(exp(x²)) · exp(x²) · 2x.
        let h = |x: Dual| -> NumericResult<Dual> { Ok(x.powi(2).exp().sin()) };
        for &xv in &[-0.6, 0.2, 0.9] {
            let d = derivative(h, xv).expect("ok");
            let e = (xv * xv).exp();
            let want = e.cos() * e * 2.0 * xv;
            assert!((d - want).abs() < 1.0e-9, "x={xv}: {d} vs {want}");
        }
        // Deeper nest with a fallible inner: k(x) = ln(1 + tanh(x)²).
        let k = |x: Dual| -> NumericResult<Dual> {
            let t = x.tanh();
            (Dual::constant(1.0) + t * t).ln()
        };
        for &xv in &[-1.0, 0.3, 1.4] {
            let d = derivative(k, xv).expect("ok");
            let t = xv.tanh();
            let dt = 1.0 - t * t;
            let want = (2.0 * t * dt) / (1.0 + t * t);
            assert!((d - want).abs() < 1.0e-10, "x={xv}: {d} vs {want}");
        }
    }

    #[test]
    fn agrees_with_central_difference_and_complex_step() {
        // f(x) = x·sin(x) + e^{x/2}.
        let f_dual = |x: Dual| -> NumericResult<Dual> { Ok(x * x.sin() + (x / 2.0).exp()) };
        let f_real = |x: f64| -> NumericResult<f64> { Ok(x * x.sin() + (x / 2.0).exp()) };
        let f_cdual = |z: CDual| -> NumericResult<CDual> {
            Ok(z.mul(z.sin()).add((z.mul(CDual::new(0.5, 0.0))).exp()))
        };
        for &xv in &[0.3, 1.0, 2.2] {
            let ad = derivative(f_dual, xv).expect("ad");
            let cd = central_difference(f_real, xv, 1.0e-5).expect("cd");
            let cs = complex_step_derivative(f_cdual, xv, 1.0e-30).expect("cs");
            assert!((ad - cd).abs() < 1.0e-6, "central: {ad} vs {cd}");
            assert!((ad - cs).abs() < 1.0e-12, "cstep: {ad} vs {cs}");
        }
    }

    #[test]
    fn gradient_of_scalar_field() {
        // f(x,y,z) = x² y + y sin z; ∇f = (2xy, x² + sin z, y cos z).
        let f = |v: &[Dual]| -> NumericResult<Dual> { Ok(v[0].powi(2) * v[1] + v[1] * v[2].sin()) };
        let point = [1.5, -2.0, 0.7];
        let g = gradient(f, &point).expect("grad");
        let want = [
            2.0 * point[0] * point[1],
            point[0] * point[0] + point[2].sin(),
            point[1] * point[2].cos(),
        ];
        for i in 0..3 {
            assert!(
                (g[i] - want[i]).abs() < 1.0e-10,
                "grad[{i}]: {} vs {}",
                g[i],
                want[i]
            );
        }
        // value_and_gradient consistency.
        let (val, g2) = value_and_gradient(f, &point).expect("vg");
        let wval = point[0] * point[0] * point[1] + point[1] * point[2].sin();
        assert!((val - wval).abs() < 1.0e-12);
        for i in 0..3 {
            assert!((g2[i] - g[i]).abs() < TOL);
        }
    }

    #[test]
    fn jacobian_of_vector_field() {
        // f(x,y) = [x² + y, x·y, sin x]; J = [[2x, 1], [y, x], [cos x, 0]].
        let f = |v: &[Dual]| -> NumericResult<Vec<Dual>> {
            Ok(vec![v[0].powi(2) + v[1], v[0] * v[1], v[0].sin()])
        };
        let point = [0.8, 1.3];
        let (m, n, jac) = jacobian(f, &point).expect("jac");
        assert_eq!((m, n), (3, 2));
        let want = [
            [2.0 * point[0], 1.0],
            [point[1], point[0]],
            [point[0].cos(), 0.0],
        ];
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (jac[i * n + j] - want[i][j]).abs() < 1.0e-10,
                    "J[{i}][{j}]: {} vs {}",
                    jac[i * n + j],
                    want[i][j]
                );
            }
        }
    }

    #[test]
    fn jacobian_vector_product_single_sweep() {
        // Same f as above; JVP must equal J·v.
        let f = |v: &[Dual]| -> NumericResult<Vec<Dual>> {
            Ok(vec![v[0].powi(2) + v[1], v[0] * v[1], v[0].sin()])
        };
        let point = [0.8, 1.3];
        let dir = [2.0, -1.0];
        let jv = jacobian_vector_product(f, &point, &dir).expect("jvp");
        // Compute J·v from the analytic Jacobian.
        let jmat = [
            [2.0 * point[0], 1.0],
            [point[1], point[0]],
            [point[0].cos(), 0.0],
        ];
        for i in 0..3 {
            let want = jmat[i][0] * dir[0] + jmat[i][1] * dir[1];
            assert!(
                (jv[i] - want).abs() < 1.0e-10,
                "JV[{i}]: {} vs {want}",
                jv[i]
            );
        }
        // Directional derivative of a scalar field along v == ∇f·v.
        let g = |v: &[Dual]| -> NumericResult<Dual> { Ok(v[0].powi(2) + v[1]) };
        let dd = directional_derivative(g, &point, &dir).expect("dd");
        let grad = [2.0 * point[0], 1.0];
        assert!((dd - (grad[0] * dir[0] + grad[1] * dir[1])).abs() < 1.0e-10);
    }

    #[test]
    fn powi_recip_abs_special_values() {
        // powi with negative exponent: x^{-2} at x=2 → 0.25, deriv -2·2^{-3}=-0.25.
        let z = Dual::variable(2.0).powi(-2);
        assert!((z.real - 0.25).abs() < TOL && (z.dual + 0.25).abs() < TOL);
        // powi exponent 0 → constant 1, deriv 0.
        let one = Dual::variable(5.0).powi(0);
        assert!((one.real - 1.0).abs() < TOL && one.dual.abs() < TOL);
        // sqrt at 0: value 0, derivative ∞ for unit seed (surfaced, not panicked).
        let s0 = Dual::variable(0.0).sqrt().expect("sqrt0");
        assert_eq!(s0.real, 0.0);
        assert!(s0.dual.is_infinite());
        // abs at 0 → sub-gradient 0.
        let a0 = Dual::variable(0.0).abs();
        assert_eq!(a0.real, 0.0);
        assert_eq!(a0.dual, 0.0);
    }

    #[test]
    fn utility_error_paths() {
        let f = |_v: &[Dual]| -> NumericResult<Dual> { Ok(Dual::constant(0.0)) };
        assert!(matches!(gradient(f, &[]), Err(NumericError::EmptyInput)));
        let fv = |_v: &[Dual]| -> NumericResult<Vec<Dual>> { Ok(vec![Dual::constant(0.0)]) };
        assert!(matches!(jacobian(fv, &[]), Err(NumericError::EmptyInput)));
        assert!(matches!(
            jacobian_vector_product(fv, &[1.0], &[1.0, 2.0]),
            Err(NumericError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            jacobian_vector_product(fv, &[], &[]),
            Err(NumericError::EmptyInput)
        ));
        // Propagated function error through gradient.
        let ferr = |_v: &[Dual]| -> NumericResult<Dual> { Err(NumericError::EmptyInput) };
        assert!(matches!(
            gradient(ferr, &[1.0]),
            Err(NumericError::EmptyInput)
        ));
    }
}
