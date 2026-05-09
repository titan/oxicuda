//! Forward-mode automatic differentiation via dual numbers.

/// A dual number `a + b·ε` where `ε² = 0`.
///
/// Used for forward-mode AD: `value` carries the primal, `dvalue` carries the
/// directional derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Primal value.
    pub value: f32,
    /// Derivative (tangent) component.
    pub dvalue: f32,
}

impl Dual {
    /// Create a dual number representing an input variable at `v` (derivative = 1).
    #[must_use]
    pub fn variable(v: f32) -> Self {
        Dual {
            value: v,
            dvalue: 1.0,
        }
    }

    /// Create a dual number representing a constant (derivative = 0).
    #[must_use]
    pub fn constant(v: f32) -> Self {
        Dual {
            value: v,
            dvalue: 0.0,
        }
    }

    /// `sin(a + b·ε) = sin(a) + b·cos(a)·ε`
    #[must_use]
    pub fn sin(self) -> Self {
        Dual {
            value: self.value.sin(),
            dvalue: self.dvalue * self.value.cos(),
        }
    }

    /// `cos(a + b·ε) = cos(a) - b·sin(a)·ε`
    #[must_use]
    pub fn cos(self) -> Self {
        Dual {
            value: self.value.cos(),
            dvalue: -self.dvalue * self.value.sin(),
        }
    }

    /// `exp(a + b·ε) = exp(a) + b·exp(a)·ε`
    #[must_use]
    pub fn exp(self) -> Self {
        let ev = self.value.exp();
        Dual {
            value: ev,
            dvalue: self.dvalue * ev,
        }
    }

    /// `ln(a + b·ε) = ln(a) + b/a·ε`
    #[must_use]
    pub fn ln(self) -> Self {
        Dual {
            value: self.value.ln(),
            dvalue: self.dvalue / self.value,
        }
    }

    /// `sqrt(a + b·ε) = sqrt(a) + b/(2·sqrt(a))·ε`
    #[must_use]
    pub fn sqrt(self) -> Self {
        let sv = self.value.sqrt();
        Dual {
            value: sv,
            dvalue: self.dvalue / (2.0 * sv),
        }
    }

    /// `tanh(a + b·ε) = tanh(a) + b·(1 - tanh²(a))·ε`
    #[must_use]
    pub fn tanh(self) -> Self {
        let tv = self.value.tanh();
        Dual {
            value: tv,
            dvalue: self.dvalue * (1.0 - tv * tv),
        }
    }

    /// `a^n` via power rule: `n * a^(n-1) · ε`
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        Dual {
            value: self.value.powi(n),
            dvalue: self.dvalue * n as f32 * self.value.powi(n - 1),
        }
    }

    /// `|a + b·ε| = |a| + b·sign(a)·ε`
    #[must_use]
    pub fn abs(self) -> Self {
        Dual {
            value: self.value.abs(),
            dvalue: self.dvalue * self.value.signum(),
        }
    }

    /// `relu(a) = max(0, a)` derivative = 1 if a > 0 else 0.
    #[must_use]
    pub fn relu(self) -> Self {
        if self.value > 0.0 {
            self
        } else {
            Dual {
                value: 0.0,
                dvalue: 0.0,
            }
        }
    }

    /// GeLU approximation via `tanh`.
    #[must_use]
    pub fn gelu(self) -> Self {
        let sqrt_2_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
        let inner = sqrt_2_pi * (self.value + 0.044715 * self.value.powi(3));
        let t = inner.tanh();
        let gelu_v = 0.5 * self.value * (1.0 + t);
        // d/dx gelu = 0.5*(1+tanh) + 0.5*x*(1-tanh^2)*d(inner)/dx
        let d_inner_dx = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * self.value * self.value);
        let d_gelu_dx = 0.5 * (1.0 + t) + 0.5 * self.value * (1.0 - t * t) * d_inner_dx;
        Dual {
            value: gelu_v,
            dvalue: self.dvalue * d_gelu_dx,
        }
    }
}

impl std::ops::Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Dual {
            value: self.value + rhs.value,
            dvalue: self.dvalue + rhs.dvalue,
        }
    }
}

impl std::ops::Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Dual {
            value: self.value - rhs.value,
            dvalue: self.dvalue - rhs.dvalue,
        }
    }
}

impl std::ops::Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Dual {
            value: self.value * rhs.value,
            dvalue: self.value * rhs.dvalue + self.dvalue * rhs.value,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let v = self.value / rhs.value;
        // (a/b)' = (a'b - ab') / b^2
        let d = (self.dvalue * rhs.value - self.value * rhs.dvalue) / (rhs.value * rhs.value);
        Dual {
            value: v,
            dvalue: d,
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Dual {
            value: -self.value,
            dvalue: -self.dvalue,
        }
    }
}

impl std::ops::Add<f32> for Dual {
    type Output = Self;
    fn add(self, rhs: f32) -> Self {
        Dual {
            value: self.value + rhs,
            dvalue: self.dvalue,
        }
    }
}

impl std::ops::Mul<f32> for Dual {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Dual {
            value: self.value * rhs,
            dvalue: self.dvalue * rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn dual_variable_dvalue_one() {
        let x = Dual::variable(3.0);
        assert_eq!(x.value, 3.0);
        assert_eq!(x.dvalue, 1.0);
    }

    #[test]
    fn dual_constant_dvalue_zero() {
        let c = Dual::constant(5.0);
        assert_eq!(c.value, 5.0);
        assert_eq!(c.dvalue, 0.0);
    }

    #[test]
    fn dual_x_squared_derivative() {
        // f(x) = x^2 → f'(x) = 2x
        let x = Dual::variable(3.0);
        let f = x * x;
        assert!((f.value - 9.0).abs() < EPS);
        assert!((f.dvalue - 6.0).abs() < EPS);
    }

    #[test]
    fn dual_powi_matches_mul() {
        let x = Dual::variable(2.5);
        let f_mul = x * x;
        let f_pow = x.powi(2);
        assert!((f_mul.value - f_pow.value).abs() < EPS);
        assert!((f_mul.dvalue - f_pow.dvalue).abs() < EPS);
    }

    #[test]
    fn dual_sin_x_squared_chain_rule() {
        // f(x) = sin(x^2), f'(x) = cos(x^2) * 2x
        let x_val = 2.0_f32;
        let x = Dual::variable(x_val);
        let f = (x * x).sin();
        let expected_d = (x_val * x_val).cos() * 2.0 * x_val;
        assert!((f.dvalue - expected_d).abs() < 1e-4);
    }

    #[test]
    fn dual_exp_derivative() {
        // f(x) = exp(x), f'(x) = exp(x)
        let x = Dual::variable(1.5);
        let f = x.exp();
        assert!((f.value - 1.5_f32.exp()).abs() < EPS);
        assert!((f.dvalue - 1.5_f32.exp()).abs() < 1e-4);
    }

    #[test]
    fn dual_cos_derivative() {
        // f(x) = cos(x), f'(x) = -sin(x)
        let x_val = 1.0_f32;
        let x = Dual::variable(x_val);
        let f = x.cos();
        assert!((f.value - x_val.cos()).abs() < EPS);
        assert!((f.dvalue - (-x_val.sin())).abs() < 1e-4);
    }

    #[test]
    fn dual_tanh_derivative() {
        let x_val = 0.5_f32;
        let x = Dual::variable(x_val);
        let f = x.tanh();
        let t = x_val.tanh();
        assert!((f.value - t).abs() < EPS);
        assert!((f.dvalue - (1.0 - t * t)).abs() < 1e-4);
    }

    #[test]
    fn dual_ln_derivative() {
        let x_val = 2.0_f32;
        let x = Dual::variable(x_val);
        let f = x.ln();
        assert!((f.value - x_val.ln()).abs() < EPS);
        assert!((f.dvalue - 1.0 / x_val).abs() < EPS);
    }

    #[test]
    fn dual_sqrt_derivative() {
        let x_val = 4.0_f32;
        let x = Dual::variable(x_val);
        let f = x.sqrt();
        assert!((f.value - 2.0).abs() < EPS);
        assert!((f.dvalue - 0.25).abs() < EPS);
    }

    #[test]
    fn dual_div_quotient_rule() {
        // f(x) = x / (x+1), f'(x) = 1/(x+1)^2
        let x_val = 2.0_f32;
        let x = Dual::variable(x_val);
        let one = Dual::constant(1.0);
        let f = x / (x + one);
        let expected_d = 1.0 / (x_val + 1.0).powi(2);
        assert!((f.dvalue - expected_d).abs() < 1e-4);
    }

    #[test]
    fn dual_chain_exp_sin() {
        // f(x) = exp(sin(x)), f'(x) = exp(sin(x)) * cos(x)
        let x_val = 0.7_f32;
        let x = Dual::variable(x_val);
        let f = x.sin().exp();
        let expected_d = x_val.sin().exp() * x_val.cos();
        assert!((f.dvalue - expected_d).abs() < 1e-4);
    }

    #[test]
    fn dual_neg_op() {
        let x = Dual::variable(3.0);
        let f = -x;
        assert_eq!(f.value, -3.0);
        assert_eq!(f.dvalue, -1.0);
    }

    #[test]
    fn dual_abs_positive() {
        let x = Dual::variable(2.0);
        let f = x.abs();
        assert_eq!(f.value, 2.0);
        assert_eq!(f.dvalue, 1.0);
    }

    #[test]
    fn dual_abs_negative() {
        let x = Dual::variable(-3.0);
        let f = x.abs();
        assert_eq!(f.value, 3.0);
        assert_eq!(f.dvalue, -1.0);
    }

    #[test]
    fn dual_add_sub_ops() {
        let a = Dual::variable(2.0);
        let b = Dual::constant(3.0);
        let s = a + b;
        assert_eq!(s.value, 5.0);
        assert_eq!(s.dvalue, 1.0);
        let d = a - b;
        assert_eq!(d.value, -1.0);
        assert_eq!(d.dvalue, 1.0);
    }
}
