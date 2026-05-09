//! Multi-dimensional dual numbers for simultaneous partial derivatives.

/// A multi-dimensional dual number that tracks `N` partial derivatives simultaneously.
///
/// `MultiDual<N>` holds one primal `value` and an `N`-element gradient vector.
/// Use `variable(v, i)` to create the i-th input variable (sets `grad[i] = 1`,
/// all others 0), and `constant(v)` for constants (`grad = [0; N]`).
#[derive(Debug, Clone)]
pub struct MultiDual<const N: usize> {
    /// Primal value.
    pub value: f32,
    /// Partial derivatives ∂/∂xᵢ for i ∈ 0..N.
    pub grad: [f32; N],
}

impl<const N: usize> MultiDual<N> {
    /// Create the `var_idx`-th input variable at value `v`.
    /// Sets `grad[var_idx] = 1.0`, all others 0.
    #[must_use]
    pub fn variable(value: f32, var_idx: usize) -> Self {
        let mut grad = [0.0_f32; N];
        if var_idx < N {
            grad[var_idx] = 1.0;
        }
        Self { value, grad }
    }

    /// Create a constant (all partial derivatives = 0).
    #[must_use]
    pub fn constant(value: f32) -> Self {
        Self {
            value,
            grad: [0.0_f32; N],
        }
    }

    /// Apply a unary function: `f(x)` with derivative `df_dx`.
    fn unary(self, fv: f32, df_dx: f32) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, &sg) in grad.iter_mut().zip(self.grad.iter()) {
            *g = df_dx * sg;
        }
        Self { value: fv, grad }
    }

    /// `sin` with chain rule.
    #[must_use]
    pub fn sin(self) -> Self {
        let sv = self.value.sin();
        let cv = self.value.cos();
        self.unary(sv, cv)
    }

    /// `cos` with chain rule.
    #[must_use]
    pub fn cos(self) -> Self {
        let cv = self.value.cos();
        let neg_s = -self.value.sin();
        self.unary(cv, neg_s)
    }

    /// `exp` with chain rule.
    #[must_use]
    pub fn exp(self) -> Self {
        let ev = self.value.exp();
        self.unary(ev, ev)
    }

    /// `ln` with chain rule.
    #[must_use]
    pub fn ln(self) -> Self {
        let lv = self.value.ln();
        let d = 1.0 / self.value;
        self.unary(lv, d)
    }

    /// `tanh` with chain rule.
    #[must_use]
    pub fn tanh(self) -> Self {
        let tv = self.value.tanh();
        let d = 1.0 - tv * tv;
        self.unary(tv, d)
    }

    /// `sqrt` with chain rule.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let sv = self.value.sqrt();
        let d = 0.5 / sv;
        self.unary(sv, d)
    }

    /// `powi` with chain rule.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        let pv = self.value.powi(n);
        let d = n as f32 * self.value.powi(n - 1);
        self.unary(pv, d)
    }

    /// Addition: `∂(a+b)/∂xᵢ = ∂a/∂xᵢ + ∂b/∂xᵢ`.
    #[must_use]
    pub fn dual_add(self, other: Self) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, (a, b)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = a + b;
        }
        Self {
            value: self.value + other.value,
            grad,
        }
    }

    /// Subtraction.
    #[must_use]
    pub fn dual_sub(self, other: Self) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, (a, b)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = a - b;
        }
        Self {
            value: self.value - other.value,
            grad,
        }
    }

    /// Multiplication (product rule): `∂(ab)/∂xᵢ = a·∂b/∂xᵢ + b·∂a/∂xᵢ`.
    #[must_use]
    pub fn dual_mul(self, other: Self) -> Self {
        let a_val = self.value;
        let b_val = other.value;
        let mut grad = [0.0_f32; N];
        for (g, (da, db)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = a_val * db + b_val * da;
        }
        Self {
            value: a_val * b_val,
            grad,
        }
    }

    /// Division (quotient rule).
    #[must_use]
    pub fn dual_div(self, other: Self) -> Self {
        let a_val = self.value;
        let b_val = other.value;
        let inv_b = 1.0 / b_val;
        let mut grad = [0.0_f32; N];
        for (g, (da, db)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = (da * b_val - a_val * db) * inv_b * inv_b;
        }
        Self {
            value: a_val * inv_b,
            grad,
        }
    }

    /// Negation.
    #[must_use]
    pub fn dual_neg(self) -> Self {
        let mut grad = self.grad;
        for g in &mut grad {
            *g = -*g;
        }
        Self {
            value: -self.value,
            grad,
        }
    }

    /// Scale by scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        let mut grad = self.grad;
        for g in &mut grad {
            *g *= s;
        }
        Self {
            value: self.value * s,
            grad,
        }
    }

    /// Add scalar.
    #[must_use]
    pub fn add_scalar(self, s: f32) -> Self {
        Self {
            value: self.value + s,
            grad: self.grad,
        }
    }
}

impl<const N: usize> std::ops::Add for MultiDual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.dual_add(rhs)
    }
}

impl<const N: usize> std::ops::Mul for MultiDual<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.dual_mul(rhs)
    }
}

impl<const N: usize> std::ops::Sub for MultiDual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.dual_sub(rhs)
    }
}

impl<const N: usize> std::ops::Div for MultiDual<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.dual_div(rhs)
    }
}

impl<const N: usize> std::ops::Neg for MultiDual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        self.dual_neg()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multidual_variable_gradient_unit() {
        let x: MultiDual<2> = MultiDual::variable(3.0, 0);
        assert_eq!(x.grad[0], 1.0);
        assert_eq!(x.grad[1], 0.0);
        let y: MultiDual<2> = MultiDual::variable(4.0, 1);
        assert_eq!(y.grad[0], 0.0);
        assert_eq!(y.grad[1], 1.0);
    }

    #[test]
    fn multidual_x_squared_plus_y_squared() {
        // f(x, y) = x^2 + y^2; ∂f/∂x = 2x, ∂f/∂y = 2y
        let x: MultiDual<2> = MultiDual::variable(3.0, 0);
        let y: MultiDual<2> = MultiDual::variable(4.0, 1);
        let x2 = x.clone().powi(2);
        let y2 = y.clone().powi(2);
        let f = x2.dual_add(y2);
        assert!(
            (f.grad[0] - 6.0).abs() < 1e-5,
            "∂f/∂x should be 6, got {}",
            f.grad[0]
        );
        assert!(
            (f.grad[1] - 8.0).abs() < 1e-5,
            "∂f/∂y should be 8, got {}",
            f.grad[1]
        );
    }

    #[test]
    fn multidual_product_partial_derivs() {
        // f(x, y) = x*y; ∂f/∂x = y, ∂f/∂y = x
        let x_val = 2.0_f32;
        let y_val = 5.0_f32;
        let x: MultiDual<2> = MultiDual::variable(x_val, 0);
        let y: MultiDual<2> = MultiDual::variable(y_val, 1);
        let f = x.dual_mul(y);
        assert!((f.grad[0] - y_val).abs() < 1e-5);
        assert!((f.grad[1] - x_val).abs() < 1e-5);
    }

    #[test]
    fn multidual_sin_cos_chain() {
        // f(x) = sin(x^2), ∂f/∂x = cos(x^2)*2x
        let x_val = 1.5_f32;
        let x: MultiDual<1> = MultiDual::variable(x_val, 0);
        let f = x.powi(2).sin();
        let expected = (x_val * x_val).cos() * 2.0 * x_val;
        assert!((f.grad[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn multidual_exp_derivative() {
        let x_val = 1.0_f32;
        let x: MultiDual<1> = MultiDual::variable(x_val, 0);
        let f = x.exp();
        assert!((f.grad[0] - x_val.exp()).abs() < 1e-5);
    }

    #[test]
    fn multidual_constant_zero_grad() {
        let c: MultiDual<3> = MultiDual::constant(7.0);
        assert!(c.grad.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn multidual_division_grad() {
        // f(x, y) = x/y; ∂f/∂x = 1/y, ∂f/∂y = -x/y^2
        let x_val = 6.0_f32;
        let y_val = 3.0_f32;
        let x: MultiDual<2> = MultiDual::variable(x_val, 0);
        let y: MultiDual<2> = MultiDual::variable(y_val, 1);
        let f = x.dual_div(y);
        assert!((f.grad[0] - 1.0 / y_val).abs() < 1e-5);
        assert!((f.grad[1] - (-x_val / y_val.powi(2))).abs() < 1e-5);
    }

    #[test]
    fn multidual_three_vars_gradient() {
        // f(x, y, z) = x*y + z; ∂/∂x = y, ∂/∂y = x, ∂/∂z = 1
        let x: MultiDual<3> = MultiDual::variable(2.0, 0);
        let y: MultiDual<3> = MultiDual::variable(3.0, 1);
        let z: MultiDual<3> = MultiDual::variable(1.0, 2);
        let xy = x.dual_mul(y);
        let f = xy.dual_add(z);
        assert!((f.grad[0] - 3.0).abs() < 1e-5);
        assert!((f.grad[1] - 2.0).abs() < 1e-5);
        assert!((f.grad[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn multidual_tanh_grad() {
        let x_val = 0.8_f32;
        let x: MultiDual<1> = MultiDual::variable(x_val, 0);
        let f = x.tanh();
        let t = x_val.tanh();
        assert!((f.grad[0] - (1.0 - t * t)).abs() < 1e-5);
    }
}
