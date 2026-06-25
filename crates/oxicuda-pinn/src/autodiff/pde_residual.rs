//! Forward-mode automatic differentiation of PDE residuals w.r.t. inputs.
//!
//! [`crate::autodiff::multidim::MultiDual`] gives *first* partial derivatives in
//! one forward sweep, but PDE residuals routinely need *second* spatial
//! derivatives (`u_xx`, the Laplacian, …). This module supplies a second-order
//! forward-AD number — [`HyperDual`] — that carries a value, an `N`-vector
//! gradient, and the full symmetric `N×N` Hessian simultaneously. It is the
//! natural extension of `MultiDual` to second order (forward-over-forward AD)
//! and computes `u_t`, `u_x`, `u_xx`, `u_xy`, … *exactly* (to machine
//! precision, no finite differences) from a single evaluation of the network /
//! field closure.
//!
//! Convenience residual builders assemble the standard PDE residuals (heat,
//! Poisson, generic 2nd-order, Burgers) directly from the AD output, closing
//! the loop "differentiate the PDE residual w.r.t. its inputs via forward AD"
//! that previously only the reverse-mode [`crate::autodiff::tape::Tape`]
//! supported.

use crate::error::{PinnError, PinnResult};

/// Second-order forward-AD number tracking value, gradient, and Hessian over
/// `N` independent inputs.
///
/// For a scalar field `u(x_0, …, x_{N-1})`, a `HyperDual<N>` holds
/// `value = u`, `grad[i] = ∂u/∂x_i`, and `hess[i*N + j] = ∂²u/∂x_i∂x_j`.
/// All elementary operations propagate value, gradient, and Hessian by the
/// product / chain rules, so the resulting derivatives are exact.
#[derive(Debug, Clone)]
pub struct HyperDual<const N: usize> {
    /// Primal value.
    pub value: f32,
    /// First partial derivatives `∂/∂x_i`.
    pub grad: [f32; N],
    /// Second partial derivatives `∂²/∂x_i∂x_j`, row-major `N×N`.
    pub hess: Vec<f32>,
}

impl<const N: usize> HyperDual<N> {
    /// The `var_idx`-th input variable at value `v`
    /// (`grad[var_idx] = 1`, Hessian zero).
    #[must_use]
    pub fn variable(value: f32, var_idx: usize) -> Self {
        let mut grad = [0.0_f32; N];
        if var_idx < N {
            grad[var_idx] = 1.0;
        }
        Self {
            value,
            grad,
            hess: vec![0.0_f32; N * N],
        }
    }

    /// A constant (gradient and Hessian zero).
    #[must_use]
    pub fn constant(value: f32) -> Self {
        Self {
            value,
            grad: [0.0_f32; N],
            hess: vec![0.0_f32; N * N],
        }
    }

    /// Apply a unary function with first derivative `d1 = f'(x)` and second
    /// derivative `d2 = f''(x)`, propagating value/gradient/Hessian:
    /// `g_i = d1·u_i`, `H_ij = d1·u_ij + d2·u_i·u_j`.
    fn unary(&self, fv: f32, d1: f32, d2: f32) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, &sg) in grad.iter_mut().zip(self.grad.iter()) {
            *g = d1 * sg;
        }
        let mut hess = vec![0.0_f32; N * N];
        for (i, gi) in self.grad.iter().enumerate() {
            for (j, gj) in self.grad.iter().enumerate() {
                hess[i * N + j] = d1 * self.hess[i * N + j] + d2 * gi * gj;
            }
        }
        Self {
            value: fv,
            grad,
            hess,
        }
    }

    /// Addition.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, (&a, &b)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = a + b;
        }
        let hess: Vec<f32> = self
            .hess
            .iter()
            .zip(other.hess.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Self {
            value: self.value + other.value,
            grad,
            hess,
        }
    }

    /// Subtraction.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, (&a, &b)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = a - b;
        }
        let hess: Vec<f32> = self
            .hess
            .iter()
            .zip(other.hess.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Self {
            value: self.value - other.value,
            grad,
            hess,
        }
    }

    /// Multiplication (full second-order product rule):
    /// `H_ij = a·b_ij + b·a_ij + a_i·b_j + a_j·b_i`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let a = self.value;
        let b = other.value;
        let mut grad = [0.0_f32; N];
        for (g, (&sg, &og)) in grad.iter_mut().zip(self.grad.iter().zip(other.grad.iter())) {
            *g = a * og + b * sg;
        }
        let mut hess = vec![0.0_f32; N * N];
        for (i, row) in hess.chunks_mut(N).enumerate() {
            let sgi = self.grad[i];
            let ogi = other.grad[i];
            let sh = &self.hess[i * N..i * N + N];
            let oh = &other.hess[i * N..i * N + N];
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = a * oh[j] + b * sh[j] + sgi * other.grad[j] + self.grad[j] * ogi;
            }
        }
        Self {
            value: a * b,
            grad,
            hess,
        }
    }

    /// Scale by a constant scalar.
    #[must_use]
    pub fn scale(&self, s: f32) -> Self {
        let mut grad = [0.0_f32; N];
        for (g, &sg) in grad.iter_mut().zip(self.grad.iter()) {
            *g = s * sg;
        }
        let hess: Vec<f32> = self.hess.iter().map(|&h| s * h).collect();
        Self {
            value: s * self.value,
            grad,
            hess,
        }
    }

    /// Add a constant scalar.
    #[must_use]
    pub fn add_scalar(&self, s: f32) -> Self {
        Self {
            value: self.value + s,
            grad: self.grad,
            hess: self.hess.clone(),
        }
    }

    /// `sin` with exact first/second derivatives.
    #[must_use]
    pub fn sin(&self) -> Self {
        let s = self.value.sin();
        let c = self.value.cos();
        self.unary(s, c, -s)
    }

    /// `cos`.
    #[must_use]
    pub fn cos(&self) -> Self {
        let c = self.value.cos();
        let s = self.value.sin();
        self.unary(c, -s, -c)
    }

    /// `exp`.
    #[must_use]
    pub fn exp(&self) -> Self {
        let e = self.value.exp();
        self.unary(e, e, e)
    }

    /// `tanh` with `f' = 1 - t²`, `f'' = -2t(1 - t²)`.
    #[must_use]
    pub fn tanh(&self) -> Self {
        let t = self.value.tanh();
        let d1 = 1.0 - t * t;
        let d2 = -2.0 * t * d1;
        self.unary(t, d1, d2)
    }

    /// Integer power `x^n` (n >= 0).
    #[must_use]
    pub fn powi(&self, n: i32) -> Self {
        let v = self.value;
        let fv = v.powi(n);
        let d1 = n as f32 * v.powi(n - 1);
        let d2 = n as f32 * (n - 1) as f32 * v.powi(n - 2);
        self.unary(fv, d1, d2)
    }

    /// Read `∂/∂x_i`.
    #[must_use]
    pub fn d(&self, i: usize) -> f32 {
        if i < N { self.grad[i] } else { 0.0 }
    }

    /// Read `∂²/∂x_i∂x_j`.
    #[must_use]
    pub fn d2(&self, i: usize, j: usize) -> f32 {
        if i < N && j < N {
            self.hess[i * N + j]
        } else {
            0.0
        }
    }
}

// ─── PDE residual builders ─────────────────────────────────────────────────────

/// Heat-equation residual `R = u_t - α u_xx` from a forward-AD field value.
///
/// Convention: coordinate index `0 = x`, `1 = t`. The closure `field` receives
/// `[HyperDual x, HyperDual t]` and returns `u` as a `HyperDual<2>`.
pub fn heat_residual_ad<F>(field: F, x: f32, t: f32, alpha: f32) -> PinnResult<f32>
where
    F: Fn(&[HyperDual<2>]) -> HyperDual<2>,
{
    if alpha <= 0.0 {
        return Err(PinnError::InvalidPdeCoefficient {
            name: "alpha",
            value: alpha,
        });
    }
    let xx = HyperDual::<2>::variable(x, 0);
    let tt = HyperDual::<2>::variable(t, 1);
    let u = field(&[xx, tt]);
    let u_t = u.d(1);
    let u_xx = u.d2(0, 0);
    let r = u_t - alpha * u_xx;
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "heat_residual_ad",
        });
    }
    Ok(r)
}

/// 2D Poisson residual `R = u_xx + u_yy - f` (Laplacian minus source).
///
/// Coordinate indices `0 = x`, `1 = y`.
pub fn poisson_residual_ad<F>(field: F, x: f32, y: f32, f_source: f32) -> PinnResult<f32>
where
    F: Fn(&[HyperDual<2>]) -> HyperDual<2>,
{
    let xx = HyperDual::<2>::variable(x, 0);
    let yy = HyperDual::<2>::variable(y, 1);
    let u = field(&[xx, yy]);
    let lap = u.d2(0, 0) + u.d2(1, 1);
    let r = lap - f_source;
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "poisson_residual_ad",
        });
    }
    Ok(r)
}

/// Viscous Burgers residual `R = u_t + u·u_x - ν u_xx`.
///
/// Coordinate indices `0 = x`, `1 = t`.
pub fn burgers_residual_ad<F>(field: F, x: f32, t: f32, nu: f32) -> PinnResult<f32>
where
    F: Fn(&[HyperDual<2>]) -> HyperDual<2>,
{
    if nu < 0.0 {
        return Err(PinnError::InvalidPdeCoefficient {
            name: "nu",
            value: nu,
        });
    }
    let xx = HyperDual::<2>::variable(x, 0);
    let tt = HyperDual::<2>::variable(t, 1);
    let u = field(&[xx, tt]);
    let r = u.d(1) + u.value * u.d(0) - nu * u.d2(0, 0);
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "burgers_residual_ad",
        });
    }
    Ok(r)
}

/// Generic linear 2nd-order residual
/// `R = a·u_xx + b·u_yy + c·u_x + d·u_y + e·u - f`, coords `0 = x`, `1 = y`.
#[allow(clippy::too_many_arguments)]
pub fn linear_2nd_order_residual_ad<F>(
    field: F,
    x: f32,
    y: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f_source: f32,
) -> PinnResult<f32>
where
    F: Fn(&[HyperDual<2>]) -> HyperDual<2>,
{
    let xx = HyperDual::<2>::variable(x, 0);
    let yy = HyperDual::<2>::variable(y, 1);
    let u = field(&[xx, yy]);
    let r = a * u.d2(0, 0) + b * u.d2(1, 1) + c * u.d(0) + d * u.d(1) + e * u.value - f_source;
    if !r.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "linear_2nd_order_residual_ad",
        });
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn hyperdual_quadratic_second_derivative() {
        // f(x) = x^3; f' = 3x^2, f'' = 6x at x = 2 => 12, 12.
        let x = HyperDual::<1>::variable(2.0, 0);
        let f = x.powi(3);
        assert!((f.value - 8.0).abs() < 1e-4);
        assert!((f.d(0) - 12.0).abs() < 1e-3, "f' = {}", f.d(0));
        assert!((f.d2(0, 0) - 12.0).abs() < 1e-3, "f'' = {}", f.d2(0, 0));
    }

    #[test]
    fn hyperdual_sin_second_derivative() {
        // f(x) = sin(x); f'' = -sin(x).
        let x_val = 0.7_f32;
        let x = HyperDual::<1>::variable(x_val, 0);
        let f = x.sin();
        assert!((f.d2(0, 0) + x_val.sin()).abs() < 1e-4);
        assert!((f.d(0) - x_val.cos()).abs() < 1e-4);
    }

    #[test]
    fn hyperdual_mixed_partial() {
        // f(x, y) = x^2 * y; f_xy = 2x at (3, 4) => 6.
        let x = HyperDual::<2>::variable(3.0, 0);
        let y = HyperDual::<2>::variable(4.0, 1);
        let f = x.powi(2).mul(&y);
        assert!((f.d2(0, 1) - 6.0).abs() < 1e-3, "f_xy = {}", f.d2(0, 1));
        assert!((f.d2(1, 0) - 6.0).abs() < 1e-3, "symmetry");
        // f_xx = 2y = 8.
        assert!((f.d2(0, 0) - 8.0).abs() < 1e-3, "f_xx = {}", f.d2(0, 0));
    }

    #[test]
    fn hyperdual_product_rule_value_grad() {
        // f(x, y) = x*y; ∂f/∂x = y, ∂f/∂y = x.
        let x = HyperDual::<2>::variable(2.0, 0);
        let y = HyperDual::<2>::variable(5.0, 1);
        let f = x.mul(&y);
        assert!((f.value - 10.0).abs() < 1e-5);
        assert!((f.d(0) - 5.0).abs() < 1e-5);
        assert!((f.d(1) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn heat_residual_ad_on_analytic_solution_is_zero() {
        // u(x,t) = sin(pi x) exp(-alpha pi^2 t) solves the heat equation =>
        // residual must vanish.
        let alpha = 0.3_f32;
        let field = |vars: &[HyperDual<2>]| -> HyperDual<2> {
            let x = vars[0].clone();
            let t = vars[1].clone();
            // sin(pi*x)
            let sx = x.scale(PI).sin();
            // exp(-alpha*pi^2*t)
            let et = t.scale(-alpha * PI * PI).exp();
            sx.mul(&et)
        };
        for i in 0..7 {
            for j in 0..5 {
                let x = 0.1 + 0.12 * i as f32;
                let t = 0.05 * j as f32;
                let r = heat_residual_ad(field, x, t, alpha).expect("residual");
                assert!(r.abs() < 1e-3, "heat residual at ({x},{t}) = {r}");
            }
        }
    }

    #[test]
    fn poisson_residual_ad_on_analytic_solution_is_zero() {
        // u = sin(pi x) sin(pi y) solves ∇²u = -2 pi^2 sin(pi x) sin(pi y).
        let field = |vars: &[HyperDual<2>]| -> HyperDual<2> {
            let x = vars[0].clone();
            let y = vars[1].clone();
            let sx = x.scale(PI).sin();
            let sy = y.scale(PI).sin();
            sx.mul(&sy)
        };
        for i in 1..6 {
            for j in 1..6 {
                let x = 0.15 * i as f32;
                let y = 0.15 * j as f32;
                let f_src = -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
                let r = poisson_residual_ad(field, x, y, f_src).expect("residual");
                assert!(r.abs() < 2e-3, "poisson residual at ({x},{y}) = {r}");
            }
        }
    }

    #[test]
    fn burgers_residual_ad_on_steady_state() {
        // Constant field u = const is trivially steady: u_t = u_x = u_xx = 0.
        let field = |_vars: &[HyperDual<2>]| -> HyperDual<2> { HyperDual::<2>::constant(1.5) };
        let r = burgers_residual_ad(field, 0.3, 0.2, 0.01).expect("residual");
        assert!(r.abs() < 1e-6, "steady burgers residual = {r}");
    }

    #[test]
    fn burgers_residual_ad_nonzero_for_wrong_field() {
        // u = x is not a solution: u_t = 0, u_x = 1, u_xx = 0 => R = u*u_x = x.
        let field = |vars: &[HyperDual<2>]| -> HyperDual<2> { vars[0].clone() };
        let r = burgers_residual_ad(field, 0.7, 0.0, 0.0).expect("residual");
        assert!((r - 0.7).abs() < 1e-4, "expected R = x = 0.7, got {r}");
    }

    #[test]
    fn linear_2nd_order_residual_matches_manual() {
        // u = x^2 + y^2: u_xx = 2, u_yy = 2, u_x = 2x, u_y = 2y, u = x^2+y^2.
        let field = |vars: &[HyperDual<2>]| -> HyperDual<2> {
            let x = vars[0].clone();
            let y = vars[1].clone();
            x.powi(2).add(&y.powi(2))
        };
        let (x, y) = (1.0_f32, 2.0_f32);
        // a=1,b=1,c=0,d=0,e=0 => R = u_xx + u_yy - f = 4 - f.
        let r = linear_2nd_order_residual_ad(field, x, y, 1.0, 1.0, 0.0, 0.0, 0.0, 4.0)
            .expect("residual");
        assert!(r.abs() < 1e-4, "laplacian residual = {r}");
    }

    #[test]
    fn invalid_coefficients_error() {
        let field = |_v: &[HyperDual<2>]| HyperDual::<2>::constant(0.0);
        assert!(heat_residual_ad(field, 0.0, 0.0, -1.0).is_err());
        assert!(burgers_residual_ad(field, 0.0, 0.0, -1.0).is_err());
    }
}
