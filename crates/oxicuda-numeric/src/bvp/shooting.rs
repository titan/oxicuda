//! Single-shooting solver for the two-point boundary-value problem
//!
//! ```text
//! y'' = f(x, y, y'),     x ∈ [a, b],     y(a) = α,   y(b) = β.
//! ```
//!
//! The second-order ODE is written as the first-order system
//! `u₀' = u₁`, `u₁' = f(x, u₀, u₁)` with `u(a) = (α, s)`, where the unknown initial
//! slope `s = y'(a)` is chosen so that the **shooting residual**
//!
//! ```text
//! φ(s) = y(b; s) − β
//! ```
//!
//! vanishes — `y(b; s)` being the value at `b` produced by integrating the IVP from
//! the trial slope `s`. The scalar root of `φ` is found with the secant method
//! (two slope guesses, no derivative of `φ` required), which converges
//! super-linearly and is exact-after-two-steps for *linear* problems because `φ`
//! is then affine in `s`. Each residual evaluation integrates the IVP with the
//! crate's classical RK4.
//!
//! Reference: H. B. Keller, *Numerical Methods for Two-Point Boundary-Value
//! Problems*, Blaisdell (1968); U. M. Ascher, R. M. R. Mattheij and R. D. Russell,
//! *Numerical Solution of Boundary Value Problems for Ordinary Differential
//! Equations*, SIAM Classics (1995), §4.

use crate::error::{NumericError, NumericResult};
use crate::ode::rk4::rk4;

/// Configuration for the single-[`shooting`](crate::bvp::shooting) BVP solver.
#[derive(Debug, Clone, Copy)]
pub struct ShootingConfig {
    /// Number of uniform RK4 sub-intervals across `[a, b]` per integration.
    pub n_steps: usize,
    /// Absolute tolerance on the boundary residual `|y(b; s) − β|`.
    pub tol: f64,
    /// Maximum secant iterations on the slope `s`.
    pub max_iter: usize,
    /// First trial slope `s₀`.
    pub slope0: f64,
    /// Second trial slope `s₁` (must differ from `slope0`).
    pub slope1: f64,
}

impl Default for ShootingConfig {
    fn default() -> Self {
        Self {
            n_steps: 200,
            tol: 1.0e-10,
            max_iter: 100,
            slope0: 0.0,
            slope1: 1.0,
        }
    }
}

/// Solution returned by [`solve_shooting`].
#[derive(Debug, Clone)]
pub struct ShootingSolution {
    /// Grid abscissae `x₀ = a, …, x_N = b` (length `n_steps + 1`).
    pub x: Vec<f64>,
    /// Solution values `y(xᵢ)` on the grid.
    pub y: Vec<f64>,
    /// Derivative values `y'(xᵢ)` on the grid.
    pub yp: Vec<f64>,
    /// Converged initial slope `s = y'(a)`.
    pub slope: f64,
    /// Secant iterations performed.
    pub iterations: usize,
    /// Final boundary residual `y(b; s) − β`.
    pub residual: f64,
}

/// Solve `y'' = f(x, y, y')` on `[a, b]` with `y(a) = alpha`, `y(b) = beta` by
/// single shooting with a secant root-find on the initial slope.
///
/// `f` receives `(x, y, y')` and returns `y''`.
///
/// # Errors
/// * [`NumericError::InvalidParameter`] if `b ≤ a`, the endpoints/BCs are
///   non-finite, `n_steps == 0`, or the two trial slopes coincide.
/// * [`NumericError::NumericalInstability`] if the secant step stalls (the
///   residual difference underflows) before reaching the tolerance.
/// * [`NumericError::NotConverged`] if the residual is not driven below `tol`
///   within `max_iter` iterations.
pub fn solve_shooting<F>(
    f: F,
    a: f64,
    b: f64,
    alpha: f64,
    beta: f64,
    config: &ShootingConfig,
) -> NumericResult<ShootingSolution>
where
    F: Fn(f64, f64, f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || b <= a {
        return Err(NumericError::InvalidParameter(
            "shooting: require finite a < b".into(),
        ));
    }
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(NumericError::InvalidParameter(
            "shooting: boundary values must be finite".into(),
        ));
    }
    if config.n_steps == 0 {
        return Err(NumericError::InvalidParameter(
            "shooting: n_steps must be >= 1".into(),
        ));
    }
    if config.slope0 == config.slope1 {
        return Err(NumericError::InvalidParameter(
            "shooting: the two trial slopes must differ".into(),
        ));
    }

    let h = (b - a) / config.n_steps as f64;
    // First-order system u = (y, y'):  u₀' = u₁,  u₁' = f(x, u₀, u₁).
    let rhs = |x: f64, u: &[f64]| -> NumericResult<Vec<f64>> { Ok(vec![u[1], f(x, u[0], u[1])]) };

    // Residual φ(s) = y(b; s) − β, returning the final state for reuse.
    let shoot = |s: f64| -> NumericResult<(f64, Vec<f64>, Vec<Vec<f64>>)> {
        let (xs, us) = rk4(rhs, a, b, &[alpha, s], h)?;
        let last = us.last().ok_or(NumericError::EmptyInput)?;
        let yb = last[0];
        Ok((yb - beta, xs, us))
    };

    // Secant iteration on the slope.
    let mut s_prev = config.slope0;
    let mut s_curr = config.slope1;
    let (mut phi_prev, _, _) = shoot(s_prev)?;
    let (mut phi_curr, mut xs, mut us) = shoot(s_curr)?;

    let mut iterations = 0_usize;
    let mut converged = phi_curr.abs() <= config.tol;
    while !converged && iterations < config.max_iter {
        let denom = phi_curr - phi_prev;
        if denom.abs() < 1.0e-300 {
            return Err(NumericError::NumericalInstability(
                "shooting: secant denominator underflowed (residuals equal)".into(),
            ));
        }
        let s_next = s_curr - phi_curr * (s_curr - s_prev) / denom;
        if !s_next.is_finite() {
            return Err(NumericError::NumericalInstability(
                "shooting: secant produced a non-finite slope".into(),
            ));
        }
        let (phi_next, xn, un) = shoot(s_next)?;
        s_prev = s_curr;
        phi_prev = phi_curr;
        s_curr = s_next;
        phi_curr = phi_next;
        xs = xn;
        us = un;
        iterations += 1;
        converged = phi_curr.abs() <= config.tol;
    }

    if !converged {
        return Err(NumericError::NotConverged {
            iter: iterations,
            residual: phi_curr.abs(),
        });
    }

    let n = xs.len();
    let mut y = Vec::with_capacity(n);
    let mut yp = Vec::with_capacity(n);
    for state in &us {
        y.push(state[0]);
        yp.push(state[1]);
    }

    Ok(ShootingSolution {
        x: xs,
        y,
        yp,
        slope: s_curr,
        iterations,
        residual: phi_curr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_bvp_matches_sinh() {
        // y'' = y, y(0)=0, y(1)=1  ⇒  y = sinh(x)/sinh(1).
        let f = |_x: f64, y: f64, _yp: f64| y;
        let cfg = ShootingConfig::default();
        let sol = solve_shooting(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        let sinh1 = 1.0_f64.sinh();
        let mut max_err = 0.0_f64;
        for (&x, &yi) in sol.x.iter().zip(sol.y.iter()) {
            let exact = x.sinh() / sinh1;
            max_err = max_err.max((yi - exact).abs());
        }
        assert!(max_err < 1.0e-6, "max_err = {max_err:e}");
        // Linear problems: secant is exact after two evaluations.
        assert!(sol.iterations <= 2, "iters = {}", sol.iterations);
    }

    #[test]
    fn boundary_conditions_satisfied_exactly() {
        let f = |_x: f64, y: f64, _yp: f64| y;
        let cfg = ShootingConfig::default();
        let sol = solve_shooting(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        // y(a) = α exactly (initial condition); y(b) = β to tolerance.
        assert!((sol.y[0] - 0.0).abs() < 1.0e-14);
        assert!((sol.y[sol.y.len() - 1] - 1.0).abs() <= cfg.tol);
        assert!(sol.residual.abs() <= cfg.tol);
    }

    #[test]
    fn nonzero_bcs_linear_with_forcing() {
        // y'' = 4y, y(0)=1, y(1)=cosh(2)+? — use exact solution y=cosh(2x).
        // y'' = 4 cosh(2x) = 4 y, y(0)=1, y(1)=cosh(2).
        let f = |_x: f64, y: f64, _yp: f64| 4.0 * y;
        let cfg = ShootingConfig::default();
        let beta = (2.0_f64).cosh();
        let sol = solve_shooting(f, 0.0, 1.0, 1.0, beta, &cfg).expect("ok");
        let mut max_err = 0.0_f64;
        for (&x, &yi) in sol.x.iter().zip(sol.y.iter()) {
            max_err = max_err.max((yi - (2.0 * x).cosh()).abs());
        }
        assert!(max_err < 1.0e-6, "max_err = {max_err:e}");
        // Recovered initial slope y'(0) = 0 for cosh(2x).
        assert!(sol.slope.abs() < 1.0e-6, "slope = {}", sol.slope);
    }

    #[test]
    fn nonlinear_bvp_three_halves_square() {
        // y'' = (3/2) y², y(0)=4, y(1)=1  ⇒  y = 4/(1+x)²  (a classic test).
        let f = |_x: f64, y: f64, _yp: f64| 1.5 * y * y;
        let cfg = ShootingConfig {
            slope0: -10.0,
            slope1: -6.0,
            ..ShootingConfig::default()
        };
        let sol = solve_shooting(f, 0.0, 1.0, 4.0, 1.0, &cfg).expect("ok");
        let mut max_err = 0.0_f64;
        for (&x, &yi) in sol.x.iter().zip(sol.y.iter()) {
            let exact = 4.0 / (1.0 + x).powi(2);
            max_err = max_err.max((yi - exact).abs());
        }
        assert!(max_err < 1.0e-4, "max_err = {max_err:e}");
        // Exact initial slope is y'(0) = -8.
        assert!((sol.slope + 8.0).abs() < 1.0e-3, "slope = {}", sol.slope);
    }

    #[test]
    fn first_derivative_in_rhs() {
        // y'' = -y' (damping), y(0)=0, y(1)=1 ⇒ y = (1-e^{-x})/(1-e^{-1}).
        let f = |_x: f64, _y: f64, yp: f64| -yp;
        let cfg = ShootingConfig::default();
        let sol = solve_shooting(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        let denom = 1.0 - (-1.0_f64).exp();
        let mut max_err = 0.0_f64;
        for (&x, &yi) in sol.x.iter().zip(sol.y.iter()) {
            let exact = (1.0 - (-x).exp()) / denom;
            max_err = max_err.max((yi - exact).abs());
        }
        assert!(max_err < 1.0e-6, "max_err = {max_err:e}");
    }

    #[test]
    fn rejects_bad_interval_and_slopes() {
        let f = |_x: f64, y: f64, _yp: f64| y;
        assert!(solve_shooting(f, 1.0, 0.0, 0.0, 1.0, &ShootingConfig::default()).is_err());
        let bad = ShootingConfig {
            slope0: 1.0,
            slope1: 1.0,
            ..ShootingConfig::default()
        };
        assert!(solve_shooting(f, 0.0, 1.0, 0.0, 1.0, &bad).is_err());
        let zero_steps = ShootingConfig {
            n_steps: 0,
            ..ShootingConfig::default()
        };
        assert!(solve_shooting(f, 0.0, 1.0, 0.0, 1.0, &zero_steps).is_err());
    }
}
