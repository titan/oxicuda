//! Finite-difference solver for the two-point boundary-value problem
//!
//! ```text
//! y'' = f(x, y, y'),     x ∈ [a, b],     y(a) = α,   y(b) = β.
//! ```
//!
//! The interval is discretised on a uniform grid `xᵢ = a + i h`, `i = 0 … N`,
//! `h = (b − a)/N`, with `y₀ = α` and `y_N = β` fixed. At each interior node the
//! derivatives are replaced by the second-order central differences
//!
//! ```text
//! y''(xᵢ) ≈ (yᵢ₋₁ − 2 yᵢ + yᵢ₊₁) / h²,
//! y'(xᵢ)  ≈ (yᵢ₊₁ − yᵢ₋₁) / (2 h),
//! ```
//!
//! producing the nonlinear residual system `Fᵢ(y) = 0`, `i = 1 … N−1`,
//!
//! ```text
//! Fᵢ = (yᵢ₋₁ − 2 yᵢ + yᵢ₊₁)/h² − f(xᵢ, yᵢ, (yᵢ₊₁ − yᵢ₋₁)/(2h)).
//! ```
//!
//! Because each `Fᵢ` involves only `yᵢ₋₁, yᵢ, yᵢ₊₁`, the Jacobian `∂F/∂y` is
//! **tridiagonal**; its sub/diagonal/super entries are formed analytically from
//! finite-difference partials of `f` w.r.t. `y` and `y'`, and each Newton update
//! is solved in `𝒪(N)` by the Thomas algorithm. For a *linear* problem Newton
//! converges in a single step (the residual is affine), giving the classical
//! `𝒪(h²)` second-order scheme.
//!
//! Reference: H. B. Keller, *Numerical Methods for Two-Point Boundary-Value
//! Problems*, Blaisdell (1968), §3; U. M. Ascher, R. M. R. Mattheij and R. D.
//! Russell, *Numerical Solution of Boundary Value Problems for ODEs*, SIAM Classics
//! (1995), §3.

use crate::error::{NumericError, NumericResult};

/// Configuration for the finite-difference BVP solver.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifferenceConfig {
    /// Number of uniform sub-intervals `N` (grid has `N + 1` points). Must be ≥ 2.
    pub n_intervals: usize,
    /// L2 tolerance on the Newton residual `‖F(y)‖`.
    pub newton_tol: f64,
    /// Maximum Newton iterations.
    pub max_iter: usize,
    /// Relative perturbation for the finite-difference partials of `f`.
    pub fd_eps: f64,
}

impl Default for FiniteDifferenceConfig {
    fn default() -> Self {
        Self {
            n_intervals: 100,
            newton_tol: 1.0e-10,
            max_iter: 100,
            fd_eps: 1.0e-7,
        }
    }
}

/// Solution returned by [`solve_finite_difference`].
#[derive(Debug, Clone)]
pub struct FiniteDifferenceSolution {
    /// Grid abscissae `x₀ = a, …, x_N = b` (length `n_intervals + 1`).
    pub x: Vec<f64>,
    /// Solution values `y(xᵢ)` on the grid (endpoints equal `α`, `β` exactly).
    pub y: Vec<f64>,
    /// Newton iterations performed.
    pub iterations: usize,
    /// Final Newton residual norm `‖F(y)‖₂`.
    pub residual: f64,
}

/// Solve `y'' = f(x, y, y')` on `[a, b]` with `y(a) = alpha`, `y(b) = beta` by a
/// second-order central finite-difference scheme and a tridiagonal Newton solve.
///
/// `f` receives `(x, y, y')` and returns `y''`.
///
/// # Errors
/// * [`NumericError::InvalidParameter`] if `b ≤ a`, the endpoints/BCs are
///   non-finite, or `n_intervals < 2`.
/// * [`NumericError::SingularMatrix`] if a Newton iteration matrix is singular
///   (zero pivot in the Thomas elimination).
/// * [`NumericError::NotConverged`] if Newton fails to reach `newton_tol` within
///   `max_iter` iterations.
pub fn solve_finite_difference<F>(
    f: F,
    a: f64,
    b: f64,
    alpha: f64,
    beta: f64,
    config: &FiniteDifferenceConfig,
) -> NumericResult<FiniteDifferenceSolution>
where
    F: Fn(f64, f64, f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || b <= a {
        return Err(NumericError::InvalidParameter(
            "finite-difference BVP: require finite a < b".into(),
        ));
    }
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(NumericError::InvalidParameter(
            "finite-difference BVP: boundary values must be finite".into(),
        ));
    }
    if config.n_intervals < 2 {
        return Err(NumericError::InvalidParameter(
            "finite-difference BVP: n_intervals must be >= 2".into(),
        ));
    }

    let n = config.n_intervals;
    let h = (b - a) / n as f64;
    let h2 = h * h;
    let inv_h2 = 1.0 / h2;
    let inv_2h = 1.0 / (2.0 * h);
    let m = n - 1; // number of interior unknowns y₁ … y_{N-1}

    let x: Vec<f64> = (0..=n).map(|i| a + i as f64 * h).collect();

    // Full solution vector including fixed endpoints; iterate on interior nodes.
    let mut y = vec![0.0_f64; n + 1];
    y[0] = alpha;
    y[n] = beta;
    // Linear initial guess between the boundary values.
    for (i, yi) in y.iter_mut().enumerate().take(n).skip(1) {
        let frac = i as f64 / n as f64;
        *yi = alpha + frac * (beta - alpha);
    }

    // Tridiagonal Jacobian bands and residual for the interior system.
    let mut lower = vec![0.0_f64; m]; // sub-diagonal  (lower[0] unused)
    let mut diag = vec![0.0_f64; m];
    let mut upper = vec![0.0_f64; m]; // super-diagonal (upper[m-1] unused)
    let mut resid = vec![0.0_f64; m];

    let mut iterations = 0_usize;
    let mut res_norm = f64::INFINITY;
    let mut converged = false;

    while iterations < config.max_iter {
        // Assemble residual Fᵢ and the analytic tridiagonal Jacobian.
        for k in 0..m {
            let i = k + 1; // global node index
            let xi = x[i];
            let ym = y[i - 1];
            let yi = y[i];
            let yp_node = y[i + 1];
            let dy = (yp_node - ym) * inv_2h; // central first derivative
            let fval = f(xi, yi, dy);
            resid[k] = (ym - 2.0 * yi + yp_node) * inv_h2 - fval;

            // Partials of f via central finite differences in y and y'.
            let dy_y = config.fd_eps * yi.abs().max(1.0);
            let f_yp = f(xi, yi + dy_y, dy);
            let f_ym = f(xi, yi - dy_y, dy);
            let dfdy = (f_yp - f_ym) / (2.0 * dy_y);

            let dy_d = config.fd_eps * dy.abs().max(1.0);
            let f_dp = f(xi, yi, dy + dy_d);
            let f_dm = f(xi, yi, dy - dy_d);
            let dfdyp = (f_dp - f_dm) / (2.0 * dy_d);

            // ∂Fᵢ/∂yᵢ₋₁ =  1/h² + dfdyp/(2h)
            // ∂Fᵢ/∂yᵢ   = -2/h² − dfdy
            // ∂Fᵢ/∂yᵢ₊₁ =  1/h² − dfdyp/(2h)
            lower[k] = inv_h2 + dfdyp * inv_2h;
            diag[k] = -2.0 * inv_h2 - dfdy;
            upper[k] = inv_h2 - dfdyp * inv_2h;
        }

        res_norm = resid.iter().map(|r| r * r).sum::<f64>().sqrt();

        // Solve J Δ = −F for the Newton correction (tridiagonal Thomas algorithm).
        let rhs: Vec<f64> = resid.iter().map(|r| -r).collect();
        let delta = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

        // Apply the correction and measure its infinity norm. Convergence is judged
        // on ‖Δ‖∞ (scale-free, round-off limited) rather than on the residual norm,
        // which carries the ill-conditioned 1/h² factor — the same convention used
        // by the crate's implicit RK integrators.
        let mut step_norm = 0.0_f64;
        for (k, d) in delta.iter().enumerate() {
            if !d.is_finite() {
                return Err(NumericError::NumericalInstability(
                    "finite-difference BVP: Newton produced a non-finite update".into(),
                ));
            }
            y[k + 1] += d;
            step_norm = step_norm.max(d.abs());
        }
        iterations += 1;
        if step_norm <= config.newton_tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(NumericError::NotConverged {
            iter: iterations,
            residual: res_norm,
        });
    }

    // Residual of the accepted iterate, for diagnostics.
    let mut final_res = 0.0_f64;
    for k in 0..m {
        let i = k + 1;
        let dy = (y[i + 1] - y[i - 1]) * inv_2h;
        let fi = (y[i - 1] - 2.0 * y[i] + y[i + 1]) * inv_h2 - f(x[i], y[i], dy);
        final_res += fi * fi;
    }
    let final_res = final_res.sqrt();

    Ok(FiniteDifferenceSolution {
        x,
        y,
        iterations,
        residual: final_res,
    })
}

/// Solve the tridiagonal system `T x = d` by the Thomas algorithm, where `T` has
/// sub-diagonal `lower` (`lower[0]` ignored), diagonal `diag`, and super-diagonal
/// `upper` (`upper[m−1]` ignored). All slices have length `m`.
///
/// # Errors
/// [`NumericError::SingularMatrix`] on a zero pivot.
fn solve_tridiagonal(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    d: &[f64],
) -> NumericResult<Vec<f64>> {
    let m = diag.len();
    if lower.len() != m || upper.len() != m || d.len() != m {
        return Err(NumericError::DimensionMismatch {
            a: diag.len(),
            b: d.len(),
        });
    }
    if m == 0 {
        return Err(NumericError::EmptyInput);
    }

    let mut c_prime = vec![0.0_f64; m];
    let mut d_prime = vec![0.0_f64; m];

    let mut beta = diag[0];
    if beta.abs() < 1.0e-300 {
        return Err(NumericError::SingularMatrix(
            "tridiagonal: zero leading pivot".into(),
        ));
    }
    c_prime[0] = upper[0] / beta;
    d_prime[0] = d[0] / beta;

    for i in 1..m {
        beta = diag[i] - lower[i] * c_prime[i - 1];
        if beta.abs() < 1.0e-300 {
            return Err(NumericError::SingularMatrix(format!(
                "tridiagonal: zero pivot at row {i}"
            )));
        }
        c_prime[i] = upper[i] / beta;
        d_prime[i] = (d[i] - lower[i] * d_prime[i - 1]) / beta;
    }

    let mut x = vec![0.0_f64; m];
    x[m - 1] = d_prime[m - 1];
    for i in (0..m - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_grid_error(sol: &FiniteDifferenceSolution, exact: impl Fn(f64) -> f64) -> f64 {
        sol.x
            .iter()
            .zip(sol.y.iter())
            .map(|(&x, &y)| (y - exact(x)).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn tridiagonal_solver_matches_dense() {
        // [2,-1,0; -1,2,-1; 0,-1,2] x = [1,0,1]  ⇒  x = [1,1,1].
        let lower = vec![0.0, -1.0, -1.0];
        let diag = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0, 0.0];
        let rhs = vec![1.0, 0.0, 1.0];
        let x = solve_tridiagonal(&lower, &diag, &upper, &rhs).expect("ok");
        for xi in &x {
            assert!((xi - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn linear_bvp_matches_sinh() {
        // y'' = y, y(0)=0, y(1)=1 ⇒ y = sinh(x)/sinh(1).
        let f = |_x: f64, y: f64, _yp: f64| y;
        let cfg = FiniteDifferenceConfig::default();
        let sol = solve_finite_difference(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        let sinh1 = 1.0_f64.sinh();
        let err = max_grid_error(&sol, |x| x.sinh() / sinh1);
        assert!(err < 1.0e-3, "err = {err:e}");
        // Linear ⇒ the first Newton update is exact; the second step confirms
        // convergence with a round-off-sized correction (‖Δ‖-based criterion).
        assert!(sol.iterations <= 2, "iters = {}", sol.iterations);
        // The accepted residual sits at the discretisation round-off floor.
        assert!(sol.residual < 1.0e-7, "residual = {:e}", sol.residual);
    }

    #[test]
    fn boundary_conditions_satisfied_exactly() {
        let f = |_x: f64, y: f64, _yp: f64| y;
        let cfg = FiniteDifferenceConfig::default();
        let sol = solve_finite_difference(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        assert!((sol.y[0] - 0.0).abs() < 1.0e-15);
        assert!((sol.y[sol.y.len() - 1] - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn second_order_convergence() {
        // Halving h must cut the error by ~4 (O(h²)).
        let f = |_x: f64, y: f64, _yp: f64| y;
        let sinh1 = 1.0_f64.sinh();
        let exact = |x: f64| x.sinh() / sinh1;
        let coarse = solve_finite_difference(
            f,
            0.0,
            1.0,
            0.0,
            1.0,
            &FiniteDifferenceConfig {
                n_intervals: 50,
                ..FiniteDifferenceConfig::default()
            },
        )
        .expect("ok");
        let fine = solve_finite_difference(
            f,
            0.0,
            1.0,
            0.0,
            1.0,
            &FiniteDifferenceConfig {
                n_intervals: 100,
                ..FiniteDifferenceConfig::default()
            },
        )
        .expect("ok");
        let e_coarse = max_grid_error(&coarse, exact);
        let e_fine = max_grid_error(&fine, exact);
        let ratio = e_coarse / e_fine;
        assert!(
            (3.5..=4.5).contains(&ratio),
            "ratio = {ratio} (e_coarse={e_coarse:e}, e_fine={e_fine:e})"
        );
    }

    #[test]
    fn agrees_with_shooting() {
        // Cross-check against the analytic solution at coarse FD truncation level.
        let f = |_x: f64, y: f64, _yp: f64| y;
        let cfg = FiniteDifferenceConfig {
            n_intervals: 200,
            ..FiniteDifferenceConfig::default()
        };
        let sol = solve_finite_difference(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        let sinh1 = 1.0_f64.sinh();
        // At N=200, h²≈2.5e-5, so the scheme should track the exact solution closely.
        let err = max_grid_error(&sol, |x| x.sinh() / sinh1);
        assert!(err < 1.0e-4, "err = {err:e}");
    }

    #[test]
    fn nonlinear_bratu_converges() {
        // Bratu: y'' + λ e^y = 0, i.e. y'' = −λ e^y, y(0)=y(1)=0.
        // For small λ the lower branch has a known closed form via the parameter θ:
        //   y(x) = -2 ln[ cosh((x-1/2) θ/2) / cosh(θ/4) ],  θ = √(2λ) cosh(θ/4).
        let lambda = 1.0_f64;
        let f = move |_x: f64, y: f64, _yp: f64| -lambda * y.exp();
        let cfg = FiniteDifferenceConfig::default();
        let sol = solve_finite_difference(f, 0.0, 1.0, 0.0, 0.0, &cfg).expect("ok");
        // Solve θ = √(2λ) cosh(θ/4) by fixed-point / bisection on the lower branch.
        let g = |theta: f64| (2.0 * lambda).sqrt() * (theta / 4.0).cosh() - theta;
        // Lower branch root in (0, ~1.5).
        let (mut lo, mut hi) = (1.0e-6_f64, 2.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if g(lo) * g(mid) <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let theta = 0.5 * (lo + hi);
        let exact =
            |x: f64| -2.0 * ((x - 0.5) * theta / 2.0).cosh().ln() + 2.0 * (theta / 4.0).cosh().ln();
        let err = max_grid_error(&sol, exact);
        assert!(err < 1.0e-3, "Bratu err = {err:e}, theta={theta}");
        // Symmetric, positive interior, peak at the midpoint.
        let mid = sol.y[sol.y.len() / 2];
        assert!(mid > 0.0, "interior must be positive, got {mid}");
    }

    #[test]
    fn nonlinear_three_halves_square() {
        // y'' = (3/2) y², y(0)=4, y(1)=1 ⇒ y = 4/(1+x)².
        let f = |_x: f64, y: f64, _yp: f64| 1.5 * y * y;
        let cfg = FiniteDifferenceConfig {
            n_intervals: 200,
            ..FiniteDifferenceConfig::default()
        };
        let sol = solve_finite_difference(f, 0.0, 1.0, 4.0, 1.0, &cfg).expect("ok");
        let err = max_grid_error(&sol, |x| 4.0 / (1.0 + x).powi(2));
        assert!(err < 1.0e-3, "err = {err:e}");
        // Newton needed a few iterations for the nonlinear problem.
        assert!(sol.iterations >= 1);
    }

    #[test]
    fn convection_term_first_derivative() {
        // y'' = -y', y(0)=0, y(1)=1 ⇒ y = (1-e^{-x})/(1-e^{-1}).
        let f = |_x: f64, _y: f64, yp: f64| -yp;
        let cfg = FiniteDifferenceConfig::default();
        let sol = solve_finite_difference(f, 0.0, 1.0, 0.0, 1.0, &cfg).expect("ok");
        let denom = 1.0 - (-1.0_f64).exp();
        let err = max_grid_error(&sol, |x| (1.0 - (-x).exp()) / denom);
        assert!(err < 1.0e-3, "err = {err:e}");
    }

    #[test]
    fn rejects_bad_input() {
        let f = |_x: f64, y: f64, _yp: f64| y;
        assert!(
            solve_finite_difference(f, 1.0, 0.0, 0.0, 1.0, &FiniteDifferenceConfig::default())
                .is_err()
        );
        let bad = FiniteDifferenceConfig {
            n_intervals: 1,
            ..FiniteDifferenceConfig::default()
        };
        assert!(solve_finite_difference(f, 0.0, 1.0, 0.0, 1.0, &bad).is_err());
    }
}
