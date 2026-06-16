//! Semi-explicit index-1 differential-algebraic equation (DAE) solver.
//!
//! Integrates the coupled system of `n_x` differential and `n_y` algebraic
//! equations
//!
//! ```text
//! x'(t) = f(x, y, t),          (differential part)
//! 0     = g(x, y, t),          (algebraic constraint)
//! ```
//!
//! under the **index-1** assumption that the constraint Jacobian `∂g/∂y` is
//! nonsingular along the solution, so that `y` is (locally) an implicit algebraic
//! function of `x` and `t`.
//!
//! The backward-Euler discretisation solves, at each step `t_n → t_{n+1} = t_n + h`,
//! the combined nonlinear system in `z = (x_{n+1}, y_{n+1})`
//!
//! ```text
//! G(z) = [ x_{n+1} − x_n − h f(x_{n+1}, y_{n+1}, t_{n+1}) ]  = 0,
//!        [ g(x_{n+1}, y_{n+1}, t_{n+1})                    ]
//! ```
//!
//! by Newton's method. The `(n_x + n_y) × (n_x + n_y)` Jacobian `∂G/∂z` is formed
//! with the crate's forward-difference Jacobian helper and factorised by LU with
//! partial pivoting; the differential rows scale the constraint Jacobian by `h`
//! while the algebraic rows carry the full constraint Jacobian, so the combined
//! matrix is nonsingular exactly when `∂g/∂y` is — the index-1 condition. Implicit
//! Euler is `𝒪(h)` accurate and L-stable, appropriate for the stiff/constrained
//! dynamics typical of index-1 DAEs.
//!
//! Consistent initial conditions are handled by [`DaeSolver::consistent_initial`],
//! which solves `g(x₀, y₀, t₀) = 0` for `y₀` given `x₀` by Newton before the first
//! step.
//!
//! Reference: K. E. Brenan, S. L. Campbell and L. R. Petzold, *Numerical Solution
//! of Initial-Value Problems in Differential-Algebraic Equations*, SIAM Classics
//! (1996); E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*,
//! 2nd ed., Springer (1996), §VI–VII.

use crate::error::{NumericError, NumericResult};
use crate::linalg::lu_decomp::{lu_decompose, lu_solve};

use super::finite_diff_jacobian;

/// Configuration for the [`DaeSolver`].
#[derive(Debug, Clone, Copy)]
pub struct DaeConfig {
    /// L2 tolerance on the combined Newton residual `‖G(z)‖`.
    pub newton_tol: f64,
    /// Maximum Newton iterations per step (and for the consistent-IC solve).
    pub max_newton_iter: usize,
    /// Relative perturbation used for the forward-difference Jacobian.
    pub fd_eps: f64,
}

impl Default for DaeConfig {
    fn default() -> Self {
        Self {
            newton_tol: 1.0e-10,
            max_newton_iter: 50,
            fd_eps: 1.0e-7,
        }
    }
}

/// Trajectory returned by [`DaeSolver::integrate`].
#[derive(Debug, Clone)]
pub struct DaeSolution {
    /// Time grid `t₀ … t_end`.
    pub t: Vec<f64>,
    /// Differential states `x(tᵢ)` (each of length `n_x`).
    pub x: Vec<Vec<f64>>,
    /// Algebraic states `y(tᵢ)` (each of length `n_y`).
    pub y: Vec<Vec<f64>>,
    /// Maximum constraint residual `max_i ‖g(xᵢ, yᵢ, tᵢ)‖₂` over the trajectory.
    pub max_constraint_residual: f64,
}

/// Backward-Euler integrator for semi-explicit index-1 DAEs.
#[derive(Debug, Clone, Copy, Default)]
pub struct DaeSolver {
    /// Solver configuration.
    pub config: DaeConfig,
}

impl DaeSolver {
    /// Create a solver with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a solver with the supplied configuration.
    pub fn with_config(config: DaeConfig) -> Self {
        Self { config }
    }

    /// Solve `g(x₀, y₀, t₀) = 0` for the algebraic variables `y₀` given the
    /// differential state `x₀` and an initial guess `y_guess`, by Newton iteration.
    ///
    /// This produces a *consistent* initial condition before integration begins.
    ///
    /// # Errors
    /// * [`NumericError::EmptyInput`] if `y_guess` is empty.
    /// * [`NumericError::ShapeMismatch`] if `g` does not return `n_y` values.
    /// * [`NumericError::SingularMatrix`] if `∂g/∂y` is singular at the iterate
    ///   (the problem is not index 1 there).
    /// * [`NumericError::NotConverged`] if Newton fails within `max_newton_iter`.
    pub fn consistent_initial<G>(
        &self,
        g: &G,
        x0: &[f64],
        y_guess: &[f64],
        t0: f64,
    ) -> NumericResult<Vec<f64>>
    where
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let ny = y_guess.len();
        if ny == 0 {
            return Err(NumericError::EmptyInput);
        }
        let eps = self.config.fd_eps;
        let mut y = y_guess.to_vec();
        let mut residual = f64::INFINITY;
        for _ in 0..self.config.max_newton_iter {
            let gv = g(x0, &y, t0);
            if gv.len() != ny {
                return Err(NumericError::ShapeMismatch {
                    expected: vec![ny],
                    got: vec![gv.len()],
                });
            }
            let gnorm = gv.iter().map(|v| v * v).sum::<f64>().sqrt();
            residual = gnorm;
            if gnorm <= self.config.newton_tol {
                return Ok(y);
            }
            // Jacobian ∂g/∂y by forward differences (x₀, t₀ held fixed).
            let gy = |_t: f64, yy: &[f64]| g(x0, yy, t0);
            let jac = finite_diff_jacobian(&gy, t0, &y, eps)?;
            let (lu, piv, _) = lu_decompose(&jac, ny)?;
            let delta = lu_solve(&lu, &piv, ny, &gv)?;
            for (yi, d) in y.iter_mut().zip(delta.iter()) {
                *yi -= d;
            }
            if !y.iter().all(|v| v.is_finite()) {
                return Err(NumericError::NumericalInstability(
                    "DAE consistent-IC: Newton diverged".into(),
                ));
            }
        }
        Err(NumericError::NotConverged {
            iter: self.config.max_newton_iter,
            residual,
        })
    }

    /// Integrate the index-1 DAE from `t0` to `t_end` with fixed step `h`,
    /// starting from the (assumed consistent) initial state `(x0, y0)`.
    ///
    /// `f(x, y, t) → x'` returns `n_x` values; `g(x, y, t) → 0` returns `n_y` values.
    /// The interval is covered by `ceil((t_end − t0)/h)` uniform sub-steps so the
    /// integration lands exactly on `t_end`.
    ///
    /// # Errors
    /// * [`NumericError::InvalidStepSize`] if `h ≤ 0` or non-finite.
    /// * [`NumericError::InvalidParameter`] if the endpoints are non-finite, `x0`
    ///   or `y0` is empty, or `t_end < t0`.
    /// * [`NumericError::ShapeMismatch`] if `f`/`g` return wrong lengths.
    /// * [`NumericError::SingularMatrix`] if a step Jacobian is singular (loss of
    ///   the index-1 property).
    /// * [`NumericError::NotConverged`] if a step's Newton iteration fails.
    pub fn integrate<F, G>(
        &self,
        f: F,
        g: G,
        t0: f64,
        x0: &[f64],
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> NumericResult<DaeSolution>
    where
        F: Fn(&[f64], &[f64], f64) -> Vec<f64>,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        if !h.is_finite() || h <= 0.0 {
            return Err(NumericError::InvalidStepSize { step: h });
        }
        if !t0.is_finite() || !t_end.is_finite() {
            return Err(NumericError::InvalidParameter(
                "DAE: t0 and t_end must be finite".into(),
            ));
        }
        if x0.is_empty() || y0.is_empty() {
            return Err(NumericError::InvalidParameter(
                "DAE: x0 and y0 must be non-empty".into(),
            ));
        }
        let nx = x0.len();
        let ny = y0.len();
        let nz = nx + ny;
        let total = t_end - t0;

        let mut t_grid = vec![t0];
        let mut x_traj = vec![x0.to_vec()];
        let mut y_traj = vec![y0.to_vec()];

        // Verify the initial constraint residual contributes to the reported max.
        let g0 = g(x0, y0, t0);
        if g0.len() != ny {
            return Err(NumericError::ShapeMismatch {
                expected: vec![ny],
                got: vec![g0.len()],
            });
        }
        let mut max_constraint_residual = g0.iter().map(|v| v * v).sum::<f64>().sqrt();

        if total == 0.0 {
            return Ok(DaeSolution {
                t: t_grid,
                x: x_traj,
                y: y_traj,
                max_constraint_residual,
            });
        }
        if total < 0.0 {
            return Err(NumericError::InvalidParameter(
                "DAE: t_end must be ≥ t0 (forward integration with h > 0)".into(),
            ));
        }

        let n_steps = (total / h).ceil().max(1.0) as usize;
        let h_step = total / n_steps as f64;
        let eps = self.config.fd_eps;

        let mut t = t0;
        let mut x = x0.to_vec();
        let mut y = y0.to_vec();

        for _ in 0..n_steps {
            let t_next = t + h_step;
            let x_prev = x.clone();

            // Combined residual G(z), z = [x_{n+1}; y_{n+1}] of length nz.
            let big_g = |z: &[f64]| -> NumericResult<Vec<f64>> {
                let (xn, yn) = z.split_at(nx);
                let fv = f(xn, yn, t_next);
                if fv.len() != nx {
                    return Err(NumericError::ShapeMismatch {
                        expected: vec![nx],
                        got: vec![fv.len()],
                    });
                }
                let gv = g(xn, yn, t_next);
                if gv.len() != ny {
                    return Err(NumericError::ShapeMismatch {
                        expected: vec![ny],
                        got: vec![gv.len()],
                    });
                }
                let mut res = vec![0.0_f64; nz];
                for i in 0..nx {
                    res[i] = xn[i] - x_prev[i] - h_step * fv[i];
                }
                res[nx..nz].copy_from_slice(&gv);
                Ok(res)
            };

            // Newton on z, initialised at the previous (x, y).
            let mut z = Vec::with_capacity(nz);
            z.extend_from_slice(&x);
            z.extend_from_slice(&y);

            let mut converged = false;
            let mut residual = f64::INFINITY;
            for _ in 0..self.config.max_newton_iter {
                let gz = big_g(&z)?;
                let gnorm = gz.iter().map(|v| v * v).sum::<f64>().sqrt();
                residual = gnorm;
                if gnorm <= self.config.newton_tol {
                    converged = true;
                    break;
                }
                // Combined Jacobian ∂G/∂z via forward differences. The dummy time
                // argument is ignored by big_g; pass t_next for clarity.
                let gwrap = |_tt: f64, zz: &[f64]| big_g(zz).unwrap_or_else(|_| vec![f64::NAN; nz]);
                let jac = finite_diff_jacobian(&gwrap, t_next, &z, eps)?;
                if jac.iter().any(|v| !v.is_finite()) {
                    return Err(NumericError::NumericalInstability(
                        "DAE: non-finite Jacobian entry (constraint evaluation failed)".into(),
                    ));
                }
                let (lu, piv, _) = lu_decompose(&jac, nz)?;
                let delta = lu_solve(&lu, &piv, nz, &gz)?;
                for (zi, d) in z.iter_mut().zip(delta.iter()) {
                    *zi -= d;
                }
                if !z.iter().all(|v| v.is_finite()) {
                    return Err(NumericError::NumericalInstability(format!(
                        "DAE: Newton iteration diverged at t={t_next}"
                    )));
                }
            }
            if !converged {
                return Err(NumericError::NotConverged {
                    iter: self.config.max_newton_iter,
                    residual,
                });
            }

            let (xn, yn) = z.split_at(nx);
            x = xn.to_vec();
            y = yn.to_vec();
            t = t_next;

            let gv = g(&x, &y, t);
            let gnorm = gv.iter().map(|v| v * v).sum::<f64>().sqrt();
            max_constraint_residual = max_constraint_residual.max(gnorm);

            t_grid.push(t);
            x_traj.push(x.clone());
            y_traj.push(y.clone());
        }

        Ok(DaeSolution {
            t: t_grid,
            x: x_traj,
            y: y_traj,
            max_constraint_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_solution_decay_with_square_constraint() {
        // x' = -x, 0 = y - x², x(0)=1 ⇒ x=e^{-t}, y=e^{-2t}.
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - x[0] * x[0]];
        let sol = solver
            .integrate(f, g, 0.0, &[1.0], &[1.0], 1.0, 0.001)
            .expect("ok");
        let xt = sol.x.last().expect("nonempty")[0];
        let yt = sol.y.last().expect("nonempty")[0];
        let exact_x = (-1.0_f64).exp();
        let exact_y = (-2.0_f64).exp();
        assert!((xt - exact_x).abs() < 1.0e-3, "x={xt}, exact={exact_x}");
        assert!((yt - exact_y).abs() < 2.0e-3, "y={yt}, exact={exact_y}");
        // Algebraic relation y = x² holds along the whole trajectory.
        for (xi, yi) in sol.x.iter().zip(sol.y.iter()) {
            assert!((yi[0] - xi[0] * xi[0]).abs() < 1.0e-8);
        }
    }

    #[test]
    fn constraint_residual_small_every_step() {
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - x[0] * x[0]];
        let sol = solver
            .integrate(f, g, 0.0, &[1.0], &[1.0], 1.0, 0.01)
            .expect("ok");
        assert!(
            sol.max_constraint_residual <= solver.config.newton_tol,
            "max g residual = {:e}",
            sol.max_constraint_residual
        );
        // Re-check each node directly.
        for (xi, yi) in sol.x.iter().zip(sol.y.iter()) {
            let gnorm = (yi[0] - xi[0] * xi[0]).abs();
            assert!(gnorm <= solver.config.newton_tol, "g = {gnorm:e}");
        }
    }

    #[test]
    fn linear_index1_dae_closed_form() {
        // Linear index-1 system (mass-matrix / RC-circuit form):
        //   x' = -x + y,   0 = y - 2 x   ⇒  x' = x,  but with x(0)=1, y=2x.
        // Substituting: x' = -x + 2x = x ⇒ x = e^{t}, y = 2 e^{t}.
        let solver = DaeSolver::with_config(DaeConfig {
            newton_tol: 1.0e-12,
            ..DaeConfig::default()
        });
        let f = |x: &[f64], y: &[f64], _t: f64| vec![-x[0] + y[0]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - 2.0 * x[0]];
        let sol = solver
            .integrate(f, g, 0.0, &[1.0], &[2.0], 0.5, 0.0005)
            .expect("ok");
        let xt = sol.x.last().expect("nonempty")[0];
        let yt = sol.y.last().expect("nonempty")[0];
        let exact_x = (0.5_f64).exp();
        assert!((xt - exact_x).abs() < 1.0e-2, "x={xt}, exact={exact_x}");
        assert!((yt - 2.0 * xt).abs() < 1.0e-9, "y must equal 2x");
    }

    #[test]
    fn implicit_euler_first_order_convergence() {
        // Halving h roughly halves the error (O(h)) for implicit Euler.
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - x[0] * x[0]];
        let exact_x = (-1.0_f64).exp();
        let coarse = solver
            .integrate(f, g, 0.0, &[1.0], &[1.0], 1.0, 0.02)
            .expect("ok")
            .x
            .last()
            .expect("nonempty")[0];
        let fine = solver
            .integrate(f, g, 0.0, &[1.0], &[1.0], 1.0, 0.01)
            .expect("ok")
            .x
            .last()
            .expect("nonempty")[0];
        let e_coarse = (coarse - exact_x).abs();
        let e_fine = (fine - exact_x).abs();
        let ratio = e_coarse / e_fine;
        // O(h): expect ~2× reduction; allow a generous band.
        assert!(
            (1.6..=2.6).contains(&ratio),
            "ratio={ratio} (coarse={e_coarse:e}, fine={e_fine:e})"
        );
    }

    #[test]
    fn consistent_initial_solves_constraint() {
        // Given x₀=2, solve 0 = y - x³ for y₀ (true y₀ = 8) from a poor guess.
        let solver = DaeSolver::new();
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - x[0].powi(3)];
        let y0 = solver
            .consistent_initial(&g, &[2.0], &[0.0], 0.0)
            .expect("ok");
        assert!((y0[0] - 8.0).abs() < 1.0e-8, "y0 = {}", y0[0]);
        // A genuinely nonlinear constraint: 0 = y² - x, x=4 ⇒ y=2 (guess near +2).
        let g2 = |x: &[f64], y: &[f64], _t: f64| vec![y[0] * y[0] - x[0]];
        let y0b = solver
            .consistent_initial(&g2, &[4.0], &[1.0], 0.0)
            .expect("ok");
        assert!((y0b[0] - 2.0).abs() < 1.0e-8, "y0 = {}", y0b[0]);
    }

    #[test]
    fn consistent_initial_then_integrate() {
        // Multi-variable: x'=-x, 0 = y - (x² + sin t at t) ... use g = y - x²-1.
        // Recover y₀ from x₀=1 ⇒ y₀ = 2, then integrate and re-verify constraint.
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - x[0] * x[0] - 1.0];
        let y0 = solver
            .consistent_initial(&g, &[1.0], &[0.0], 0.0)
            .expect("ok");
        assert!((y0[0] - 2.0).abs() < 1.0e-9);
        let sol = solver
            .integrate(f, g, 0.0, &[1.0], &y0, 0.5, 0.005)
            .expect("ok");
        assert!(sol.max_constraint_residual <= solver.config.newton_tol);
    }

    #[test]
    fn singular_constraint_jacobian_detected() {
        // 0 = x - 2 has NO dependence on y ⇒ ∂g/∂y ≡ 0 (singular). With x₀ = 1 the
        // constraint is violated, so Newton must take a step and hit the singular
        // Jacobian rather than accepting the guess outright.
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0]];
        let g = |x: &[f64], _y: &[f64], _t: f64| vec![x[0] - 2.0];
        // Consistent-IC solve cannot determine y (singular Jacobian ∂g/∂y = 0).
        let ic = solver.consistent_initial(&g, &[1.0], &[0.5], 0.0);
        assert!(ic.is_err(), "expected singular-Jacobian error");
        // A full step also fails because the y-column of the combined Jacobian is
        // identically zero (the index-1 property is lost).
        let step = solver.integrate(f, g, 0.0, &[1.0], &[0.5], 0.1, 0.05);
        assert!(step.is_err(), "expected singular combined Jacobian error");
    }

    #[test]
    fn two_variable_index1_system() {
        // x₁'=-x₁, x₂'=-2x₂, 0 = y - (x₁ + x₂). x(0)=(1,1) ⇒ y = e^{-t}+e^{-2t}.
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0], -2.0 * x[1]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - (x[0] + x[1])];
        let sol = solver
            .integrate(f, g, 0.0, &[1.0, 1.0], &[2.0], 1.0, 0.002)
            .expect("ok");
        let xf = sol.x.last().expect("nonempty");
        let yf = sol.y.last().expect("nonempty")[0];
        let ex1 = (-1.0_f64).exp();
        let ex2 = (-2.0_f64).exp();
        assert!((xf[0] - ex1).abs() < 1.0e-2);
        assert!((xf[1] - ex2).abs() < 1.0e-2);
        assert!((yf - (xf[0] + xf[1])).abs() < 1.0e-8);
    }

    #[test]
    fn rejects_bad_step_and_empty() {
        let solver = DaeSolver::new();
        let f = |x: &[f64], _y: &[f64], _t: f64| vec![-x[0]];
        let g = |x: &[f64], y: &[f64], _t: f64| vec![y[0] - x[0] * x[0]];
        assert!(
            solver
                .integrate(f, g, 0.0, &[1.0], &[1.0], 1.0, 0.0)
                .is_err()
        );
        assert!(
            solver
                .integrate(f, g, 0.0, &[1.0], &[1.0], 1.0, f64::NAN)
                .is_err()
        );
        assert!(solver.integrate(f, g, 0.0, &[], &[1.0], 1.0, 0.1).is_err());
    }
}
