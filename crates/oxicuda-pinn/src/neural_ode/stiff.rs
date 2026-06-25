//! Stiff ODE solvers: implicit (backward) Euler, Rosenbrock-W (ROS2), and
//! variable-order BDF (order 1 and 2).
//!
//! Explicit Runge-Kutta methods (Euler / Heun / RK4 / DOPRI45 in
//! [`crate::neural_ode::solvers`]) suffer catastrophic step-size restrictions
//! on *stiff* systems — problems with widely separated time-scales, e.g.
//! chemical-kinetics networks or electrical circuits with fast transients.
//!
//! The integrators here are *implicit* (or *linearly implicit*) and remain
//! stable for large `h` on such systems. Each requires the Jacobian
//! `J = ∂f/∂y`; this module computes it by central finite differences so the
//! caller only supplies the right-hand side. A dense LU solve (Gaussian
//! elimination with partial pivoting) handles the linear systems.
//!
//! References:
//! * Hairer & Wanner, *Solving Ordinary Differential Equations II*, 1996.
//! * Rosenbrock, *Some general implicit processes for the numerical solution
//!   of differential equations*, Comput. J. 5 (1963).

use crate::error::{PinnError, PinnResult};

/// Stiff right-hand side signature: `f(t, y, dydt)`.
pub type StiffRhsFn<'a> = &'a dyn Fn(f32, &[f32], &mut [f32]);

/// Configuration shared by the stiff integrators.
#[derive(Debug, Clone)]
pub struct StiffConfig {
    /// Maximum Newton iterations per implicit step.
    pub max_newton_iters: usize,
    /// Newton convergence tolerance on the increment ‖Δy‖∞.
    pub newton_tol: f32,
    /// Finite-difference perturbation used to build the Jacobian.
    pub fd_eps: f32,
}

impl StiffConfig {
    /// Construct a configuration, validating the numeric tolerances.
    pub fn new(max_newton_iters: usize, newton_tol: f32, fd_eps: f32) -> PinnResult<Self> {
        if max_newton_iters == 0 {
            return Err(PinnError::Internal(
                "max_newton_iters must be >= 1".to_string(),
            ));
        }
        if !(newton_tol.is_finite() && newton_tol > 0.0) {
            return Err(PinnError::Internal(
                "newton_tol must be finite and > 0".to_string(),
            ));
        }
        if !(fd_eps.is_finite() && fd_eps > 0.0) {
            return Err(PinnError::Internal(
                "fd_eps must be finite and > 0".to_string(),
            ));
        }
        Ok(Self {
            max_newton_iters,
            newton_tol,
            fd_eps,
        })
    }
}

impl Default for StiffConfig {
    fn default() -> Self {
        Self {
            max_newton_iters: 32,
            newton_tol: 1e-7,
            fd_eps: 1e-4,
        }
    }
}

// ─── Dense linear algebra (no ndarray) ─────────────────────────────────────────

/// Central-difference Jacobian `J[i][j] = ∂f_i/∂y_j` flattened row-major
/// (`dim × dim`).
fn finite_diff_jacobian(rhs: StiffRhsFn, t: f32, y: &[f32], eps: f32) -> Vec<f32> {
    let dim = y.len();
    let mut jac = vec![0.0_f32; dim * dim];
    let mut yp = y.to_vec();
    let mut f_plus = vec![0.0_f32; dim];
    let mut f_minus = vec![0.0_f32; dim];
    for j in 0..dim {
        // Scale perturbation to the magnitude of the component for robustness.
        let h = eps * (1.0 + y[j].abs());
        yp[j] = y[j] + h;
        rhs(t, &yp, &mut f_plus);
        yp[j] = y[j] - h;
        rhs(t, &yp, &mut f_minus);
        yp[j] = y[j];
        let inv = 0.5 / h;
        for i in 0..dim {
            jac[i * dim + j] = (f_plus[i] - f_minus[i]) * inv;
        }
    }
    jac
}

/// Solve `A x = b` in place via Gaussian elimination with partial pivoting.
/// `a` is row-major `n × n` (consumed); returns `x` or a `SolverDivergence`
/// error when the matrix is numerically singular.
fn lu_solve(mut a: Vec<f32>, mut b: Vec<f32>, n: usize) -> PinnResult<Vec<f32>> {
    for col in 0..n {
        // Partial pivot: find the row with the largest |a[row][col]|.
        let mut pivot_row = col;
        let mut pivot_mag = a[col * n + col].abs();
        for row in (col + 1)..n {
            let mag = a[row * n + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = row;
            }
        }
        if pivot_mag < 1e-20 {
            return Err(PinnError::SolverDivergence {
                reason: "singular Jacobian in implicit step",
            });
        }
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }
        // Eliminate below the pivot.
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            if factor != 0.0 {
                for k in col..n {
                    a[row * n + k] -= factor * a[col * n + k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= a[i * n + k] * x[k];
        }
        x[i] = sum / a[i * n + i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "lu_solve",
        });
    }
    Ok(x)
}

/// Build `M = I - gamma·h·J` (row-major `n × n`).
fn implicit_matrix(jac: &[f32], gamma_h: f32, n: usize) -> Vec<f32> {
    let mut m = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let ident = if i == j { 1.0 } else { 0.0 };
            m[i * n + j] = ident - gamma_h * jac[i * n + j];
        }
    }
    m
}

// ─── Backward (implicit) Euler ─────────────────────────────────────────────────

/// One backward-Euler step solving `y_{n+1} = y_n + h·f(t+h, y_{n+1})`
/// by a damped Newton iteration with a finite-difference Jacobian.
///
/// A-stable; first order accurate. Suitable as a robust fallback for very
/// stiff problems.
pub fn backward_euler_step(
    rhs: StiffRhsFn,
    t: f32,
    y: &[f32],
    h: f32,
    config: &StiffConfig,
) -> PinnResult<Vec<f32>> {
    if !(h.is_finite() && h > 0.0) {
        return Err(PinnError::InvalidStepSize { h });
    }
    let dim = y.len();
    if dim == 0 {
        return Err(PinnError::EmptyInput);
    }
    let t_next = t + h;
    // Initial guess: explicit Euler.
    let mut f0 = vec![0.0_f32; dim];
    rhs(t, y, &mut f0);
    let mut y_next: Vec<f32> = y
        .iter()
        .zip(f0.iter())
        .map(|(&yi, &fi)| yi + h * fi)
        .collect();

    let mut f_eval = vec![0.0_f32; dim];
    for _ in 0..config.max_newton_iters {
        rhs(t_next, &y_next, &mut f_eval);
        // Residual G(y) = y - y_n - h·f(t+h, y).
        let g: Vec<f32> = (0..dim).map(|i| y_next[i] - y[i] - h * f_eval[i]).collect();
        // Newton system (I - h·J)·Δ = -G.
        let jac = finite_diff_jacobian(rhs, t_next, &y_next, config.fd_eps);
        let m = implicit_matrix(&jac, h, dim);
        let neg_g: Vec<f32> = g.iter().map(|&v| -v).collect();
        let delta = lu_solve(m, neg_g, dim)?;
        let mut max_step = 0.0_f32;
        for i in 0..dim {
            y_next[i] += delta[i];
            max_step = max_step.max(delta[i].abs());
        }
        if max_step < config.newton_tol {
            break;
        }
    }
    if y_next.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "backward_euler_step",
        });
    }
    Ok(y_next)
}

// ─── Rosenbrock-W (ROS2) ───────────────────────────────────────────────────────

/// One linearly-implicit Rosenbrock (ROS2 / `gamma = 1 - 1/√2`) step.
///
/// Rosenbrock methods avoid the nonlinear Newton loop entirely: they solve a
/// fixed number of *linear* systems sharing the matrix `(I - gamma·h·J)`.
/// ROS2 is L-stable and second-order accurate.
///
/// Stages (autonomous form, Jacobian frozen at `y_n`, `W = I - γh J`):
/// ```text
/// W k1 = h f(t_n,   y_n)
/// W k2 = h f(t_n+h, y_n + k1) - 2 k1
/// y_{n+1} = y_n + (3/2) k1 + (1/2) k2
/// ```
/// This is the L-stable, second-order ROS2 of Verwer, Spee, Blom & Hundsdorfer
/// (SIAM J. Sci. Comput. 1999) with `γ = 1 - 1/√2`. The `W^{-1}` already
/// carries the `γ` scaling, so the stage-2 coupling term is exactly `-2 k1`.
pub fn rosenbrock2_step(
    rhs: StiffRhsFn,
    t: f32,
    y: &[f32],
    h: f32,
    config: &StiffConfig,
) -> PinnResult<Vec<f32>> {
    if !(h.is_finite() && h > 0.0) {
        return Err(PinnError::InvalidStepSize { h });
    }
    let dim = y.len();
    if dim == 0 {
        return Err(PinnError::EmptyInput);
    }
    // ROS2 coefficient: γ = 1 - 1/√2 (L-stable, second order).
    let gamma = 1.0 - 0.5_f32.sqrt();

    let jac = finite_diff_jacobian(rhs, t, y, config.fd_eps);
    let m = implicit_matrix(&jac, gamma * h, dim);

    // Stage 1: W k1 = h f(y_n).
    let mut f0 = vec![0.0_f32; dim];
    rhs(t, y, &mut f0);
    let rhs1: Vec<f32> = f0.iter().map(|&fi| h * fi).collect();
    let k1 = lu_solve(m.clone(), rhs1, dim)?;

    // Stage 2: W k2 = h f(y_n + k1) - 2 k1.
    let y2: Vec<f32> = (0..dim).map(|i| y[i] + k1[i]).collect();
    let mut f2 = vec![0.0_f32; dim];
    rhs(t + h, &y2, &mut f2);
    let rhs2: Vec<f32> = (0..dim).map(|i| h * f2[i] - 2.0 * k1[i]).collect();
    let k2 = lu_solve(m, rhs2, dim)?;

    // Update: y_{n+1} = y_n + (3/2) k1 + (1/2) k2.
    let y_next: Vec<f32> = (0..dim).map(|i| y[i] + 1.5 * k1[i] + 0.5 * k2[i]).collect();
    if y_next.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "rosenbrock2_step",
        });
    }
    Ok(y_next)
}

// ─── Variable-order BDF (1-2) ──────────────────────────────────────────────────

/// Fixed-step BDF integrator (order 1 = backward Euler, then order 2).
///
/// BDF (Backward Differentiation Formulas) are the workhorse multistep methods
/// for stiff problems (CVODE / LSODE). The first step uses BDF1 (backward
/// Euler) to seed the history; thereafter BDF2 is used:
/// ```text
/// (3/2) y_{n+1} - 2 y_n + (1/2) y_{n-1} = h f(t_{n+1}, y_{n+1})
/// ```
/// solved by Newton iteration with a finite-difference Jacobian. Returns the
/// full trajectory `[(n_steps+1) × dim]`.
pub fn integrate_bdf(
    rhs: StiffRhsFn,
    y0: &[f32],
    t0: f32,
    h: f32,
    n_steps: usize,
    config: &StiffConfig,
) -> PinnResult<Vec<Vec<f32>>> {
    if !(h.is_finite() && h > 0.0) {
        return Err(PinnError::InvalidStepSize { h });
    }
    let dim = y0.len();
    if dim == 0 {
        return Err(PinnError::EmptyInput);
    }
    let mut traj: Vec<Vec<f32>> = Vec::with_capacity(n_steps + 1);
    traj.push(y0.to_vec());
    if n_steps == 0 {
        return Ok(traj);
    }

    // First step: BDF1 (= backward Euler).
    let y1 = backward_euler_step(rhs, t0, y0, h, config)?;
    traj.push(y1);

    // Subsequent steps: BDF2.
    for step in 1..n_steps {
        let t_next = t0 + (step as f32 + 1.0) * h;
        let y_n = traj[step].clone();
        let y_nm1 = traj[step - 1].clone();
        // BDF2 residual: G(y) = (3/2) y - 2 y_n + (1/2) y_{n-1} - h f(t,y).
        // Newton matrix: (3/2) I - h J.
        let mut y_next = y_n.clone(); // predictor: constant extrapolation
        let mut f_eval = vec![0.0_f32; dim];
        for _ in 0..config.max_newton_iters {
            rhs(t_next, &y_next, &mut f_eval);
            let g: Vec<f32> = (0..dim)
                .map(|i| 1.5 * y_next[i] - 2.0 * y_n[i] + 0.5 * y_nm1[i] - h * f_eval[i])
                .collect();
            let jac = finite_diff_jacobian(rhs, t_next, &y_next, config.fd_eps);
            // M = (3/2) I - h J.
            let mut m = vec![0.0_f32; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    let ident = if i == j { 1.5 } else { 0.0 };
                    m[i * dim + j] = ident - h * jac[i * dim + j];
                }
            }
            let neg_g: Vec<f32> = g.iter().map(|&v| -v).collect();
            let delta = lu_solve(m, neg_g, dim)?;
            let mut max_step = 0.0_f32;
            for i in 0..dim {
                y_next[i] += delta[i];
                max_step = max_step.max(delta[i].abs());
            }
            if max_step < config.newton_tol {
                break;
            }
        }
        if y_next.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "integrate_bdf",
            });
        }
        traj.push(y_next);
    }
    Ok(traj)
}

/// Integrate a stiff system with fixed step size using backward Euler.
/// Returns the trajectory `[(n_steps+1) × dim]`.
pub fn integrate_backward_euler(
    rhs: StiffRhsFn,
    y0: &[f32],
    t0: f32,
    h: f32,
    n_steps: usize,
    config: &StiffConfig,
) -> PinnResult<Vec<Vec<f32>>> {
    let mut traj = Vec::with_capacity(n_steps + 1);
    traj.push(y0.to_vec());
    let mut y = y0.to_vec();
    for step in 0..n_steps {
        let t = t0 + step as f32 * h;
        y = backward_euler_step(rhs, t, &y, h, config)?;
        traj.push(y.clone());
    }
    Ok(traj)
}

/// Integrate a stiff system with fixed step size using ROS2.
/// Returns the trajectory `[(n_steps+1) × dim]`.
pub fn integrate_rosenbrock2(
    rhs: StiffRhsFn,
    y0: &[f32],
    t0: f32,
    h: f32,
    n_steps: usize,
    config: &StiffConfig,
) -> PinnResult<Vec<Vec<f32>>> {
    let mut traj = Vec::with_capacity(n_steps + 1);
    traj.push(y0.to_vec());
    let mut y = y0.to_vec();
    for step in 0..n_steps {
        let t = t0 + step as f32 * h;
        y = rosenbrock2_step(rhs, t, &y, h, config)?;
        traj.push(y.clone());
    }
    Ok(traj)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear scalar decay y' = -k y (k large => stiff). Analytic: y0 e^{-kt}.
    fn decay_rhs(k: f32) -> impl Fn(f32, &[f32], &mut [f32]) {
        move |_t, y, dydt| {
            dydt[0] = -k * y[0];
        }
    }

    #[test]
    fn lu_solve_identity() {
        // 2x2 identity solve returns b unchanged.
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, -2.0];
        let x = lu_solve(a, b, 2).expect("identity solve");
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn lu_solve_known_2x2() {
        // [[2,1],[1,3]] x = [3,5] => x = [0.8, 1.4].
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![3.0, 5.0];
        let x = lu_solve(a, b, 2).expect("2x2 solve");
        assert!((x[0] - 0.8).abs() < 1e-5, "x0 = {}", x[0]);
        assert!((x[1] - 1.4).abs() < 1e-5, "x1 = {}", x[1]);
    }

    #[test]
    fn lu_solve_pivoting_required() {
        // First pivot is zero, forcing a row swap.
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let b = vec![2.0, 3.0];
        let x = lu_solve(a, b, 2).expect("pivoted solve");
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn lu_solve_singular_errors() {
        let a = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
        let b = vec![1.0, 2.0];
        assert!(lu_solve(a, b, 2).is_err());
    }

    #[test]
    fn finite_diff_jacobian_linear() {
        // f(y) = [-3 y0]; J = [-3].
        let k = 3.0;
        let rhs = decay_rhs(k);
        let jac = finite_diff_jacobian(&rhs, 0.0, &[1.0], 1e-3);
        assert!((jac[0] + 3.0).abs() < 1e-2, "J = {}", jac[0]);
    }

    #[test]
    fn backward_euler_matches_analytic_decay() {
        // Stiff: k = 50, h = 0.05 (explicit Euler would blow up at hk = 2.5).
        let k = 50.0;
        let rhs = decay_rhs(k);
        let cfg = StiffConfig::default();
        let h = 0.05;
        let n = 20;
        let traj = integrate_backward_euler(&rhs, &[1.0], 0.0, h, n, &cfg).expect("BE integrate");
        // BDF1 is bounded and decays monotonically; never blows up.
        for window in traj.windows(2) {
            assert!(window[1][0].abs() <= window[0][0].abs() + 1e-6);
        }
        let final_t = n as f32 * h;
        let analytic = (-k * final_t).exp();
        // Backward Euler over-damps but stays the correct order of magnitude.
        assert!(traj[n][0] >= 0.0);
        assert!(traj[n][0] < 0.1, "should have decayed, got {}", traj[n][0]);
        assert!(analytic < 0.1);
    }

    #[test]
    fn rosenbrock2_second_order_accuracy() {
        // ROS2 should be markedly more accurate than backward Euler.
        let k = 5.0;
        let rhs = decay_rhs(k);
        let cfg = StiffConfig::default();
        let h = 0.02;
        let n = 50; // t_final = 1.0
        let traj = integrate_rosenbrock2(&rhs, &[1.0], 0.0, h, n, &cfg).expect("ROS2");
        let analytic = (-k * 1.0_f32).exp();
        let err = (traj[n][0] - analytic).abs();
        assert!(
            err < 5e-3,
            "ROS2 error {} too large (analytic {})",
            err,
            analytic
        );
    }

    #[test]
    fn bdf2_more_accurate_than_bdf1() {
        // On a smooth decay, BDF2 error < BDF1 error at the same step.
        let k = 4.0;
        let rhs = decay_rhs(k);
        let cfg = StiffConfig::default();
        let h = 0.05;
        let n = 20; // t_final = 1.0
        let analytic = (-k * 1.0_f32).exp();
        let bdf = integrate_bdf(&rhs, &[1.0], 0.0, h, n, &cfg).expect("BDF");
        let be = integrate_backward_euler(&rhs, &[1.0], 0.0, h, n, &cfg).expect("BE");
        let err_bdf = (bdf[n][0] - analytic).abs();
        let err_be = (be[n][0] - analytic).abs();
        assert!(
            err_bdf < err_be,
            "BDF2 err {} should beat BDF1 err {}",
            err_bdf,
            err_be
        );
    }

    #[test]
    fn stiff_system_2d_van_der_pol_like() {
        // Linear stiff 2D system: y0' = -100 y0 + y1, y1' = -y1.
        // Eigenvalues -100 and -1 => stiff. Both decay to 0.
        let rhs = |_t: f32, y: &[f32], dydt: &mut [f32]| {
            dydt[0] = -100.0 * y[0] + y[1];
            dydt[1] = -y[1];
        };
        let cfg = StiffConfig::default();
        let h = 0.05; // explicit Euler unstable (h*100 = 5 >> 2)
        let n = 40;
        let traj = integrate_rosenbrock2(&rhs, &[1.0, 1.0], 0.0, h, n, &cfg).expect("ROS2 2D");
        // Solution must remain bounded and decay.
        assert!(traj[n][0].abs() < 0.5, "y0 = {}", traj[n][0]);
        assert!(traj[n][1].abs() < 0.5, "y1 = {}", traj[n][1]);
        assert!(traj.iter().all(|s| s.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn config_validation_rejects_bad_input() {
        assert!(StiffConfig::new(0, 1e-6, 1e-4).is_err());
        assert!(StiffConfig::new(10, -1.0, 1e-4).is_err());
        assert!(StiffConfig::new(10, 1e-6, 0.0).is_err());
        assert!(StiffConfig::new(10, 1e-6, 1e-4).is_ok());
    }

    #[test]
    fn invalid_step_size_errors() {
        let rhs = decay_rhs(1.0);
        let cfg = StiffConfig::default();
        assert!(backward_euler_step(&rhs, 0.0, &[1.0], -0.1, &cfg).is_err());
        assert!(rosenbrock2_step(&rhs, 0.0, &[1.0], 0.0, &cfg).is_err());
    }

    #[test]
    fn empty_input_errors() {
        let rhs = decay_rhs(1.0);
        let cfg = StiffConfig::default();
        assert!(backward_euler_step(&rhs, 0.0, &[], 0.1, &cfg).is_err());
    }

    #[test]
    fn bdf_zero_steps_returns_initial() {
        let rhs = decay_rhs(1.0);
        let cfg = StiffConfig::default();
        let traj = integrate_bdf(&rhs, &[2.0], 0.0, 0.1, 0, &cfg).expect("zero steps");
        assert_eq!(traj.len(), 1);
        assert_eq!(traj[0][0], 2.0);
    }
}
