//! IMEX (implicit-explicit) additive Runge-Kutta schemes for split ODE systems.
//!
//! For an additively split right-hand side
//!
//! ```text
//! y' = f_E(t, y) + f_I(t, y)
//! ```
//!
//! where `f_E` is non-stiff (treated *explicitly*) and `f_I` is stiff (treated
//! *implicitly*), these additive Runge-Kutta (ARK) methods combine an explicit
//! Butcher tableau `(Â, b̂, ĉ)` with a singly-diagonally-implicit tableau
//! `(A, b, c)` that shares the abscissae `c = ĉ`. The implicit stage equations
//! are solved with a damped-free Newton iteration that uses a user-supplied
//! Jacobian of the stiff part, factorised by the crate's dense Gaussian
//! elimination.
//!
//! Two schemes are provided:
//! - **IMEX Euler** — first order (forward Euler on `f_E`, backward Euler on
//!   `f_I`); a single implicit solve per step.
//! - **ARS(2,2,2)** (Ascher, Ruuth & Spiteri 1997) — second order, L-stable in
//!   the implicit part, two implicit stages, stiffly accurate.
//!
//! Unlike [`crate::time::imex::imex_step`] (a first-order, *linear*, sparse-matrix
//! IMEX scheme using CG), this module handles a fully **nonlinear** stiff part
//! `f_I` through Newton iterations and offers a second-order method.
//!
//! # Reference
//! U. M. Ascher, S. J. Ruuth, R. J. Spiteri, *Implicit-explicit Runge-Kutta
//! methods for time-dependent partial differential equations*, Appl. Numer.
//! Math. 25 (1997), 151-167.

use crate::error::{PdeError, PdeResult};
use crate::spectral::chebyshev::gauss_solve_dense;
use crate::time::sdirk::SdirkConfig;

// ── Internal helpers ────────────────────────────────────────────────────────────

/// Validate the common preconditions shared by every stepper.
fn validate(u: &[f64], dt: f64, cfg: &SdirkConfig) -> PdeResult<()> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "dt".into(),
            reason: format!("must be a finite value > 0, got {dt}"),
        });
    }
    if u.is_empty() {
        return Err(PdeError::InvalidParameter {
            name: "u".into(),
            reason: "state vector must be non-empty".into(),
        });
    }
    if cfg.tol <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "tol".into(),
            reason: "must be positive".into(),
        });
    }
    if cfg.max_iter == 0 {
        return Err(PdeError::InvalidParameter {
            name: "max_iter".into(),
            reason: "must be at least 1".into(),
        });
    }
    Ok(())
}

/// Check that a closure returned a vector of the expected length.
fn check_len(got: usize, expected: usize) -> PdeResult<()> {
    if got != expected {
        return Err(PdeError::DimensionMismatch {
            a: got,
            b: expected,
        });
    }
    Ok(())
}

/// Solve the diagonally-implicit stage equation
///
/// ```text
/// Y = rhs + (dt·γ) · f_I(t_s, Y)
/// ```
///
/// for `Y` by Newton iteration. `y` carries the initial guess in and the
/// converged stage value out. The Newton system is
/// `(I − dt·γ·J_I) ΔY = −(Y − dt·γ·f_I − rhs)` with `J_I = jac_i(t_s, Y)`.
///
/// Returns [`PdeError::NotConverged`] if the inf-norm of the Newton update does
/// not drop below `cfg.tol` within `cfg.max_iter` iterations.
fn newton_stage<FI, JI>(
    y: &mut [f64],
    rhs: &[f64],
    t_s: f64,
    dt_gamma: f64,
    f_i: &FI,
    jac_i: &JI,
    cfg: &SdirkConfig,
) -> PdeResult<()>
where
    FI: Fn(f64, &[f64]) -> Vec<f64>,
    JI: Fn(f64, &[f64]) -> Vec<f64>,
{
    let d = rhs.len();
    let mut residual = f64::INFINITY;
    for _ in 0..cfg.max_iter {
        let fi = f_i(t_s, y);
        check_len(fi.len(), d)?;
        // g = Y − dt·γ·f_I − rhs   (negated for the RHS below)
        let mut neg_g = vec![0.0_f64; d];
        for i in 0..d {
            neg_g[i] = -(y[i] - dt_gamma * fi[i] - rhs[i]);
        }
        // A = I − dt·γ·J_I
        let jac = jac_i(t_s, y);
        check_len(jac.len(), d * d)?;
        let mut a = vec![0.0_f64; d * d];
        for r in 0..d {
            for c in 0..d {
                let mut v = -dt_gamma * jac[r * d + c];
                if r == c {
                    v += 1.0;
                }
                a[r * d + c] = v;
            }
        }
        let delta = gauss_solve_dense(&mut a, &mut neg_g, d)?;
        residual = 0.0;
        for i in 0..d {
            y[i] += delta[i];
            residual = residual.max(delta[i].abs());
        }
        if residual < cfg.tol {
            return Ok(());
        }
    }
    Err(PdeError::NotConverged {
        iter: cfg.max_iter,
        residual,
    })
}

// ── Driver struct ───────────────────────────────────────────────────────────────

/// Driver for IMEX additive Runge-Kutta time stepping.
///
/// Holds the Newton-solve configuration ([`SdirkConfig`], reused here for its
/// `{tol, max_iter}` pair) used by the implicit stage solves.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImexArk {
    /// Newton iteration controls for the implicit stages.
    pub cfg: SdirkConfig,
}

impl ImexArk {
    /// Build a driver with the default Newton configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a driver with a custom Newton configuration.
    #[must_use]
    pub fn with_config(cfg: SdirkConfig) -> Self {
        Self { cfg }
    }

    /// One **IMEX Euler** step (first order). Advances `u` in place by `dt`:
    ///
    /// ```text
    /// u_{n+1} = u_n + dt·f_E(t_n, u_n) + dt·f_I(t_{n+1}, u_{n+1}).
    /// ```
    ///
    /// # Errors
    /// [`PdeError::InvalidParameter`] for bad `dt`/`u`/`cfg`,
    /// [`PdeError::DimensionMismatch`] if a closure returns the wrong length, or
    /// [`PdeError::NotConverged`] if the implicit Newton solve fails.
    pub fn imex_euler_step<FE, FI, JI>(
        &self,
        u: &mut [f64],
        t: f64,
        dt: f64,
        f_e: FE,
        f_i: FI,
        jac_i: JI,
    ) -> PdeResult<()>
    where
        FE: Fn(f64, &[f64]) -> Vec<f64>,
        FI: Fn(f64, &[f64]) -> Vec<f64>,
        JI: Fn(f64, &[f64]) -> Vec<f64>,
    {
        validate(u, dt, &self.cfg)?;
        let d = u.len();
        let fe = f_e(t, u);
        check_len(fe.len(), d)?;
        // rhs = u_n + dt·f_E(t_n, u_n)
        let mut rhs = vec![0.0_f64; d];
        for i in 0..d {
            rhs[i] = u[i] + dt * fe[i];
        }
        let mut y = rhs.clone();
        newton_stage(&mut y, &rhs, t + dt, dt, &f_i, &jac_i, &self.cfg)?;
        u.copy_from_slice(&y);
        Ok(())
    }

    /// One **ARS(2,2,2)** step (second order, L-stable implicit part).
    ///
    /// With `γ = (2−√2)/2` and `δ = 1 − 1/(2γ)` the stages are
    ///
    /// ```text
    /// Y1 = u_n
    /// Y2 = u_n + dt·γ·f_E(t_n, Y1) + dt·γ·f_I(t_n+γ·dt, Y2)
    /// Y3 = u_n + dt·[δ·f_E(t_n,Y1) + (1−δ)·f_E(t_n+γ·dt, Y2)]
    ///          + dt·[(1−γ)·f_I(t_n+γ·dt, Y2) + γ·f_I(t_n+dt, Y3)]
    /// u_{n+1} = Y3   (stiffly accurate).
    /// ```
    ///
    /// # Errors
    /// As [`Self::imex_euler_step`].
    pub fn ars222_step<FE, FI, JI>(
        &self,
        u: &mut [f64],
        t: f64,
        dt: f64,
        f_e: FE,
        f_i: FI,
        jac_i: JI,
    ) -> PdeResult<()>
    where
        FE: Fn(f64, &[f64]) -> Vec<f64>,
        FI: Fn(f64, &[f64]) -> Vec<f64>,
        JI: Fn(f64, &[f64]) -> Vec<f64>,
    {
        validate(u, dt, &self.cfg)?;
        let d = u.len();
        let gamma = (2.0 - std::f64::consts::SQRT_2) / 2.0;
        let delta = 1.0 - 1.0 / (2.0 * gamma);

        // Stage 1 (explicit, trivial): Y1 = u_n.
        let y1 = u.to_vec();
        let fe1 = f_e(t, &y1);
        check_len(fe1.len(), d)?;

        // Stage 2 (implicit): Y2 = u_n + dt·γ·f_E(t,Y1) + dt·γ·f_I(t+γ·dt, Y2).
        let t2 = t + gamma * dt;
        let mut rhs2 = vec![0.0_f64; d];
        for i in 0..d {
            rhs2[i] = u[i] + dt * gamma * fe1[i];
        }
        let mut y2 = rhs2.clone();
        newton_stage(&mut y2, &rhs2, t2, dt * gamma, &f_i, &jac_i, &self.cfg)?;
        let fe2 = f_e(t2, &y2);
        check_len(fe2.len(), d)?;
        let fi2 = f_i(t2, &y2);
        check_len(fi2.len(), d)?;

        // Stage 3 (implicit): Y3 = rhs3 + dt·γ·f_I(t+dt, Y3).
        let t3 = t + dt;
        let mut rhs3 = vec![0.0_f64; d];
        for i in 0..d {
            rhs3[i] =
                u[i] + dt * (delta * fe1[i] + (1.0 - delta) * fe2[i]) + dt * (1.0 - gamma) * fi2[i];
        }
        let mut y3 = y2.clone();
        newton_stage(&mut y3, &rhs3, t3, dt * gamma, &f_i, &jac_i, &self.cfg)?;

        u.copy_from_slice(&y3);
        Ok(())
    }

    /// Integrate `n_steps` of IMEX Euler from `t0` to `t0 + n_steps·dt`.
    ///
    /// # Errors
    /// As [`Self::imex_euler_step`], plus [`PdeError::InvalidParameter`] when
    /// `n_steps == 0`.
    pub fn imex_euler_integrate<FE, FI, JI>(
        &self,
        u: &mut [f64],
        t0: f64,
        dt: f64,
        n_steps: usize,
        f_e: FE,
        f_i: FI,
        jac_i: JI,
    ) -> PdeResult<()>
    where
        FE: Fn(f64, &[f64]) -> Vec<f64>,
        FI: Fn(f64, &[f64]) -> Vec<f64>,
        JI: Fn(f64, &[f64]) -> Vec<f64>,
    {
        validate(u, dt, &self.cfg)?;
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be positive".into(),
            });
        }
        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            self.imex_euler_step(u, t, dt, &f_e, &f_i, &jac_i)?;
        }
        Ok(())
    }

    /// Integrate `n_steps` of ARS(2,2,2) from `t0` to `t0 + n_steps·dt`.
    ///
    /// # Errors
    /// As [`Self::ars222_step`], plus [`PdeError::InvalidParameter`] when
    /// `n_steps == 0`.
    pub fn ars222_integrate<FE, FI, JI>(
        &self,
        u: &mut [f64],
        t0: f64,
        dt: f64,
        n_steps: usize,
        f_e: FE,
        f_i: FI,
        jac_i: JI,
    ) -> PdeResult<()>
    where
        FE: Fn(f64, &[f64]) -> Vec<f64>,
        FI: Fn(f64, &[f64]) -> Vec<f64>,
        JI: Fn(f64, &[f64]) -> Vec<f64>,
    {
        validate(u, dt, &self.cfg)?;
        if n_steps == 0 {
            return Err(PdeError::InvalidParameter {
                name: "n_steps".into(),
                reason: "must be positive".into(),
            });
        }
        for step in 0..n_steps {
            let t = t0 + step as f64 * dt;
            self.ars222_step(u, t, dt, &f_e, &f_i, &jac_i)?;
        }
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Scalar split test problem: y' = a·y (explicit) + b·y (implicit), exact
    // solution y(t) = y0·exp((a+b)·t). Closures are capture-free / capture only
    // `Copy` values, hence themselves `Copy`, so they are passed by value and
    // may be reused across several calls.

    #[test]
    fn validation_errors() {
        let solver = ImexArk::new();
        let mut u = vec![1.0];
        let fe = |_t: f64, _y: &[f64]| vec![0.0];
        let fi = |_t: f64, y: &[f64]| vec![-y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-1.0];
        assert!(
            solver
                .imex_euler_step(&mut u, 0.0, 0.0, fe, fi, ji)
                .is_err()
        ); // dt=0
        let mut empty: Vec<f64> = vec![];
        assert!(
            solver
                .imex_euler_step(&mut empty, 0.0, 0.1, fe, fi, ji)
                .is_err()
        );
        assert!(
            solver
                .ars222_integrate(&mut u, 0.0, 0.1, 0, fe, fi, ji)
                .is_err()
        ); // n_steps=0
    }

    #[test]
    fn imex_euler_stiff_stays_bounded() {
        // y' = 1·y + (−50)·y, exact y(t) = exp(−49 t). Explicit Euler on the
        // −50 y term would need dt < 2/50 = 0.04 to be stable; here dt = 0.1 is
        // far past that, yet IMEX-Euler treats it implicitly and stays bounded.
        let solver = ImexArk::new();
        let fe = |_t: f64, y: &[f64]| vec![y[0]];
        let fi = |_t: f64, y: &[f64]| vec![-50.0 * y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-50.0];
        let mut u = vec![1.0];
        solver
            .imex_euler_integrate(&mut u, 0.0, 0.1, 20, fe, fi, ji)
            .expect("integrate");
        assert!(
            u[0].is_finite() && u[0] >= 0.0 && u[0] < 1.0,
            "u = {}",
            u[0]
        );
    }

    #[test]
    fn ars222_matches_exact_decay() {
        // Pure implicit decay y' = 0·y + (−1)·y → y(1) = exp(−1).
        let solver = ImexArk::new();
        let fe = |_t: f64, _y: &[f64]| vec![0.0];
        let fi = |_t: f64, y: &[f64]| vec![-y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-1.0];
        let mut u = vec![1.0];
        solver
            .ars222_integrate(&mut u, 0.0, 0.02, 50, fe, fi, ji)
            .expect("integrate");
        let expected = (-1.0_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1e-4,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    #[test]
    fn ars222_matches_split_exact() {
        // y' = 0.5·y + (−5)·y, exact y(1) = exp(−4.5).
        let solver = ImexArk::new();
        let fe = |_t: f64, y: &[f64]| vec![0.5 * y[0]];
        let fi = |_t: f64, y: &[f64]| vec![-5.0 * y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-5.0];
        let mut u = vec![1.0];
        solver
            .ars222_integrate(&mut u, 0.0, 0.01, 100, fe, fi, ji)
            .expect("integrate");
        let expected = (-4.5_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1e-4,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    #[test]
    fn ars222_second_order_in_dt() {
        // For the smooth split problem the global error should scale ~dt²;
        // halving dt cuts the error by ~4×.
        let solver = ImexArk::new();
        let fe = |_t: f64, y: &[f64]| vec![0.5 * y[0]];
        let fi = |_t: f64, y: &[f64]| vec![-2.0 * y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-2.0];
        let exact = (-1.5_f64).exp();

        let mut u1 = vec![1.0];
        solver
            .ars222_integrate(&mut u1, 0.0, 0.1, 10, fe, fi, ji)
            .expect("ok");
        let e1 = (u1[0] - exact).abs();

        let mut u2 = vec![1.0];
        solver
            .ars222_integrate(&mut u2, 0.0, 0.05, 20, fe, fi, ji)
            .expect("ok");
        let e2 = (u2[0] - exact).abs();

        let ratio = e1 / e2.max(1e-15);
        assert!(ratio > 3.0, "expected ≈4× reduction, got {ratio:.2}");
    }

    #[test]
    fn ars222_nonlinear_logistic() {
        // Implicit logistic y' = 0·y + y(1−y), y(0)=0.5, exact y(t)=1/(1+e^{−t}).
        // Exercises the Newton iteration through a nonlinear Jacobian 1 − 2y.
        let solver = ImexArk::new();
        let fe = |_t: f64, _y: &[f64]| vec![0.0];
        let fi = |_t: f64, y: &[f64]| vec![y[0] * (1.0 - y[0])];
        let ji = |_t: f64, y: &[f64]| vec![1.0 - 2.0 * y[0]];
        let mut u = vec![0.5];
        solver
            .ars222_integrate(&mut u, 0.0, 0.05, 40, fe, fi, ji)
            .expect("integrate");
        let t = 2.0_f64;
        let expected = 1.0 / (1.0 + (-t).exp());
        assert!(
            (u[0] - expected).abs() < 1e-3,
            "u = {}, exp = {}",
            u[0],
            expected
        );
    }

    #[test]
    fn step_matches_integrate_loop() {
        let solver = ImexArk::new();
        let fe = |_t: f64, y: &[f64]| vec![0.5 * y[0]];
        let fi = |_t: f64, y: &[f64]| vec![-3.0 * y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-3.0];
        let dt = 0.02;

        let mut u_loop = vec![1.0];
        for k in 0..10usize {
            solver
                .ars222_step(&mut u_loop, k as f64 * dt, dt, fe, fi, ji)
                .expect("ok");
        }
        let mut u_int = vec![1.0];
        solver
            .ars222_integrate(&mut u_int, 0.0, dt, 10, fe, fi, ji)
            .expect("ok");
        assert!((u_loop[0] - u_int[0]).abs() < 1e-14);
    }

    #[test]
    fn newton_not_converged_with_one_iteration() {
        // A single Newton iteration is not enough for the |ΔY|-based stopping
        // test to pass, so the implicit solve reports NotConverged.
        let cfg = SdirkConfig {
            tol: 1e-12,
            max_iter: 1,
        };
        let solver = ImexArk::with_config(cfg);
        let fe = |_t: f64, _y: &[f64]| vec![0.0];
        let fi = |_t: f64, y: &[f64]| vec![-10.0 * y[0]];
        let ji = |_t: f64, _y: &[f64]| vec![-10.0];
        let mut u = vec![1.0];
        let res = solver.imex_euler_step(&mut u, 0.0, 0.1, fe, fi, ji);
        assert!(matches!(res, Err(PdeError::NotConverged { .. })));
    }

    #[test]
    fn two_dimensional_system() {
        // Decoupled 2-D split system; each component decays as exp((a+b)t).
        let solver = ImexArk::new();
        let fe = |_t: f64, y: &[f64]| vec![0.1 * y[0], 0.2 * y[1]];
        let fi = |_t: f64, y: &[f64]| vec![-2.0 * y[0], -3.0 * y[1]];
        let ji = |_t: f64, _y: &[f64]| vec![-2.0, 0.0, 0.0, -3.0];
        let mut u = vec![1.0, 1.0];
        solver
            .ars222_integrate(&mut u, 0.0, 0.01, 100, fe, fi, ji)
            .expect("integrate");
        assert!((u[0] - (-1.9_f64).exp()).abs() < 1e-3, "u0 = {}", u[0]);
        assert!((u[1] - (-2.8_f64).exp()).abs() < 1e-3, "u1 = {}", u[1]);
    }
}
