//! Singly Diagonally Implicit Runge-Kutta (SDIRK) methods for stiff ODEs.
//!
//! All diagonal Butcher tableau entries are equal (= γ), so every implicit
//! stage equation can be solved with the same factored Jacobian.  Here we use
//! fixed-point (Picard) iteration for the implicit stage solves, which is
//! suitable when the ODE is not too stiff or the step size is moderate.
//!
//! Two methods are provided:
//! - **SDIRK2** (Alexander 1977) — 2nd order, L-stable, 2-stage.
//! - **SDIRK3** (Crouzeix 1975) — 3rd order, A-stable, 2-stage.

use crate::error::{PdeError, PdeResult};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for SDIRK implicit stage solves.
#[derive(Debug, Clone, Copy)]
pub struct SdirkConfig {
    /// Fixed-point iteration tolerance: iteration stops when
    /// `|Y^{k+1} − Y^k|_∞ < tol`. Default: `1e-12`.
    pub tol: f64,
    /// Maximum fixed-point iterations per stage. Default: `100`.
    pub max_iter: usize,
}

impl Default for SdirkConfig {
    fn default() -> Self {
        Self {
            tol: 1.0e-12,
            max_iter: 100,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Validate common preconditions.
fn validate_sdirk(u: &[f64], dt: f64, cfg: &SdirkConfig) -> PdeResult<()> {
    if dt <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "dt".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    if u.is_empty() {
        return Err(PdeError::InvalidParameter {
            name: "u".to_string(),
            reason: "must be non-empty".to_string(),
        });
    }
    if cfg.tol <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "tol".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    if cfg.max_iter == 0 {
        return Err(PdeError::InvalidParameter {
            name: "max_iter".to_string(),
            reason: "must be at least 1".to_string(),
        });
    }
    Ok(())
}

/// Fixed-point iteration that solves `Y = rhs_const + h * gamma * f(t_s, Y)`.
///
/// `rhs_const[i]` is the already-accumulated part of the right-hand side that
/// does not depend on `Y`.  The initial guess is passed in `y_cur` (in-place).
///
/// Returns `Ok(())` with `y_cur` updated to the converged value, or
/// `Err(NotConverged)` if the iteration did not satisfy the tolerance within
/// `cfg.max_iter` steps.
fn fixed_point_stage<F>(
    y_cur: &mut [f64],
    rhs_const: &[f64],
    t_s: f64,
    h: f64,
    gamma: f64,
    f: &F,
    cfg: &SdirkConfig,
) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let d = rhs_const.len();
    let mut y_new = vec![0.0_f64; d];
    let mut residual = f64::INFINITY;

    for iter in 0..cfg.max_iter {
        let fval = f(t_s, y_cur);
        for i in 0..d {
            y_new[i] = rhs_const[i] + h * gamma * fval[i];
        }
        residual = y_cur
            .iter()
            .zip(y_new.iter())
            .map(|(a, b)| (b - a).abs())
            .fold(0.0_f64, f64::max);
        // copy y_new → y_cur
        y_cur.copy_from_slice(&y_new);
        if residual < cfg.tol {
            let _ = iter; // satisfy the borrow checker / clippy
            return Ok(());
        }
    }
    Err(PdeError::NotConverged {
        iter: cfg.max_iter,
        residual,
    })
}

// ── SDIRK2 ────────────────────────────────────────────────────────────────────

/// γ for SDIRK2: `1 − 1/√2 ≈ 0.2928932…`  (Alexander 1977, 2-stage, L-stable)
const GAMMA2: f64 = 1.0 - std::f64::consts::FRAC_1_SQRT_2;

/// One SDIRK2 (Alexander 1977) step — 2nd order, L-stable.
///
/// Butcher tableau  (γ = 1 − 1/√2):
/// ```text
/// γ   | γ    0
/// 1   | 1-γ  γ
/// ────|─────────
///     | 1-γ  γ
/// ```
///
/// Stage 1: `Y₁ = u + h·γ·f(t + γh, Y₁)`  (fixed-point)
/// Stage 2: `Y₂ = u + h·(1−γ)·k₁ + h·γ·f(t + h, Y₂)` where `k₁ = f(t+γh, Y₁)`
/// `u_{n+1} = Y₂`
///
/// # Errors
/// Returns [`PdeError::InvalidParameter`] for invalid `dt`, empty `u`, or bad
/// `cfg`; [`PdeError::NotConverged`] if a stage's fixed-point fails.
pub fn sdirk2_step<F>(u: &mut [f64], t: f64, dt: f64, f: F, cfg: &SdirkConfig) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_sdirk(u, dt, cfg)?;
    let d = u.len();
    let gamma = GAMMA2;

    // ── Stage 1: Y1 = u + h*γ*f(t + γ*h, Y1) ──────────────────────────────
    let t_s1 = t + gamma * dt;
    // rhs_const = u  (the part independent of Y1)
    let rhs1: Vec<f64> = u.to_vec();
    let mut y1 = rhs1.clone(); // initial guess = u
    fixed_point_stage(&mut y1, &rhs1, t_s1, dt, gamma, &f, cfg)?;

    // k1 = f(t + γ*h, Y1)
    let k1 = f(t_s1, &y1);

    // ── Stage 2: Y2 = u + h*(1-γ)*k1 + h*γ*f(t + h, Y2) ──────────────────
    let t_s2 = t + dt;
    // rhs_const[i] = u[i] + h*(1-γ)*k1[i]
    let mut rhs2 = vec![0.0_f64; d];
    for i in 0..d {
        rhs2[i] = u[i] + dt * (1.0 - gamma) * k1[i];
    }
    // warm start from Y1
    let mut y2 = y1;
    fixed_point_stage(&mut y2, &rhs2, t_s2, dt, gamma, &f, cfg)?;

    // u_{n+1} = Y2
    u.copy_from_slice(&y2);
    Ok(())
}

/// Multiple SDIRK2 steps from `t = t0` to `t = t0 + n_steps * dt`.
///
/// # Errors
/// Same as [`sdirk2_step`] plus [`PdeError::InvalidParameter`] if `n_steps == 0`.
pub fn sdirk2<F>(
    u: &mut [f64],
    t0: f64,
    dt: f64,
    n_steps: usize,
    f: F,
    cfg: &SdirkConfig,
) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_sdirk(u, dt, cfg)?;
    if n_steps == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n_steps".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    let gamma = GAMMA2;
    let d = u.len();

    for step in 0..n_steps {
        let t = t0 + step as f64 * dt;

        // Stage 1
        let t_s1 = t + gamma * dt;
        let rhs1: Vec<f64> = u.to_vec();
        let mut y1 = rhs1.clone();
        fixed_point_stage(&mut y1, &rhs1, t_s1, dt, gamma, &f, cfg)?;
        let k1 = f(t_s1, &y1);

        // Stage 2
        let t_s2 = t + dt;
        let mut rhs2 = vec![0.0_f64; d];
        for i in 0..d {
            rhs2[i] = u[i] + dt * (1.0 - gamma) * k1[i];
        }
        let mut y2 = y1;
        fixed_point_stage(&mut y2, &rhs2, t_s2, dt, gamma, &f, cfg)?;

        u.copy_from_slice(&y2);
    }
    Ok(())
}

// ── SDIRK3 ────────────────────────────────────────────────────────────────────

/// One SDIRK3 (Crouzeix 1975) step — 3rd order, A-stable.
///
/// Butcher tableau  (γ = (3+√3)/6 ≈ 0.7887):
/// ```text
/// γ   | γ        0
/// 1-γ | 1−2γ     γ
/// ────|────────────
///     | 0.5      0.5
/// ```
///
/// Stage 1: `Y₁ = u + h·γ·f(t + γh, Y₁)`  (fixed-point)
/// Stage 2: `Y₂ = u + h·(1−2γ)·k₁ + h·γ·f(t+(1−γ)h, Y₂)`  where `k₁ = f(t+γh, Y₁)`
/// `u_{n+1} = u + h·0.5·k₁ + h·0.5·k₂`  where `k₂ = f(t+(1−γ)h, Y₂)`
///
/// # Errors
/// Returns [`PdeError::InvalidParameter`] for invalid inputs;
/// [`PdeError::NotConverged`] if a stage's fixed-point fails.
pub fn sdirk3_step<F>(u: &mut [f64], t: f64, dt: f64, f: F, cfg: &SdirkConfig) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_sdirk(u, dt, cfg)?;
    let d = u.len();
    let gamma3 = (3.0 + 3.0_f64.sqrt()) / 6.0;

    // ── Stage 1: Y1 = u + h*γ*f(t + γ*h, Y1) ──────────────────────────────
    let t_s1 = t + gamma3 * dt;
    let rhs1: Vec<f64> = u.to_vec();
    let mut y1 = rhs1.clone();
    fixed_point_stage(&mut y1, &rhs1, t_s1, dt, gamma3, &f, cfg)?;
    let k1 = f(t_s1, &y1);

    // ── Stage 2: Y2 = u + h*(1-2γ)*k1 + h*γ*f(t+(1-γ)*h, Y2) ─────────────
    let t_s2 = t + (1.0 - gamma3) * dt;
    let mut rhs2 = vec![0.0_f64; d];
    for i in 0..d {
        rhs2[i] = u[i] + dt * (1.0 - 2.0 * gamma3) * k1[i];
    }
    let mut y2 = u.to_vec(); // initial guess = u
    fixed_point_stage(&mut y2, &rhs2, t_s2, dt, gamma3, &f, cfg)?;
    let k2 = f(t_s2, &y2);

    // ── Update: u = u + h * 0.5 * k1 + h * 0.5 * k2 ───────────────────────
    for i in 0..d {
        u[i] += dt * 0.5 * k1[i] + dt * 0.5 * k2[i];
    }
    Ok(())
}

/// Multiple SDIRK3 steps from `t = t0` to `t = t0 + n_steps * dt`.
///
/// # Errors
/// Same as [`sdirk3_step`] plus [`PdeError::InvalidParameter`] if `n_steps == 0`.
pub fn sdirk3<F>(
    u: &mut [f64],
    t0: f64,
    dt: f64,
    n_steps: usize,
    f: F,
    cfg: &SdirkConfig,
) -> PdeResult<()>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    validate_sdirk(u, dt, cfg)?;
    if n_steps == 0 {
        return Err(PdeError::InvalidParameter {
            name: "n_steps".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    let d = u.len();
    let gamma3 = (3.0 + 3.0_f64.sqrt()) / 6.0;

    for step in 0..n_steps {
        let t = t0 + step as f64 * dt;

        // Stage 1
        let t_s1 = t + gamma3 * dt;
        let rhs1: Vec<f64> = u.to_vec();
        let mut y1 = rhs1.clone();
        fixed_point_stage(&mut y1, &rhs1, t_s1, dt, gamma3, &f, cfg)?;
        let k1 = f(t_s1, &y1);

        // Stage 2
        let t_s2 = t + (1.0 - gamma3) * dt;
        let mut rhs2 = vec![0.0_f64; d];
        for i in 0..d {
            rhs2[i] = u[i] + dt * (1.0 - 2.0 * gamma3) * k1[i];
        }
        let mut y2 = u.to_vec();
        fixed_point_stage(&mut y2, &rhs2, t_s2, dt, gamma3, &f, cfg)?;
        let k2 = f(t_s2, &y2);

        // Update
        for i in 0..d {
            u[i] += dt * 0.5 * k1[i] + dt * 0.5 * k2[i];
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn default_cfg() -> SdirkConfig {
        SdirkConfig::default()
    }

    // ── SDIRK2 error cases ───────────────────────────────────────────────────

    #[test]
    fn sdirk2_err_dt_zero() {
        let mut u = vec![1.0];
        let result = sdirk2_step(&mut u, 0.0, 0.0, |_, x| vec![-x[0]], &default_cfg());
        assert!(result.is_err());
    }

    #[test]
    fn sdirk2_err_empty() {
        let mut u: Vec<f64> = vec![];
        let result = sdirk2_step(&mut u, 0.0, 0.1, |_, _| vec![], &default_cfg());
        assert!(result.is_err());
    }

    // ── SDIRK2 correctness ───────────────────────────────────────────────────

    #[test]
    fn sdirk2_exponential_decay() {
        // du/dt = -u, u(0)=1 → u(1) = exp(-1).
        // SDIRK2 is 2nd order; with dt=0.01 the error is O(dt²) ≈ O(1e-4).
        let mut u = vec![1.0];
        let dt = 0.01;
        sdirk2(&mut u, 0.0, dt, 100, |_, x| vec![-x[0]], &default_cfg()).expect("ok");
        let expected = (-1.0_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1.0e-5,
            "u={}, expected={}",
            u[0],
            expected
        );
    }

    #[test]
    fn sdirk2_stiff_system() {
        // du/dt = -100*u, u(0)=1.
        // Fixed-point iteration for stage 1 requires |h*γ*100| < 1,
        // i.e., dt < 1/(100*γ) ≈ 0.034.  Use dt=0.01 so the iteration converges.
        // Explicit Euler with dt=0.01 would give growth factor (1 - 100*0.01) = -0.0 — stable
        // only barely; SDIRK2 stays robustly stable.
        let mut u = vec![1.0];
        let dt = 0.01;
        let n = 50usize; // t_final = 0.5
        sdirk2(
            &mut u,
            0.0,
            dt,
            n,
            |_, x| vec![-100.0 * x[0]],
            &default_cfg(),
        )
        .expect("ok");
        let t_final = dt * n as f64;
        let expected = (-100.0 * t_final).exp();
        assert!(
            (u[0] - expected).abs() < 1.0e-3,
            "stiff SDIRK2: u={}, expected={}",
            u[0],
            expected
        );
    }

    #[test]
    fn sdirk2_harmonic_oscillator() {
        // [u, v]' = [v, -u]; u(0)=1, v(0)=0 → u(2π) ≈ 1, v(2π) ≈ 0
        // Use small dt so 2nd-order SDIRK2 gives reasonable accuracy.
        let mut x = vec![1.0, 0.0];
        let dt = 0.001;
        let n = (2.0 * PI / dt).round() as usize;
        sdirk2(&mut x, 0.0, dt, n, |_, s| vec![s[1], -s[0]], &default_cfg()).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-2, "harmonic u after 2π: {}", x[0]);
        assert!(x[1].abs() < 1.0e-2, "harmonic v after 2π: {}", x[1]);
    }

    #[test]
    fn sdirk2_multi_step() {
        // sdirk2(n=10) == 10 × sdirk2_step
        let init = vec![1.0_f64];
        let dt = 0.01;
        let cfg = default_cfg();

        let mut u1 = init.clone();
        sdirk2(&mut u1, 0.0, dt, 10, |_, x| vec![-x[0]], &cfg).expect("ok");

        let mut u2 = init.clone();
        for k in 0..10usize {
            let t = k as f64 * dt;
            sdirk2_step(&mut u2, t, dt, |_, x| vec![-x[0]], &cfg).expect("ok");
        }

        assert!(
            (u1[0] - u2[0]).abs() < 1.0e-14,
            "multi vs loop: {} vs {}",
            u1[0],
            u2[0]
        );
    }

    #[test]
    fn sdirk2_order_check() {
        // For du/dt = -u, error at t=1 should scale as O(dt²).
        // Halving dt reduces error by ~4.
        let exact = (-1.0_f64).exp();
        let cfg = default_cfg();

        let mut u1 = vec![1.0];
        let dt1 = 0.05;
        sdirk2(&mut u1, 0.0, dt1, 20, |_, x| vec![-x[0]], &cfg).expect("ok");
        let err1 = (u1[0] - exact).abs();

        let mut u2 = vec![1.0];
        let dt2 = 0.025;
        sdirk2(&mut u2, 0.0, dt2, 40, |_, x| vec![-x[0]], &cfg).expect("ok");
        let err2 = (u2[0] - exact).abs();

        let ratio = err1 / err2;
        assert!(ratio > 3.0, "expected ≈4× reduction, got ratio={ratio:.2}");
    }

    #[test]
    fn sdirk2_converge_failure_low_max_iter() {
        // du/dt = -1000*u, dt=0.5, max_iter=1 → should fail to converge
        let mut u = vec![1.0];
        let cfg = SdirkConfig {
            tol: 1.0e-12,
            max_iter: 1,
        };
        let result = sdirk2_step(&mut u, 0.0, 0.5, |_, x| vec![-1000.0 * x[0]], &cfg);
        assert!(result.is_err(), "should fail to converge with max_iter=1");
    }

    #[test]
    fn sdirk2_zero_rhs() {
        // f=0 → u stays unchanged
        let init = vec![1.5, 2.71];
        let mut u = init.clone();
        let cfg = default_cfg();
        sdirk2(&mut u, 0.0, 0.1, 10, |_, _| vec![0.0, 0.0], &cfg).expect("ok");
        for i in 0..2 {
            assert!(
                (u[i] - init[i]).abs() < 1.0e-14,
                "u[{i}] changed from {} to {}",
                init[i],
                u[i]
            );
        }
    }

    // ── SDIRK3 error cases ───────────────────────────────────────────────────

    #[test]
    fn sdirk3_err_dt_zero() {
        let mut u = vec![1.0];
        let result = sdirk3_step(&mut u, 0.0, 0.0, |_, x| vec![-x[0]], &default_cfg());
        assert!(result.is_err());
    }

    // ── SDIRK3 correctness ───────────────────────────────────────────────────

    #[test]
    fn sdirk3_exponential_decay() {
        // du/dt = -u, u(0)=1 → u(1) = exp(-1); SDIRK3 is 3rd order.
        // Use dt=0.02 (50 steps) so 3rd-order accuracy is clearly visible.
        let mut u = vec![1.0];
        let dt = 0.02;
        sdirk3(&mut u, 0.0, dt, 50, |_, x| vec![-x[0]], &default_cfg()).expect("ok");
        let expected = (-1.0_f64).exp();
        assert!(
            (u[0] - expected).abs() < 1.0e-6,
            "u={}, expected={}",
            u[0],
            expected
        );
    }

    #[test]
    fn sdirk3_stiff_system() {
        // du/dt = -100*u, u(0)=1.
        // Fixed-point iteration for SDIRK3 stage 1 requires |h*γ3*100| < 1
        // where γ3 ≈ 0.789, so dt < 1/(100*0.789) ≈ 0.0127.  Use dt=0.005.
        let mut u = vec![1.0];
        let dt = 0.005;
        let n = 100usize; // t_final = 0.5
        sdirk3(
            &mut u,
            0.0,
            dt,
            n,
            |_, x| vec![-100.0 * x[0]],
            &default_cfg(),
        )
        .expect("ok");
        let t_final = dt * n as f64;
        let expected = (-100.0 * t_final).exp();
        assert!(
            (u[0] - expected).abs() < 1.0e-3,
            "stiff SDIRK3: u={}, expected={}",
            u[0],
            expected
        );
    }

    #[test]
    fn sdirk3_order_check() {
        // For du/dt = -u at t=1, error should scale as O(dt³).
        // Halving dt reduces error by ~8.
        let exact = (-1.0_f64).exp();
        let cfg = default_cfg();

        let mut u1 = vec![1.0];
        let dt1 = 0.1;
        sdirk3(&mut u1, 0.0, dt1, 10, |_, x| vec![-x[0]], &cfg).expect("ok");
        let err1 = (u1[0] - exact).abs();

        let mut u2 = vec![1.0];
        let dt2 = 0.05;
        sdirk3(&mut u2, 0.0, dt2, 20, |_, x| vec![-x[0]], &cfg).expect("ok");
        let err2 = (u2[0] - exact).abs();

        let ratio = err1 / err2;
        assert!(ratio > 5.0, "expected ≈8× reduction, got ratio={ratio:.2}");
    }

    #[test]
    fn sdirk3_vs_sdirk2_accuracy() {
        // For the same dt and the smooth du/dt = -u problem, SDIRK3 error < SDIRK2 error.
        // Use dt=0.05 so that both methods' fixed-point iterations converge.
        let exact = (-1.0_f64).exp();
        let dt = 0.05;
        let cfg = default_cfg();

        let mut u2 = vec![1.0];
        sdirk2(&mut u2, 0.0, dt, 20, |_, x| vec![-x[0]], &cfg).expect("ok");
        let err2 = (u2[0] - exact).abs();

        let mut u3 = vec![1.0];
        sdirk3(&mut u3, 0.0, dt, 20, |_, x| vec![-x[0]], &cfg).expect("ok");
        let err3 = (u3[0] - exact).abs();

        assert!(
            err3 < err2,
            "SDIRK3 error ({err3}) should be < SDIRK2 error ({err2})"
        );
    }

    #[test]
    fn sdirk3_harmonic_oscillator() {
        // [u, v]' = [v, -u]; u(0)=1, v(0)=0 → u(2π) ≈ 1, v(2π) ≈ 0.
        // SDIRK3 uses implicit fixed-point iteration; for oscillatory problems
        // the amplitude and phase errors accumulate.  With dt=0.001 and n=6284
        // steps the 3rd-order method returns close to the initial state.
        let mut x = vec![1.0, 0.0];
        let n = 6284usize; // ≈ 2π / 0.001
        let dt = 2.0 * PI / n as f64;
        sdirk3(&mut x, 0.0, dt, n, |_, s| vec![s[1], -s[0]], &default_cfg()).expect("ok");
        assert!(
            (x[0] - 1.0).abs() < 1.0e-3,
            "SDIRK3 harmonic u after 2π: {}",
            x[0]
        );
        assert!(x[1].abs() < 1.0e-3, "SDIRK3 harmonic v after 2π: {}", x[1]);
    }

    #[test]
    fn sdirk3_zero_rhs() {
        // f=0 → u stays unchanged
        let init = vec![1.0, 2.0, 3.0];
        let mut u = init.clone();
        let cfg = default_cfg();
        sdirk3(&mut u, 0.0, 0.1, 10, |_, _| vec![0.0, 0.0, 0.0], &cfg).expect("ok");
        for i in 0..3 {
            assert!(
                (u[i] - init[i]).abs() < 1.0e-14,
                "u[{i}] changed: {} → {}",
                init[i],
                u[i]
            );
        }
    }

    #[test]
    fn sdirk3_gamma_value() {
        // (3 + √3) / 6 ≈ 0.78867513...
        let gamma3 = (3.0 + 3.0_f64.sqrt()) / 6.0;
        assert!(
            (gamma3 - 0.788_675_134_594_812_9).abs() < 1.0e-14,
            "gamma3 = {gamma3}"
        );
    }

    #[test]
    fn sdirk_multi_dimension() {
        // 3-D system: each component decays independently.
        let mut u2 = vec![1.0, 2.0, 3.0];
        let cfg = default_cfg();
        sdirk2(&mut u2, 0.0, 0.1, 5, |_, x| vec![-x[0], -x[1], -x[2]], &cfg)
            .expect("SDIRK2 3-D ok");

        let mut u3 = vec![1.0, 2.0, 3.0];
        sdirk3(&mut u3, 0.0, 0.1, 5, |_, x| vec![-x[0], -x[1], -x[2]], &cfg)
            .expect("SDIRK3 3-D ok");

        // Both must produce finite, plausible values (not NaN/inf, and decaying)
        for i in 0..3 {
            assert!(u2[i].is_finite(), "SDIRK2 u[{i}] not finite");
            assert!(u3[i].is_finite(), "SDIRK3 u[{i}] not finite");
        }
    }
}
