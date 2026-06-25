//! Self-contained 1-D heat-equation application with adaptive time stepping.
//!
//! Solves `∂u/∂t = α ∂²u/∂x²` on `[x₀, x₁]` with Dirichlet endpoints, marching
//! in time with **backward Euler** (unconditionally stable, so the step size is
//! limited only by accuracy, never by a CFL condition) and an embedded
//! **step-doubling Richardson** error estimator that grows or shrinks `dt`
//! automatically.
//!
//! # Adaptive controller
//!
//! At each accepted state we form two estimates of `u(t + dt)`:
//!
//! * `u_big` — one backward-Euler step of size `dt`;
//! * `u_two` — two backward-Euler steps of size `dt/2`.
//!
//! Because backward Euler is first-order, the leading error of `u_two` is half
//! that of `u_big`, so the Richardson-extrapolated value `u* = 2 u_two − u_big`
//! is second-order accurate and `err = ‖u_two − u_big‖∞` estimates the local
//! truncation error of `u_two`. The classic elementary controller
//!
//! ```text
//! dt_new = safety · dt · (tol / err)^{1/2}
//! ```
//!
//! (exponent `1/(p+1) = 1/2` for the order-`p = 1` base method) accepts the
//! step when `err ≤ tol` and otherwise retries with the reduced `dt`. The
//! returned, extrapolated state `u*` is what is propagated, giving an
//! effectively second-order-accurate, error-controlled integrator.
//!
//! # References
//!
//! * E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential
//!   Equations I*, 2nd ed., Springer, 1993, §II.4 (step-size control).
//! * W. H. Press et al., *Numerical Recipes*, 3rd ed., CUP, 2007, §17.2
//!   (adaptive step-size control with step doubling).

use crate::error::{PdeError, PdeResult};
use crate::fdm::poisson_1d::thomas_solve;

/// Adaptive 1-D heat-equation solver with Dirichlet boundary conditions.
#[derive(Debug, Clone)]
pub struct HeatEquation {
    /// Thermal diffusivity `α > 0`.
    pub alpha: f64,
    /// Uniform grid spacing `dx > 0`.
    pub dx: f64,
    /// Number of grid nodes (`n ≥ 3`).
    pub n: usize,
    /// Fixed left endpoint value `u[0]`.
    pub left: f64,
    /// Fixed right endpoint value `u[n−1]`.
    pub right: f64,
}

/// Parameters controlling the adaptive time-stepping loop.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Absolute local-error tolerance per step (max-norm).
    pub tol: f64,
    /// Initial trial time step.
    pub dt_init: f64,
    /// Lower bound on the time step (guards against stalling).
    pub dt_min: f64,
    /// Upper bound on the time step.
    pub dt_max: f64,
    /// Safety factor in `(0, 1]` applied to the predicted step.
    pub safety: f64,
    /// Maximum factor by which `dt` may grow between accepted steps.
    pub max_grow: f64,
    /// Maximum number of accepted steps before giving up.
    pub max_steps: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            tol: 1.0e-6,
            dt_init: 1.0e-4,
            dt_min: 1.0e-12,
            dt_max: 1.0e-1,
            safety: 0.9,
            max_grow: 4.0,
            max_steps: 100_000,
        }
    }
}

/// Diagnostics returned by an adaptive integration to a target time.
#[derive(Debug, Clone)]
pub struct AdaptiveReport {
    /// Final solution vector at `t = t_final`.
    pub u: Vec<f64>,
    /// Final simulation time (equal to the requested target on success).
    pub t: f64,
    /// Number of accepted steps.
    pub accepted: usize,
    /// Number of rejected steps.
    pub rejected: usize,
    /// Time step in force at the end of the integration.
    pub last_dt: f64,
}

impl HeatEquation {
    /// Build a solver, validating `α > 0`, `dx > 0`, `n ≥ 3`, and finite data.
    ///
    /// # Errors
    ///
    /// [`PdeError::InvalidParameter`] / [`PdeError::InvalidGrid`] on bad inputs.
    pub fn new(alpha: f64, dx: f64, n: usize, left: f64, right: f64) -> PdeResult<Self> {
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "alpha".into(),
                reason: format!("diffusivity must be finite and > 0, got {alpha}"),
            });
        }
        if !(dx.is_finite() && dx > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dx".into(),
                reason: format!("grid spacing must be finite and > 0, got {dx}"),
            });
        }
        if n < 3 {
            return Err(PdeError::InvalidGrid(format!(
                "heat equation requires n >= 3, got {n}"
            )));
        }
        if !(left.is_finite() && right.is_finite()) {
            return Err(PdeError::InvalidParameter {
                name: "boundary".into(),
                reason: "Dirichlet values must be finite".into(),
            });
        }
        Ok(Self {
            alpha,
            dx,
            n,
            left,
            right,
        })
    }

    /// One backward-Euler step `(I − r A) u^{n+1} = u^n` (+ boundary terms),
    /// with `r = α·dt/dx²` and `A` the interior 1-D Laplacian.
    ///
    /// Returns a fresh vector; `u` is left unchanged. Endpoints of the result
    /// are clamped to the Dirichlet values.
    ///
    /// # Errors
    ///
    /// [`PdeError::ShapeMismatch`] if `u.len() != n`; propagates Thomas-solver
    /// failures.
    pub fn backward_euler_step(&self, u: &[f64], dt: f64) -> PdeResult<Vec<f64>> {
        if u.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![u.len()],
            });
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "dt".into(),
                reason: format!("must be finite and > 0, got {dt}"),
            });
        }
        let r = self.alpha * dt / (self.dx * self.dx);
        let m = self.n - 2;
        let mut sub = vec![-r; m];
        let mut diag = vec![1.0 + 2.0 * r; m];
        let mut sup = vec![-r; m];
        let mut rhs = vec![0.0_f64; m];
        rhs.copy_from_slice(&u[1..self.n - 1]);
        rhs[0] += r * self.left;
        rhs[m - 1] += r * self.right;
        sub[0] = 0.0;
        sup[m - 1] = 0.0;
        let interior = thomas_solve(&sub, &mut diag, &mut sup, &mut rhs)?;
        let mut next = vec![0.0_f64; self.n];
        next[0] = self.left;
        next[self.n - 1] = self.right;
        next[1..self.n - 1].copy_from_slice(&interior);
        Ok(next)
    }

    /// Take one *accepted* adaptive step from `u` (at time `t`).
    ///
    /// Returns `(u_new, t_new, dt_used, dt_next, rejections)` where `u_new` is
    /// the Richardson-extrapolated (second-order) state, `dt_used ≤ dt_try` is
    /// the step actually accepted, `dt_next` is the controller's suggestion for
    /// the next step, and `rejections` counts the retries inside this call.
    ///
    /// # Errors
    ///
    /// [`PdeError::NotConverged`] if the step is rejected down to `dt_min`
    /// without meeting `tol`; propagates solver failures.
    pub fn adaptive_step(
        &self,
        u: &[f64],
        t: f64,
        dt_try: f64,
        cfg: &AdaptiveConfig,
    ) -> PdeResult<(Vec<f64>, f64, f64, f64, usize)> {
        if u.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![u.len()],
            });
        }
        validate_config(cfg)?;
        let mut dt = dt_try.clamp(cfg.dt_min, cfg.dt_max);
        let mut rejections = 0_usize;
        loop {
            let u_big = self.backward_euler_step(u, dt)?;
            let u_half = self.backward_euler_step(u, 0.5 * dt)?;
            let u_two = self.backward_euler_step(&u_half, 0.5 * dt)?;
            let err = max_norm_diff(&u_big, &u_two);
            // Richardson extrapolation: u* = 2 u_two − u_big (order p+1 = 2).
            let mut u_star = vec![0.0_f64; self.n];
            for i in 0..self.n {
                u_star[i] = 2.0 * u_two[i] - u_big[i];
            }
            // Controller factor; guard the err = 0 case.
            let factor = if err > 0.0 {
                cfg.safety * (cfg.tol / err).sqrt()
            } else {
                cfg.max_grow
            };
            let factor = factor.clamp(0.2, cfg.max_grow);
            if err <= cfg.tol || dt <= cfg.dt_min * (1.0 + 1.0e-12) {
                let dt_used = dt;
                let dt_next = (dt * factor).clamp(cfg.dt_min, cfg.dt_max);
                return Ok((u_star, t + dt_used, dt_used, dt_next, rejections));
            }
            // Reject and shrink.
            rejections += 1;
            let dt_new = (dt * factor).clamp(cfg.dt_min, cfg.dt_max);
            if dt_new >= dt {
                // Cannot shrink further; force the floor.
                if (dt - cfg.dt_min).abs() < cfg.dt_min * 1.0e-9 {
                    return Err(PdeError::NotConverged {
                        iter: rejections,
                        residual: err,
                    });
                }
                dt = cfg.dt_min;
            } else {
                dt = dt_new;
            }
        }
    }

    /// Integrate the initial state `u0` to `t_final` with adaptive stepping.
    ///
    /// The final step is trimmed so the integration lands exactly on
    /// `t_final`. Endpoints of `u0` are overwritten by the Dirichlet values.
    ///
    /// # Errors
    ///
    /// [`PdeError::ShapeMismatch`] on a length mismatch; [`PdeError::NotConverged`]
    /// if `max_steps` is exceeded or a step stalls at `dt_min`.
    pub fn integrate(
        &self,
        u0: &[f64],
        t_final: f64,
        cfg: &AdaptiveConfig,
    ) -> PdeResult<AdaptiveReport> {
        if u0.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![u0.len()],
            });
        }
        if !(t_final.is_finite() && t_final > 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "t_final".into(),
                reason: format!("must be finite and > 0, got {t_final}"),
            });
        }
        validate_config(cfg)?;
        let mut u = u0.to_vec();
        u[0] = self.left;
        u[self.n - 1] = self.right;
        let mut t = 0.0_f64;
        let mut dt = cfg.dt_init.clamp(cfg.dt_min, cfg.dt_max);
        let mut accepted = 0_usize;
        let mut rejected = 0_usize;
        while t < t_final * (1.0 - 1.0e-12) {
            // Do not overshoot the target.
            let dt_try = dt.min(t_final - t);
            let (u_new, t_new, _dt_used, dt_next, rej) = self.adaptive_step(&u, t, dt_try, cfg)?;
            u = u_new;
            t = t_new;
            dt = dt_next;
            accepted += 1;
            rejected += rej;
            if accepted >= cfg.max_steps {
                return Err(PdeError::NotConverged {
                    iter: accepted,
                    residual: t_final - t,
                });
            }
        }
        Ok(AdaptiveReport {
            u,
            t,
            accepted,
            rejected,
            last_dt: dt,
        })
    }
}

/// Validate an [`AdaptiveConfig`].
fn validate_config(cfg: &AdaptiveConfig) -> PdeResult<()> {
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "tol".into(),
            reason: "must be positive and finite".into(),
        });
    }
    if !(cfg.dt_min > 0.0 && cfg.dt_min.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "dt_min".into(),
            reason: "must be positive and finite".into(),
        });
    }
    if !(cfg.dt_max >= cfg.dt_min && cfg.dt_max.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "dt_max".into(),
            reason: "must be finite and ≥ dt_min".into(),
        });
    }
    if !(cfg.safety > 0.0 && cfg.safety <= 1.0) {
        return Err(PdeError::InvalidParameter {
            name: "safety".into(),
            reason: "must be in (0, 1]".into(),
        });
    }
    if !(cfg.max_grow > 1.0 && cfg.max_grow.is_finite()) {
        return Err(PdeError::InvalidParameter {
            name: "max_grow".into(),
            reason: "must be finite and > 1".into(),
        });
    }
    if cfg.max_steps == 0 {
        return Err(PdeError::InvalidParameter {
            name: "max_steps".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    Ok(())
}

/// Max-norm of the elementwise difference of two equal-length vectors.
fn max_norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    /// Initial data `sin(π x)` on a unit-length domain with `n` nodes.
    fn sine_initial(n: usize) -> (Vec<f64>, f64) {
        let dx = 1.0 / (n as f64 - 1.0);
        let u: Vec<f64> = (0..n).map(|i| (PI * i as f64 * dx).sin()).collect();
        (u, dx)
    }

    #[test]
    fn single_backward_euler_step_runs() {
        let (u, dx) = sine_initial(21);
        let solver = HeatEquation::new(1.0, dx, 21, 0.0, 0.0).expect("ok");
        let next = solver.backward_euler_step(&u, 1.0e-3).expect("ok");
        assert_eq!(next.len(), 21);
        assert_eq!(next[0], 0.0);
        assert_eq!(next[20], 0.0);
        // Amplitude must decay (heat dissipates).
        let peak0 = u.iter().cloned().fold(0.0_f64, f64::max);
        let peak1 = next.iter().cloned().fold(0.0_f64, f64::max);
        assert!(peak1 < peak0, "peak0={peak0} peak1={peak1}");
    }

    #[test]
    fn adaptive_matches_analytic_decay() {
        // u(x,0)=sin(π x) ⇒ u(x,t)=sin(π x) exp(−π² α t).
        let n = 81;
        let (u0, dx) = sine_initial(n);
        let alpha = 1.0_f64;
        let solver = HeatEquation::new(alpha, dx, n, 0.0, 0.0).expect("ok");
        let t_final = 0.02_f64;
        let cfg = AdaptiveConfig {
            tol: 1.0e-7,
            dt_init: 1.0e-4,
            dt_max: 5.0e-3,
            ..AdaptiveConfig::default()
        };
        let rep = solver.integrate(&u0, t_final, &cfg).expect("integrate ok");
        assert!((rep.t - t_final).abs() < 1e-10, "t={}", rep.t);
        let decay = (-PI * PI * alpha * t_final).exp();
        let center = rep.u[n / 2];
        let exact = (PI * (n / 2) as f64 * dx).sin() * decay;
        assert!(
            (center - exact).abs() < 2e-3,
            "center={center} exact={exact}"
        );
    }

    #[test]
    fn adaptive_grows_step_for_smooth_decay() {
        // For a decaying smooth solution the controller should enlarge dt over
        // time (curvature in time decreases), so the final step exceeds the
        // initial trial step.
        let n = 41;
        let (u0, dx) = sine_initial(n);
        let solver = HeatEquation::new(1.0, dx, n, 0.0, 0.0).expect("ok");
        let cfg = AdaptiveConfig {
            tol: 1.0e-6,
            dt_init: 2.0e-5,
            dt_max: 1.0e-2,
            ..AdaptiveConfig::default()
        };
        let rep = solver.integrate(&u0, 0.05, &cfg).expect("ok");
        assert!(
            rep.last_dt > cfg.dt_init,
            "dt did not grow: last={} init={}",
            rep.last_dt,
            cfg.dt_init
        );
        assert!(rep.accepted >= 1);
    }

    #[test]
    fn steady_state_reaches_linear_profile() {
        // With nonzero, unequal Dirichlet data and zero forcing, the long-time
        // solution is the straight line between the endpoints.
        let n = 33;
        let dx = 1.0 / (n as f64 - 1.0);
        let solver = HeatEquation::new(1.0, dx, n, 1.0, 3.0).expect("ok");
        let u0 = vec![0.0_f64; n];
        let cfg = AdaptiveConfig {
            tol: 1.0e-6,
            dt_max: 0.2,
            ..AdaptiveConfig::default()
        };
        let rep = solver.integrate(&u0, 5.0, &cfg).expect("ok");
        for i in 0..n {
            let x = i as f64 * dx;
            let line = 1.0 + 2.0 * x; // left + (right−left)·x, domain length 1
            assert!(
                (rep.u[i] - line).abs() < 5e-3,
                "i={i} u={} line={line}",
                rep.u[i]
            );
        }
    }

    #[test]
    fn tighter_tolerance_costs_more_steps() {
        let n = 41;
        let (u0, dx) = sine_initial(n);
        let solver = HeatEquation::new(1.0, dx, n, 0.0, 0.0).expect("ok");
        let base = AdaptiveConfig {
            dt_init: 1.0e-4,
            dt_max: 1.0e-2,
            ..AdaptiveConfig::default()
        };
        let loose = AdaptiveConfig {
            tol: 1.0e-4,
            ..base.clone()
        };
        let tight = AdaptiveConfig {
            tol: 1.0e-8,
            ..base
        };
        let r_loose = solver.integrate(&u0, 0.05, &loose).expect("ok");
        let r_tight = solver.integrate(&u0, 0.05, &tight).expect("ok");
        assert!(
            r_tight.accepted >= r_loose.accepted,
            "tight {} should need ≥ loose {}",
            r_tight.accepted,
            r_loose.accepted
        );
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(HeatEquation::new(-1.0, 0.1, 11, 0.0, 0.0).is_err());
        assert!(HeatEquation::new(1.0, 0.0, 11, 0.0, 0.0).is_err());
        assert!(HeatEquation::new(1.0, 0.1, 2, 0.0, 0.0).is_err());
        assert!(HeatEquation::new(1.0, 0.1, 11, f64::NAN, 0.0).is_err());
    }

    #[test]
    fn rejects_bad_config() {
        let solver = HeatEquation::new(1.0, 0.1, 11, 0.0, 0.0).expect("ok");
        let u0 = vec![0.0_f64; 11];
        let bad = AdaptiveConfig {
            tol: 0.0,
            ..AdaptiveConfig::default()
        };
        assert!(matches!(
            solver.integrate(&u0, 1.0, &bad),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rejects_wrong_length() {
        let solver = HeatEquation::new(1.0, 0.1, 11, 0.0, 0.0).expect("ok");
        assert!(matches!(
            solver.backward_euler_step(&[0.0; 5], 1e-3),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }
}
