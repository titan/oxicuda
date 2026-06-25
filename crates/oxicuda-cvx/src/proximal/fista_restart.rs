//! FISTA with adaptive restart (O'Donoghue & Candès 2015).
//!
//! Plain FISTA (Beck & Teboulle 2009) accelerates proximal gradient from `O(1/k)`
//! to `O(1/k²)` via Nesterov momentum, but the momentum makes the objective
//! **non-monotone**: it can overshoot and oscillate, especially on strongly-convex
//! problems where the optimal restart period depends on the unknown condition
//! number.  *Adaptive restart* resets the momentum (`t ← 1`, `y ← x`) whenever a
//! cheap heuristic detects that momentum has begun to hurt, recovering the linear
//! convergence rate of accelerated gradient on strongly-convex objectives without
//! knowing `μ` or `L`.
//!
//! Two restart criteria are provided ([`RestartRule`]):
//!
//! * **Function restart** — restart when `F(x_{k+1}) > F(x_k)`, i.e. the composite
//!   objective `F = f + g` increases.  Requires one extra objective evaluation per
//!   step.
//! * **Gradient restart** — restart when the *gradient-mapping* makes an obtuse
//!   angle with the last step, `⟨y_k − x_{k+1}, x_{k+1} − x_k⟩ > 0`.  This is the
//!   restart of O'Donoghue–Candès §3.2: it needs no extra objective evaluation
//!   (the quantity `y_k − x_{k+1}` is, up to the step size, the gradient mapping
//!   at `y_k`) and is generally the more robust choice.
//!
//! The composite step is the standard forward-backward (ISTA) step with optional
//! backtracking on the smooth part `f`:
//!
//! ```text
//!   x_{k+1} = prox_{s g}( y_k − s ∇f(y_k) ),
//!   t_{k+1} = ½(1 + √(1 + 4 t_k²)),
//!   y_{k+1} = x_{k+1} + ((t_k − 1)/t_{k+1}) (x_{k+1} − x_k).
//! ```
//!
//! # References
//!
//! - B. O'Donoghue & E. Candès (2015), "Adaptive Restart for Accelerated Gradient
//!   Schemes", *Foundations of Computational Mathematics* 15(3):715-732.
//! - A. Beck & M. Teboulle (2009), "A Fast Iterative Shrinkage-Thresholding
//!   Algorithm for Linear Inverse Problems", *SIAM J. Imaging Sciences* 2(1):183-202.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Adaptive-restart criterion for [`fista_restart`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartRule {
    /// No restarting — equivalent to plain FISTA.
    None,
    /// Restart when the composite objective `F(x_{k+1}) > F(x_k)` increases.
    Function,
    /// Restart when `⟨y_k − x_{k+1}, x_{k+1} − x_k⟩ > 0` (gradient-mapping rule).
    Gradient,
}

/// Configuration for [`fista_restart`].
#[derive(Debug, Clone)]
pub struct FistaRestartConfig {
    /// Initial step size `s` (should satisfy `s ≤ 1/L` when `backtrack = false`).
    pub step: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stop when `‖x_{k+1} − x_k‖₂ < tol`.
    pub tol: f64,
    /// Restart criterion.
    pub rule: RestartRule,
    /// Enable backtracking line search on the smooth majorant of `f`.
    pub backtrack: bool,
    /// Backtracking shrink factor in `(0, 1)` (only used when `backtrack = true`).
    pub backtrack_shrink: f64,
}

impl Default for FistaRestartConfig {
    fn default() -> Self {
        Self {
            step: 1.0,
            max_iter: 1000,
            tol: 1e-9,
            rule: RestartRule::Gradient,
            backtrack: false,
            backtrack_shrink: 0.5,
        }
    }
}

impl FistaRestartConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`CvxError::InvalidParameter`] for a non-positive step, zero
    /// `max_iter`, or a backtracking factor outside `(0, 1)`.
    pub fn validate(&self) -> CvxResult<()> {
        if !(self.step.is_finite() && self.step > 0.0) {
            return Err(CvxError::InvalidParameter(format!(
                "step must be a positive finite number, got {}",
                self.step
            )));
        }
        if self.max_iter == 0 {
            return Err(CvxError::InvalidParameter("max_iter must be ≥ 1".into()));
        }
        if !(self.tol.is_finite() && self.tol >= 0.0) {
            return Err(CvxError::InvalidParameter(
                "tol must be non-negative and finite".into(),
            ));
        }
        if self.backtrack && !(self.backtrack_shrink > 0.0 && self.backtrack_shrink < 1.0) {
            return Err(CvxError::InvalidParameter(
                "backtrack_shrink must lie in (0, 1)".into(),
            ));
        }
        Ok(())
    }
}

/// Result returned by [`fista_restart`].
#[derive(Debug, Clone)]
pub struct FistaRestartResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Final composite objective `F(x) = f(x) + g(x)`.
    pub objective: f64,
    /// Iterations performed.
    pub iter: usize,
    /// Number of momentum restarts that fired.
    pub restarts: usize,
    /// Final step size (may differ from the initial value if backtracking was on).
    pub step: f64,
    /// Whether the `‖Δx‖ < tol` stopping rule fired.
    pub converged: bool,
    /// Composite-objective history (one entry per iteration).
    pub obj_history: Vec<f64>,
}

/// FISTA with adaptive restart.
///
/// Minimises `F(x) = f(x) + g(x)` where `f` is convex and differentiable (with a
/// Lipschitz gradient) and `g` is convex with a tractable proximal operator.
///
/// # Parameters
/// * `x0`     — starting point (non-empty).
/// * `f`      — smooth part `f(x)` (also used by the `Function` restart rule and
///   by backtracking).
/// * `grad_f` — gradient `∇f(x)`.
/// * `g`      — non-smooth part `g(x)`; only consulted to report the composite
///   objective and to drive the `Function` restart rule.
/// * `prox_g` — proximal operator `prox_{s g}(v)` for step size `s`.
/// * `config` — step / restart settings.
///
/// # Errors
/// * [`CvxError::EmptyInput`] when `x0` is empty.
/// * [`CvxError::InvalidParameter`] for an invalid configuration.
/// * [`CvxError::DimensionMismatch`] when a closure returns a wrongly-sized vector.
/// * [`CvxError::LineSearchFailed`] when backtracking underflows the step size.
pub fn fista_restart<F, G, GF, P>(
    x0: &[f64],
    f: F,
    grad_f: GF,
    g: G,
    prox_g: P,
    config: &FistaRestartConfig,
) -> CvxResult<FistaRestartResult>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    GF: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    G: Fn(&[f64]) -> CvxResult<f64>,
    P: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    config.validate()?;

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut y = x0.to_vec();
    let mut t = 1.0_f64;
    let mut step = config.step;

    // Composite objective F(x) = f(x) + g(x) of the current accepted iterate.
    let mut obj = f(&x)? + g(&x)?;
    let mut obj_history = Vec::with_capacity(config.max_iter);

    let mut restarts = 0usize;
    let mut iters = 0usize;
    let mut converged = false;

    for it in 0..config.max_iter {
        iters = it + 1;

        // ── forward-backward step at the extrapolation point y ──────────────
        let fy = f(&y)?;
        let gy = grad_f(&y)?;
        if gy.len() != n {
            return Err(CvxError::DimensionMismatch { a: gy.len(), b: n });
        }

        let mut s = step;
        let mut x_new: Vec<f64>;
        loop {
            let v: Vec<f64> = (0..n).map(|i| y[i] - s * gy[i]).collect();
            x_new = prox_g(&v, s)?;
            if x_new.len() != n {
                return Err(CvxError::DimensionMismatch {
                    a: x_new.len(),
                    b: n,
                });
            }
            if !config.backtrack {
                break;
            }
            // Sufficient-decrease majorant: f(x⁺) ≤ f(y) + ⟨∇f(y), x⁺−y⟩ + ‖x⁺−y‖²/(2s).
            let f_new = f(&x_new)?;
            let mut dot_g = 0.0_f64;
            let mut sq = 0.0_f64;
            for i in 0..n {
                let d = x_new[i] - y[i];
                dot_g += gy[i] * d;
                sq += d * d;
            }
            if f_new <= fy + dot_g + sq / (2.0 * s) + 1e-12 {
                step = s;
                break;
            }
            s *= config.backtrack_shrink;
            if s < 1e-300 {
                return Err(CvxError::LineSearchFailed(
                    "fista_restart: step underflowed during backtracking".into(),
                ));
            }
        }

        // ── composite objective at the momentum trial point ────────────────
        let mut obj_new = f(&x_new)? + g(&x_new)?;

        // ── restart test ────────────────────────────────────────────────────
        let do_restart = match config.rule {
            RestartRule::None => false,
            RestartRule::Function => obj_new > obj + 1e-12,
            RestartRule::Gradient => {
                // ⟨y − x⁺, x⁺ − x⟩ > 0  (gradient-mapping vs. progress direction).
                let mut ip = 0.0_f64;
                for i in 0..n {
                    ip += (y[i] - x_new[i]) * (x_new[i] - x[i]);
                }
                ip > 0.0
            }
        };

        if do_restart {
            restarts += 1;
            t = 1.0;
            if config.rule == RestartRule::Function {
                // Discard the uphill momentum point and take a plain proximal-
                // gradient (ISTA) step from the last accepted iterate x.  With a
                // valid step `s ≤ 1/L` this is guaranteed monotone descent, so the
                // recorded objective sequence never increases.
                let gx = grad_f(&x)?;
                if gx.len() != n {
                    return Err(CvxError::DimensionMismatch { a: gx.len(), b: n });
                }
                let v: Vec<f64> = (0..n).map(|i| x[i] - step * gx[i]).collect();
                let x_ista = prox_g(&v, step)?;
                if x_ista.len() != n {
                    return Err(CvxError::DimensionMismatch {
                        a: x_ista.len(),
                        b: n,
                    });
                }
                obj_new = f(&x_ista)? + g(&x_ista)?;
                x_new = x_ista;
            }
            // Restart momentum: next extrapolation point coincides with x_new.
            y = x_new.clone();
        } else {
            let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
            let beta = (t - 1.0) / t_new;
            y = (0..n)
                .map(|i| x_new[i] + beta * (x_new[i] - x[i]))
                .collect();
            t = t_new;
        }

        // ── accept the new iterate ──────────────────────────────────────────
        let diff: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let d_nrm = norm2(&diff);

        x = x_new;
        obj = obj_new;
        obj_history.push(obj);

        if d_nrm < config.tol {
            converged = true;
            break;
        }
    }

    Ok(FistaRestartResult {
        x,
        objective: obj,
        iter: iters,
        restarts,
        step,
        converged,
        obj_history,
    })
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::prox_ops::l1::{prox_l1, soft_threshold};

    // Smooth diagonal quadratic f(x) = ½ xᵀ diag(d) x − bᵀ x with d_i > 0
    // (so L = max d_i, μ = min d_i, minimiser x*_i = b_i / d_i). A spread of
    // d_i makes the problem ill-conditioned, which is exactly where Nesterov
    // momentum overshoots and the restart heuristics earn their keep.
    fn quad_diag(
        diag: Vec<f64>,
        b: Vec<f64>,
    ) -> (
        impl Fn(&[f64]) -> CvxResult<f64>,
        impl Fn(&[f64]) -> CvxResult<Vec<f64>>,
    ) {
        let d_f = diag.clone();
        let b_f = b.clone();
        let f = move |x: &[f64]| -> CvxResult<f64> {
            let mut v = 0.0_f64;
            for i in 0..x.len() {
                v += 0.5 * d_f[i] * x[i] * x[i] - b_f[i] * x[i];
            }
            Ok(v)
        };
        let d_g = diag;
        let b_g = b;
        let gf = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok((0..x.len()).map(|i| d_g[i] * x[i] - b_g[i]).collect())
        };
        (f, gf)
    }

    fn zero_g(x: &[f64]) -> CvxResult<f64> {
        let _ = x;
        Ok(0.0)
    }

    #[test]
    fn gradient_restart_solves_lasso() {
        // min ½‖x − d‖² + λ‖x‖₁ ⇒ x* = soft_threshold(d, λ).
        let d = vec![3.0_f64, -2.0, 0.5, 4.0, -0.1];
        let lambda = 1.0;
        let (f, gf) = {
            let d_c = d.clone();
            let f = move |x: &[f64]| -> CvxResult<f64> {
                Ok((0..x.len()).map(|i| 0.5 * (x[i] - d_c[i]).powi(2)).sum())
            };
            let d_g = d.clone();
            let gf = move |x: &[f64]| -> CvxResult<Vec<f64>> {
                Ok((0..x.len()).map(|i| x[i] - d_g[i]).collect())
            };
            (f, gf)
        };
        let g = {
            let d_g = lambda;
            move |x: &[f64]| -> CvxResult<f64> { Ok(d_g * x.iter().map(|v| v.abs()).sum::<f64>()) }
        };
        let prox = move |v: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l1(v, s * lambda) };
        let cfg = FistaRestartConfig {
            step: 1.0,
            max_iter: 2000,
            tol: 1e-12,
            rule: RestartRule::Gradient,
            ..Default::default()
        };
        let res = fista_restart(&[0.0; 5], &f, &gf, &g, &prox, &cfg).expect("solves");
        for i in 0..d.len() {
            let want = soft_threshold(d[i], lambda);
            assert!(
                (res.x[i] - want).abs() < 1e-5,
                "x[{i}]={} want {want}",
                res.x[i]
            );
        }
    }

    #[test]
    fn restart_beats_plain_fista_on_ill_conditioned() {
        // Ill-conditioned strongly-convex quadratic: condition number κ ≈ 1000.
        // Plain FISTA oscillates badly here (the canonical O'Donoghue–Candès
        // setting); adaptive restart recovers the linear rate and converges in
        // far fewer iterations to x*_i = b_i / d_i.
        let mut rng = LcgRng::new(20260621);
        let n = 40;
        // Geometrically spaced eigenvalues in [1e-3, 1] ⇒ κ ≈ 1000, L = 1.
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                10f64.powf(-3.0 * (1.0 - t)) // 1e-3 … 1.0
            })
            .collect();
        let b: Vec<f64> = (0..n).map(|_| rng.next_range(-1.0, 1.0)).collect();
        let l_max = diag.iter().cloned().fold(0.0_f64, f64::max);
        let x_star: Vec<f64> = (0..n).map(|i| b[i] / diag[i]).collect();
        let (f, gf) = quad_diag(diag, b);

        let base = FistaRestartConfig {
            step: 1.0 / l_max,
            max_iter: 20000,
            tol: 1e-9,
            backtrack: false,
            ..Default::default()
        };
        let prox = |v: &[f64], _s: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };

        let cfg_none = FistaRestartConfig {
            rule: RestartRule::None,
            ..base.clone()
        };
        let plain = fista_restart(&vec![0.0; n], &f, &gf, &zero_g, &prox, &cfg_none).expect("ok");

        let cfg_grad = FistaRestartConfig {
            rule: RestartRule::Gradient,
            ..base.clone()
        };
        let grad = fista_restart(&vec![0.0; n], &f, &gf, &zero_g, &prox, &cfg_grad).expect("ok");

        let cfg_fun = FistaRestartConfig {
            rule: RestartRule::Function,
            ..base
        };
        let fun = fista_restart(&vec![0.0; n], &f, &gf, &zero_g, &prox, &cfg_fun).expect("ok");

        // Both restart variants must actually restart and converge to x*.
        assert!(grad.restarts > 0, "gradient rule never restarted");
        assert!(fun.restarts > 0, "function rule never restarted");
        for variant in [&grad, &fun, &plain] {
            for i in 0..n {
                assert!(
                    (variant.x[i] - x_star[i]).abs() < 1e-3,
                    "wrong optimum at {i}"
                );
            }
        }
        // Restarted FISTA reaches the tolerance strictly faster than plain FISTA
        // on this ill-conditioned problem.
        assert!(
            grad.iter < plain.iter,
            "gradient-restart {} should beat plain {}",
            grad.iter,
            plain.iter
        );
        assert!(
            fun.iter < plain.iter,
            "function-restart {} should beat plain {}",
            fun.iter,
            plain.iter
        );
    }

    #[test]
    fn function_rule_objective_is_monotone() {
        // The function-restart rule restarts on *any* objective increase, so the
        // recorded composite-objective sequence must be non-increasing throughout.
        let mut rng = LcgRng::new(7);
        let n = 16;
        // Mildly ill-conditioned (κ = 100) so momentum is active.
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                10f64.powf(-2.0 * (1.0 - t))
            })
            .collect();
        let b: Vec<f64> = (0..n).map(|_| rng.next_range(-2.0, 2.0)).collect();
        let l_max = diag.iter().cloned().fold(0.0_f64, f64::max);
        let (f, gf) = quad_diag(diag, b);
        let cfg = FistaRestartConfig {
            step: 1.0 / l_max,
            max_iter: 2000,
            tol: 1e-12,
            rule: RestartRule::Function,
            ..Default::default()
        };
        let prox = |v: &[f64], _s: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let res = fista_restart(&vec![3.0; n], &f, &gf, &zero_g, &prox, &cfg).expect("ok");
        for w in res.obj_history.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "objective increased: {} → {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn backtracking_finds_step_without_known_lipschitz() {
        // Pass a too-large initial step and let backtracking shrink it.
        let d = vec![1.0_f64, -1.0, 2.0];
        let (f, gf) = {
            let d_c = d.clone();
            let f = move |x: &[f64]| -> CvxResult<f64> {
                Ok((0..x.len()).map(|i| 0.5 * (x[i] - d_c[i]).powi(2)).sum())
            };
            let d_g = d.clone();
            let gf = move |x: &[f64]| -> CvxResult<Vec<f64>> {
                Ok((0..x.len()).map(|i| x[i] - d_g[i]).collect())
            };
            (f, gf)
        };
        let cfg = FistaRestartConfig {
            step: 100.0, // way bigger than 1/L = 1
            max_iter: 1000,
            tol: 1e-10,
            rule: RestartRule::Gradient,
            backtrack: true,
            backtrack_shrink: 0.5,
        };
        let prox = |v: &[f64], _s: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let res = fista_restart(&[0.0; 3], &f, &gf, &zero_g, &prox, &cfg).expect("ok");
        assert!(res.step <= 1.0 + 1e-9, "step not reduced: {}", res.step);
        for i in 0..d.len() {
            assert!((res.x[i] - d[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn rejects_empty_and_bad_config() {
        let f = |_: &[f64]| Ok(0.0);
        let gf = |x: &[f64]| Ok(vec![0.0; x.len()]);
        let prox = |v: &[f64], _s: f64| Ok(v.to_vec());
        let cfg = FistaRestartConfig::default();
        let r = fista_restart(&[], &f, &gf, &zero_g, &prox, &cfg);
        assert!(matches!(r, Err(CvxError::EmptyInput)), "{r:?}");

        let bad = FistaRestartConfig {
            step: -1.0,
            ..Default::default()
        };
        let r2 = fista_restart(&[1.0], &f, &gf, &zero_g, &prox, &bad);
        assert!(matches!(r2, Err(CvxError::InvalidParameter(_))), "{r2:?}");
    }
}
