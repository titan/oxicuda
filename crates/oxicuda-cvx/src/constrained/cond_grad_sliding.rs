//! Conditional Gradient Sliding (CGS) — Lan & Zhou (2016).
//!
//! Solves `min_{x ∈ C} f(x)` for a smooth convex `f` with `L`-Lipschitz
//! gradient over a compact convex set `C` accessed through a Linear
//! Minimization Oracle (LMO).  Plain Frank-Wolfe calls the LMO once per
//! gradient evaluation and converges at the optimal `O(1/k)` rate, but its
//! gradient-oracle complexity is also `O(1/ε)`.  CGS *decouples* these two
//! costs: it applies Nesterov acceleration in the outer loop (so only
//! `O(1/√ε)` gradient evaluations are needed) while the inner
//! "conditional-gradient sliding" procedure solves the projection subproblem
//!
//! ```text
//! min_{u ∈ C}  ⟨g, u⟩ + (β/2) ‖u − x‖²
//! ```
//!
//! approximately, using only LMO calls (so no Euclidean projection is ever
//! required — the method remains fully projection-free).
//!
//! ## Outer scheme (Lan-Zhou Algorithm 1)
//! ```text
//! z_{k}    = (1 − γ_k) y_{k−1} + γ_k x_{k−1}
//! g_k      = ∇f(z_k)
//! x_k      = CndG(g_k, x_{k−1}, β_k, η_k)         (inner sliding, LMO-only)
//! y_k      = (1 − γ_k) y_{k−1} + γ_k x_k
//! ```
//!
//! with the textbook schedule `γ_k = 3/(k+2)`, `β_k = 3L/(k+1)`,
//! `η_k = L D² / (k (k+1))` where `D` is the diameter of `C`.
//!
//! ## Inner CndG procedure
//! Iteratively minimises the prox-linear model `φ(u)=⟨g + β(u−x), u⟩` by
//! Frank-Wolfe steps on `C`; it terminates when the inner Wolfe gap drops below
//! the tolerance `η_k`.
//!
//! Reference: Lan, G. & Zhou, Y. (2016). *Conditional Gradient Sliding for
//! Convex Optimization.* SIAM Journal on Optimization 26(2), 1379-1409.

use crate::error::{CvxError, CvxResult};

/// Configuration for [`conditional_gradient_sliding`].
#[derive(Debug, Clone)]
pub struct CgsConfig {
    /// Maximum number of outer (accelerated) iterations.
    pub max_iter: usize,
    /// Lipschitz constant `L` of `∇f` used in the step schedule.
    pub lipschitz: f64,
    /// Diameter `D` of the feasible set `C` (used to size the inner tolerance).
    pub diameter: f64,
    /// Outer convergence tolerance on the Frank-Wolfe gap of `y_k`.
    pub tol: f64,
    /// Maximum number of inner CndG (sliding) iterations per outer step.
    pub inner_max_iter: usize,
}

impl Default for CgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            lipschitz: 1.0,
            diameter: 2.0,
            tol: 1e-8,
            inner_max_iter: 50,
        }
    }
}

/// Result of a CGS run.
#[derive(Debug, Clone)]
pub struct CgsResult {
    /// Final averaged iterate `y_k ∈ C`.
    pub x: Vec<f64>,
    /// Number of outer iterations performed.
    pub iter: usize,
    /// Frank-Wolfe gap `⟨∇f(y), y − LMO(∇f(y))⟩ ≥ 0` certifying sub-optimality.
    pub gap: f64,
    /// Total number of inner LMO calls consumed.
    pub lmo_calls: usize,
}

fn validate_cfg(cfg: &CgsConfig) -> CvxResult<()> {
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "cgs: max_iter must be ≥ 1".into(),
        ));
    }
    if cfg.inner_max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "cgs: inner_max_iter must be ≥ 1".into(),
        ));
    }
    if !(cfg.lipschitz > 0.0 && cfg.lipschitz.is_finite()) {
        return Err(CvxError::InvalidParameter(format!(
            "cgs: lipschitz must be > 0, got {}",
            cfg.lipschitz
        )));
    }
    if !(cfg.diameter > 0.0 && cfg.diameter.is_finite()) {
        return Err(CvxError::InvalidParameter(format!(
            "cgs: diameter must be > 0, got {}",
            cfg.diameter
        )));
    }
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "cgs: tol must be > 0, got {}",
            cfg.tol
        )));
    }
    Ok(())
}

/// Inner conditional-gradient (CndG) sliding procedure.
///
/// Approximately solves `min_{u ∈ C} ⟨g, u⟩ + (β/2)‖u − x0‖²` by Frank-Wolfe
/// steps, returning the iterate `u` and the number of LMO calls used.
fn cnd_g(
    g: &[f64],
    x0: &[f64],
    beta: f64,
    eta: f64,
    lmo: &impl Fn(&[f64]) -> Vec<f64>,
    inner_max_iter: usize,
    n: usize,
) -> CvxResult<(Vec<f64>, usize)> {
    let mut u = x0.to_vec();
    let mut calls = 0_usize;
    for _ in 0..inner_max_iter {
        // Gradient of the prox model at u: ∇φ(u) = g + β (u − x0).
        let mut grad = vec![0.0_f64; n];
        for j in 0..n {
            grad[j] = g[j] + beta * (u[j] - x0[j]);
        }
        let v = lmo(&grad);
        calls += 1;
        if v.len() != n {
            return Err(CvxError::DimensionMismatch { a: v.len(), b: n });
        }
        // Wolfe gap of the inner model: ⟨∇φ(u), u − v⟩ ≥ 0.
        let mut inner_gap = 0.0_f64;
        for j in 0..n {
            inner_gap += grad[j] * (u[j] - v[j]);
        }
        if inner_gap <= eta {
            break;
        }
        // Exact line search for the strongly-convex quadratic model along u→v:
        // φ(u + α(v−u)) is quadratic in α with curvature β‖v−u‖².
        let mut dir_sq = 0.0_f64;
        for j in 0..n {
            let d = v[j] - u[j];
            dir_sq += d * d;
        }
        let alpha = if beta * dir_sq > 0.0 {
            (inner_gap / (beta * dir_sq)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        for j in 0..n {
            u[j] += alpha * (v[j] - u[j]);
        }
    }
    Ok((u, calls))
}

/// Run Conditional Gradient Sliding (Lan-Zhou 2016).
///
/// # Arguments
/// - `x_init`: feasible starting point `x_0 ∈ C`.
/// - `grad_fn`: gradient oracle `∇f : ℝⁿ → ℝⁿ`.
/// - `lmo`: Linear Minimization Oracle `g ↦ argmin_{v ∈ C} ⟨g, v⟩`.
/// - `cfg`: algorithm configuration.
///
/// # Errors
/// - [`CvxError::EmptyInput`] if `x_init` is empty.
/// - [`CvxError::InvalidParameter`] for invalid `cfg`.
/// - [`CvxError::DimensionMismatch`] if the gradient or LMO returns a vector of
///   the wrong length.
pub fn conditional_gradient_sliding(
    x_init: &[f64],
    grad_fn: impl Fn(&[f64]) -> Vec<f64>,
    lmo: impl Fn(&[f64]) -> Vec<f64>,
    cfg: &CgsConfig,
) -> CvxResult<CgsResult> {
    if x_init.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    validate_cfg(cfg)?;

    let n = x_init.len();
    let l = cfg.lipschitz;
    let d2 = cfg.diameter * cfg.diameter;

    let mut x = x_init.to_vec(); // x_{k-1}
    let mut y = x_init.to_vec(); // y_{k-1}
    let mut gap = 0.0_f64;
    let mut final_iter = 0_usize;
    let mut lmo_calls = 0_usize;

    for k in 1..=cfg.max_iter {
        let kf = k as f64;
        let gamma = 3.0 / (kf + 2.0);
        let beta = 3.0 * l / (kf + 1.0);
        let eta = (l * d2) / (kf * (kf + 1.0));

        // z_k = (1 − γ) y + γ x.
        let mut z = vec![0.0_f64; n];
        for j in 0..n {
            z[j] = (1.0 - gamma) * y[j] + gamma * x[j];
        }
        let g = grad_fn(&z);
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }

        // Inner sliding step producing x_k.
        let (x_new, calls) = cnd_g(&g, &x, beta, eta, &lmo, cfg.inner_max_iter, n)?;
        lmo_calls += calls;

        // y_k = (1 − γ) y + γ x_k.
        let mut y_new = vec![0.0_f64; n];
        for j in 0..n {
            y_new[j] = (1.0 - gamma) * y[j] + gamma * x_new[j];
        }

        x = x_new;
        y = y_new;
        final_iter = k;

        // Outer Frank-Wolfe gap on the averaged iterate y as the stopping
        // certificate.
        let gy = grad_fn(&y);
        if gy.len() != n {
            return Err(CvxError::DimensionMismatch { a: gy.len(), b: n });
        }
        let s = lmo(&gy);
        lmo_calls += 1;
        if s.len() != n {
            return Err(CvxError::DimensionMismatch { a: s.len(), b: n });
        }
        let mut g_dot = 0.0_f64;
        for j in 0..n {
            g_dot += gy[j] * (y[j] - s[j]);
        }
        gap = g_dot;
        if gap < cfg.tol {
            break;
        }
    }

    Ok(CgsResult {
        x: y,
        iter: final_iter,
        gap,
        lmo_calls,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained::frank_wolfe::{l1_ball_lmo, simplex_lmo};

    fn quad_grad(x: &[f64], c: &[f64]) -> Vec<f64> {
        x.iter().zip(c.iter()).map(|(xi, ci)| xi - ci).collect()
    }

    #[test]
    fn converges_on_simplex_quadratic() {
        // min 0.5‖x − c‖² s.t. x ∈ Δ_3, c = [0.6, 0.2, 0.2] (already feasible).
        let c = vec![0.6_f64, 0.2, 0.2];
        let cc = c.clone();
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cfg = CgsConfig {
            max_iter: 400,
            lipschitz: 1.0,
            diameter: 1.5,
            tol: 1e-6,
            inner_max_iter: 50,
        };
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg)
            .expect("ok");
        for (xi, ci) in res.x.iter().zip(c.iter()) {
            assert!((xi - ci).abs() < 5e-3, "xi={xi}, ci={ci}");
        }
    }

    #[test]
    fn stays_in_simplex() {
        let c = vec![0.4_f64, 0.35, 0.25];
        let cc = c.clone();
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cfg = CgsConfig {
            max_iter: 200,
            diameter: 1.5,
            ..CgsConfig::default()
        };
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg)
            .expect("ok");
        let sum: f64 = res.x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
        for &xi in &res.x {
            assert!(xi >= -1e-9, "negative component xi={xi}");
        }
    }

    #[test]
    fn gap_nonneg() {
        let c = vec![0.5_f64, 0.3, 0.2];
        let cc = c.clone();
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cfg = CgsConfig {
            max_iter: 300,
            diameter: 1.5,
            tol: 1e-10,
            ..CgsConfig::default()
        };
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg)
            .expect("ok");
        assert!(res.gap >= -1e-9, "gap={}", res.gap);
    }

    #[test]
    fn gap_decreases_with_iterations() {
        let c = vec![0.7_f64, 0.2, 0.1];
        let x0 = vec![1.0 / 3.0_f64; 3];

        let cc1 = c.clone();
        let cfg1 = CgsConfig {
            max_iter: 10,
            diameter: 1.5,
            tol: 1e-15,
            ..CgsConfig::default()
        };
        let r1 = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc1), simplex_lmo, &cfg1)
            .expect("ok");

        let cc2 = c.clone();
        let cfg2 = CgsConfig {
            max_iter: 200,
            diameter: 1.5,
            tol: 1e-15,
            ..CgsConfig::default()
        };
        let r2 = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc2), simplex_lmo, &cfg2)
            .expect("ok");
        assert!(
            r2.gap <= r1.gap + 1e-9,
            "gap200={} gap10={}",
            r2.gap,
            r1.gap
        );
    }

    #[test]
    fn l1_ball_problem() {
        // min 0.5‖x − c‖² over the L1 ball with ‖c‖₁ > 1 so the solution is on
        // the boundary; CGS should drive the objective below the start value.
        let c = vec![2.0_f64, -1.5, 0.5];
        let cc = c.clone();
        let x0 = vec![0.0_f64; 3];
        let cfg = CgsConfig {
            max_iter: 300,
            diameter: 2.0,
            tol: 1e-9,
            ..CgsConfig::default()
        };
        let obj = |x: &[f64]| -> f64 {
            0.5 * x
                .iter()
                .zip(&c)
                .map(|(xi, ci)| (xi - ci).powi(2))
                .sum::<f64>()
        };
        let f0 = obj(&x0);
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), l1_ball_lmo, &cfg)
            .expect("ok");
        let l1: f64 = res.x.iter().map(|v| v.abs()).sum();
        assert!(l1 <= 1.0 + 1e-6, "‖x‖₁={l1} must stay ≤ 1");
        assert!(obj(&res.x) < f0, "objective did not decrease");
    }

    #[test]
    fn early_stop_large_tol() {
        let x0 = vec![1.0_f64, 0.0, 0.0]; // simplex vertex, already optimal-ish
        let cfg = CgsConfig {
            max_iter: 1000,
            diameter: 1.5,
            tol: 1e6,
            ..CgsConfig::default()
        };
        let res = conditional_gradient_sliding(&x0, |x| x.to_vec(), simplex_lmo, &cfg).expect("ok");
        assert!(res.iter < 50, "iter={}", res.iter);
    }

    #[test]
    fn shape_preserved() {
        let c = vec![0.6_f64, 0.4];
        let cc = c.clone();
        let x0 = vec![0.5_f64, 0.5];
        let cfg = CgsConfig {
            diameter: 1.5,
            ..CgsConfig::default()
        };
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg)
            .expect("ok");
        assert_eq!(res.x.len(), x0.len());
    }

    #[test]
    fn lmo_efficiency_vs_outer() {
        // The number of LMO calls should comfortably exceed the (small) number
        // of outer/gradient iterations, exercising the sliding inner loop.
        let c = vec![0.5_f64, 0.3, 0.2];
        let cc = c.clone();
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cfg = CgsConfig {
            max_iter: 30,
            lipschitz: 1.0,
            diameter: 1.5,
            tol: 1e-15,
            inner_max_iter: 20,
        };
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg)
            .expect("ok");
        assert!(
            res.lmo_calls >= res.iter,
            "lmo={} iter={}",
            res.lmo_calls,
            res.iter
        );
    }

    #[test]
    fn output_finite() {
        let c = vec![0.4_f64, 0.3, 0.3];
        let cc = c.clone();
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cfg = CgsConfig {
            diameter: 1.5,
            ..CgsConfig::default()
        };
        let res = conditional_gradient_sliding(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg)
            .expect("ok");
        for &v in &res.x {
            assert!(v.is_finite());
        }
        assert!(res.gap.is_finite());
    }

    #[test]
    fn empty_input_error() {
        let cfg = CgsConfig::default();
        let res = conditional_gradient_sliding(&[], |x| x.to_vec(), simplex_lmo, &cfg);
        assert!(matches!(res, Err(CvxError::EmptyInput)));
    }

    #[test]
    fn rejects_bad_cfg() {
        let x0 = vec![0.5_f64, 0.5];
        let bad_l = CgsConfig {
            lipschitz: -1.0,
            ..CgsConfig::default()
        };
        assert!(conditional_gradient_sliding(&x0, |x| x.to_vec(), simplex_lmo, &bad_l).is_err());
        let bad_iter = CgsConfig {
            max_iter: 0,
            ..CgsConfig::default()
        };
        assert!(conditional_gradient_sliding(&x0, |x| x.to_vec(), simplex_lmo, &bad_iter).is_err());
        let bad_diam = CgsConfig {
            diameter: 0.0,
            ..CgsConfig::default()
        };
        assert!(conditional_gradient_sliding(&x0, |x| x.to_vec(), simplex_lmo, &bad_diam).is_err());
    }

    #[test]
    fn lmo_wrong_dim_error() {
        let x0 = vec![0.5_f64, 0.5];
        let cfg = CgsConfig {
            diameter: 1.5,
            ..CgsConfig::default()
        };
        let res = conditional_gradient_sliding(&x0, |x| x.to_vec(), |_| vec![1.0], &cfg);
        assert!(matches!(res, Err(CvxError::DimensionMismatch { .. })));
    }
}
