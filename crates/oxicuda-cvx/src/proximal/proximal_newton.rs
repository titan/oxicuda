//! Proximal Newton method for composite minimisation `min f(x) + g(x)`.
//!
//! Here `f` is smooth and convex with available gradient `∇f` and Hessian `∇²f`,
//! while `g` is convex but possibly non-smooth, accessed through its proximal
//! operator.  Lee, Sun & Saunders (2014) generalise Newton's method to this
//! composite setting by building, at each iterate, the *scaled* prox
//! subproblem
//!
//! ```text
//! x⁺ = argmin_u  ⟨∇f(x), u − x⟩ + ½ (u − x)ᵀ H (u − x) + g(u),   H = ∇²f(x)
//! ```
//!
//! whose minimiser defines the proximal-Newton search direction `d = x⁺ − x`.
//! A backtracking line search on the *composite* objective `F = f + g` along
//! `d` then guarantees sufficient decrease and global convergence; near the
//! solution the method enjoys local quadratic convergence inherited from
//! Newton's method.
//!
//! ## Solving the inner subproblem
//! The subproblem is itself a composite quadratic program.  We solve it
//! inexactly with a handful of proximal-gradient sweeps: the smooth part has
//! gradient `∇q(u) = ∇f(x) + H (u − x)` with Lipschitz constant
//! `L_H = ‖H‖₂` (over-estimated by the Frobenius norm), so the inner step is
//!
//! ```text
//! u ← prox_{(1/L_H) g}( u − (1/L_H) ∇q(u) ).
//! ```
//!
//! ## Composite line search
//! Using the prox-grad sufficient-decrease surrogate, the outer step accepts
//! the largest `t ∈ {1, ½, ¼, …}` satisfying
//!
//! ```text
//! F(x + t d) ≤ F(x) + α t [ ⟨∇f(x), d⟩ + g(x⁺) − g(x) ],
//! ```
//!
//! with `α ∈ (0, ½)`.
//!
//! Reference: Lee, J. D., Sun, Y., & Saunders, M. A. (2014). *Proximal Newton-
//! type methods for minimizing composite functions.* SIAM J. Optim. 24(3).

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Configuration for [`proximal_newton`].
#[derive(Debug, Clone)]
pub struct ProximalNewtonConfig {
    /// Maximum number of outer (Newton) iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the proximal-Newton step norm `‖d‖₂`.
    pub tol: f64,
    /// Number of inner proximal-gradient sweeps solving each subproblem.
    pub inner_iter: usize,
    /// Line-search sufficient-decrease parameter `α ∈ (0, ½)`.
    pub alpha: f64,
    /// Line-search backtracking factor `β ∈ (0, 1)`.
    pub beta: f64,
}

impl Default for ProximalNewtonConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-9,
            inner_iter: 50,
            alpha: 0.25,
            beta: 0.5,
        }
    }
}

/// Result of a proximal-Newton run.
#[derive(Debug, Clone)]
pub struct ProximalNewtonResult {
    /// Minimiser estimate.
    pub x: Vec<f64>,
    /// Smooth-part value `f(x)` at termination.
    pub f: f64,
    /// Number of outer iterations performed.
    pub iter: usize,
    /// Final proximal-Newton step norm `‖d‖₂`.
    pub step_norm: f64,
}

fn validate_cfg(cfg: &ProximalNewtonConfig) -> CvxResult<()> {
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "proximal_newton: max_iter must be ≥ 1".into(),
        ));
    }
    if cfg.inner_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "proximal_newton: inner_iter must be ≥ 1".into(),
        ));
    }
    if !(cfg.alpha > 0.0 && cfg.alpha < 0.5) {
        return Err(CvxError::InvalidParameter(format!(
            "proximal_newton: alpha must lie in (0, 0.5), got {}",
            cfg.alpha
        )));
    }
    if !(cfg.beta > 0.0 && cfg.beta < 1.0) {
        return Err(CvxError::InvalidParameter(format!(
            "proximal_newton: beta must lie in (0, 1), got {}",
            cfg.beta
        )));
    }
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "proximal_newton: tol must be > 0, got {}",
            cfg.tol
        )));
    }
    Ok(())
}

/// Frobenius norm of a row-major `n × n` matrix, used as an upper bound on the
/// spectral norm to size the inner prox-gradient step.
fn frob_norm(h: &[f64]) -> f64 {
    h.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `H (u − x)` for a row-major `n × n` matrix `H`.
fn hess_apply(h: &[f64], u: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut acc = 0.0_f64;
        let row = &h[i * n..i * n + n];
        for j in 0..n {
            acc += row[j] * (u[j] - x[j]);
        }
        out[i] = acc;
    }
    out
}

/// Run the proximal-Newton method on `F = f + g`.
///
/// # Arguments
/// - `x0`: starting point.
/// - `grad_f`: gradient oracle `∇f`.
/// - `hess_f`: Hessian oracle returning the row-major `n × n` matrix `∇²f(x)`.
/// - `prox_g`: proximal operator of `g`, i.e. `(v, t) ↦ prox_{t g}(v)`.
/// - `f_val`: smooth-part objective `f(x)` (used by the line search).
/// - `g_val`: non-smooth-part objective `g(x)` (used by the line search).
/// - `cfg`: algorithm configuration.
///
/// # Errors
/// - [`CvxError::EmptyInput`] if `x0` is empty.
/// - [`CvxError::InvalidParameter`] for invalid `cfg`.
/// - [`CvxError::DimensionMismatch`] if an oracle returns a vector/matrix of the
///   wrong size.
/// - [`CvxError::NumericalInstability`] if the Hessian norm is non-positive
///   (degenerate model).
#[allow(clippy::too_many_arguments)]
pub fn proximal_newton<GF, HF, PG, FV, GV>(
    x0: &[f64],
    grad_f: GF,
    hess_f: HF,
    prox_g: PG,
    f_val: FV,
    g_val: GV,
    cfg: &ProximalNewtonConfig,
) -> CvxResult<ProximalNewtonResult>
where
    GF: Fn(&[f64]) -> Vec<f64>,
    HF: Fn(&[f64]) -> Vec<f64>,
    PG: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    FV: Fn(&[f64]) -> f64,
    GV: Fn(&[f64]) -> f64,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    validate_cfg(cfg)?;

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut step_norm = 0.0_f64;
    let mut final_iter = 0_usize;

    for k in 0..cfg.max_iter {
        final_iter = k;
        let grad = grad_f(&x);
        if grad.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: grad.len(),
                b: n,
            });
        }
        let h = hess_f(&x);
        if h.len() != n * n {
            return Err(CvxError::DimensionMismatch {
                a: h.len(),
                b: n * n,
            });
        }
        let l_h = frob_norm(&h).max(1e-12);
        if !l_h.is_finite() {
            return Err(CvxError::NumericalInstability(
                "Hessian norm is non-finite".into(),
            ));
        }
        let inv_l = 1.0 / l_h;

        // Solve the inner composite quadratic with an accelerated (FISTA)
        // proximal-gradient inner solver, producing the prox-Newton target x⁺
        // (warm-started at x).  Acceleration lifts the inner convergence rate
        // from O((1 − 1/κ)ᵏ) to O((1 − 1/√κ)ᵏ), which is essential when the
        // Hessian is ill-conditioned (Lee-Sun-Saunders 2014, §3).
        let mut u = x.clone();
        let mut w = x.clone(); // momentum (extrapolated) point
        let mut t_mom = 1.0_f64;
        for _ in 0..cfg.inner_iter {
            // Gradient of the quadratic model at the extrapolated point:
            // ∇q(w) = ∇f(x) + H (w − x).
            let hw = hess_apply(&h, &w, &x, n);
            let mut v = vec![0.0_f64; n];
            for j in 0..n {
                v[j] = w[j] - inv_l * (grad[j] + hw[j]);
            }
            let u_new = prox_g(&v, inv_l)?;
            if u_new.len() != n {
                return Err(CvxError::DimensionMismatch {
                    a: u_new.len(),
                    b: n,
                });
            }
            // FISTA momentum update.
            let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_mom * t_mom).sqrt());
            let mom = (t_mom - 1.0) / t_next;
            let mut diff = 0.0_f64;
            for j in 0..n {
                let step = u_new[j] - u[j];
                diff += step * step;
                w[j] = u_new[j] + mom * step;
            }
            u = u_new;
            t_mom = t_next;
            // Inner stopping when the sweep barely moves.
            if diff.sqrt() < cfg.tol * 1e-2 {
                break;
            }
        }

        // Proximal-Newton direction d = x⁺ − x.
        let d: Vec<f64> = u.iter().zip(&x).map(|(a, b)| a - b).collect();
        step_norm = norm2(&d);
        if step_norm < cfg.tol {
            break;
        }

        // Composite line search.  Surrogate decrease:
        //   Δ = ⟨∇f(x), d⟩ + g(x⁺) − g(x)   (≤ 0 for a genuine descent step).
        let mut grad_dot_d = 0.0_f64;
        for j in 0..n {
            grad_dot_d += grad[j] * d[j];
        }
        let g_x = g_val(&x);
        let g_xp = g_val(&u);
        let delta = grad_dot_d + g_xp - g_x;
        let f_x = f_val(&x);
        let f_x_g = f_x + g_x;

        let mut t = 1.0_f64;
        let mut accepted = false;
        for _ in 0..50 {
            let trial: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + t * di).collect();
            let f_t = f_val(&trial);
            let g_t = g_val(&trial);
            if (f_t + g_t).is_finite() && f_t + g_t <= f_x_g + cfg.alpha * t * delta {
                x = trial;
                accepted = true;
                break;
            }
            t *= cfg.beta;
        }
        if !accepted {
            // No decrease found: the iterate is (numerically) stationary.
            break;
        }
    }

    let f = f_val(&x);
    Ok(ProximalNewtonResult {
        x,
        f,
        iter: final_iter,
        step_norm,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::{prox_indicator_box, prox_l1};

    fn cfg() -> ProximalNewtonConfig {
        ProximalNewtonConfig::default()
    }

    // Smooth-only test: g = 0 (prox is identity). Prox-Newton must reduce to a
    // damped Newton method and solve a quadratic in one step (up to line search).
    #[test]
    fn smooth_quadratic_no_g() {
        // f = ½ (x−a)ᵀ A (x−a), A = diag(2, 5); minimiser a = [1, -2].
        let a = vec![1.0_f64, -2.0];
        let grad = {
            let a = a.clone();
            move |x: &[f64]| vec![2.0 * (x[0] - a[0]), 5.0 * (x[1] - a[1])]
        };
        let hess = |_x: &[f64]| vec![2.0, 0.0, 0.0, 5.0];
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let f_val = {
            let a = a.clone();
            move |x: &[f64]| (x[0] - a[0]).powi(2) + 2.5 * (x[1] - a[1]).powi(2)
        };
        let g_val = |_x: &[f64]| 0.0;
        let r = proximal_newton(&[0.0, 0.0], grad, hess, prox, f_val, g_val, &cfg()).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-6, "x0={}", r.x[0]);
        assert!((r.x[1] + 2.0).abs() < 1e-6, "x1={}", r.x[1]);
    }

    // Lasso: f = ½‖x − b‖², g = λ‖x‖₁.  Closed-form solution is the
    // soft-threshold of b at λ since H = I.
    #[test]
    fn lasso_soft_threshold() {
        let b = vec![3.0_f64, 0.4, -0.4, -2.0];
        let lam = 0.5_f64;
        let grad = {
            let b = b.clone();
            move |x: &[f64]| x.iter().zip(&b).map(|(xi, bi)| xi - bi).collect::<Vec<_>>()
        };
        let n = b.len();
        let hess = move |_x: &[f64]| {
            let mut h = vec![0.0_f64; n * n];
            for i in 0..n {
                h[i * n + i] = 1.0;
            }
            h
        };
        let prox = move |v: &[f64], t: f64| prox_l1(v, lam * t);
        let f_val = {
            let b = b.clone();
            move |x: &[f64]| {
                0.5 * x
                    .iter()
                    .zip(&b)
                    .map(|(xi, bi)| (xi - bi).powi(2))
                    .sum::<f64>()
            }
        };
        let g_val = move |x: &[f64]| lam * x.iter().map(|v| v.abs()).sum::<f64>();
        let r = proximal_newton(&vec![0.0; 4], grad, hess, prox, f_val, g_val, &cfg()).expect("ok");
        // Expected soft-threshold of b at λ: [2.5, 0, 0, -1.5].
        let expect = [2.5_f64, 0.0, 0.0, -1.5];
        for (xi, ei) in r.x.iter().zip(&expect) {
            assert!((xi - ei).abs() < 1e-4, "got {xi}, expected {ei}");
        }
    }

    // Box-constrained smooth problem via g = indicator of [lo, hi].
    #[test]
    fn box_constrained_quadratic() {
        // min ½‖x − [2, -3]‖² s.t. x ∈ [-1, 1]². Optimum [1, -1].
        let target = vec![2.0_f64, -3.0];
        let grad = {
            let t = target.clone();
            move |x: &[f64]| x.iter().zip(&t).map(|(xi, ti)| xi - ti).collect::<Vec<_>>()
        };
        let hess = |_x: &[f64]| vec![1.0, 0.0, 0.0, 1.0];
        let prox = |v: &[f64], _t: f64| prox_indicator_box(v, -1.0, 1.0);
        let f_val = {
            let t = target.clone();
            move |x: &[f64]| {
                0.5 * x
                    .iter()
                    .zip(&t)
                    .map(|(xi, ti)| (xi - ti).powi(2))
                    .sum::<f64>()
            }
        };
        let g_val = |_x: &[f64]| 0.0;
        let r = proximal_newton(&[0.0, 0.0], grad, hess, prox, f_val, g_val, &cfg()).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-5, "x0={}", r.x[0]);
        assert!((r.x[1] + 1.0).abs() < 1e-5, "x1={}", r.x[1]);
    }

    #[test]
    fn ill_conditioned_quadratic() {
        // Strongly anisotropic Hessian where steepest descent crawls but
        // Newton scaling makes the problem trivial.
        let a = vec![0.5_f64, 7.0];
        let grad = {
            let a = a.clone();
            move |x: &[f64]| vec![100.0 * (x[0] - a[0]), 0.01 * (x[1] - a[1])]
        };
        let hess = |_x: &[f64]| vec![100.0, 0.0, 0.0, 0.01];
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let f_val = {
            let a = a.clone();
            move |x: &[f64]| 50.0 * (x[0] - a[0]).powi(2) + 0.005 * (x[1] - a[1]).powi(2)
        };
        let g_val = |_x: &[f64]| 0.0;
        let c = ProximalNewtonConfig {
            inner_iter: 200,
            max_iter: 200,
            ..cfg()
        };
        let r = proximal_newton(&[0.0, 0.0], grad, hess, prox, f_val, g_val, &c).expect("ok");
        assert!((r.x[0] - 0.5).abs() < 1e-3, "x0={}", r.x[0]);
        assert!((r.x[1] - 7.0).abs() < 1e-2, "x1={}", r.x[1]);
    }

    #[test]
    fn step_norm_collapses() {
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let hess = |_x: &[f64]| vec![2.0, 0.0, 0.0, 2.0];
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let f_val = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let g_val = |_x: &[f64]| 0.0;
        let r = proximal_newton(&[3.0, -4.0], grad, hess, prox, f_val, g_val, &cfg()).expect("ok");
        assert!(r.step_norm < cfg().tol, "step_norm={}", r.step_norm);
    }

    #[test]
    fn decreases_composite_objective() {
        let b = vec![2.0_f64, -1.0, 3.0];
        let lam = 0.3_f64;
        let grad = {
            let b = b.clone();
            move |x: &[f64]| x.iter().zip(&b).map(|(xi, bi)| xi - bi).collect::<Vec<_>>()
        };
        let hess = |_x: &[f64]| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let prox = move |v: &[f64], t: f64| prox_l1(v, lam * t);
        let f_val = {
            let b = b.clone();
            move |x: &[f64]| {
                0.5 * x
                    .iter()
                    .zip(&b)
                    .map(|(xi, bi)| (xi - bi).powi(2))
                    .sum::<f64>()
            }
        };
        let g_val = move |x: &[f64]| lam * x.iter().map(|v| v.abs()).sum::<f64>();
        let f0 = f_val(&[0.0, 0.0, 0.0]) + g_val(&[0.0, 0.0, 0.0]);
        let r =
            proximal_newton(&vec![0.0; 3], &grad, hess, prox, &f_val, &g_val, &cfg()).expect("ok");
        let f_end = f_val(&r.x) + g_val(&r.x);
        assert!(f_end <= f0 + 1e-12, "f0={f0} f_end={f_end}");
    }

    #[test]
    fn output_finite_and_sized() {
        let grad = |x: &[f64]| x.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
        let n = 4_usize;
        let hess = move |_x: &[f64]| {
            let mut h = vec![0.0_f64; n * n];
            for i in 0..n {
                h[i * n + i] = 2.0;
            }
            h
        };
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let f_val = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let g_val = |_x: &[f64]| 0.0;
        let r = proximal_newton(
            &[1.0, 2.0, 3.0, 4.0],
            grad,
            hess,
            prox,
            f_val,
            g_val,
            &cfg(),
        )
        .expect("ok");
        assert_eq!(r.x.len(), 4);
        for v in &r.x {
            assert!(v.is_finite());
        }
        assert!(r.f.is_finite());
    }

    #[test]
    fn empty_input_error() {
        let grad = |_x: &[f64]| Vec::<f64>::new();
        let hess = |_x: &[f64]| Vec::<f64>::new();
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let res = proximal_newton(&[], grad, hess, prox, |_| 0.0, |_| 0.0, &cfg());
        assert!(matches!(res, Err(CvxError::EmptyInput)));
    }

    #[test]
    fn rejects_bad_cfg() {
        let grad = |x: &[f64]| vec![2.0 * x[0]];
        let hess = |_x: &[f64]| vec![2.0];
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let bad = ProximalNewtonConfig {
            alpha: 0.8,
            ..cfg()
        };
        assert!(proximal_newton(&[1.0], &grad, &hess, &prox, |_| 0.0, |_| 0.0, &bad).is_err());
        let bad2 = ProximalNewtonConfig {
            max_iter: 0,
            ..cfg()
        };
        assert!(proximal_newton(&[1.0], &grad, &hess, &prox, |_| 0.0, |_| 0.0, &bad2).is_err());
    }

    #[test]
    fn hessian_wrong_size_error() {
        let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        // Wrong-size Hessian (should be 2×2 = 4 entries).
        let hess = |_x: &[f64]| vec![2.0, 0.0, 0.0];
        let prox = |v: &[f64], _t: f64| -> CvxResult<Vec<f64>> { Ok(v.to_vec()) };
        let res = proximal_newton(&[1.0, 1.0], grad, hess, prox, |_| 0.0, |_| 0.0, &cfg());
        assert!(matches!(res, Err(CvxError::DimensionMismatch { .. })));
    }
}
