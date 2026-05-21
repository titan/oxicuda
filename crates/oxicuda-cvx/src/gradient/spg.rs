//! Spectral Projected Gradient (SPG) method.
//!
//! Implements the nonmonotone spectral projected gradient method of Birgin,
//! Martínez & Raydan (2000), "Nonmonotone Spectral Projected Gradient Methods on
//! Convex Sets" (SIAM J. Optim. 10(4)).
//!
//! SPG minimises a smooth convex function `f` over a closed convex set `C`
//! (supplied implicitly through a Euclidean projection closure `P_C`). It combines
//! two ingredients:
//!
//! 1. **Barzilai-Borwein spectral step length** — the scalar
//!    `α = (sᵀs) / (sᵀy)` (clamped to `[α_min, α_max]`, with a safeguard
//!    `α = α_max` when `sᵀy ≤ 0`) approximates the inverse Hessian along the most
//!    recent step `s = x_{k+1} − x_k`, `y = ∇f(x_{k+1}) − ∇f(x_k)`.
//! 2. **Nonmonotone (GLL) line search** — the reference value
//!    `f_ref = max_{0 ≤ j < min(k+1, M)} f(x_{k−j})` (Grippo-Lampariello-Lucidi)
//!    allows temporary objective increases while still guaranteeing global
//!    convergence.
//!
//! At iteration `k` the search direction is the projected-gradient direction
//! `d = P_C(x − α ∇f(x)) − x`. The stationarity test `‖d‖_∞ < tol` certifies a
//! first-order optimal point. The step `λ ∈ {1, ½, ¼, …}` is accepted once it
//! satisfies the sufficient-decrease condition `f(x + λ d) ≤ f_ref + γ λ gᵀd`.

use crate::error::{CvxError, CvxResult};

/// Configuration for the [`Spg`] solver.
#[derive(Debug, Clone)]
pub struct SpgConfig {
    /// Maximum number of outer iterations (`≥ 1`).
    pub max_iter: usize,
    /// Nonmonotone memory `M` (`≥ 1`); `M = 1` reduces to a monotone search.
    pub memory: usize,
    /// Lower clamp on the spectral step length (`> 0`).
    pub alpha_min: f32,
    /// Upper clamp on the spectral step length (`> alpha_min`).
    pub alpha_max: f32,
    /// Sufficient-decrease parameter `γ ∈ (0, 1)`.
    pub gamma: f32,
    /// Stationarity tolerance on `‖d‖_∞` (`> 0`).
    pub tol: f32,
}

impl Default for SpgConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            memory: 10,
            alpha_min: 1.0e-10,
            alpha_max: 1.0e10,
            gamma: 1.0e-4,
            tol: 1.0e-6,
        }
    }
}

/// Result of an [`Spg`] minimisation.
#[derive(Debug, Clone)]
pub struct SpgResult {
    /// Final iterate `x`.
    pub x: Vec<f32>,
    /// Objective value `f(x)` at the final iterate.
    pub f: f32,
    /// Number of outer iterations performed.
    pub iterations: usize,
    /// Whether the stationarity criterion `‖d‖_∞ < tol` was met.
    pub converged: bool,
}

/// Spectral Projected Gradient solver.
pub struct Spg;

impl Spg {
    /// Minimise `f` over the convex set defined by `project`, starting from `x0`.
    ///
    /// * `f` — objective `x ↦ f(x)`.
    /// * `grad` — gradient `x ↦ ∇f(x)` (must return a vector of the same length as `x`).
    /// * `project` — Euclidean projection `x ↦ P_C(x)` onto the feasible set.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::InvalidParameter`] / [`CvxError::EmptyInput`] on malformed
    /// configuration, [`CvxError::DimensionMismatch`] if `grad` or `project` change
    /// the dimension, and [`CvxError::LineSearchFailed`] if backtracking cannot find
    /// an acceptable step.
    pub fn minimize<F, G, P>(
        x0: &[f32],
        f: F,
        grad: G,
        project: P,
        cfg: &SpgConfig,
    ) -> CvxResult<SpgResult>
    where
        F: Fn(&[f32]) -> f32,
        G: Fn(&[f32]) -> Vec<f32>,
        P: Fn(&[f32]) -> Vec<f32>,
    {
        Self::validate(x0, cfg)?;
        let dim = x0.len();

        // Start feasible.
        let mut x = project(x0);
        if x.len() != dim {
            return Err(CvxError::DimensionMismatch { a: x.len(), b: dim });
        }

        let mut g = grad(&x);
        if g.len() != dim {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: dim });
        }
        let mut f_x = f(&x);

        // Nonmonotone history ring buffer of recent objective values.
        let mut history: Vec<f32> = Vec::with_capacity(cfg.memory);
        history.push(f_x);

        // Spectral step length: α = 1 on the first iteration.
        let mut alpha = 1.0_f32;
        let mut iterations = 0usize;
        let mut converged = false;

        for it in 0..cfg.max_iter {
            iterations = it + 1;

            // Projected-gradient search direction: d = P_C(x − α g) − x.
            let trial: Vec<f32> = (0..dim).map(|j| x[j] - alpha * g[j]).collect();
            let proj = project(&trial);
            if proj.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: proj.len(),
                    b: dim,
                });
            }
            let d: Vec<f32> = (0..dim).map(|j| proj[j] - x[j]).collect();

            // Stationarity test on the infinity norm of the projected step.
            let mut d_inf = 0.0_f32;
            for &di in &d {
                d_inf = d_inf.max(di.abs());
            }
            if d_inf < cfg.tol {
                converged = true;
                break;
            }

            // Directional derivative gᵀd (≤ 0 for a descent direction here).
            let g_dot_d: f32 = (0..dim).map(|j| g[j] * d[j]).sum();

            // Nonmonotone reference: max over the last min(k+1, memory) values.
            let mut f_ref = f32::NEG_INFINITY;
            for &fh in &history {
                f_ref = f_ref.max(fh);
            }

            // Backtracking on λ ∈ {1, ½, ¼, …} (GLL sufficient decrease).
            let mut lambda = 1.0_f32;
            let mut x_new = vec![0.0_f32; dim];
            let mut f_new = f_x;
            let mut accepted = false;
            for _ in 0..60 {
                for j in 0..dim {
                    x_new[j] = x[j] + lambda * d[j];
                }
                f_new = f(&x_new);
                if f_new <= f_ref + cfg.gamma * lambda * g_dot_d {
                    accepted = true;
                    break;
                }
                lambda *= 0.5;
                if lambda < 1.0e-20 {
                    break;
                }
            }
            if !accepted {
                return Err(CvxError::LineSearchFailed(format!(
                    "SPG backtracking failed at iteration {it} (‖d‖_∞ = {d_inf})"
                )));
            }

            // Gradient at the new iterate.
            let g_new = grad(&x_new);
            if g_new.len() != dim {
                return Err(CvxError::DimensionMismatch {
                    a: g_new.len(),
                    b: dim,
                });
            }

            // Spectral (BB) step update from s = x_new − x, y = g_new − g.
            let s: Vec<f32> = (0..dim).map(|j| x_new[j] - x[j]).collect();
            let y_vec: Vec<f32> = (0..dim).map(|j| g_new[j] - g[j]).collect();
            alpha = Self::bb_step(&s, &y_vec, cfg.alpha_min, cfg.alpha_max)?;

            // Advance the iterate and the nonmonotone history.
            x = x_new.clone();
            g = g_new;
            f_x = f_new;
            if history.len() == cfg.memory {
                history.remove(0);
            }
            history.push(f_x);
        }

        Ok(SpgResult {
            x,
            f: f_x,
            iterations,
            converged,
        })
    }

    /// Barzilai-Borwein spectral step length `α = (sᵀs) / (sᵀy)`.
    ///
    /// The result is clamped to `[alpha_min, alpha_max]`. When `sᵀy ≤ 0` (or the
    /// inner products are not finite) the safeguard value `alpha_max` is returned,
    /// preventing a non-descent or unbounded step.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::DimensionMismatch`] if `s` and `y` differ in length,
    /// and [`CvxError::InvalidParameter`] if the clamps are inconsistent.
    pub fn bb_step(s: &[f32], y: &[f32], alpha_min: f32, alpha_max: f32) -> CvxResult<f32> {
        if s.len() != y.len() {
            return Err(CvxError::DimensionMismatch {
                a: s.len(),
                b: y.len(),
            });
        }
        if alpha_min <= 0.0 || !alpha_min.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "SPG alpha_min must be > 0, got {alpha_min}"
            )));
        }
        if alpha_max <= alpha_min || !alpha_max.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "SPG alpha_max must exceed alpha_min, got max={alpha_max}, min={alpha_min}"
            )));
        }
        let s_dot_s: f32 = s.iter().map(|v| v * v).sum();
        let s_dot_y: f32 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        if s_dot_y <= 0.0 || !s_dot_s.is_finite() || !s_dot_y.is_finite() {
            return Ok(alpha_max);
        }
        let alpha = s_dot_s / s_dot_y;
        if !alpha.is_finite() {
            return Ok(alpha_max);
        }
        Ok(alpha.clamp(alpha_min, alpha_max))
    }

    fn validate(x0: &[f32], cfg: &SpgConfig) -> CvxResult<()> {
        if x0.is_empty() {
            return Err(CvxError::EmptyInput);
        }
        if cfg.max_iter == 0 {
            return Err(CvxError::InvalidParameter(
                "SPG requires max_iter ≥ 1".to_string(),
            ));
        }
        if cfg.memory == 0 {
            return Err(CvxError::InvalidParameter(
                "SPG requires memory ≥ 1".to_string(),
            ));
        }
        if cfg.alpha_min <= 0.0 || !cfg.alpha_min.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "SPG alpha_min must be > 0, got {}",
                cfg.alpha_min
            )));
        }
        if cfg.alpha_max <= cfg.alpha_min || !cfg.alpha_max.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "SPG alpha_max must exceed alpha_min, got max={}, min={}",
                cfg.alpha_max, cfg.alpha_min
            )));
        }
        if cfg.gamma <= 0.0 || cfg.gamma >= 1.0 || !cfg.gamma.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "SPG gamma must lie in (0, 1), got {}",
                cfg.gamma
            )));
        }
        if cfg.tol <= 0.0 || !cfg.tol.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "SPG tol must be > 0, got {}",
                cfg.tol
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity projection (unconstrained).
    fn identity(x: &[f32]) -> Vec<f32> {
        x.to_vec()
    }

    /// Project onto the box `[lo, hi]^n`.
    fn box_project(x: &[f32], lo: f32, hi: f32) -> Vec<f32> {
        x.iter().map(|v| v.clamp(lo, hi)).collect()
    }

    /// Project onto the probability simplex `{x ≥ 0, Σx = 1}` (sort-based).
    fn simplex_project(x: &[f32]) -> Vec<f32> {
        let mut u = x.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0_f32;
        let mut tau = 0.0_f32;
        for (k, &uk) in u.iter().enumerate() {
            cum += uk;
            let cand = (cum - 1.0) / (k as f32 + 1.0);
            if uk - cand > 0.0 {
                tau = cand;
            } else {
                break;
            }
        }
        x.iter().map(|v| (v - tau).max(0.0)).collect()
    }

    #[test]
    fn strictly_convex_quadratic_identity() {
        // min ½ xᵀA x − bᵀx, A = diag(2, 4), b = (2, 8) → x = A⁻¹b = (1, 2).
        let a = [2.0_f32, 4.0];
        let b = [2.0_f32, 8.0];
        let f = |x: &[f32]| -> f32 {
            0.5 * (a[0] * x[0] * x[0] + a[1] * x[1] * x[1]) - (b[0] * x[0] + b[1] * x[1])
        };
        let grad = |x: &[f32]| -> Vec<f32> { vec![a[0] * x[0] - b[0], a[1] * x[1] - b[1]] };
        let res =
            Spg::minimize(&[0.0, 0.0], f, grad, identity, &SpgConfig::default()).expect("minimize");
        assert!(res.converged);
        assert!((res.x[0] - 1.0).abs() < 1.0e-4, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 2.0).abs() < 1.0e-4, "x1 = {}", res.x[1]);
    }

    #[test]
    fn box_constrained_quadratic_stationarity() {
        // min ½‖x − t‖² over [0, 1]^2, t = (2, -1) → optimum (1, 0).
        let t = [2.0_f32, -1.0];
        let f = |x: &[f32]| -> f32 { 0.5 * ((x[0] - t[0]).powi(2) + (x[1] - t[1]).powi(2)) };
        let grad = |x: &[f32]| -> Vec<f32> { vec![x[0] - t[0], x[1] - t[1]] };
        let proj = |x: &[f32]| box_project(x, 0.0, 1.0);
        let res =
            Spg::minimize(&[0.5, 0.5], f, grad, proj, &SpgConfig::default()).expect("minimize");
        assert!(res.converged);
        // Projected-gradient stationarity ‖P_C(x − g) − x‖_∞ small.
        let g = vec![res.x[0] - t[0], res.x[1] - t[1]];
        let step: Vec<f32> = (0..2).map(|j| res.x[j] - g[j]).collect();
        let proj_step = box_project(&step, 0.0, 1.0);
        let mut stat = 0.0_f32;
        for (&ps, &xj) in proj_step.iter().zip(res.x.iter()) {
            stat = stat.max((ps - xj).abs());
        }
        assert!(stat < 1.0e-4, "stationarity = {stat}");
        assert!((res.x[0] - 1.0).abs() < 1.0e-3, "x0 = {}", res.x[0]);
        assert!(res.x[1].abs() < 1.0e-3, "x1 = {}", res.x[1]);
    }

    #[test]
    fn bb_step_hand_example() {
        // s = (2, 0), y = (1, 0) → α = (sᵀs)/(sᵀy) = 4 / 2 = 2.
        let alpha = Spg::bb_step(&[2.0, 0.0], &[1.0, 0.0], 1.0e-10, 1.0e10).expect("bb");
        assert!((alpha - 2.0).abs() < 1.0e-6, "alpha = {alpha}");
    }

    #[test]
    fn bb_step_clamps_to_bounds() {
        // s = (10, 0), y = (1e-3, 0) → α = 100/1e-3 = 1e5, clamp to 10.
        let alpha = Spg::bb_step(&[10.0, 0.0], &[1.0e-3, 0.0], 0.1, 10.0).expect("bb");
        assert!((alpha - 10.0).abs() < 1.0e-5, "alpha = {alpha}");
        // Tiny ratio clamps up to alpha_min.
        let alpha2 = Spg::bb_step(&[1.0e-4, 0.0], &[100.0, 0.0], 0.5, 10.0).expect("bb");
        assert!((alpha2 - 0.5).abs() < 1.0e-6, "alpha2 = {alpha2}");
    }

    #[test]
    fn bb_step_fallback_when_sty_nonpositive() {
        // sᵀy = -1 ≤ 0 → fallback alpha_max.
        let alpha = Spg::bb_step(&[1.0, 0.0], &[-1.0, 0.0], 1.0e-3, 7.0).expect("bb");
        assert!((alpha - 7.0).abs() < 1.0e-6, "alpha = {alpha}");
        // sᵀy = 0 → fallback alpha_max as well.
        let alpha0 = Spg::bb_step(&[1.0, 1.0], &[0.0, 0.0], 1.0e-3, 5.0).expect("bb");
        assert!((alpha0 - 5.0).abs() < 1.0e-6, "alpha0 = {alpha0}");
    }

    #[test]
    fn converged_flag_easy_problem() {
        let f = |x: &[f32]| -> f32 { x.iter().map(|v| v * v).sum() };
        let grad = |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 2.0 * v).collect() };
        let res = Spg::minimize(&[3.0, -2.0], f, grad, identity, &SpgConfig::default())
            .expect("minimize");
        assert!(res.converged);
        assert!(res.x.iter().all(|v| v.abs() < 1.0e-4));
    }

    #[test]
    fn nonmonotone_memory_allows_increase() {
        // A mildly oscillatory but convex objective; memory > 1 should still converge.
        let f = |x: &[f32]| -> f32 { 0.5 * (x[0] * x[0] + 10.0 * x[1] * x[1]) };
        let grad = |x: &[f32]| -> Vec<f32> { vec![x[0], 10.0 * x[1]] };
        let cfg = SpgConfig {
            memory: 5,
            ..SpgConfig::default()
        };
        let res = Spg::minimize(&[5.0, 5.0], f, grad, identity, &cfg).expect("minimize");
        assert!(res.converged);
        assert!(res.x[0].abs() < 1.0e-3 && res.x[1].abs() < 1.0e-3);
    }

    #[test]
    fn monotone_memory_one_converges() {
        let f = |x: &[f32]| -> f32 { 0.5 * (x[0] * x[0] + 4.0 * x[1] * x[1]) };
        let grad = |x: &[f32]| -> Vec<f32> { vec![x[0], 4.0 * x[1]] };
        let cfg = SpgConfig {
            memory: 1,
            ..SpgConfig::default()
        };
        let res = Spg::minimize(&[2.0, -3.0], f, grad, identity, &cfg).expect("minimize");
        assert!(res.converged);
        assert!(res.x[0].abs() < 1.0e-3 && res.x[1].abs() < 1.0e-3);
    }

    #[test]
    fn deterministic_repeated_runs() {
        let f = |x: &[f32]| -> f32 { x.iter().map(|v| v * v).sum() };
        let grad = |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 2.0 * v).collect() };
        let r1 = Spg::minimize(&[1.0, 2.0, 3.0], f, grad, identity, &SpgConfig::default())
            .expect("minimize");
        let r2 = Spg::minimize(&[1.0, 2.0, 3.0], f, grad, identity, &SpgConfig::default())
            .expect("minimize");
        assert_eq!(r1.iterations, r2.iterations);
        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.f, r2.f);
    }

    #[test]
    fn projection_onto_simplex_constrains_iterate() {
        // min ½‖x − t‖² over the probability simplex, t = (0.7, 0.2, 0.6).
        let t = [0.7_f32, 0.2, 0.6];
        let f = |x: &[f32]| -> f32 { 0.5 * (0..3).map(|j| (x[j] - t[j]).powi(2)).sum::<f32>() };
        let grad = |x: &[f32]| -> Vec<f32> { (0..3).map(|j| x[j] - t[j]).collect() };
        let res = Spg::minimize(
            &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            f,
            grad,
            simplex_project,
            &SpgConfig::default(),
        )
        .expect("minimize");
        assert!(res.converged);
        let sum: f32 = res.x.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-4, "sum = {sum}");
        assert!(res.x.iter().all(|&v| v >= -1.0e-5));
    }

    #[test]
    fn one_dimensional_minimization() {
        // min (x − 4)² → x = 4.
        let f = |x: &[f32]| -> f32 { (x[0] - 4.0).powi(2) };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * (x[0] - 4.0)] };
        let res =
            Spg::minimize(&[0.0], f, grad, identity, &SpgConfig::default()).expect("minimize");
        assert!(res.converged);
        assert!((res.x[0] - 4.0).abs() < 1.0e-4, "x = {}", res.x[0]);
    }

    #[test]
    fn separable_convex_function() {
        // min Σ (x_j − j)² for j = 1..4 → x = (1, 2, 3, 4).
        let target = [1.0_f32, 2.0, 3.0, 4.0];
        let f = |x: &[f32]| -> f32 { (0..4).map(|j| (x[j] - target[j]).powi(2)).sum() };
        let grad = |x: &[f32]| -> Vec<f32> { (0..4).map(|j| 2.0 * (x[j] - target[j])).collect() };
        let res = Spg::minimize(
            &[0.0, 0.0, 0.0, 0.0],
            f,
            grad,
            identity,
            &SpgConfig::default(),
        )
        .expect("minimize");
        assert!(res.converged);
        for (j, (&xj, &tj)) in res.x.iter().zip(target.iter()).enumerate() {
            assert!((xj - tj).abs() < 1.0e-3, "x[{j}] = {xj}");
        }
    }

    #[test]
    fn objective_decreases_overall() {
        let f = |x: &[f32]| -> f32 { x.iter().map(|v| v * v).sum() };
        let grad = |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 2.0 * v).collect() };
        let x0 = [4.0_f32, -5.0, 6.0];
        let f0 = f(&x0);
        let res = Spg::minimize(&x0, f, grad, identity, &SpgConfig::default()).expect("minimize");
        assert!(res.f < f0, "final f {} >= initial {}", res.f, f0);
    }

    #[test]
    fn err_x0_empty() {
        let f = |_: &[f32]| -> f32 { 0.0 };
        let grad = |_: &[f32]| -> Vec<f32> { Vec::new() };
        assert!(matches!(
            Spg::minimize(&[], f, grad, identity, &SpgConfig::default()),
            Err(CvxError::EmptyInput)
        ));
    }

    #[test]
    fn err_max_iter_zero() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            max_iter: 0,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_memory_zero() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            memory: 0,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_alpha_min_non_positive() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            alpha_min: 0.0,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_alpha_max_le_alpha_min() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            alpha_min: 1.0,
            alpha_max: 0.5,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_gamma_zero() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            gamma: 0.0,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_gamma_ge_one() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            gamma: 1.0,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn err_tol_non_positive() {
        let f = |x: &[f32]| -> f32 { x[0] * x[0] };
        let grad = |x: &[f32]| -> Vec<f32> { vec![2.0 * x[0]] };
        let cfg = SpgConfig {
            tol: 0.0,
            ..SpgConfig::default()
        };
        assert!(matches!(
            Spg::minimize(&[1.0], f, grad, identity, &cfg),
            Err(CvxError::InvalidParameter(_))
        ));
    }

    #[test]
    fn bb_step_dimension_mismatch() {
        assert!(matches!(
            Spg::bb_step(&[1.0, 2.0], &[1.0], 1.0e-10, 1.0e10),
            Err(CvxError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn bb_step_rejects_bad_clamps() {
        assert!(matches!(
            Spg::bb_step(&[1.0], &[1.0], 0.0, 1.0),
            Err(CvxError::InvalidParameter(_))
        ));
        assert!(matches!(
            Spg::bb_step(&[1.0], &[1.0], 2.0, 1.0),
            Err(CvxError::InvalidParameter(_))
        ));
    }
}
