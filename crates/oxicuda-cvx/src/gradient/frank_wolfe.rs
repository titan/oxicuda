//! Frank-Wolfe / Conditional Gradient method for constrained convex optimisation.
//!
//! Solves `min_{x ∈ C} f(x)` where `C` is described implicitly via a Linear Minimization
//! Oracle (LMO): `s_k = argmin_{s ∈ C} <∇f(x_k), s>`.
//!
//! Update rule: `x_{k+1} = x_k + γ_k (s_k − x_k)` where:
//! - Open-loop step: `γ_k = 2 / (k + 2)`
//! - Backtracking: Armijo with direction `d_k = s_k − x_k`.
//!
//! References:
//! - Frank & Wolfe (1956), "An algorithm for quadratic programming".
//! - Jaggi (2013), "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization", ICML.

use crate::error::{CvxError, CvxResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Step-size strategy for the Frank-Wolfe algorithm.
#[derive(Debug, Clone)]
pub enum FwStepSize {
    /// Classic open-loop schedule: `γ_k = 2 / (k + 2)`.
    OpenLoop,
    /// Backtracking Armijo: starting from `γ = 1`, multiply by `rho` until
    /// `f(x + γ d) ≤ f(x) + c · γ · ⟨∇f(x), d⟩`.
    ///
    /// `c ∈ (0, 1)`, `rho ∈ (0, 1)`.
    Backtrack {
        /// Armijo sufficient-decrease coefficient.
        c: f64,
        /// Reduction factor per backtracking step.
        rho: f64,
    },
}

/// Configuration for the Frank-Wolfe solver.
#[derive(Debug, Clone)]
pub struct FrankWolfeConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance: stop when Frank-Wolfe gap < `tol`.
    pub tol: f64,
    /// Step-size strategy.
    pub step_size: FwStepSize,
}

impl Default for FrankWolfeConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            step_size: FwStepSize::OpenLoop,
        }
    }
}

/// Output of a Frank-Wolfe run.
#[derive(Debug, Clone)]
pub struct FrankWolfeResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Final Frank-Wolfe gap `⟨∇f(x), x − s⟩ ≥ 0` (upper bound on sub-optimality).
    pub fw_gap: f64,
    /// Whether the FW gap dropped below `tol`.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Internal helper: plain dot product
// ---------------------------------------------------------------------------

#[inline]
fn dot_plain(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_config(cfg: &FrankWolfeConfig) -> CvxResult<()> {
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0, got {}",
            cfg.tol
        )));
    }
    if let FwStepSize::Backtrack { c, rho } = cfg.step_size {
        if c <= 0.0 || c >= 1.0 {
            return Err(CvxError::InvalidParameter(format!(
                "backtrack c must be in (0, 1), got {c}"
            )));
        }
        if rho <= 0.0 || rho >= 1.0 {
            return Err(CvxError::InvalidParameter(format!(
                "backtrack rho must be in (0, 1), got {rho}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the Frank-Wolfe (Conditional Gradient) algorithm.
///
/// # Arguments
/// - `x0`: initial feasible point in `C` (non-empty).
/// - `f`: objective function `ℝⁿ → ℝ`.
/// - `grad_f`: gradient `ℝⁿ → CvxResult<ℝⁿ>`.
/// - `lmo`: Linear Minimization Oracle: `g ↦ argmin_{s ∈ C} ⟨g, s⟩`.
/// - `cfg`: algorithm configuration.
///
/// # Errors
/// Returns [`CvxError::EmptyInput`] if `x0` is empty, [`CvxError::InvalidParameter`]
/// for invalid configuration, [`CvxError::DimensionMismatch`] if the LMO returns a
/// vector of the wrong length, or any error propagated from `grad_f` / `lmo`.
pub fn frank_wolfe<F, G, LMO>(
    x0: &[f64],
    f: F,
    grad_f: G,
    lmo: LMO,
    cfg: &FrankWolfeConfig,
) -> CvxResult<FrankWolfeResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    LMO: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    validate_config(cfg)?;

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fw_gap = 0.0_f64;
    let mut converged = false;
    let mut n_iter = 0usize;

    for k in 0..cfg.max_iter {
        let g = grad_f(&x)?;
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }

        let s = lmo(&g)?;
        if s.len() != n {
            return Err(CvxError::DimensionMismatch { a: s.len(), b: n });
        }

        // Frank-Wolfe gap: ⟨∇f(x), x − s⟩ = dot(g, x) − dot(g, s) ≥ 0 at optimum
        fw_gap = dot_plain(&g, &x) - dot_plain(&g, &s);

        if fw_gap < cfg.tol {
            converged = true;
            break;
        }

        // Direction d = s − x
        let d: Vec<f64> = s.iter().zip(x.iter()).map(|(si, xi)| si - xi).collect();

        // Step size
        let gamma = match &cfg.step_size {
            FwStepSize::OpenLoop => 2.0 / (k as f64 + 2.0),
            FwStepSize::Backtrack { c, rho } => {
                let f0 = f(&x);
                // slope = c * dot(g, d) = c * (dot(g, s) − dot(g, x)) = −c * fw_gap ≤ 0
                let slope = *c * dot_plain(&g, &d);
                let mut gamma = 1.0_f64;
                loop {
                    let x_trial: Vec<f64> = x
                        .iter()
                        .zip(d.iter())
                        .map(|(xi, di)| xi + gamma * di)
                        .collect();
                    if f(&x_trial) <= f0 + gamma * slope {
                        break;
                    }
                    gamma *= rho;
                    if gamma < 1e-15 {
                        gamma = 1e-15;
                        break;
                    }
                }
                gamma
            }
        };

        // Update: x ← x + γ · d
        for i in 0..n {
            x[i] += gamma * d[i];
        }

        n_iter += 1;
    }

    Ok(FrankWolfeResult {
        x,
        n_iter,
        fw_gap,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // LMO helpers used throughout tests
    // ----------------------------------------------------------------

    /// Simplex LMO: argmin_{s ∈ Δ_n} ⟨g, s⟩ — put all weight on the index of min(g).
    fn simplex_lmo(g: &[f64]) -> CvxResult<Vec<f64>> {
        let n = g.len();
        let mut best = 0usize;
        for i in 1..n {
            if g[i] < g[best] {
                best = i;
            }
        }
        let mut s = vec![0.0_f64; n];
        s[best] = 1.0;
        Ok(s)
    }

    /// L2-ball LMO (radius 1): argmin_{‖s‖≤1} ⟨g, s⟩ = −g / ‖g‖ (or zero if g = 0).
    fn l2_ball_lmo(g: &[f64]) -> CvxResult<Vec<f64>> {
        let nrm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if nrm < 1e-300 {
            return Ok(vec![0.0; g.len()]);
        }
        Ok(g.iter().map(|gi| -gi / nrm).collect())
    }

    // ----------------------------------------------------------------
    // Quadratic centred at c on the probability simplex
    // ----------------------------------------------------------------

    fn sq_f(x: &[f64], c: &[f64]) -> f64 {
        x.iter()
            .zip(c.iter())
            .map(|(xi, ci)| 0.5 * (xi - ci).powi(2))
            .sum()
    }

    fn sq_g(x: &[f64], c: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(x.iter().zip(c.iter()).map(|(xi, ci)| xi - ci).collect())
    }

    macro_rules! simplex_quad {
        ($c:expr) => {{
            let c_vec: &[f64] = $c;
            let n = c_vec.len();
            let x0: Vec<f64> = vec![1.0 / n as f64; n];
            let cv_f = c_vec.to_vec();
            let cv_g = c_vec.to_vec();
            (
                x0,
                move |x: &[f64]| sq_f(x, &cv_f),
                move |x: &[f64]| sq_g(x, &cv_g),
            )
        }};
    }

    // ----------------------------------------------------------------

    #[test]
    fn simplex_quadratic() {
        let c = vec![0.6, 0.2, 0.2];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig {
            max_iter: 5000,
            tol: 1e-6,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        for (xi, ci) in res.x.iter().zip(c.iter()) {
            assert!((xi - ci).abs() < 1e-3, "xi={xi}, ci={ci}");
        }
    }

    #[test]
    fn fw_gap_nonneg() {
        let c = vec![0.5, 0.3, 0.2];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig::default();
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.fw_gap >= -1e-10, "fw_gap={}", res.fw_gap);
    }

    #[test]
    fn converged_flag() {
        let c = vec![0.5, 0.5];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig {
            max_iter: 5000,
            tol: 1e-4,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.converged);
    }

    #[test]
    fn n_iter_lt_max() {
        let c = vec![0.5, 0.5];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig {
            max_iter: 5000,
            tol: 1e-4,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.n_iter < cfg.max_iter);
    }

    #[test]
    fn result_length() {
        let c = vec![0.3, 0.3, 0.4];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig::default();
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert_eq!(res.x.len(), x0.len());
    }

    #[test]
    fn empty_x0_err() {
        let f = |_: &[f64]| 0.0_f64;
        let grad_f = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![]) };
        let lmo = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![]) };
        let cfg = FrankWolfeConfig::default();
        match frank_wolfe(&[], f, grad_f, lmo, &cfg) {
            Err(CvxError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    #[test]
    fn invalid_tol_err() {
        let f = |_: &[f64]| 0.0_f64;
        let grad_f = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0]) };
        let lmo = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0]) };
        let cfg = FrankWolfeConfig {
            tol: -1.0,
            ..FrankWolfeConfig::default()
        };
        match frank_wolfe(&[0.5], f, grad_f, lmo, &cfg) {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn open_loop_converges() {
        let c = vec![0.7, 0.1, 0.2];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig {
            max_iter: 5000,
            tol: 1e-5,
            step_size: FwStepSize::OpenLoop,
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.converged || res.fw_gap < 1e-3, "fw_gap={}", res.fw_gap);
    }

    #[test]
    fn backtrack_converges() {
        let c = vec![0.7, 0.1, 0.2];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig {
            max_iter: 5000,
            tol: 1e-5,
            step_size: FwStepSize::Backtrack { c: 0.1, rho: 0.5 },
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.converged || res.fw_gap < 1e-3, "fw_gap={}", res.fw_gap);
    }

    #[test]
    fn l2_ball_lmo_test() {
        // min 0.5 * ||x - c||^2 s.t. ||x|| ≤ 1, c = [0.3, 0.4]
        // Solution is c / max(1, ||c||) = [0.3, 0.4] (already inside ball)
        let c = vec![0.3, 0.4];
        let f = |x: &[f64]| -> f64 {
            x.iter()
                .zip(c.iter())
                .map(|(xi, ci)| 0.5 * (xi - ci).powi(2))
                .sum()
        };
        let grad_f = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(c.iter()).map(|(xi, ci)| xi - ci).collect())
        };
        // x0 = (-1, 0) is on the boundary of the L2 ball
        let x0 = vec![-1.0, 0.0];
        let cfg = FrankWolfeConfig {
            max_iter: 5000,
            tol: 1e-5,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, l2_ball_lmo, &cfg).expect("ok");
        // Result should be close to c (inside ball)
        assert!((res.x[0] - 0.3).abs() < 5e-3, "x[0]={}", res.x[0]);
        assert!((res.x[1] - 0.4).abs() < 5e-3, "x[1]={}", res.x[1]);
    }

    #[test]
    fn x_stays_in_simplex() {
        // Frank-Wolfe iterates are convex combinations of simplex vertices → remain in simplex
        let c = vec![0.4, 0.3, 0.3];
        let (x0, f, grad_f) = simplex_quad!(&c);
        let cfg = FrankWolfeConfig {
            max_iter: 200,
            tol: 1e-8,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        let sum: f64 = res.x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum={sum}");
        for &xi in &res.x {
            assert!(xi >= -1e-12, "negative component xi={xi}");
        }
    }

    #[test]
    fn fw_gap_zero_at_opt() {
        // At the optimum c = [1/3, 1/3, 1/3] (centroid, which is x0), fw_gap should be tiny.
        let c = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let (x0, f, grad_f) = simplex_quad!(&c);
        // x0 is already the optimum → gradient is 0 → fw_gap = 0
        let cfg = FrankWolfeConfig {
            max_iter: 1,
            tol: 1e-12,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.fw_gap < 1e-4, "fw_gap={}", res.fw_gap);
    }

    #[test]
    fn backtrack_invalid_c() {
        let f = |_: &[f64]| 0.0_f64;
        let grad_f = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0]) };
        let lmo = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0]) };
        let cfg = FrankWolfeConfig {
            step_size: FwStepSize::Backtrack { c: 1.5, rho: 0.5 },
            ..FrankWolfeConfig::default()
        };
        match frank_wolfe(&[0.5], f, grad_f, lmo, &cfg) {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn backtrack_invalid_rho() {
        let f = |_: &[f64]| 0.0_f64;
        let grad_f = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0]) };
        let lmo = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![0.0]) };
        let cfg = FrankWolfeConfig {
            step_size: FwStepSize::Backtrack { c: 0.1, rho: 1.0 },
            ..FrankWolfeConfig::default()
        };
        match frank_wolfe(&[0.5], f, grad_f, lmo, &cfg) {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn lmo_dim_mismatch_err() {
        let f = |_: &[f64]| 0.0_f64;
        let grad_f = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![1.0, 0.0]) };
        // LMO returns wrong length (1 instead of 2)
        let bad_lmo = |_: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![1.0]) };
        let cfg = FrankWolfeConfig::default();
        let x0 = vec![0.5, 0.5];
        match frank_wolfe(&x0, f, grad_f, bad_lmo, &cfg) {
            Err(CvxError::DimensionMismatch { .. }) => {}
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn d100_simplex() {
        let n = 100usize;
        // Target: uniform on first 10 coordinates (scaled), rest zero — but must be on simplex
        let mut c = vec![0.0_f64; n];
        // First 10 components each get 0.1 → sum = 1.0
        for ci in c.iter_mut().take(10) {
            *ci = 0.1;
        }
        let x0 = vec![1.0 / n as f64; n];
        let c_f = c.clone();
        let c_g = c.clone();
        let f = move |x: &[f64]| -> f64 {
            x.iter()
                .zip(c_f.iter())
                .map(|(xi, ci)| 0.5 * (xi - ci).powi(2))
                .sum()
        };
        let grad_f = move |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(c_g.iter()).map(|(xi, ci)| xi - ci).collect())
        };
        let cfg = FrankWolfeConfig {
            max_iter: 10000,
            tol: 1e-4,
            ..FrankWolfeConfig::default()
        };
        let res = frank_wolfe(&x0, f, grad_f, simplex_lmo, &cfg).expect("ok");
        assert!(res.converged || res.fw_gap < 1e-3, "fw_gap={}", res.fw_gap);
        // Check result is approximately on the simplex
        let sum: f64 = res.x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum={sum}");
    }
}
