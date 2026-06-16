//! Frank-Wolfe (Conditional Gradient) method for constrained convex optimisation.
//!
//! Solves `min_{x ∈ C} f(x)` where `C` is a convex set accessed via a Linear
//! Minimization Oracle (LMO).
//!
//! ## Algorithm (open-loop schedule)
//! ```text
//! For t = 0, 1, 2, …
//!   g_t  = ∇f(x_t)
//!   s_t  = LMO(g_t)  = argmin_{v ∈ C}  g_t · v
//!   d_t  = s_t − x_t               (Frank-Wolfe direction)
//!   γ_t  = 2 / (t + 2)             (open-loop step size)
//!   gap  = − g_t · d_t ≥ 0         (convergence certificate)
//!   x_{t+1} = x_t + γ_t · d_t
//! ```
//!
//! References:
//! - Frank & Wolfe (1956), "An algorithm for quadratic programming".
//! - Jaggi (2013), "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization", ICML.

use crate::error::{CvxError, CvxResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the Frank-Wolfe solver.
#[derive(Debug, Clone)]
pub struct FrankWolfeConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stop when the Frank-Wolfe gap < `tol` (non-negative, certificates sub-optimality).
    pub tol: f64,
}

impl Default for FrankWolfeConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
        }
    }
}

/// Result of a Frank-Wolfe run.
#[derive(Debug, Clone)]
pub struct FwResult {
    /// Final iterate `x_t ∈ C`.
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub iter: usize,
    /// Final Frank-Wolfe gap `−∇f(x) · (LMO(∇f(x)) − x) ≥ 0`.
    pub gap: f64,
}

// ---------------------------------------------------------------------------
// LMO helpers
// ---------------------------------------------------------------------------

/// Linear Minimization Oracle for the probability simplex `Δ_n = {x ≥ 0 : Σ x_i = 1}`.
///
/// Returns `e_k` where `k = argmin_j grad[j]`.
///
/// # Panics
/// Does not panic; returns all-zero vector of length `grad.len()` if `grad` is empty.
#[must_use]
pub fn simplex_lmo(grad: &[f64]) -> Vec<f64> {
    if grad.is_empty() {
        return Vec::new();
    }
    let mut best = 0usize;
    for i in 1..grad.len() {
        if grad[i] < grad[best] {
            best = i;
        }
    }
    let mut s = vec![0.0_f64; grad.len()];
    s[best] = 1.0;
    s
}

/// Linear Minimization Oracle for the L1 ball `{x : ‖x‖₁ ≤ 1}`.
///
/// Returns `±e_k` where `k = argmax_j |grad[j]|`, with sign `−sign(grad[k])`.
///
/// # Panics
/// Does not panic; returns all-zero vector if `grad` is empty or all-zero.
#[must_use]
pub fn l1_ball_lmo(grad: &[f64]) -> Vec<f64> {
    if grad.is_empty() {
        return Vec::new();
    }
    let mut best = 0usize;
    for i in 1..grad.len() {
        if grad[i].abs() > grad[best].abs() {
            best = i;
        }
    }
    let mut s = vec![0.0_f64; grad.len()];
    if grad[best].abs() > 0.0 {
        s[best] = -grad[best].signum();
    }
    s
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_cfg(cfg: &FrankWolfeConfig) -> CvxResult<()> {
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "frank_wolfe: max_iter must be ≥ 1".into(),
        ));
    }
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "frank_wolfe: tol must be > 0, got {}",
            cfg.tol
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Run Frank-Wolfe (Conditional Gradient) with open-loop step size `γ_t = 2/(t+2)`.
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
/// - [`CvxError::DimensionMismatch`] if `lmo` returns a vector of wrong length.
pub fn frank_wolfe(
    x_init: &[f64],
    grad_fn: impl Fn(&[f64]) -> Vec<f64>,
    lmo: impl Fn(&[f64]) -> Vec<f64>,
    cfg: &FrankWolfeConfig,
) -> CvxResult<FwResult> {
    if x_init.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    validate_cfg(cfg)?;

    let n = x_init.len();
    let mut x = x_init.to_vec();
    let mut gap = 0.0_f64;
    let mut final_iter = 0usize;

    for t in 0..cfg.max_iter {
        let g = grad_fn(&x);
        let s = lmo(&g);

        if s.len() != n {
            return Err(CvxError::DimensionMismatch { a: s.len(), b: n });
        }

        // Frank-Wolfe gap: gap = − g · (s − x) = g · (x − s) ≥ 0 at optimum.
        let mut dot_gs = 0.0_f64;
        let mut dot_gx = 0.0_f64;
        for j in 0..n {
            dot_gs += g[j] * s[j];
            dot_gx += g[j] * x[j];
        }
        gap = dot_gx - dot_gs;

        final_iter = t + 1;

        if gap < cfg.tol {
            break;
        }

        // Open-loop step size: γ_t = 2 / (t + 2).
        let gamma = 2.0 / (t as f64 + 2.0);

        // Update: x ← x + γ · (s − x).
        for j in 0..n {
            x[j] += gamma * (s[j] - x[j]);
        }
    }

    Ok(FwResult {
        x,
        iter: final_iter,
        gap,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn quad_grad(x: &[f64], c: &[f64]) -> Vec<f64> {
        x.iter().zip(c.iter()).map(|(xi, ci)| xi - ci).collect()
    }

    #[test]
    fn convergence_on_quadratic_simplex() {
        // min 0.5*||x - c||^2  s.t. x ∈ Δ_3,  c = [0.6, 0.2, 0.2]
        // Solution: x* = c (already in simplex since sum=1 and all ≥ 0).
        let c = vec![0.6_f64, 0.2, 0.2];
        let cc = c.clone();
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cfg = FrankWolfeConfig {
            max_iter: 2000,
            tol: 1e-7,
        };
        let res = frank_wolfe(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg).expect("ok");
        for (xi, ci) in res.x.iter().zip(c.iter()) {
            assert!((xi - ci).abs() < 1e-3, "xi={xi}, ci={ci}");
        }
    }

    #[test]
    fn gap_decreases() {
        // FW gap at more iterations ≤ gap at fewer iterations.
        let c = vec![0.5_f64, 0.3, 0.2];
        let x0 = vec![1.0 / 3.0_f64; 3];

        let cc1 = c.clone();
        let cfg1 = FrankWolfeConfig {
            max_iter: 10,
            tol: 1e-15,
        };
        let res1 = frank_wolfe(&x0, move |x| quad_grad(x, &cc1), simplex_lmo, &cfg1).expect("ok");

        let cc2 = c.clone();
        let cfg2 = FrankWolfeConfig {
            max_iter: 200,
            tol: 1e-15,
        };
        let res2 = frank_wolfe(&x0, move |x| quad_grad(x, &cc2), simplex_lmo, &cfg2).expect("ok");

        assert!(
            res2.gap <= res1.gap + 1e-10,
            "gap_200={} should be ≤ gap_10={}",
            res2.gap,
            res1.gap
        );
    }

    #[test]
    fn x_stays_in_simplex() {
        // Frank-Wolfe iterates are convex combinations of simplex vertices → remain in simplex.
        let c = vec![0.4_f64, 0.3, 0.3];
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cc = c.clone();
        let cfg = FrankWolfeConfig {
            max_iter: 300,
            tol: 1e-9,
        };
        let res = frank_wolfe(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg).expect("ok");
        let sum: f64 = res.x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
        for &xi in &res.x {
            assert!(xi >= -1e-12, "negative component xi={xi}");
        }
    }

    #[test]
    fn simplex_lmo_correct() {
        // argmin_{e_k} g·e_k where g = [0.1, -0.5, 0.3] → index 1 (value -0.5).
        let g = vec![0.1_f64, -0.5, 0.3];
        let s = simplex_lmo(&g);
        assert_eq!(s.len(), 3);
        assert_eq!(s[1], 1.0, "should pick index 1");
        assert_eq!(s[0], 0.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn l1_ball_lmo_correct() {
        // argmin_{±e_k, k=argmax|g|} g·v  where g = [0.1, -0.8, 0.3]
        // Max |g| at index 1 (|−0.8| = 0.8), sign of g[1] is negative → s[1] = +1.
        let g = vec![0.1_f64, -0.8, 0.3];
        let s = l1_ball_lmo(&g);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], 0.0);
        assert!((s[1] - 1.0).abs() < 1e-12, "s[1]={}", s[1]);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn x_shape_preserved() {
        let x0 = vec![0.5_f64, 0.5];
        let cfg = FrankWolfeConfig::default();
        let c = vec![0.7_f64, 0.3];
        let cc = c.clone();
        let res = frank_wolfe(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg).expect("ok");
        assert_eq!(res.x.len(), x0.len());
    }

    #[test]
    fn tol_triggers_early_stop() {
        // Very large tolerance: gap < tol immediately after 1st iter.
        let x0 = vec![1.0_f64, 0.0, 0.0]; // already a simplex vertex
        let cfg = FrankWolfeConfig {
            max_iter: 1000,
            tol: 1e6,
        };
        let res = frank_wolfe(&x0, |x| x.to_vec(), simplex_lmo, &cfg).expect("ok");
        // Should have stopped very early.
        assert!(res.iter < 100, "iter={}", res.iter);
    }

    #[test]
    fn max_iter_1_ok() {
        let x0 = vec![0.5_f64, 0.5];
        let cfg = FrankWolfeConfig {
            max_iter: 1,
            tol: 1e-15,
        };
        let c = vec![0.7_f64, 0.3];
        let cc = c.clone();
        let res = frank_wolfe(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg).expect("ok");
        assert_eq!(res.iter, 1);
        assert_eq!(res.x.len(), 2);
    }

    #[test]
    fn fw_gap_nonneg() {
        // The FW gap must be non-negative at every iterate (it upper-bounds sub-optimality).
        let c = vec![0.5_f64, 0.3, 0.2];
        let x0 = vec![1.0 / 3.0_f64; 3];
        let cc = c.clone();
        let cfg = FrankWolfeConfig {
            max_iter: 500,
            tol: 1e-10,
        };
        let res = frank_wolfe(&x0, move |x| quad_grad(x, &cc), simplex_lmo, &cfg).expect("ok");
        assert!(res.gap >= -1e-10, "gap={}", res.gap);
    }

    #[test]
    fn empty_x_init_error() {
        let cfg = FrankWolfeConfig::default();
        let result = frank_wolfe(&[], |x| x.to_vec(), simplex_lmo, &cfg);
        match result {
            Err(CvxError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    #[test]
    fn lmo_wrong_dim_error() {
        let x0 = vec![0.5_f64, 0.5];
        let cfg = FrankWolfeConfig::default();
        // LMO returns wrong length.
        let result = frank_wolfe(&x0, |x| x.to_vec(), |_| vec![1.0], &cfg);
        match result {
            Err(CvxError::DimensionMismatch { .. }) => {}
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn l1_ball_lmo_positive_grad() {
        // g = [0.3, 0.9, -0.1] → argmax|g| = index 1 (0.9), sign positive → s[1] = -1.
        let g = vec![0.3_f64, 0.9, -0.1];
        let s = l1_ball_lmo(&g);
        assert!((s[1] - (-1.0)).abs() < 1e-12, "s[1]={}", s[1]);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[2], 0.0);
    }
}
