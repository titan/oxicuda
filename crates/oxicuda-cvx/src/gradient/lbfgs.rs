//! Limited-memory BFGS (L-BFGS) for large-scale smooth unconstrained convex optimisation.
//!
//! Implements the Nocedal (1980) / Liu-Nocedal (1989) compact inverse-Hessian approximation
//! with a two-loop recursion and a strong-Wolfe bracket line search.
//!
//! References:
//! - Nocedal & Wright (2006), "Numerical Optimization", Chapter 7, Algorithm 7.4.
//! - Liu & Nocedal (1989), "On the limited memory method for large scale optimization".

use std::collections::VecDeque;

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the L-BFGS solver.
#[derive(Debug, Clone)]
pub struct LbfgsConfig {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// History length (number of (s, y) pairs retained). Must be ≥ 1.
    pub m: usize,
    /// Convergence tolerance: stop when L∞-norm of gradient < `tol`.
    pub tol: f64,
    /// Armijo sufficient-decrease constant c₁ ∈ (0, 1).
    pub c1: f64,
    /// Strong-Wolfe curvature constant c₂ ∈ (c₁, 1).
    pub c2: f64,
    /// Maximum line-search bracket iterations per outer step.
    pub max_ls_iter: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            m: 10,
            tol: 1e-8,
            c1: 1e-4,
            c2: 0.9,
            max_ls_iter: 25,
        }
    }
}

/// Output of a successful L-BFGS run.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Number of outer iterations performed.
    pub n_iter: usize,
    /// Whether the solver converged within `tol`.
    pub converged: bool,
    /// L2 norm of the final gradient.
    pub final_grad_norm: f64,
}

// ---------------------------------------------------------------------------
// Helper: plain (unchecked) dot product over matched slices
// ---------------------------------------------------------------------------

#[inline]
fn dot_unchecked(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ---------------------------------------------------------------------------
// Two-loop recursion: Nocedal & Wright §7.4, Algorithm 7.4
// ---------------------------------------------------------------------------

fn two_loop(
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
    rho_hist: &VecDeque<f64>,
    g: &[f64],
) -> Vec<f64> {
    let hist_len = s_hist.len();
    let n = g.len();
    let mut q = g.to_vec();
    let mut alpha = vec![0.0_f64; hist_len];

    // Backward pass
    for i in (0..hist_len).rev() {
        alpha[i] = rho_hist[i] * dot_unchecked(&s_hist[i], &q);
        let ai = alpha[i];
        for j in 0..n {
            q[j] -= ai * y_hist[i][j];
        }
    }

    // Initial Hessian scale H₀ = γ I where γ = (s[-1]ᵀ y[-1]) / (y[-1]ᵀ y[-1])
    let gamma = if hist_len > 0 {
        let last = hist_len - 1;
        let sy = dot_unchecked(&s_hist[last], &y_hist[last]);
        let yy = dot_unchecked(&y_hist[last], &y_hist[last]);
        (sy / (yy + 1e-30)).clamp(1e-12, 1e12)
    } else {
        1.0_f64
    };

    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

    // Forward pass
    for i in 0..hist_len {
        let beta = rho_hist[i] * dot_unchecked(&y_hist[i], &r);
        let diff = alpha[i] - beta;
        for j in 0..n {
            r[j] += diff * s_hist[i][j];
        }
    }

    // Descent direction d = -r
    r.iter_mut().for_each(|ri| *ri = -*ri);
    r
}

// ---------------------------------------------------------------------------
// Strong-Wolfe line search
// ---------------------------------------------------------------------------

/// Phase 2 (zoom): bisect [a_lo, a_hi] until strong-Wolfe conditions are met.
/// Returns the accepted step length.
fn zoom<F, G>(
    x: &[f64],
    d: &[f64],
    phi0: f64,
    dphi0: f64,
    a_lo_in: f64,
    a_hi_in: f64,
    f: &F,
    grad_f: &G,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> CvxResult<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    let mut a_lo = a_lo_in;
    let mut a_hi = a_hi_in;
    let mut phi_lo = {
        let x_trial: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + a_lo * di)
            .collect();
        f(&x_trial)
    };

    for _ in 0..max_iter {
        let a_j = 0.5 * (a_lo + a_hi);
        let x_j: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + a_j * di)
            .collect();
        let phi_j = f(&x_j);

        if phi_j > phi0 + c1 * a_j * dphi0 || phi_j >= phi_lo {
            a_hi = a_j;
        } else {
            let g_j = grad_f(&x_j)?;
            let dphi_j: f64 = g_j.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();
            if dphi_j.abs() <= -c2 * dphi0 {
                return Ok(a_j);
            }
            if dphi_j * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
            }
            a_lo = a_j;
            phi_lo = phi_j;
        }

        // Interval collapsed
        if (a_hi - a_lo).abs() < 1e-15 {
            return Ok(a_lo);
        }
    }

    // Return best found
    Ok(a_lo)
}

/// Strong-Wolfe line search (bracket phase + zoom).
/// `phi0 = f(x)`, `dphi0 = dot(grad_f(x), d)` (must be < 0).
fn wolfe_line_search<F, G>(
    x: &[f64],
    d: &[f64],
    phi0: f64,
    dphi0: f64,
    f: &F,
    grad_f: &G,
    c1: f64,
    c2: f64,
    max_ls_iter: usize,
) -> CvxResult<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    let initial_step = 1.0_f64;
    let max_step = 1e10_f64;
    let mut alpha_lo = 0.0_f64;
    let mut alpha_hi = initial_step;
    let mut phi_lo = phi0;

    for iter in 0..max_ls_iter {
        let x_hi: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha_hi * di)
            .collect();
        let phi_hi = f(&x_hi);

        if phi_hi > phi0 + c1 * alpha_hi * dphi0 || (iter > 0 && phi_hi >= phi_lo) {
            let accepted = zoom(
                x,
                d,
                phi0,
                dphi0,
                alpha_lo,
                alpha_hi,
                f,
                grad_f,
                c1,
                c2,
                max_ls_iter,
            )?;
            return Ok(accepted);
        }

        let g_hi = grad_f(&x_hi)?;
        let dphi_hi: f64 = g_hi.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();

        if dphi_hi.abs() <= -c2 * dphi0 {
            return Ok(alpha_hi);
        }

        if dphi_hi >= 0.0 {
            let accepted = zoom(
                x,
                d,
                phi0,
                dphi0,
                alpha_hi,
                alpha_lo,
                f,
                grad_f,
                c1,
                c2,
                max_ls_iter,
            )?;
            return Ok(accepted);
        }

        phi_lo = phi_hi;
        alpha_lo = alpha_hi;
        alpha_hi = (2.0 * alpha_hi).min(max_step);
    }

    // Armijo check at current best alpha_lo
    let x_lo: Vec<f64> = x
        .iter()
        .zip(d.iter())
        .map(|(xi, di)| xi + alpha_lo * di)
        .collect();
    let phi_lo_check = f(&x_lo);
    if phi_lo_check <= phi0 + c1 * alpha_lo * dphi0 {
        return Ok(alpha_lo);
    }

    Err(CvxError::LineSearchFailed(format!(
        "strong-Wolfe line search did not satisfy Armijo after {max_ls_iter} bracket iterations"
    )))
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run L-BFGS on an unconstrained smooth convex problem.
///
/// # Arguments
/// - `x0`: starting point (non-empty).
/// - `f`: objective function `ℝⁿ → ℝ`.
/// - `grad_f`: gradient `ℝⁿ → CvxResult<ℝⁿ>`.
/// - `cfg`: solver configuration.
///
/// # Errors
/// Returns [`CvxError::EmptyInput`] if `x0` is empty, [`CvxError::InvalidParameter`]
/// for bad configuration values, [`CvxError::LineSearchFailed`] if the Wolfe line
/// search fails to satisfy the Armijo condition, or any error propagated from `grad_f`.
pub fn lbfgs<F, G>(x0: &[f64], f: F, grad_f: G, cfg: &LbfgsConfig) -> CvxResult<LbfgsResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    // ----- Validation -----
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if cfg.m == 0 {
        return Err(CvxError::InvalidParameter("m must be ≥ 1".to_owned()));
    }
    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0, got {}",
            cfg.tol
        )));
    }
    if cfg.c1 <= 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "c1 must be > 0, got {}",
            cfg.c1
        )));
    }
    if cfg.c2 <= cfg.c1 {
        return Err(CvxError::InvalidParameter(format!(
            "c2 must be > c1 ({} > {} violated)",
            cfg.c2, cfg.c1
        )));
    }
    if cfg.c2 >= 1.0 {
        return Err(CvxError::InvalidParameter(format!(
            "c2 must be < 1, got {}",
            cfg.c2
        )));
    }

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad_f(&x)?;
    if g.len() != n {
        return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
    }

    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(cfg.m);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(cfg.m);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(cfg.m);

    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..cfg.max_iter {
        // Convergence check (L∞ norm of gradient)
        let g_inf = g.iter().map(|gi| gi.abs()).fold(0.0_f64, f64::max);
        if g_inf < cfg.tol {
            converged = true;
            break;
        }

        // Two-loop recursion → descent direction
        let d = two_loop(&s_hist, &y_hist, &rho_hist, &g);

        // Directional derivative for line search
        let dphi0: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();

        // Safeguard: if direction is not descent, reset to steepest descent
        let d = if dphi0 >= 0.0 {
            g.iter().map(|gi| -*gi).collect::<Vec<_>>()
        } else {
            d
        };
        let dphi0_safe: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();

        let phi0 = f(&x);

        // Strong-Wolfe line search
        let alpha = wolfe_line_search(
            &x,
            &d,
            phi0,
            dphi0_safe,
            &f,
            &grad_f,
            cfg.c1,
            cfg.c2,
            cfg.max_ls_iter,
        )?;

        // Update x
        let s: Vec<f64> = d.iter().map(|di| alpha * di).collect();
        let x_new: Vec<f64> = x.iter().zip(s.iter()).map(|(xi, si)| xi + si).collect();

        // New gradient
        let g_new = grad_f(&x_new)?;
        if g_new.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: g_new.len(),
                b: n,
            });
        }

        let y: Vec<f64> = g_new
            .iter()
            .zip(g.iter())
            .map(|(gi_new, gi)| gi_new - gi)
            .collect();

        // Curvature condition: only update history if s·y > ε·‖y‖
        let sy = dot_unchecked(&s, &y);
        let y_nrm = norm2(&y);
        if sy > 1e-15 * y_nrm {
            if s_hist.len() == cfg.m {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
            rho_hist.push_back(1.0 / sy);
        }

        x = x_new;
        g = g_new;
        n_iter += 1;
    }

    let final_grad_norm = norm2(&g);

    Ok(LbfgsResult {
        x,
        n_iter,
        converged,
        final_grad_norm,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Sum-of-squares centred at `target`: f(x) = 0.5 ‖x − target‖².
    fn quad_f(x: &[f64], target: &[f64]) -> f64 {
        x.iter()
            .zip(target.iter())
            .map(|(xi, ti)| 0.5 * (xi - ti).powi(2))
            .sum()
    }

    fn quad_g(x: &[f64], target: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(x.iter()
            .zip(target.iter())
            .map(|(xi, ti)| xi - ti)
            .collect())
    }

    macro_rules! make_quad_closures {
        ($target:expr) => {{
            let t: Vec<f64> = $target;
            let t_f = t.clone();
            let t_g = t.clone();
            (
                move |x: &[f64]| quad_f(x, &t_f),
                move |x: &[f64]| quad_g(x, &t_g),
            )
        }};
    }

    // Rosenbrock: f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2
    fn rosenbrock_f(x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn rosenbrock_g(x: &[f64]) -> CvxResult<Vec<f64>> {
        let df0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let df1 = 200.0 * (x[1] - x[0] * x[0]);
        Ok(vec![df0, df1])
    }

    #[test]
    fn quadratic_2d() {
        let (f, grad_f) = make_quad_closures!(vec![0.0, 0.0]);
        let cfg = LbfgsConfig::default();
        let res = lbfgs(&[3.0, -4.0], f, grad_f, &cfg).expect("ok");
        assert!(res.converged);
        assert!(res.x[0].abs() < 1e-5, "x[0]={}", res.x[0]);
        assert!(res.x[1].abs() < 1e-5, "x[1]={}", res.x[1]);
    }

    #[test]
    fn quadratic_10d() {
        let target: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (f, grad_f) = make_quad_closures!(target.clone());
        let cfg = LbfgsConfig::default();
        let x0: Vec<f64> = vec![0.0; 10];
        let res = lbfgs(&x0, f, grad_f, &cfg).expect("ok");
        assert!(res.converged);
        for (xi, ti) in res.x.iter().zip(target.iter()) {
            assert!((xi - ti).abs() < 1e-5, "xi={xi}, ti={ti}");
        }
    }

    #[test]
    fn rosenbrock_converges() {
        let cfg = LbfgsConfig {
            max_iter: 2000,
            tol: 1e-6,
            ..LbfgsConfig::default()
        };
        let res = lbfgs(&[-1.0, -1.0], rosenbrock_f, rosenbrock_g, &cfg).expect("ok");
        assert!(
            (res.x[0] - 1.0).abs() < 1e-3 && (res.x[1] - 1.0).abs() < 1e-3,
            "x={:?}",
            res.x
        );
    }

    #[test]
    fn ill_conditioned_quadratic() {
        // f(x) = 0.5 * Σ i * xi^2  with condition number 100
        let f = |x: &[f64]| -> f64 {
            x.iter()
                .enumerate()
                .map(|(i, xi)| 0.5 * (i + 1) as f64 * xi * xi)
                .sum()
        };
        let grad_f = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter()
                .enumerate()
                .map(|(i, xi)| (i + 1) as f64 * xi)
                .collect())
        };
        let x0: Vec<f64> = vec![1.0; 10];
        let cfg = LbfgsConfig {
            max_iter: 500,
            tol: 1e-7,
            ..LbfgsConfig::default()
        };
        let res = lbfgs(&x0, f, grad_f, &cfg).expect("ok");
        assert!(
            res.converged,
            "did not converge; final_grad_norm={}",
            res.final_grad_norm
        );
        for xi in &res.x {
            assert!(xi.abs() < 1e-4, "xi={xi}");
        }
    }

    #[test]
    fn result_converged_flag() {
        let (f, grad_f) = make_quad_closures!(vec![1.0]);
        let cfg = LbfgsConfig::default();
        let res = lbfgs(&[0.0], f, grad_f, &cfg).expect("ok");
        assert!(res.converged);
    }

    #[test]
    fn n_iter_finite() {
        let (f, grad_f) = make_quad_closures!(vec![1.0, 2.0]);
        let cfg = LbfgsConfig::default();
        let res = lbfgs(&[0.0, 0.0], f, grad_f, &cfg).expect("ok");
        assert!(res.n_iter < cfg.max_iter);
    }

    #[test]
    fn final_grad_norm_small() {
        let (f, grad_f) = make_quad_closures!(vec![0.0, 0.0, 0.0]);
        let cfg = LbfgsConfig::default();
        let res = lbfgs(&[5.0, -3.0, 2.0], f, grad_f, &cfg).expect("ok");
        assert!(
            res.final_grad_norm < cfg.tol * 10.0,
            "norm={}",
            res.final_grad_norm
        );
    }

    #[test]
    fn history_size_m1() {
        let (f, grad_f) = make_quad_closures!(vec![1.0, 2.0]);
        let cfg = LbfgsConfig {
            m: 1,
            ..LbfgsConfig::default()
        };
        let res = lbfgs(&[0.0, 0.0], f, grad_f, &cfg).expect("ok");
        assert!(res.converged, "m=1 did not converge");
    }

    #[test]
    fn history_size_m20() {
        let (f, grad_f) = make_quad_closures!(vec![1.0, 2.0]);
        let cfg = LbfgsConfig {
            m: 20,
            ..LbfgsConfig::default()
        };
        let res = lbfgs(&[0.0, 0.0], f, grad_f, &cfg).expect("ok");
        assert!(res.converged, "m=20 did not converge");
    }

    #[test]
    fn empty_x0_err() {
        let (f, grad_f) = make_quad_closures!(vec![]);
        let cfg = LbfgsConfig::default();
        match lbfgs(&[], f, grad_f, &cfg) {
            Err(CvxError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {:?}", other),
        }
    }

    #[test]
    fn invalid_m_err() {
        let (f, grad_f) = make_quad_closures!(vec![0.0]);
        let cfg = LbfgsConfig {
            m: 0,
            ..LbfgsConfig::default()
        };
        match lbfgs(&[1.0], f, grad_f, &cfg) {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn invalid_tol_err() {
        let (f, grad_f) = make_quad_closures!(vec![0.0]);
        let cfg = LbfgsConfig {
            tol: 0.0,
            ..LbfgsConfig::default()
        };
        match lbfgs(&[1.0], f, grad_f, &cfg) {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn single_variable() {
        let f = |x: &[f64]| -> f64 { (x[0] - 5.0).powi(2) };
        let grad_f = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![2.0 * (x[0] - 5.0)]) };
        let cfg = LbfgsConfig::default();
        let res = lbfgs(&[0.0], f, grad_f, &cfg).expect("ok");
        assert!(res.converged);
        assert!((res.x[0] - 5.0).abs() < 1e-5, "x={}", res.x[0]);
    }

    #[test]
    fn result_x_length() {
        let n = 7;
        let (f, grad_f) = make_quad_closures!(vec![0.0; n]);
        let cfg = LbfgsConfig::default();
        let x0 = vec![1.0; n];
        let res = lbfgs(&x0, f, grad_f, &cfg).expect("ok");
        assert_eq!(res.x.len(), n);
    }

    #[test]
    fn zero_gradient_start() {
        // x0 is already the optimum → gradient is 0 at the start
        let (f, grad_f) = make_quad_closures!(vec![3.0, 3.0]);
        let cfg = LbfgsConfig::default();
        let res = lbfgs(&[3.0, 3.0], f, grad_f, &cfg).expect("ok");
        // Converged with 0 outer iterations (gradient check fires immediately)
        assert!(res.converged);
        assert_eq!(res.n_iter, 0);
    }

    #[test]
    fn c1_c2_violation_err() {
        let (f, grad_f) = make_quad_closures!(vec![0.0]);
        // c2 < c1 — violates curvature-constant ordering
        let cfg = LbfgsConfig {
            c1: 0.5,
            c2: 0.1,
            ..LbfgsConfig::default()
        };
        match lbfgs(&[1.0], f, grad_f, &cfg) {
            Err(CvxError::InvalidParameter(_)) => {}
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn sum_of_squares_large() {
        let n = 100;
        let (f, grad_f) = make_quad_closures!(vec![0.0; n]);
        let cfg = LbfgsConfig::default();
        let x0: Vec<f64> = (0..n).map(|i| (i as f64 % 7.0) - 3.0).collect();
        let res = lbfgs(&x0, f, grad_f, &cfg).expect("ok");
        assert!(res.converged, "100-d sum-of-squares did not converge");
        assert!(res.final_grad_norm < cfg.tol * 100.0);
    }
}
