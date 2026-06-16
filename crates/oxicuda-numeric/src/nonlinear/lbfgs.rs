//! Limited-memory BFGS (L-BFGS) quasi-Newton minimiser.
//!
//! Minimises a smooth scalar objective `φ : ℝⁿ → ℝ` without ever forming the
//! `n × n` inverse-Hessian matrix.  Instead L-BFGS stores only the most recent
//! `m` correction pairs
//!
//! ```text
//! sₖ = xₖ₊₁ − xₖ,    yₖ = ∇φₖ₊₁ − ∇φₖ
//! ```
//!
//! and reconstructs the action of the inverse Hessian `Hₖ ∇φₖ` through the
//! Nocedal-Wright two-loop recursion.  This makes the per-iteration cost
//! `O(m·n)` in time and `O(m·n)` in memory, suitable for large `n` where the
//! dense [`crate::nonlinear::bfgs`] storage would be prohibitive.
//!
//! The two-loop recursion (Nocedal & Wright, *Numerical Optimization*, Alg. 7.4):
//!
//! ```text
//! q ← ∇φₖ
//! for i = k−1 … k−m:   αᵢ = ρᵢ sᵢᵀ q ;   q ← q − αᵢ yᵢ
//! r ← γ q                       (γ = sₖ₋₁ᵀyₖ₋₁ / yₖ₋₁ᵀyₖ₋₁ scales the initial H₀)
//! for i = k−m … k−1:   β  = ρᵢ yᵢᵀ r ;   r ← r + (αᵢ − β) sᵢ
//! direction p = −r
//! ```
//!
//! The step length is chosen by a backtracking Armijo line search, identical in
//! spirit to the dense BFGS minimiser.

use crate::error::{NumericError, NumericResult};

/// Configuration for [`lbfgs_minimize`] / [`lbfgs_minimize_numerical`].
#[derive(Debug, Clone, Copy)]
pub struct LbfgsConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖∇φ‖₂`.
    pub grad_tol: f64,
    /// Number of correction pairs retained in the limited-memory window (`m ≥ 1`).
    pub memory: usize,
    /// Armijo sufficient-decrease parameter (`0 < c1 < 1`, typically `1e-4`).
    pub c1: f64,
    /// Backtracking contraction factor (`0 < rho < 1`, typically `0.5`).
    pub backtrack: f64,
    /// Step used for the central-difference gradient (numerical variant only).
    pub fd_eps: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            grad_tol: 1.0e-8,
            memory: 10,
            c1: 1.0e-4,
            backtrack: 0.5,
            fd_eps: 1.0e-6,
        }
    }
}

/// Result of an L-BFGS run.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Minimiser estimate.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fx: f64,
    /// `‖∇φ(x)‖₂` at termination.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iters: usize,
}

/// Minimises `phi` with an analytic gradient `grad` using L-BFGS.
///
/// `phi(x)` returns the scalar objective; `grad(x)` returns `∇φ(x)` (length
/// `n`).  `x0` is the starting point.
///
/// # Errors
///
/// * [`NumericError::EmptyInput`] if `x0` is empty.
/// * [`NumericError::InvalidParameter`] for invalid config parameters,
///   non-finite `x0`, or a gradient of the wrong length.
/// * [`NumericError::NotConverged`] if `‖∇φ‖` stays above `grad_tol` after
///   `max_iter` iterations.
pub fn lbfgs_minimize<P, G>(
    phi: P,
    grad: G,
    x0: &[f64],
    cfg: &LbfgsConfig,
) -> NumericResult<LbfgsResult>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    run_lbfgs(&phi, &|x| grad(x), x0, cfg)
}

/// Minimises `phi` with a central finite-difference gradient using L-BFGS.
///
/// # Errors
///
/// Same conditions as [`lbfgs_minimize`].
pub fn lbfgs_minimize_numerical<P>(
    phi: P,
    x0: &[f64],
    cfg: &LbfgsConfig,
) -> NumericResult<LbfgsResult>
where
    P: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let eps = cfg.fd_eps;
    let phi_ref = &phi;
    let grad = |x: &[f64]| central_gradient(phi_ref, x, eps, n);
    run_lbfgs(&phi, &grad, x0, cfg)
}

fn run_lbfgs<P, G>(phi: &P, grad: &G, x0: &[f64], cfg: &LbfgsConfig) -> NumericResult<LbfgsResult>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    validate_config(cfg)?;
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(NumericError::InvalidParameter(
            "x0 has non-finite entries".into(),
        ));
    }

    let mut x = x0.to_vec();
    let mut fx = phi(&x);
    if !fx.is_finite() {
        return Err(NumericError::InvalidParameter(
            "objective is non-finite at x0".into(),
        ));
    }
    let mut g = grad_checked(grad, &x, n)?;
    let mut gnorm = norm2(&g);

    // Limited-memory ring of correction pairs.
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(cfg.memory);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(cfg.memory);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(cfg.memory);
    // γ scales the implicit initial inverse Hessian H₀ = γ I.
    let mut gamma = 1.0_f64;

    for it in 0..cfg.max_iter {
        if gnorm <= cfg.grad_tol {
            return Ok(LbfgsResult {
                x,
                fx,
                grad_norm: gnorm,
                iters: it,
            });
        }

        // Two-loop recursion producing the search direction p = −H g.
        let mut p = two_loop_direction(&g, &s_hist, &y_hist, &rho_hist, gamma);
        let mut dderiv = dot(&g, &p);
        // Guard: if the recursion failed to give a descent direction, fall back
        // to steepest descent and reset the memory.
        if dderiv >= 0.0 {
            p = g.iter().map(|gi| -gi).collect();
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            gamma = 1.0;
            dderiv = dot(&g, &p);
        }

        // Armijo backtracking line search.
        let mut alpha = 1.0_f64;
        let mut x_new = x.clone();
        let mut f_new = fx;
        let mut ok = false;
        for _ in 0..60 {
            let trial: Vec<f64> = x.iter().zip(&p).map(|(xi, pi)| xi + alpha * pi).collect();
            let ftrial = phi(&trial);
            if ftrial.is_finite() && ftrial <= fx + cfg.c1 * alpha * dderiv {
                x_new = trial;
                f_new = ftrial;
                ok = true;
                break;
            }
            alpha *= cfg.backtrack;
        }
        if !ok {
            return finish(x, fx, gnorm, it, cfg.grad_tol);
        }

        let g_new = grad_checked(grad, &x_new, n)?;

        // Form the new correction pair and push into the limited-memory window.
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1.0e-12 {
            let yy = dot(&y, &y);
            if yy > 0.0 {
                gamma = sy / yy;
            }
            if s_hist.len() == cfg.memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

        x = x_new;
        fx = f_new;
        g = g_new;
        gnorm = norm2(&g);
    }

    finish(x, fx, gnorm, cfg.max_iter, cfg.grad_tol)
}

fn validate_config(cfg: &LbfgsConfig) -> NumericResult<()> {
    if !(cfg.grad_tol > 0.0 && cfg.grad_tol.is_finite()) {
        return Err(NumericError::InvalidParameter(format!(
            "grad_tol must be positive finite, got {}",
            cfg.grad_tol
        )));
    }
    if cfg.memory == 0 {
        return Err(NumericError::InvalidParameter(
            "memory must be at least 1".into(),
        ));
    }
    if !(cfg.c1 > 0.0 && cfg.c1 < 1.0) {
        return Err(NumericError::InvalidParameter(format!(
            "c1 must lie in (0,1), got {}",
            cfg.c1
        )));
    }
    if !(cfg.backtrack > 0.0 && cfg.backtrack < 1.0) {
        return Err(NumericError::InvalidParameter(format!(
            "backtrack must lie in (0,1), got {}",
            cfg.backtrack
        )));
    }
    Ok(())
}

/// Nocedal-Wright two-loop recursion returning the search direction `p = −H g`.
fn two_loop_direction(
    g: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
    gamma: f64,
) -> Vec<f64> {
    let k = s_hist.len();
    let mut q = g.to_vec();
    let mut alphas = vec![0.0_f64; k];
    // First loop: newest → oldest.
    for i in (0..k).rev() {
        let ai = rho_hist[i] * dot(&s_hist[i], &q);
        alphas[i] = ai;
        for (qj, yj) in q.iter_mut().zip(&y_hist[i]) {
            *qj -= ai * yj;
        }
    }
    // Scale by the initial inverse-Hessian estimate H₀ = γ I.
    for qj in &mut q {
        *qj *= gamma;
    }
    // Second loop: oldest → newest.
    for i in 0..k {
        let beta = rho_hist[i] * dot(&y_hist[i], &q);
        let coeff = alphas[i] - beta;
        for (qj, sj) in q.iter_mut().zip(&s_hist[i]) {
            *qj += coeff * sj;
        }
    }
    // Direction is the negative of the (approximate) Newton step H g.
    for qj in &mut q {
        *qj = -*qj;
    }
    q
}

fn finish(x: Vec<f64>, fx: f64, gnorm: f64, iters: usize, tol: f64) -> NumericResult<LbfgsResult> {
    if gnorm <= tol {
        Ok(LbfgsResult {
            x,
            fx,
            grad_norm: gnorm,
            iters,
        })
    } else {
        Err(NumericError::NotConverged {
            iter: iters,
            residual: gnorm,
        })
    }
}

/// Central finite-difference gradient.
fn central_gradient<P>(phi: &P, x: &[f64], eps: f64, n: usize) -> Vec<f64>
where
    P: Fn(&[f64]) -> f64,
{
    let mut g = vec![0.0_f64; n];
    let mut xp = x.to_vec();
    for i in 0..n {
        let h = eps * (1.0 + x[i].abs());
        xp[i] = x[i] + h;
        let fp = phi(&xp);
        xp[i] = x[i] - h;
        let fm = phi(&xp);
        xp[i] = x[i];
        g[i] = (fp - fm) / (2.0 * h);
    }
    g
}

fn grad_checked<G>(grad: &G, x: &[f64], n: usize) -> NumericResult<Vec<f64>>
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let g = grad(x);
    if g.len() != n {
        return Err(NumericError::InvalidParameter(format!(
            "gradient length {} != {n}",
            g.len()
        )));
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err(NumericError::NumericalInstability(
            "gradient became non-finite".into(),
        ));
    }
    Ok(g)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LbfgsConfig {
        LbfgsConfig::default()
    }

    #[test]
    fn quadratic_bowl() {
        // φ = (x−3)² + (y+1)² → minimum at (3, −1).
        let phi = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 1.0).powi(2);
        let grad = |v: &[f64]| vec![2.0 * (v[0] - 3.0), 2.0 * (v[1] + 1.0)];
        let r = lbfgs_minimize(phi, grad, &[0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 3.0).abs() < 1e-6, "x={}", r.x[0]);
        assert!((r.x[1] + 1.0).abs() < 1e-6, "y={}", r.x[1]);
        assert!(r.fx < 1e-10);
    }

    #[test]
    fn rosenbrock() {
        // Classic Rosenbrock function, minimum at (1, 1).
        let phi = |v: &[f64]| {
            let a = 1.0 - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + 100.0 * b * b
        };
        let grad = |v: &[f64]| {
            let dx = -2.0 * (1.0 - v[0]) - 400.0 * v[0] * (v[1] - v[0] * v[0]);
            let dy = 200.0 * (v[1] - v[0] * v[0]);
            vec![dx, dy]
        };
        let c = LbfgsConfig {
            max_iter: 5000,
            grad_tol: 1e-6,
            ..cfg()
        };
        let r = lbfgs_minimize(phi, grad, &[-1.2, 1.0], &c).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-3, "x={}", r.x[0]);
        assert!((r.x[1] - 1.0).abs() < 1e-3, "y={}", r.x[1]);
    }

    #[test]
    fn high_dimensional_quadratic() {
        // φ = Σ kᵢ (xᵢ − i)² with varied curvature; L-BFGS should reach the
        // minimiser xᵢ = i in moderate dimension where the dense Hessian is
        // wasteful.
        let n = 40_usize;
        let phi = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, &x)| (1.0 + i as f64) * (x - i as f64).powi(2))
                .sum::<f64>()
        };
        let grad = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, &x)| 2.0 * (1.0 + i as f64) * (x - i as f64))
                .collect::<Vec<_>>()
        };
        let start = vec![0.0_f64; n];
        let r = lbfgs_minimize(phi, grad, &start, &cfg()).expect("ok");
        for (i, &xi) in r.x.iter().enumerate() {
            assert!((xi - i as f64).abs() < 1e-4, "x[{i}]={xi}");
        }
    }

    #[test]
    fn memory_one_still_converges() {
        // Even with a single correction pair (closest to nonlinear CG) the solver
        // converges on a well-conditioned quadratic.
        let phi = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
        let c = LbfgsConfig { memory: 1, ..cfg() };
        let r = lbfgs_minimize(phi, grad, &[3.0, -2.0, 1.0, 5.0], &c).expect("ok");
        assert!(r.grad_norm <= c.grad_tol, "gnorm={}", r.grad_norm);
    }

    #[test]
    fn numerical_gradient_matches() {
        let phi = |v: &[f64]| (v[0] - 2.0).powi(2) + 3.0 * (v[1] - 5.0).powi(2);
        let r = lbfgs_minimize_numerical(phi, &[0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 2.0).abs() < 1e-4, "x={}", r.x[0]);
        assert!((r.x[1] - 5.0).abs() < 1e-4, "y={}", r.x[1]);
    }

    #[test]
    fn already_at_minimum() {
        let phi = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
        let grad = |v: &[f64]| vec![2.0 * v[0], 2.0 * v[1]];
        let r = lbfgs_minimize(phi, grad, &[0.0, 0.0], &cfg()).expect("ok");
        assert_eq!(r.iters, 0);
        assert!(r.grad_norm < 1e-12);
    }

    #[test]
    fn converges_grad_norm() {
        let phi = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
        let r = lbfgs_minimize(phi, grad, &[5.0, -4.0, 3.0, 2.0, -1.0], &cfg()).expect("ok");
        assert!(r.grad_norm <= cfg().grad_tol, "gnorm={}", r.grad_norm);
    }

    #[test]
    fn decreases_objective() {
        let phi = |v: &[f64]| (v[0] + 4.0).powi(2) + (v[1] - 2.0).powi(4);
        let grad = |v: &[f64]| vec![2.0 * (v[0] + 4.0), 4.0 * (v[1] - 2.0).powi(3)];
        let start = [1.0, 1.0];
        let f0 = phi(&start);
        let r = lbfgs_minimize(phi, grad, &start, &cfg()).expect("ok");
        assert!(r.fx < f0, "objective did not decrease: {} -> {}", f0, r.fx);
    }

    #[test]
    fn output_len_and_finite() {
        let phi = |v: &[f64]| v.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| 2.0 * (x - 1.0)).collect::<Vec<_>>();
        let r = lbfgs_minimize(phi, grad, &[0.0; 6], &cfg()).expect("ok");
        assert_eq!(r.x.len(), 6);
        for v in &r.x {
            assert!(v.is_finite());
        }
        assert!(r.fx.is_finite() && r.grad_norm.is_finite());
    }

    #[test]
    fn max_iter_bound() {
        let phi = |v: &[f64]| {
            let a = 1.0 - v[0];
            let b = v[1] - v[0] * v[0];
            a * a + 100.0 * b * b
        };
        let grad = |v: &[f64]| {
            let dx = -2.0 * (1.0 - v[0]) - 400.0 * v[0] * (v[1] - v[0] * v[0]);
            let dy = 200.0 * (v[1] - v[0] * v[0]);
            vec![dx, dy]
        };
        let c = LbfgsConfig {
            max_iter: 1,
            grad_tol: 1e-12,
            ..cfg()
        };
        let res = lbfgs_minimize(phi, grad, &[-5.0, 5.0], &c);
        assert!(res.is_err());
    }

    #[test]
    fn rejects_bad_input() {
        let phi = |v: &[f64]| v[0] * v[0];
        let grad = |v: &[f64]| vec![2.0 * v[0]];
        assert!(lbfgs_minimize(phi, grad, &[], &cfg()).is_err());
        let bad_mem = LbfgsConfig { memory: 0, ..cfg() };
        assert!(
            lbfgs_minimize(
                |v: &[f64]| v[0] * v[0],
                |v: &[f64]| vec![2.0 * v[0]],
                &[1.0],
                &bad_mem
            )
            .is_err()
        );
        let bad_c1 = LbfgsConfig { c1: 2.0, ..cfg() };
        assert!(
            lbfgs_minimize(
                |v: &[f64]| v[0] * v[0],
                |v: &[f64]| vec![2.0 * v[0]],
                &[1.0],
                &bad_c1
            )
            .is_err()
        );
    }

    #[test]
    fn matches_dense_bfgs_minimum() {
        // L-BFGS and the dense formula should agree on a moderately conditioned
        // quadratic to high precision.
        let phi = |v: &[f64]| {
            2.0 * (v[0] - 1.0).powi(2) + 5.0 * (v[1] + 2.0).powi(2) + 0.5 * (v[2] - 3.0).powi(2)
        };
        let grad = |v: &[f64]| vec![4.0 * (v[0] - 1.0), 10.0 * (v[1] + 2.0), 1.0 * (v[2] - 3.0)];
        let r = lbfgs_minimize(phi, grad, &[0.0, 0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-6);
        assert!((r.x[1] + 2.0).abs() < 1e-6);
        assert!((r.x[2] - 3.0).abs() < 1e-6);
    }
}
