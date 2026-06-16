//! BFGS quasi-Newton unconstrained minimiser.
//!
//! Minimises a smooth scalar objective `φ : ℝⁿ → ℝ` using the
//! Broyden-Fletcher-Goldfarb-Shanno update of the inverse Hessian
//! approximation `H ≈ (∇²φ)⁻¹`, combined with a backtracking Armijo line
//! search.  The search direction is `p = −H ∇φ`; after each accepted step the
//! inverse Hessian is updated by the rank-2 BFGS formula
//!
//! ```text
//! ρ = 1 / (yᵀ s)
//! H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
//! ```
//!
//! where `s = x_{k+1} − x_k` and `y = ∇φ_{k+1} − ∇φ_k`.  The gradient is taken
//! analytically when supplied, or approximated by central finite differences.

use crate::error::{NumericError, NumericResult};

/// Configuration for [`bfgs_minimize`] / [`bfgs_minimize_numerical`].
#[derive(Debug, Clone, Copy)]
pub struct BfgsConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖∇φ‖₂`.
    pub grad_tol: f64,
    /// Armijo sufficient-decrease parameter (`0 < c1 < 1`, typically `1e-4`).
    pub c1: f64,
    /// Backtracking contraction factor (`0 < rho < 1`, typically `0.5`).
    pub backtrack: f64,
    /// Step used for the central-difference gradient (numerical variant only).
    pub fd_eps: f64,
}

impl Default for BfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            grad_tol: 1.0e-8,
            c1: 1.0e-4,
            backtrack: 0.5,
            fd_eps: 1.0e-6,
        }
    }
}

/// Result of a BFGS run.
#[derive(Debug, Clone)]
pub struct BfgsResult {
    /// Minimiser estimate.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fx: f64,
    /// `‖∇φ(x)‖₂` at termination.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iters: usize,
}

/// Minimises `phi` with an analytic gradient `grad`.
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
pub fn bfgs_minimize<P, G>(
    phi: P,
    grad: G,
    x0: &[f64],
    cfg: &BfgsConfig,
) -> NumericResult<BfgsResult>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    run_bfgs(&phi, &|x| grad(x), x0, cfg)
}

/// Minimises `phi` with a central finite-difference gradient.
///
/// # Errors
///
/// Same conditions as [`bfgs_minimize`].
pub fn bfgs_minimize_numerical<P>(phi: P, x0: &[f64], cfg: &BfgsConfig) -> NumericResult<BfgsResult>
where
    P: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let eps = cfg.fd_eps;
    let phi_ref = &phi;
    let grad = |x: &[f64]| central_gradient(phi_ref, x, eps, n);
    run_bfgs(&phi, &grad, x0, cfg)
}

fn run_bfgs<P, G>(phi: &P, grad: &G, x0: &[f64], cfg: &BfgsConfig) -> NumericResult<BfgsResult>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    if !(cfg.grad_tol > 0.0 && cfg.grad_tol.is_finite()) {
        return Err(NumericError::InvalidParameter(format!(
            "grad_tol must be positive finite, got {}",
            cfg.grad_tol
        )));
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

    // Inverse-Hessian approximation H, row-major n×n, initialised to identity.
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        h[i * n + i] = 1.0;
    }

    for it in 0..cfg.max_iter {
        if gnorm <= cfg.grad_tol {
            return Ok(BfgsResult {
                x,
                fx,
                grad_norm: gnorm,
                iters: it,
            });
        }

        // Search direction p = −H g.
        let mut p = matvec(&h, &g, n);
        for pi in &mut p {
            *pi = -*pi;
        }
        // Guard: ensure a descent direction; if not, reset to steepest descent.
        let mut dderiv = dot(&g, &p);
        if dderiv >= 0.0 {
            p = g.iter().map(|gi| -gi).collect();
            // Reset H to identity to recover.
            h.fill(0.0);
            for i in 0..n {
                h[i * n + i] = 1.0;
            }
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
            // Line search failed to find decrease → treat as converged if the
            // gradient is already tiny, otherwise report non-convergence.
            return finish(x, fx, gnorm, it, cfg.grad_tol);
        }

        let g_new = grad_checked(grad, &x_new, n)?;

        // s = x_new − x ; y = g_new − g.
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);

        // BFGS inverse-Hessian update (skipped if curvature condition fails).
        if sy > 1.0e-12 {
            bfgs_update(&mut h, &s, &y, sy, n);
        }

        x = x_new;
        fx = f_new;
        g = g_new;
        gnorm = norm2(&g);
    }

    finish(x, fx, gnorm, cfg.max_iter, cfg.grad_tol)
}

fn finish(x: Vec<f64>, fx: f64, gnorm: f64, iters: usize, tol: f64) -> NumericResult<BfgsResult> {
    if gnorm <= tol {
        Ok(BfgsResult {
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

/// Applies `H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ` in place.
fn bfgs_update(h: &mut [f64], s: &[f64], y: &[f64], sy: f64, n: usize) {
    let rho = 1.0 / sy;
    // a = H y (column), b = yᵀ H (row, = (H y)ᵀ since H symmetric).
    let hy = matvec(h, y, n);
    let yhy = dot(y, &hy);

    // H_new[i][j] = H[i][j]
    //             − ρ (s_i (Hy)_j + (Hy)_i s_j)
    //             + ρ s_i s_j (1 + ρ yᵀHy)
    let coeff = rho * (1.0 + rho * yhy);
    for i in 0..n {
        let si = s[i];
        let hyi = hy[i];
        let base = i * n;
        for j in 0..n {
            let term = rho * (si * hy[j] + hyi * s[j]) - coeff * si * s[j];
            h[base + j] -= term;
        }
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

fn matvec(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for (i, oi) in out.iter_mut().enumerate() {
        *oi = dot(&a[i * n..i * n + n], v);
    }
    out
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

    fn cfg() -> BfgsConfig {
        BfgsConfig::default()
    }

    #[test]
    fn quadratic_bowl() {
        // φ = (x−3)² + (y+1)² → minimum at (3, −1).
        let phi = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 1.0).powi(2);
        let grad = |v: &[f64]| vec![2.0 * (v[0] - 3.0), 2.0 * (v[1] + 1.0)];
        let r = bfgs_minimize(phi, grad, &[0.0, 0.0], &cfg()).expect("ok");
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
        let c = BfgsConfig {
            max_iter: 2000,
            grad_tol: 1e-6,
            ..cfg()
        };
        let r = bfgs_minimize(phi, grad, &[-1.2, 1.0], &c).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-3, "x={}", r.x[0]);
        assert!((r.x[1] - 1.0).abs() < 1e-3, "y={}", r.x[1]);
    }

    #[test]
    fn converges_grad_norm() {
        let phi = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
        let r = bfgs_minimize(phi, grad, &[5.0, -4.0, 3.0], &cfg()).expect("ok");
        assert!(r.grad_norm <= cfg().grad_tol, "gnorm={}", r.grad_norm);
    }

    #[test]
    fn numerical_gradient_matches() {
        // The finite-difference variant reaches the same minimiser.
        let phi = |v: &[f64]| (v[0] - 2.0).powi(2) + 3.0 * (v[1] - 5.0).powi(2);
        let r = bfgs_minimize_numerical(phi, &[0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 2.0).abs() < 1e-4, "x={}", r.x[0]);
        assert!((r.x[1] - 5.0).abs() < 1e-4, "y={}", r.x[1]);
    }

    #[test]
    fn already_at_minimum() {
        let phi = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
        let grad = |v: &[f64]| vec![2.0 * v[0], 2.0 * v[1]];
        let r = bfgs_minimize(phi, grad, &[0.0, 0.0], &cfg()).expect("ok");
        assert_eq!(r.iters, 0);
        assert!(r.grad_norm < 1e-12);
    }

    #[test]
    fn max_iter_bound() {
        // One iteration on Rosenbrock from a far start cannot converge.
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
        let c = BfgsConfig {
            max_iter: 1,
            grad_tol: 1e-12,
            ..cfg()
        };
        let res = bfgs_minimize(phi, grad, &[-5.0, 5.0], &c);
        assert!(res.is_err());
    }

    #[test]
    fn output_len() {
        let phi = |v: &[f64]| v.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| 2.0 * (x - 1.0)).collect::<Vec<_>>();
        let r = bfgs_minimize(phi, grad, &[0.0; 4], &cfg()).expect("ok");
        assert_eq!(r.x.len(), 4);
    }

    #[test]
    fn decreases_objective() {
        let phi = |v: &[f64]| (v[0] + 4.0).powi(2) + (v[1] - 2.0).powi(4);
        let grad = |v: &[f64]| vec![2.0 * (v[0] + 4.0), 4.0 * (v[1] - 2.0).powi(3)];
        let start = [1.0, 1.0];
        let f0 = phi(&start);
        let r = bfgs_minimize(phi, grad, &start, &cfg()).expect("ok");
        assert!(r.fx < f0, "objective did not decrease: {} -> {}", f0, r.fx);
    }

    #[test]
    fn finite() {
        let phi = |v: &[f64]| v.iter().map(|x| x.cosh()).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| x.sinh()).collect::<Vec<_>>();
        let r = bfgs_minimize(phi, grad, &[0.5, -0.5], &cfg()).expect("ok");
        for v in &r.x {
            assert!(v.is_finite());
        }
        assert!(r.fx.is_finite() && r.grad_norm.is_finite());
    }

    #[test]
    fn rejects_bad_input() {
        let phi = |v: &[f64]| v[0] * v[0];
        let grad = |v: &[f64]| vec![2.0 * v[0]];
        assert!(bfgs_minimize(phi, grad, &[], &cfg()).is_err());
        let bad = BfgsConfig { c1: 2.0, ..cfg() };
        assert!(
            bfgs_minimize(
                |v: &[f64]| v[0] * v[0],
                |v: &[f64]| vec![2.0 * v[0]],
                &[1.0],
                &bad
            )
            .is_err()
        );
    }
}
