//! Nonlinear conjugate-gradient (CG) minimiser.
//!
//! Minimises a smooth scalar objective `φ : ℝⁿ → ℝ` using conjugate search
//! directions built from successive gradients.  Unlike (L-)BFGS it stores no
//! Hessian approximation at all — only the previous gradient and search
//! direction — giving `O(n)` memory, while still converging much faster than
//! plain steepest descent on ill-conditioned quadratics.
//!
//! Direction update:
//!
//! ```text
//! d₀ = −∇φ₀
//! dₖ = −∇φₖ + βₖ dₖ₋₁
//! ```
//!
//! with the conjugacy parameter `βₖ` chosen by one of:
//!
//! * **Fletcher-Reeves**:  `βₖ = (gₖᵀgₖ) / (gₖ₋₁ᵀgₖ₋₁)`
//! * **Polak-Ribière (PR+)**: `βₖ = max(0, gₖᵀ(gₖ − gₖ₋₁) / (gₖ₋₁ᵀgₖ₋₁))`
//! * **Hestenes-Stiefel**: `βₖ = gₖᵀ(gₖ − gₖ₋₁) / (dₖ₋₁ᵀ(gₖ − gₖ₋₁))`
//!
//! Automatic restarts to steepest descent occur every `n` iterations and
//! whenever the generated direction loses descent (`dₖᵀgₖ ≥ 0`), guaranteeing
//! global convergence under the strong-Wolfe step lengths used here.

use crate::error::{NumericError, NumericResult};

/// Choice of conjugacy update formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgVariant {
    /// Fletcher-Reeves `β = ‖gₖ‖² / ‖gₖ₋₁‖²`.
    FletcherReeves,
    /// Polak-Ribière with non-negative clipping (PR+).
    PolakRibiere,
    /// Hestenes-Stiefel.
    HestenesStiefel,
}

/// Configuration for [`conjugate_gradient_minimize`].
#[derive(Debug, Clone, Copy)]
pub struct CgConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖∇φ‖₂`.
    pub grad_tol: f64,
    /// Conjugacy update variant.
    pub variant: CgVariant,
    /// Armijo sufficient-decrease parameter (`0 < c1 < c2 < 1`, typically `1e-4`).
    pub c1: f64,
    /// Strong-Wolfe curvature parameter (`c1 < c2 < 1`, typically `0.1`).
    pub c2: f64,
    /// Step used for the central-difference gradient (numerical variant only).
    pub fd_eps: f64,
}

impl Default for CgConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            grad_tol: 1.0e-8,
            variant: CgVariant::PolakRibiere,
            c1: 1.0e-4,
            c2: 0.1,
            fd_eps: 1.0e-6,
        }
    }
}

/// Accepted line-search point: `(x_new, f_new, g_new)`.
type LineStep = (Vec<f64>, f64, Vec<f64>);

/// Result of a nonlinear CG run.
#[derive(Debug, Clone)]
pub struct CgResult {
    /// Minimiser estimate.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fx: f64,
    /// `‖∇φ(x)‖₂` at termination.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iters: usize,
}

/// Minimises `phi` with an analytic gradient `grad` via nonlinear CG.
///
/// # Errors
///
/// * [`NumericError::EmptyInput`] if `x0` is empty.
/// * [`NumericError::InvalidParameter`] for invalid config parameters,
///   non-finite `x0`, or a gradient of the wrong length.
/// * [`NumericError::NotConverged`] if `‖∇φ‖` stays above `grad_tol` after
///   `max_iter` iterations.
pub fn conjugate_gradient_minimize<P, G>(
    phi: P,
    grad: G,
    x0: &[f64],
    cfg: &CgConfig,
) -> NumericResult<CgResult>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    run_cg(&phi, &|x| grad(x), x0, cfg)
}

/// Minimises `phi` with a central finite-difference gradient via nonlinear CG.
///
/// # Errors
///
/// Same conditions as [`conjugate_gradient_minimize`].
pub fn conjugate_gradient_minimize_numerical<P>(
    phi: P,
    x0: &[f64],
    cfg: &CgConfig,
) -> NumericResult<CgResult>
where
    P: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let eps = cfg.fd_eps;
    let phi_ref = &phi;
    let grad = |x: &[f64]| central_gradient(phi_ref, x, eps, n);
    run_cg(&phi, &grad, x0, cfg)
}

fn run_cg<P, G>(phi: &P, grad: &G, x0: &[f64], cfg: &CgConfig) -> NumericResult<CgResult>
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

    // Initial direction = steepest descent.
    let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();
    let mut since_restart = 0_usize;

    for it in 0..cfg.max_iter {
        if gnorm <= cfg.grad_tol {
            return Ok(CgResult {
                x,
                fx,
                grad_norm: gnorm,
                iters: it,
            });
        }

        let mut dderiv = dot(&g, &d);
        // Ensure a descent direction; restart to steepest descent if not.
        if dderiv >= 0.0 {
            d = g.iter().map(|gi| -gi).collect();
            since_restart = 0;
            dderiv = dot(&g, &d);
        }

        // Strong-Wolfe line search along d.
        let ls = strong_wolfe_line_search(phi, grad, &x, fx, &g, &d, dderiv, cfg, n)?;
        let (x_new, f_new, g_new) = match ls {
            Some(triple) => triple,
            None => return finish(x, fx, gnorm, it, cfg.grad_tol),
        };

        let gnew_norm = norm2(&g_new);

        // Conjugacy parameter β.
        let gg_old = dot(&g, &g);
        let beta = if since_restart + 1 >= n || gg_old <= 0.0 {
            // Periodic restart: behave as steepest descent next step.
            0.0
        } else {
            compute_beta(cfg.variant, &g, &g_new, &d, gg_old)
        };

        // New direction dₖ = −gₖ + β dₖ₋₁.
        let mut d_new = vec![0.0_f64; n];
        for k in 0..n {
            d_new[k] = -g_new[k] + beta * d[k];
        }

        x = x_new;
        fx = f_new;
        g = g_new;
        gnorm = gnew_norm;
        d = d_new;
        if beta == 0.0 {
            since_restart = 0;
        } else {
            since_restart += 1;
        }
    }

    finish(x, fx, gnorm, cfg.max_iter, cfg.grad_tol)
}

fn compute_beta(variant: CgVariant, g: &[f64], g_new: &[f64], d: &[f64], gg_old: f64) -> f64 {
    match variant {
        CgVariant::FletcherReeves => dot(g_new, g_new) / gg_old,
        CgVariant::PolakRibiere => {
            let mut num = 0.0_f64;
            for (gn, go) in g_new.iter().zip(g) {
                num += gn * (gn - go);
            }
            (num / gg_old).max(0.0)
        }
        CgVariant::HestenesStiefel => {
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for k in 0..g.len() {
                let y = g_new[k] - g[k];
                num += g_new[k] * y;
                den += d[k] * y;
            }
            if den.abs() < 1.0e-14 {
                0.0
            } else {
                (num / den).max(0.0)
            }
        }
    }
}

/// Strong-Wolfe line search (bracketing + zoom), returning the accepted point
/// `(x_new, f_new, g_new)` or `None` if no acceptable step could be found.
#[allow(clippy::too_many_arguments)]
fn strong_wolfe_line_search<P, G>(
    phi: &P,
    grad: &G,
    x: &[f64],
    f0: f64,
    g0: &[f64],
    d: &[f64],
    dderiv0: f64,
    cfg: &CgConfig,
    n: usize,
) -> NumericResult<Option<LineStep>>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let eval = |alpha: f64| -> (Vec<f64>, f64) {
        let xt: Vec<f64> = x.iter().zip(d).map(|(xi, di)| xi + alpha * di).collect();
        let ft = phi(&xt);
        (xt, ft)
    };

    let mut alpha_prev = 0.0_f64;
    let mut f_prev = f0;
    let mut alpha = 1.0_f64;
    let alpha_max = 1.0e10_f64;

    for i in 0..50 {
        let (xt, ft) = eval(alpha);
        if !ft.is_finite() {
            // Step too large; shrink by bisecting toward the previous bracket.
            alpha = 0.5 * (alpha_prev + alpha);
            continue;
        }
        // Armijo (sufficient decrease) failure or non-monotone increase → zoom.
        if ft > f0 + cfg.c1 * alpha * dderiv0 || (i > 0 && ft >= f_prev) {
            return zoom(
                phi, grad, x, f0, g0, d, dderiv0, alpha_prev, f_prev, alpha, cfg, n,
            );
        }
        let gt = grad_checked(grad, &xt, n)?;
        let slope = dot(&gt, d);
        if slope.abs() <= -cfg.c2 * dderiv0 {
            // Both strong-Wolfe conditions met.
            return Ok(Some((xt, ft, gt)));
        }
        if slope >= 0.0 {
            // Overshot the minimum; zoom with reversed bracket.
            return zoom(
                phi, grad, x, f0, g0, d, dderiv0, alpha, ft, alpha_prev, cfg, n,
            );
        }
        alpha_prev = alpha;
        f_prev = ft;
        alpha = (2.0 * alpha).min(alpha_max);
        if alpha >= alpha_max {
            break;
        }
    }
    Ok(None)
}

/// Strong-Wolfe "zoom" refining the interval `[a_lo, a_hi]`.
#[allow(clippy::too_many_arguments)]
fn zoom<P, G>(
    phi: &P,
    grad: &G,
    x: &[f64],
    f0: f64,
    _g0: &[f64],
    d: &[f64],
    dderiv0: f64,
    mut a_lo: f64,
    mut f_lo: f64,
    mut a_hi: f64,
    cfg: &CgConfig,
    n: usize,
) -> NumericResult<Option<LineStep>>
where
    P: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    for _ in 0..50 {
        let alpha = 0.5 * (a_lo + a_hi);
        let xt: Vec<f64> = x.iter().zip(d).map(|(xi, di)| xi + alpha * di).collect();
        let ft = phi(&xt);
        if !ft.is_finite() {
            a_hi = alpha;
            continue;
        }
        if ft > f0 + cfg.c1 * alpha * dderiv0 || ft >= f_lo {
            a_hi = alpha;
        } else {
            let gt = grad_checked(grad, &xt, n)?;
            let slope = dot(&gt, d);
            if slope.abs() <= -cfg.c2 * dderiv0 {
                return Ok(Some((xt, ft, gt)));
            }
            if slope * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
            }
            a_lo = alpha;
            f_lo = ft;
        }
        if (a_hi - a_lo).abs() < 1.0e-16 {
            break;
        }
    }
    Ok(None)
}

fn validate_config(cfg: &CgConfig) -> NumericResult<()> {
    if !(cfg.grad_tol > 0.0 && cfg.grad_tol.is_finite()) {
        return Err(NumericError::InvalidParameter(format!(
            "grad_tol must be positive finite, got {}",
            cfg.grad_tol
        )));
    }
    if !(cfg.c1 > 0.0 && cfg.c1 < cfg.c2 && cfg.c2 < 1.0) {
        return Err(NumericError::InvalidParameter(format!(
            "require 0 < c1 < c2 < 1, got c1={}, c2={}",
            cfg.c1, cfg.c2
        )));
    }
    Ok(())
}

fn finish(x: Vec<f64>, fx: f64, gnorm: f64, iters: usize, tol: f64) -> NumericResult<CgResult> {
    if gnorm <= tol {
        Ok(CgResult {
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

    fn cfg() -> CgConfig {
        CgConfig::default()
    }

    #[test]
    fn quadratic_bowl_pr() {
        let phi = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 1.0).powi(2);
        let grad = |v: &[f64]| vec![2.0 * (v[0] - 3.0), 2.0 * (v[1] + 1.0)];
        let r = conjugate_gradient_minimize(phi, grad, &[0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 3.0).abs() < 1e-6, "x={}", r.x[0]);
        assert!((r.x[1] + 1.0).abs() < 1e-6, "y={}", r.x[1]);
        assert!(r.fx < 1e-10);
    }

    #[test]
    fn quadratic_bowl_fr() {
        let phi = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 1.0).powi(2);
        let grad = |v: &[f64]| vec![2.0 * (v[0] - 3.0), 2.0 * (v[1] + 1.0)];
        let c = CgConfig {
            variant: CgVariant::FletcherReeves,
            ..cfg()
        };
        let r = conjugate_gradient_minimize(phi, grad, &[0.0, 0.0], &c).expect("ok");
        assert!((r.x[0] - 3.0).abs() < 1e-6);
        assert!((r.x[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn quadratic_bowl_hs() {
        let phi = |v: &[f64]| (v[0] - 3.0).powi(2) + (v[1] + 1.0).powi(2);
        let grad = |v: &[f64]| vec![2.0 * (v[0] - 3.0), 2.0 * (v[1] + 1.0)];
        let c = CgConfig {
            variant: CgVariant::HestenesStiefel,
            ..cfg()
        };
        let r = conjugate_gradient_minimize(phi, grad, &[0.0, 0.0], &c).expect("ok");
        assert!((r.x[0] - 3.0).abs() < 1e-6);
        assert!((r.x[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cg_finishes_quadratic_in_few_iters() {
        // On a 5-D SPD quadratic, CG should converge in roughly n iterations.
        let a = [3.0_f64, 1.0, 4.0, 1.0, 5.0];
        let phi = move |v: &[f64]| v.iter().zip(&a).map(|(x, c)| c * x * x).sum::<f64>();
        let grad = move |v: &[f64]| {
            v.iter()
                .zip(&a)
                .map(|(x, c)| 2.0 * c * x)
                .collect::<Vec<_>>()
        };
        let r = conjugate_gradient_minimize(phi, grad, &[1.0; 5], &cfg()).expect("ok");
        assert!(r.grad_norm <= cfg().grad_tol);
        assert!(r.iters <= 60, "iters={}", r.iters);
    }

    #[test]
    fn rosenbrock() {
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
        let c = CgConfig {
            max_iter: 10000,
            grad_tol: 1e-6,
            ..cfg()
        };
        let r = conjugate_gradient_minimize(phi, grad, &[-1.2, 1.0], &c).expect("ok");
        assert!((r.x[0] - 1.0).abs() < 1e-3, "x={}", r.x[0]);
        assert!((r.x[1] - 1.0).abs() < 1e-3, "y={}", r.x[1]);
    }

    #[test]
    fn numerical_gradient_matches() {
        let phi = |v: &[f64]| (v[0] - 2.0).powi(2) + 3.0 * (v[1] - 5.0).powi(2);
        let r = conjugate_gradient_minimize_numerical(phi, &[0.0, 0.0], &cfg()).expect("ok");
        assert!((r.x[0] - 2.0).abs() < 1e-4, "x={}", r.x[0]);
        assert!((r.x[1] - 5.0).abs() < 1e-4, "y={}", r.x[1]);
    }

    #[test]
    fn already_at_minimum() {
        let phi = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
        let grad = |v: &[f64]| vec![2.0 * v[0], 2.0 * v[1]];
        let r = conjugate_gradient_minimize(phi, grad, &[0.0, 0.0], &cfg()).expect("ok");
        assert_eq!(r.iters, 0);
        assert!(r.grad_norm < 1e-12);
    }

    #[test]
    fn decreases_objective() {
        let phi = |v: &[f64]| (v[0] + 4.0).powi(2) + (v[1] - 2.0).powi(4);
        let grad = |v: &[f64]| vec![2.0 * (v[0] + 4.0), 4.0 * (v[1] - 2.0).powi(3)];
        let start = [1.0, 1.0];
        let f0 = phi(&start);
        let r = conjugate_gradient_minimize(phi, grad, &start, &cfg()).expect("ok");
        assert!(r.fx < f0);
    }

    #[test]
    fn high_dim_well_conditioned() {
        let n = 30_usize;
        let phi = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, &x)| (x - i as f64).powi(2))
                .sum::<f64>()
        };
        let grad = |v: &[f64]| {
            v.iter()
                .enumerate()
                .map(|(i, &x)| 2.0 * (x - i as f64))
                .collect::<Vec<_>>()
        };
        let r = conjugate_gradient_minimize(phi, grad, &vec![0.0; n], &cfg()).expect("ok");
        for (i, &xi) in r.x.iter().enumerate() {
            assert!((xi - i as f64).abs() < 1e-4, "x[{i}]={xi}");
        }
    }

    #[test]
    fn output_finite() {
        let phi = |v: &[f64]| v.iter().map(|x| x.cosh()).sum::<f64>();
        let grad = |v: &[f64]| v.iter().map(|x| x.sinh()).collect::<Vec<_>>();
        let r = conjugate_gradient_minimize(phi, grad, &[0.5, -0.5], &cfg()).expect("ok");
        for v in &r.x {
            assert!(v.is_finite());
        }
        assert!(r.fx.is_finite() && r.grad_norm.is_finite());
    }

    #[test]
    fn rejects_bad_input() {
        let phi = |v: &[f64]| v[0] * v[0];
        let grad = |v: &[f64]| vec![2.0 * v[0]];
        assert!(conjugate_gradient_minimize(phi, grad, &[], &cfg()).is_err());
        let bad = CgConfig {
            c1: 0.5,
            c2: 0.1,
            ..cfg()
        };
        assert!(
            conjugate_gradient_minimize(
                |v: &[f64]| v[0] * v[0],
                |v: &[f64]| vec![2.0 * v[0]],
                &[1.0],
                &bad
            )
            .is_err()
        );
    }
}
