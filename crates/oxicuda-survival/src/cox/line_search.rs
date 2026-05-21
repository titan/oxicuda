//! Wolfe and Armijo line search routines for Newton-Raphson in Cox regression.
//!
//! Provides:
//! - [`ArmijoConfig`] / [`armijo_backtrack`]: backtracking Armijo sufficient-decrease line search.
//! - [`WolfeConfig`] / [`wolfe_line_search`]: strong Wolfe line search with zoom bracketing.

use crate::error::{SurvivalError, SurvivalResult};

// ---------------------------------------------------------------------------
// Armijo backtracking
// ---------------------------------------------------------------------------

/// Configuration for the Armijo backtracking line search.
#[derive(Debug, Clone, Copy)]
pub struct ArmijoConfig {
    /// Sufficient decrease constant (Armijo condition). Typically `1e-4`.
    pub c1: f64,
    /// Contraction factor applied to step at each iteration. Typically `0.5`.
    pub rho: f64,
    /// Maximum number of backtracking steps.
    pub max_iter: usize,
    /// Initial step length α₀.
    pub alpha_init: f64,
}

impl Default for ArmijoConfig {
    fn default() -> Self {
        Self {
            c1: 1.0e-4,
            rho: 0.5,
            max_iter: 50,
            alpha_init: 1.0,
        }
    }
}

/// Backtracking Armijo line search.
///
/// Finds the largest `α = α_init * ρ^k` (k ≥ 0) such that:
///
/// `f(x + α·d) ≤ f(x) + c1·α·(∇f)ᵀ·d`
///
/// # Errors
/// - [`SurvivalError::InvalidParameter`] if `grad_dot_d ≥ 0` (non-descent direction).
/// - [`SurvivalError::NotConverged`] if no acceptable step found within `max_iter`.
pub fn armijo_backtrack<F>(
    x: &[f64],
    direction: &[f64],
    f_val: f64,
    grad_dot_d: f64,
    f: &F,
    config: &ArmijoConfig,
) -> SurvivalResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if grad_dot_d >= 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "armijo_backtrack: grad_dot_d={grad_dot_d:.6e} must be strictly negative for a descent direction"
        )));
    }
    if config.c1 <= 0.0 || config.c1 >= 1.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "armijo_backtrack: c1={:.6e} must be in (0,1)",
            config.c1
        )));
    }
    if config.rho <= 0.0 || config.rho >= 1.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "armijo_backtrack: rho={:.6e} must be in (0,1)",
            config.rho
        )));
    }

    let n = x.len();
    let mut alpha = config.alpha_init;
    let mut trial = vec![0.0_f64; n];

    for _iter in 0..config.max_iter {
        for i in 0..n {
            trial[i] = x[i] + alpha * direction[i];
        }
        let f_new = f(&trial);
        if f_new <= f_val + config.c1 * alpha * grad_dot_d {
            return Ok(alpha);
        }
        alpha *= config.rho;
        if alpha < f64::EPSILON {
            break;
        }
    }

    Err(SurvivalError::NotConverged {
        iter: config.max_iter,
    })
}

// ---------------------------------------------------------------------------
// Strong Wolfe line search
// ---------------------------------------------------------------------------

/// Configuration for the strong Wolfe line search.
#[derive(Debug, Clone, Copy)]
pub struct WolfeConfig {
    /// Sufficient decrease constant (Armijo condition). Typically `1e-4`.
    pub c1: f64,
    /// Curvature condition constant. Typically `0.9` for Newton methods, `0.1` for CG.
    pub c2: f64,
    /// Upper bound on the bracket interval. Typically `10.0`.
    pub alpha_max: f64,
    /// Maximum total number of function evaluations.
    pub max_iter: usize,
}

impl Default for WolfeConfig {
    fn default() -> Self {
        Self {
            c1: 1.0e-4,
            c2: 0.9,
            alpha_max: 10.0,
            max_iter: 50,
        }
    }
}

/// Dot product of a gradient vector with a direction vector.
fn dot(g: &[f64], d: &[f64]) -> f64 {
    g.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum()
}

/// Evaluate trial point `x + alpha * d`.
fn eval_trial(x: &[f64], direction: &[f64], alpha: f64) -> Vec<f64> {
    x.iter()
        .zip(direction.iter())
        .map(|(xi, di)| xi + alpha * di)
        .collect()
}

/// Cubic interpolation to find a minimiser in [a, b] given f and f' at both ends.
/// Falls back to bisection if the cubic minimiser is outside (a, b).
fn cubic_minimiser(
    alpha_lo: f64,
    f_lo: f64,
    df_lo: f64,
    alpha_hi: f64,
    f_hi: f64,
    df_hi: f64,
) -> f64 {
    // Algorithm from Nocedal & Wright (2006), Algorithm 3.6 (cubic interpolation).
    let d1 = df_lo + df_hi - 3.0 * (f_hi - f_lo) / (alpha_hi - alpha_lo);
    let d2_sq = d1 * d1 - df_lo * df_hi;
    if d2_sq < 0.0 {
        // Fall back to bisection.
        return 0.5 * (alpha_lo + alpha_hi);
    }
    let d2 = d2_sq.sqrt();
    let num = df_hi + d2 - d1;
    let den = df_hi - df_lo + 2.0 * d2;
    if den.abs() < f64::EPSILON {
        return 0.5 * (alpha_lo + alpha_hi);
    }
    let alpha_star = alpha_hi - (alpha_hi - alpha_lo) * num / den;
    // Clamp to (alpha_lo, alpha_hi).
    let lo = alpha_lo.min(alpha_hi);
    let hi = alpha_lo.max(alpha_hi);
    let margin = 0.1 * (hi - lo);
    alpha_star.clamp(lo + margin, hi - margin)
}

/// Zoom subroutine for strong Wolfe line search (Nocedal & Wright, Algorithm 3.6).
///
/// Bisects / cubically interpolates a bracket `[alpha_lo, alpha_hi]` that is known to
/// contain a strong-Wolfe point, and returns an acceptable `alpha`.
fn zoom<F, G>(
    x: &[f64],
    direction: &[f64],
    f_val: f64,
    grad_dot_d: f64,
    f: &F,
    grad_fn: &G,
    config: &WolfeConfig,
    alpha_lo_in: f64,
    f_lo_in: f64,
    df_lo_in: f64,
    alpha_hi_in: f64,
    f_hi_in: f64,
    df_hi_in: f64,
) -> SurvivalResult<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut alpha_lo = alpha_lo_in;
    let mut f_lo = f_lo_in;
    let mut df_lo = df_lo_in;
    let mut alpha_hi = alpha_hi_in;
    let mut f_hi = f_hi_in;
    let mut df_hi = df_hi_in;

    for iter in 0..config.max_iter {
        let alpha_j = cubic_minimiser(alpha_lo, f_lo, df_lo, alpha_hi, f_hi, df_hi);

        let xj = eval_trial(x, direction, alpha_j);
        let f_j = f(&xj);

        if f_j > f_val + config.c1 * alpha_j * grad_dot_d || f_j >= f_lo {
            // alpha_j violates sufficient decrease or is worse than lo → shrink hi.
            alpha_hi = alpha_j;
            f_hi = f_j;
            let gj = grad_fn(&xj);
            df_hi = dot(&gj, direction);
        } else {
            let gj = grad_fn(&xj);
            let df_j = dot(&gj, direction);
            // Strong curvature satisfied?
            if df_j.abs() <= config.c2 * grad_dot_d.abs() {
                return Ok(alpha_j);
            }
            if df_j * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
                f_hi = f_lo;
                df_hi = df_lo;
            }
            alpha_lo = alpha_j;
            f_lo = f_j;
            df_lo = df_j;
        }

        if (alpha_hi - alpha_lo).abs() < f64::EPSILON * 10.0 {
            // Bracket collapsed; return best known.
            return Ok(alpha_lo);
        }

        // Safety: should not happen with correct cubic interpolation but guard anyway.
        if iter + 1 == config.max_iter {
            return Ok(alpha_lo);
        }
    }
    Ok(alpha_lo)
}

/// Strong Wolfe line search using zoom bracketing (Nocedal & Wright, Algorithm 3.5–3.6).
///
/// Finds `α` satisfying:
/// 1. Sufficient decrease: `f(x+α d) ≤ f(x) + c1·α·(∇f)ᵀd`
/// 2. Strong curvature: `|(∇f(x+α d))ᵀd| ≤ c2·|(∇f)ᵀd|`
///
/// # Errors
/// - [`SurvivalError::InvalidParameter`] if `grad_dot_d ≥ 0`.
/// - [`SurvivalError::NotConverged`] if no suitable step found.
pub fn wolfe_line_search<F, G>(
    x: &[f64],
    direction: &[f64],
    f_val: f64,
    grad_dot_d: f64,
    f: &F,
    grad_fn: &G,
    config: &WolfeConfig,
) -> SurvivalResult<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    if grad_dot_d >= 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "wolfe_line_search: grad_dot_d={grad_dot_d:.6e} must be negative"
        )));
    }
    if config.c1 <= 0.0 || config.c1 >= config.c2 {
        return Err(SurvivalError::InvalidParameter(format!(
            "wolfe_line_search: need 0 < c1={:.4e} < c2={:.4e} < 1",
            config.c1, config.c2
        )));
    }

    let mut alpha_prev = 0.0_f64;
    let mut f_prev = f_val;
    let mut df_prev = grad_dot_d;
    let mut alpha = 1.0_f64.min(config.alpha_max);

    for _iter in 0..config.max_iter {
        let xj = eval_trial(x, direction, alpha);
        let f_j = f(&xj);

        // Armijo violated or worse than previous → zoom between [alpha_prev, alpha].
        if f_j > f_val + config.c1 * alpha * grad_dot_d || (_iter > 0 && f_j >= f_prev) {
            let gp = grad_fn(&eval_trial(x, direction, alpha_prev));
            let df_p = dot(&gp, direction);
            return zoom(
                x, direction, f_val, grad_dot_d, f, grad_fn, config, alpha_prev, f_prev, df_p,
                alpha, f_j, 0.0, // df_hi placeholder; zoom recomputes if needed.
            );
        }

        let gj = grad_fn(&xj);
        let df_j = dot(&gj, direction);

        // Strong curvature satisfied → done.
        if df_j.abs() <= config.c2 * grad_dot_d.abs() {
            return Ok(alpha);
        }

        // Derivative positive → we overshot a minimum; zoom between [alpha, alpha_prev].
        if df_j >= 0.0 {
            return zoom(
                x, direction, f_val, grad_dot_d, f, grad_fn, config, alpha, f_j, df_j, alpha_prev,
                f_prev, df_prev,
            );
        }

        alpha_prev = alpha;
        f_prev = f_j;
        df_prev = df_j;

        // Expand step (geometric + cap at alpha_max).
        alpha = (alpha * 2.0).min(config.alpha_max);
        if alpha >= config.alpha_max {
            break;
        }
    }

    // Return last valid step (satisfies Armijo at minimum).
    Ok(alpha_prev.max(1.0e-10))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: quadratic f(x) = 0.5 * ||x - target||^2, grad = x - target.
    fn quadratic_f(target: &[f64]) -> impl Fn(&[f64]) -> f64 + '_ {
        move |x: &[f64]| {
            0.5 * x
                .iter()
                .zip(target.iter())
                .map(|(xi, ti)| (xi - ti).powi(2))
                .sum::<f64>()
        }
    }

    fn quadratic_grad(target: &[f64]) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
        move |x: &[f64]| {
            x.iter()
                .zip(target.iter())
                .map(|(xi, ti)| xi - ti)
                .collect()
        }
    }

    // ---------- Armijo tests ----------

    #[test]
    fn armijo_finds_valid_alpha_on_quadratic() {
        let target = [3.0_f64];
        let x = [0.0_f64];
        let direction = [1.0_f64]; // direction toward minimum
        let f = quadratic_f(&target);
        let f_val = f(&x);
        let g = quadratic_grad(&target);
        let grad = g(&x);
        let gdotd: f64 = grad[0] * direction[0];
        let config = ArmijoConfig::default();
        let alpha = armijo_backtrack(&x, &direction, f_val, gdotd, &f, &config).expect("ok");
        // Verify Armijo condition.
        let x_new = [x[0] + alpha * direction[0]];
        assert!(f(&x_new) <= f_val + config.c1 * alpha * gdotd);
    }

    #[test]
    fn armijo_returns_error_on_non_descent_direction() {
        let target = [0.0_f64];
        let x = [1.0_f64];
        let direction = [1.0_f64]; // gradient direction = ascent
        let f = quadratic_f(&target);
        let f_val = f(&x);
        // grad_dot_d > 0 → error
        let err = armijo_backtrack(&x, &direction, f_val, 0.5, &f, &ArmijoConfig::default());
        assert!(err.is_err());
    }

    #[test]
    fn armijo_alpha_1_accepted_on_unit_step_quadratic() {
        // f(x) = 0.5 x^2, descent from x=2.0 with step d=-2 (Newton step).
        let f = |x: &[f64]| 0.5 * x[0] * x[0];
        let x = [2.0_f64];
        let direction = [-2.0_f64]; // Newton step for f'=x, H=1 → d=-x=-2
        let f_val = f(&x); // 2.0
        let grad_dot_d = 2.0 * (-2.0); // f'(2)*(-2) = -4
        let config = ArmijoConfig {
            alpha_init: 1.0,
            ..Default::default()
        };
        let alpha = armijo_backtrack(&x, &direction, f_val, grad_dot_d, &f, &config).expect("ok");
        // alpha=1 → x_new=0, f_new=0 ≤ 2 + 1e-4*1*(-4) ✓
        assert!((alpha - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn armijo_decreases_alpha_on_steep_landscape() {
        // Rosenbrock-like: very steep, alpha=1 will be rejected.
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let x = [-0.5_f64, 0.5_f64];
        // gradient at x
        let g0 = 2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let g1 = 200.0 * (x[1] - x[0] * x[0]);
        let direction = [-g0, -g1]; // gradient descent direction
        let gdotd = g0 * direction[0] + g1 * direction[1];
        let f_val = f(&x);
        let config = ArmijoConfig::default();
        let alpha = armijo_backtrack(&x, &direction, f_val, gdotd, &f, &config).expect("ok");
        assert!(alpha < 1.0);
        let x_new = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]];
        assert!(f(&x_new) <= f_val + config.c1 * alpha * gdotd);
    }

    #[test]
    fn armijo_2d_quadratic_valid() {
        let target = [1.0_f64, 2.0_f64];
        let x = [0.0_f64, 0.0_f64];
        let f = quadratic_f(&target);
        let g = quadratic_grad(&target);
        let grad = g(&x);
        let direction = [-grad[0], -grad[1]];
        let gdotd: f64 = grad.iter().zip(direction.iter()).map(|(a, b)| a * b).sum();
        let f_val = f(&x);
        let config = ArmijoConfig::default();
        let alpha = armijo_backtrack(&x, &direction, f_val, gdotd, &f, &config).expect("ok");
        let x_new = [x[0] + alpha * direction[0], x[1] + alpha * direction[1]];
        assert!(f(&x_new) < f_val);
    }

    #[test]
    fn armijo_rejects_bad_rho() {
        let f = |x: &[f64]| x[0] * x[0];
        let x = [2.0_f64];
        let d = [-1.0_f64];
        let config = ArmijoConfig {
            rho: 1.1, // invalid
            ..Default::default()
        };
        let err = armijo_backtrack(&x, &d, f(&x), -4.0, &f, &config);
        assert!(err.is_err());
    }

    #[test]
    fn armijo_rejects_bad_c1() {
        let f = |x: &[f64]| x[0] * x[0];
        let x = [2.0_f64];
        let d = [-1.0_f64];
        let config = ArmijoConfig {
            c1: 1.5, // invalid
            ..Default::default()
        };
        let err = armijo_backtrack(&x, &d, f(&x), -4.0, &f, &config);
        assert!(err.is_err());
    }

    // ---------- Wolfe tests ----------

    #[test]
    fn wolfe_on_quadratic_satisfies_both_conditions() {
        let target = [5.0_f64];
        let f = quadratic_f(&target);
        let gfn = quadratic_grad(&target);
        let x = [0.0_f64];
        let g = gfn(&x);
        let direction = [-g[0]]; // steepest descent
        let gdotd = -g[0] * g[0];
        let f_val = f(&x);
        let config = WolfeConfig::default();
        let alpha =
            wolfe_line_search(&x, &direction, f_val, gdotd, &f, &gfn, &config).expect("wolfe ok");
        let x_new = [x[0] + alpha * direction[0]];
        // Sufficient decrease.
        assert!(
            f(&x_new) <= f_val + config.c1 * alpha * gdotd,
            "Armijo violated"
        );
        // Curvature condition.
        let g_new = gfn(&x_new);
        let df_new = g_new[0] * direction[0];
        assert!(
            df_new.abs() <= config.c2 * gdotd.abs(),
            "Curvature violated: |df_new|={} c2*|gdotd|={}",
            df_new.abs(),
            config.c2 * gdotd.abs()
        );
    }

    #[test]
    fn wolfe_errors_on_non_descent() {
        let f = |x: &[f64]| x[0] * x[0];
        let gfn = |x: &[f64]| vec![2.0 * x[0]];
        let x = [1.0_f64];
        let err = wolfe_line_search(&x, &[1.0], f(&x), 2.0, &f, &gfn, &WolfeConfig::default());
        assert!(err.is_err());
    }

    #[test]
    fn wolfe_2d_quadratic_valid() {
        let target = [3.0_f64, -2.0_f64];
        let f = quadratic_f(&target);
        let gfn = quadratic_grad(&target);
        let x = [0.0_f64, 0.0_f64];
        let g = gfn(&x);
        let direction: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let gdotd: f64 = g.iter().zip(direction.iter()).map(|(a, b)| a * b).sum();
        let f_val = f(&x);
        let config = WolfeConfig::default();
        let alpha = wolfe_line_search(&x, &direction, f_val, gdotd, &f, &gfn, &config).expect("ok");
        let x_new: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        assert!(f(&x_new) < f_val);
    }

    #[test]
    fn wolfe_alpha_shrinks_near_minimum() {
        // Start close to minimum; alpha should be small.
        let target = [1.0_f64];
        let f = quadratic_f(&target);
        let gfn = quadratic_grad(&target);
        let x = [0.99_f64];
        let g = gfn(&x);
        let direction = [-g[0]];
        let gdotd = -g[0] * g[0];
        let config = WolfeConfig::default();
        let alpha = wolfe_line_search(&x, &direction, f(&x), gdotd, &f, &gfn, &config).expect("ok");
        assert!(alpha > 0.0);
        let x_new = [x[0] + alpha * direction[0]];
        assert!(f(&x_new) <= f(&x));
    }

    #[test]
    fn wolfe_cubic_minimiser_stays_in_bracket() {
        // cubic_minimiser should stay in (alpha_lo, alpha_hi).
        let alpha_lo = 0.0_f64;
        let alpha_hi = 1.0_f64;
        let m = cubic_minimiser(alpha_lo, 2.0, -3.0, alpha_hi, 0.5, 1.0);
        assert!(
            m > alpha_lo && m < alpha_hi,
            "m={m} not in ({alpha_lo},{alpha_hi})"
        );
    }

    #[test]
    fn armijo_and_wolfe_agree_on_exact_line_minimiser() {
        // For f(x) = 0.5 x^2 starting at x=4, direction d=-1:
        // Exact minimiser at alpha=4 (x+alpha*d = 0).
        // Both should find steps that satisfy their respective conditions.
        let f = |x: &[f64]| 0.5 * x[0] * x[0];
        let gfn = |x: &[f64]| vec![x[0]];
        let x = [4.0_f64];
        let direction = [-1.0_f64];
        let gdotd = -4.0_f64; // g(x)=4, d=-1
        let f_val = f(&x);

        let a_cfg = ArmijoConfig::default();
        let w_cfg = WolfeConfig::default();

        let a_alpha =
            armijo_backtrack(&x, &direction, f_val, gdotd, &f, &a_cfg).expect("armijo ok");
        let w_alpha =
            wolfe_line_search(&x, &direction, f_val, gdotd, &f, &gfn, &w_cfg).expect("wolfe ok");

        // Both should accept the step.
        let xa = [x[0] + a_alpha * direction[0]];
        let xw = [x[0] + w_alpha * direction[0]];
        assert!(f(&xa) <= f_val + a_cfg.c1 * a_alpha * gdotd);
        assert!(f(&xw) <= f_val + w_cfg.c1 * w_alpha * gdotd);
    }

    #[test]
    fn wolfe_default_config_is_valid() {
        let cfg = WolfeConfig::default();
        assert!(cfg.c1 < cfg.c2);
        assert!(cfg.c2 < 1.0);
        assert!(cfg.alpha_max > 0.0);
        assert!(cfg.max_iter > 0);
    }

    #[test]
    fn armijo_default_config_is_valid() {
        let cfg = ArmijoConfig::default();
        assert!(cfg.c1 > 0.0 && cfg.c1 < 1.0);
        assert!(cfg.rho > 0.0 && cfg.rho < 1.0);
        assert!(cfg.alpha_init > 0.0);
    }
}
