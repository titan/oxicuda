//! Polyak step-size subgradient method (Polyak 1969).
//!
//! For a non-smooth convex objective `f`, the projected subgradient iteration is
//!
//! ```text
//! x_{k+1} = Π_C( x_k − α_k g_k ),   g_k ∈ ∂f(x_k)
//! ```
//!
//! The **Polyak step-size** chooses
//!
//! ```text
//! α_k = (f(x_k) − f★) / ‖g_k‖²
//! ```
//!
//! which is provably optimal in the sense that it drives `‖x_k − x★‖` down by the
//! largest guaranteed amount per iteration.  When the optimal value `f★` is known
//! exactly the method enjoys geometric convergence for sharp (strongly-convex with
//! sharp subgradients) problems.
//!
//! When `f★` is **unknown** we use the *estimated-target* variant of Brännlund (1993)
//! / Goffin–Kiwiel (1999): keep a running estimate `f_est = f_rec − δ_l` where
//! `f_rec` is the best (lowest) objective seen so far and `δ_l > 0` is a target
//! level.  The level is halved whenever the path length since the last improvement
//! exceeds a budget `B`, and is grown geometrically while progress continues.  This
//! converges to the true optimum without prior knowledge of `f★`.
//!
//! # References
//!
//! - B. T. Polyak (1969), "Minimization of unsmooth functionals", USSR Comput. Math.
//!   & Math. Phys. 9(3):14-29.
//! - S. Boyd, L. Xiao & A. Mutapcic (2003), "Subgradient methods", lecture notes,
//!   Stanford EE392o.
//! - K. C. Kiwiel (1996), "The efficiency of subgradient projection methods for
//!   convex optimization", SIAM J. Control Optim.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Target-level rule for the Polyak step-size.
#[derive(Debug, Clone)]
pub enum PolyakTarget {
    /// The optimal value `f★` is known exactly.
    Known {
        /// Known optimal objective value.
        f_star: f64,
    },
    /// `f★` is unknown; estimate it with an adaptive target level (Brännlund 1993).
    ///
    /// `delta0` is the initial target level `δ_0 > 0`, `path_budget` is the path-length
    /// budget `B` before the level is halved.
    Estimated {
        /// Initial target level `δ_0 > 0`.
        delta0: f64,
        /// Path-length budget `B > 0` between level reductions.
        path_budget: f64,
    },
}

/// Configuration for the Polyak subgradient method.
#[derive(Debug, Clone)]
pub struct PolyakConfig {
    /// Maximum number of iterations (default `1000`).
    pub max_iter: usize,
    /// Target-level rule (default `Estimated { delta0: 1.0, path_budget: 1.0 }`).
    pub target: PolyakTarget,
    /// Stop when `‖g_k‖₂ < tol` (default `1 × 10⁻⁸`).
    pub grad_tol: f64,
    /// Stop when the best objective is within `obj_tol` of the (known) target
    /// (default `1 × 10⁻⁹`). Only consulted for [`PolyakTarget::Known`].
    pub obj_tol: f64,
    /// Lower clamp on the step size to avoid stalling (default `1 × 10⁻¹²`).
    pub step_min: f64,
    /// Upper clamp on the step size to avoid blow-up (default `1 × 10⁶`).
    pub step_max: f64,
}

impl Default for PolyakConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            target: PolyakTarget::Estimated {
                delta0: 1.0,
                path_budget: 1.0,
            },
            grad_tol: 1e-8,
            obj_tol: 1e-9,
            step_min: 1e-12,
            step_max: 1e6,
        }
    }
}

/// Result of a Polyak subgradient run.
#[derive(Debug, Clone)]
pub struct PolyakResult {
    /// Best iterate found (the one attaining `best_obj`).
    pub x: Vec<f64>,
    /// Best (lowest) objective value seen.
    pub best_obj: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether a stopping criterion (gradient or objective) fired.
    pub converged: bool,
    /// Best-objective history (one entry per iteration).
    pub obj_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------

/// Run the projected subgradient method with the Polyak step-size.
///
/// * `x0` — starting point.
/// * `f` — objective evaluator `f(x)`.
/// * `subgrad` — returns **a** subgradient `g ∈ ∂f(x)`.
/// * `project` — projection onto the feasible set `C` (use the identity closure
///   `|x| Ok(x.to_vec())` for unconstrained problems).
/// * `config` — algorithm configuration.
///
/// Returns the **best** iterate seen (subgradient methods are not monotone, so the
/// running minimiser is returned rather than the last iterate).
pub fn polyak_subgradient<F, G, P>(
    x0: &[f64],
    f: F,
    subgrad: G,
    project: P,
    config: &PolyakConfig,
) -> CvxResult<PolyakResult>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    P: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let n = x0.len();
    if config.step_min <= 0.0 || config.step_max < config.step_min {
        return Err(CvxError::InvalidParameter(format!(
            "polyak step bounds invalid: [{}, {}]",
            config.step_min, config.step_max
        )));
    }

    // Validate / initialise the target-level state.
    let (mut delta, path_budget, estimated) = match config.target {
        PolyakTarget::Known { f_star } => {
            if !f_star.is_finite() {
                return Err(CvxError::InvalidParameter(
                    "polyak f_star must be finite".into(),
                ));
            }
            (0.0, 0.0, false)
        }
        PolyakTarget::Estimated {
            delta0,
            path_budget,
        } => {
            if delta0 <= 0.0 || !delta0.is_finite() {
                return Err(CvxError::InvalidParameter(format!(
                    "polyak delta0 must be > 0, got {delta0}"
                )));
            }
            if path_budget <= 0.0 || !path_budget.is_finite() {
                return Err(CvxError::InvalidParameter(format!(
                    "polyak path_budget must be > 0, got {path_budget}"
                )));
            }
            (delta0, path_budget, true)
        }
    };

    let mut x = project(x0)?;
    if x.len() != n {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
    }

    let mut best_obj = f(&x)?;
    if !best_obj.is_finite() {
        return Err(CvxError::NumericalInstability(
            "polyak: objective not finite at x0".into(),
        ));
    }
    let mut best_x = x.clone();
    // `f_rec` is the lowest objective since the last level reset.
    let mut f_rec = best_obj;
    let mut path_since_reset = 0.0_f64;
    let mut obj_history = Vec::with_capacity(config.max_iter);

    let mut converged = false;
    let mut iters = 0usize;
    for it in 0..config.max_iter {
        iters = it + 1;
        let fx = f(&x)?;
        if !fx.is_finite() {
            return Err(CvxError::NumericalInstability(
                "polyak: objective became non-finite".into(),
            ));
        }
        if fx < best_obj {
            best_obj = fx;
            best_x.copy_from_slice(&x);
        }
        obj_history.push(best_obj);

        // Target value used by the Polyak rule.
        let f_target = match config.target {
            PolyakTarget::Known { f_star } => {
                if best_obj - f_star <= config.obj_tol {
                    converged = true;
                    break;
                }
                f_star
            }
            PolyakTarget::Estimated { .. } => {
                if fx < f_rec {
                    f_rec = fx;
                    // Progress: grow the target level modestly so we keep moving.
                    delta *= 1.5;
                    path_since_reset = 0.0;
                }
                f_rec - delta
            }
        };

        let g = subgrad(&x)?;
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }
        let g_norm = norm2(&g);
        if g_norm < config.grad_tol {
            converged = true;
            break;
        }
        let g_sq = g_norm * g_norm;

        // Polyak step: α = (f(x) − f_target) / ‖g‖².  Numerator clamped non-negative.
        let numer = (fx - f_target).max(0.0);
        let mut alpha = numer / g_sq;
        if !alpha.is_finite() {
            alpha = config.step_min;
        }
        alpha = alpha.clamp(config.step_min, config.step_max);

        // Subgradient step + projection.
        let y: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .map(|(xi, gi)| xi - alpha * gi)
            .collect();
        let x_new = project(&y)?;
        if x_new.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: n,
            });
        }

        // Track path length for the adaptive target-level rule.
        if estimated {
            let mut step_len_sq = 0.0_f64;
            for i in 0..n {
                let d = x_new[i] - x[i];
                step_len_sq += d * d;
            }
            path_since_reset += step_len_sq.sqrt();
            if path_since_reset > path_budget {
                // No sufficient progress within the budget: halve the target level.
                delta *= 0.5;
                path_since_reset = 0.0;
                f_rec = best_obj;
                if delta < config.step_min {
                    converged = true;
                    // The best iterate is tracked separately; stop here.
                    break;
                }
            }
        }

        x = x_new;
    }

    Ok(PolyakResult {
        x: best_x,
        best_obj,
        n_iter: iters,
        converged,
        obj_history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_proj(x: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(x.to_vec())
    }

    // f(x) = |x − 3| on R.  Optimum x★ = 3, f★ = 0.
    fn abs_obj(x: &[f64]) -> CvxResult<f64> {
        Ok((x[0] - 3.0).abs())
    }
    fn abs_subgrad(x: &[f64]) -> CvxResult<Vec<f64>> {
        let g = if x[0] > 3.0 {
            1.0
        } else if x[0] < 3.0 {
            -1.0
        } else {
            0.0
        };
        Ok(vec![g])
    }

    #[test]
    fn known_target_abs_converges() {
        let cfg = PolyakConfig {
            target: PolyakTarget::Known { f_star: 0.0 },
            max_iter: 200,
            ..Default::default()
        };
        let res =
            polyak_subgradient(&[0.0], abs_obj, abs_subgrad, identity_proj, &cfg).expect("ok");
        assert!(res.converged, "should converge with known f_star");
        assert!((res.x[0] - 3.0).abs() < 1e-6, "x = {}", res.x[0]);
        assert!(res.best_obj < 1e-6, "obj = {}", res.best_obj);
    }

    #[test]
    fn estimated_target_abs_converges() {
        let cfg = PolyakConfig {
            target: PolyakTarget::Estimated {
                delta0: 1.0,
                path_budget: 0.5,
            },
            max_iter: 2000,
            ..Default::default()
        };
        let res =
            polyak_subgradient(&[-5.0], abs_obj, abs_subgrad, identity_proj, &cfg).expect("ok");
        assert!(res.best_obj < 1e-3, "obj = {}", res.best_obj);
        assert!((res.x[0] - 3.0).abs() < 1e-2, "x = {}", res.x[0]);
    }

    // f(x) = max_i (a_i·x − b_i): piecewise-linear convex in R^2.
    fn piecewise_max(a: &[[f64; 2]], b: &[f64], x: &[f64]) -> (f64, usize) {
        let mut best = f64::NEG_INFINITY;
        let mut arg = 0usize;
        for (k, ak) in a.iter().enumerate() {
            let v = ak[0] * x[0] + ak[1] * x[1] - b[k];
            if v > best {
                best = v;
                arg = k;
            }
        }
        (best, arg)
    }

    #[test]
    fn piecewise_linear_min_known() {
        // Rows chosen so the min of the max is 0 at x = (0, 0).
        let a = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
        let b = [0.0, 0.0, 0.0, 0.0];
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(piecewise_max(&a, &b, x).0) };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            let (_, arg) = piecewise_max(&a, &b, x);
            Ok(vec![a[arg][0], a[arg][1]])
        };
        let cfg = PolyakConfig {
            target: PolyakTarget::Known { f_star: 0.0 },
            max_iter: 500,
            ..Default::default()
        };
        let res = polyak_subgradient(&[2.0, -3.0], f, g, identity_proj, &cfg).expect("ok");
        assert!(res.best_obj < 1e-4, "obj = {}", res.best_obj);
        assert!(norm2(&res.x) < 1e-2, "‖x‖ = {}", norm2(&res.x));
    }

    #[test]
    fn projection_keeps_feasible() {
        // Minimise |x − 5| over the box [0, 1]; optimum on the boundary at x = 1.
        let proj = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![x[0].clamp(0.0, 1.0)]) };
        let f = |x: &[f64]| -> CvxResult<f64> { Ok((x[0] - 5.0).abs()) };
        let g =
            |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![if x[0] > 5.0 { 1.0 } else { -1.0 }]) };
        let cfg = PolyakConfig {
            target: PolyakTarget::Estimated {
                delta0: 2.0,
                path_budget: 1.0,
            },
            max_iter: 500,
            ..Default::default()
        };
        let res = polyak_subgradient(&[0.5], f, g, proj, &cfg).expect("ok");
        assert!(
            res.x[0] >= -1e-9 && res.x[0] <= 1.0 + 1e-9,
            "x = {}",
            res.x[0]
        );
        assert!((res.x[0] - 1.0).abs() < 1e-2, "x = {}", res.x[0]);
    }

    #[test]
    fn best_iterate_is_returned() {
        // Objective history must be non-increasing (it tracks the running best).
        let cfg = PolyakConfig {
            target: PolyakTarget::Known { f_star: 0.0 },
            max_iter: 100,
            ..Default::default()
        };
        let res =
            polyak_subgradient(&[10.0], abs_obj, abs_subgrad, identity_proj, &cfg).expect("ok");
        for w in res.obj_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "history not monotone: {w:?}");
        }
        // best_obj equals f(best_x).
        let recomputed = abs_obj(&res.x).expect("ok");
        assert!((recomputed - res.best_obj).abs() < 1e-12);
    }

    #[test]
    fn empty_input_errors() {
        let cfg = PolyakConfig::default();
        let err = polyak_subgradient(&[], abs_obj, abs_subgrad, identity_proj, &cfg);
        assert!(matches!(err, Err(CvxError::EmptyInput)));
    }

    #[test]
    fn invalid_delta0_errors() {
        let cfg = PolyakConfig {
            target: PolyakTarget::Estimated {
                delta0: -1.0,
                path_budget: 1.0,
            },
            ..Default::default()
        };
        let err = polyak_subgradient(&[0.0], abs_obj, abs_subgrad, identity_proj, &cfg);
        assert!(matches!(err, Err(CvxError::InvalidParameter(_))));
    }

    #[test]
    fn invalid_step_bounds_error() {
        let cfg = PolyakConfig {
            step_min: 1.0,
            step_max: 0.5,
            ..Default::default()
        };
        let err = polyak_subgradient(&[0.0], abs_obj, abs_subgrad, identity_proj, &cfg);
        assert!(matches!(err, Err(CvxError::InvalidParameter(_))));
    }

    #[test]
    fn dimension_mismatch_subgradient() {
        let bad_sub = |_x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![1.0, 2.0]) };
        let cfg = PolyakConfig {
            target: PolyakTarget::Known { f_star: 0.0 },
            ..Default::default()
        };
        let err = polyak_subgradient(&[0.0], abs_obj, bad_sub, identity_proj, &cfg);
        assert!(matches!(err, Err(CvxError::DimensionMismatch { .. })));
    }

    #[test]
    fn l1_regression_subgradient() {
        // Least-absolute-deviations: min_x Σ_i |x − d_i| in R^1.
        // Optimum is the median of d.
        let d = [1.0_f64, 2.0, 2.0, 8.0, 100.0]; // median = 2
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(d.iter().map(|di| (x[0] - di).abs()).sum()) };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            let s: f64 = d
                .iter()
                .map(|di| {
                    if x[0] > *di {
                        1.0
                    } else if x[0] < *di {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .sum();
            Ok(vec![s])
        };
        let cfg = PolyakConfig {
            target: PolyakTarget::Estimated {
                delta0: 5.0,
                path_budget: 2.0,
            },
            max_iter: 5000,
            ..Default::default()
        };
        let res = polyak_subgradient(&[50.0], f, g, identity_proj, &cfg).expect("ok");
        assert!(
            (res.x[0] - 2.0).abs() < 0.2,
            "median estimate = {}",
            res.x[0]
        );
    }

    #[test]
    fn convergence_flag_false_when_capped() {
        // One iteration on a far point cannot reach tolerance.
        let cfg = PolyakConfig {
            target: PolyakTarget::Estimated {
                delta0: 0.01,
                path_budget: 100.0,
            },
            max_iter: 1,
            grad_tol: 1e-12,
            ..Default::default()
        };
        let res =
            polyak_subgradient(&[100.0], abs_obj, abs_subgrad, identity_proj, &cfg).expect("ok");
        assert!(!res.converged);
        assert_eq!(res.n_iter, 1);
    }
}
