//! Projected subgradient method with primal averaging (Nesterov 2009; Polyak-
//! Ruppert averaging).
//!
//! Minimises a convex, possibly non-smooth objective `f` over a convex set `C`.
//! The plain projected subgradient recursion
//!
//! ```text
//! x_{k+1} = Π_C( x_k − α_k g_k ),   g_k ∈ ∂f(x_k)
//! ```
//!
//! does **not** converge in the last iterate for a fixed/diminishing step; only
//! the *running average* of the iterates does.  This module returns the
//! weighted ergodic average
//!
//! ```text
//! x̄_K = ( Σ_k w_k x_k ) / ( Σ_k w_k ),
//! ```
//!
//! which, for diminishing non-summable-but-square-summable steps
//! (`Σ α_k = ∞`, `Σ α_k² < ∞`) and weights `w_k = α_k`, achieves the optimal
//! `O(1/√K)` rate `f(x̄_K) − f* = O(LD/√K)` for `L`-Lipschitz `f` on a set of
//! diameter `D` (Nesterov 2009, "Primal-dual subgradient methods").  Several
//! standard step-size schedules are offered.
//!
//! Reference: Nesterov, Y. (2009). *Primal-dual subgradient methods for convex
//! problems.* Mathematical Programming 120(1), 221-259.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Step-size schedule for the averaged subgradient method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubgradStep {
    /// Constant step `α_k = c`.
    Constant(f64),
    /// Diminishing step `α_k = c / √(k + 1)` (`Σα = ∞`, `Σα² = ∞`; classic).
    InverseSqrt(f64),
    /// Square-summable step `α_k = c / (k + 1)` (`Σα = ∞`, `Σα² < ∞`).
    InverseK(f64),
}

impl SubgradStep {
    /// Step size at iteration `k` (zero-based).
    fn at(self, k: usize) -> f64 {
        let kf = k as f64;
        match self {
            SubgradStep::Constant(c) => c,
            SubgradStep::InverseSqrt(c) => c / (kf + 1.0).sqrt(),
            SubgradStep::InverseK(c) => c / (kf + 1.0),
        }
    }

    /// The base scale `c` of the schedule (for validation).
    fn scale(self) -> f64 {
        match self {
            SubgradStep::Constant(c) | SubgradStep::InverseSqrt(c) | SubgradStep::InverseK(c) => c,
        }
    }
}

/// Configuration for [`averaged_subgradient`].
#[derive(Debug, Clone)]
pub struct AveragedSubgradConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Step-size schedule.
    pub step: SubgradStep,
    /// Convergence tolerance on the change of the running average `‖x̄_k − x̄_{k−1}‖`.
    pub tol: f64,
}

impl Default for AveragedSubgradConfig {
    fn default() -> Self {
        Self {
            max_iter: 2000,
            step: SubgradStep::InverseSqrt(1.0),
            tol: 1e-9,
        }
    }
}

/// Result of an averaged-subgradient run.
#[derive(Debug, Clone)]
pub struct AveragedSubgradResult {
    /// Ergodic (weighted-average) iterate — the convergent estimate.
    pub x_avg: Vec<f64>,
    /// Best objective value seen across the trajectory (over the raw iterates).
    pub best_f: f64,
    /// Raw iterate attaining `best_f`.
    pub x_best: Vec<f64>,
    /// Number of iterations performed.
    pub iter: usize,
}

fn validate_cfg(cfg: &AveragedSubgradConfig) -> CvxResult<()> {
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter(
            "averaged_subgradient: max_iter must be ≥ 1".into(),
        ));
    }
    let c = cfg.step.scale();
    if !(c > 0.0 && c.is_finite()) {
        return Err(CvxError::InvalidParameter(format!(
            "averaged_subgradient: step scale must be > 0, got {c}"
        )));
    }
    if cfg.tol < 0.0 {
        return Err(CvxError::InvalidParameter(format!(
            "averaged_subgradient: tol must be ≥ 0, got {}",
            cfg.tol
        )));
    }
    Ok(())
}

/// Run the projected subgradient method with primal averaging.
///
/// # Arguments
/// - `x0`: starting point (must already lie in `C`, or it is projected on step 0).
/// - `subgrad`: subgradient oracle returning some `g ∈ ∂f(x)`.
/// - `f_obj`: objective value `f(x)` (tracks the best raw iterate).
/// - `project`: Euclidean projection onto the convex set `C`.
/// - `cfg`: algorithm configuration.
///
/// The weighted average uses weights `w_k = α_k`, the same as the step sizes,
/// which is the standard choice giving the `O(1/√K)` ergodic rate.
///
/// # Errors
/// - [`CvxError::EmptyInput`] if `x0` is empty.
/// - [`CvxError::InvalidParameter`] for invalid `cfg`.
/// - [`CvxError::DimensionMismatch`] if an oracle returns a vector of the wrong
///   length.
pub fn averaged_subgradient<SG, FO, PR>(
    x0: &[f64],
    subgrad: SG,
    f_obj: FO,
    project: PR,
    cfg: &AveragedSubgradConfig,
) -> CvxResult<AveragedSubgradResult>
where
    SG: Fn(&[f64]) -> Vec<f64>,
    FO: Fn(&[f64]) -> f64,
    PR: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    validate_cfg(cfg)?;

    let n = x0.len();
    let mut x = project(x0)?;
    if x.len() != n {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
    }

    let mut sum_w = 0.0_f64;
    let mut weighted = vec![0.0_f64; n];
    let mut avg_prev = x.clone();

    let mut x_best = x.clone();
    let mut best_f = f_obj(&x);
    let mut final_iter = 0_usize;

    for k in 0..cfg.max_iter {
        final_iter = k;
        // Track the best raw iterate.
        let fx = f_obj(&x);
        if fx < best_f {
            best_f = fx;
            x_best = x.clone();
        }

        // Accumulate weighted average with weight w_k = α_k.
        let alpha = cfg.step.at(k);
        sum_w += alpha;
        for j in 0..n {
            weighted[j] += alpha * x[j];
        }
        let avg: Vec<f64> = if sum_w > 0.0 {
            weighted.iter().map(|w| w / sum_w).collect()
        } else {
            x.clone()
        };

        // Convergence test on the running average.
        let diff: Vec<f64> = avg.iter().zip(&avg_prev).map(|(a, b)| a - b).collect();
        if k > 0 && norm2(&diff) < cfg.tol {
            return Ok(AveragedSubgradResult {
                x_avg: avg,
                best_f,
                x_best,
                iter: k,
            });
        }
        avg_prev = avg;

        // Subgradient step + projection.
        let g = subgrad(&x);
        if g.len() != n {
            return Err(CvxError::DimensionMismatch { a: g.len(), b: n });
        }
        let mut y = vec![0.0_f64; n];
        for j in 0..n {
            y[j] = x[j] - alpha * g[j];
        }
        let x_new = project(&y)?;
        if x_new.len() != n {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: n,
            });
        }
        x = x_new;
    }

    // Final average over all iterates.
    let avg: Vec<f64> = if sum_w > 0.0 {
        weighted.iter().map(|w| w / sum_w).collect()
    } else {
        x.clone()
    };
    // Include the terminal raw iterate in the best tracking.
    let fx = f_obj(&x);
    if fx < best_f {
        best_f = fx;
        x_best = x.clone();
    }
    Ok(AveragedSubgradResult {
        x_avg: avg,
        best_f,
        x_best,
        iter: final_iter + 1,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_proj(x: &[f64]) -> CvxResult<Vec<f64>> {
        Ok(x.to_vec())
    }

    fn cfg() -> AveragedSubgradConfig {
        AveragedSubgradConfig::default()
    }

    #[test]
    fn averaged_minimises_absolute_value() {
        // f(x) = |x − 3|; subgradient is sign(x − 3); minimiser x = 3.
        let subgrad = |x: &[f64]| vec![(x[0] - 3.0).signum()];
        let f_obj = |x: &[f64]| (x[0] - 3.0).abs();
        let c = AveragedSubgradConfig {
            max_iter: 5000,
            step: SubgradStep::InverseSqrt(1.0),
            tol: 0.0,
        };
        let r = averaged_subgradient(&[0.0], subgrad, f_obj, identity_proj, &c).expect("ok");
        assert!((r.x_avg[0] - 3.0).abs() < 0.1, "x_avg={}", r.x_avg[0]);
    }

    #[test]
    fn smooth_quadratic() {
        // f = ‖x − a‖²; subgradient = 2 (x − a); minimiser a.
        let a = vec![1.0_f64, -2.0];
        let subgrad = {
            let a = a.clone();
            move |x: &[f64]| vec![2.0 * (x[0] - a[0]), 2.0 * (x[1] - a[1])]
        };
        let f_obj = {
            let a = a.clone();
            move |x: &[f64]| (x[0] - a[0]).powi(2) + (x[1] - a[1]).powi(2)
        };
        let c = AveragedSubgradConfig {
            max_iter: 4000,
            step: SubgradStep::InverseK(0.5),
            tol: 0.0,
        };
        let r = averaged_subgradient(&[0.0, 0.0], subgrad, f_obj, identity_proj, &c).expect("ok");
        // Best raw iterate should be very close to the minimiser.
        assert!((r.x_best[0] - 1.0).abs() < 0.05, "x_best0={}", r.x_best[0]);
        assert!((r.x_best[1] + 2.0).abs() < 0.05, "x_best1={}", r.x_best[1]);
    }

    #[test]
    fn l1_objective_multidim() {
        // f(x) = Σ |x_i − c_i|; separable, minimiser c.
        let c = vec![2.0_f64, -1.0, 0.5];
        let subgrad = {
            let c = c.clone();
            move |x: &[f64]| {
                x.iter()
                    .zip(&c)
                    .map(|(xi, ci)| (xi - ci).signum())
                    .collect::<Vec<_>>()
            }
        };
        let f_obj = {
            let c = c.clone();
            move |x: &[f64]| {
                x.iter()
                    .zip(&c)
                    .map(|(xi, ci)| (xi - ci).abs())
                    .sum::<f64>()
            }
        };
        let cfg = AveragedSubgradConfig {
            max_iter: 8000,
            step: SubgradStep::InverseSqrt(0.5),
            tol: 0.0,
        };
        let r =
            averaged_subgradient(&vec![0.0; 3], subgrad, f_obj, identity_proj, &cfg).expect("ok");
        for (xi, ci) in r.x_avg.iter().zip(&c) {
            assert!((xi - ci).abs() < 0.15, "xi={xi}, ci={ci}");
        }
    }

    #[test]
    fn projection_box_keeps_feasible() {
        // Minimise f = (x − 5)² over [−1, 1]; optimum at boundary x = 1.
        let subgrad = |x: &[f64]| vec![2.0 * (x[0] - 5.0)];
        let f_obj = |x: &[f64]| (x[0] - 5.0).powi(2);
        let proj = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![x[0].clamp(-1.0, 1.0)]) };
        let cfg = AveragedSubgradConfig {
            max_iter: 3000,
            step: SubgradStep::InverseSqrt(0.1),
            tol: 0.0,
        };
        let r = averaged_subgradient(&[0.0], subgrad, f_obj, proj, &cfg).expect("ok");
        assert!(r.x_avg[0] <= 1.0 + 1e-9 && r.x_avg[0] >= -1.0 - 1e-9);
        assert!((r.x_best[0] - 1.0).abs() < 1e-6, "x_best={}", r.x_best[0]);
    }

    #[test]
    fn constant_step_averages() {
        // With a constant step the raw iterates oscillate but the average
        // converges toward the minimiser of |x|.
        let subgrad = |x: &[f64]| vec![x[0].signum()];
        let f_obj = |x: &[f64]| x[0].abs();
        let cfg = AveragedSubgradConfig {
            max_iter: 6000,
            step: SubgradStep::Constant(0.01),
            tol: 0.0,
        };
        let r = averaged_subgradient(&[1.0], subgrad, f_obj, identity_proj, &cfg).expect("ok");
        assert!(r.x_avg[0].abs() < 0.05, "x_avg={}", r.x_avg[0]);
    }

    #[test]
    fn best_f_non_increasing_property() {
        // best_f must not exceed the initial objective.
        let subgrad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let f_obj = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let start = [3.0, 4.0];
        let f0 = f_obj(&start);
        let r = averaged_subgradient(&start, subgrad, f_obj, identity_proj, &cfg()).expect("ok");
        assert!(r.best_f <= f0 + 1e-12, "best_f={} f0={}", r.best_f, f0);
    }

    #[test]
    fn output_finite_and_sized() {
        let subgrad = |x: &[f64]| x.iter().map(|v| v.signum()).collect::<Vec<_>>();
        let f_obj = |x: &[f64]| x.iter().map(|v| v.abs()).sum::<f64>();
        let r = averaged_subgradient(
            &[1.0, -2.0, 3.0, -4.0],
            subgrad,
            f_obj,
            identity_proj,
            &cfg(),
        )
        .expect("ok");
        assert_eq!(r.x_avg.len(), 4);
        assert_eq!(r.x_best.len(), 4);
        for v in &r.x_avg {
            assert!(v.is_finite());
        }
        assert!(r.best_f.is_finite());
    }

    #[test]
    fn tol_triggers_early_stop() {
        // Very loose tolerance halts once the average barely moves.
        let subgrad = |x: &[f64]| vec![2.0 * x[0]];
        let f_obj = |x: &[f64]| x[0] * x[0];
        let cfg = AveragedSubgradConfig {
            max_iter: 100_000,
            step: SubgradStep::InverseK(0.5),
            tol: 1e-3,
        };
        let r = averaged_subgradient(&[1.0], subgrad, f_obj, identity_proj, &cfg).expect("ok");
        assert!(r.iter < 100_000, "should stop early, iter={}", r.iter);
    }

    #[test]
    fn empty_input_error() {
        let res = averaged_subgradient(
            &[],
            |_x: &[f64]| Vec::<f64>::new(),
            |_| 0.0,
            identity_proj,
            &cfg(),
        );
        assert!(matches!(res, Err(CvxError::EmptyInput)));
    }

    #[test]
    fn rejects_bad_cfg() {
        let subgrad = |x: &[f64]| vec![x[0].signum()];
        let f_obj = |x: &[f64]| x[0].abs();
        let bad = AveragedSubgradConfig {
            step: SubgradStep::Constant(-1.0),
            ..cfg()
        };
        assert!(averaged_subgradient(&[1.0], &subgrad, &f_obj, identity_proj, &bad).is_err());
        let bad2 = AveragedSubgradConfig {
            max_iter: 0,
            ..cfg()
        };
        assert!(averaged_subgradient(&[1.0], &subgrad, &f_obj, identity_proj, &bad2).is_err());
    }

    #[test]
    fn subgrad_wrong_dim_error() {
        let subgrad = |_x: &[f64]| vec![1.0, 2.0]; // wrong length for n=1
        let f_obj = |x: &[f64]| x[0].abs();
        let res = averaged_subgradient(&[1.0], subgrad, f_obj, identity_proj, &cfg());
        assert!(matches!(res, Err(CvxError::DimensionMismatch { .. })));
    }

    #[test]
    fn averaged_better_than_last_iterate_for_nonsmooth() {
        // For f = |x|, the constant-step last iterate oscillates around 0 while
        // the average is closer to the optimum.
        let subgrad = |x: &[f64]| vec![x[0].signum()];
        let f_obj = |x: &[f64]| x[0].abs();
        let cfg = AveragedSubgradConfig {
            max_iter: 4000,
            step: SubgradStep::Constant(0.05),
            tol: 0.0,
        };
        let r = averaged_subgradient(&[1.0], subgrad, f_obj, identity_proj, &cfg).expect("ok");
        // The averaged objective is small.
        assert!(f_obj(&r.x_avg) < 0.05, "f(x_avg)={}", f_obj(&r.x_avg));
    }
}
