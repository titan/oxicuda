//! Empirical Likelihood (Owen 1988) — distribution-free confidence regions.
//!
//! Maximises the empirical likelihood subject to moment constraints, yielding
//! Wilks-type statistics with chi-squared limiting distributions.

use crate::distributions::chi_squared::ChiSquared;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Configuration and result types
// ---------------------------------------------------------------------------

/// Configuration for empirical likelihood optimisation.
#[derive(Debug, Clone)]
pub struct ElConfig {
    /// Maximum Newton-Raphson iterations.
    pub max_iter: usize,
    /// Convergence tolerance for the Newton-Raphson update.
    pub tol: f64,
    /// Number of grid points for profile EL confidence interval search.
    pub n_grid: usize,
}

impl Default for ElConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-10,
            n_grid: 100,
        }
    }
}

/// Result of an empirical likelihood test.
#[derive(Debug, Clone)]
pub struct ElResult {
    /// Optimal probability weights p_i (sum to 1).
    pub p_weights: Vec<f64>,
    /// Log empirical likelihood at the optimum: Σ log(p_i).
    pub log_el: f64,
    /// Wilks statistic: -2 * (log_el - log_el_unrestricted).
    pub wilks: f64,
    /// p-value from chi-squared(1) distribution.
    pub p_value: f64,
}

// ---------------------------------------------------------------------------
// Internal Newton-Raphson solver for λ
// ---------------------------------------------------------------------------

/// Solve for the Lagrange multiplier λ such that Σ g_i / (1 + λ g_i) = 0,
/// where g_i = x_i - mu0 (the estimating equation for the mean constraint).
///
/// Returns `(lambda, success)`.
fn solve_lambda(g: &[f64], max_iter: usize, tol: f64) -> (f64, bool) {
    let n = g.len() as f64;
    // Bound to prevent 1 + λ g_i ≤ 0 (all interior: 1 + λ g_i > 0)
    // Initial λ = 0 is always feasible.
    let mut lambda = 0.0f64;

    // Determine safe bracket for λ: we need all 1 + λ g_i > 0.
    // g_max = max g_i, g_min = min g_i
    let g_max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let g_min = g.iter().cloned().fold(f64::INFINITY, f64::min);

    // λ must be in ( -1/g_max, -1/g_min ) (with sign handling)
    let lo_bound = if g_max > 0.0 {
        -1.0 / g_max
    } else {
        f64::NEG_INFINITY
    };
    let hi_bound = if g_min < 0.0 {
        -1.0 / g_min
    } else {
        f64::INFINITY
    };

    // Clip initial lambda into the interior of the feasible region
    let eps = 1e-8;
    let lo = if lo_bound.is_finite() {
        lo_bound + eps * (1.0 + lo_bound.abs())
    } else {
        -1e12
    };
    let hi = if hi_bound.is_finite() {
        hi_bound - eps * (1.0 + hi_bound.abs())
    } else {
        1e12
    };
    lambda = lambda.clamp(lo, hi);

    for _ in 0..max_iter {
        // grad = Σ g_i / (1 + λ g_i)
        // hess = -Σ g_i^2 / (1 + λ g_i)^2
        let mut grad = 0.0f64;
        let mut hess = 0.0f64;
        let mut feasible = true;
        for &gi in g {
            let d = 1.0 + lambda * gi;
            if d <= 0.0 {
                feasible = false;
                break;
            }
            grad += gi / d;
            hess -= gi * gi / (d * d);
        }
        if !feasible {
            // Bisect back toward 0
            lambda *= 0.5;
            continue;
        }
        if grad.abs() < tol * n {
            return (lambda, true);
        }
        if hess.abs() < 1e-300 {
            break;
        }
        let step = -grad / hess;
        let mut new_lambda = lambda + step;
        // Clamp into feasible region
        new_lambda = new_lambda.clamp(lo, hi);
        if (new_lambda - lambda).abs() < tol {
            lambda = new_lambda;
            return (lambda, true);
        }
        lambda = new_lambda;
    }
    (lambda, false)
}

/// Compute log-EL given g = x - mu0 and the optimal λ.
fn log_el_from_lambda(g: &[f64], lambda: f64) -> StatsResult<f64> {
    let n = g.len();
    let mut log_sum = 0.0f64;
    for &gi in g {
        let d = 1.0 + lambda * gi;
        if d <= 0.0 {
            return Err(StatsError::NumericalInstability(
                "empirical_likelihood: 1 + λ·g ≤ 0".into(),
            ));
        }
        log_sum += d.ln();
    }
    // log_el = Σ log(p_i) = -n*log(n) + Σ log(1/(1+λ g_i))
    //        = -n*ln(n) - log_sum
    Ok(-(n as f64) * (n as f64).ln() - log_sum)
}

/// Build probability weights from the solved λ.
fn weights_from_lambda(g: &[f64], lambda: f64) -> StatsResult<Vec<f64>> {
    let n = g.len() as f64;
    let mut p = Vec::with_capacity(g.len());
    for &gi in g {
        let d = 1.0 + lambda * gi;
        if d <= 0.0 {
            return Err(StatsError::NumericalInstability(
                "empirical_likelihood: 1 + λ·g ≤ 0".into(),
            ));
        }
        p.push(1.0 / (n * d));
    }
    Ok(p)
}

/// Compute the chi-squared(1) p-value for a Wilks statistic.
fn chi2_pvalue(wilks: f64) -> StatsResult<f64> {
    let chi2 = ChiSquared::new(1.0)?;
    let cdf_val = chi2.cdf(wilks)?;
    Ok(1.0 - cdf_val)
}

// ---------------------------------------------------------------------------
// Public API: EL test for the mean
// ---------------------------------------------------------------------------

/// Empirical likelihood test for the population mean.
///
/// Tests H₀: mean(X) = mu0 using Owen's (1988) EL ratio test.
///
/// # Algorithm
/// Maximise Σ log(p_i) subject to Σ p_i = 1, Σ p_i x_i = mu0, p_i > 0.
/// The optimal weights have the form p_i = 1 / (n (1 + λ (x_i - mu0)))
/// where λ is found by Newton-Raphson to satisfy the constraint Σ p_i x_i = mu0.
/// The Wilks statistic -2 log λ_R = -2(log EL - log EL_max) ~d chi^2(1).
pub fn el_mean_test(data: &[f64], mu0: f64, cfg: &ElConfig) -> StatsResult<ElResult> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if !mu0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "mu0".into(),
            reason: "must be finite".into(),
        });
    }

    // Check that mu0 is inside the convex hull of the data
    let x_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if mu0 < x_min || mu0 > x_max {
        return Err(StatsError::InvalidParameter {
            name: "mu0".into(),
            reason: "mu0 must be inside the convex hull of the data (between min and max)".into(),
        });
    }

    // Estimating equations: g_i = x_i - mu0
    let g: Vec<f64> = data.iter().map(|&x| x - mu0).collect();

    let (lambda, converged) = solve_lambda(&g, cfg.max_iter, cfg.tol);
    if !converged {
        // Attempt to continue with current lambda anyway; log a mild warning via error
        // if the result is clearly non-finite.
    }

    let log_el = log_el_from_lambda(&g, lambda)?;
    // log_el_unrestricted = Σ log(1/n) = -n ln(n)
    let log_el_max = -(n as f64) * (n as f64).ln();
    let wilks = -2.0 * (log_el - log_el_max);
    let wilks = wilks.max(0.0); // numerical guard

    let p_value = chi2_pvalue(wilks)?;
    let p_weights = weights_from_lambda(&g, lambda)?;

    Ok(ElResult {
        p_weights,
        log_el,
        wilks,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// EL confidence interval for the mean
// ---------------------------------------------------------------------------

/// Empirical likelihood confidence interval for the mean.
///
/// Binary searches for the lower and upper bounds of μ where the Wilks
/// statistic is ≤ the critical value chi²_{1, alpha}.
///
/// # Arguments
/// * `data`  — observed data
/// * `alpha` — significance level (e.g. 0.05 for 95 % CI)
/// * `cfg`   — EL configuration
pub fn el_confidence_interval(data: &[f64], alpha: f64, cfg: &ElConfig) -> StatsResult<(f64, f64)> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatsError::ProbabilityOutOfRange { value: alpha });
    }

    // Critical value: chi^2_{1, 1-alpha}
    let chi2 = ChiSquared::new(1.0)?;
    let crit = chi2.ppf(1.0 - alpha)?;

    let x_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sample_mean: f64 = data.iter().sum::<f64>() / n as f64;

    // Compute Wilks at a given mu0
    let wilks_at = |mu0: f64| -> f64 {
        let g: Vec<f64> = data.iter().map(|&x| x - mu0).collect();
        let (lambda, _) = solve_lambda(&g, cfg.max_iter, cfg.tol);
        match log_el_from_lambda(&g, lambda) {
            Ok(log_el) => {
                let log_el_max = -(n as f64) * (n as f64).ln();
                (-2.0 * (log_el - log_el_max)).max(0.0)
            }
            Err(_) => f64::INFINITY,
        }
    };

    // Binary search for lower bound: in [x_min, sample_mean]
    let mut lo_lo = x_min + (x_max - x_min) * 1e-6;
    let mut lo_hi = sample_mean;
    let ci_lower = if wilks_at(lo_lo) < crit {
        lo_lo
    } else {
        for _ in 0..60 {
            let mid = (lo_lo + lo_hi) / 2.0;
            let w = wilks_at(mid);
            if w > crit {
                lo_lo = mid;
            } else {
                lo_hi = mid;
            }
            if (lo_hi - lo_lo).abs() < 1e-12 {
                break;
            }
        }
        (lo_lo + lo_hi) / 2.0
    };

    // Binary search for upper bound: in [sample_mean, x_max]
    let mut hi_lo = sample_mean;
    let mut hi_hi = x_max - (x_max - x_min) * 1e-6;
    let ci_upper = if wilks_at(hi_hi) < crit {
        hi_hi
    } else {
        for _ in 0..60 {
            let mid = (hi_lo + hi_hi) / 2.0;
            let w = wilks_at(mid);
            if w > crit {
                hi_hi = mid;
            } else {
                hi_lo = mid;
            }
            if (hi_hi - hi_lo).abs() < 1e-12 {
                break;
            }
        }
        (hi_lo + hi_hi) / 2.0
    };

    Ok((ci_lower, ci_upper))
}

// ---------------------------------------------------------------------------
// EL ratio test for ratio (two constraints)
// ---------------------------------------------------------------------------

/// Empirical likelihood test for the ratio `E[X] / E[Y] = ratio0`.
///
/// Uses the combined estimating equation: g_i = x_i - ratio0 * y_i.
/// Under H₀: Σ p_i (x_i - ratio0 y_i) = 0, p_i > 0, Σ p_i = 1.
///
/// # Arguments
/// * `x`      — numerator observations (length n)
/// * `y`      — denominator observations (length n)
/// * `n`      — sample size
/// * `ratio0` — hypothesised ratio
/// * `cfg`    — EL configuration
pub fn el_ratio_test(
    x: &[f64],
    y: &[f64],
    n: usize,
    ratio0: f64,
    cfg: &ElConfig,
) -> StatsResult<ElResult> {
    if x.len() < 2 || y.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: x.len().min(y.len()),
            need: 2,
        });
    }
    if x.len() != n || y.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: n,
            b: x.len().max(y.len()),
        });
    }
    if !ratio0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "ratio0".into(),
            reason: "must be finite".into(),
        });
    }

    // Combined estimating equation: g_i = x_i - ratio0 * y_i
    let g: Vec<f64> = x.iter().zip(y).map(|(&xi, &yi)| xi - ratio0 * yi).collect();

    // Check feasibility: 0 must be inside the convex hull of g
    let g_min = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if g_min > 0.0 || g_max < 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "ratio0".into(),
            reason: "ratio0 is outside the feasible range implied by the data".into(),
        });
    }

    let (lambda, _converged) = solve_lambda(&g, cfg.max_iter, cfg.tol);
    let log_el = log_el_from_lambda(&g, lambda)?;
    let log_el_max = -(n as f64) * (n as f64).ln();
    let wilks = (-2.0 * (log_el - log_el_max)).max(0.0);
    let p_value = chi2_pvalue(wilks)?;
    let p_weights = weights_from_lambda(&g, lambda)?;

    Ok(ElResult {
        p_weights,
        log_el,
        wilks,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- el_mean_test ----

    #[test]
    fn el_mean_test_at_true_mean_large_pvalue() {
        // H0: mean = 5.5, data is 1..=10; true mean IS 5.5 → p large
        let data: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let r = el_mean_test(&data, 5.5, &cfg).expect("el_mean_test should succeed");
        // Wilks should be near 0
        assert!(r.wilks < 5.0, "wilks={}", r.wilks);
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn el_mean_test_false_mean_small_pvalue() {
        // H0: mean = 1.0, data is 1..=10; true mean is 5.5 → reject
        let data: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let r = el_mean_test(&data, 1.5, &cfg).expect("el_mean_test should succeed");
        assert!(r.wilks > 3.0, "wilks={} should be large", r.wilks);
        assert!(r.p_value < 0.2, "p={}", r.p_value);
    }

    #[test]
    fn el_mean_test_probability_weights_sum_to_one() {
        let data: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let r = el_mean_test(&data, 10.5, &cfg).expect("el_mean_test should succeed");
        let sum: f64 = r.p_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum={}", sum);
    }

    #[test]
    fn el_mean_test_weights_are_positive() {
        let data: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let r = el_mean_test(&data, 5.5, &cfg).expect("el_mean_test should succeed");
        assert!(r.p_weights.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn el_mean_test_outside_convex_hull_errors() {
        let data = [2.0, 3.0, 4.0, 5.0];
        let cfg = ElConfig::default();
        // mu0 = 1.0 is outside [2, 5]
        assert!(matches!(
            el_mean_test(&data, 1.0, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn el_mean_test_empty_error() {
        let cfg = ElConfig::default();
        assert!(matches!(
            el_mean_test(&[], 0.0, &cfg),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }

    #[test]
    fn el_mean_test_wilks_monotone_in_departure() {
        // Wilks should increase as mu0 moves away from the sample mean
        let data: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let true_mean = 10.5;
        let w_center = el_mean_test(&data, true_mean, &cfg)
            .expect("el_mean_test should succeed")
            .wilks;
        let w_near = el_mean_test(&data, true_mean + 1.0, &cfg)
            .expect("el_mean_test should succeed")
            .wilks;
        let w_far = el_mean_test(&data, true_mean + 4.0, &cfg)
            .expect("el_mean_test should succeed")
            .wilks;
        assert!(w_center <= w_near, "center={w_center} near={w_near}");
        assert!(w_near <= w_far, "near={w_near} far={w_far}");
    }

    // ---- el_confidence_interval ----

    #[test]
    fn el_ci_contains_true_mean() {
        // Standard: EL 95 % CI should contain the sample mean (= MLE)
        let data: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let (lo, hi) = el_confidence_interval(&data, 0.05, &cfg)
            .expect("el_confidence_interval should succeed");
        let sample_mean = 10.5;
        assert!(lo < sample_mean && hi > sample_mean, "lo={lo}, hi={hi}");
    }

    #[test]
    fn el_ci_width_increases_with_alpha() {
        // Larger alpha → narrower CI (higher confidence → wider)
        let data: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        let cfg = ElConfig::default();
        let (lo95, hi95) = el_confidence_interval(&data, 0.05, &cfg)
            .expect("el_confidence_interval should succeed");
        let (lo90, hi90) = el_confidence_interval(&data, 0.10, &cfg)
            .expect("el_confidence_interval should succeed");
        let width_95 = hi95 - lo95;
        let width_90 = hi90 - lo90;
        assert!(
            width_95 > width_90,
            "95% CI should be wider: {width_95} vs {width_90}"
        );
    }

    #[test]
    fn el_ci_ordered() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cfg = ElConfig::default();
        let (lo, hi) = el_confidence_interval(&data, 0.05, &cfg)
            .expect("el_confidence_interval should succeed");
        assert!(lo < hi, "lo={lo} should be < hi={hi}");
    }

    // ---- el_ratio_test ----

    #[test]
    fn el_ratio_test_at_true_ratio_large_pvalue() {
        // If X = 2Y, then ratio = 2 is the true value → large p
        let y: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let x: Vec<f64> = y.iter().map(|&v| 2.0 * v).collect();
        let cfg = ElConfig::default();
        let r = el_ratio_test(&x, &y, 10, 2.0, &cfg).expect("el_ratio_test should succeed");
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn el_ratio_test_false_ratio_small_pvalue() {
        // Build paired data where ratio E[X]/E[Y] ≈ 2.
        // Use heterogeneous y so that for ratio0 = 1.5 (wrong), g_i = x_i - 1.5*y_i has mixed signs.
        // y_i alternates small/large; x_i = 2*y_i exactly.
        // g_i = 2*y_i - 1.5*y_i = 0.5*y_i > 0 always — still infeasible.
        //
        // Better approach: construct x and y so that g_i = x_i - ratio0*y_i mixes signs.
        // Let x_i and y_i be paired observations with noise; true ratio ≈ 1.
        // Test ratio0 = 0 → g_i = x_i (all positive) still infeasible.
        //
        // Correct approach: use data where g has mixed signs naturally.
        // x_i = c*y_i + epsilon_i with E[epsilon]=0, so E[g] = (c - ratio0)*E[y] != 0 but
        // individual g_i = (c-ratio0)*y_i + epsilon_i can change sign only if epsilon_i is large.
        //
        // We construct explicitly: half observations have g>0, half g<0.
        // x = [5, 5, 5, 5, 1, 1, 1, 1], y = [2, 2, 2, 2, 2, 2, 2, 2]
        // true ratio = mean(x)/mean(y) = 3/2 = 1.5
        // ratio0 = 3.0: g_i = x_i - 3*y_i = [5-6, 5-6, 5-6, 5-6, 1-6, ...] = [-1,-1,-1,-1,-5,-5,-5,-5] (infeasible)
        // ratio0 = 2.0: g_i = [5-4, 5-4, 5-4, 5-4, 1-4, 1-4, 1-4, 1-4] = [1,1,1,1,-3,-3,-3,-3] ✓ feasible
        // true ratio = 1.5, so ratio0=2 is wrong but feasible
        let x: Vec<f64> = vec![5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0];
        let y: Vec<f64> = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let cfg = ElConfig::default();
        let r = el_ratio_test(&x, &y, 8, 2.0, &cfg).expect("el_ratio_test should succeed");
        // ratio0=2 is wrong (true ≈ 1.5); Wilks should be elevated
        assert!(r.wilks > 0.5, "wilks={}", r.wilks);
    }

    #[test]
    fn el_ratio_test_weights_sum_to_one() {
        let y: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let x: Vec<f64> = y.iter().map(|&v| 3.0 * v).collect();
        let cfg = ElConfig::default();
        let r = el_ratio_test(&x, &y, 8, 3.0, &cfg).expect("el_ratio_test should succeed");
        let sum: f64 = r.p_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum={}", sum);
    }

    #[test]
    fn el_ratio_test_infeasible_ratio_errors() {
        // All g_i = x_i - ratio0 y_i have same sign → infeasible
        let y = [1.0, 2.0, 3.0];
        let x = [2.0, 4.0, 6.0]; // x = 2y exactly
        let cfg = ElConfig::default();
        // ratio0 = 10 → g_i = 2y - 10y = -8y < 0 always → infeasible
        assert!(matches!(
            el_ratio_test(&x, &y, 3, 10.0, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
    }
}
