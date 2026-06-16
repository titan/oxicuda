//! Bayesian Optimization with GP surrogate and EI/UCB/PI acquisition.
//!
//! Implements the sequential model-based optimization loop of Mockus (1978),
//! Srinivas et al. (2010), and Brochu et al. (2010 tutorial).
//!
//! # Algorithm
//!
//! 1. Evaluate the objective at `n_init` random points.
//! 2. For each iteration `t = 1..n_iter`:
//!    a. Fit a GP to all observations so far.
//!    b. Sample `n_candidates` random candidate points in the bounds.
//!    c. Evaluate the acquisition function (EI / UCB / PI) at each candidate.
//!    d. Evaluate the objective at the argmax candidate and record the result.
//! 3. Return the best observed (x, y) pair.
//!
//! The GP surrogate uses the RBF kernel from [`crate::gp::gpr`].

use crate::error::{BayesError, BayesResult};
use crate::gp::gpr::{GprConfig, GprKernel, gpr_fit, gpr_predict};
use crate::handle::LcgRng;

// Re-export GprKernel so callers do not need to reach into gp internals.
pub use crate::gp::gpr::GprKernel as GprKernelReexport;

// ─── Acquisition function ─────────────────────────────────────────────────────

/// Acquisition function variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionFn {
    /// Expected Improvement (Mockus 1978).
    ExpectedImprovement,
    /// Upper Confidence Bound (Srinivas 2010).
    UpperConfidenceBound,
    /// Probability of Improvement.
    ProbabilityOfImprovement,
}

// ─── Normal CDF / PDF (Abramowitz & Stegun 7.1.26) ───────────────────────────

/// Standard normal PDF.
#[inline]
fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF via Abramowitz & Stegun 7.1.26 rational approximation.
/// Maximum absolute error < 7.5e-8.
#[inline]
fn normal_cdf(z: f64) -> f64 {
    let z_abs = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * z_abs);
    let phi = (-0.5 * z_abs * z_abs).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let p = 1.0 - phi * poly;
    if z >= 0.0 { p } else { 1.0 - p }
}

// ─── Acquisition value ────────────────────────────────────────────────────────

/// Evaluate a single acquisition function value.
///
/// - `mu`: GP posterior mean at the candidate point.
/// - `sigma`: GP posterior standard deviation (≥ 0).
/// - `f_best`: current best observed objective value.
/// - `kappa`: UCB exploration weight.
/// - `xi`: EI/PI jitter (small positive constant).
#[must_use]
pub fn acquisition_value(
    acq: AcquisitionFn,
    mu: f64,
    sigma: f64,
    f_best: f64,
    kappa: f64,
    xi: f64,
) -> f64 {
    match acq {
        AcquisitionFn::ExpectedImprovement => {
            if sigma <= 0.0 {
                return 0.0;
            }
            let improvement = mu - f_best - xi;
            let z = improvement / sigma;
            improvement * normal_cdf(z) + sigma * normal_pdf(z)
        }
        AcquisitionFn::UpperConfidenceBound => mu + kappa * sigma,
        AcquisitionFn::ProbabilityOfImprovement => {
            if sigma <= 0.0 {
                return if mu > f_best + xi { 1.0 } else { 0.0 };
            }
            normal_cdf((mu - f_best - xi) / sigma)
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Bayesian Optimization.
#[derive(Debug, Clone)]
pub struct BayesOptConfig {
    /// Input dimensionality.
    pub dim: usize,
    /// Search bounds per dimension: `(lo, hi)` with `lo < hi`.
    pub bounds: Vec<(f64, f64)>,
    /// Number of initial random evaluations (before the GP loop).
    pub n_init: usize,
    /// Number of Bayesian optimization iterations after the warm-start.
    pub n_iter: usize,
    /// Number of random candidate points evaluated per acquisition step.
    pub n_candidates: usize,
    /// Acquisition function to use.
    pub acquisition: AcquisitionFn,
    /// UCB exploration weight κ.
    pub ucb_kappa: f64,
    /// EI/PI jitter ξ.
    pub ei_xi: f64,
    /// GP observation noise variance.
    pub noise_variance: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for BayesOptConfig {
    fn default() -> Self {
        Self {
            dim: 1,
            bounds: vec![(0.0, 1.0)],
            n_init: 5,
            n_iter: 20,
            n_candidates: 500,
            acquisition: AcquisitionFn::ExpectedImprovement,
            ucb_kappa: 2.0,
            ei_xi: 0.01,
            noise_variance: 1e-4,
            seed: 0,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result returned by [`bayesopt`].
#[derive(Debug, Clone)]
pub struct BayesOptResult {
    /// Best input found (length `dim`).
    pub best_x: Vec<f64>,
    /// Best objective value found.
    pub best_y: f64,
    /// All evaluated inputs in evaluation order, row-major `(n_evaluations × dim)`.
    pub all_x: Vec<f64>,
    /// All evaluated objective values in evaluation order.
    pub all_y: Vec<f64>,
    /// Total number of objective evaluations (`n_init + n_iter`).
    pub n_evaluations: usize,
}

// ─── Validation ───────────────────────────────────────────────────────────────

fn validate_bo_config(config: &BayesOptConfig) -> BayesResult<()> {
    if config.dim == 0 {
        return Err(BayesError::InvalidConfig("dim must be > 0".into()));
    }
    if config.bounds.len() != config.dim {
        return Err(BayesError::DimensionMismatch {
            expected: config.dim,
            got: config.bounds.len(),
        });
    }
    for (lo, hi) in &config.bounds {
        if lo >= hi {
            return Err(BayesError::InvalidConfig("bounds must have lo < hi".into()));
        }
    }
    if config.n_init == 0 {
        return Err(BayesError::InvalidConfig("n_init must be > 0".into()));
    }
    if config.n_candidates == 0 {
        return Err(BayesError::InvalidConfig("n_candidates must be > 0".into()));
    }
    Ok(())
}

// ─── Random point in bounds ───────────────────────────────────────────────────

/// Sample a single random point within the box defined by `bounds`.
///
/// Uses the safe recipe: `lo + (hi - lo) * (next_u32 / 2^31)`.
fn random_in_bounds(bounds: &[(f64, f64)], rng: &mut LcgRng) -> Vec<f64> {
    bounds
        .iter()
        .map(|(lo, hi)| {
            let u = rng.next_u32() as f64 / 4_294_967_296.0;
            lo + (hi - lo) * u
        })
        .collect()
}

// ─── GP configuration ─────────────────────────────────────────────────────────

fn bo_gpr_config(noise_variance: f64) -> GprConfig {
    GprConfig {
        kernel: GprKernel::Rbf {
            length_scale: 0.3,
            signal_variance: 1.0,
        },
        noise_variance,
        normalize_y: true,
        jitter: 1e-6,
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Run Bayesian Optimization.
///
/// # Errors
/// - `InvalidConfig` if `dim == 0`, `n_init == 0`, `n_candidates == 0`.
/// - `DimensionMismatch` if `bounds.len() != dim`.
/// - `InvalidConfig` if any `bounds[i].lo >= bounds[i].hi`.
/// - `SingularMatrix` if the GP covariance matrix is numerically singular.
pub fn bayesopt(
    config: &BayesOptConfig,
    objective: &dyn Fn(&[f64]) -> f64,
) -> BayesResult<BayesOptResult> {
    validate_bo_config(config)?;

    let dim = config.dim;
    let n_total = config.n_init + config.n_iter;
    let mut rng = LcgRng::new(config.seed);

    let mut all_x: Vec<f64> = Vec::with_capacity(n_total * dim);
    let mut all_y: Vec<f64> = Vec::with_capacity(n_total);

    // ── Phase 1: random initial evaluations ──────────────────────────────────
    for _ in 0..config.n_init {
        let x = random_in_bounds(&config.bounds, &mut rng);
        let y = objective(&x);
        all_x.extend_from_slice(&x);
        all_y.push(y);
    }

    // ── Phase 2: Bayesian optimization iterations ─────────────────────────────
    let gpr_config = bo_gpr_config(config.noise_variance);

    for _ in 0..config.n_iter {
        let n_obs = all_y.len();
        let f_best = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Fit GP to all observations so far
        let fit = gpr_fit(&all_x, &all_y, n_obs, dim, &gpr_config)?;

        // Sample candidates and evaluate acquisition
        let mut best_acq = f64::NEG_INFINITY;
        let mut best_x_cand = random_in_bounds(&config.bounds, &mut rng);

        for _ in 0..config.n_candidates {
            let x_cand = random_in_bounds(&config.bounds, &mut rng);
            let (means, stds) = gpr_predict(&fit, &x_cand, 1, true)?;
            let mu = means[0];
            let sigma = stds.as_ref().map_or(0.0, |s| s[0]);
            let acq = acquisition_value(
                config.acquisition,
                mu,
                sigma,
                f_best,
                config.ucb_kappa,
                config.ei_xi,
            );
            if acq > best_acq {
                best_acq = acq;
                best_x_cand = x_cand;
            }
        }

        let y_next = objective(&best_x_cand);
        all_x.extend_from_slice(&best_x_cand);
        all_y.push(y_next);
    }

    // ── Find best ─────────────────────────────────────────────────────────────
    let best_idx = all_y
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_y = all_y[best_idx];
    let best_x = all_x[best_idx * dim..(best_idx + 1) * dim].to_vec();
    let n_evaluations = all_y.len();

    Ok(BayesOptResult {
        best_x,
        best_y,
        all_x,
        all_y,
        n_evaluations,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn config_1d_ei() -> BayesOptConfig {
        BayesOptConfig {
            dim: 1,
            bounds: vec![(0.0, 1.0)],
            n_init: 5,
            n_iter: 20,
            n_candidates: 500,
            acquisition: AcquisitionFn::ExpectedImprovement,
            ucb_kappa: 2.0,
            ei_xi: 0.01,
            noise_variance: 1e-4,
            seed: 42,
        }
    }

    // ── Test 1: 1D quadratic EI convergence ──────────────────────────────────
    #[test]
    fn bo_1d_quadratic_ei_converges() {
        let config = config_1d_ei();
        // Maximize -(x - 0.3)^2
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.3).powi(2)).expect("value should be present");
        assert!(
            (result.best_x[0] - 0.3).abs() < 0.20,
            "best_x={:.4} should be within 0.20 of 0.3",
            result.best_x[0]
        );
    }

    // ── Test 2: 2D quadratic EI convergence ──────────────────────────────────
    #[test]
    fn bo_2d_quadratic_ei_converges() {
        let config = BayesOptConfig {
            dim: 2,
            bounds: vec![(0.0, 1.0), (0.0, 1.0)],
            n_init: 5,
            n_iter: 30,
            n_candidates: 500,
            acquisition: AcquisitionFn::ExpectedImprovement,
            seed: 7,
            ..BayesOptConfig::default()
        };
        let result = bayesopt(&config, &|x: &[f64]| {
            -(x[0] - 0.5).powi(2) - (x[1] - 0.5).powi(2)
        })
        .expect("value should be present");
        let dist = ((result.best_x[0] - 0.5).powi(2) + (result.best_x[1] - 0.5).powi(2)).sqrt();
        assert!(
            dist < 0.25,
            "distance to optimum={dist:.4} should be < 0.25"
        );
    }

    // ── Test 3: all queried x within bounds ───────────────────────────────────
    #[test]
    fn bo_all_x_within_bounds() {
        let config = config_1d_ei();
        let bounds = config.bounds.clone();
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.5).powi(2)).expect("value should be present");
        for i in 0..result.n_evaluations {
            let row = &result.all_x[i * config.dim..(i + 1) * config.dim];
            for (j, (&v, &(lo, hi))) in row.iter().zip(bounds.iter()).enumerate() {
                assert!(
                    v >= lo && v <= hi,
                    "x[{i}][{j}]={v} out of bounds [{lo},{hi}]"
                );
            }
        }
    }

    // ── Test 4: n_evaluations == n_init + n_iter ──────────────────────────────
    #[test]
    fn bo_n_evaluations_correct() {
        let config = config_1d_ei();
        let result = bayesopt(&config, &|x: &[f64]| x[0]).expect("bayesopt should succeed");
        assert_eq!(
            result.n_evaluations,
            config.n_init + config.n_iter,
            "n_evaluations={} expected={}",
            result.n_evaluations,
            config.n_init + config.n_iter
        );
    }

    // ── Test 5: best_y == max(all_y) ─────────────────────────────────────────
    #[test]
    fn bo_best_y_is_max() {
        let config = config_1d_ei();
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.5).powi(2)).expect("value should be present");
        let max_y = result
            .all_y
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_y - max_y).abs() < 1e-10,
            "best_y={} max_y={}",
            result.best_y,
            max_y
        );
    }

    // ── Test 6: best_x.len() == dim ──────────────────────────────────────────
    #[test]
    fn bo_best_x_len_is_dim() {
        let config = config_1d_ei();
        let result = bayesopt(&config, &|x: &[f64]| x[0]).expect("bayesopt should succeed");
        assert_eq!(result.best_x.len(), config.dim);
    }

    // ── Test 7: all_x.len() == n_evaluations * dim ───────────────────────────
    #[test]
    fn bo_all_x_len_correct() {
        let config = config_1d_ei();
        let result = bayesopt(&config, &|x: &[f64]| x[0]).expect("bayesopt should succeed");
        assert_eq!(result.all_x.len(), result.n_evaluations * config.dim);
    }

    // ── Test 8: all_y.len() == n_evaluations ─────────────────────────────────
    #[test]
    fn bo_all_y_len_correct() {
        let config = config_1d_ei();
        let result = bayesopt(&config, &|x: &[f64]| x[0]).expect("bayesopt should succeed");
        assert_eq!(result.all_y.len(), result.n_evaluations);
    }

    // ── Test 9: EI > 0 when sigma > 0 and mu > f_best ────────────────────────
    #[test]
    fn acquisition_ei_positive_when_improvable() {
        let val = acquisition_value(AcquisitionFn::ExpectedImprovement, 1.0, 0.5, 0.5, 2.0, 0.01);
        assert!(val > 0.0, "EI should be > 0 when mu > f_best, got {val}");
    }

    // ── Test 10: UCB monotone in sigma ───────────────────────────────────────
    #[test]
    fn acquisition_ucb_monotone_in_sigma() {
        let ucb_small = acquisition_value(
            AcquisitionFn::UpperConfidenceBound,
            0.0,
            0.1,
            0.0,
            2.0,
            0.01,
        );
        let ucb_large = acquisition_value(
            AcquisitionFn::UpperConfidenceBound,
            0.0,
            0.5,
            0.0,
            2.0,
            0.01,
        );
        assert!(
            ucb_large > ucb_small,
            "UCB(sigma=0.5)={ucb_large} should be > UCB(sigma=0.1)={ucb_small}"
        );
    }

    // ── Test 11: PI in [0, 1] ─────────────────────────────────────────────────
    #[test]
    fn acquisition_pi_in_unit_interval() {
        for &(mu, sigma) in &[(0.5, 0.3_f64), (0.0, 0.1), (1.0, 0.5), (-0.5, 0.2)] {
            let pi = acquisition_value(
                AcquisitionFn::ProbabilityOfImprovement,
                mu,
                sigma,
                0.0,
                2.0,
                0.01,
            );
            assert!(
                (0.0..=1.0).contains(&pi),
                "PI={pi} out of [0,1] for mu={mu} sigma={sigma}"
            );
        }
    }

    // ── Test 12: EI with sigma=0 → 0 ─────────────────────────────────────────
    #[test]
    fn acquisition_ei_sigma_zero() {
        let val = acquisition_value(AcquisitionFn::ExpectedImprovement, 1.0, 0.0, 0.5, 2.0, 0.01);
        assert_eq!(val, 0.0, "EI with sigma=0 should be 0, got {val}");
    }

    // ── Test 13: PI with sigma=0 → deterministic ─────────────────────────────
    #[test]
    fn acquisition_pi_sigma_zero() {
        // mu > f_best + xi → 1.0
        let v1 = acquisition_value(
            AcquisitionFn::ProbabilityOfImprovement,
            1.0,
            0.0,
            0.5,
            2.0,
            0.01,
        );
        assert_eq!(v1, 1.0, "PI sigma=0, mu=1>f_best+xi should be 1, got {v1}");

        // mu <= f_best + xi → 0.0
        let v2 = acquisition_value(
            AcquisitionFn::ProbabilityOfImprovement,
            0.0,
            0.0,
            0.5,
            2.0,
            0.01,
        );
        assert_eq!(v2, 0.0, "PI sigma=0, mu=0<=f_best+xi should be 0, got {v2}");
    }

    // ── Test 14: seed reproducibility ────────────────────────────────────────
    #[test]
    fn bo_seed_reproducible() {
        let cfg_a = config_1d_ei();
        let cfg_b = config_1d_ei();
        let r_a =
            bayesopt(&cfg_a, &|x: &[f64]| -(x[0] - 0.3).powi(2)).expect("value should be present");
        let r_b =
            bayesopt(&cfg_b, &|x: &[f64]| -(x[0] - 0.3).powi(2)).expect("value should be present");
        assert_eq!(r_a.best_x, r_b.best_x, "same seed should give same best_x");
    }

    // ── Test 15: all_y all finite ─────────────────────────────────────────────
    #[test]
    fn bo_all_y_finite() {
        let config = config_1d_ei();
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.5).powi(2)).expect("value should be present");
        for &y in &result.all_y {
            assert!(y.is_finite(), "non-finite y={y}");
        }
    }

    // ── Test 16: n_iter=0 yields only n_init evaluations ─────────────────────
    #[test]
    fn bo_n_iter_zero() {
        let config = BayesOptConfig {
            n_iter: 0,
            ..config_1d_ei()
        };
        let result = bayesopt(&config, &|x: &[f64]| x[0]).expect("bayesopt should succeed");
        assert_eq!(result.n_evaluations, config.n_init);
    }

    // ── Test 17: n_init=1 works ───────────────────────────────────────────────
    #[test]
    fn bo_n_init_one() {
        let config = BayesOptConfig {
            n_init: 1,
            n_iter: 3,
            n_candidates: 10,
            ..config_1d_ei()
        };
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.5).powi(2)).expect("value should be present");
        assert_eq!(result.n_evaluations, 4);
    }

    // ── Test 18: UCB convergence on 1D quadratic ──────────────────────────────
    #[test]
    fn bo_1d_quadratic_ucb_converges() {
        let config = BayesOptConfig {
            acquisition: AcquisitionFn::UpperConfidenceBound,
            ucb_kappa: 2.0,
            n_iter: 25,
            seed: 99,
            ..config_1d_ei()
        };
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.3).powi(2)).expect("value should be present");
        assert!(
            (result.best_x[0] - 0.3).abs() < 0.30,
            "UCB best_x={:.4} should be within 0.3 of 0.3",
            result.best_x[0]
        );
    }

    // ── Test 19: PI convergence on 1D quadratic ───────────────────────────────
    #[test]
    fn bo_1d_quadratic_pi_converges() {
        let config = BayesOptConfig {
            acquisition: AcquisitionFn::ProbabilityOfImprovement,
            n_iter: 25,
            seed: 77,
            ..config_1d_ei()
        };
        let result =
            bayesopt(&config, &|x: &[f64]| -(x[0] - 0.3).powi(2)).expect("value should be present");
        assert!(
            (result.best_x[0] - 0.3).abs() < 0.30,
            "PI best_x={:.4} should be within 0.3 of 0.3",
            result.best_x[0]
        );
    }

    // ── Test 20: dim=0 → InvalidConfig ───────────────────────────────────────
    #[test]
    fn bo_dim_zero_error() {
        let config = BayesOptConfig {
            dim: 0,
            bounds: vec![],
            ..config_1d_ei()
        };
        let result = bayesopt(&config, &|_: &[f64]| 0.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    // ── Test 21: n_init=0 → InvalidConfig ────────────────────────────────────
    #[test]
    fn bo_n_init_zero_error() {
        let config = BayesOptConfig {
            n_init: 0,
            ..config_1d_ei()
        };
        let result = bayesopt(&config, &|_: &[f64]| 0.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }

    // ── Test 22: bounds with lo >= hi → InvalidConfig ─────────────────────────
    #[test]
    fn bo_bounds_invalid_error() {
        let config = BayesOptConfig {
            bounds: vec![(1.0, 0.0)], // lo > hi
            ..config_1d_ei()
        };
        let result = bayesopt(&config, &|_: &[f64]| 0.0);
        assert!(
            matches!(result, Err(BayesError::InvalidConfig(_))),
            "expected InvalidConfig, got {result:?}"
        );
    }
}
