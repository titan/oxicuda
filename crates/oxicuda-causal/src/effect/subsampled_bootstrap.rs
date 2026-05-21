//! Subsampled bootstrap for honest confidence intervals.
//!
//! Implements the subsampling method of Politis, Romano & Wolf (1999) as used
//! in Generalized Random Forests (Athey, Tibshirani & Wager 2019). Unlike the
//! bootstrap-with-replacement, subsampling draws B subsets of size m < n
//! WITHOUT replacement and scales the variance by m/n to obtain an honest
//! estimate of the full-sample standard error. This avoids the bias that
//! standard bootstrap-with-replacement introduces in semiparametric methods.

use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

/// Configuration for subsampled bootstrap.
#[derive(Debug, Clone, Copy)]
pub struct SubsampledBootstrapConfig {
    /// Number of subsamples (B_sub, e.g. 200 or 500).
    pub n_bootstrap: usize,
    /// Subsample size as fraction of n: m = max(1, floor(fraction * n)).
    /// Typical value: 0.5 (i.e., m ≈ n/2).
    pub subsample_fraction: f32,
    /// Significance level α ∈ (0, 0.5) for CI: e.g. 0.05 → 95% CI.
    pub alpha: f32,
}

impl Default for SubsampledBootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 200,
            subsample_fraction: 0.5,
            alpha: 0.05,
        }
    }
}

/// Result of subsampled bootstrap.
#[derive(Debug, Clone)]
pub struct SubsampledBootstrapResult {
    /// Full-sample estimate θ̂.
    pub estimate: f32,
    /// Subsampling standard error: sqrt((m/n) * Var_B(θ̂_b)).
    pub se: f32,
    /// Lower bound of (1-α) CI: estimate - z_{1-α/2} * se.
    pub ci_lower: f32,
    /// Upper bound of (1-α) CI: estimate + z_{1-α/2} * se.
    pub ci_upper: f32,
    /// Number of subsamples that succeeded.
    pub n_valid: usize,
}

/// Abramowitz & Stegun approximation of Φ^{-1}(q) for q ∈ (0, 1).
///
/// Uses the rational approximation from A&S §26.2.17 (formula with ≤4.9% error)
/// to compute the inverse normal CDF (probit function). For q < 0.5 uses
/// symmetry: Φ^{-1}(q) = -Φ^{-1}(1-q). For q == 0.5 returns 0.0 directly.
fn probit(q: f32) -> f32 {
    // Handle the symmetric half: map q < 0.5 to the upper half via negation.
    // We test strictly < 0.5 to avoid infinite recursion at the boundary.
    if q < 0.5 {
        return -probit_upper(1.0 - q);
    }
    probit_upper(q)
}

/// Core A&S rational approximation for q ∈ [0.5, 1).
fn probit_upper(q: f32) -> f32 {
    // At q = 0.5, t = 0 and the approximation evaluates to 0 - c0/(1+0) ≠ 0.
    // The symmetry fix: for q very close to 0.5 the result should be ≈ 0.
    // Protect against ln(0) when q ≥ 1.
    if q >= 1.0 {
        return f32::INFINITY;
    }
    let p = 1.0 - q;
    if p <= 0.0 {
        return f32::INFINITY;
    }
    // t = sqrt(-2 ln(1-q))
    let t = (-2.0_f32 * p.ln()).sqrt();
    // Rational approximation coefficients (A&S 26.2.17)
    let c0 = 2.515_517_f32;
    let c1 = 0.802_853_f32;
    let c2 = 0.010_328_f32;
    let d1 = 1.432_788_f32;
    let d2 = 0.189_269_f32;
    let d3 = 0.001_308_f32;
    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    t - numerator / denominator
}

/// Validate the SubsampledBootstrapConfig fields.
fn validate_config(cfg: &SubsampledBootstrapConfig) -> CausalResult<()> {
    if cfg.n_bootstrap == 0 {
        return Err(CausalError::InvalidParameter {
            reason: "n_bootstrap must be ≥ 1".to_string(),
        });
    }
    if cfg.subsample_fraction <= 0.0 || cfg.subsample_fraction >= 1.0 {
        return Err(CausalError::InvalidParameter {
            reason: format!(
                "subsample_fraction must be in (0, 1), got {}",
                cfg.subsample_fraction
            ),
        });
    }
    if cfg.alpha <= 0.0 || cfg.alpha >= 0.5 {
        return Err(CausalError::InvalidParameter {
            reason: format!("alpha must be in (0, 0.5), got {}", cfg.alpha),
        });
    }
    Ok(())
}

/// Perform partial Fisher-Yates shuffle to select m indices from 0..n without
/// replacement. After the call, `indices[0..m]` holds the selected indices.
fn partial_fisher_yates(indices: &mut [usize], m: usize, rng: &mut LcgRng) {
    let n = indices.len();
    for i in 0..m {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
}

/// Compute the variance (population, ddof=0) of a slice of f32.
fn population_variance(values: &[f32]) -> f32 {
    let n = values.len();
    if n <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / n as f32;
    values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32
}

/// Run subsampled bootstrap for a scalar estimator.
///
/// # Arguments
/// * `data`      — flat array of n observations (each observation is a scalar).
/// * `estimator` — function mapping a subsample slice to a scalar estimate.
/// * `cfg`       — configuration.
/// * `rng`       — random number generator for subsample selection.
///
/// # Errors
/// Returns `CausalError::EmptyInput` if data is empty.
/// Returns `CausalError::InvalidParameter` for invalid config values.
/// Returns `CausalError::Internal` if all subsamples failed.
pub fn subsampled_bootstrap<F>(
    data: &[f32],
    estimator: F,
    cfg: &SubsampledBootstrapConfig,
    rng: &mut LcgRng,
) -> CausalResult<SubsampledBootstrapResult>
where
    F: Fn(&[f32]) -> CausalResult<f32>,
{
    if data.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    validate_config(cfg)?;

    let n = data.len();
    let m = ((cfg.subsample_fraction * n as f32).floor() as usize).max(1);

    // Full-sample estimate
    let theta_hat = estimator(data)?;

    // Draw B subsamples of size m WITHOUT replacement
    let mut indices: Vec<usize> = (0..n).collect();
    let mut subsample_buf = vec![0.0_f32; m];
    let mut theta_bs: Vec<f32> = Vec::with_capacity(cfg.n_bootstrap);

    for _ in 0..cfg.n_bootstrap {
        // Reset indices each time to allow re-selection across bootstrap rounds
        for (idx, val) in indices.iter_mut().enumerate() {
            *val = idx;
        }
        partial_fisher_yates(&mut indices, m, rng);
        for (j, &idx) in indices[..m].iter().enumerate() {
            subsample_buf[j] = data[idx];
        }
        if let Ok(theta_b) = estimator(&subsample_buf) {
            theta_bs.push(theta_b);
        }
    }

    let n_valid = theta_bs.len();
    if n_valid == 0 {
        return Err(CausalError::Internal {
            msg: "all subsamples failed to produce an estimate".to_string(),
        });
    }

    // Scale variance by m/n — the key subsampling correction
    let var_b = population_variance(&theta_bs);
    let se = ((m as f32 / n as f32) * var_b).sqrt();

    let z = probit(1.0 - cfg.alpha / 2.0);
    let ci_lower = theta_hat - z * se;
    let ci_upper = theta_hat + z * se;

    Ok(SubsampledBootstrapResult {
        estimate: theta_hat,
        se,
        ci_lower,
        ci_upper,
        n_valid,
    })
}

/// Run subsampled bootstrap for a multivariate estimator (returns `Vec<f32>`).
///
/// Returns one `SubsampledBootstrapResult` per output coordinate.
///
/// # Arguments
/// * `data`      — n rows of d-dimensional data, row-major: length n*d.
/// * `n`         — number of observations.
/// * `d`         — dimension of each observation.
/// * `estimator` — function mapping (subsample_rows_flat, n_sub) → `Vec<f32>` of length d.
/// * `cfg`       — configuration.
/// * `rng`       — random number generator.
///
/// # Errors
/// Returns `CausalError::EmptyInput` if data is empty or n/d are zero.
/// Returns `CausalError::IncompatibleData` if data.len() ≠ n*d.
/// Returns `CausalError::InvalidParameter` for invalid config values.
/// Returns `CausalError::Internal` if all subsamples failed or dimension mismatch.
pub fn subsampled_bootstrap_vec<F>(
    data: &[f32],
    n: usize,
    d: usize,
    estimator: F,
    cfg: &SubsampledBootstrapConfig,
    rng: &mut LcgRng,
) -> CausalResult<Vec<SubsampledBootstrapResult>>
where
    F: Fn(&[f32], usize) -> CausalResult<Vec<f32>>,
{
    if n == 0 || d == 0 || data.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    if data.len() != n * d {
        return Err(CausalError::IncompatibleData);
    }
    validate_config(cfg)?;

    let m = ((cfg.subsample_fraction * n as f32).floor() as usize).max(1);

    // Full-sample estimate
    let theta_hat = estimator(data, n)?;
    if theta_hat.len() != d {
        return Err(CausalError::Internal {
            msg: format!(
                "estimator returned {} values, expected d={}",
                theta_hat.len(),
                d
            ),
        });
    }

    // Draw B subsamples
    let mut indices: Vec<usize> = (0..n).collect();
    let mut sub_data = vec![0.0_f32; m * d];
    // Per-coordinate subsample estimates
    let mut coord_estimates: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_bootstrap); d];
    let mut n_valid = 0_usize;

    for _ in 0..cfg.n_bootstrap {
        // Reset indices
        for (idx, val) in indices.iter_mut().enumerate() {
            *val = idx;
        }
        partial_fisher_yates(&mut indices, m, rng);
        // Copy selected rows into sub_data
        for (j, &row_idx) in indices[..m].iter().enumerate() {
            let src_start = row_idx * d;
            let dst_start = j * d;
            sub_data[dst_start..dst_start + d].copy_from_slice(&data[src_start..src_start + d]);
        }
        match estimator(&sub_data, m) {
            Ok(theta_b) if theta_b.len() == d => {
                n_valid += 1;
                for (k, &val) in theta_b.iter().enumerate() {
                    coord_estimates[k].push(val);
                }
            }
            _ => {
                // Count as failed; skip
            }
        }
    }

    if n_valid == 0 {
        return Err(CausalError::Internal {
            msg: "all subsamples failed to produce an estimate".to_string(),
        });
    }

    let z = probit(1.0 - cfg.alpha / 2.0);
    let scale = m as f32 / n as f32;

    let results = (0..d)
        .map(|k| {
            let var_b = population_variance(&coord_estimates[k]);
            let se = (scale * var_b).sqrt();
            SubsampledBootstrapResult {
                estimate: theta_hat[k],
                se,
                ci_lower: theta_hat[k] - z * se,
                ci_upper: theta_hat[k] + z * se,
                n_valid,
            }
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_estimator(data: &[f32]) -> CausalResult<f32> {
        if data.is_empty() {
            return Err(CausalError::EmptyInput);
        }
        Ok(data.iter().sum::<f32>() / data.len() as f32)
    }

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    #[test]
    fn mean_estimator_basic() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(42);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        // Full-sample mean = 9.5
        assert!(
            (result.estimate - 9.5).abs() < 1e-4,
            "estimate={}",
            result.estimate
        );
    }

    #[test]
    fn ci_contains_true_mean() {
        // Data: 0..100, true mean = 49.5
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 300,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(7);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        let sample_mean = 49.5_f32;
        assert!(
            result.ci_lower <= sample_mean && sample_mean <= result.ci_upper,
            "CI=[{}, {}] does not contain {}",
            result.ci_lower,
            result.ci_upper,
            sample_mean
        );
    }

    #[test]
    fn se_is_nonneg() {
        let data: Vec<f32> = (0..30).map(|i| i as f32 * 0.1).collect();
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(13);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        assert!(result.se >= 0.0, "se={}", result.se);
    }

    #[test]
    fn ci_lower_leq_upper() {
        let data: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.5).collect();
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(99);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        assert!(
            result.ci_lower <= result.estimate,
            "ci_lower={} > estimate={}",
            result.ci_lower,
            result.estimate
        );
        assert!(
            result.estimate <= result.ci_upper,
            "estimate={} > ci_upper={}",
            result.estimate,
            result.ci_upper
        );
    }

    #[test]
    fn n_valid_leq_n_bootstrap() {
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 100,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(5);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        assert!(result.n_valid <= cfg.n_bootstrap);
    }

    #[test]
    fn both_fractions_give_valid_cis() {
        // Test that both fraction=0.3 and fraction=0.8 produce valid (lower ≤ upper) CIs
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        for &frac in &[0.3_f32, 0.8_f32] {
            let cfg = SubsampledBootstrapConfig {
                n_bootstrap: 100,
                subsample_fraction: frac,
                alpha: 0.05,
            };
            let mut rng = make_rng(17);
            let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
            assert!(
                result.ci_lower <= result.ci_upper,
                "fraction={}: ci_lower={} > ci_upper={}",
                frac,
                result.ci_lower,
                result.ci_upper
            );
        }
    }

    #[test]
    fn n_bootstrap_1_works() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 1,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(3);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        // With 1 subsample, Var = 0, so se = 0, CI is degenerate
        assert_eq!(result.se, 0.0);
        assert_eq!(result.n_valid, 1);
    }

    #[test]
    fn n_bootstrap_200_no_error() {
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 200,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(21);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn err_empty_data() {
        let data: Vec<f32> = vec![];
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(1);
        assert!(matches!(
            subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng),
            Err(CausalError::EmptyInput)
        ));
    }

    #[test]
    fn err_n_bootstrap_zero() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 0,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(1);
        assert!(matches!(
            subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_alpha_out_of_range() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut rng = make_rng(1);

        let cfg_zero = SubsampledBootstrapConfig {
            n_bootstrap: 10,
            subsample_fraction: 0.5,
            alpha: 0.0,
        };
        assert!(matches!(
            subsampled_bootstrap(&data, mean_estimator, &cfg_zero, &mut rng),
            Err(CausalError::InvalidParameter { .. })
        ));

        let cfg_large = SubsampledBootstrapConfig {
            n_bootstrap: 10,
            subsample_fraction: 0.5,
            alpha: 0.6,
        };
        assert!(matches!(
            subsampled_bootstrap(&data, mean_estimator, &cfg_large, &mut rng),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_fraction_out_of_range() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut rng = make_rng(1);

        let cfg_zero = SubsampledBootstrapConfig {
            n_bootstrap: 10,
            subsample_fraction: 0.0,
            alpha: 0.05,
        };
        assert!(matches!(
            subsampled_bootstrap(&data, mean_estimator, &cfg_zero, &mut rng),
            Err(CausalError::InvalidParameter { .. })
        ));

        let cfg_one = SubsampledBootstrapConfig {
            n_bootstrap: 10,
            subsample_fraction: 1.0,
            alpha: 0.05,
        };
        assert!(matches!(
            subsampled_bootstrap(&data, mean_estimator, &cfg_one, &mut rng),
            Err(CausalError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn default_config_valid() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(55);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng);
        assert!(result.is_ok(), "default config should be valid");
    }

    #[test]
    fn probit_known_values() {
        // Φ^{-1}(0.975) ≈ 1.96
        let z = probit(0.975);
        assert!(
            (z - 1.959_964).abs() < 0.01,
            "probit(0.975) = {}, expected ≈ 1.96",
            z
        );
        // Φ^{-1}(0.5) = 0
        let z_half = probit(0.5);
        assert!(z_half.abs() < 0.01, "probit(0.5) = {}, expected 0", z_half);
    }

    #[test]
    fn vec_version_shape() {
        // d=3, n=30: subsampled_bootstrap_vec returns Vec of length 3
        let n = 30;
        let d = 3;
        let data: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let estimator = |sub: &[f32], n_sub: usize| -> CausalResult<Vec<f32>> {
            if n_sub == 0 {
                return Err(CausalError::EmptyInput);
            }
            // Compute column means
            let mut means = vec![0.0_f32; d];
            for row in 0..n_sub {
                for col in 0..d {
                    means[col] += sub[row * d + col];
                }
            }
            for v in means.iter_mut() {
                *v /= n_sub as f32;
            }
            Ok(means)
        };
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 50,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(77);
        let results = subsampled_bootstrap_vec(&data, n, d, estimator, &cfg, &mut rng).unwrap();
        assert_eq!(
            results.len(),
            d,
            "expected {} results, got {}",
            d,
            results.len()
        );
    }

    #[test]
    fn vec_version_ci_lower_leq_upper() {
        let n = 40;
        let d = 2;
        let data: Vec<f32> = (0..n * d).map(|i| i as f32 * 0.1).collect();
        let estimator = |sub: &[f32], n_sub: usize| -> CausalResult<Vec<f32>> {
            if n_sub == 0 {
                return Err(CausalError::EmptyInput);
            }
            let mut means = vec![0.0_f32; d];
            for row in 0..n_sub {
                for col in 0..d {
                    means[col] += sub[row * d + col];
                }
            }
            for v in means.iter_mut() {
                *v /= n_sub as f32;
            }
            Ok(means)
        };
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(88);
        let results = subsampled_bootstrap_vec(&data, n, d, estimator, &cfg, &mut rng).unwrap();
        for (k, r) in results.iter().enumerate() {
            assert!(
                r.ci_lower <= r.ci_upper,
                "coord {}: ci_lower={} > ci_upper={}",
                k,
                r.ci_lower,
                r.ci_upper
            );
        }
    }

    #[test]
    fn constant_data_zero_se() {
        // All data equal → variance of subsample estimates ≈ 0 → se ≈ 0.
        // f32 arithmetic on identical values can still produce tiny floating-point
        // residuals, so we check that se is very small rather than exactly zero.
        let const_val = 7.77_f32;
        let data = vec![const_val; 50];
        let cfg = SubsampledBootstrapConfig::default();
        let mut rng = make_rng(11);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng).unwrap();
        assert!(
            result.se < 1e-4,
            "se={} expected very small for constant data",
            result.se
        );
        assert!((result.estimate - const_val).abs() < 1e-5);
    }

    #[test]
    fn small_n() {
        // n=5, very small dataset: m ≥ 1, should not panic
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let cfg = SubsampledBootstrapConfig {
            n_bootstrap: 20,
            subsample_fraction: 0.5,
            alpha: 0.05,
        };
        let mut rng = make_rng(33);
        let result = subsampled_bootstrap(&data, mean_estimator, &cfg, &mut rng);
        assert!(result.is_ok(), "n=5 should not fail: {:?}", result);
        let r = result.unwrap();
        assert!(r.n_valid > 0);
        assert!(r.se.is_finite());
    }
}
