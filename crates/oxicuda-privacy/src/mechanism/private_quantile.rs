//! Private quantile / median via the exponential mechanism.
//!
//! Reference: Smith (2011), "Privacy-preserving statistical estimation with
//! optimal convergence rates", STOC.  See also the "Joint Exponential
//! Mechanism" line of work for which this is the single-quantile base case.
//!
//! # Construction
//! Given data `x` within bounds `[lo, hi]` and a target quantile `q ∈ (0, 1)`
//! (`q = 0.5` ⇒ median), we clamp the data to `[lo, hi]`, sort it, and prepend
//! `lo` / append `hi`, yielding sentinel points `y₀ ≤ … ≤ yₙ` with `n` data
//! points in between.  These define `n` intervals; interval `i` spans
//! `[yᵢ, yᵢ₊₁]` with width `wᵢ = yᵢ₊₁ − yᵢ ≥ 0` and has exactly `i` data points
//! to its left.
//!
//! # Utility
//! Let the target rank be `m = q · n` where `n` is the number of data points.
//! Interval `i` has rank utility `uᵢ = −|i − m|`.  Adding or removing one data
//! point shifts every rank by at most `1`, so the rank-utility sensitivity is
//! `Δu = 1`.
//!
//! # Exponential mechanism
//! Interval `i` is selected with probability
//!
//! `P(i) ∝ wᵢ · exp(ε · uᵢ / (2 · Δu))`.
//!
//! The width factor `wᵢ` makes this a proper density over the continuum;
//! zero-width intervals receive zero probability.  After choosing an interval
//! we sample uniformly within `[yᵢ, yᵢ₊₁]`.  A numerically stable shifted
//! exponent (subtract `max uᵢ`) is used.  If the data are all equal (so every
//! interior interval has zero width and total weight is zero), the midpoint of
//! `[lo, hi]` is returned.
//!
//! As `ε` grows the selected value concentrates near the true `q`-quantile.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the private quantile mechanism.
#[derive(Debug, Clone)]
pub struct QuantileConfig {
    /// Privacy parameter `ε > 0`.
    pub epsilon: f64,
    /// Lower bound of the data domain (`lo < hi`).
    pub lo: f64,
    /// Upper bound of the data domain (`lo < hi`).
    pub hi: f64,
    /// Target quantile `q ∈ (0, 1)`.
    pub q: f64,
}

impl QuantileConfig {
    /// Construct and validate a `QuantileConfig`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `epsilon ≤ 0`, `lo ≥ hi`, or
    /// `q ∉ (0, 1)`.
    pub fn new(epsilon: f64, lo: f64, hi: f64, q: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 || epsilon.is_nan() {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon must be positive, got {epsilon}"
            )));
        }
        if lo >= hi || lo.is_nan() || hi.is_nan() {
            return Err(PrivacyError::InvalidParameter(format!(
                "lo must be < hi, got lo={lo}, hi={hi}"
            )));
        }
        if q <= 0.0 || q >= 1.0 || q.is_nan() {
            return Err(PrivacyError::InvalidParameter(format!(
                "q must be in (0,1), got {q}"
            )));
        }
        Ok(Self { epsilon, lo, hi, q })
    }
}

/// Select a private `q`-quantile of `data` via the exponential mechanism.
///
/// The returned value lies in `[lo, hi]`.  Data outside `[lo, hi]` is clamped
/// before processing.  As `ε` grows the result concentrates near the true
/// quantile.
///
/// # Errors
/// - `EmptyInput` if `data` is empty.
pub fn private_quantile(
    data: &[f64],
    cfg: &QuantileConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<f64> {
    if data.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }

    // Clamp data to [lo, hi] and sort.
    let mut sorted: Vec<f64> = data.iter().map(|&v| v.clamp(cfg.lo, cfg.hi)).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();

    // Sentinel-augmented points: y₀ = lo, y₁..yₙ = sorted, yₙ₊₁ = hi.
    // This yields n+1 intervals (indices 0..=n); interval i has i points left.
    // Build the boundary vector y of length n+2.
    let mut y = Vec::with_capacity(n + 2);
    y.push(cfg.lo);
    y.extend_from_slice(&sorted);
    y.push(cfg.hi);

    let n_f = n as f64;
    let target_rank = cfg.q * n_f;
    // Sensitivity of the rank-utility is 1; exponential mechanism scale.
    let scale = cfg.epsilon / (2.0 * 1.0);

    let num_intervals = n + 1;

    // Utilities uᵢ = −|i − m| for i in 0..=n.
    // Shift by the maximum utility (closest interval to target rank) for
    // numerical stability; max utility ≤ 0 always.
    let mut max_u = f64::NEG_INFINITY;
    for i in 0..num_intervals {
        let u = -((i as f64) - target_rank).abs();
        if u > max_u {
            max_u = u;
        }
    }

    // Weights wᵢ · exp(scale · (uᵢ − max_u)).
    let mut weights = Vec::with_capacity(num_intervals);
    let mut total = 0.0f64;
    for i in 0..num_intervals {
        let width = (y[i + 1] - y[i]).max(0.0);
        let u = -((i as f64) - target_rank).abs();
        let w = width * (scale * (u - max_u)).exp();
        weights.push(w);
        total += w;
    }

    // All-equal data (or degenerate widths) ⇒ total weight 0: return midpoint.
    if total <= 0.0 || total.is_nan() {
        return Ok(0.5 * (cfg.lo + cfg.hi));
    }

    // Sample an interval proportional to its weight.
    let u_draw = rng.next_f64() * total;
    let mut cumsum = 0.0f64;
    let mut chosen = num_intervals - 1;
    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if cumsum >= u_draw {
            chosen = i;
            break;
        }
    }

    // Sample uniformly within the chosen interval [y[chosen], y[chosen+1]].
    let left = y[chosen];
    let right = y[chosen + 1];
    let value = left + rng.next_f64() * (right - left);
    Ok(value.clamp(cfg.lo, cfg.hi))
}

/// Convenience wrapper for the private median (`q = 0.5`).
///
/// # Errors
/// - `EmptyInput` if `data` is empty.
/// - `InvalidParameter` if `epsilon ≤ 0` or `lo ≥ hi`.
pub fn private_median(
    data: &[f64],
    epsilon: f64,
    lo: f64,
    hi: f64,
    rng: &mut LcgRng,
) -> PrivacyResult<f64> {
    let cfg = QuantileConfig::new(epsilon, lo, hi, 0.5)?;
    private_quantile(data, &cfg, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        assert!(QuantileConfig::new(0.0, 0.0, 1.0, 0.5).is_err());
        assert!(QuantileConfig::new(-1.0, 0.0, 1.0, 0.5).is_err());
        assert!(QuantileConfig::new(1.0, 1.0, 1.0, 0.5).is_err());
        assert!(QuantileConfig::new(1.0, 2.0, 1.0, 0.5).is_err());
        assert!(QuantileConfig::new(1.0, 0.0, 1.0, 0.0).is_err());
        assert!(QuantileConfig::new(1.0, 0.0, 1.0, 1.0).is_err());
        assert!(QuantileConfig::new(1.0, 0.0, 1.0, 1.5).is_err());
        assert!(QuantileConfig::new(1.0, 0.0, 1.0, 0.5).is_ok());
    }

    #[test]
    fn test_returns_in_range() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let cfg = QuantileConfig::new(1.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(42);
        for _ in 0..500 {
            let v = private_quantile(&data, &cfg, &mut rng).expect("ok");
            assert!((0.0..=100.0).contains(&v), "out of range: {v}");
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn test_median_centered_dataset() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let cfg = QuantileConfig::new(2.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(7);
        let mut sum = 0.0;
        let trials = 200;
        for _ in 0..trials {
            sum += private_quantile(&data, &cfg, &mut rng).expect("ok");
        }
        let avg = sum / trials as f64;
        assert!((avg - 50.0).abs() < 10.0, "avg median {avg} not near 50");
    }

    #[test]
    fn test_high_epsilon_close_to_true() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let cfg = QuantileConfig::new(50.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(13);
        let mut sum = 0.0;
        let trials = 200;
        for _ in 0..trials {
            sum += private_quantile(&data, &cfg, &mut rng).expect("ok");
        }
        let avg = sum / trials as f64;
        assert!(
            (avg - 50.0).abs() < 3.0,
            "avg median {avg} not within 3 of 50"
        );
    }

    #[test]
    fn test_quantile_25() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let cfg = QuantileConfig::new(5.0, 0.0, 100.0, 0.25).expect("ok");
        let mut rng = LcgRng::new(21);
        let mut sum = 0.0;
        let trials = 200;
        for _ in 0..trials {
            sum += private_quantile(&data, &cfg, &mut rng).expect("ok");
        }
        let avg = sum / trials as f64;
        assert!((avg - 25.0).abs() < 12.0, "avg q25 {avg} not near 25");
    }

    #[test]
    fn test_all_equal_data() {
        let data = vec![42.0f64; 50];
        let cfg = QuantileConfig::new(1.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(3);
        for _ in 0..100 {
            let v = private_quantile(&data, &cfg, &mut rng).expect("ok");
            assert!((0.0..=100.0).contains(&v), "out of range: {v}");
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn test_empty_data_error() {
        let cfg = QuantileConfig::new(1.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(0);
        assert!(private_quantile(&[], &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_clamping_out_of_range_data() {
        // Data extends well beyond [lo, hi]; result must still be in-range.
        let data: Vec<f64> = vec![-1000.0, -500.0, 50.0, 500.0, 1000.0];
        let cfg = QuantileConfig::new(2.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng = LcgRng::new(55);
        for _ in 0..200 {
            let v = private_quantile(&data, &cfg, &mut rng).expect("ok");
            assert!((0.0..=100.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn test_determinism_same_seed() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let cfg = QuantileConfig::new(2.0, 0.0, 100.0, 0.5).expect("ok");
        let mut rng_a = LcgRng::new(2024);
        let mut rng_b = LcgRng::new(2024);
        for _ in 0..100 {
            let a = private_quantile(&data, &cfg, &mut rng_a).expect("ok");
            let b = private_quantile(&data, &cfg, &mut rng_b).expect("ok");
            assert!((a - b).abs() < 1e-12, "non-deterministic: {a} vs {b}");
        }
    }

    #[test]
    fn test_monotone_quantiles() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let cfg_lo = QuantileConfig::new(5.0, 0.0, 100.0, 0.1).expect("ok");
        let cfg_hi = QuantileConfig::new(5.0, 0.0, 100.0, 0.9).expect("ok");
        let mut rng = LcgRng::new(99);
        let trials = 200;
        let mut sum_lo = 0.0;
        let mut sum_hi = 0.0;
        for _ in 0..trials {
            sum_lo += private_quantile(&data, &cfg_lo, &mut rng).expect("ok");
            sum_hi += private_quantile(&data, &cfg_hi, &mut rng).expect("ok");
        }
        let mean_lo = sum_lo / trials as f64;
        let mean_hi = sum_hi / trials as f64;
        assert!(
            mean_hi > mean_lo,
            "q=0.9 mean {mean_hi} should exceed q=0.1 mean {mean_lo}"
        );
    }

    #[test]
    fn test_private_median_wrapper() {
        let data: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let mut rng = LcgRng::new(8);
        let v = private_median(&data, 1.0, 0.0, 100.0, &mut rng).expect("ok");
        assert!((0.0..=100.0).contains(&v));
    }

    #[test]
    fn test_private_median_wrapper_validates() {
        let data = vec![1.0, 2.0, 3.0];
        let mut rng = LcgRng::new(8);
        // bad epsilon
        assert!(private_median(&data, 0.0, 0.0, 100.0, &mut rng).is_err());
        // bad bounds
        assert!(private_median(&data, 1.0, 100.0, 0.0, &mut rng).is_err());
    }
}
