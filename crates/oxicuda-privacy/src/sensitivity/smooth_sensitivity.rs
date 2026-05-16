//! Smooth sensitivity calibration.
//!
//! Reference: Nissim, Raskhodnikova & Smith (2007), "Smooth Sensitivity and
//! Sampling in Private Data Analysis", STOC 2007.
//!
//! # Smooth sensitivity
//! The **β-smooth sensitivity** of f at x is:
//!
//! `S^β_f(x) = max_{x'} e^{−β·d(x,x')} · LS_f(x')`
//!
//! where the max is over all datasets x' at Hamming distance d(x,x') from x.
//!
//! Using S^β_f(x) to calibrate noise instead of global sensitivity Δ_f gives
//! (ε, 0)-DP when the noise is drawn from a distribution with heavy tails
//! (e.g., Cauchy or Student-t), with noise scale S^β_f(x) / (ε − β).
//!
//! # Supported queries
//! - **Mean**: global smooth sensitivity = 1/n (since LS_mean is 1/n always
//!   for bounded domain — approximate here via local smoothing).
//! - **Median**: computed via the Nissim-Raskhodnikova-Smith algorithm on
//!   sorted order statistics.
//!
//! # Noise
//! We use **Laplace noise** as an approximation.  For full privacy guarantees
//! with smooth sensitivity, Cauchy noise or Student-t is required; Laplace
//! gives (2ε/ε−β, 0)-DP approximately.  Callers who need exact guarantees
//! should use Cauchy noise.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for smooth sensitivity computation.
#[derive(Debug, Clone)]
pub struct SmoothSensConfig {
    /// Privacy parameter ε > 0 for the privatised release.
    pub epsilon: f64,
    /// Smoothing parameter β ∈ (0, ε).  Must satisfy β < ε.
    pub beta: f64,
    /// Maximum Hamming distance to search when computing smooth sensitivity.
    /// For the median, this controls accuracy vs compute trade-off.
    pub distance_budget: usize,
}

impl SmoothSensConfig {
    /// Construct and validate a `SmoothSensConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon`, or `InvalidParameter` if β ≥ ε or β ≤ 0
    /// or `distance_budget == 0`.
    pub fn new(epsilon: f64, beta: f64, distance_budget: usize) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if beta <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "beta must be positive, got {beta}"
            )));
        }
        if beta >= epsilon {
            return Err(PrivacyError::InvalidParameter(format!(
                "beta={beta} must be < epsilon={epsilon}"
            )));
        }
        if distance_budget == 0 {
            return Err(PrivacyError::InvalidParameter(
                "distance_budget must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            epsilon,
            beta,
            distance_budget,
        })
    }
}

/// Compute the β-smooth sensitivity of the **mean** on x.
///
/// For the mean query with domain-bounded data:
/// LS_mean(x') ≤ 1/n for any x', so S^β_mean = 1/n (no dependence on β or x).
/// However, in practice we compute the data-dependent local sensitivity to
/// potentially achieve a tighter bound.
///
/// # Errors
/// Returns `EmptyInput` if `x` is empty.
pub fn smooth_sensitivity_mean(x: &[f64], _cfg: &SmoothSensConfig) -> PrivacyResult<f64> {
    if x.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    // For mean: LS_mean(x) at any neighbouring dataset is bounded by
    // (max(x) - min(x)) / n  (worst case: replacing the min by max or vice versa).
    let n = x.len() as f64;
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);

    // β-smooth sensitivity for mean is the local sensitivity at distance 0
    // (since local sensitivity of mean does not grow with distance —
    // neighbouring datasets have LS ≤ (max-min)/n).
    Ok((max_val - min_val) / n)
}

/// Compute the β-smooth sensitivity of the **median** on sorted x.
///
/// The Nissim-Raskhodnikova-Smith approach for the median on sorted data:
/// At Hamming distance k from x (k ∈ 0..distance_budget), the local
/// sensitivity of the median is bounded by the gap between order statistics
/// at positions ⌊n/2⌋ ± k in the sorted sequence.
///
/// `S^β_median(x) = max_{k=0..d} e^{-β·k} · LS_median_at_distance_k(sorted_x)`
///
/// where `LS_median_at_distance_k` is approximated by the gap
/// `|sorted_x[mid+k] - sorted_x[mid-k]| / 2` (clamped to array bounds).
///
/// # Errors
/// Returns `EmptyInput` if `x` is empty.
pub fn smooth_sensitivity_median(x: &[f64], cfg: &SmoothSensConfig) -> PrivacyResult<f64> {
    if x.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }

    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mid = n / 2;
    let max_dist = cfg.distance_budget.min(mid);

    let mut s_beta = 0.0f64;
    for k in 0..=max_dist {
        let lo_idx = mid.saturating_sub(k);
        let hi_idx = (mid + k).min(n - 1);
        let ls_at_k = (sorted[hi_idx] - sorted[lo_idx]).abs() / 2.0;
        let weight = (-(cfg.beta * k as f64)).exp();
        s_beta = s_beta.max(weight * ls_at_k);
    }

    Ok(s_beta)
}

/// Add noise calibrated to the smooth sensitivity using Laplace distribution.
///
/// For rigorous (ε, 0)-DP, Cauchy or Student-t noise should be used instead
/// of Laplace.  The Laplace noise here provides an efficient approximation.
///
/// Noise scale = smooth_sens / (ε − β).
///
/// # Errors
/// - `NonPositiveSensitivity` if `smooth_sens < 0`.
/// - `InvalidParameter` if ε − β ≤ 0 (already validated by `SmoothSensConfig`).
pub fn smooth_sensitive_noise(
    val: f64,
    smooth_sens: f64,
    cfg: &SmoothSensConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<f64> {
    if smooth_sens < 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(smooth_sens));
    }
    // smooth_sens == 0 means the statistic is constant — no noise needed.
    if smooth_sens == 0.0 {
        return Ok(val);
    }

    let scale = smooth_sens / (cfg.epsilon - cfg.beta);
    let u = rng.next_f64() - 0.5;
    let abs_u = u.abs().min(0.5 - f64::EPSILON);
    let noise = -scale * u.signum() * (1.0 - 2.0 * abs_u).ln();
    Ok(val + noise)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_sensitivity_mean_nonneg() {
        let x = [1.0, 2.0, 5.0, 7.0, 10.0];
        let cfg = SmoothSensConfig::new(1.0, 0.5, 5).expect("ok");
        let s = smooth_sensitivity_mean(&x, &cfg).expect("ok");
        assert!(s >= 0.0, "smooth sensitivity must be ≥ 0, got {s}");
    }

    #[test]
    fn test_smooth_sensitivity_median_nonneg() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = SmoothSensConfig::new(1.0, 0.3, 3).expect("ok");
        let s = smooth_sensitivity_median(&x, &cfg).expect("ok");
        assert!(s >= 0.0, "smooth sensitivity must be ≥ 0, got {s}");
    }

    #[test]
    fn test_smooth_sensitivity_median_tight_data() {
        // All equal data → median is constant, local sensitivity = 0 → S^β = 0.
        let x = [5.0; 10];
        let cfg = SmoothSensConfig::new(1.0, 0.5, 5).expect("ok");
        let s = smooth_sensitivity_median(&x, &cfg).expect("ok");
        assert!(
            (s - 0.0).abs() < 1e-10,
            "tight data: S^β should be 0, got {s}"
        );
    }

    #[test]
    fn test_smooth_sensitive_noise_finite() {
        let cfg = SmoothSensConfig::new(1.0, 0.3, 3).expect("ok");
        let mut rng = LcgRng::new(42);
        let noisy = smooth_sensitive_noise(5.0, 0.5, &cfg, &mut rng).expect("ok");
        assert!(noisy.is_finite());
    }

    #[test]
    fn test_smooth_config_bad_beta() {
        // β ≥ ε → error.
        assert!(SmoothSensConfig::new(1.0, 1.0, 5).is_err());
        assert!(SmoothSensConfig::new(1.0, 1.5, 5).is_err());
    }
}
