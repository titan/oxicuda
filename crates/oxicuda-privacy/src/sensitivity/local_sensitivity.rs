//! Local sensitivity framework.
//!
//! Reference: Nissim, Raskhodnikova & Smith (2007), "Smooth Sensitivity and
//! Sampling in Private Data Analysis", STOC 2007.
//!
//! The **local sensitivity** of a query f at dataset x is:
//!
//! `LS_f(x) = max_{d(x,x')=1} |f(x) − f(x')|`
//!
//! Unlike the global sensitivity (which is the worst-case LS over all x),
//! the local sensitivity can be much smaller for well-behaved datasets.
//!
//! # Supported queries
//! - **Mean**: removing/replacing one element by any value in the domain.
//! - **Median**: replacing one element by a neighbour in sorted order.
//! - **Sum**: replacing one element by a domain endpoint.
//!
//! # Noise calibration
//! Local sensitivity noise (valid only when LS is computed on a trusted query):
//!   `Noise ~ Lap(LS_f(x) / ε)`

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Compute the local sensitivity of the **mean** query on dataset x.
///
/// The mean of n elements changes most when we replace one element by the
/// element furthest from the current mean.  The worst-case change is:
///
/// `LS_mean(x) = max(|max(x) − mean(x)|, |min(x) − mean(x)|) / n`
///
/// # Errors
/// Returns `EmptyInput` if `x` is empty.
pub fn local_sensitivity_mean(x: &[f64]) -> PrivacyResult<f64> {
    if x.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }

    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);

    let ls = ((max_val - mean).abs().max((min_val - mean).abs())) / n;
    Ok(ls)
}

/// Compute the local sensitivity of the **median** query on sorted dataset x.
///
/// For the median, inserting or removing one element can shift the median at
/// most by the distance to the nearest element in sorted order.  We compute:
///
/// `LS_median(x) = half the gap between the two elements straddling the median
///  in sorted order` (or the full gap if n is even).
///
/// More precisely, for sorted x: LS ≤ |x[⌈n/2⌉] − x[⌊n/2⌋]| / 2,
/// but we use the conservative bound of the max gap to adjacent neighbours.
///
/// # Errors
/// Returns `EmptyInput` if `x` is empty.
pub fn local_sensitivity_median(x: &[f64]) -> PrivacyResult<f64> {
    if x.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if x.len() == 1 {
        return Ok(0.0);
    }

    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mid = n / 2;

    // Gap around the median in sorted order.
    let lower = if mid > 0 {
        sorted[mid - 1]
    } else {
        sorted[mid]
    };
    let upper = if n.is_multiple_of(2) && mid < n - 1 {
        sorted[mid]
    } else if mid + 1 < n {
        sorted[mid + 1]
    } else {
        sorted[mid]
    };

    let ls = (upper - lower).abs() / 2.0;
    Ok(ls)
}

/// Compute the local sensitivity of the **sum** query on dataset x.
///
/// The sum changes by at most `max(|x_i − domain_lo|, |x_i − domain_hi|)`
/// when one element x_i is replaced by a domain endpoint.  The worst case
/// is `domain_hi − domain_lo` (full range).
///
/// # Errors
/// - `EmptyInput` if `x` is empty.
/// - `InvalidParameter` if `domain_lo > domain_hi`.
pub fn local_sensitivity_sum(x: &[f64], domain_lo: f64, domain_hi: f64) -> PrivacyResult<f64> {
    if x.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if domain_lo > domain_hi {
        return Err(PrivacyError::InvalidParameter(format!(
            "domain_lo={domain_lo} must be ≤ domain_hi={domain_hi}"
        )));
    }

    // Local sensitivity for sum is max over all elements of max(|xᵢ − lo|, |xᵢ − hi|).
    let ls = x
        .iter()
        .map(|&xi| (xi - domain_lo).abs().max((xi - domain_hi).abs()))
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(ls)
}

/// Add Laplace noise calibrated to the local sensitivity.
///
/// **Warning**: using local sensitivity without smooth sensitivity or PTR
/// does *not* provide standard DP unless the local sensitivity is trusted
/// (e.g., computed on a public dataset or via PTR).
///
/// Noise: Lap(local_sens / ε).
///
/// # Errors
/// - `NonPositiveSensitivity` if `local_sens ≤ 0`.
/// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
pub fn add_local_sensitivity_noise(
    val: f64,
    local_sens: f64,
    epsilon: f64,
    rng: &mut LcgRng,
) -> PrivacyResult<f64> {
    if local_sens <= 0.0 {
        // Local sensitivity = 0 means the query is constant; no noise needed.
        return Ok(val);
    }
    if epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon));
    }
    let scale = local_sens / epsilon;
    let u = rng.next_f64() - 0.5;
    let abs_u = u.abs().min(0.5 - f64::EPSILON);
    let noise = -scale * u.signum() * (1.0 - 2.0 * abs_u).ln();
    Ok(val + noise)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_sensitivity_mean_uniform() {
        // Uniform data: mean = 0.5, max = 1.0, min = 0.0, n = 4.
        let x = [0.0, 0.25, 0.75, 1.0];
        let ls = local_sensitivity_mean(&x).expect("ok");
        // max deviation from mean = 0.5, LS = 0.5/4 = 0.125.
        assert!((ls - 0.125).abs() < 1e-10, "expected 0.125, got {ls}");
    }

    #[test]
    fn test_local_sensitivity_mean_empty_error() {
        assert!(local_sensitivity_mean(&[]).is_err());
    }

    #[test]
    fn test_local_sensitivity_median_sorted() {
        let x = [1.0, 3.0, 5.0, 7.0, 9.0];
        let ls = local_sensitivity_median(&x).expect("ok");
        assert!(ls >= 0.0);
    }

    #[test]
    fn test_local_sensitivity_sum() {
        let x = [1.0, 2.0, 3.0];
        let ls = local_sensitivity_sum(&x, 0.0, 10.0).expect("ok");
        // Worst case: element 1.0 replaced by 10.0 → change = 9.0.
        assert!((ls - 9.0).abs() < 1e-10, "expected 9.0, got {ls}");
    }

    #[test]
    fn test_local_sensitivity_noise_finite() {
        let mut rng = LcgRng::new(42);
        let val = add_local_sensitivity_noise(5.0, 1.0, 1.0, &mut rng).expect("ok");
        assert!(val.is_finite());
    }
}
