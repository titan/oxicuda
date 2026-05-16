//! Privacy amplification by subsampling.
//!
//! Subsampling a dataset before applying an (ε, δ)-DP mechanism amplifies the
//! privacy guarantee by effectively reducing the probability that any individual
//! participates in a given computation.
//!
//! # Poisson subsampling (Balle et al. 2018, Theorem 9)
//! Each element is included independently with probability q ∈ (0, 1].
//! The amplified guarantee is:
//!
//! `(ln(1 + q·(e^ε − 1)), q·δ)-DP`
//!
//! # Uniform without-replacement subsampling
//! Sample m elements from n without replacement (q = m/n).
//! By the Balle et al. analysis the same Poisson formula applies with q = m/n,
//! plus an additional factor from the without-replacement correction.
//!
//! The exact without-replacement bound is tighter but complex; we use the
//! Poisson upper bound (conservative).

use super::advanced::CompositionResult;
use crate::error::{PrivacyError, PrivacyResult};

/// Privacy amplification via Poisson subsampling.
///
/// Each record is included independently with probability q.
///
/// Amplified ε: `ε' = ln(1 + q·(e^ε − 1))`
/// Amplified δ: `δ' = q·δ`
///
/// # Errors
/// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
/// - `InvalidParameter` if `q ∉ (0, 1]`.
/// - `InvalidDelta` if `delta < 0`.
pub fn amplify_poisson(epsilon: f64, delta: f64, q: f64) -> PrivacyResult<CompositionResult> {
    if epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon));
    }
    if !(q > 0.0 && q <= 1.0) {
        return Err(PrivacyError::InvalidParameter(format!(
            "subsampling rate q must be in (0, 1], got {q}"
        )));
    }
    if delta < 0.0 {
        return Err(PrivacyError::InvalidDelta(delta));
    }

    let amplified_epsilon = (1.0 + q * (epsilon.exp() - 1.0)).ln();
    let amplified_delta = q * delta;

    Ok(CompositionResult::new(amplified_epsilon, amplified_delta))
}

/// Privacy amplification via uniform without-replacement subsampling.
///
/// Samples m records from a dataset of size n (n > 0, m ≤ n).
/// Uses the Poisson-subsampling upper bound with rate q = m/n.
///
/// # Errors
/// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
/// - `InvalidParameter` if `m > n`, `n == 0`, or `m == 0`.
/// - `InvalidDelta` if `delta < 0`.
pub fn amplify_uniform(
    epsilon: f64,
    delta: f64,
    m: usize,
    n: usize,
) -> PrivacyResult<CompositionResult> {
    if epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon));
    }
    if n == 0 {
        return Err(PrivacyError::InvalidParameter(
            "dataset size n must be > 0".into(),
        ));
    }
    if m == 0 {
        return Err(PrivacyError::InvalidParameter(
            "batch size m must be > 0".into(),
        ));
    }
    if m > n {
        return Err(PrivacyError::InvalidParameter(format!(
            "batch size m={m} must be ≤ dataset size n={n}"
        )));
    }
    if delta < 0.0 {
        return Err(PrivacyError::InvalidDelta(delta));
    }

    let q = m as f64 / n as f64;
    amplify_poisson(epsilon, delta, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_amplification_reduces_epsilon() {
        let original = 1.0;
        let res = amplify_poisson(original, 1e-5, 0.01).expect("ok");
        assert!(
            res.epsilon < original,
            "amplified ε={} should be < original ε={}",
            res.epsilon,
            original
        );
    }

    #[test]
    fn test_poisson_q_one_is_identity() {
        let res = amplify_poisson(0.5, 1e-5, 1.0).expect("ok");
        // ln(1 + 1*(e^ε-1)) = ln(e^ε) = ε exactly.
        assert!((res.epsilon - 0.5).abs() < 1e-12);
        assert!((res.delta - 1e-5).abs() < 1e-18);
    }

    #[test]
    fn test_uniform_amplification() {
        let res = amplify_uniform(1.0, 1e-5, 10, 1000).expect("ok");
        assert!(res.epsilon < 1.0);
        assert!(res.delta < 1e-5);
    }

    #[test]
    fn test_uniform_m_exceeds_n_error() {
        assert!(amplify_uniform(1.0, 1e-5, 100, 50).is_err());
    }

    #[test]
    fn test_poisson_bad_q_error() {
        assert!(amplify_poisson(1.0, 1e-5, 0.0).is_err());
        assert!(amplify_poisson(1.0, 1e-5, 1.5).is_err());
    }
}
