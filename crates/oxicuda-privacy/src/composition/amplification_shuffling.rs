//! Privacy amplification by shuffling.
//!
//! Reference: Erlingsson, Feldman, Mironov, Raghunathan, Talwar & Thakurta
//! (2019), "Amplification by Shuffling: From Local to Central Differential
//! Privacy via Anonymity", SODA 2019.
//!
//! # Protocol
//! n users each apply a local (ε₀, 0)-DP randomizer independently.
//! A trusted shuffler permutes all n reports uniformly at random before
//! sending them to the server.  The shuffling breaks the link between
//! individual reports and individuals, yielding a much stronger central
//! guarantee.
//!
//! # Exact bound (Theorem 3.1 of Erlingsson et al.)
//! `ε_central ≤ ln(1 + (e^ε₀ − 1)/(e^ε₀ + 1) · 8·√(2·ln(4/δ)) / √n)`
//!
//! This captures the O(ε₀·√(ln(1/δ)/n)) scaling for small ε₀.

use super::advanced::CompositionResult;
use crate::error::{PrivacyError, PrivacyResult};

/// Privacy amplification via the shuffling model.
///
/// Applies the exact Erlingsson et al. bound.
///
/// # Arguments
/// - `epsilon_local`: local DP parameter ε₀ > 0 (per-user randomizer).
/// - `delta`: target central δ ∈ (0, 1).
/// - `n`: number of users (must be ≥ 1).
///
/// # Returns
/// `CompositionResult { epsilon: ε_central, delta }`.
///
/// # Errors
/// - `NonPositiveEpsilon` if `epsilon_local ≤ 0`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
/// - `InvalidParameter` if `n == 0`.
pub fn amplify_shuffle(
    epsilon_local: f64,
    delta: f64,
    n: usize,
) -> PrivacyResult<CompositionResult> {
    if epsilon_local <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon_local));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    if n == 0 {
        return Err(PrivacyError::InvalidParameter(
            "number of users n must be ≥ 1".into(),
        ));
    }

    let exp_eps = epsilon_local.exp();

    // Ratio (e^ε₀ - 1) / (e^ε₀ + 1)
    let ratio = (exp_eps - 1.0) / (exp_eps + 1.0);

    // Factor 8·√(2·ln(4/δ)) / √n
    let ln_term = (4.0 / delta).ln();
    let factor = 8.0 * (2.0 * ln_term).sqrt() / (n as f64).sqrt();

    // ε_central = ln(1 + ratio * factor)
    let argument = 1.0 + ratio * factor;
    let epsilon_central = if argument > 1.0 {
        argument.ln()
    } else {
        // Degenerate case: return 0 (mechanism is already very private)
        0.0
    };

    Ok(CompositionResult::new(epsilon_central, delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_reduces_epsilon_vs_local() {
        // With n=10000 users and ε₀=1.0, the central ε should be much smaller.
        let local_eps = 1.0;
        let res = amplify_shuffle(local_eps, 1e-6, 10_000).expect("ok");
        assert!(
            res.epsilon < local_eps,
            "central ε={} should be < local ε={}",
            res.epsilon,
            local_eps
        );
    }

    #[test]
    fn test_shuffle_more_users_smaller_epsilon() {
        let res_small = amplify_shuffle(2.0, 1e-5, 100).expect("ok");
        let res_large = amplify_shuffle(2.0, 1e-5, 10_000).expect("ok");
        assert!(
            res_large.epsilon < res_small.epsilon,
            "more users → smaller ε: {} vs {}",
            res_large.epsilon,
            res_small.epsilon
        );
    }

    #[test]
    fn test_shuffle_zero_users_error() {
        assert!(amplify_shuffle(1.0, 1e-5, 0).is_err());
    }

    #[test]
    fn test_shuffle_bad_delta_error() {
        assert!(amplify_shuffle(1.0, 0.0, 100).is_err());
        assert!(amplify_shuffle(1.0, 1.5, 100).is_err());
    }

    #[test]
    fn test_shuffle_epsilon_is_finite() {
        let res = amplify_shuffle(0.5, 1e-8, 1_000).expect("ok");
        assert!(res.epsilon.is_finite(), "epsilon must be finite");
    }
}
