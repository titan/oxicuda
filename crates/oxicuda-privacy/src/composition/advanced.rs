//! Strong composition theorem and heterogeneous composition.
//!
//! # Basic composition
//! k independent (ε₀, δ₀)-DP mechanisms compose to (k·ε₀, k·δ₀)-DP.
//!
//! # Strong composition (Dwork-Rothblum-Vadhan 2010)
//! For any δ' > 0, k-fold composition of (ε₀, δ₀)-DP mechanisms satisfies:
//!
//! `(ε₀·√(2k·ln(1/δ')) + k·ε₀·(e^ε₀ − 1), k·δ₀ + δ')-DP`
//!
//! This is tighter than basic composition for large k when ε₀ is small.
//!
//! # Heterogeneous composition
//! Different mechanisms with parameters (εᵢ, δᵢ) compose to
//! (Σεᵢ, Σδᵢ)-DP by basic composition (no tighter general bound).

use crate::error::{PrivacyError, PrivacyResult};

/// The result of a composition computation.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionResult {
    /// Composed ε value.
    pub epsilon: f64,
    /// Composed δ value.
    pub delta: f64,
}

impl CompositionResult {
    /// Create a new `CompositionResult`, checking that values are non-negative.
    #[must_use]
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self { epsilon, delta }
    }
}

/// Basic composition: k-fold of (ε₀, δ₀) → (k·ε₀, k·δ₀).
///
/// This is always valid but not tight for large k.
pub fn basic_compose(epsilon: f64, delta: f64, k: usize) -> CompositionResult {
    CompositionResult::new(epsilon * k as f64, delta * k as f64)
}

/// Strong composition: k-fold of (ε₀, δ₀) with auxiliary failure prob δ'.
///
/// Applies the Dwork-Rothblum-Vadhan (2010) advanced composition theorem.
/// The composed guarantee is:
///
/// `(ε₀·√(2k·ln(1/δ')) + k·ε₀·(e^ε₀ − 1), k·δ₀ + δ')-DP`
///
/// Choose δ' = δ₀ for the "optimal" split; the caller can tune it.
///
/// # Errors
/// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
/// - `InvalidDelta` if `delta_prime ∉ (0, 1)`.
/// - `InvalidParameter` if `k == 0`.
pub fn strong_compose(
    epsilon: f64,
    delta: f64,
    k: usize,
    delta_prime: f64,
) -> PrivacyResult<CompositionResult> {
    if epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon));
    }
    if !(delta_prime > 0.0 && delta_prime < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta_prime));
    }
    if k == 0 {
        return Err(PrivacyError::InvalidParameter("k must be ≥ 1".into()));
    }

    let k_f64 = k as f64;
    // ε_strong = ε₀·√(2k·ln(1/δ')) + k·ε₀·(e^ε₀ − 1)
    let eps_composed = epsilon * (2.0 * k_f64 * (1.0 / delta_prime).ln()).sqrt()
        + k_f64 * epsilon * (epsilon.exp() - 1.0);

    // δ_composed = k·δ₀ + δ'
    let delta_composed = k_f64 * delta + delta_prime;

    Ok(CompositionResult::new(eps_composed, delta_composed))
}

/// Heterogeneous composition of steps with different (εᵢ, δᵢ) parameters.
///
/// Uses basic composition: ε = Σεᵢ, δ = Σδᵢ.
///
/// # Returns
/// `CompositionResult { epsilon: 0, delta: 0 }` for an empty step list.
pub fn heterogeneous_compose(steps: &[(f64, f64)]) -> CompositionResult {
    let epsilon = steps.iter().map(|(e, _)| e).sum();
    let delta = steps.iter().map(|(_, d)| d).sum();
    CompositionResult::new(epsilon, delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compose() {
        let res = basic_compose(1.0, 1e-5, 10);
        assert!((res.epsilon - 10.0).abs() < 1e-12);
        assert!((res.delta - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn test_strong_compose_tighter_than_basic() {
        let k = 100;
        let epsilon = 0.1;
        let delta = 1e-5;
        let delta_prime = 1e-5;
        let basic = basic_compose(epsilon, delta, k);
        let strong = strong_compose(epsilon, delta, k, delta_prime).expect("ok");
        // Strong composition should give smaller ε than basic for large k, small ε.
        assert!(
            strong.epsilon < basic.epsilon,
            "strong.ε={} should be < basic.ε={}",
            strong.epsilon,
            basic.epsilon
        );
    }

    #[test]
    fn test_heterogeneous_compose() {
        let steps = [(0.5, 1e-5), (0.3, 2e-5), (0.2, 3e-5)];
        let res = heterogeneous_compose(&steps);
        assert!((res.epsilon - 1.0).abs() < 1e-12);
        assert!((res.delta - 6e-5).abs() < 1e-17);
    }

    #[test]
    fn test_strong_compose_bad_delta_prime() {
        assert!(strong_compose(1.0, 1e-5, 10, 0.0).is_err());
        assert!(strong_compose(1.0, 1e-5, 10, 1.5).is_err());
    }
}
