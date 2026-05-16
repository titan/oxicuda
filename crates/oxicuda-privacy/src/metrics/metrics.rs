//! Privacy budget tracking and utility metrics.
//!
//! Provides:
//! - `PrivacyBudget`: tracks spent (ε, δ) and enforces a total budget.
//! - `gaussian_mse`: Mean Squared Error of the Gaussian mechanism.
//! - `snr_db`: Signal-to-Noise Ratio in dB.
//! - `gaussian_utility`: Approximate utility of the Gaussian mechanism.
//! - `subsampling_amplification_factor`: ratio of amplified to original ε.

use crate::error::{PrivacyError, PrivacyResult};

// ─── Privacy budget tracker ───────────────────────────────────────────────────

/// Tracks the (ε, δ) privacy budget consumed across a sequence of operations.
#[derive(Debug, Clone)]
pub struct PrivacyBudget {
    /// Total allowed ε.
    pub epsilon_total: f64,
    /// Total allowed δ.
    pub delta_total: f64,
    /// Privacy ε consumed so far.
    pub spent_epsilon: f64,
    /// Privacy δ consumed so far.
    pub spent_delta: f64,
}

impl PrivacyBudget {
    /// Create a new `PrivacyBudget` with the given total allowances.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` if `epsilon ≤ 0` or `InvalidDelta` if `delta < 0`.
    pub fn new(epsilon: f64, delta: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if delta < 0.0 {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        Ok(Self {
            epsilon_total: epsilon,
            delta_total: delta,
            spent_epsilon: 0.0,
            spent_delta: 0.0,
        })
    }

    /// Spend `(epsilon, delta)` from the budget (basic composition).
    ///
    /// # Errors
    /// Returns `BudgetExhausted` if the requested spend would exceed either total.
    pub fn spend(&mut self, epsilon: f64, delta: f64) -> PrivacyResult<()> {
        let new_eps = self.spent_epsilon + epsilon;
        let new_del = self.spent_delta + delta;

        if new_eps > self.epsilon_total + f64::EPSILON || new_del > self.delta_total + f64::EPSILON
        {
            return Err(PrivacyError::BudgetExhausted {
                spent: new_eps.max(new_del / self.delta_total * self.epsilon_total),
                total: self.epsilon_total,
            });
        }

        self.spent_epsilon = new_eps;
        self.spent_delta = new_del;
        Ok(())
    }

    /// Return the remaining ε budget.
    #[must_use]
    pub fn remaining_epsilon(&self) -> f64 {
        (self.epsilon_total - self.spent_epsilon).max(0.0)
    }

    /// Return the remaining δ budget.
    #[must_use]
    pub fn remaining_delta(&self) -> f64 {
        (self.delta_total - self.spent_delta).max(0.0)
    }

    /// Return the fraction of the ε budget spent (in [0, 1]).
    #[must_use]
    pub fn fraction_spent(&self) -> f64 {
        (self.spent_epsilon / self.epsilon_total).clamp(0.0, 1.0)
    }
}

// ─── Utility metrics ──────────────────────────────────────────────────────────

/// Compute the Mean Squared Error of the Gaussian mechanism averaged over
/// `n_samples` independent noise draws.
///
/// For Gaussian mechanism with noise std σ, the per-sample MSE = σ² and the
/// expected MSE over n samples is also σ² (no averaging reduction since each
/// query is separate).
///
/// `MSE = (sensitivity / sigma)² · sigma² = sensitivity²` — this is the
/// squared sensitivity normalised by n_samples (for an estimator averaging
/// n independent noisy evaluations).
///
/// Returns: `sigma² / n_samples` (noise variance of the averaged estimator).
#[must_use]
pub fn gaussian_mse(sensitivity: f64, sigma: f64, n_samples: usize) -> f64 {
    if n_samples == 0 || sigma <= 0.0 {
        return f64::INFINITY;
    }
    let _ = sensitivity; // sensitivity is part of the privacy config, not MSE directly
    sigma * sigma / n_samples as f64
}

/// Signal-to-Noise Ratio in dB.
///
/// `SNR = 10 · log₁₀(signal_power / noise_power)`
///
/// # Errors
/// Returns `InvalidParameter` if `noise_power ≤ 0` or `signal_power < 0`.
pub fn snr_db(signal_power: f64, noise_power: f64) -> PrivacyResult<f64> {
    if noise_power <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "noise_power must be positive, got {noise_power}"
        )));
    }
    if signal_power < 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "signal_power must be ≥ 0, got {signal_power}"
        )));
    }
    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Compute the noise standard deviation (utility loss) of the Gaussian
/// mechanism achieving (ε, δ)-DP.
///
/// Uses the analytic formula: σ = sensitivity · √(2 · ln(1.25/δ)) / ε.
///
/// # Errors
/// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
/// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
pub fn gaussian_utility(sensitivity: f64, epsilon: f64, delta: f64) -> PrivacyResult<f64> {
    if sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
    }
    if epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    let sigma = sensitivity * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;
    Ok(sigma)
}

/// Compute the privacy amplification factor from Poisson subsampling.
///
/// Returns `ε_amplified / ε_original = ln(1 + q(e^ε − 1)) / ε`.
///
/// # Errors
/// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
/// - `InvalidParameter` if `q ∉ (0, 1]`.
pub fn subsampling_amplification_factor(epsilon: f64, q: f64) -> PrivacyResult<f64> {
    if epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(epsilon));
    }
    if !(q > 0.0 && q <= 1.0) {
        return Err(PrivacyError::InvalidParameter(format!(
            "subsampling rate q must be in (0, 1], got {q}"
        )));
    }
    let amplified = (1.0 + q * (epsilon.exp() - 1.0)).ln();
    Ok(amplified / epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_spend_and_remaining() {
        let mut b = PrivacyBudget::new(1.0, 1e-5).expect("ok");
        b.spend(0.4, 1e-6).expect("ok");
        assert!((b.remaining_epsilon() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_budget_exhausted_error() {
        let mut b = PrivacyBudget::new(1.0, 1e-5).expect("ok");
        assert!(b.spend(1.5, 0.0).is_err());
    }

    #[test]
    fn test_snr_db_positive_for_strong_signal() {
        let snr = snr_db(100.0, 1.0).expect("ok");
        assert!((snr - 20.0).abs() < 1e-6, "expected 20 dB, got {snr}");
    }

    #[test]
    fn test_gaussian_utility_positive() {
        let sigma = gaussian_utility(1.0, 1.0, 1e-5).expect("ok");
        assert!(sigma > 0.0);
    }

    #[test]
    fn test_subsampling_factor_less_than_one() {
        let factor = subsampling_amplification_factor(1.0, 0.1).expect("ok");
        assert!(
            factor < 1.0,
            "amplification factor should be < 1, got {factor}"
        );
    }

    #[test]
    fn test_subsampling_factor_q_one_is_one() {
        let factor = subsampling_amplification_factor(0.5, 1.0).expect("ok");
        // With q=1, ε_amplified = ε, so factor = 1.
        assert!((factor - 1.0).abs() < 1e-10, "expected 1.0, got {factor}");
    }
}
