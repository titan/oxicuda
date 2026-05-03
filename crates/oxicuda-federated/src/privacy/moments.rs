//! Moments accountant for differential privacy composition.
//!
//! Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
//!
//! The moments accountant tracks the privacy loss of iteratively applied
//! Gaussian mechanisms by computing log-moments and converting to (ε, δ)-DP
//! using Markov's inequality.

use crate::error::{FedError, FedResult};

/// Moments accountant for composing Gaussian DP mechanisms.
#[derive(Debug, Clone)]
pub struct MomentsAccountant {
    /// Noise multiplier: σ / sensitivity (must be > 0).
    pub noise_multiplier: f32,
    /// Number of mechanism applications (compositions) tracked so far.
    pub steps: usize,
    /// Target δ for (ε, δ)-DP conversion.
    pub delta: f32,
}

impl MomentsAccountant {
    /// Create a new moments accountant.
    ///
    /// # Errors
    /// Returns `InvalidNoiseMultiplier` if `noise_multiplier ≤ 0`, or
    /// `InvalidPrivacyBudget` if `delta ≤ 0`.
    pub fn new(noise_multiplier: f32, delta: f32) -> FedResult<Self> {
        if !(noise_multiplier > 0.0 && noise_multiplier.is_finite()) {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        Ok(Self {
            noise_multiplier,
            steps: 0,
            delta,
        })
    }

    /// Compute the log of the α-th moment of the privacy loss random variable
    /// for a single Gaussian mechanism application.
    ///
    /// Under the subsampled Gaussian mechanism (sampling probability q=1),
    /// the log moment is bounded by:
    ///
    /// `log E[exp((α-1) * ℓ)] ≤ (α-1) * α / (2 * σ²) + O(1/σ⁴)`
    ///
    /// The exact formula for a single application (q=1) is:
    /// `log_moment(α) = (α-1)*α / (2*σ²)`
    ///
    /// For `steps` compositions: `log_moment_total = steps * log_moment(α)`.
    #[must_use]
    pub fn log_moment(&self, alpha: f32) -> f32 {
        // For the Gaussian mechanism with noise std = sigma * sensitivity,
        // the Rényi divergence of order alpha is:
        // D_alpha(M(D) || M(D')) = alpha / (2 * sigma^2)
        // The log moment is:
        // log E[exp((alpha-1) * privacy_loss)] = (alpha-1) * alpha / (2 * sigma^2)
        // For steps compositions: multiply by steps
        let sigma = self.noise_multiplier as f64;
        let a = alpha as f64;
        let single_moment = (a - 1.0) * a / (2.0 * sigma * sigma);
        (self.steps as f64 * single_moment) as f32
    }

    /// Convert the α-th moment to an epsilon bound using Markov's inequality.
    ///
    /// `ε(α) = (log_moment(α) − log(δ)) / (α − 1)`
    #[must_use]
    pub fn epsilon_from_moment(&self, alpha: f32) -> f32 {
        if alpha <= 1.0 {
            return f32::INFINITY;
        }
        let log_m = self.log_moment(alpha) as f64;
        let log_delta = (self.delta as f64).ln();
        let eps = (log_m - log_delta) / (alpha as f64 - 1.0);
        eps.max(0.0) as f32
    }

    /// Compute the tightest (ε, δ)-DP bound by optimising over α ∈ [2, 128].
    ///
    /// Returns the smallest ε achievable over all integer moments α.
    ///
    /// # Errors
    /// Returns `InvalidNoiseMultiplier` if no finite bound can be found
    /// (e.g., steps = 0).
    pub fn compute_epsilon(&self) -> FedResult<f32> {
        if self.steps == 0 {
            return Ok(0.0);
        }
        let mut best_eps = f32::INFINITY;
        for alpha_int in 2_u32..=128 {
            let alpha = alpha_int as f32;
            let eps = self.epsilon_from_moment(alpha);
            if eps.is_finite() && eps < best_eps {
                best_eps = eps;
            }
        }
        if best_eps.is_infinite() {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        Ok(best_eps)
    }

    /// Increment the composition counter by `n_steps`.
    pub fn compose_steps(&mut self, n_steps: usize) {
        self.steps = self.steps.saturating_add(n_steps);
    }

    /// Reset the composition counter to zero.
    pub fn reset(&mut self) {
        self.steps = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_accountant_new_valid() {
        let ma =
            MomentsAccountant::new(1.5, 1e-5).expect("test invariant: valid moments accountant");
        assert_eq!(ma.steps, 0);
        assert!((ma.noise_multiplier - 1.5).abs() < 1e-6);
    }

    #[test]
    fn moments_accountant_invalid_noise() {
        assert!(matches!(
            MomentsAccountant::new(0.0, 1e-5),
            Err(FedError::InvalidNoiseMultiplier)
        ));
    }

    #[test]
    fn moments_accountant_invalid_delta() {
        assert!(matches!(
            MomentsAccountant::new(1.0, 0.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn moments_zero_steps_zero_epsilon() {
        let ma =
            MomentsAccountant::new(1.0, 1e-5).expect("test invariant: valid moments accountant");
        assert_eq!(ma.steps, 0);
        let eps = ma.compute_epsilon().expect("test invariant: valid epsilon");
        assert_eq!(eps, 0.0);
    }

    #[test]
    fn moments_compose_increases_epsilon() {
        let mut ma =
            MomentsAccountant::new(1.0, 1e-5).expect("test invariant: valid moments accountant");
        ma.compose_steps(10);
        let eps10 = ma
            .compute_epsilon()
            .expect("test invariant: valid epsilon 10 steps");
        ma.compose_steps(10);
        let eps20 = ma
            .compute_epsilon()
            .expect("test invariant: valid epsilon 20 steps");
        assert!(eps20 > eps10, "more steps → larger epsilon");
    }

    #[test]
    fn moments_log_moment_positive() {
        let mut ma =
            MomentsAccountant::new(1.0, 1e-5).expect("test invariant: valid moments accountant");
        ma.compose_steps(1);
        let lm = ma.log_moment(4.0);
        assert!(lm >= 0.0, "log moment should be non-negative");
    }

    #[test]
    fn moments_reset() {
        let mut ma =
            MomentsAccountant::new(1.0, 1e-5).expect("test invariant: valid moments accountant");
        ma.compose_steps(100);
        ma.reset();
        assert_eq!(ma.steps, 0);
    }
}
