//! Gaussian mechanism for differential privacy.
//!
//! Implements the analytic Gaussian mechanism for (ε, δ)-differential privacy,
//! as used in DP-SGD (Abadi et al., 2016).

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Gaussian mechanism for (ε, δ)-differential privacy.
#[derive(Debug, Clone)]
pub struct GaussianMechanism {
    /// L2 sensitivity of the function being privatised.
    pub sensitivity: f32,
    /// Privacy budget ε (must be > 0).
    pub epsilon: f32,
    /// Privacy budget δ (must be in (0, 1)).
    pub delta: f32,
}

impl GaussianMechanism {
    /// Create a new Gaussian mechanism with validation.
    ///
    /// # Errors
    /// Returns `InvalidPrivacyBudget` if ε or δ are out of range, or
    /// `InvalidClipNorm` if `sensitivity` is non-positive.
    pub fn new(sensitivity: f32, epsilon: f32, delta: f32) -> FedResult<Self> {
        if !(epsilon > 0.0 && epsilon.is_finite() && delta > 0.0 && delta < 1.0) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if !(sensitivity > 0.0 && sensitivity.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        Ok(Self {
            sensitivity,
            epsilon,
            delta,
        })
    }

    /// Compute the noise standard deviation σ for this mechanism.
    ///
    /// Uses the analytic Gaussian formula:
    /// `σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε`
    #[must_use]
    pub fn sigma(&self) -> f32 {
        let factor = (2.0 * (1.25 / self.delta).ln()).sqrt();
        self.sensitivity * factor / self.epsilon
    }

    /// Clip a gradient vector to the L2 ball of radius `clip_norm`.
    ///
    /// Computes `g ← g * min(1, clip_norm / max(||g||₂, 1e-6))`.
    ///
    /// # Errors
    /// Returns `InvalidClipNorm` if `clip_norm` is non-positive or non-finite.
    pub fn clip_gradient(gradient: &mut [f32], clip_norm: f32) -> FedResult<()> {
        if !(clip_norm > 0.0 && clip_norm.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        let norm_sq: f32 = gradient.iter().map(|&g| g * g).sum();
        let norm = norm_sq.sqrt();
        let norm_safe = norm.max(1e-6);
        if norm_safe > clip_norm {
            let scale = clip_norm / norm_safe;
            for g in gradient.iter_mut() {
                *g *= scale;
            }
        }
        Ok(())
    }

    /// Add isotropic Gaussian noise `N(0, σ²·I)` to a gradient.
    ///
    /// Uses Box-Muller transform via the provided `LcgRng`.
    ///
    /// # Errors
    /// Returns `InvalidNoiseMultiplier` if `sigma()` is non-finite.
    pub fn add_noise(&self, gradient: &mut [f32], rng: &mut LcgRng) -> FedResult<()> {
        let sigma = self.sigma();
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        let mut i = 0;
        while i < gradient.len() {
            let (z1, z2) = rng.next_normal_pair();
            gradient[i] += sigma * z1;
            i += 1;
            if i < gradient.len() {
                gradient[i] += sigma * z2;
                i += 1;
            }
        }
        Ok(())
    }

    /// Apply DP-SGD step: clip gradient then add Gaussian noise.
    ///
    /// # Errors
    /// Returns `InvalidClipNorm` or `InvalidNoiseMultiplier` on failure.
    pub fn dp_sgd_step(&self, gradient: &mut [f32], rng: &mut LcgRng) -> FedResult<()> {
        Self::clip_gradient(gradient, self.sensitivity)?;
        self.add_noise(gradient, rng)
    }

    /// Compute the L2 norm of a gradient vector.
    #[must_use]
    pub fn l2_norm(gradient: &[f32]) -> f32 {
        gradient.iter().map(|&g| g * g).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_sigma_formula() {
        // sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        let mech = GaussianMechanism::new(1.0, 1.0, 0.1)
            .expect("test invariant: valid gaussian mechanism");
        let expected = (2.0 * (1.25_f32 / 0.1).ln()).sqrt();
        assert!(
            (mech.sigma() - expected).abs() < 1e-5,
            "sigma={}",
            mech.sigma()
        );
    }

    #[test]
    fn gaussian_clip_respects_norm() {
        let mut grad = vec![3.0f32, 4.0]; // norm = 5.0
        GaussianMechanism::clip_gradient(&mut grad, 1.0).expect("test invariant: valid clip");
        let norm = GaussianMechanism::l2_norm(&grad);
        assert!(
            norm <= 1.0 + 1e-5,
            "clipped norm={norm} exceeds clip_norm=1.0"
        );
    }

    #[test]
    fn gaussian_clip_identity_when_below_norm() {
        let mut grad = vec![0.3f32, 0.4]; // norm = 0.5 < 2.0
        GaussianMechanism::clip_gradient(&mut grad, 2.0).expect("test invariant: valid clip");
        assert!((grad[0] - 0.3).abs() < 1e-6 && (grad[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn gaussian_clip_invalid_norm() {
        let mut grad = vec![1.0f32];
        assert!(matches!(
            GaussianMechanism::clip_gradient(&mut grad, -1.0),
            Err(FedError::InvalidClipNorm)
        ));
    }

    #[test]
    fn gaussian_add_noise_changes_gradient() {
        let mech = GaussianMechanism::new(1.0, 0.5, 0.01)
            .expect("test invariant: valid gaussian mechanism");
        let mut grad = vec![0.0f32; 10];
        let mut rng = LcgRng::new(42);
        mech.add_noise(&mut grad, &mut rng)
            .expect("test invariant: valid add noise");
        assert!(
            grad.iter().any(|&v| v != 0.0),
            "noise should have changed the gradient"
        );
    }

    #[test]
    fn gaussian_mechanism_invalid_epsilon() {
        assert!(matches!(
            GaussianMechanism::new(1.0, 0.0, 0.1),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn gaussian_mechanism_invalid_delta() {
        assert!(matches!(
            GaussianMechanism::new(1.0, 1.0, 1.5),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn dp_sgd_step_clips_and_noises() {
        let mech = GaussianMechanism::new(1.0, 0.5, 0.01)
            .expect("test invariant: valid gaussian mechanism");
        let mut grad = vec![10.0f32, 10.0, 10.0, 10.0]; // large gradient
        let mut rng = LcgRng::new(7);
        mech.dp_sgd_step(&mut grad, &mut rng)
            .expect("test invariant: valid dp_sgd_step");
        // After clipping to sensitivity=1.0 and adding noise, some change occurred
        assert!(grad.iter().all(|v| v.is_finite()));
    }
}
