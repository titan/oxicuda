//! Laplacian mechanism for differential privacy.
//!
//! The Laplace mechanism achieves ε-differential privacy for a function
//! with L1 sensitivity Δ by adding noise from Laplace(0, Δ/ε).

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Configuration for the Laplace mechanism.
#[derive(Debug, Clone)]
pub struct LaplacianMechanism {
    /// L1 sensitivity of the function being privatised.
    pub sensitivity: f32,
    /// Privacy budget ε (must be > 0).
    pub epsilon: f32,
}

impl LaplacianMechanism {
    /// Create a validated Laplace mechanism.
    ///
    /// # Errors
    /// Returns `InvalidPrivacyBudget` if `epsilon ≤ 0`, or
    /// `InvalidClipNorm` if `sensitivity ≤ 0`.
    pub fn new(sensitivity: f32, epsilon: f32) -> FedResult<Self> {
        if !(epsilon > 0.0 && epsilon.is_finite()) {
            return Err(FedError::InvalidPrivacyBudget);
        }
        if !(sensitivity > 0.0 && sensitivity.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        Ok(Self {
            sensitivity,
            epsilon,
        })
    }

    /// Compute the Laplace scale parameter b = sensitivity / epsilon.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.sensitivity / self.epsilon
    }

    /// Add Laplace noise Lap(0, b) to each element of `data`.
    ///
    /// Uses the inverse CDF method:
    /// `noise = −b * sign(u − 0.5) * ln(1 − 2*|u − 0.5|)` where `u ∼ Uniform(0,1)`.
    ///
    /// # Errors
    /// Returns `InvalidNoiseMultiplier` if scale is non-finite.
    pub fn add_noise(&self, data: &mut [f32], rng: &mut LcgRng) -> FedResult<()> {
        let b = self.scale();
        if !b.is_finite() || b <= 0.0 {
            return Err(FedError::InvalidNoiseMultiplier);
        }
        for val in data.iter_mut() {
            *val += rng.next_laplace(b);
        }
        Ok(())
    }

    /// Calibrate the sensitivity for a given target epsilon.
    ///
    /// Returns the scale b = sensitivity / epsilon for use in noise generation.
    #[must_use]
    pub fn calibrated_scale(sensitivity: f32, epsilon: f32) -> f32 {
        (sensitivity / epsilon).max(1e-10)
    }

    /// Clip a gradient to L1 norm ball of radius `clip_norm`.
    ///
    /// For each element: `g_i = g_i * min(1, clip_norm / max(||g||₁, 1e-6))`.
    ///
    /// # Errors
    /// Returns `InvalidClipNorm` if `clip_norm ≤ 0`.
    pub fn clip_l1(gradient: &mut [f32], clip_norm: f32) -> FedResult<()> {
        if !(clip_norm > 0.0 && clip_norm.is_finite()) {
            return Err(FedError::InvalidClipNorm);
        }
        let norm: f32 = gradient.iter().map(|&g| g.abs()).sum();
        let norm_safe = norm.max(1e-6);
        if norm_safe > clip_norm {
            let scale = clip_norm / norm_safe;
            for g in gradient.iter_mut() {
                *g *= scale;
            }
        }
        Ok(())
    }
}

/// Add Laplace noise from Lap(0, b) directly to each element of `data`.
///
/// This function is a standalone convenience wrapper.
///
/// # Errors
/// Returns `InvalidNoiseMultiplier` if `b ≤ 0` or non-finite.
pub fn add_laplacian_noise(data: &mut [f32], b: f32, rng: &mut LcgRng) -> FedResult<()> {
    if !(b > 0.0 && b.is_finite()) {
        return Err(FedError::InvalidNoiseMultiplier);
    }
    for val in data.iter_mut() {
        *val += rng.next_laplace(b);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplacian_scale_formula() {
        let mech =
            LaplacianMechanism::new(2.0, 0.5).expect("test invariant: valid laplacian mechanism");
        assert!((mech.scale() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn laplacian_add_noise_changes_data() {
        let mech =
            LaplacianMechanism::new(1.0, 1.0).expect("test invariant: valid laplacian mechanism");
        let mut data = vec![0.0f32; 20];
        let mut rng = LcgRng::new(42);
        mech.add_noise(&mut data, &mut rng)
            .expect("test invariant: valid add noise");
        assert!(data.iter().any(|&v| v != 0.0));
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn laplacian_invalid_epsilon() {
        assert!(matches!(
            LaplacianMechanism::new(1.0, -1.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn laplacian_clip_l1() {
        let mut grad = vec![0.5f32, 0.5, 0.5, 0.5]; // L1 = 2.0
        LaplacianMechanism::clip_l1(&mut grad, 1.0).expect("test invariant: valid L1 clip");
        let l1: f32 = grad.iter().map(|&g| g.abs()).sum();
        assert!(l1 <= 1.0 + 1e-5, "L1 norm={l1} exceeds clip_norm=1.0");
    }

    #[test]
    fn add_laplacian_noise_finite() {
        let mut data = vec![0.0f32; 50];
        let mut rng = LcgRng::new(17);
        add_laplacian_noise(&mut data, 1.0, &mut rng)
            .expect("test invariant: valid laplacian noise");
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn add_laplacian_noise_invalid_b() {
        let mut data = vec![0.0f32; 5];
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            add_laplacian_noise(&mut data, 0.0, &mut rng),
            Err(FedError::InvalidNoiseMultiplier)
        ));
    }
}
