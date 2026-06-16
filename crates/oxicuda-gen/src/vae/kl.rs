//! Gaussian latent space with KL divergence loss.
//!
//! Implements the reparameterisation trick and KL divergence
//! `KL(N(μ, σ²) || N(0,1))` used in VAE training.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── GaussianLatent ───────────────────────────────────────────────────────────

/// Gaussian latent variable with mean `μ` and log-variance `log σ²`.
///
/// Stores the parameters of the approximate posterior `q(z|x) = N(μ, σ²I)`.
#[derive(Debug, Clone)]
pub struct GaussianLatent {
    /// Mean of the approximate posterior.
    pub mu: Vec<f32>,
    /// Log-variance of the approximate posterior: `log σ²`.
    pub logvar: Vec<f32>,
}

impl GaussianLatent {
    /// Create a new Gaussian latent from mean and log-variance vectors.
    ///
    /// # Errors
    /// - `EmptyInput` if vectors are empty
    /// - `DimensionMismatch` if lengths differ
    pub fn new(mu: Vec<f32>, logvar: Vec<f32>) -> GenResult<Self> {
        if mu.is_empty() {
            return Err(GenError::EmptyInput("mu is empty"));
        }
        if mu.len() != logvar.len() {
            return Err(GenError::DimensionMismatch {
                expected: mu.len(),
                got: logvar.len(),
            });
        }
        Ok(Self { mu, logvar })
    }

    /// Create a standard normal latent: `μ = 0`, `log σ² = 0` (σ = 1).
    pub fn standard_normal(dim: usize) -> GenResult<Self> {
        if dim == 0 {
            return Err(GenError::EmptyInput("dim must be > 0"));
        }
        Ok(Self {
            mu: vec![0.0_f32; dim],
            logvar: vec![0.0_f32; dim],
        })
    }

    /// KL divergence from the approximate posterior to the standard normal.
    ///
    /// `KL(N(μ, σ²) || N(0, 1)) = 0.5 * Σ(μ_i² + σ_i² - 1 - log σ_i²)`
    ///
    /// Returns the average KL per dimension.
    ///
    /// # Errors
    /// - `NonFiniteCommitmentLoss` if any element is non-finite
    pub fn kl_loss(&self) -> GenResult<f32> {
        let n = self.mu.len() as f32;
        let mut total = 0.0_f32;
        for (&m, &lv) in self.mu.iter().zip(&self.logvar) {
            // Clamp logvar for numerical stability: σ ∈ [exp(-30), exp(20)]
            let lv_clamped = lv.clamp(-30.0, 20.0);
            let sigma_sq = lv_clamped.exp();
            total += m * m + sigma_sq - 1.0 - lv_clamped;
        }
        let kl = 0.5 * total / n;
        if !kl.is_finite() {
            return Err(GenError::NonFiniteCommitmentLoss(kl));
        }
        Ok(kl)
    }

    /// Per-element KL loss (unreduced).
    ///
    /// Returns a vector of `0.5 * (μ_i² + σ_i² - 1 - log σ_i²)` for each `i`.
    pub fn kl_loss_elementwise(&self) -> Vec<f32> {
        self.mu
            .iter()
            .zip(&self.logvar)
            .map(|(&m, &lv)| {
                let lv_clamped = lv.clamp(-30.0, 20.0);
                0.5 * (m * m + lv_clamped.exp() - 1.0 - lv_clamped)
            })
            .collect()
    }

    /// Sample from the approximate posterior using the reparameterisation trick.
    ///
    /// `z = μ + σ * ε` where `ε ~ N(0, I)`.
    pub fn sample(&self, rng: &mut LcgRng) -> Vec<f32> {
        let sigma = self.std();
        let mut eps = vec![0.0_f32; self.mu.len()];
        rng.fill_normal(&mut eps);
        self.mu
            .iter()
            .zip(&sigma)
            .zip(&eps)
            .map(|((&m, &s), &e)| m + s * e)
            .collect()
    }

    /// Compute the standard deviation vector: `σ_i = exp(0.5 * logvar_i)`.
    ///
    /// Values are clipped to `[exp(-15), exp(10)]` for numerical stability.
    pub fn std(&self) -> Vec<f32> {
        self.logvar
            .iter()
            .map(|&lv| {
                let lv_clamped = lv.clamp(-30.0, 20.0);
                (0.5 * lv_clamped).exp()
            })
            .collect()
    }

    /// Compute the variance vector: `σ²_i = exp(logvar_i)`.
    pub fn var(&self) -> Vec<f32> {
        self.logvar
            .iter()
            .map(|&lv| lv.clamp(-30.0, 20.0).exp())
            .collect()
    }

    /// Return the dimensionality of the latent space.
    pub fn dim(&self) -> usize {
        self.mu.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn kl_zero_for_standard_normal() {
        // μ=0, logvar=0 → KL = 0.5*(0 + 1 - 1 - 0) = 0
        let latent = GaussianLatent::standard_normal(64).expect("standard_normal should succeed");
        let kl = latent.kl_loss().expect("kl_loss should succeed");
        assert!(kl.abs() < EPS, "KL should be ~0 for standard normal: {kl}");
    }

    #[test]
    fn kl_positive_for_nonstandard() {
        // Any deviation from standard normal should give positive KL
        let mu = vec![1.0_f32; 16];
        let logvar = vec![0.0_f32; 16];
        let latent = GaussianLatent::new(mu, logvar).expect("new should succeed");
        let kl = latent.kl_loss().expect("kl_loss should succeed");
        assert!(kl > 0.0, "KL should be positive for μ≠0: {kl}");
    }

    #[test]
    fn kl_positive_for_large_logvar() {
        let mu = vec![0.0_f32; 16];
        let logvar = vec![2.0_f32; 16]; // large variance
        let latent = GaussianLatent::new(mu, logvar).expect("new should succeed");
        let kl = latent.kl_loss().expect("kl_loss should succeed");
        assert!(kl > 0.0, "KL should be positive for logvar≠0: {kl}");
    }

    #[test]
    fn sample_shape() {
        let latent = GaussianLatent::standard_normal(128).expect("standard_normal should succeed");
        let mut rng = LcgRng::new(42);
        let z = latent.sample(&mut rng);
        assert_eq!(z.len(), 128);
    }

    #[test]
    fn sample_is_finite() {
        let latent = GaussianLatent::standard_normal(64).expect("standard_normal should succeed");
        let mut rng = LcgRng::new(7);
        for _ in 0..10 {
            let z = latent.sample(&mut rng);
            assert!(z.iter().all(|v| v.is_finite()), "non-finite sample");
        }
    }

    #[test]
    fn std_for_standard_normal() {
        let latent = GaussianLatent::standard_normal(8).expect("standard_normal should succeed");
        let std = latent.std();
        for &s in &std {
            assert!((s - 1.0).abs() < EPS, "std should be 1.0 for logvar=0: {s}");
        }
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mu = vec![0.0_f32; 8];
        let logvar = vec![0.0_f32; 4];
        assert!(matches!(
            GaussianLatent::new(mu, logvar),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn kl_elementwise_sums_to_total() {
        let mu: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let logvar: Vec<f32> = (0..16).map(|i| -(i as f32) * 0.05).collect();
        let latent = GaussianLatent::new(mu, logvar).expect("new should succeed");
        let kl_total = latent.kl_loss().expect("kl_loss should succeed");
        let kl_elem: f32 = latent.kl_loss_elementwise().iter().sum::<f32>() / 16.0;
        assert!(
            (kl_total - kl_elem).abs() < EPS,
            "elementwise sum should equal total: {kl_total} vs {kl_elem}"
        );
    }

    #[test]
    fn std_bounds_with_extreme_logvar() {
        // Even with extreme logvar, std should be finite
        let mu = vec![0.0_f32; 4];
        let logvar = vec![100.0_f32; 4]; // would overflow exp without clamping
        let latent = GaussianLatent::new(mu, logvar).expect("new should succeed");
        let std = latent.std();
        assert!(
            std.iter().all(|v| v.is_finite()),
            "std should be finite: {std:?}"
        );
    }
}
