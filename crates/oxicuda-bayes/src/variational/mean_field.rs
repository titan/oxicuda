//! Mean-field variational distribution: factored q(z) = Π_i q(z_i).
//!
//! Provides entropy computation, ELBO objective, and marginal sampling.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;
use crate::variational::elbo::{elbo, kl_gaussian_vec};
use crate::variational::reparam::sample_gaussian_vec;

/// Mean-field Gaussian variational distribution: q(z) = Π_i N(μ_i, σ_i²).
///
/// Each dimension is independent (no correlation), which makes
/// the KL divergence and entropy factorize into per-dimension terms.
#[derive(Debug, Clone)]
pub struct MeanFieldDist {
    /// Variational means μ_i for each latent dimension.
    pub mu: Vec<f32>,
    /// Variational log-variances log(σ_i²) for each latent dimension.
    pub log_var: Vec<f32>,
}

impl MeanFieldDist {
    /// Create a new mean-field distribution with given parameters.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if `mu` and `log_var` differ in length,
    /// or `BayesError::EmptyInputs` if empty.
    pub fn new(mu: Vec<f32>, log_var: Vec<f32>) -> BayesResult<Self> {
        if mu.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if mu.len() != log_var.len() {
            return Err(BayesError::DimensionMismatch {
                expected: mu.len(),
                got: log_var.len(),
            });
        }
        Ok(Self { mu, log_var })
    }

    /// Latent dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.mu.len()
    }

    /// Differential entropy of the mean-field Gaussian:
    /// `H[q] = Σ_i 0.5 * (1 + ln(2π) + log_var_i)`.
    ///
    /// # Errors
    /// Returns `BayesError::NanEncountered` if any log_var is non-finite.
    pub fn entropy(&self) -> BayesResult<f32> {
        let log2pi1 = 1.0 + (2.0 * std::f32::consts::PI).ln();
        let mut h = 0.0_f32;
        for &lv in &self.log_var {
            if !lv.is_finite() {
                return Err(BayesError::NanEncountered {
                    location: "MeanFieldDist::entropy: non-finite log_var",
                });
            }
            h += 0.5 * (log2pi1 + lv);
        }
        Ok(h)
    }

    /// KL divergence KL(q(z) ‖ p(z)) where p(z) = N(0, I).
    ///
    /// Due to factorization: KL = Σ_i KL(N(μ_i, σ_i²) ‖ N(0, 1)).
    ///
    /// # Errors
    /// Returns `BayesError::NanEncountered` if computation produces NaN.
    pub fn kl_divergence(&self) -> BayesResult<f32> {
        kl_gaussian_vec(&self.mu, &self.log_var)
    }

    /// ELBO = -reconstruction_loss - β * KL(q ‖ p).
    ///
    /// # Errors
    /// Propagates errors from `kl_divergence`.
    pub fn elbo(&self, reconstruction_loss: f32, beta: f32) -> BayesResult<f32> {
        let kl = self.kl_divergence()?;
        Ok(elbo(reconstruction_loss, kl, beta))
    }

    /// Sample a single latent vector z ~ q(z).
    ///
    /// # Errors
    /// Returns any error from the Gaussian sampler.
    pub fn sample(&self, rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        sample_gaussian_vec(&self.mu, &self.log_var, rng)
    }

    /// Sample `n` independent latent vectors from q(z).
    ///
    /// # Errors
    /// Returns errors from sampling or `BayesError::InsufficientSamples` if `n == 0`.
    pub fn sample_n(&self, n: usize, rng: &mut LcgRng) -> BayesResult<Vec<Vec<f32>>> {
        if n == 0 {
            return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
        }
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_field_new_valid() {
        let dist = MeanFieldDist::new(vec![0.0; 4], vec![0.0; 4])
            .expect("test invariant: MeanFieldDist::new must succeed");
        assert_eq!(dist.dim(), 4);
    }

    #[test]
    fn mean_field_new_mismatch() {
        assert!(MeanFieldDist::new(vec![0.0; 3], vec![0.0; 4]).is_err());
    }

    #[test]
    fn mean_field_new_empty() {
        assert!(MeanFieldDist::new(vec![], vec![]).is_err());
    }

    #[test]
    fn mean_field_entropy_standard_normal() {
        // For standard normal: H = 0.5*(1 + ln(2π)) per dimension
        let d = 4;
        let dist = MeanFieldDist::new(vec![0.0; d], vec![0.0; d])
            .expect("test invariant: MeanFieldDist::new must succeed");
        let h = dist
            .entropy()
            .expect("test invariant: entropy must succeed");
        let expected = d as f32 * 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln());
        assert!((h - expected).abs() < 1e-4, "expected {expected}, got {h}");
    }

    #[test]
    fn mean_field_kl_standard_normal_zero() {
        let dist = MeanFieldDist::new(vec![0.0; 4], vec![0.0; 4])
            .expect("test invariant: MeanFieldDist::new must succeed");
        let kl = dist
            .kl_divergence()
            .expect("test invariant: kl_divergence must succeed");
        assert!(kl.abs() < 1e-5, "KL(N(0,I)||N(0,I)) must be 0, got {kl}");
    }

    #[test]
    fn mean_field_sample_shape() {
        let mut rng = LcgRng::new(42);
        let dist = MeanFieldDist::new(vec![1.0; 8], vec![0.0; 8])
            .expect("test invariant: MeanFieldDist::new must succeed");
        let s = dist
            .sample(&mut rng)
            .expect("test invariant: sample must succeed");
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn mean_field_sample_n() {
        let mut rng = LcgRng::new(7);
        let dist = MeanFieldDist::new(vec![0.0; 3], vec![0.0; 3])
            .expect("test invariant: MeanFieldDist::new must succeed");
        let samples = dist
            .sample_n(5, &mut rng)
            .expect("test invariant: sample_n must succeed");
        assert_eq!(samples.len(), 5);
        for s in &samples {
            assert_eq!(s.len(), 3);
        }
    }
}
