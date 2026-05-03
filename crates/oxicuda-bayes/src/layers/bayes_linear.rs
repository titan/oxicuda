//! Bayesian linear layer via Bayes-by-Backprop (BBB).
//!
//! Weight distribution: q(W) = N(W_mu, softplus(W_rho)²).
//! Prior: p(W) = N(0, prior_sigma²).

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;
use crate::variational::reparam::gaussian_sample;

/// Numerically stable softplus: `ln(1 + exp(x))`.
///
/// For `x > 20`, returns `x` to avoid overflow.
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Bayesian linear layer using Bayes-by-Backprop (BBB).
///
/// Parameters:
/// - `w_mu`, `w_rho`: weight mean and rho (σ = softplus(ρ)), shape `[out × in]`
/// - `b_mu`, `b_rho`: bias mean and rho, shape `[out]`
///
/// Prior: p(W) = N(0, prior_sigma²) for all weights.
#[derive(Debug, Clone)]
pub struct BayesLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Weight means `[out × in]`.
    pub w_mu: Vec<f32>,
    /// Weight rho (σ = softplus(ρ)) `[out × in]`.
    pub w_rho: Vec<f32>,
    /// Bias means `[out]`.
    pub b_mu: Vec<f32>,
    /// Bias rho `[out]`.
    pub b_rho: Vec<f32>,
    /// Prior standard deviation.
    pub prior_sigma: f32,
}

impl BayesLinear {
    /// Create a new BayesLinear layer.
    ///
    /// Initialization: w_mu ~ N(0, 0.1), w_rho = -3.0 (small initial sigma).
    ///
    /// # Errors
    /// Returns `BayesError::InvalidPriorVariance` if `prior_sigma <= 0` or non-finite.
    pub fn new(
        in_features: usize,
        out_features: usize,
        prior_sigma: f32,
        rng: &mut LcgRng,
    ) -> BayesResult<Self> {
        if prior_sigma <= 0.0 || !prior_sigma.is_finite() {
            return Err(BayesError::InvalidPriorVariance);
        }
        let n_weights = in_features * out_features;
        let mut w_mu = vec![0.0_f32; n_weights];
        rng.fill_normal(&mut w_mu);
        for v in w_mu.iter_mut() {
            *v *= 0.1;
        }
        let w_rho = vec![-3.0_f32; n_weights];
        let mut b_mu = vec![0.0_f32; out_features];
        rng.fill_normal(&mut b_mu);
        for v in b_mu.iter_mut() {
            *v *= 0.01;
        }
        let b_rho = vec![-3.0_f32; out_features];
        Ok(Self {
            in_features,
            out_features,
            w_mu,
            w_rho,
            b_mu,
            b_rho,
            prior_sigma,
        })
    }

    /// Sample weights and compute a stochastic forward pass.
    ///
    /// `w_sample[i] = w_mu[i] + softplus(w_rho[i]) * ε_i`, `ε_i ~ N(0,1)`.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if `x.len() != in_features`.
    pub fn forward_sample(&self, x: &[f32], rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(BayesError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = self.b_mu.clone();
        // Sample biases
        for (o, b_r) in out.iter_mut().zip(self.b_rho.iter()) {
            let sigma = softplus(*b_r);
            let log_var = (sigma * sigma).ln();
            *o += gaussian_sample(0.0, log_var, rng);
        }
        // Matrix-vector multiply with sampled weights
        for (oc, o) in out.iter_mut().enumerate() {
            for (ic, &xi) in x.iter().enumerate() {
                let idx = oc * self.in_features + ic;
                let sigma = softplus(self.w_rho[idx]);
                let log_var = (sigma * sigma).ln();
                let w_sample = gaussian_sample(self.w_mu[idx], log_var, rng);
                *o += w_sample * xi;
            }
        }
        Ok(out)
    }

    /// Deterministic forward pass using mean weights (for inference).
    ///
    /// `out = W_mu @ x + b_mu`.
    ///
    /// # Errors
    /// Returns `BayesError::DimensionMismatch` if `x.len() != in_features`.
    pub fn forward_mean(&self, x: &[f32]) -> BayesResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(BayesError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = self.b_mu.clone();
        for (oc, o) in out.iter_mut().enumerate() {
            for (ic, &xi) in x.iter().enumerate() {
                *o += self.w_mu[oc * self.in_features + ic] * xi;
            }
        }
        Ok(out)
    }

    /// KL divergence KL(q(W) ‖ p(W)) summed over all weights and biases.
    ///
    /// For each parameter w_i:
    /// `KL(N(μ, σ²) ‖ N(0, σ_p²)) = log(σ_p/σ) + (σ² + μ²)/(2σ_p²) - 0.5`
    ///
    /// # Errors
    /// Returns `BayesError::NanEncountered` if any parameter is non-finite.
    pub fn kl_divergence(&self) -> BayesResult<f32> {
        let prior_var = self.prior_sigma * self.prior_sigma;
        let log_prior_sigma = self.prior_sigma.ln();

        let mut kl = 0.0_f32;

        // Weights
        for (&mu, &rho) in self.w_mu.iter().zip(self.w_rho.iter()) {
            let sigma = softplus(rho);
            let sigma_sq = sigma * sigma;
            let log_sigma = sigma.ln();
            // KL = log(σ_p/σ) + (σ² + μ²)/(2σ_p²) - 0.5
            kl += log_prior_sigma - log_sigma + (sigma_sq + mu * mu) / (2.0 * prior_var) - 0.5;
        }

        // Biases
        for (&mu, &rho) in self.b_mu.iter().zip(self.b_rho.iter()) {
            let sigma = softplus(rho);
            let sigma_sq = sigma * sigma;
            let log_sigma = sigma.ln();
            kl += log_prior_sigma - log_sigma + (sigma_sq + mu * mu) / (2.0 * prior_var) - 0.5;
        }

        if !kl.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "BayesLinear::kl_divergence: non-finite result",
            });
        }
        Ok(kl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayes_linear_new_valid() {
        let mut rng = LcgRng::new(42);
        let layer = BayesLinear::new(4, 2, 1.0, &mut rng)
            .expect("test invariant: BayesLinear::new must succeed");
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 2);
        assert_eq!(layer.w_mu.len(), 8);
    }

    #[test]
    fn bayes_linear_invalid_prior() {
        let mut rng = LcgRng::new(1);
        assert!(BayesLinear::new(4, 2, 0.0, &mut rng).is_err());
        assert!(BayesLinear::new(4, 2, -1.0, &mut rng).is_err());
    }

    #[test]
    fn forward_mean_shape() {
        let mut rng = LcgRng::new(5);
        let layer = BayesLinear::new(4, 3, 1.0, &mut rng)
            .expect("test invariant: BayesLinear::new must succeed");
        let x = vec![1.0_f32; 4];
        let out = layer
            .forward_mean(&x)
            .expect("test invariant: forward_mean must succeed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn forward_sample_shape() {
        let mut rng = LcgRng::new(9);
        let layer = BayesLinear::new(4, 3, 1.0, &mut rng)
            .expect("test invariant: BayesLinear::new must succeed");
        let x = vec![0.5_f32; 4];
        let out = layer
            .forward_sample(&x, &mut rng)
            .expect("test invariant: forward_sample must succeed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn kl_divergence_positive() {
        let mut rng = LcgRng::new(77);
        let layer = BayesLinear::new(3, 2, 1.0, &mut rng)
            .expect("test invariant: BayesLinear::new must succeed");
        let kl = layer
            .kl_divergence()
            .expect("test invariant: kl_divergence must succeed");
        assert!(kl >= 0.0, "KL must be non-negative, got {kl}");
    }

    #[test]
    fn forward_mean_dim_mismatch() {
        let mut rng = LcgRng::new(3);
        let layer = BayesLinear::new(4, 2, 1.0, &mut rng)
            .expect("test invariant: BayesLinear::new must succeed");
        assert!(layer.forward_mean(&[1.0; 3]).is_err());
    }

    #[test]
    fn softplus_stable_large() {
        assert!((softplus(100.0) - 100.0).abs() < 0.01);
    }
}
