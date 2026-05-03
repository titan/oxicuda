//! ELBO (Evidence Lower Bound) computation for variational inference.
//!
//! Provides ELBO, KL divergence, and Importance-Weighted ELBO (IWAE).

use crate::error::{BayesError, BayesResult};

/// Configuration for ELBO computation.
#[derive(Debug, Clone)]
pub struct ElboConfig {
    /// Number of Monte Carlo samples for expectation estimation.
    pub n_samples: usize,
    /// KL weight β: β > 1 for disentanglement (β-VAE); 1.0 for standard ELBO.
    pub beta: f32,
}

impl ElboConfig {
    /// Create a standard ELBO config (β = 1.0).
    #[must_use]
    pub fn standard(n_samples: usize) -> Self {
        Self {
            n_samples,
            beta: 1.0,
        }
    }

    /// Create a β-VAE config.
    #[must_use]
    pub fn beta_vae(n_samples: usize, beta: f32) -> Self {
        Self { n_samples, beta }
    }
}

/// Gaussian KL divergence: KL(N(μ, σ²) ‖ N(0, 1)) for a single element.
///
/// Formula: `0.5 * (μ² + σ² - 1 - ln(σ²))` where `σ² = exp(log_var)`.
///
/// # Errors
/// Returns `BayesError::NanEncountered` if inputs produce NaN.
pub fn kl_gaussian(mu: f32, log_var: f32) -> BayesResult<f32> {
    if !mu.is_finite() || !log_var.is_finite() {
        return Err(BayesError::NanEncountered {
            location: "kl_gaussian: non-finite input",
        });
    }
    // σ² = exp(log_var)
    let sigma_sq = log_var.exp();
    // KL = 0.5 * (μ² + σ² - 1 - log_var)  [since ln(σ²) = log_var]
    let kl = 0.5 * (mu * mu + sigma_sq - 1.0 - log_var);
    if !kl.is_finite() {
        return Err(BayesError::NanEncountered {
            location: "kl_gaussian: non-finite output",
        });
    }
    Ok(kl.max(0.0))
}

/// Gaussian KL divergence summed over a vector of (μ, log_var) pairs.
///
/// # Errors
/// Returns `BayesError::DimensionMismatch` if lengths differ, or
/// `BayesError::EmptyInputs` if inputs are empty.
pub fn kl_gaussian_vec(mu: &[f32], log_var: &[f32]) -> BayesResult<f32> {
    if mu.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    if mu.len() != log_var.len() {
        return Err(BayesError::DimensionMismatch {
            expected: mu.len(),
            got: log_var.len(),
        });
    }
    let mut total = 0.0_f32;
    for (&m, &lv) in mu.iter().zip(log_var.iter()) {
        total += kl_gaussian(m, lv)?;
    }
    Ok(total)
}

/// Compute ELBO = E[log p(x|z)] - β * KL(q(z) ‖ p(z)).
///
/// `reconstruction_loss` is the negative log-likelihood (positive scalar).
/// ELBO = -reconstruction_loss - β * kl.
#[must_use]
pub fn elbo(reconstruction_loss: f32, kl: f32, beta: f32) -> f32 {
    -reconstruction_loss - beta * kl
}

/// Importance-Weighted ELBO (IWAE): `log(1/K * Σ_k p(x|z_k)/q(z_k|x) * p(z_k))`.
///
/// Using the log-sum-exp trick:
/// `IWAE = log_sum_exp(log_weights) - log(K)`
/// where `log_weights[k] = log_p_xz[k] + log_p_z[k] - log_q_z[k]`.
///
/// # Errors
/// Returns `BayesError::InsufficientSamples` if fewer than 1 sample provided,
/// or `BayesError::DimensionMismatch` if arrays have different lengths.
pub fn iwae(log_likelihoods: &[f32], log_q: &[f32], log_p: &[f32]) -> BayesResult<f32> {
    let k = log_likelihoods.len();
    if k == 0 {
        return Err(BayesError::InsufficientSamples { min: 1, got: 0 });
    }
    if log_q.len() != k {
        return Err(BayesError::DimensionMismatch {
            expected: k,
            got: log_q.len(),
        });
    }
    if log_p.len() != k {
        return Err(BayesError::DimensionMismatch {
            expected: k,
            got: log_p.len(),
        });
    }

    // log_weight[k] = log_likelihood[k] + log_p[k] - log_q[k]
    let log_weights: Vec<f32> = (0..k)
        .map(|i| log_likelihoods[i] + log_p[i] - log_q[i])
        .collect();

    // Numerically stable log-sum-exp
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_lw.is_finite() {
        return Err(BayesError::NanEncountered {
            location: "iwae: non-finite log_weights",
        });
    }

    let sum_exp: f32 = log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();
    let lse = max_lw + sum_exp.ln();

    // IWAE = lse - log(K)
    let iwae_val = lse - (k as f32).ln();
    Ok(iwae_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_standard_normal_is_zero() {
        // KL(N(0,1) || N(0,1)) = 0
        let kl = kl_gaussian(0.0, 0.0).expect("test invariant: kl_gaussian(0,0) must succeed");
        assert!(kl.abs() < 1e-6, "expected 0, got {kl}");
    }

    #[test]
    fn kl_gaussian_nonnegative() {
        let kl = kl_gaussian(1.0, 0.5).expect("test invariant: kl must succeed");
        assert!(kl >= 0.0, "KL must be non-negative, got {kl}");
    }

    #[test]
    fn elbo_config_standard() {
        let cfg = ElboConfig::standard(10);
        assert_eq!(cfg.n_samples, 10);
        assert!((cfg.beta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn elbo_config_beta_vae() {
        let cfg = ElboConfig::beta_vae(5, 4.0);
        assert!((cfg.beta - 4.0).abs() < 1e-6);
    }

    #[test]
    fn elbo_computation() {
        // With recon=1.0, kl=0.5, beta=1.0: elbo = -1.0 - 0.5 = -1.5
        let val = elbo(1.0, 0.5, 1.0);
        assert!((val + 1.5).abs() < 1e-6, "expected -1.5, got {val}");
    }

    #[test]
    fn kl_gaussian_vec_empty() {
        let res = kl_gaussian_vec(&[], &[]);
        assert!(res.is_err());
    }

    #[test]
    fn kl_gaussian_vec_mismatch() {
        let res = kl_gaussian_vec(&[0.0; 3], &[0.0; 2]);
        assert!(res.is_err());
    }

    #[test]
    fn iwae_insufficient_samples() {
        let res = iwae(&[], &[], &[]);
        assert!(res.is_err());
    }

    #[test]
    fn iwae_single_sample_equals_elbo_approx() {
        // With 1 sample: IWAE = log_p(x|z) + log_p(z) - log_q(z)
        let ll = &[-0.5_f32];
        let lq = &[-0.3_f32];
        let lp = &[-0.2_f32];
        let val = iwae(ll, lq, lp).expect("test invariant: iwae must succeed");
        let expected = -0.5 + (-0.2) - (-0.3);
        assert!(
            (val - expected).abs() < 1e-5,
            "expected {expected}, got {val}"
        );
    }
}
