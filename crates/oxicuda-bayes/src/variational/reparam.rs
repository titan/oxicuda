//! Reparameterization tricks for various distributions.
//!
//! Provides Gaussian, Laplacian reparameterization sampling and log-probability
//! computation, plus a straight-through estimator identity.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Gaussian ────────────────────────────────────────────────────────────────

/// Sample from N(μ, exp(log_var)) using the reparameterization trick.
///
/// `z = μ + σ * ε`, where `σ = exp(0.5 * log_var)` and `ε ~ N(0, 1)`.
/// Uses Box-Muller transform for Gaussian sampling.
#[must_use]
pub fn gaussian_sample(mu: f32, log_var: f32, rng: &mut LcgRng) -> f32 {
    let u1 = (rng.next_f32() + 1e-6_f32).min(1.0 - 1e-7_f32);
    let u2 = rng.next_f32();
    let eps = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    let sigma = (0.5 * log_var).exp();
    mu + sigma * eps
}

/// Log probability of x under N(μ, exp(log_var)).
///
/// Formula: `-0.5 * (((x - μ)/σ)² + ln(2π) + log_var)`.
///
/// # Errors
/// Returns `BayesError::NonPositiveSigma` if `log_var` produces σ ≤ 0,
/// or `BayesError::NanEncountered` if the result is not finite.
pub fn gaussian_log_prob(x: f32, mu: f32, log_var: f32) -> BayesResult<f32> {
    if !log_var.is_finite() {
        return Err(BayesError::NanEncountered {
            location: "gaussian_log_prob: non-finite log_var",
        });
    }
    let sigma_sq = log_var.exp();
    if sigma_sq <= 0.0 {
        return Err(BayesError::NonPositiveSigma);
    }
    let diff = x - mu;
    let log2pi = (2.0 * std::f32::consts::PI).ln();
    let lp = -0.5 * ((diff * diff) / sigma_sq + log2pi + log_var);
    if !lp.is_finite() {
        return Err(BayesError::NanEncountered {
            location: "gaussian_log_prob: non-finite result",
        });
    }
    Ok(lp)
}

// ─── Laplacian ────────────────────────────────────────────────────────────────

/// Sample from Laplace(μ, b) using the inverse CDF method.
///
/// `x = μ - b * sign(u - 0.5) * ln(1 - 2|u - 0.5|)` where `u ~ Uniform(0,1)`.
///
/// # Errors
/// Returns `BayesError::NonPositiveSigma` if `b <= 0`,
/// or `BayesError::NanEncountered` if result is not finite.
pub fn laplacian_sample(mu: f32, b: f32, rng: &mut LcgRng) -> BayesResult<f32> {
    if b <= 0.0 {
        return Err(BayesError::NonPositiveSigma);
    }
    let u = rng.next_f32();
    let u_shifted = u - 0.5;
    let abs_u_shifted = u_shifted.abs();
    // Clamp to avoid log(0)
    let arg = (1.0 - 2.0 * abs_u_shifted).max(1e-10_f32);
    let sign = if u_shifted >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let x = mu - b * sign * arg.ln();
    if !x.is_finite() {
        return Err(BayesError::NanEncountered {
            location: "laplacian_sample: non-finite result",
        });
    }
    Ok(x)
}

/// Log probability of x under Laplace(μ, b).
///
/// Formula: `-ln(2b) - |x - μ| / b`.
///
/// # Errors
/// Returns `BayesError::NonPositiveSigma` if `b <= 0`.
pub fn laplacian_log_prob(x: f32, mu: f32, b: f32) -> BayesResult<f32> {
    if b <= 0.0 {
        return Err(BayesError::NonPositiveSigma);
    }
    let lp = -(2.0_f32 * b).ln() - (x - mu).abs() / b;
    Ok(lp)
}

// ─── Straight-through estimator ──────────────────────────────────────────────

/// Straight-through estimator: forward pass is identity (no rounding in pure Rust).
///
/// In a real autograd system, the backward pass would treat `round(x)` as having
/// gradient 1. Here we return `x` unchanged, representing the identity forward pass.
#[must_use]
#[inline]
pub fn straight_through(x: f32) -> f32 {
    x
}

// ─── Vectorized helpers ───────────────────────────────────────────────────────

/// Sample a vector from N(μ_i, exp(log_var_i)) for each element.
///
/// # Errors
/// Returns `BayesError::DimensionMismatch` if lengths differ,
/// or `BayesError::EmptyInputs` if empty.
pub fn sample_gaussian_vec(mu: &[f32], log_var: &[f32], rng: &mut LcgRng) -> BayesResult<Vec<f32>> {
    if mu.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    if mu.len() != log_var.len() {
        return Err(BayesError::DimensionMismatch {
            expected: mu.len(),
            got: log_var.len(),
        });
    }
    let samples = mu
        .iter()
        .zip(log_var.iter())
        .map(|(&m, &lv)| gaussian_sample(m, lv, rng))
        .collect();
    Ok(samples)
}

/// Sum of log probabilities log p(x_i | μ_i, exp(log_var_i)) over all elements.
///
/// # Errors
/// Returns `BayesError::DimensionMismatch` if lengths differ,
/// or `BayesError::EmptyInputs` if empty.
pub fn log_prob_gaussian_vec(x: &[f32], mu: &[f32], log_var: &[f32]) -> BayesResult<f32> {
    if x.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    if x.len() != mu.len() {
        return Err(BayesError::DimensionMismatch {
            expected: x.len(),
            got: mu.len(),
        });
    }
    if x.len() != log_var.len() {
        return Err(BayesError::DimensionMismatch {
            expected: x.len(),
            got: log_var.len(),
        });
    }
    let mut total = 0.0_f32;
    for ((&xi, &mi), &lvi) in x.iter().zip(mu.iter()).zip(log_var.iter()) {
        total += gaussian_log_prob(xi, mi, lvi)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_sample_finite() {
        let mut rng = LcgRng::new(42);
        let s = gaussian_sample(0.0, 0.0, &mut rng);
        assert!(s.is_finite());
    }

    #[test]
    fn gaussian_log_prob_standard_normal_at_zero() {
        // log p(0 | 0, 1) = -0.5 * ln(2π)
        let expected = -0.5 * (2.0 * std::f32::consts::PI).ln();
        let lp = gaussian_log_prob(0.0, 0.0, 0.0).expect("test invariant: log_prob must succeed");
        assert!(
            (lp - expected).abs() < 1e-4,
            "expected {expected}, got {lp}"
        );
    }

    #[test]
    fn gaussian_log_prob_nonpositive_sigma_error() {
        // log_var = -inf produces sigma_sq = 0 → error
        let res = gaussian_log_prob(0.0, 0.0, f32::NEG_INFINITY);
        assert!(res.is_err());
    }

    #[test]
    fn laplacian_sample_finite() {
        let mut rng = LcgRng::new(7);
        let s = laplacian_sample(0.0, 1.0, &mut rng).expect("test invariant: sample must succeed");
        assert!(s.is_finite());
    }

    #[test]
    fn laplacian_sample_invalid_b() {
        let mut rng = LcgRng::new(1);
        assert!(laplacian_sample(0.0, 0.0, &mut rng).is_err());
        assert!(laplacian_sample(0.0, -1.0, &mut rng).is_err());
    }

    #[test]
    fn laplacian_log_prob_at_mode() {
        // log p(mu | mu, b) = -ln(2b)
        let b = 2.0_f32;
        let expected = -(2.0 * b).ln();
        let lp = laplacian_log_prob(0.0, 0.0, b).expect("test invariant: log_prob must succeed");
        assert!(
            (lp - expected).abs() < 1e-5,
            "expected {expected}, got {lp}"
        );
    }

    #[test]
    fn straight_through_identity() {
        assert!((straight_through(0.7) - 0.7).abs() < 1e-7);
        assert!((straight_through(-1.3) + 1.3).abs() < 1e-7);
    }

    #[test]
    fn sample_gaussian_vec_length() {
        let mut rng = LcgRng::new(99);
        let mu = vec![0.0_f32; 4];
        let lv = vec![0.0_f32; 4];
        let s = sample_gaussian_vec(&mu, &lv, &mut rng)
            .expect("test invariant: sample_gaussian_vec must succeed");
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn log_prob_gaussian_vec_mismatch() {
        assert!(log_prob_gaussian_vec(&[0.0; 3], &[0.0; 2], &[0.0; 3]).is_err());
    }
}
