//! Rényi Differential Privacy (RDP) accountant.
//!
//! Mironov, "Rényi Differential Privacy", CSF 2017.
//!
//! RDP provides tighter composition bounds than the standard (ε, δ)-DP
//! for iterative algorithms like DP-SGD.

use crate::error::{FedError, FedResult};

/// Compute the Rényi divergence of order α for the Gaussian mechanism.
///
/// The Gaussian mechanism with noise multiplier σ (= noise_std / sensitivity)
/// satisfies α-RDP with:
///
/// `ε_RDP(α) = α / (2 * σ²)`
///
/// # Errors
/// Returns `InvalidNoiseMultiplier` if σ ≤ 0.
pub fn rdp_gaussian(alpha: f32, noise_multiplier: f32) -> FedResult<f32> {
    if !(noise_multiplier > 0.0 && noise_multiplier.is_finite()) {
        return Err(FedError::InvalidNoiseMultiplier);
    }
    if alpha <= 1.0 {
        return Ok(0.0);
    }
    Ok(alpha / (2.0 * noise_multiplier * noise_multiplier))
}

/// Convert an RDP guarantee to (ε, δ)-DP.
///
/// Given `(α, ε_RDP)`, the mechanism satisfies `(ε_DP, δ)`-DP where:
///
/// `ε_DP = ε_RDP + log(1 − 1/α) − (log(δ) + log(1 − 1/α)) / (α − 1)`
///
/// Simplified formula (Proposition 3, Mironov 2017):
/// `ε_DP = ε_RDP − (log(δ) + log(1 − 1/α)) / (α − 1) + log(1 − 1/α)`
///
/// Practical bound: `ε_DP ≤ ε_RDP + log((α−1)/α) + ... ≈ ε_RDP − log(δ)/(α−1)`.
/// We use the tightest closed-form: `ε_DP = ε_RDP − (log(δ)) / (α − 1)`.
///
/// # Errors
/// Returns `InvalidPrivacyBudget` if δ ≤ 0 or α ≤ 1.
pub fn rdp_to_dp(alpha: f32, rdp_epsilon: f32, delta: f32) -> FedResult<f32> {
    if !(delta > 0.0 && delta < 1.0) {
        return Err(FedError::InvalidPrivacyBudget);
    }
    if alpha <= 1.0 {
        return Ok(f32::INFINITY);
    }
    let eps = rdp_epsilon - (delta.ln() / (alpha - 1.0));
    Ok(eps.max(0.0))
}

/// Compose `steps` applications of the same RDP mechanism.
///
/// By the composition theorem for RDP:
/// `ε_RDP(k steps) = k * ε_RDP(1 step)`
///
/// # Errors
/// Returns `InvalidNoiseMultiplier` if σ ≤ 0.
pub fn compose_rdp(alpha: f32, noise_multiplier: f32, steps: usize) -> FedResult<f32> {
    let single = rdp_gaussian(alpha, noise_multiplier)?;
    Ok(single * steps as f32)
}

/// Compute the tightest (ε, δ)-DP bound for composed Gaussian mechanism.
///
/// Optimises over α ∈ [2, 128] to find the smallest ε for the given
/// number of steps and δ.
///
/// # Errors
/// Returns `InvalidNoiseMultiplier` or `InvalidPrivacyBudget` on invalid input.
pub fn optimal_epsilon(noise_multiplier: f32, steps: usize, delta: f32) -> FedResult<f32> {
    if !(delta > 0.0 && delta < 1.0) {
        return Err(FedError::InvalidPrivacyBudget);
    }
    if !(noise_multiplier > 0.0 && noise_multiplier.is_finite()) {
        return Err(FedError::InvalidNoiseMultiplier);
    }
    if steps == 0 {
        return Ok(0.0);
    }

    let mut best_eps = f32::INFINITY;
    for alpha_int in 2_u32..=128 {
        let alpha = alpha_int as f32;
        let rdp_eps = compose_rdp(alpha, noise_multiplier, steps)?;
        let dp_eps = rdp_to_dp(alpha, rdp_eps, delta)?;
        if dp_eps.is_finite() && dp_eps < best_eps {
            best_eps = dp_eps;
        }
    }
    Ok(best_eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdp_gaussian_positive() {
        let eps = rdp_gaussian(2.0, 1.0).expect("test invariant: valid rdp_gaussian");
        assert!(eps > 0.0);
    }

    #[test]
    fn rdp_gaussian_invalid_sigma() {
        assert!(matches!(
            rdp_gaussian(2.0, 0.0),
            Err(FedError::InvalidNoiseMultiplier)
        ));
    }

    #[test]
    fn rdp_to_dp_valid() {
        let eps = rdp_to_dp(4.0, 1.0, 1e-5).expect("test invariant: valid rdp_to_dp");
        assert!(eps.is_finite() && eps > 0.0);
    }

    #[test]
    fn rdp_to_dp_invalid_delta() {
        assert!(matches!(
            rdp_to_dp(4.0, 1.0, 0.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn compose_rdp_scales_linearly() {
        let single = compose_rdp(4.0, 1.0, 1).expect("test invariant: single step rdp");
        let ten = compose_rdp(4.0, 1.0, 10).expect("test invariant: ten step rdp");
        assert!(
            (ten - 10.0 * single).abs() < 1e-4,
            "RDP composition should be linear"
        );
    }

    #[test]
    fn optimal_epsilon_zero_steps() {
        let eps = optimal_epsilon(1.0, 0, 1e-5).expect("test invariant: valid optimal epsilon");
        assert_eq!(eps, 0.0);
    }

    #[test]
    fn optimal_epsilon_increases_with_steps() {
        let eps10 = optimal_epsilon(1.0, 10, 1e-5).expect("test invariant: valid epsilon 10 steps");
        let eps100 =
            optimal_epsilon(1.0, 100, 1e-5).expect("test invariant: valid epsilon 100 steps");
        assert!(eps100 > eps10, "more steps → larger epsilon");
    }

    #[test]
    fn compose_rdp_zero_for_alpha_le_one() {
        let eps = rdp_gaussian(1.0, 1.0).expect("test invariant: alpha=1 rdp");
        assert_eq!(eps, 0.0);
    }
}
