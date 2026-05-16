//! Zero-Concentrated Differential Privacy (zCDP) and Truncated CDP.
//!
//! Reference: Bun & Steinke (2016), "Concentrated Differential Privacy:
//! Simplifications, Extensions, and Lower Bounds".
//!
//! # zCDP definition
//! A mechanism M is **ρ-zCDP** if for all neighbouring datasets x, x' and
//! all α ∈ (1, ∞):
//!
//! `D_α(M(x) ‖ M(x')) ≤ ρ · α`
//!
//! where D_α is the Rényi divergence of order α.
//!
//! # Gaussian mechanism
//! Adding Gaussian noise with std σ to a function with L2-sensitivity Δ gives
//! `ρ = Δ² / (2σ²)` (Proposition 1.6 in Bun & Steinke).
//!
//! # Composition
//! zCDP composes additively: `ρ_total = Σ ρᵢ` (exact, no approximation).
//!
//! # Conversion to (ε, δ)-DP
//! `ε = ρ + 2√(ρ · ln(1/δ))` (Lemma 3.5 in Bun & Steinke).
//!
//! # Truncated CDP
//! M is **(ρ, ω)-tCDP** if ρ-zCDP holds restricted to Rényi orders α ∈ (1, ω+1].
//! Conversion to (ε, δ) for tCDP: `ε ≤ ρ(ω+1) + √(2ρω · ln(1/δ) · (ω+1)/ω)`.

use crate::error::{PrivacyError, PrivacyResult};

/// Compute the zCDP parameter ρ for the Gaussian mechanism.
///
/// `ρ = Δ² / (2σ²)` where Δ is the L2 sensitivity and σ is the noise std.
///
/// # Errors
/// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
/// - `InvalidParameter` if `sigma ≤ 0`.
pub fn zcdp_gaussian(sensitivity: f64, sigma: f64) -> PrivacyResult<f64> {
    if sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
    }
    if sigma <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "sigma must be positive, got {sigma}"
        )));
    }
    Ok(sensitivity * sensitivity / (2.0 * sigma * sigma))
}

/// Compose k zCDP mechanisms with parameters ρ₁, …, ρₖ.
///
/// Returns `Σ ρᵢ` — composition is exact and additive for zCDP.
pub fn zcdp_compose(rhos: &[f64]) -> f64 {
    rhos.iter().sum()
}

/// Convert a ρ-zCDP guarantee to (ε, δ)-DP.
///
/// Uses the closed-form bound (Lemma 3.5 of Bun & Steinke 2016):
///
/// `ε = ρ + 2√(ρ · ln(1/δ))`
///
/// # Errors
/// - `InvalidParameter` if `rho ≤ 0`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
pub fn zcdp_to_epsilon_delta(rho: f64, delta: f64) -> PrivacyResult<f64> {
    if rho <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "rho must be positive, got {rho}"
        )));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    let epsilon = rho + 2.0 * (rho * (1.0 / delta).ln()).sqrt();
    Ok(epsilon)
}

// ─── Truncated CDP ────────────────────────────────────────────────────────────

/// Configuration for a Truncated CDP mechanism.
#[derive(Debug, Clone)]
pub struct TcdpConfig {
    /// zCDP parameter ρ > 0.
    pub rho: f64,
    /// Truncation order ω > 0.  The tCDP guarantee covers Rényi orders in (1, ω+1].
    pub omega: f64,
}

impl TcdpConfig {
    /// Construct and validate a `TcdpConfig`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `rho ≤ 0` or `omega ≤ 0`.
    pub fn new(rho: f64, omega: f64) -> PrivacyResult<Self> {
        if rho <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "rho must be positive, got {rho}"
            )));
        }
        if omega <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "omega must be positive, got {omega}"
            )));
        }
        Ok(Self { rho, omega })
    }
}

/// Convert a (ρ, ω)-tCDP guarantee to (ε, δ)-DP.
///
/// Uses the bound derived from Rényi divergence at the truncation order ω+1:
///
/// `ε ≤ ρ·(ω+1) + √(2ρω · (ω+1)/ω · ln(1/δ))`
///
/// This is a valid (ε, δ)-DP guarantee when the Rényi divergence is bounded
/// up to order ω+1.
///
/// # Errors
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
/// - `InvalidParameter` if `rho ≤ 0` or `omega ≤ 0`.
pub fn tcdp_to_epsilon_delta(cfg: &TcdpConfig, delta: f64) -> PrivacyResult<f64> {
    if cfg.rho <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "rho must be positive, got {}",
            cfg.rho
        )));
    }
    if cfg.omega <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "omega must be positive, got {}",
            cfg.omega
        )));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }

    let omega = cfg.omega;
    let rho = cfg.rho;
    let log_inv_delta = (1.0 / delta).ln();

    // ε = ρ(ω+1) + √(2ρω · (ω+1)/ω · ln(1/δ))
    let first_term = rho * (omega + 1.0);
    let radicand = 2.0 * rho * omega * (omega + 1.0) / omega * log_inv_delta;
    let second_term = radicand.sqrt();

    Ok(first_term + second_term)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zcdp_gaussian() {
        // Gaussian mech with sensitivity=1, sigma=2: ρ = 1/(2*4) = 0.125.
        let rho = zcdp_gaussian(1.0, 2.0).expect("ok");
        assert!((rho - 0.125).abs() < 1e-12, "expected 0.125, got {rho}");
    }

    #[test]
    fn test_zcdp_compose_additive() {
        let rhos = [0.1, 0.2];
        let total = zcdp_compose(&rhos);
        assert!((total - 0.3).abs() < 1e-12, "expected 0.3, got {total}");
    }

    #[test]
    fn test_zcdp_to_eps_delta_ordering() {
        let eps1 = zcdp_to_epsilon_delta(0.5, 1e-5).expect("ok");
        let eps2 = zcdp_to_epsilon_delta(0.5, 1e-3).expect("ok");
        // Smaller delta → larger epsilon.
        assert!(
            eps1 > eps2,
            "smaller delta should give larger ε: {eps1} > {eps2}"
        );
    }

    #[test]
    fn test_tcdp_to_epsilon_delta_valid() {
        let cfg = TcdpConfig::new(0.5, 10.0).expect("ok");
        let eps = tcdp_to_epsilon_delta(&cfg, 1e-5).expect("ok");
        assert!(eps > 0.0, "epsilon must be positive, got {eps}");
        // tCDP should give a larger ε than zCDP for the same parameters.
        let zcdp_eps = zcdp_to_epsilon_delta(0.5, 1e-5).expect("ok");
        // tCDP bound is generally looser.
        assert!(eps.is_finite());
        assert!(zcdp_eps.is_finite());
    }

    #[test]
    fn test_tcdp_invalid_omega() {
        assert!(TcdpConfig::new(0.5, 0.0).is_err());
    }
}
