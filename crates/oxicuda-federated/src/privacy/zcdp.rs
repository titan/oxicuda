//! Zero-concentrated differential privacy (zCDP) accountant and interop.
//!
//! Bun & Steinke, "Concentrated Differential Privacy: Simplifications,
//! Extensions, and Lower Bounds", TCC 2016.
//!
//! zCDP is a relaxation of differential privacy parameterised by a single
//! number `ρ` that composes additively and converts cleanly to and from Rényi
//! DP. It is the natural accountant for the Gaussian mechanism: a mechanism
//! that adds `N(0, σ²·Δ²)` noise to an `Δ`-sensitive query is `ρ`-zCDP with
//! `ρ = 1 / (2σ²)`.
//!
//! # Relationships used here
//!
//! * **Gaussian mechanism →  zCDP** (Bun-Steinke, Prop. 1.6):
//!   `ρ = 1 / (2σ²)` for noise multiplier `σ = noise_std / sensitivity`.
//! * **zCDP ↔ Rényi DP** (Bun-Steinke, Prop. 1.4): `ρ`-zCDP is exactly
//!   `(α, ρα)`-RDP for *every* order `α > 1`. Hence
//!   `ρ = ε_RDP(α) / α`, and an `α`-RDP guarantee implies `(ε_RDP/α)`-zCDP.
//! * **Composition**: `ρ`-zCDP composes additively —
//!   `k` rounds of `ρ`-zCDP give `kρ`-zCDP.
//! * **zCDP → (ε, δ)-DP** (Bun-Steinke, Prop. 1.3, optimised over the Rényi
//!   order): `ε = ρ + 2·√(ρ · ln(1/δ))`, which is the tightest closed form.
//!
//! All accounting is done in `f64` for numerical headroom and returned as
//! `f32` to match the rest of the crate.

use crate::error::{FedError, FedResult};

/// Convert a Gaussian-mechanism noise multiplier to its zCDP parameter `ρ`.
///
/// For a query of L2 sensitivity `Δ` perturbed with `N(0, (σΔ)²)` noise, the
/// noise multiplier is `σ` and the mechanism is `ρ`-zCDP with
/// `ρ = 1 / (2σ²)`.
///
/// # Errors
/// Returns [`FedError::InvalidNoiseMultiplier`] if `noise_multiplier ≤ 0` or is
/// non-finite.
pub fn zcdp_gaussian(noise_multiplier: f32) -> FedResult<f32> {
    if !(noise_multiplier > 0.0 && noise_multiplier.is_finite()) {
        return Err(FedError::InvalidNoiseMultiplier);
    }
    let sigma = noise_multiplier as f64;
    Ok((1.0 / (2.0 * sigma * sigma)) as f32)
}

/// Convert a Rényi-DP guarantee `(α, ε_RDP)` to a zCDP parameter `ρ`.
///
/// Because `ρ`-zCDP is `(α, ρα)`-RDP for every order, an `α`-RDP bound
/// `ε_RDP` corresponds to `ρ = ε_RDP / α`. Optimising the bound over the order
/// is the caller's responsibility (a Gaussian mechanism gives the same `ρ` at
/// every order, so any `α > 1` is tight).
///
/// # Errors
/// Returns [`FedError::InvalidPrivacyBudget`] if `α ≤ 1` or `ε_RDP < 0`.
pub fn rdp_to_zcdp(alpha: f32, rdp_epsilon: f32) -> FedResult<f32> {
    if alpha <= 1.0 {
        return Err(FedError::InvalidPrivacyBudget);
    }
    if rdp_epsilon < 0.0 || !rdp_epsilon.is_finite() {
        return Err(FedError::InvalidPrivacyBudget);
    }
    Ok((rdp_epsilon as f64 / alpha as f64) as f32)
}

/// Convert a zCDP parameter `ρ` to the Rényi-DP guarantee `ε_RDP(α) = ρ·α`.
///
/// This is the exact `(α, ρα)`-RDP correspondence and lets the zCDP accountant
/// feed [`crate::privacy::rdp::rdp_to_dp`] at any chosen order.
///
/// # Errors
/// Returns [`FedError::InvalidPrivacyBudget`] if `ρ < 0` or `α ≤ 1`.
pub fn zcdp_to_rdp(rho: f32, alpha: f32) -> FedResult<f32> {
    if rho < 0.0 || !rho.is_finite() {
        return Err(FedError::InvalidPrivacyBudget);
    }
    if alpha <= 1.0 {
        return Err(FedError::InvalidPrivacyBudget);
    }
    Ok((rho as f64 * alpha as f64) as f32)
}

/// Convert a zCDP parameter `ρ` to an `(ε, δ)`-DP guarantee.
///
/// Uses the optimised Bun-Steinke bound
/// `ε = ρ + 2·√(ρ · ln(1/δ))`,
/// which already minimises over the Rényi order analytically.
///
/// # Errors
/// Returns [`FedError::InvalidPrivacyBudget`] if `ρ < 0` or `δ ∉ (0, 1)`.
pub fn zcdp_to_dp(rho: f32, delta: f32) -> FedResult<f32> {
    if rho < 0.0 || !rho.is_finite() {
        return Err(FedError::InvalidPrivacyBudget);
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(FedError::InvalidPrivacyBudget);
    }
    let rho = rho as f64;
    if rho == 0.0 {
        return Ok(0.0);
    }
    let ln_inv_delta = (1.0 / delta as f64).ln();
    let eps = rho + 2.0 * (rho * ln_inv_delta).sqrt();
    Ok(eps as f32)
}

/// A zCDP accountant that composes per-round Gaussian privacy losses.
///
/// The privacy loss of `k` identical `ρ₀`-zCDP rounds is `kρ₀`-zCDP; the
/// accountant simply accumulates `ρ`. Conversion to `(ε, δ)`-DP happens on
/// demand via [`ZcdpAccountant::epsilon`].
#[derive(Debug, Clone)]
pub struct ZcdpAccountant {
    /// Accumulated zCDP parameter `ρ` across all composed rounds.
    rho: f64,
    /// Per-round zCDP parameter from the fixed Gaussian noise multiplier.
    rho_per_round: f64,
}

impl ZcdpAccountant {
    /// Build an accountant for a fixed Gaussian-mechanism noise multiplier.
    ///
    /// Each composed round contributes `ρ₀ = 1 / (2σ²)`.
    ///
    /// # Errors
    /// Returns [`FedError::InvalidNoiseMultiplier`] if `noise_multiplier ≤ 0`
    /// or is non-finite.
    pub fn from_gaussian(noise_multiplier: f32) -> FedResult<Self> {
        let rho_per_round = zcdp_gaussian(noise_multiplier)? as f64;
        Ok(Self {
            rho: 0.0,
            rho_per_round,
        })
    }

    /// Compose `n_rounds` applications of the fixed per-round mechanism.
    pub fn compose(&mut self, n_rounds: usize) {
        self.rho += self.rho_per_round * n_rounds as f64;
    }

    /// Directly add an externally-computed `ρ` budget (heterogeneous rounds).
    ///
    /// # Errors
    /// Returns [`FedError::InvalidPrivacyBudget`] if `rho < 0` or is non-finite.
    pub fn add_rho(&mut self, rho: f32) -> FedResult<()> {
        if rho < 0.0 || !rho.is_finite() {
            return Err(FedError::InvalidPrivacyBudget);
        }
        self.rho += rho as f64;
        Ok(())
    }

    /// Current accumulated zCDP parameter `ρ`.
    #[must_use]
    pub fn rho(&self) -> f32 {
        self.rho as f32
    }

    /// Convert the accumulated budget to `(ε, δ)`-DP at the target `δ`.
    ///
    /// # Errors
    /// Returns [`FedError::InvalidPrivacyBudget`] if `δ ∉ (0, 1)`.
    pub fn epsilon(&self, delta: f32) -> FedResult<f32> {
        zcdp_to_dp(self.rho as f32, delta)
    }

    /// Reset the accumulated budget to zero (keeps the per-round rate).
    pub fn reset(&mut self) {
        self.rho = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::rdp::{rdp_gaussian, rdp_to_dp};

    #[test]
    fn zcdp_gaussian_matches_closed_form() {
        // σ = 1 → ρ = 1/2.
        let rho = zcdp_gaussian(1.0).expect("valid sigma");
        assert!((rho - 0.5).abs() < 1e-6);
        // σ = 2 → ρ = 1/8.
        let rho2 = zcdp_gaussian(2.0).expect("valid sigma");
        assert!((rho2 - 0.125).abs() < 1e-6);
    }

    #[test]
    fn zcdp_gaussian_rejects_bad_sigma() {
        assert!(matches!(
            zcdp_gaussian(0.0),
            Err(FedError::InvalidNoiseMultiplier)
        ));
        assert!(matches!(
            zcdp_gaussian(-1.0),
            Err(FedError::InvalidNoiseMultiplier)
        ));
    }

    #[test]
    fn rdp_and_zcdp_are_mutually_consistent() {
        // For the Gaussian mechanism, ε_RDP(α) = α/(2σ²) and ρ = 1/(2σ²),
        // so rdp_to_zcdp(α, ε_RDP) must recover exactly ρ for every order.
        let sigma = 1.3_f32;
        let rho_direct = zcdp_gaussian(sigma).expect("rho direct");
        for &alpha in &[2.0_f32, 4.0, 8.0, 32.0] {
            let rdp_eps = rdp_gaussian(alpha, sigma).expect("rdp");
            let rho_from_rdp = rdp_to_zcdp(alpha, rdp_eps).expect("rdp_to_zcdp");
            assert!(
                (rho_from_rdp - rho_direct).abs() < 1e-5,
                "α={alpha}: rho_from_rdp={rho_from_rdp}, rho_direct={rho_direct}"
            );
            // Round-trip back to RDP.
            let rdp_back = zcdp_to_rdp(rho_direct, alpha).expect("zcdp_to_rdp");
            assert!((rdp_back - rdp_eps).abs() < 1e-4, "α={alpha} round-trip");
        }
    }

    #[test]
    fn zcdp_to_dp_is_positive_and_grows_with_rho() {
        let delta = 1e-5;
        let e1 = zcdp_to_dp(0.1, delta).expect("eps 0.1");
        let e2 = zcdp_to_dp(0.5, delta).expect("eps 0.5");
        assert!(e1 > 0.0 && e2 > 0.0);
        assert!(e2 > e1, "larger ρ → larger ε");
        // ρ = 0 → ε = 0.
        assert_eq!(zcdp_to_dp(0.0, delta).expect("zero rho"), 0.0);
    }

    #[test]
    fn zcdp_to_dp_rejects_bad_delta() {
        assert!(matches!(
            zcdp_to_dp(0.1, 0.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
        assert!(matches!(
            zcdp_to_dp(0.1, 1.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn zcdp_dp_bound_is_no_looser_than_rdp_at_fixed_order() {
        // The optimised zCDP→DP bound minimises over the order, so it must be
        // at least as tight as the RDP→DP conversion at any single fixed α.
        let sigma = 1.0_f32;
        let delta = 1e-5_f32;
        let steps = 50_usize;
        let rho = zcdp_gaussian(sigma).expect("rho") * steps as f32;
        let eps_zcdp = zcdp_to_dp(rho, delta).expect("zcdp eps");

        // RDP path at a single representative order α = 16.
        let alpha = 16.0_f32;
        let rdp_eps = rdp_gaussian(alpha, sigma).expect("rdp") * steps as f32;
        let eps_rdp_fixed = rdp_to_dp(alpha, rdp_eps, delta).expect("rdp_to_dp");

        assert!(
            eps_zcdp <= eps_rdp_fixed + 1e-3,
            "zCDP optimised bound {eps_zcdp} should not exceed fixed-order RDP {eps_rdp_fixed}"
        );
    }

    #[test]
    fn accountant_composes_additively() {
        let mut acc = ZcdpAccountant::from_gaussian(1.0).expect("accountant");
        acc.compose(10);
        let rho10 = acc.rho();
        // 10 rounds of ρ₀ = 0.5 → 5.0.
        assert!((rho10 - 5.0).abs() < 1e-5);
        acc.compose(10);
        assert!((acc.rho() - 10.0).abs() < 1e-5);
        // Epsilon must grow with composition.
        let eps10 = acc.epsilon(1e-5).expect("eps10");
        assert!(eps10 > 0.0);
    }

    #[test]
    fn accountant_epsilon_increases_with_rounds() {
        let mut a = ZcdpAccountant::from_gaussian(2.0).expect("a");
        a.compose(5);
        let e5 = a.epsilon(1e-6).expect("e5");
        a.compose(5);
        let e10 = a.epsilon(1e-6).expect("e10");
        assert!(e10 > e5, "more rounds → larger ε");
    }

    #[test]
    fn accountant_add_rho_and_reset() {
        let mut a = ZcdpAccountant::from_gaussian(1.0).expect("a");
        a.add_rho(0.25).expect("add");
        assert!((a.rho() - 0.25).abs() < 1e-6);
        a.reset();
        assert_eq!(a.rho(), 0.0);
        assert!(matches!(
            a.add_rho(-1.0),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }

    #[test]
    fn rdp_to_zcdp_rejects_bad_input() {
        assert!(matches!(
            rdp_to_zcdp(1.0, 0.5),
            Err(FedError::InvalidPrivacyBudget)
        ));
        assert!(matches!(
            rdp_to_zcdp(4.0, -0.5),
            Err(FedError::InvalidPrivacyBudget)
        ));
    }
}
