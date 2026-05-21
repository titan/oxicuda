//! Full Truncated Concentrated Differential Privacy (tCDP).
//!
//! References:
//! - Bun M, Steinke T (2016) "Concentrated Differential Privacy: Simplifications, Extensions,
//!   and Lower Bounds", TCC 2016-B, LNCS 9985:635–658.
//! - Mironov I (2017) "Rényi Differential Privacy of the Gaussian Mechanism", IEEE CSF 2017.
//!
//! # Definition
//! A mechanism M is **(ρ, ω)-tCDP** if for all neighbouring datasets x, x' and all
//! α ∈ (1, ω]:
//!
//! `D_α(M(x) ‖ M(x')) ≤ ρ · α`
//!
//! # Conversion to (ε, δ)-DP
//! Minimize `f(α) = ρ·α + ln(1/δ)/(α − 1)` over α ∈ (1, ω].
//!
//! Interior critical point: `α* = 1 + √(ln(1/δ)/ρ)`.
//!
//! - If ω = ∞ or α* ≤ ω: `ε = ρ + 2√(ρ·ln(1/δ))` (recovers zCDP formula).
//! - If α* > ω (boundary applies): `ε = ρ·ω + ln(1/δ)/(ω − 1)`.

use crate::error::{PrivacyError, PrivacyResult};

// ─── TcdpMechanism ───────────────────────────────────────────────────────────

/// A (ρ, ω)-tCDP mechanism with ω ∈ (0, ∞].
///
/// Set `omega = f64::INFINITY` for pure zCDP behaviour (Gaussian mechanism).
/// The tCDP guarantee asserts that Rényi divergence `D_α ≤ ρ·α` for all α ∈ (1, ω].
#[derive(Clone, Debug)]
pub struct TcdpMechanism {
    /// Rényi divergence parameter ρ > 0.
    pub rho: f64,
    /// Truncation order ω > 0 (may be `f64::INFINITY` for full zCDP).
    pub omega: f64,
}

impl TcdpMechanism {
    /// Gaussian mechanism with L2 sensitivity Δ and noise std σ.
    ///
    /// `ρ = Δ²/(2σ²)`, ω = ∞ (Gaussian satisfies zCDP for all α).
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
    /// - `InvalidParameter` if `sigma ≤ 0`.
    pub fn gaussian(sensitivity: f64, sigma: f64) -> PrivacyResult<Self> {
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        Ok(Self {
            rho: sensitivity * sensitivity / (2.0 * sigma * sigma),
            omega: f64::INFINITY,
        })
    }

    /// Laplace mechanism with L2 sensitivity Δ and scale b = Δ/ε₀.
    ///
    /// Uses the upper bound: treats as ε₀-DP via ρ = ε₀²/2, ω = ∞.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
    /// - `InvalidParameter` if `scale ≤ 0`.
    pub fn laplace(sensitivity: f64, scale: f64) -> PrivacyResult<Self> {
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if scale <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "scale must be positive, got {scale}"
            )));
        }
        let eps_0 = sensitivity / scale;
        Ok(Self {
            rho: eps_0 * eps_0 / 2.0,
            omega: f64::INFINITY,
        })
    }

    /// Convert an ε₀-DP guarantee to tCDP via Bun-Steinke Proposition 1.3.
    ///
    /// `ρ = ε₀·(e^ε₀ − 1) / 2`, ω = ∞.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
    pub fn from_pure_dp(epsilon: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        let rho = epsilon * (epsilon.exp() - 1.0) / 2.0;
        Ok(Self {
            rho,
            omega: f64::INFINITY,
        })
    }

    /// Optimal α for the given δ, minimizing `ρ·α + ln(1/δ)/(α−1)` over α ∈ (1, ω].
    ///
    /// # Errors
    /// - `InvalidParameter` if `rho ≤ 0` or `omega ≤ 0`, or finite `omega ≤ 1`.
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn optimal_alpha(&self, delta: f64) -> PrivacyResult<f64> {
        Self::validate_mechanism_params(self.rho, self.omega)?;
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        let log_inv_d = (1.0 / delta).ln();
        let alpha_star = 1.0 + (log_inv_d / self.rho).sqrt();
        if self.omega.is_infinite() {
            Ok(alpha_star)
        } else {
            Ok(alpha_star.min(self.omega))
        }
    }

    /// Optimal (ε, δ) conversion via minimization of `ρ·α + ln(1/δ)/(α−1)` over α ∈ (1, ω].
    ///
    /// - Interior optimum (ω = ∞ or α* ≤ ω): `ε = ρ + 2√(ρ·ln(1/δ))`.
    /// - Boundary (α* > ω): `ε = ρ·ω + ln(1/δ)/(ω − 1)`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `rho ≤ 0` or `omega ≤ 0`, or finite `omega ≤ 1`.
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn to_epsilon_delta(&self, delta: f64) -> PrivacyResult<f64> {
        Self::validate_mechanism_params(self.rho, self.omega)?;
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        let log_inv_d = (1.0 / delta).ln();
        let alpha_star = 1.0 + (log_inv_d / self.rho).sqrt();

        let eps = if self.omega.is_infinite() || alpha_star <= self.omega {
            // Interior optimum: recovers zCDP formula.
            self.rho + 2.0 * (self.rho * log_inv_d).sqrt()
        } else {
            // Boundary: α* > ω, evaluate at α = ω.
            self.rho * self.omega + log_inv_d / (self.omega - 1.0)
        };
        Ok(eps)
    }

    /// δ(ε): find δ ∈ (0,1) such that `to_epsilon_delta(δ) = ε_target`.
    ///
    /// Uses geometric-midpoint bisection over 80 iterations (log-scale search).
    ///
    /// # Errors
    /// - `InvalidParameter` if `epsilon_target ≤ 0`.
    /// - `InvalidParameter` if `epsilon_target` is smaller than achievable.
    /// - Propagates validation errors from `to_epsilon_delta`.
    pub fn to_delta_epsilon(&self, epsilon_target: f64) -> PrivacyResult<f64> {
        if epsilon_target <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_target must be positive, got {epsilon_target}"
            )));
        }
        Self::validate_mechanism_params(self.rho, self.omega)?;

        // As δ → 1⁻, ln(1/δ) → 0, so ε → ρ (minimum achievable ε).
        // As δ → 0⁺, ln(1/δ) → ∞, so ε → ∞.
        // We bisect: if to_epsilon_delta(lo) gives eps > eps_target, lo is too small.
        let lo_init = 1e-300_f64;
        let hi_init = 1.0 - 1e-15_f64;

        // Check feasibility: maximum epsilon at lo (smallest delta) should exceed target.
        let eps_at_lo = self.to_epsilon_delta(lo_init)?;
        if eps_at_lo < epsilon_target {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_target {epsilon_target:.6} is too large to invert (eps at delta=1e-300 is {eps_at_lo:.6})"
            )));
        }

        // Check: at hi (delta near 1), epsilon should be small. If it exceeds target, no solution.
        let eps_at_hi = self.to_epsilon_delta(hi_init)?;
        if eps_at_hi >= epsilon_target {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_target {epsilon_target:.6} is too small to invert (min epsilon is approximately {eps_at_hi:.6})"
            )));
        }

        // Geometric-midpoint bisection: lo → small delta (large eps), hi → large delta (small eps).
        // Invariant: to_epsilon_delta(lo) >= eps_target, to_epsilon_delta(hi) < eps_target.
        let mut lo = lo_init;
        let mut hi = hi_init;

        for _ in 0..80 {
            // Geometric mean in log space: avoids clustering near zero for tiny delta ranges.
            let mid = (lo.ln() + hi.ln()).mul_add(0.5, 0.0).exp();
            let eps_mid = self.to_epsilon_delta(mid)?;
            if eps_mid > epsilon_target {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Ok((lo + hi) / 2.0)
    }

    /// Sequential composition: (ρ₁+ρ₂, min(ω₁,ω₂)).
    ///
    /// # Errors
    /// Propagates validation errors from both mechanisms.
    pub fn compose(&self, other: &TcdpMechanism) -> PrivacyResult<TcdpMechanism> {
        Self::validate_mechanism_params(self.rho, self.omega)?;
        Self::validate_mechanism_params(other.rho, other.omega)?;
        Ok(TcdpMechanism {
            rho: self.rho + other.rho,
            omega: self.omega.min(other.omega),
        })
    }

    /// k-fold self-composition: (ρ·k, ω).
    ///
    /// # Errors
    /// - `InvalidParameter` if `k == 0`.
    /// - Propagates validation errors.
    pub fn compose_k(&self, k: usize) -> PrivacyResult<TcdpMechanism> {
        Self::validate_mechanism_params(self.rho, self.omega)?;
        if k == 0 {
            return Err(PrivacyError::InvalidParameter(
                "compose_k requires k ≥ 1".into(),
            ));
        }
        Ok(TcdpMechanism {
            rho: self.rho * k as f64,
            omega: self.omega,
        })
    }

    /// Poisson subsampling amplification at rate q ∈ (0,1].
    ///
    /// First-order bound: `ρ_sub = q²·ρ`, same ω.
    ///
    /// # Errors
    /// - `InvalidParameter` if `q ≤ 0` or `q > 1`.
    /// - Propagates validation errors.
    pub fn subsample_poisson(&self, q: f64) -> PrivacyResult<TcdpMechanism> {
        Self::validate_mechanism_params(self.rho, self.omega)?;
        if q <= 0.0 || q > 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sampling rate q must be in (0, 1], got {q}"
            )));
        }
        Ok(TcdpMechanism {
            rho: q * q * self.rho,
            omega: self.omega,
        })
    }

    /// Returns `(tcdp_epsilon, zcdp_formula_epsilon)` for the given δ.
    ///
    /// `zcdp_formula_epsilon = ρ + 2√(ρ·ln(1/δ))` (Bun-Steinke Lemma 3.5).
    /// When ω = ∞, these should be equal; for finite ω with α* > ω, tCDP may give a
    /// larger ε than the interior-optimum formula.
    ///
    /// # Errors
    /// Propagates `to_epsilon_delta` errors.
    pub fn compare_with_zcdp_formula(&self, delta: f64) -> PrivacyResult<(f64, f64)> {
        let tcdp_eps = self.to_epsilon_delta(delta)?;
        let log_inv_d = (1.0 / delta).ln();
        let zcdp_eps = self.rho + 2.0 * (self.rho * log_inv_d).sqrt();
        Ok((tcdp_eps, zcdp_eps))
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn validate_mechanism_params(rho: f64, omega: f64) -> PrivacyResult<()> {
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
        if omega.is_finite() && omega <= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "finite omega must be > 1 (boundary formula diverges otherwise), got {omega}"
            )));
        }
        Ok(())
    }
}

// ─── TcdpAccountant ──────────────────────────────────────────────────────────

/// Accountant that accumulates tCDP compositions from possibly heterogeneous mechanisms.
///
/// Uses sequential composition: total ρ = Σ ρᵢ, ω_min = min(ωᵢ).
#[derive(Clone, Debug)]
pub struct TcdpAccountant {
    /// Accumulated ρ (sum of all observed mechanism ρ values).
    pub rho_total: f64,
    /// Minimum ω seen across all observed mechanisms.
    pub omega_min: f64,
    /// Number of individual mechanism observations (counting k-fold as k).
    pub step_count: usize,
}

impl Default for TcdpAccountant {
    fn default() -> Self {
        Self::new()
    }
}

impl TcdpAccountant {
    /// Create a fresh accountant with no observed mechanisms.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rho_total: 0.0,
            omega_min: f64::INFINITY,
            step_count: 0,
        }
    }

    /// Record one application of `mechanism`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `mechanism.rho ≤ 0` or `mechanism.omega ≤ 0`, or
    ///   finite `mechanism.omega ≤ 1`.
    pub fn observe(&mut self, mechanism: &TcdpMechanism) -> PrivacyResult<()> {
        TcdpMechanism::validate_mechanism_params(mechanism.rho, mechanism.omega)?;
        self.rho_total += mechanism.rho;
        self.omega_min = self.omega_min.min(mechanism.omega);
        self.step_count += 1;
        Ok(())
    }

    /// Record `k` applications of `mechanism` (batch update).
    ///
    /// # Errors
    /// - `InvalidParameter` if `k == 0`.
    /// - Propagates `observe` validation errors.
    pub fn observe_k(&mut self, mechanism: &TcdpMechanism, k: usize) -> PrivacyResult<()> {
        if k == 0 {
            return Err(PrivacyError::InvalidParameter(
                "observe_k requires k ≥ 1".into(),
            ));
        }
        TcdpMechanism::validate_mechanism_params(mechanism.rho, mechanism.omega)?;
        self.rho_total += mechanism.rho * k as f64;
        self.omega_min = self.omega_min.min(mechanism.omega);
        self.step_count += k;
        Ok(())
    }

    /// Construct the accumulated `TcdpMechanism` from all observed steps.
    ///
    /// # Errors
    /// - `InvalidParameter` if no mechanisms have been observed.
    pub fn privacy_guarantee(&self) -> PrivacyResult<TcdpMechanism> {
        if self.rho_total <= 0.0 {
            return Err(PrivacyError::InvalidParameter(
                "no mechanisms observed (rho_total = 0)".into(),
            ));
        }
        Ok(TcdpMechanism {
            rho: self.rho_total,
            omega: self.omega_min,
        })
    }

    /// Convert accumulated privacy cost to (ε, δ)-DP.
    ///
    /// # Errors
    /// Propagates errors from `privacy_guarantee` and `to_epsilon_delta`.
    pub fn to_epsilon_delta(&self, delta: f64) -> PrivacyResult<f64> {
        self.privacy_guarantee()?.to_epsilon_delta(delta)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_rho() {
        // Δ=1, σ=2 → ρ = 1/(2·4) = 0.125, ω = ∞.
        let m = TcdpMechanism::gaussian(1.0, 2.0).expect("ok");
        assert!((m.rho - 0.125).abs() < 1e-12);
        assert!(m.omega.is_infinite());
    }

    #[test]
    fn test_gaussian_invalid_sensitivity() {
        assert!(TcdpMechanism::gaussian(0.0, 1.0).is_err());
        assert!(TcdpMechanism::gaussian(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_gaussian_invalid_sigma() {
        assert!(TcdpMechanism::gaussian(1.0, 0.0).is_err());
        assert!(TcdpMechanism::gaussian(1.0, -0.5).is_err());
    }

    #[test]
    fn test_laplace_rho() {
        // Δ=1, b=2 → ε₀=0.5, ρ = 0.5²/2 = 0.125.
        let m = TcdpMechanism::laplace(1.0, 2.0).expect("ok");
        assert!((m.rho - 0.125).abs() < 1e-12);
        assert!(m.omega.is_infinite());
    }

    #[test]
    fn test_from_pure_dp_rho() {
        // ε₀=0.1 → ρ = 0.1·(e^0.1 − 1)/2 ≈ 0.005254.
        let m = TcdpMechanism::from_pure_dp(0.1).expect("ok");
        let expected_rho = 0.1 * (0.1_f64.exp() - 1.0) / 2.0;
        assert!((m.rho - expected_rho).abs() < 1e-12);
        assert!(m.omega.is_infinite());
    }

    #[test]
    fn test_from_pure_dp_invalid() {
        assert!(TcdpMechanism::from_pure_dp(0.0).is_err());
        assert!(TcdpMechanism::from_pure_dp(-0.5).is_err());
    }

    // ── Epsilon-delta conversion ─────────────────────────────────────────────

    #[test]
    fn test_to_epsilon_delta_infinity_omega_matches_zcdp() {
        // For ω = ∞, should match ρ + 2√(ρ·ln(1/δ)) exactly.
        let m = TcdpMechanism::gaussian(1.0, 2.0).expect("ok"); // ρ=0.125
        let delta = 1e-5_f64;
        let (tcdp_eps, zcdp_eps) = m.compare_with_zcdp_formula(delta).expect("ok");
        assert!(
            (tcdp_eps - zcdp_eps).abs() < 1e-12,
            "tCDP={tcdp_eps:.8}, zCDP={zcdp_eps:.8}"
        );
    }

    #[test]
    fn test_to_epsilon_delta_finite_omega_interior() {
        // α* = 1 + √(ln(1/δ)/ρ). If α* ≤ ω, use interior formula.
        // ρ=0.5, δ=1e-3 → α* = 1 + √(ln(1000)/0.5) ≈ 1 + √13.816 ≈ 4.72
        // ω=10 > 4.72, so interior formula applies → same as zCDP formula.
        let m = TcdpMechanism {
            rho: 0.5,
            omega: 10.0,
        };
        let (tcdp_eps, zcdp_eps) = m.compare_with_zcdp_formula(1e-3).expect("ok");
        assert!(
            (tcdp_eps - zcdp_eps).abs() < 1e-12,
            "alpha_star <= omega: tCDP={tcdp_eps:.8}, zCDP={zcdp_eps:.8}"
        );
    }

    #[test]
    fn test_to_epsilon_delta_finite_omega_boundary() {
        // Boundary case: α* > ω. Use ε = ρ·ω + ln(1/δ)/(ω − 1).
        // ρ=0.01, ω=2.0, δ=1e-5 → α* = 1 + √(11.513/0.01) = 1 + 33.96 = 34.96 > ω=2
        // ε_boundary = 0.01·2 + 11.513/(2−1) = 0.02 + 11.513 = 11.533
        let m = TcdpMechanism {
            rho: 0.01,
            omega: 2.0,
        };
        let delta = 1e-5_f64;
        let eps = m.to_epsilon_delta(delta).expect("ok");
        let log_inv_d = (1.0 / delta).ln();
        let expected = 0.01 * 2.0 + log_inv_d / (2.0 - 1.0);
        assert!(
            (eps - expected).abs() < 1e-10,
            "eps={eps:.8}, expected={expected:.8}"
        );
    }

    #[test]
    fn test_invalid_delta() {
        let m = TcdpMechanism::gaussian(1.0, 1.0).expect("ok");
        assert!(m.to_epsilon_delta(0.0).is_err());
        assert!(m.to_epsilon_delta(1.0).is_err());
        assert!(m.to_epsilon_delta(-0.5).is_err());
        assert!(m.to_epsilon_delta(1.5).is_err());
    }

    #[test]
    fn test_delta_to_zero_gives_large_epsilon() {
        let m = TcdpMechanism::gaussian(1.0, 1.0).expect("ok");
        let eps_small_delta = m.to_epsilon_delta(1e-100).expect("ok");
        let eps_large_delta = m.to_epsilon_delta(0.1).expect("ok");
        assert!(
            eps_small_delta > eps_large_delta,
            "smaller δ should give larger ε: {eps_small_delta:.4} > {eps_large_delta:.4}"
        );
    }

    #[test]
    fn test_to_delta_epsilon_inverts_to_epsilon_delta() {
        let m = TcdpMechanism::gaussian(1.0, 2.0).expect("ok"); // ρ=0.125, ω=∞
        let delta_target = 1e-5_f64;
        let eps_forward = m.to_epsilon_delta(delta_target).expect("ok");
        let delta_recovered = m.to_delta_epsilon(eps_forward).expect("ok");
        // Round-trip: δ should match within numerical tolerance.
        let relative_err = (delta_recovered - delta_target).abs() / delta_target;
        assert!(
            relative_err < 1e-6,
            "δ mismatch: got {delta_recovered:.2e}, expected {delta_target:.2e}, rel_err={relative_err:.2e}"
        );
    }

    #[test]
    fn test_to_delta_epsilon_invalid_epsilon() {
        let m = TcdpMechanism::gaussian(1.0, 1.0).expect("ok");
        assert!(m.to_delta_epsilon(0.0).is_err());
        assert!(m.to_delta_epsilon(-1.0).is_err());
    }

    // ── Optimal alpha ───────────────────────────────────────────────────────

    #[test]
    fn test_optimal_alpha_infinite_omega() {
        let m = TcdpMechanism {
            rho: 0.5,
            omega: f64::INFINITY,
        };
        let delta = 1e-5_f64;
        let alpha = m.optimal_alpha(delta).expect("ok");
        let expected = 1.0 + (delta.recip().ln() / 0.5_f64).sqrt();
        assert!((alpha - expected).abs() < 1e-12);
    }

    #[test]
    fn test_optimal_alpha_finite_omega_clamped() {
        // If α* > ω, optimal_alpha returns ω.
        let m = TcdpMechanism {
            rho: 0.01,
            omega: 3.0,
        };
        let delta = 1e-5_f64; // α* ≈ 34.96, clamped to 3.0
        let alpha = m.optimal_alpha(delta).expect("ok");
        assert!(
            (alpha - 3.0).abs() < 1e-12,
            "expected omega=3.0, got {alpha}"
        );
    }

    // ── Composition ─────────────────────────────────────────────────────────

    #[test]
    fn test_compose_rho_sum_omega_min() {
        let m1 = TcdpMechanism {
            rho: 0.1,
            omega: 5.0,
        };
        let m2 = TcdpMechanism {
            rho: 0.2,
            omega: 8.0,
        };
        let composed = m1.compose(&m2).expect("ok");
        assert!((composed.rho - 0.3).abs() < 1e-12);
        assert!((composed.omega - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_compose_k_rho_scaled() {
        let m = TcdpMechanism {
            rho: 0.05,
            omega: f64::INFINITY,
        };
        let composed = m.compose_k(10).expect("ok");
        assert!((composed.rho - 0.5).abs() < 1e-12);
        assert!(composed.omega.is_infinite());
    }

    #[test]
    fn test_compose_k_zero_fails() {
        let m = TcdpMechanism {
            rho: 0.1,
            omega: f64::INFINITY,
        };
        assert!(m.compose_k(0).is_err());
    }

    #[test]
    fn test_compose_finite_omega_takes_min() {
        let m1 = TcdpMechanism {
            rho: 0.3,
            omega: 4.0,
        };
        let m2 = TcdpMechanism {
            rho: 0.2,
            omega: 7.0,
        };
        let composed = m1.compose(&m2).expect("ok");
        assert!((composed.omega - 4.0).abs() < 1e-12);
        assert!((composed.rho - 0.5).abs() < 1e-12);
    }

    // ── Subsampling ─────────────────────────────────────────────────────────

    #[test]
    fn test_subsample_poisson_rho_scaled() {
        let m = TcdpMechanism {
            rho: 1.0,
            omega: f64::INFINITY,
        };
        let sub = m.subsample_poisson(0.1).expect("ok");
        assert!((sub.rho - 0.01).abs() < 1e-12);
        assert!(sub.omega.is_infinite());
    }

    #[test]
    fn test_subsample_poisson_invalid_q() {
        let m = TcdpMechanism {
            rho: 1.0,
            omega: f64::INFINITY,
        };
        assert!(m.subsample_poisson(0.0).is_err());
        assert!(m.subsample_poisson(1.1).is_err());
        assert!(m.subsample_poisson(-0.5).is_err());
    }

    // ── Accountant ──────────────────────────────────────────────────────────

    #[test]
    fn test_accountant_empty_fails() {
        let acc = TcdpAccountant::new();
        assert!(acc.privacy_guarantee().is_err());
    }

    #[test]
    fn test_accountant_observe_twice_rho_sum() {
        let mut acc = TcdpAccountant::new();
        let m = TcdpMechanism {
            rho: 0.1,
            omega: f64::INFINITY,
        };
        acc.observe(&m).expect("ok");
        acc.observe(&m).expect("ok");
        assert!((acc.rho_total - 0.2).abs() < 1e-12);
        assert_eq!(acc.step_count, 2);
    }

    #[test]
    fn test_accountant_observe_k_batch() {
        let mut acc = TcdpAccountant::new();
        let m = TcdpMechanism {
            rho: 0.05,
            omega: 4.0,
        };
        acc.observe_k(&m, 20).expect("ok");
        assert!((acc.rho_total - 1.0).abs() < 1e-12);
        assert_eq!(acc.step_count, 20);
        assert!((acc.omega_min - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_accountant_to_epsilon_delta() {
        let mut acc = TcdpAccountant::new();
        let m = TcdpMechanism::gaussian(1.0, 2.0).expect("ok");
        acc.observe_k(&m, 5).expect("ok");
        let eps = acc.to_epsilon_delta(1e-5).expect("ok");
        // Should equal m.compose_k(5).to_epsilon_delta(1e-5).
        let composed = m.compose_k(5).expect("ok");
        let eps_ref = composed.to_epsilon_delta(1e-5).expect("ok");
        assert!((eps - eps_ref).abs() < 1e-12);
    }

    #[test]
    fn test_accountant_omega_min_tracked() {
        let mut acc = TcdpAccountant::new();
        acc.observe(&TcdpMechanism {
            rho: 0.1,
            omega: 10.0,
        })
        .expect("ok");
        acc.observe(&TcdpMechanism {
            rho: 0.1,
            omega: 3.0,
        })
        .expect("ok");
        acc.observe(&TcdpMechanism {
            rho: 0.1,
            omega: 7.0,
        })
        .expect("ok");
        assert!((acc.omega_min - 3.0).abs() < 1e-12);
    }
}
