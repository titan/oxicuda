//! Concentrated Differential Privacy (original Dwork-Rothblum definition).
//!
//! Reference: Dwork & Rothblum (2016), "Concentrated Differential Privacy",
//! arXiv:1603.01887.
//!
//! # Mean-Concentrated DP
//! A randomised mechanism `M` is **(μ, τ)-mCDP** (mean-concentrated DP) if for
//! all neighbouring datasets `x, x'` the privacy-loss random variable
//! `L = ln( P[M(x)=o] / P[M(x')=o] )` (with `o ~ M(x)`) is **subgaussian**
//! with mean at most `μ` and standard parameter `τ`:
//!
//! `E[L] ≤ μ`   and   `E[exp(λ(L − E[L]))] ≤ exp(λ²τ²/2)`  for all `λ ∈ ℝ`.
//!
//! Intuitively the expected privacy loss is small (`μ`) and the loss
//! concentrates sharply around its mean with subgaussian width `τ`.
//!
//! # Gaussian mechanism
//! Adding `𝒩(0, σ²)` noise to a query with L2 sensitivity `Δ` gives a privacy
//! loss that is *exactly* Gaussian with mean `Δ²/(2σ²)` and variance
//! `Δ²/σ²`. Hence the Gaussian mechanism is `(Δ²/(2σ²), Δ/σ)`-mCDP — note the
//! identity `μ = τ²/2`, the canonical "concentrated" relationship.
//!
//! # Composition
//! mCDP composes by adding means and adding variances (subgaussian parameters
//! add in quadrature):
//!
//! `(μ₁, τ₁) ∘ … ∘ (μₖ, τₖ) → (Σμᵢ, √Στᵢ²)`.
//!
//! # Conversion to (ε, δ)-DP
//! Using the subgaussian tail bound on `L` (Dwork-Rothblum 2016, Lemma 3.5 /
//! Markov on the moment generating function), an `(μ, τ)`-mCDP mechanism is
//! `(ε, δ)`-DP with
//!
//! `δ(ε) = exp( −(ε − μ)² / (2τ²) )`   for `ε ≥ μ`,
//!
//! equivalently `ε(δ) = μ + τ·√(2·ln(1/δ))`.
//!
//! # Relationship to zCDP
//! Bun-Steinke zCDP is a relaxation of mCDP; a `ρ`-zCDP mechanism is
//! `(ρ, √(2ρ))`-mCDP-like in its conversion constant. We expose
//! [`mcdp_from_zcdp`] for the canonical Gaussian correspondence `μ = ρ`,
//! `τ = √(2ρ)`.

use crate::error::{PrivacyError, PrivacyResult};

/// A mean-concentrated DP guarantee `(μ, τ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mcdp {
    /// Mean privacy-loss bound `μ ≥ 0`.
    pub mu: f64,
    /// Subgaussian standard parameter `τ > 0`.
    pub tau: f64,
}

impl Mcdp {
    /// Construct and validate an mCDP guarantee.
    ///
    /// # Errors
    /// - `InvalidParameter` if `mu < 0`, `tau ≤ 0`, or either is non-finite.
    pub fn new(mu: f64, tau: f64) -> PrivacyResult<Self> {
        if !mu.is_finite() || mu < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "mu must be finite and ≥ 0, got {mu}"
            )));
        }
        if !tau.is_finite() || tau <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "tau must be finite and > 0, got {tau}"
            )));
        }
        Ok(Self { mu, tau })
    }

    /// mCDP parameters of the Gaussian mechanism with L2 sensitivity `Δ` and
    /// noise std `σ`: `μ = Δ²/(2σ²)`, `τ = Δ/σ`.
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
        let tau = sensitivity / sigma;
        Ok(Self {
            mu: tau * tau / 2.0,
            tau,
        })
    }

    /// Compose `self` with `other` (independent mechanisms): means add, the
    /// subgaussian parameters add in quadrature.
    #[must_use]
    pub fn compose(self, other: Self) -> Self {
        Self {
            mu: self.mu + other.mu,
            tau: (self.tau * self.tau + other.tau * other.tau).sqrt(),
        }
    }

    /// `k`-fold homogeneous composition: `(kμ, √k·τ)`.
    #[must_use]
    pub fn compose_k(self, k: usize) -> Self {
        let kf = k as f64;
        Self {
            mu: self.mu * kf,
            tau: self.tau * kf.sqrt(),
        }
    }

    /// `δ(ε)` from the subgaussian tail bound: `exp(−(ε−μ)²/(2τ²))` for
    /// `ε ≥ μ`, clamped to `1.0` for `ε < μ`.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
    pub fn delta(&self, epsilon: f64) -> PrivacyResult<f64> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if epsilon < self.mu {
            return Ok(1.0);
        }
        let z = (epsilon - self.mu) / self.tau;
        Ok((-0.5 * z * z).exp())
    }

    /// `ε(δ)` inverting the subgaussian tail bound:
    /// `ε = μ + τ·√(2·ln(1/δ))`.
    ///
    /// # Errors
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    pub fn epsilon(&self, delta: f64) -> PrivacyResult<f64> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        Ok(self.mu + self.tau * (2.0 * (1.0 / delta).ln()).sqrt())
    }
}

/// Canonical mCDP parameters corresponding to a `ρ`-zCDP guarantee:
/// `μ = ρ`, `τ = √(2ρ)` (the Gaussian-mechanism correspondence).
///
/// # Errors
/// - `InvalidParameter` if `rho ≤ 0`.
pub fn mcdp_from_zcdp(rho: f64) -> PrivacyResult<Mcdp> {
    if rho <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "rho must be positive, got {rho}"
        )));
    }
    Mcdp::new(rho, (2.0 * rho).sqrt())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Gaussian mechanism obeys the canonical μ = τ²/2 identity.
    #[test]
    fn gaussian_mu_tau_identity() {
        let m = Mcdp::gaussian(2.0, 3.0).expect("g");
        assert!((m.tau - 2.0 / 3.0).abs() < 1e-12, "tau {}", m.tau);
        assert!(
            (m.mu - m.tau * m.tau / 2.0).abs() < 1e-12,
            "μ should equal τ²/2"
        );
    }

    // 2. Composition adds means.
    #[test]
    fn compose_adds_mu() {
        let a = Mcdp::new(0.1, 0.4).expect("a");
        let b = Mcdp::new(0.2, 0.3).expect("b");
        let c = a.compose(b);
        assert!((c.mu - 0.3).abs() < 1e-12, "μ should add: {}", c.mu);
    }

    // 3. Composition adds subgaussian parameters in quadrature.
    #[test]
    fn compose_quadrature_tau() {
        let a = Mcdp::new(0.1, 0.3).expect("a");
        let b = Mcdp::new(0.1, 0.4).expect("b");
        let c = a.compose(b);
        // √(0.3² + 0.4²) = 0.5.
        assert!((c.tau - 0.5).abs() < 1e-12, "τ quadrature: {}", c.tau);
    }

    // 4. k-fold composition matches repeated pairwise composition.
    #[test]
    fn compose_k_matches_repeated() {
        let base = Mcdp::gaussian(1.0, 2.0).expect("g");
        let k = 9;
        let folded = base.compose_k(k);
        let mut acc = base;
        for _ in 1..k {
            acc = acc.compose(base);
        }
        assert!((folded.mu - acc.mu).abs() < 1e-10, "μ mismatch");
        assert!((folded.tau - acc.tau).abs() < 1e-10, "τ mismatch");
    }

    // 5. Smaller δ → larger ε.
    #[test]
    fn epsilon_monotone_in_delta() {
        let m = Mcdp::gaussian(1.0, 1.0).expect("g");
        let e_tight = m.epsilon(1e-7).expect("e");
        let e_loose = m.epsilon(1e-3).expect("e");
        assert!(e_tight > e_loose, "{e_tight} > {e_loose}");
    }

    // 6. δ and ε are mutual inverses through the tail bound.
    #[test]
    fn delta_epsilon_round_trip() {
        let m = Mcdp::gaussian(1.0, 1.5).expect("g");
        let target_delta = 1e-5;
        let eps = m.epsilon(target_delta).expect("e");
        let back = m.delta(eps).expect("d");
        assert!(
            (back - target_delta).abs() < 1e-9,
            "round trip δ: {back} vs {target_delta}"
        );
    }

    // 7. ε below μ yields δ = 1 (no useful guarantee).
    #[test]
    fn epsilon_below_mu_delta_one() {
        let m = Mcdp::new(2.0, 0.5).expect("m");
        let d = m.delta(1.0).expect("d");
        assert!((d - 1.0).abs() < 1e-12, "δ should be 1 for ε<μ, got {d}");
    }

    // 8. δ decreases as ε grows beyond μ.
    #[test]
    fn delta_decreases_with_epsilon() {
        let m = Mcdp::gaussian(1.0, 1.0).expect("g");
        let d_small = m.delta(m.mu + 0.5).expect("d");
        let d_large = m.delta(m.mu + 3.0).expect("d");
        assert!(d_large < d_small, "{d_large} < {d_small}");
    }

    // 9. More noise → smaller μ and τ.
    #[test]
    fn more_noise_smaller_params() {
        let lo = Mcdp::gaussian(1.0, 1.0).expect("g");
        let hi = Mcdp::gaussian(1.0, 5.0).expect("g");
        assert!(hi.mu < lo.mu, "μ should shrink with σ");
        assert!(hi.tau < lo.tau, "τ should shrink with σ");
    }

    // 10. zCDP correspondence: μ = ρ, τ = √(2ρ).
    #[test]
    fn from_zcdp_correspondence() {
        let m = mcdp_from_zcdp(0.5).expect("z");
        assert!((m.mu - 0.5).abs() < 1e-12, "μ = ρ");
        assert!((m.tau - 1.0).abs() < 1e-12, "τ = √(2·0.5) = 1");
    }

    // 11. Invalid construction parameters are rejected.
    #[test]
    fn invalid_params_error() {
        assert!(Mcdp::new(-0.1, 1.0).is_err());
        assert!(Mcdp::new(0.1, 0.0).is_err());
        assert!(Mcdp::gaussian(0.0, 1.0).is_err());
        assert!(Mcdp::gaussian(1.0, 0.0).is_err());
        assert!(mcdp_from_zcdp(0.0).is_err());
    }

    // 12. Out-of-range privacy parameters are rejected by conversions.
    #[test]
    fn conversion_boundary_errors() {
        let m = Mcdp::gaussian(1.0, 1.0).expect("g");
        assert!(m.epsilon(0.0).is_err());
        assert!(m.epsilon(1.0).is_err());
        assert!(m.delta(0.0).is_err());
        assert!(m.delta(-1.0).is_err());
    }

    // 13. Composed ε is larger than a single-mechanism ε (privacy degrades).
    #[test]
    fn composition_increases_epsilon() {
        let base = Mcdp::gaussian(1.0, 2.0).expect("g");
        let single = base.epsilon(1e-5).expect("e");
        let many = base.compose_k(20).epsilon(1e-5).expect("e");
        assert!(many > single, "composed ε larger: {many} > {single}");
    }
}
