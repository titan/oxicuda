//! Sampled Gaussian Mechanism (SGM) — Rényi-DP amplification by Poisson
//! subsampling.
//!
//! References:
//! - Mironov, Talwar & Zhang (2019), "Rényi Differential Privacy of the
//!   Sampled Gaussian Mechanism", arXiv:1908.10530.
//! - Balle, Barthe, Gaboardi, Hsu & Sato (2020), "Hypothesis Testing
//!   Interpretations and Rényi Differential Privacy", AISTATS 2020 (optimal
//!   RDP → (ε, δ) conversion).
//!
//! # Sampled Gaussian Mechanism
//! Given a query `f` with L2 sensitivity Δ, Poisson subsampling rate `q`, and
//! noise multiplier `σ` (so the noise std is `σ·Δ`), the SGM releases
//!
//! `SG_{q,σ}(D) = f({xᵢ : i ∈ Poisson(q)}) + 𝒩(0, σ²Δ²·I)`.
//!
//! The privacy curve is most naturally expressed in Rényi-DP. For integer
//! order `α ≥ 2` the tight upper bound (Mironov-Talwar-Zhang 2019, Theorem 11)
//! is
//!
//! `ε(α) ≤ (1/(α−1)) · ln( Σ_{k=0}^{α} C(α,k)·(1−q)^{α−k}·qᵏ · exp(k(k−1)/(2σ²)) )`.
//!
//! The `k = 0` and `k = 1` terms contribute `(1−q)^α` and `α·q·(1−q)^{α−1}`
//! respectively (their exponential factor is 1), and the remaining terms carry
//! the Gaussian log-moment `exp(k(k−1)/(2σ²))`.
//!
//! This file works entirely with the **noise multiplier** `σ` (i.e. it folds
//! the L2 sensitivity Δ into the noise std), which is the convention used by
//! the moments accountant of DP-SGD. It is intentionally distinct from
//! `accounting::rdp_subsampling`, which keeps Δ explicit and uses a different
//! (looser) `T_k = exp((k−1)·ε_M(k))` per-term bound.

use crate::accounting::rdp_subsampling::RdpSubsampling;
use crate::error::{PrivacyError, PrivacyResult};

/// Configuration for the Sampled Gaussian Mechanism RDP accountant.
#[derive(Debug, Clone)]
pub struct SampledGaussianConfig {
    /// Poisson subsampling rate q ∈ (0, 1].
    pub sampling_rate: f64,
    /// Gaussian noise multiplier σ > 0 (noise std = σ · L2-sensitivity).
    pub noise_multiplier: f64,
    /// Maximum integer Rényi order α (inclusive), must be ≥ 2.
    pub max_order: usize,
}

impl SampledGaussianConfig {
    /// Construct and validate a [`SampledGaussianConfig`].
    ///
    /// # Errors
    /// - `InvalidParameter` if `sampling_rate ∉ (0, 1]`, `noise_multiplier ≤ 0`,
    ///   or `max_order < 2`.
    pub fn new(sampling_rate: f64, noise_multiplier: f64, max_order: usize) -> PrivacyResult<Self> {
        if !(sampling_rate > 0.0 && sampling_rate <= 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "sampling_rate must be in (0, 1], got {sampling_rate}"
            )));
        }
        if noise_multiplier <= 0.0 || !noise_multiplier.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise_multiplier must be positive and finite, got {noise_multiplier}"
            )));
        }
        if max_order < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "max_order must be ≥ 2, got {max_order}"
            )));
        }
        Ok(Self {
            sampling_rate,
            noise_multiplier,
            max_order,
        })
    }
}

/// The Sampled Gaussian Mechanism RDP accountant.
///
/// Holds the accumulated RDP curve over Rényi orders `2..=max_order` so that
/// repeated SGM steps (DP-SGD training iterations) can be composed additively
/// before a single RDP → (ε, δ) conversion.
#[derive(Debug, Clone)]
pub struct SampledGaussianMechanism {
    sampling_rate: f64,
    noise_multiplier: f64,
    /// Evaluated integer orders `[2, 3, …, max_order]`.
    orders: Vec<usize>,
    /// Accumulated RDP value `ε(α)` for each order (sum across composed steps).
    rdp: Vec<f64>,
}

impl SampledGaussianMechanism {
    /// Create a fresh accountant with zero accumulated RDP.
    ///
    /// # Errors
    /// Propagates configuration validation errors.
    pub fn new(cfg: &SampledGaussianConfig) -> PrivacyResult<Self> {
        let orders: Vec<usize> = (2..=cfg.max_order).collect();
        let rdp = vec![0.0_f64; orders.len()];
        Ok(Self {
            sampling_rate: cfg.sampling_rate,
            noise_multiplier: cfg.noise_multiplier,
            orders,
            rdp,
        })
    }

    /// Single-step RDP of the SGM at integer order `alpha ≥ 2`.
    ///
    /// `ε(α) = (1/(α−1))·ln( Σ_{k=0}^{α} C(α,k)·(1−q)^{α−k}·qᵏ·exp(k(k−1)/(2σ²)) )`.
    ///
    /// The summand is accumulated in a numerically careful order (small terms
    /// first) and the helper returns the bound clamped to be non-negative.
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha < 2`.
    /// - `InvalidParameter` if the binomial moment sum is NaN.
    pub fn rdp_step(q: f64, noise_multiplier: f64, alpha: usize) -> PrivacyResult<f64> {
        if alpha < 2 {
            return Err(PrivacyError::InvalidParameter(format!(
                "alpha must be ≥ 2, got {alpha}"
            )));
        }
        let sigma2 = noise_multiplier * noise_multiplier;
        let one_minus_q = 1.0 - q;

        // Accumulate the binomial-moment sum. Each term's coefficient
        // `C(α,k)·qᵏ·(1−q)^{α−k}` is in [0, 1]; multiplying a finite (possibly
        // zero) coefficient by a possibly-huge `exp_factor` can overflow, so we
        // skip terms whose coefficient is exactly zero before forming the
        // product to avoid `0 · ∞ = NaN`.
        let mut sum = 0.0_f64;
        for k in 0..=alpha {
            let binom = RdpSubsampling::binomial_coeff(alpha, k);
            let q_pow = q.powi(k as i32);
            let omq_pow = one_minus_q.powi((alpha - k) as i32);
            let coeff = binom * q_pow * omq_pow;
            if coeff == 0.0 {
                continue;
            }
            // Gaussian log-moment factor: exp(k(k−1)/(2σ²)). For k∈{0,1} this is 1.
            let exp_factor = if k < 2 {
                1.0
            } else {
                ((k * (k - 1)) as f64 / (2.0 * sigma2)).exp()
            };
            sum += coeff * exp_factor;
        }

        if sum.is_nan() {
            return Err(PrivacyError::InvalidParameter(format!(
                "NaN in SGM binomial moment at alpha={alpha}"
            )));
        }
        // An overflowed (infinite) moment means the noise is far too small to
        // provide RDP at this order: report an infinite (no-privacy) bound
        // rather than failing, so callers can still account other orders.
        if sum.is_infinite() {
            return Ok(f64::INFINITY);
        }
        if sum <= 0.0 {
            return Ok(0.0);
        }
        Ok((sum.ln() / (alpha as f64 - 1.0)).max(0.0))
    }

    /// Compose `steps` independent SGM applications into the accumulated RDP
    /// curve (additive in RDP).
    ///
    /// # Errors
    /// Propagates `rdp_step` errors.
    pub fn compose(&mut self, steps: usize) -> PrivacyResult<()> {
        if steps == 0 {
            return Ok(());
        }
        for (i, &alpha) in self.orders.iter().enumerate() {
            let per_step = Self::rdp_step(self.sampling_rate, self.noise_multiplier, alpha)?;
            self.rdp[i] += (steps as f64) * per_step;
        }
        Ok(())
    }

    /// Read-only view of the evaluated Rényi orders.
    #[must_use]
    pub fn orders(&self) -> &[usize] {
        &self.orders
    }

    /// Read-only view of the accumulated RDP values aligned with [`orders`].
    ///
    /// [`orders`]: Self::orders
    #[must_use]
    pub fn rdp_curve(&self) -> &[f64] {
        &self.rdp
    }

    /// Convert the accumulated RDP curve to the tightest (ε, δ)-DP guarantee.
    ///
    /// Uses the optimal conversion (Balle et al. 2020; Canonne-Kamath-Steinke
    /// 2020): for each order α,
    ///
    /// `ε(α) = rdp(α) + ln((α−1)/α) − (ln δ + ln α)/(α−1)`
    ///
    /// and returns `max(0, min_α ε(α))`.
    ///
    /// # Errors
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    /// - `InvalidParameter` if no order yields a finite ε.
    pub fn get_epsilon(&self, delta: f64) -> PrivacyResult<f64> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        // A curve with no accumulated RDP corresponds to the identity mechanism,
        // which is perfectly (0, 0)-DP; the finite-order conversion below carries
        // slack, so short-circuit the trivial case to the exact ε = 0.
        if self.rdp.iter().all(|&r| r == 0.0) {
            return Ok(0.0);
        }
        let mut best = f64::INFINITY;
        for (i, &alpha) in self.orders.iter().enumerate() {
            let a = alpha as f64;
            let candidate = self.rdp[i] + ((a - 1.0) / a).ln() - (delta.ln() + a.ln()) / (a - 1.0);
            if candidate.is_finite() && candidate < best {
                best = candidate;
            }
        }
        if !best.is_finite() {
            return Err(PrivacyError::InvalidParameter(
                "RDP → (ε,δ) conversion produced no finite ε".into(),
            ));
        }
        Ok(best.max(0.0))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(q: f64, sigma: f64, max_order: usize) -> SampledGaussianMechanism {
        let cfg = SampledGaussianConfig::new(q, sigma, max_order).expect("cfg");
        SampledGaussianMechanism::new(&cfg).expect("new")
    }

    // 1. ε increases monotonically with the number of composed steps.
    #[test]
    fn epsilon_increases_with_steps() {
        let mut a = make(0.01, 1.0, 32);
        let mut b = make(0.01, 1.0, 32);
        a.compose(50).expect("compose");
        b.compose(200).expect("compose");
        let eps_a = a.get_epsilon(1e-5).expect("eps");
        let eps_b = b.get_epsilon(1e-5).expect("eps");
        assert!(eps_b > eps_a, "more steps → larger ε: {eps_b} > {eps_a}");
    }

    // 2. Larger noise multiplier → smaller ε.
    #[test]
    fn epsilon_decreases_with_noise() {
        let mut lo = make(0.05, 1.0, 32);
        let mut hi = make(0.05, 4.0, 32);
        lo.compose(100).expect("c");
        hi.compose(100).expect("c");
        let eps_lo = lo.get_epsilon(1e-5).expect("e");
        let eps_hi = hi.get_epsilon(1e-5).expect("e");
        assert!(
            eps_hi < eps_lo,
            "more noise → smaller ε: {eps_hi} < {eps_lo}"
        );
    }

    // 3. Smaller sampling rate → smaller privacy loss (amplification).
    #[test]
    fn more_sampling_more_privacy_loss() {
        let mut small_q = make(0.005, 1.0, 32);
        let mut large_q = make(0.05, 1.0, 32);
        small_q.compose(100).expect("c");
        large_q.compose(100).expect("c");
        let eps_small = small_q.get_epsilon(1e-5).expect("e");
        let eps_large = large_q.get_epsilon(1e-5).expect("e");
        assert!(
            eps_large > eps_small,
            "larger q → larger ε: {eps_large} > {eps_small}"
        );
    }

    // 4. RDP per-step is monotone non-decreasing in the Rényi order.
    #[test]
    fn rdp_orders_monotone() {
        let m = make(0.1, 1.5, 16);
        let mut prev = f64::NEG_INFINITY;
        for &alpha in m.orders() {
            let r = SampledGaussianMechanism::rdp_step(0.1, 1.5, alpha).expect("step");
            assert!(r >= prev - 1e-12, "non-monotone at α={alpha}: {r} < {prev}");
            prev = r;
        }
    }

    // 5. get_epsilon yields a finite, positive value.
    #[test]
    fn get_epsilon_finite() {
        let mut m = make(0.01, 1.1, 64);
        m.compose(1000).expect("c");
        let eps = m.get_epsilon(1e-6).expect("e");
        assert!(eps.is_finite() && eps > 0.0, "ε must be finite +: {eps}");
    }

    // 6. Zero composed steps ⇒ zero ε.
    #[test]
    fn zero_steps_zero_epsilon() {
        let mut m = make(0.1, 1.0, 32);
        m.compose(0).expect("c");
        let eps = m.get_epsilon(1e-5).expect("e");
        assert!(eps.abs() < 1e-9, "0 steps ⇒ ε≈0, got {eps}");
    }

    // 7. Smaller δ → larger ε for a fixed accumulated curve.
    #[test]
    fn delta_affects_epsilon() {
        let mut m = make(0.02, 1.0, 32);
        m.compose(100).expect("c");
        let eps_tight = m.get_epsilon(1e-7).expect("e");
        let eps_loose = m.get_epsilon(1e-3).expect("e");
        assert!(
            eps_tight > eps_loose,
            "smaller δ → larger ε: {eps_tight} > {eps_loose}"
        );
    }

    // 8. q = 1 (full batch) reduces to the un-subsampled Gaussian RDP α/(2σ²).
    #[test]
    fn q_1_full_batch() {
        // With q=1 only the k=α term survives, giving ε(α)=α(α−1)/(2σ²)/(α−1)
        // = α/(2σ²) — exactly the plain Gaussian-mechanism RDP.
        let sigma = 2.0;
        for alpha in 2..=10 {
            let got = SampledGaussianMechanism::rdp_step(1.0, sigma, alpha).expect("step");
            let want = alpha as f64 / (2.0 * sigma * sigma);
            assert!(
                (got - want).abs() < 1e-9,
                "q=1 α={alpha}: got {got}, want {want}"
            );
        }
    }

    // 9. As σ → ∞ the per-step RDP → 0 (noise dominates).
    #[test]
    fn noise_large_rdp_vanishes() {
        let small = SampledGaussianMechanism::rdp_step(0.1, 1.0, 4).expect("s");
        let large = SampledGaussianMechanism::rdp_step(0.1, 100.0, 4).expect("s");
        assert!(large < small, "huge σ → tiny RDP: {large} < {small}");
        assert!(large < 1e-2, "huge σ RDP should be near zero: {large}");
    }

    // 10. RDP composition is additive: composing n then m equals composing n+m.
    #[test]
    fn compose_additive() {
        let mut split = make(0.03, 1.2, 32);
        split.compose(30).expect("c");
        split.compose(70).expect("c");
        let mut once = make(0.03, 1.2, 32);
        once.compose(100).expect("c");
        for (a, b) in split.rdp_curve().iter().zip(once.rdp_curve().iter()) {
            assert!((a - b).abs() < 1e-9, "non-additive: {a} vs {b}");
        }
    }

    // 11. Invalid configurations are rejected.
    #[test]
    fn invalid_config_errors() {
        assert!(SampledGaussianConfig::new(0.0, 1.0, 4).is_err());
        assert!(SampledGaussianConfig::new(1.5, 1.0, 4).is_err());
        assert!(SampledGaussianConfig::new(0.1, 0.0, 4).is_err());
        assert!(SampledGaussianConfig::new(0.1, 1.0, 1).is_err());
    }

    // 12. get_epsilon rejects out-of-range δ; rdp_step rejects α < 2.
    #[test]
    fn boundary_errors() {
        let m = make(0.1, 1.0, 8);
        assert!(m.get_epsilon(0.0).is_err());
        assert!(m.get_epsilon(1.0).is_err());
        assert!(SampledGaussianMechanism::rdp_step(0.1, 1.0, 1).is_err());
    }

    // 13. A vanishingly small σ overflows the moment to +∞ (no privacy) rather
    //     than producing NaN or erroring.
    #[test]
    fn tiny_sigma_infinite_rdp_no_nan() {
        let r = SampledGaussianMechanism::rdp_step(1.0, 1e-6, 3).expect("step");
        assert!(r.is_infinite(), "tiny σ should give ∞ RDP, got {r}");
        assert!(!r.is_nan(), "must never be NaN");
    }
}
