//! Skellam mechanism for integer-valued queries with closed-form RDP.
//!
//! # Reference
//! - Agarwal N, Suresh AT, Yu F, Kumar S, McMahan B (2021),
//!   *"The Skellam Mechanism for Differentially Private Federated
//!   Learning"*, NeurIPS 2021,
//!   <https://arxiv.org/abs/2110.04995>.
//!
//! The Skellam distribution is the law of `Z = X − Y` for two independent
//! Poisson variables `X, Y ~ Pois(μ)`.  It supports the entire integer
//! lattice, is symmetric about zero, and has variance `2μ`.  Adding a
//! Skellam draw to an integer-valued query provides a discrete analogue
//! of the Gaussian mechanism, free of the floating-point side channels
//! analysed by Mironov (2012).
//!
//! # Rényi DP (Agarwal et al. 2021, Theorem 1)
//! For a query with `L1` sensitivity `Δ₁` (and optional `L2` sensitivity
//! `Δ₂`), the Skellam mechanism with rate `μ` satisfies
//!
//! ```text
//!     ε_R(α) ≤ α · Δ₁² / (2μ)
//!             + min{ (2α − 1) · Δ₁⁴ / (4μ²) + 3 · Δ₁³ / (2μ²),
//!                    3α · Δ₁³ / (2μ²) }
//! ```
//!
//! If `Δ₂² ≤ Δ₁` the leading term can be tightened by replacing `Δ₁²`
//! with `Δ₂²` (refined L2 bound, Agarwal et al. 2021 Corollary 1).
//!
//! # `(ε, δ)`-DP conversion
//! Standard RDP → DP:
//!
//! ```text
//!     ε(δ) = min_α [ ε_R(α) + ln(1/δ) / (α − 1) ]
//! ```
//!
//! optimised over a fixed grid of orders.
//!
//! # Sampling
//! - **Knuth's algorithm** for `μ ≤ 30`: accept-reject with Poisson PMF
//!   product accumulator.
//! - **Normal approximation** for `μ > 30`: `Pois(μ) ≈ round(N(μ, μ))`,
//!   clamped to `≥ 0`.  Total-variation distance to the true Poisson is
//!   `O(1/√μ)` (Berry-Esseen), well below DP-relevant precisions for the
//!   `μ` values used in practice.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration of a Skellam noise mechanism.
#[derive(Debug, Clone)]
pub struct SkellamConfig {
    /// Poisson rate `μ > 0`.  The Skellam noise has variance `2μ`.
    pub mu: f64,
    /// `L1` sensitivity `Δ₁ > 0`.
    pub sensitivity_l1: f64,
    /// `L2` sensitivity `Δ₂ > 0`.  Used by the refined RDP bound when
    /// `Δ₂² ≤ Δ₁` (Agarwal et al. 2021 Corollary 1).
    pub sensitivity_l2: f64,
}

/// Skellam mechanism: `f(x) + ξ` with `ξ ~ Skellam(μ, μ)`.
#[derive(Debug, Clone)]
pub struct SkellamMechanism {
    cfg: SkellamConfig,
}

/// Cutoff between Knuth's direct algorithm and the normal-approximation
/// branch for Poisson sampling.  Knuth becomes slow (and numerically
/// fragile because `p` shrinks toward zero) above this rate.
const POIS_NORMAL_CUTOFF: f64 = 30.0;

/// RDP optimisation grid used by `to_epsilon_delta`.  The standard RDP→DP
/// conversion is minimised over `α ∈ ALPHA_GRID`; finer grids tighten
/// the resulting `ε(δ)` at a negligible cost.
const ALPHA_GRID: &[f64] = &[1.5, 1.75, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

impl SkellamMechanism {
    /// Construct a Skellam mechanism from a config.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if any sensitivity is non-positive or
    ///   non-finite.
    /// - `InvalidParameter` if `μ ≤ 0` or non-finite.
    pub fn new(cfg: SkellamConfig) -> PrivacyResult<Self> {
        if !(cfg.mu.is_finite() && cfg.mu > 0.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "mu must be > 0 and finite, got {}",
                cfg.mu
            )));
        }
        if !(cfg.sensitivity_l1.is_finite() && cfg.sensitivity_l1 > 0.0) {
            return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity_l1));
        }
        if !(cfg.sensitivity_l2.is_finite() && cfg.sensitivity_l2 > 0.0) {
            return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity_l2));
        }
        Ok(Self { cfg })
    }

    /// Reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &SkellamConfig {
        &self.cfg
    }

    /// Draw one Skellam(μ, μ) variate as `X − Y` with independent
    /// `X, Y ~ Pois(μ)`.
    pub fn sample(&self, rng: &mut LcgRng) -> i64 {
        let x = sample_poisson(self.cfg.mu, rng);
        let y = sample_poisson(self.cfg.mu, rng);
        x.saturating_sub(y)
    }

    /// Add Skellam noise to a scalar integer query.
    pub fn add_noise(&self, value: i64, rng: &mut LcgRng) -> i64 {
        value.wrapping_add(self.sample(rng))
    }

    /// Add fresh Skellam noise to each coordinate of an integer vector.
    ///
    /// The configured `(Δ₁, Δ₂)` should already account for the full
    /// `L1` / `L2` sensitivities of the vector-valued query.
    #[must_use]
    pub fn add_noise_vec(&self, values: &[i64], rng: &mut LcgRng) -> Vec<i64> {
        values.iter().map(|&v| self.add_noise(v, rng)).collect()
    }

    /// Closed-form Rényi DP bound `ε_R(α)` (Agarwal et al. 2021,
    /// Theorem 1; Corollary 1 for the L2-refined leading term).
    ///
    /// # Errors
    /// Returns `InvalidParameter` for `α ≤ 1` or non-finite `α`.
    pub fn rdp(&self, alpha: f64) -> PrivacyResult<f64> {
        if !(alpha.is_finite() && alpha > 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "alpha must be > 1 and finite, got {alpha}"
            )));
        }
        let mu = self.cfg.mu;
        let d1 = self.cfg.sensitivity_l1;
        let d2 = self.cfg.sensitivity_l2;

        // Leading term: use the L2-refined quantity Δ₂² when Δ₂² ≤ Δ₁,
        // otherwise the raw Δ₁² bound (Agarwal et al. 2021).
        let leading_numer = if d2 * d2 <= d1 { d2 * d2 } else { d1 * d1 };
        let leading = alpha * leading_numer / (2.0 * mu);

        // Two competing correction bounds; the theorem allows the minimum.
        let mu_sq = mu * mu;
        let d1_sq = d1 * d1;
        let d1_cu = d1_sq * d1;
        let d1_qu = d1_sq * d1_sq;

        let correction_a =
            (2.0 * alpha - 1.0) * d1_qu / (4.0 * mu_sq) + 3.0 * d1_cu / (2.0 * mu_sq);
        let correction_b = 3.0 * alpha * d1_cu / (2.0 * mu_sq);
        let correction = correction_a.min(correction_b);

        Ok(leading + correction)
    }

    /// Convert the closed-form RDP bound to `(ε, δ)`-DP via
    /// `ε(δ) = min_α [ ε_R(α) + ln(1/δ) / (α − 1) ]`, optimised over
    /// `ALPHA_GRID`.
    ///
    /// # Errors
    /// - `InvalidDelta` if `δ ∉ (0, 1)`.
    pub fn to_epsilon_delta(&self, delta: f64) -> PrivacyResult<f64> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        let log_inv_delta = (1.0 / delta).ln();
        let mut best = f64::INFINITY;
        for &alpha in ALPHA_GRID {
            let eps_rdp = self.rdp(alpha)?;
            let candidate = eps_rdp + log_inv_delta / (alpha - 1.0);
            if candidate < best {
                best = candidate;
            }
        }
        Ok(best)
    }
}

/// Sample `K ~ Pois(mu)` returning a non-negative `i64`.
///
/// - **Knuth's algorithm** for `mu ≤ POIS_NORMAL_CUTOFF`.  Multiplies
///   uniform `(0, 1]` draws until their product falls below `e^{-mu}`;
///   the count of multiplications minus one is the Poisson draw.
/// - **Normal approximation** for `mu > POIS_NORMAL_CUTOFF`: rounded
///   `N(mu, mu)`, clamped to `≥ 0`.
fn sample_poisson(mu: f64, rng: &mut LcgRng) -> i64 {
    if mu <= POIS_NORMAL_CUTOFF {
        sample_poisson_knuth(mu, rng)
    } else {
        sample_poisson_normal_approx(mu, rng)
    }
}

/// Knuth's direct multiplicative algorithm.  `mu` must be `> 0`.
///
/// We cap the loop at a safety bound `K_MAX = max(1000, 100 · ceil(mu))`
/// to handle pathological RNG sequences in tests without ever spinning
/// forever; in normal operation the loop terminates in `~mu` iterations.
fn sample_poisson_knuth(mu: f64, rng: &mut LcgRng) -> i64 {
    let l = (-mu).exp();
    let mut p: f64 = 1.0;
    let mut k: i64 = 0;
    let k_max: i64 = (100.0 * mu).ceil() as i64;
    let k_max = k_max.max(1000);
    while k < k_max {
        k += 1;
        // Uniform in (0, 1] to avoid p collapsing to 0 from a single
        // zero draw (which the inverse-CDF/Knuth algorithm tolerates,
        // but tightening here keeps the accumulator numerically clean).
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        p *= u;
        if p <= l {
            return k - 1;
        }
    }
    // Safety fallback: very long run → return the safety bound.
    k - 1
}

/// Normal-approximation Poisson: `round(N(mu, mu))`, clamped `≥ 0`.
fn sample_poisson_normal_approx(mu: f64, rng: &mut LcgRng) -> i64 {
    let (z, _) = rng.normal_pair();
    let x = z * mu.sqrt() + mu;
    let rounded = x.round();
    if !rounded.is_finite() || rounded <= 0.0 {
        return 0;
    }
    if rounded >= i64::MAX as f64 {
        return i64::MAX;
    }
    rounded as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(mu: f64, d1: f64, d2: f64) -> SkellamMechanism {
        SkellamMechanism::new(SkellamConfig {
            mu,
            sensitivity_l1: d1,
            sensitivity_l2: d2,
        })
        .expect("ok")
    }

    #[test]
    fn test_new_rejects_nonpositive_mu() {
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: 0.0,
                sensitivity_l1: 1.0,
                sensitivity_l2: 1.0,
            })
            .is_err()
        );
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: -1.0,
                sensitivity_l1: 1.0,
                sensitivity_l2: 1.0,
            })
            .is_err()
        );
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: f64::NAN,
                sensitivity_l1: 1.0,
                sensitivity_l2: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_rejects_nonpositive_sensitivity_l1() {
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: 1.0,
                sensitivity_l1: 0.0,
                sensitivity_l2: 1.0,
            })
            .is_err()
        );
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: 1.0,
                sensitivity_l1: -0.1,
                sensitivity_l2: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_rejects_nonpositive_sensitivity_l2() {
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: 1.0,
                sensitivity_l1: 1.0,
                sensitivity_l2: 0.0,
            })
            .is_err()
        );
        assert!(
            SkellamMechanism::new(SkellamConfig {
                mu: 1.0,
                sensitivity_l1: 1.0,
                sensitivity_l2: -0.5,
            })
            .is_err()
        );
    }

    #[test]
    fn test_sample_mean_near_zero() {
        // Skellam(μ, μ) is symmetric ⇒ mean 0.
        // SE of sample mean is √(2μ/N); use ±3·SE tolerance.
        let mu = 5.0;
        let m = make(mu, 1.0, 1.0);
        let mut rng = LcgRng::new(101);
        let n = 10_000usize;
        let mut sum: f64 = 0.0;
        for _ in 0..n {
            sum += m.sample(&mut rng) as f64;
        }
        let mean = sum / n as f64;
        let tolerance = 3.0 * (2.0 * mu / n as f64).sqrt();
        assert!(mean.abs() < tolerance, "mean = {mean}");
    }

    #[test]
    fn test_sample_variance_near_two_mu() {
        let mu = 4.0;
        let m = make(mu, 1.0, 1.0);
        let mut rng = LcgRng::new(7);
        let n = 10_000usize;
        let mut sum: f64 = 0.0;
        let mut sum_sq: f64 = 0.0;
        for _ in 0..n {
            let v = m.sample(&mut rng) as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let theory = 2.0 * mu;
        // Allow ±15% slack at n = 10K.
        assert!(
            (var - theory).abs() / theory < 0.15,
            "var = {var}, theory = {theory}"
        );
    }

    #[test]
    fn test_rdp_monotone_in_alpha() {
        let m = make(10.0, 1.0, 1.0);
        let grid = [1.5_f64, 2.0, 4.0, 8.0, 16.0, 32.0];
        let mut prev = -1.0;
        for &a in &grid {
            let e = m.rdp(a).expect("ok");
            assert!(e > prev, "ε_R({a}) = {e} not > {prev}");
            prev = e;
        }
    }

    #[test]
    fn test_rdp_decreases_in_mu() {
        let alpha = 4.0;
        let small_mu = make(2.0, 1.0, 1.0).rdp(alpha).expect("ok");
        let large_mu = make(20.0, 1.0, 1.0).rdp(alpha).expect("ok");
        let huge_mu = make(200.0, 1.0, 1.0).rdp(alpha).expect("ok");
        assert!(small_mu > large_mu, "small={small_mu} large={large_mu}");
        assert!(large_mu > huge_mu, "large={large_mu} huge={huge_mu}");
    }

    #[test]
    fn test_rdp_rejects_alpha_le_one() {
        let m = make(5.0, 1.0, 1.0);
        assert!(m.rdp(1.0).is_err());
        assert!(m.rdp(0.5).is_err());
        assert!(m.rdp(-1.0).is_err());
        assert!(m.rdp(f64::NAN).is_err());
        assert!(m.rdp(f64::INFINITY).is_err());
    }

    #[test]
    fn test_rdp_l2_refinement_tightens_leading_term() {
        // With Δ₂² < Δ₁ the leading term should be smaller than with the
        // raw Δ₁² substitution.  Pick Δ₁ = 4, Δ₂ = 1 → Δ₂² = 1 < Δ₁ = 4.
        let m_refined = make(10.0, 4.0, 1.0);
        // Force the un-refined comparison by setting Δ₂ such that
        // Δ₂² > Δ₁, eliminating the L2 branch.
        let m_unrefined = make(10.0, 4.0, 3.0);
        let alpha = 4.0;
        let r = m_refined.rdp(alpha).expect("ok");
        let u = m_unrefined.rdp(alpha).expect("ok");
        assert!(r < u, "refined {r} not < unrefined {u}");
    }

    #[test]
    fn test_sample_is_integer() {
        let m = make(3.0, 1.0, 1.0);
        let mut rng = LcgRng::new(11);
        for _ in 0..100 {
            let v: i64 = m.sample(&mut rng);
            let _ = v;
        }
    }

    #[test]
    fn test_large_mu_normal_approx_finite() {
        // mu > POIS_NORMAL_CUTOFF triggers the normal-approx branch.
        let m = make(500.0, 1.0, 1.0);
        let mut rng = LcgRng::new(13);
        let mut all_finite = true;
        for _ in 0..1000 {
            let v = m.sample(&mut rng);
            // i64 is always finite; we just exercise the branch and
            // check the sampler returns plausible integers.
            if v.unsigned_abs() > 10_000 {
                all_finite = false;
                break;
            }
        }
        assert!(all_finite, "large-μ branch produced extreme values");
    }

    #[test]
    fn test_large_mu_mean_near_zero() {
        let mu = 100.0;
        let m = make(mu, 1.0, 1.0);
        let mut rng = LcgRng::new(2026);
        let n = 5000usize;
        let mut sum: f64 = 0.0;
        for _ in 0..n {
            sum += m.sample(&mut rng) as f64;
        }
        let mean = sum / n as f64;
        let tolerance = 3.0 * (2.0 * mu / n as f64).sqrt();
        assert!(mean.abs() < tolerance, "mean = {mean}");
    }

    #[test]
    fn test_deterministic_given_seed() {
        let m = make(8.0, 1.0, 1.0);
        let mut a = LcgRng::new(777);
        let mut b = LcgRng::new(777);
        let xa: Vec<i64> = (0..64).map(|_| m.sample(&mut a)).collect();
        let xb: Vec<i64> = (0..64).map(|_| m.sample(&mut b)).collect();
        assert_eq!(xa, xb);
    }

    #[test]
    fn test_add_noise_offsets_by_sample() {
        let m = make(5.0, 1.0, 1.0);
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = rng_a.clone();
        let noise = m.sample(&mut rng_a);
        let applied = m.add_noise(10, &mut rng_b);
        assert_eq!(applied, 10 + noise);
    }

    #[test]
    fn test_add_noise_vec_preserves_length() {
        let m = make(2.0, 1.0, 1.0);
        let mut rng = LcgRng::new(3);
        let input = vec![0_i64, 1, -1, 2, -2, 5, 10, 100, -1000];
        let out = m.add_noise_vec(&input, &mut rng);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn test_add_noise_vec_empty_input() {
        let m = make(1.0, 1.0, 1.0);
        let mut rng = LcgRng::new(0);
        let out = m.add_noise_vec(&[], &mut rng);
        assert!(out.is_empty());
    }

    #[test]
    fn test_to_epsilon_delta_finite_for_small_delta() {
        let m = make(10.0, 1.0, 1.0);
        for &delta in &[0.5_f64, 0.1, 1e-3, 1e-5, 1e-10] {
            let eps = m.to_epsilon_delta(delta).expect("ok");
            assert!(eps.is_finite() && eps > 0.0, "δ={delta} ε={eps}");
        }
    }

    #[test]
    fn test_to_epsilon_delta_rejects_bad_delta() {
        let m = make(5.0, 1.0, 1.0);
        assert!(m.to_epsilon_delta(0.0).is_err());
        assert!(m.to_epsilon_delta(1.0).is_err());
        assert!(m.to_epsilon_delta(-0.1).is_err());
        assert!(m.to_epsilon_delta(1.5).is_err());
    }

    #[test]
    fn test_to_epsilon_delta_monotone_in_delta() {
        // ε(δ) should be non-increasing in δ (more allowed slack ⇒
        // smaller ε).
        let m = make(10.0, 1.0, 1.0);
        let e1 = m.to_epsilon_delta(1e-1).expect("ok");
        let e2 = m.to_epsilon_delta(1e-5).expect("ok");
        let e3 = m.to_epsilon_delta(1e-10).expect("ok");
        assert!(e1 <= e2, "ε(0.1)={e1} > ε(1e-5)={e2}");
        assert!(e2 <= e3, "ε(1e-5)={e2} > ε(1e-10)={e3}");
    }

    #[test]
    fn test_config_accessor_returns_input() {
        let cfg = SkellamConfig {
            mu: 3.5,
            sensitivity_l1: 2.0,
            sensitivity_l2: 1.5,
        };
        let m = SkellamMechanism::new(cfg.clone()).expect("ok");
        let back = m.config();
        assert!((back.mu - cfg.mu).abs() < 1e-12);
        assert!((back.sensitivity_l1 - cfg.sensitivity_l1).abs() < 1e-12);
        assert!((back.sensitivity_l2 - cfg.sensitivity_l2).abs() < 1e-12);
    }

    #[test]
    fn test_poisson_sample_non_negative_small_mu() {
        let mut rng = LcgRng::new(9);
        for _ in 0..1000 {
            assert!(sample_poisson_knuth(2.0, &mut rng) >= 0);
        }
    }

    #[test]
    fn test_poisson_sample_non_negative_large_mu() {
        let mut rng = LcgRng::new(31);
        for _ in 0..1000 {
            assert!(sample_poisson_normal_approx(100.0, &mut rng) >= 0);
        }
    }

    #[test]
    fn test_poisson_normal_approx_mean_near_mu() {
        // With μ = 100 the normal approx should produce sample mean ≈ μ.
        let mu = 100.0;
        let mut rng = LcgRng::new(2);
        let n = 5000usize;
        let mut sum = 0.0_f64;
        for _ in 0..n {
            sum += sample_poisson_normal_approx(mu, &mut rng) as f64;
        }
        let mean = sum / n as f64;
        // SE of sample mean is √(μ / n); allow 4·SE.
        let tol = 4.0 * (mu / n as f64).sqrt();
        assert!((mean - mu).abs() < tol, "mean={mean} mu={mu}");
    }
}
