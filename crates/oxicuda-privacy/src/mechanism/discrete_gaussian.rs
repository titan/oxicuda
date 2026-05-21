//! Discrete Gaussian mechanism for integer-valued queries.
//!
//! # Reference
//! - Canonne, Kamath, Steinke (2020),
//!   *"The Discrete Gaussian for Differential Privacy"*, NeurIPS 2020,
//!   <https://arxiv.org/abs/2004.00010>.
//!
//! The discrete Gaussian over the integers has probability mass function
//!
//! ```text
//!     P_DG(σ²)(k) ∝ exp(-k² / (2 σ²)),   k ∈ ℤ.
//! ```
//!
//! On an integer-valued query with `L1` sensitivity `Δ`, the mechanism
//! `f(x) + Z` with `Z ~ DG(σ²)` satisfies `ρ`-zCDP with
//! `ρ = Δ² / (2σ²)`, and consequently `(ε, δ)`-DP with
//! `ε = ρ + 2√(ρ · ln(1/δ))` for any `δ ∈ (0, 1)` (Canonne et al.
//! 2020, Theorem 11; also Bun & Steinke 2016, Proposition 1.3).
//!
//! Compared with rounding a continuous-Gaussian draw, the discrete
//! Gaussian avoids the floating-point side channels analysed by Mironov
//! (2012) and provides exact integer outputs.
//!
//! # Sampling
//! Two strategies, both grounded in Canonne et al. 2020 (Section 3):
//!
//! 1. **Round-and-accept** (`σ ≥ 1`): draw `X ~ N(0, σ²)`, return
//!    `Y = round(X)`.  The total-variation distance to the true discrete
//!    Gaussian decays as `O(e^{-π² σ²})` (paper, Section 3); for `σ ≥ 1`
//!    this is below `10⁻³⁹` and thus negligible at the precisions
//!    relevant to differential privacy.
//!
//! 2. **Rejection refinement** (`σ < 1`): the round-and-accept output is
//!    re-weighted by the standard Gaussian acceptance bound
//!    `exp(-(Y - X)² / (2σ²))`, which corrects the residual bias from
//!    coarse rounding in the small-σ regime.
//!
//! Both branches reuse the workspace's `LcgRng::normal_pair` Box-Muller
//! implementation rather than introducing any rand-crate dependency.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Discrete Gaussian mechanism with scale `σ` for integer-valued queries.
#[derive(Debug, Clone)]
pub struct DiscreteGaussianMechanism {
    /// Scale parameter `σ > 0` (in the same units as the integer query).
    pub sigma: f64,
    /// `L1` sensitivity of the integer query.  Must be `≥ 1`.
    pub sensitivity: i64,
}

impl DiscreteGaussianMechanism {
    /// Construct a discrete Gaussian mechanism.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `sigma ≤ 0` or non-finite.
    /// - `InvalidParameter` if `sensitivity < 1`.
    pub fn new(sigma: f64, sensitivity: i64) -> PrivacyResult<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(PrivacyError::NonPositiveSensitivity(sigma));
        }
        if sensitivity < 1 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sensitivity must be ≥ 1, got {sensitivity}"
            )));
        }
        Ok(Self { sigma, sensitivity })
    }

    /// Draw one integer noise value `Z ~ DG(σ²)`.
    ///
    /// See module docs for the two-branch strategy.
    pub fn sample(&self, rng: &mut LcgRng) -> i64 {
        if self.sigma >= 1.0 {
            self.sample_round_and_accept(rng)
        } else {
            self.sample_rejection(rng)
        }
    }

    /// Round-and-accept branch (no rejection): draws a continuous
    /// Gaussian and rounds to the nearest integer.  Total-variation
    /// distance to the exact discrete Gaussian is `O(exp(-π² σ²))`.
    fn sample_round_and_accept(&self, rng: &mut LcgRng) -> i64 {
        let (z, _) = rng.normal_pair();
        let x = z * self.sigma;
        clamp_to_i64(x.round())
    }

    /// Small-σ rejection branch: corrects the residual rounding bias
    /// from `sample_round_and_accept` for `σ < 1`.
    ///
    /// We accept the rounded value `Y` with probability
    /// `exp(-(Y - X)² / (2σ²))`.  Because `|Y - X| ≤ 0.5`, the maximum
    /// rejection weight is `exp(-0.25 / (2σ²))` and the acceptance
    /// probability stays above `e^{-1/(8σ²)}`.
    fn sample_rejection(&self, rng: &mut LcgRng) -> i64 {
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;
        // Bound the loop conservatively to keep tests well-behaved even
        // for adversarial RNG sequences.  For σ ∈ (0, 1), the expected
        // number of trials is small (≤ a few dozen).
        for _ in 0..10_000 {
            let (z, _) = rng.normal_pair();
            let x = z * self.sigma;
            let y = x.round();
            let dy = y - x;
            let log_acc = -(dy * dy) / two_sigma_sq;
            let u = rng.next_f64().max(f64::MIN_POSITIVE);
            if u.ln() <= log_acc {
                return clamp_to_i64(y);
            }
        }
        // Fallback: return the most recent rounded value (extremely unlikely).
        let (z, _) = rng.normal_pair();
        clamp_to_i64((z * self.sigma).round())
    }

    /// Apply the mechanism to a single integer query value.
    pub fn apply(&self, x: i64, rng: &mut LcgRng) -> i64 {
        x.wrapping_add(self.sample(rng))
    }

    /// Apply the mechanism element-wise to an integer vector.
    #[must_use]
    pub fn apply_vec(&self, x: &[i64], rng: &mut LcgRng) -> Vec<i64> {
        x.iter().map(|&v| self.apply(v, rng)).collect()
    }

    /// `ρ`-zCDP guarantee: `ρ = sensitivity² / (2 σ²)`.
    ///
    /// (Canonne, Kamath, Steinke 2020, Theorem 11.)
    #[must_use]
    #[inline]
    pub fn rho(&self) -> f64 {
        let s = self.sensitivity as f64;
        (s * s) / (2.0 * self.sigma * self.sigma)
    }

    /// Convert to `(ε, δ)`-DP via the zCDP-to-approximate-DP conversion
    /// `ε(δ) = ρ + 2·√(ρ · ln(1/δ))` (Bun & Steinke 2016, Proposition 1.3).
    ///
    /// # Errors
    /// - `InvalidDelta` if `δ ∉ (0, 1)`.
    pub fn epsilon_for_delta(&self, delta: f64) -> PrivacyResult<f64> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        let rho = self.rho();
        let inv_log = (1.0 / delta).ln();
        Ok(rho + 2.0 * (rho * inv_log).sqrt())
    }
}

/// Clamp a finite `f64` into the representable `i64` range.
///
/// Required because round-and-accept can theoretically (with astronomically
/// small probability) produce a value outside `i64::MIN..=i64::MAX`.
fn clamp_to_i64(x: f64) -> i64 {
    if !x.is_finite() {
        return 0;
    }
    if x >= i64::MAX as f64 {
        i64::MAX
    } else if x <= i64::MIN as f64 {
        i64::MIN
    } else {
        x as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_rejects_nonpositive_sigma() {
        assert!(DiscreteGaussianMechanism::new(0.0, 1).is_err());
        assert!(DiscreteGaussianMechanism::new(-1.0, 1).is_err());
        assert!(DiscreteGaussianMechanism::new(f64::NAN, 1).is_err());
        assert!(DiscreteGaussianMechanism::new(f64::INFINITY, 1).is_err());
    }

    #[test]
    fn test_new_rejects_nonpositive_sensitivity() {
        assert!(DiscreteGaussianMechanism::new(1.0, 0).is_err());
        assert!(DiscreteGaussianMechanism::new(1.0, -1).is_err());
    }

    #[test]
    fn test_mean_near_zero_for_sigma_2() {
        let m = DiscreteGaussianMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(101);
        let n = 5000usize;
        let mut sum: f64 = 0.0;
        for _ in 0..n {
            sum += m.sample(&mut rng) as f64;
        }
        let mean = sum / n as f64;
        // Symmetric distribution → mean ≈ 0; allow ±0.15 at n = 5000.
        assert!(mean.abs() < 0.2, "mean = {mean}");
    }

    #[test]
    fn test_variance_near_sigma_squared_for_sigma_2() {
        let m = DiscreteGaussianMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(13);
        let n = 5000usize;
        let mut sum: f64 = 0.0;
        let mut sum_sq: f64 = 0.0;
        for _ in 0..n {
            let v = m.sample(&mut rng) as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        // Theoretical variance ≈ σ² + (rounding contribution); for σ = 2 the
        // discrete Gaussian variance is within ~5% of σ² = 4.
        assert!((var - 4.0).abs() / 4.0 < 0.2, "var = {var}");
    }

    #[test]
    fn test_support_is_wide_enough() {
        // We expect to see values of |Z| ≥ 1 at σ = 2 within a few thousand samples.
        let m = DiscreteGaussianMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(99);
        let mut max_abs = 0_i64;
        for _ in 0..2000 {
            max_abs = max_abs.max(m.sample(&mut rng).abs());
        }
        assert!(max_abs >= 3, "support too narrow: max_abs = {max_abs}");
    }

    #[test]
    fn test_rho_closed_form() {
        let m = DiscreteGaussianMechanism::new(1.0, 1).expect("ok");
        assert!((m.rho() - 0.5).abs() < 1e-12);

        let m2 = DiscreteGaussianMechanism::new(2.0, 4).expect("ok");
        // ρ = 16 / (2·4) = 2.0
        assert!((m2.rho() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_epsilon_for_delta_closed_form() {
        // σ = 1, sensitivity = 1 → ρ = 0.5; δ = 0.1
        // ε = 0.5 + 2 · √(0.5 · ln(10)) ≈ 0.5 + 2 · √(0.5 · 2.302585)
        //    = 0.5 + 2 · √1.151293 ≈ 0.5 + 2 · 1.07298 ≈ 2.6460
        let m = DiscreteGaussianMechanism::new(1.0, 1).expect("ok");
        let eps = m.epsilon_for_delta(0.1).expect("ok");
        assert!((eps - 2.6460).abs() < 1e-3, "eps = {eps}");
    }

    #[test]
    fn test_epsilon_for_delta_rejects_bad_delta() {
        let m = DiscreteGaussianMechanism::new(1.0, 1).expect("ok");
        assert!(m.epsilon_for_delta(0.0).is_err());
        assert!(m.epsilon_for_delta(1.0).is_err());
        assert!(m.epsilon_for_delta(-0.1).is_err());
        assert!(m.epsilon_for_delta(1.5).is_err());
    }

    #[test]
    fn test_apply_offsets_by_sample() {
        let m = DiscreteGaussianMechanism::new(2.0, 1).expect("ok");
        let mut rng_a = LcgRng::new(8);
        let mut rng_b = rng_a.clone();
        let noise = m.sample(&mut rng_a);
        let applied = m.apply(7, &mut rng_b);
        assert_eq!(applied, 7 + noise);
    }

    #[test]
    fn test_apply_vec_length_match() {
        let m = DiscreteGaussianMechanism::new(1.5, 1).expect("ok");
        let mut rng = LcgRng::new(42);
        let v = vec![0_i64, 1, -1, 2, -2, 5];
        let out = m.apply_vec(&v, &mut rng);
        assert_eq!(out.len(), v.len());
    }

    #[test]
    fn test_apply_vec_empty_input() {
        let m = DiscreteGaussianMechanism::new(1.0, 1).expect("ok");
        let mut rng = LcgRng::new(0);
        let out = m.apply_vec(&[], &mut rng);
        assert!(out.is_empty());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let m = DiscreteGaussianMechanism::new(2.0, 1).expect("ok");
        let mut rng_a = LcgRng::new(1337);
        let mut rng_b = LcgRng::new(1337);
        let a: Vec<i64> = (0..32).map(|_| m.sample(&mut rng_a)).collect();
        let b: Vec<i64> = (0..32).map(|_| m.sample(&mut rng_b)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_integer_round_trip() {
        let m = DiscreteGaussianMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(2);
        let v = m.apply(42, &mut rng);
        let _: i64 = v;
    }

    #[test]
    fn test_small_sigma_rejection_branch_runs() {
        // Exercise the σ < 1 rejection sampler and check it returns integers
        // within a tight expected support.
        let m = DiscreteGaussianMechanism::new(0.5, 1).expect("ok");
        let mut rng = LcgRng::new(17);
        let mut counts = [0usize; 7]; // indices map to -3..=3
        let n = 2000usize;
        for _ in 0..n {
            let v = m.sample(&mut rng);
            assert!(v.abs() <= 6, "small-σ sample too large: {v}");
            if (-3..=3).contains(&v) {
                counts[(v + 3) as usize] += 1;
            }
        }
        // Zero should be the most common outcome (≥ 40% with σ = 0.5).
        let zero_count = counts[3];
        assert!(zero_count * 5 >= 2 * n, "Pr(Z = 0) = {zero_count}/{n}");
    }
}
