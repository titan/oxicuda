//! Discrete Laplace (Geometric) mechanism for integer-valued queries.
//!
//! # Reference
//! - Ghosh, Roughgarden, Sundararajan (2012),
//!   *"Universally Utility-Maximizing Privacy Mechanisms"*, SIAM J. Comp.
//! - Canonne, Kamath, Steinke (2020),
//!   *"The Discrete Gaussian for Differential Privacy"*, NeurIPS 2020
//!   (Section 2 reviews the Discrete Laplace / Geometric mechanism).
//!
//! The discrete Laplace distribution has probability mass function
//!
//! ```text
//!     P_DL(t)(k) = (1 - e^{-1/t}) / (1 + e^{-1/t}) · e^{-|k|/t},   k ∈ ℤ
//! ```
//!
//! where `t > 0` is a scale parameter.  When applied to an integer-valued
//! query with `L1` sensitivity `Δ`, choosing `t = Δ / ε` yields a clean
//! `(ε, 0)`-differentially-private mechanism with no floating-point
//! artifacts (modulo the rounding inside the inverse-CDF call) — the
//! integer-output analogue of the continuous Laplace mechanism.
//!
//! # Sampling
//! Writes `Z = X - Y` with `X, Y ~ Geometric(p)` independent and
//! `p = 1 - e^{-1/t}`.  Geometric draws use the inverse-CDF transform
//! `k = ⌊log(1 - u) / log(1 - p)⌋` for `u ~ Uniform[0, 1)`.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Discrete Laplace (a.k.a. two-sided geometric) mechanism.
///
/// Provides `(ε, 0)`-DP for queries with integer-valued output and
/// `L1` sensitivity at most `sensitivity`, using `scale = sensitivity / ε`.
#[derive(Debug, Clone)]
pub struct DiscreteLaplaceMechanism {
    /// Scale parameter `t = sensitivity / ε`.  Must be `> 0`.
    pub scale: f64,
    /// `L1` sensitivity (integer-valued query); must be `≥ 1`.
    pub sensitivity: i64,
}

impl DiscreteLaplaceMechanism {
    /// Construct a discrete Laplace mechanism with explicit scale.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `scale ≤ 0`.
    /// - `InvalidParameter` if `sensitivity < 1`.
    pub fn new(scale: f64, sensitivity: i64) -> PrivacyResult<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(PrivacyError::NonPositiveSensitivity(scale));
        }
        if sensitivity < 1 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sensitivity must be ≥ 1, got {sensitivity}"
            )));
        }
        Ok(Self { scale, sensitivity })
    }

    /// Construct a discrete Laplace mechanism from `(ε, sensitivity)` such that
    /// `scale = sensitivity / ε`.  Provides `(ε, 0)`-DP.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon ≤ 0`.
    /// - `InvalidParameter` if `sensitivity < 1`.
    pub fn for_epsilon(epsilon: f64, sensitivity: i64) -> PrivacyResult<Self> {
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity < 1 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sensitivity must be ≥ 1, got {sensitivity}"
            )));
        }
        let scale = sensitivity as f64 / epsilon;
        Ok(Self { scale, sensitivity })
    }

    /// Effective ε of this configuration for the stated `sensitivity`.
    ///
    /// `ε = sensitivity / scale`.
    #[must_use]
    #[inline]
    pub fn epsilon(&self) -> f64 {
        self.sensitivity as f64 / self.scale
    }

    /// Sample one integer noise value `Z ~ DiscreteLaplace(scale)`.
    ///
    /// Uses the geometric-difference identity `Z = X - Y` with two
    /// independent `Geometric(p = 1 - exp(-1/t))` draws.
    pub fn sample(&self, rng: &mut LcgRng) -> i64 {
        let p = 1.0 - (-1.0 / self.scale).exp();
        let x = sample_geometric(p, rng);
        let y = sample_geometric(p, rng);
        x - y
    }

    /// Apply the mechanism to a single integer query value.
    pub fn apply(&self, x: i64, rng: &mut LcgRng) -> i64 {
        x.wrapping_add(self.sample(rng))
    }

    /// Apply the mechanism element-wise to an integer vector.
    ///
    /// Each coordinate gets its own independent discrete-Laplace draw.
    /// (The L1 sensitivity in the construction refers to the *total*
    /// query, so callers using this for a vector query should size
    /// `sensitivity` accordingly.)
    #[must_use]
    pub fn apply_vec(&self, x: &[i64], rng: &mut LcgRng) -> Vec<i64> {
        x.iter().map(|&v| self.apply(v, rng)).collect()
    }
}

/// Sample `K ~ Geometric(p)` with `P(K = k) = (1-p)^k · p`, `k = 0, 1, 2, ...`.
///
/// Uses the inverse-CDF transform `k = ⌊log(1 - u) / log(1 - p)⌋` for
/// `u ~ Uniform[0, 1)`.  Returns `0` immediately for `p == 1.0` and
/// guards against `p` very close to `1` (then `log(1 - p) → -∞` and
/// `k = 0`).  For very small `p`, the value of `k` can grow large; we
/// clamp to `i64::MAX / 2` to avoid overflow when forming `X - Y`.
fn sample_geometric(p: f64, rng: &mut LcgRng) -> i64 {
    debug_assert!(p > 0.0 && p <= 1.0);
    if p >= 1.0 {
        return 0;
    }
    // u ∈ [0, 1).  Use (1 - u) ∈ (0, 1] so the log is non-positive.
    let u = rng.next_f64();
    let one_minus_u = (1.0 - u).max(f64::MIN_POSITIVE);
    let log_one_minus_p = (1.0 - p).ln();
    if log_one_minus_p >= -f64::MIN_POSITIVE {
        // p ≈ 0 → distribution is concentrated near +∞; clamp safely.
        return i64::MAX / 4;
    }
    let k = (one_minus_u.ln() / log_one_minus_p).floor();
    // Clamp to keep X - Y in i64 range even for absurdly small p.
    let bound = (i64::MAX / 4) as f64;
    if k >= bound {
        i64::MAX / 4
    } else if k < 0.0 {
        0
    } else {
        k as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_rejects_nonpositive_scale() {
        assert!(DiscreteLaplaceMechanism::new(0.0, 1).is_err());
        assert!(DiscreteLaplaceMechanism::new(-1.0, 1).is_err());
        assert!(DiscreteLaplaceMechanism::new(f64::NAN, 1).is_err());
        assert!(DiscreteLaplaceMechanism::new(f64::INFINITY, 1).is_err());
    }

    #[test]
    fn test_new_rejects_nonpositive_sensitivity() {
        assert!(DiscreteLaplaceMechanism::new(1.0, 0).is_err());
        assert!(DiscreteLaplaceMechanism::new(1.0, -3).is_err());
    }

    #[test]
    fn test_for_epsilon_rejects_bad_epsilon() {
        assert!(DiscreteLaplaceMechanism::for_epsilon(0.0, 1).is_err());
        assert!(DiscreteLaplaceMechanism::for_epsilon(-0.5, 1).is_err());
        assert!(DiscreteLaplaceMechanism::for_epsilon(f64::NAN, 1).is_err());
    }

    #[test]
    fn test_for_epsilon_computes_scale_correctly() {
        let m = DiscreteLaplaceMechanism::for_epsilon(0.5, 1).expect("ok");
        assert!((m.scale - 2.0).abs() < 1e-12);
        assert_eq!(m.sensitivity, 1);

        let m2 = DiscreteLaplaceMechanism::for_epsilon(1.0, 5).expect("ok");
        assert!((m2.scale - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_epsilon_roundtrip() {
        let m = DiscreteLaplaceMechanism::for_epsilon(0.7, 3).expect("ok");
        assert!((m.epsilon() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_mean_near_zero_for_scale_2() {
        let m = DiscreteLaplaceMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(123);
        let n = 5000usize;
        let mut sum: f64 = 0.0;
        for _ in 0..n {
            sum += m.sample(&mut rng) as f64;
        }
        let mean = sum / n as f64;
        // Symmetric distribution → mean ≈ 0; allow ±0.2 with this sample size.
        assert!(mean.abs() < 0.25, "mean = {mean}");
    }

    #[test]
    fn test_variance_matches_theory_for_scale_2() {
        // Var(DiscreteLaplace(t)) = 2 e^{-1/t} / (1 - e^{-1/t})²
        let m = DiscreteLaplaceMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(7);
        let n = 5000usize;
        let mut sum_sq: f64 = 0.0;
        let mut sum: f64 = 0.0;
        for _ in 0..n {
            let v = m.sample(&mut rng) as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let q = (-1.0_f64 / 2.0).exp();
        let theory = 2.0 * q / (1.0 - q).powi(2);
        // Allow ±20% slack at n = 5000.
        assert!(
            (var - theory).abs() / theory < 0.2,
            "var = {var}, theory = {theory}"
        );
    }

    #[test]
    fn test_apply_offsets_by_sample() {
        let m = DiscreteLaplaceMechanism::new(3.0, 1).expect("ok");
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = rng_a.clone();
        let noise = m.sample(&mut rng_a);
        let applied = m.apply(10, &mut rng_b);
        assert_eq!(applied, 10 + noise);
    }

    #[test]
    fn test_apply_vec_length_matches() {
        let m = DiscreteLaplaceMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(4);
        let input = vec![0_i64, 1, 2, 3, 4];
        let output = m.apply_vec(&input, &mut rng);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let m = DiscreteLaplaceMechanism::new(1.5, 1).expect("ok");
        let mut rng_a = LcgRng::new(2026);
        let mut rng_b = LcgRng::new(2026);
        let a: Vec<i64> = (0..32).map(|_| m.sample(&mut rng_a)).collect();
        let b: Vec<i64> = (0..32).map(|_| m.sample(&mut rng_b)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_samples_bounded_for_normal_scale() {
        // For scale = 2, samples beyond ±100 are astronomically unlikely.
        let m = DiscreteLaplaceMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(11);
        for _ in 0..5000 {
            let v = m.sample(&mut rng);
            assert!(v.abs() <= 200, "got unbounded sample {v}");
        }
    }

    #[test]
    fn test_sample_distribution_symmetric() {
        let m = DiscreteLaplaceMechanism::new(2.5, 1).expect("ok");
        let mut rng = LcgRng::new(55);
        let n = 6000usize;
        let mut pos = 0usize;
        let mut neg = 0usize;
        for _ in 0..n {
            let v = m.sample(&mut rng);
            if v > 0 {
                pos += 1;
            } else if v < 0 {
                neg += 1;
            }
        }
        // Symmetric → roughly equal counts of positive and negative samples.
        let diff = (pos as i64 - neg as i64).unsigned_abs() as usize;
        assert!(diff < n / 10, "asymmetric pos={pos}, neg={neg}");
    }

    #[test]
    fn test_geometric_sampler_is_nonneg() {
        let mut rng = LcgRng::new(2);
        let p = 0.5;
        for _ in 0..1000 {
            assert!(sample_geometric(p, &mut rng) >= 0);
        }
    }

    #[test]
    fn test_for_epsilon_low_epsilon_has_large_scale() {
        let m = DiscreteLaplaceMechanism::for_epsilon(0.01, 1).expect("ok");
        assert!(m.scale > 50.0);
    }

    #[test]
    fn test_integer_round_trip_preserved() {
        // Even after noise, the result is integer-valued.
        let m = DiscreteLaplaceMechanism::new(2.0, 1).expect("ok");
        let mut rng = LcgRng::new(3);
        let v = m.apply(42, &mut rng);
        // Compile-time guarantee that v is i64; this assert is trivially true
        // but documents intent.
        let _: i64 = v;
    }
}
