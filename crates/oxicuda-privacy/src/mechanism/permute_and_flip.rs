//! Permute-and-Flip mechanism (McKenna & Sheldon, 2020).
//!
//! Reference: McKenna & Sheldon, "Permute-and-Flip: A new mechanism for
//! differentially private selection", NeurIPS 2020.
//!
//! Permute-and-Flip is an `(ε, 0)`-differentially private selection mechanism
//! that, like the exponential mechanism, selects a high-quality outcome from a
//! discrete candidate set. It **never has larger expected error** than the
//! exponential mechanism and is strictly better for most score profiles — it is
//! the *optimal* (Pareto) mechanism for the expected-error / regret objective.
//!
//! # Protocol
//! Let `q_i` be the quality score of candidate `i`, `Δq` the global
//! sensitivity, and `q* = max_i q_i`. Draw a uniformly random permutation of
//! the candidates; iterate through it and for each candidate `i` flip a coin
//! that lands heads with probability
//!
//! `p_i = exp( (ε / (2·Δq)) · (q_i − q*) ) ∈ (0, 1]`.
//!
//! Return the first candidate whose coin lands heads. Because `p_* = 1` for an
//! optimal candidate, the loop is guaranteed to terminate having visited each
//! candidate at most once (sampling *without replacement*, in contrast to the
//! "Flip" / rejection view of the exponential mechanism which samples *with*
//! replacement).
//!
//! # Privacy
//! The mechanism is `(ε, 0)`-DP: sampling weights match the exponential
//! mechanism's `exp(ε·q_i / (2·Δq))` up to the without-replacement coupling,
//! and the privacy proof (Theorem 1 of the paper) gives the same `ε` guarantee.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the Permute-and-Flip mechanism.
#[derive(Debug, Clone)]
pub struct PermuteFlipConfig {
    /// Privacy parameter ε > 0.
    pub epsilon: f64,
    /// Global sensitivity Δq > 0 of the quality function.
    pub sensitivity: f64,
}

impl PermuteFlipConfig {
    /// Construct and validate a `PermuteFlipConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` or `NonPositiveSensitivity` on invalid params.
    pub fn new(epsilon: f64, sensitivity: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        Ok(Self {
            epsilon,
            sensitivity,
        })
    }

    /// Selection-weight exponent factor `ε / (2·Δq)`.
    #[must_use]
    fn scale(&self) -> f64 {
        self.epsilon / (2.0 * self.sensitivity)
    }
}

/// Run the Permute-and-Flip mechanism over a discrete set of `scores`.
///
/// Returns the index of the selected candidate. Provides `(ε, 0)`-DP.
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `NonPositiveEpsilon` / `NonPositiveSensitivity` if the config is invalid.
pub fn permute_and_flip(
    scores: &[f64],
    cfg: &PermuteFlipConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<usize> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if cfg.sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
    }

    let scale = cfg.scale();
    let q_max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Build the candidate index list and shuffle it (Fisher-Yates) so we visit
    // candidates in a uniformly-random order without replacement. We draw the
    // swap index from the *high* bits of the LCG via `next_f64()` because the
    // low bits of a 64-bit LCG have short periods (modulo on `next_u64()` would
    // bias the permutation).
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        // j ∈ [0, i] uniformly.
        let j = ((rng.next_f64() * (i as f64 + 1.0)) as usize).min(i);
        order.swap(i, j);
    }

    for &idx in &order {
        // p_i = exp(scale · (q_i − q*)) ∈ (0, 1]; equals 1 for an argmax.
        let p_i = (scale * (scores[idx] - q_max)).exp();
        if rng.next_f64() < p_i {
            return Ok(idx);
        }
    }

    // Mathematically unreachable (the argmax always flips heads), but if a
    // pathological RNG draw skips it we return the last visited argmax-style
    // candidate deterministically rather than panicking.
    let mut best_idx = order[0];
    let mut best_val = scores[best_idx];
    for &idx in &order {
        if scores[idx] > best_val {
            best_val = scores[idx];
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

/// Empirical selection probabilities by running the mechanism `trials` times.
///
/// Useful for diagnostics / unit tests; returns a length-`k` vector summing to 1.
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `InvalidParameter` if `trials == 0`.
/// - Propagates configuration errors from [`permute_and_flip`].
pub fn permute_flip_empirical_probs(
    scores: &[f64],
    cfg: &PermuteFlipConfig,
    trials: usize,
    rng: &mut LcgRng,
) -> PrivacyResult<Vec<f64>> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if trials == 0 {
        return Err(PrivacyError::InvalidParameter("trials must be ≥ 1".into()));
    }
    let mut counts = vec![0u64; scores.len()];
    for _ in 0..trials {
        let idx = permute_and_flip(scores, cfg, rng)?;
        counts[idx] += 1;
    }
    let inv = 1.0 / trials as f64;
    Ok(counts.iter().map(|&c| c as f64 * inv).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PermuteFlipConfig {
        PermuteFlipConfig::new(1.0, 1.0).expect("valid config")
    }

    #[test]
    fn config_rejects_non_positive_epsilon() {
        assert!(PermuteFlipConfig::new(0.0, 1.0).is_err());
        assert!(PermuteFlipConfig::new(-1.0, 1.0).is_err());
    }

    #[test]
    fn config_rejects_non_positive_sensitivity() {
        assert!(PermuteFlipConfig::new(1.0, 0.0).is_err());
        assert!(PermuteFlipConfig::new(1.0, -2.0).is_err());
    }

    #[test]
    fn returns_valid_index() {
        let scores = vec![1.0, 5.0, 2.0, 0.5];
        let c = cfg();
        let mut rng = LcgRng::new(7);
        for _ in 0..200 {
            let idx = permute_and_flip(&scores, &c, &mut rng).expect("ok");
            assert!(idx < scores.len());
        }
    }

    #[test]
    fn empty_input_errors() {
        let c = cfg();
        let mut rng = LcgRng::new(1);
        assert!(permute_and_flip(&[], &c, &mut rng).is_err());
    }

    #[test]
    fn single_candidate_always_selected() {
        let scores = vec![42.0];
        let c = cfg();
        let mut rng = LcgRng::new(3);
        for _ in 0..50 {
            assert_eq!(permute_and_flip(&scores, &c, &mut rng).expect("ok"), 0);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let scores = vec![1.0, 2.0, 3.0, 1.5, 0.0];
        let c = cfg();
        let mut a = LcgRng::new(123);
        let mut b = LcgRng::new(123);
        for _ in 0..100 {
            assert_eq!(
                permute_and_flip(&scores, &c, &mut a).expect("ok"),
                permute_and_flip(&scores, &c, &mut b).expect("ok")
            );
        }
    }

    #[test]
    fn argmax_is_most_frequent() {
        // High ε concentrates mass on the best score.
        let scores = vec![0.0, 0.0, 10.0, 0.0];
        let c = PermuteFlipConfig::new(4.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(99);
        let probs = permute_flip_empirical_probs(&scores, &c, 4000, &mut rng).expect("ok");
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .map(|(i, _)| i)
            .expect("non-empty");
        assert_eq!(best, 2, "argmax candidate must be selected most often");
        assert!(probs[2] > 0.8, "best prob = {}", probs[2]);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let scores = vec![1.0, 2.0, 3.0];
        let c = cfg();
        let mut rng = LcgRng::new(5);
        let probs = permute_flip_empirical_probs(&scores, &c, 1000, &mut rng).expect("ok");
        let s: f64 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum = {s}");
    }

    #[test]
    fn empirical_zero_trials_errors() {
        let scores = vec![1.0, 2.0];
        let c = cfg();
        let mut rng = LcgRng::new(0);
        assert!(permute_flip_empirical_probs(&scores, &c, 0, &mut rng).is_err());
    }

    #[test]
    fn higher_epsilon_increases_best_mass() {
        let scores = vec![0.0, 3.0, 0.0];
        let lo = PermuteFlipConfig::new(0.2, 1.0).expect("ok");
        let hi = PermuteFlipConfig::new(3.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(2024);
        let p_lo = permute_flip_empirical_probs(&scores, &lo, 3000, &mut rng).expect("ok");
        let p_hi = permute_flip_empirical_probs(&scores, &hi, 3000, &mut rng).expect("ok");
        assert!(
            p_hi[1] > p_lo[1],
            "expected best-mass to grow with ε: lo={}, hi={}",
            p_lo[1],
            p_hi[1]
        );
    }

    #[test]
    fn equal_scores_roughly_uniform() {
        let scores = vec![2.0, 2.0, 2.0, 2.0];
        let c = cfg();
        let mut rng = LcgRng::new(31);
        let probs = permute_flip_empirical_probs(&scores, &c, 8000, &mut rng).expect("ok");
        for &p in &probs {
            assert!((p - 0.25).abs() < 0.05, "p = {p}, expected ≈ 0.25");
        }
    }

    #[test]
    fn never_selects_dominated_when_gap_large() {
        // With a huge score gap and high ε the worst candidate is essentially
        // never chosen.
        let scores = vec![-100.0, 0.0];
        let c = PermuteFlipConfig::new(5.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(77);
        let probs = permute_flip_empirical_probs(&scores, &c, 2000, &mut rng).expect("ok");
        assert!(probs[0] < 0.01, "dominated prob = {}", probs[0]);
        assert!(probs[1] > 0.99);
    }
}
