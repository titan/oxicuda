//! Exponential mechanism (McSherry & Talwar, 2007) with Walker–Vose alias
//! sampling.
//!
//! The exponential mechanism over a finite set of `k` outcomes selects outcome
//! `i` with probability proportional to `exp(ε · u(i) / (2 · Δ))`, where `u` is
//! the utility (quality) function and `Δ` its global sensitivity.
//!
//! For a fixed output set sampled *many* times, building Walker's alias table
//! once amortises sampling to `O(1)` per draw, versus the `O(k)` cumulative-sum
//! linear scan used by [`crate::mechanism::exponential`].  Table construction
//! uses Vose's numerically stable variant of the alias method.
//!
//! References:
//! - McSherry & Talwar (2007), "Mechanism Design via Differential Privacy",
//!   FOCS.
//! - Walker (1977), "An Efficient Method for Generating Discrete Random
//!   Variables with General Distributions", ACM TOMS 3(3).
//! - Vose (1991), "A Linear Algorithm for Generating Random Numbers with a
//!   Given Distribution", IEEE Trans. Software Eng. 17(9).

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Exponential mechanism backed by a Walker–Vose alias table for `O(1)`
/// sampling.
///
/// Construct once with [`ExponentialAlias::new`]; draw repeatedly with
/// [`ExponentialAlias::sample`].
#[derive(Debug, Clone)]
pub struct ExponentialAlias {
    /// Number of outcomes `k ≥ 1`.
    n: usize,
    /// Per-column acceptance probability, each in `[0, 1]`.
    prob: Vec<f64>,
    /// Per-column alias index (the fallback outcome on rejection).
    alias: Vec<usize>,
}

impl ExponentialAlias {
    /// Build the alias table for the exponential mechanism over `utilities`.
    ///
    /// Outcome `i` is assigned probability proportional to
    /// `exp(ε · utilities[i] / (2 · sensitivity))`.  The exponent is shifted by
    /// `max(utilities)` before exponentiating for numerical stability, so the
    /// largest weight is `1` and the sum is in `[1, k]`.
    ///
    /// # Errors
    /// - `EmptyInput` if `utilities` is empty.
    /// - `NonPositiveEpsilon` if `epsilon` is non-finite or `≤ 0`.
    /// - `NonPositiveSensitivity` if `sensitivity` is non-finite or `≤ 0`.
    /// - `InvalidParameter` if any utility is non-finite or the weights do not
    ///   sum to a positive finite value.
    pub fn new(utilities: &[f64], epsilon: f64, sensitivity: f64) -> PrivacyResult<Self> {
        if utilities.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity <= 0.0 || !sensitivity.is_finite() {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        for &u in utilities {
            if !u.is_finite() {
                return Err(PrivacyError::InvalidParameter(format!(
                    "utility must be finite, got {u}"
                )));
            }
        }

        let n = utilities.len();
        let scale = epsilon / (2.0 * sensitivity);
        let shift = utilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = utilities
            .iter()
            .map(|&u| ((u - shift) * scale).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(PrivacyError::InvalidParameter(
                "utility weights did not sum to a positive finite value".into(),
            ));
        }
        let probs: Vec<f64> = weights.iter().map(|&w| w / total).collect();
        let (prob, alias) = Self::build_alias(&probs);
        Ok(Self { n, prob, alias })
    }

    /// Build a Walker–Vose alias table from a normalised probability vector.
    ///
    /// Returns `(prob, alias)` where, for column `i` selected uniformly, the
    /// table emits `i` with probability `prob[i]` and `alias[i]` otherwise.
    fn build_alias(probs: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let n = probs.len();
        let mut prob = vec![0.0_f64; n];
        // Self-alias by default so any residual rejection mass is a no-op.
        let mut alias: Vec<usize> = (0..n).collect();
        let mut scaled: Vec<f64> = probs.iter().map(|&p| p * n as f64).collect();

        let mut small: Vec<usize> = Vec::new();
        let mut large: Vec<usize> = Vec::new();
        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            // Both worklists are non-empty here, so `pop` yields `Some`; the
            // `else` branch is unreachable but keeps library code unwrap-free.
            let (Some(less), Some(more)) = (small.pop(), large.pop()) else {
                break;
            };
            prob[less] = scaled[less];
            alias[less] = more;
            // Remaining probability mass of `more` after topping up `less`.
            scaled[more] = (scaled[more] + scaled[less]) - 1.0;
            if scaled[more] < 1.0 {
                small.push(more);
            } else {
                large.push(more);
            }
        }
        // Any leftovers (from floating-point drift) carry full mass.
        while let Some(i) = large.pop() {
            prob[i] = 1.0;
        }
        while let Some(i) = small.pop() {
            prob[i] = 1.0;
        }
        (prob, alias)
    }

    /// Number of outcomes `k`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the outcome set is empty (always `false` for a constructed
    /// table; provided for API completeness alongside [`len`]).
    ///
    /// [`len`]: Self::len
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Draw one outcome index via the alias table in `O(1)`.
    ///
    /// Consumes two uniform draws from `rng`: one selects the column, the other
    /// is the acceptance coin.
    ///
    /// # Errors
    /// - `EmptyInput` if the table holds no outcomes (cannot occur for a table
    ///   built by [`ExponentialAlias::new`], which rejects empty input).
    pub fn sample(&self, rng: &mut LcgRng) -> PrivacyResult<usize> {
        if self.n == 0 {
            return Err(PrivacyError::EmptyInput);
        }
        let column = ((rng.next_f64() * self.n as f64).floor() as usize).min(self.n - 1);
        let coin = rng.next_f64();
        if coin < self.prob[column] {
            Ok(column)
        } else {
            Ok(self.alias[column])
        }
    }

    /// Reconstruct the exact sampling distribution induced by the alias table.
    ///
    /// For column `i` (chosen with probability `1/n`) the table emits `i` with
    /// probability `prob[i]` and `alias[i]` otherwise, so
    ///
    /// ```text
    ///     P(j) = prob[j]/n + Σ_i (1 − prob[i])/n · 𝟙[alias[i] = j].
    /// ```
    ///
    /// This equals the target `softmax(ε·u/(2Δ))` distribution up to
    /// floating-point error and is intended primarily for testing/inspection.
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        let inv_n = 1.0 / self.n as f64;
        let mut p = vec![0.0_f64; self.n];
        for (i, (&pr, &al)) in self.prob.iter().zip(self.alias.iter()).enumerate() {
            p[i] += pr * inv_n;
            p[al] += (1.0 - pr) * inv_n;
        }
        p
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct (un-aliased) softmax of the exponential mechanism, for reference.
    fn reference_softmax(utilities: &[f64], epsilon: f64, sensitivity: f64) -> Vec<f64> {
        let scale = epsilon / (2.0 * sensitivity);
        let shift = utilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = utilities
            .iter()
            .map(|&u| ((u - shift) * scale).exp())
            .collect();
        let total: f64 = w.iter().sum();
        w.iter().map(|&x| x / total).collect()
    }

    // (a) alias-table probabilities match the direct softmax to ~1e-12.
    #[test]
    fn probabilities_match_softmax() {
        let utilities = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let epsilon = 1.3;
        let sensitivity = 0.7;
        let expected = reference_softmax(&utilities, epsilon, sensitivity);
        let ea = ExponentialAlias::new(&utilities, epsilon, sensitivity).expect("ok");
        let got = ea.probabilities();
        assert_eq!(got.len(), expected.len());
        let total: f64 = got.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "table probs must sum to 1, got {total}"
        );
        for (e, g) in expected.iter().zip(got.iter()) {
            assert!((e - g).abs() < 1e-12, "alias prob {g} vs softmax {e}");
        }
    }

    // (b) higher ε concentrates mass on the argmax-utility element.
    #[test]
    fn higher_epsilon_concentrates_on_argmax() {
        let utilities = [0.0, 1.0, 2.0, 5.0, 3.0];
        let argmax = 3; // utility 5.0
        let low = ExponentialAlias::new(&utilities, 0.1, 1.0).expect("ok");
        let high = ExponentialAlias::new(&utilities, 8.0, 1.0).expect("ok");
        let p_low = low.probabilities()[argmax];
        let p_high = high.probabilities()[argmax];
        assert!(
            p_high > p_low,
            "larger ε should concentrate on argmax: {p_high} > {p_low}"
        );
        assert!(
            p_high > 0.9,
            "ε=8 should put most mass on argmax, got {p_high}"
        );
    }

    // (c) ε → 0 yields a near-uniform distribution.
    #[test]
    fn small_epsilon_is_near_uniform() {
        let utilities = [0.0, 1.0, 2.0, 3.0, 4.0];
        let k = utilities.len();
        let ea = ExponentialAlias::new(&utilities, 1e-6, 1.0).expect("ok");
        let p = ea.probabilities();
        let uniform = 1.0 / k as f64;
        for &pi in &p {
            assert!(
                (pi - uniform).abs() < 1e-3,
                "ε→0 should be near-uniform: {pi} vs {uniform}"
            );
        }
    }

    // (d) empirical sample frequencies approximate the target distribution.
    #[test]
    fn empirical_frequencies_match_target() {
        let utilities = [0.0, 1.0, 2.0];
        let ea = ExponentialAlias::new(&utilities, 1.0, 1.0).expect("ok");
        let target = ea.probabilities();
        let mut rng = LcgRng::new(0x5EED_1234);
        let draws = 60_000_usize;
        let mut counts = [0_usize; 3];
        for _ in 0..draws {
            let idx = ea.sample(&mut rng).expect("sample");
            counts[idx] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let freq = c as f64 / draws as f64;
            assert!(
                (freq - target[i]).abs() < 0.02,
                "bin {i}: empirical {freq} vs target {} differs too much",
                target[i]
            );
        }
    }

    // (e) invalid inputs are rejected with the right error.
    #[test]
    fn invalid_inputs_rejected() {
        assert!(matches!(
            ExponentialAlias::new(&[], 1.0, 1.0),
            Err(PrivacyError::EmptyInput)
        ));
        assert!(matches!(
            ExponentialAlias::new(&[1.0, 2.0], 0.0, 1.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            ExponentialAlias::new(&[1.0, 2.0], -1.0, 1.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            ExponentialAlias::new(&[1.0, 2.0], 1.0, 0.0),
            Err(PrivacyError::NonPositiveSensitivity(_))
        ));
        assert!(matches!(
            ExponentialAlias::new(&[1.0, 2.0], 1.0, -2.0),
            Err(PrivacyError::NonPositiveSensitivity(_))
        ));
        assert!(ExponentialAlias::new(&[f64::NAN, 1.0], 1.0, 1.0).is_err());
    }

    // A single-outcome table always returns index 0.
    #[test]
    fn single_outcome_returns_zero() {
        let ea = ExponentialAlias::new(&[7.0], 2.0, 0.5).expect("ok");
        assert_eq!(ea.len(), 1);
        assert!(!ea.is_empty());
        let mut rng = LcgRng::new(99);
        for _ in 0..50 {
            assert_eq!(ea.sample(&mut rng).expect("sample"), 0);
        }
        let p = ea.probabilities();
        assert!((p[0] - 1.0).abs() < 1e-12);
    }

    // sample only ever returns valid indices, for a skewed distribution.
    #[test]
    fn sample_indices_in_range() {
        let utilities = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ea = ExponentialAlias::new(&utilities, 3.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(7);
        for _ in 0..500 {
            let idx = ea.sample(&mut rng).expect("sample");
            assert!(idx < utilities.len());
        }
    }
}
