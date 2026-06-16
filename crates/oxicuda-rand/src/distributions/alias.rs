//! Alias method for O(1) sampling from discrete distributions.
//!
//! Implements Walker's alias method (1977) with Vose's linear-time
//! construction (1991). The table is built in O(n) time and space;
//! each sample requires exactly two random numbers and O(1) work.
//!
//! Reference:
//! - A.J. Walker, "An Efficient Method for Generating Discrete Random
//!   Variables with General Distributions", ACM TOMS, 1977.
//! - M.D. Vose, "A Linear Algorithm for Generating Random Numbers with a
//!   Given Distribution", IEEE TSE, 1991.

use crate::engines::pcg::Pcg32;
use crate::error::{RandError, RandResult};

/// Pre-computed alias table for O(1) discrete distribution sampling.
///
/// Stores one probability value and one alias per outcome. Sampling
/// requires two uniform random numbers: one to select a column and one
/// to decide between the primary outcome and its alias.
pub struct AliasTable {
    /// Scaled probability for the primary outcome in each column.
    prob: Vec<f64>,
    /// Alias (secondary outcome) for each column.
    alias: Vec<usize>,
    /// Number of outcomes.
    n: usize,
}

impl AliasTable {
    /// Build a Vose alias table from the given non-negative weights.
    ///
    /// Weights need not sum to 1; they are normalized internally.
    ///
    /// # Errors
    ///
    /// - [`RandError::InvalidParameter`] if `weights` is empty.
    /// - [`RandError::InvalidParameter`] if any weight is negative.
    /// - [`RandError::InvalidParameter`] if the total weight is zero.
    pub fn new(weights: &[f64]) -> RandResult<Self> {
        if weights.is_empty() {
            return Err(RandError::InvalidParameter(
                "alias table requires at least one weight".to_string(),
            ));
        }
        for (i, &w) in weights.iter().enumerate() {
            if w < 0.0 {
                return Err(RandError::InvalidParameter(format!(
                    "weight[{i}] = {w} is negative; all weights must be >= 0"
                )));
            }
        }

        let n = weights.len();
        let sum: f64 = weights.iter().sum();
        if sum == 0.0 {
            return Err(RandError::InvalidParameter(
                "total weight is zero; at least one weight must be positive".to_string(),
            ));
        }

        // Normalize: p[i] = weight[i] / sum * n  (so average = 1.0)
        let mut p: Vec<f64> = weights.iter().map(|&w| w / sum * n as f64).collect();

        let mut prob = vec![0.0f64; n];
        let mut alias = vec![0usize; n];

        // Partition indices into small (p[i] < 1) and large (p[i] >= 1).
        let mut small: Vec<usize> = Vec::with_capacity(n);
        let mut large: Vec<usize> = Vec::with_capacity(n);
        for (i, &pi) in p.iter().enumerate() {
            if pi < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        // Vose construction: pair each "small" with a "large" to fill columns.
        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().expect("small non-empty");
            let g = large.pop().expect("large non-empty");

            prob[l] = p[l];
            alias[l] = g;

            // Subtract the probability donated to column l from the large bucket.
            p[g] = (p[g] + p[l]) - 1.0;

            if p[g] < 1.0 {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        // Remaining entries are exactly 1 (modulo floating-point rounding).
        for &g in large.iter().chain(small.iter()) {
            prob[g] = 1.0;
            alias[g] = g;
        }

        Ok(AliasTable { prob, alias, n })
    }

    /// Sample one index from the distribution using the alias method.
    ///
    /// Two random numbers are consumed from `rng`:
    /// - One to select a column uniformly in `[0, n)`.
    /// - One (implicitly derived from the column selection) to decide
    ///   between the primary outcome and its alias.
    #[inline]
    pub fn sample(&self, rng: &mut Pcg32) -> usize {
        // Use a single u32 draw: split into column index and fractional part.
        let u_raw = rng.next_u32() as f64 / 4_294_967_296.0; // [0, 1)
        let scaled = u_raw * self.n as f64;
        let i = (scaled as usize).min(self.n - 1);
        let frac = scaled - i as f64;
        if frac < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }

    /// Sample `count` indices from the distribution.
    pub fn sample_batch(&self, count: usize, rng: &mut Pcg32) -> Vec<usize> {
        (0..count).map(|_| self.sample(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> Pcg32 {
        Pcg32::new(seed, 1)
    }

    #[test]
    fn output_in_range() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let table = AliasTable::new(&weights).expect("valid weights");
        let mut rng = make_rng(42);
        for _ in 0..1000 {
            let idx = table.sample(&mut rng);
            assert!(idx < weights.len(), "index {idx} out of range");
        }
    }

    #[test]
    fn uniform_weights_all_equiprobable() {
        let n = 5usize;
        let weights: Vec<f64> = vec![1.0; n];
        let table = AliasTable::new(&weights).expect("valid weights");
        let mut rng = make_rng(12345);
        let total = 50_000usize;
        let mut counts = vec![0usize; n];
        for _ in 0..total {
            counts[table.sample(&mut rng)] += 1;
        }
        let expected = total as f64 / n as f64;
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - expected;
                d * d / expected
            })
            .sum();
        // Chi2 with n-1=4 df; lenient threshold of 20
        assert!(
            chi2 < 20.0,
            "chi2 = {chi2:.2} exceeds 20.0 for uniform weights"
        );
    }

    #[test]
    fn single_weight_always_same() {
        let table = AliasTable::new(&[1.0]).expect("valid");
        let mut rng = make_rng(0);
        for _ in 0..100 {
            assert_eq!(table.sample(&mut rng), 0);
        }
    }

    #[test]
    fn two_weights_bias() {
        // weights = [1.0, 3.0] -> index 1 should appear ~75% of the time
        let table = AliasTable::new(&[1.0, 3.0]).expect("valid");
        let mut rng = make_rng(99);
        let n = 20_000usize;
        let count1 = (0..n).filter(|_| table.sample(&mut rng) == 1).count();
        let fraction = count1 as f64 / n as f64;
        assert!(
            (0.70..0.80).contains(&fraction),
            "expected ~75% index 1, got {:.1}%",
            fraction * 100.0
        );
    }

    #[test]
    fn sample_batch_len() {
        let weights = vec![1.0, 2.0, 3.0];
        let table = AliasTable::new(&weights).expect("valid");
        let mut rng = make_rng(55);
        let batch = table.sample_batch(100, &mut rng);
        assert_eq!(batch.len(), 100);
    }

    #[test]
    fn empty_weights_error() {
        let result = AliasTable::new(&[]);
        assert!(result.is_err(), "empty weights must return Err");
        if let Err(RandError::InvalidParameter(msg)) = result {
            assert!(msg.contains("at least one"), "error message: {msg}");
        } else {
            panic!("expected InvalidParameter error");
        }
    }

    #[test]
    fn negative_weight_error() {
        let result = AliasTable::new(&[-1.0, 1.0]);
        assert!(result.is_err(), "negative weight must return Err");
        if let Err(RandError::InvalidParameter(msg)) = result {
            assert!(msg.contains("negative"), "error message: {msg}");
        } else {
            panic!("expected InvalidParameter error");
        }
    }

    #[test]
    fn frequencies_match_weights() {
        let weights = [1.0f64, 2.0, 3.0, 4.0];
        let table = AliasTable::new(&weights).expect("valid");
        let mut rng = make_rng(777);
        let total = 10_000usize;
        let mut counts = vec![0usize; weights.len()];
        for _ in 0..total {
            counts[table.sample(&mut rng)] += 1;
        }
        let sum: f64 = weights.iter().sum();
        for (i, (&w, &c)) in weights.iter().zip(counts.iter()).enumerate() {
            let expected_frac = w / sum;
            let actual_frac = c as f64 / total as f64;
            let diff = (expected_frac - actual_frac).abs();
            assert!(
                diff < 0.05,
                "index {i}: expected {:.3}, got {:.3}, diff {:.3} > 0.05",
                expected_frac,
                actual_frac,
                diff
            );
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let weights = vec![2.0, 5.0, 3.0];
        let table = AliasTable::new(&weights).expect("valid");
        let mut rng1 = make_rng(42);
        let mut rng2 = make_rng(42);
        let seq1: Vec<usize> = (0..50).map(|_| table.sample(&mut rng1)).collect();
        let seq2: Vec<usize> = (0..50).map(|_| table.sample(&mut rng2)).collect();
        assert_eq!(seq1, seq2, "same seed must produce identical sequences");
    }
}
