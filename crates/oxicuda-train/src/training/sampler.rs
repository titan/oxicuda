//! Data samplers — index generation strategies for training data loaders.
//!
//! A *sampler* produces the sequence of dataset indices a training loop visits
//! in one epoch.  The choice of sampler controls shuffling, class re-balancing,
//! and subset restriction without touching the dataset itself.  This module
//! mirrors the sampler family of typical deep-learning data loaders, built on
//! the crate's deterministic [`crate::handle::LcgRng`] so every epoch is
//! reproducible for a fixed seed.
//!
//! Provided samplers:
//!
//! * [`SequentialSampler`] — indices `0, 1, …, n−1` in order.
//! * [`RandomSampler`] — a uniformly random permutation each epoch (Fisher–Yates),
//!   optionally *with replacement* for a fixed number of draws.
//! * [`WeightedRandomSampler`] — sample (with replacement) proportional to
//!   per-example weights; the standard tool for class-imbalanced training.
//! * [`SubsetRandomSampler`] — a random permutation restricted to a fixed
//!   index subset (e.g. a cross-validation fold).
//! * [`BatchSampler`] — groups any index sampler's output into mini-batches,
//!   with an option to drop a ragged final batch.

use crate::error::{TrainError, TrainResult};
use crate::handle::LcgRng;

// ─── SequentialSampler ────────────────────────────────────────────────────────

/// Yields indices `0..n` in order.
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    len: usize,
}

impl SequentialSampler {
    /// Create a sequential sampler over `len` examples.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `len == 0`.
    pub fn new(len: usize) -> TrainResult<Self> {
        if len == 0 {
            return Err(TrainError::EmptyParams);
        }
        Ok(Self { len })
    }

    /// Number of indices yielded per epoch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false` (a `SequentialSampler` is never empty).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Produce the ordered index sequence for one epoch.
    #[must_use]
    pub fn indices(&self) -> Vec<usize> {
        (0..self.len).collect()
    }
}

// ─── RandomSampler ────────────────────────────────────────────────────────────

/// Yields a random index ordering each epoch.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    len: usize,
    /// When `Some(k)`, sample `k` indices *with replacement*; when `None`,
    /// produce a full permutation (sampling without replacement).
    num_samples: Option<usize>,
    rng: LcgRng,
}

impl RandomSampler {
    /// Create a without-replacement random sampler (full permutation).
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `len == 0`.
    pub fn new(len: usize, seed: u64) -> TrainResult<Self> {
        if len == 0 {
            return Err(TrainError::EmptyParams);
        }
        Ok(Self {
            len,
            num_samples: None,
            rng: LcgRng::new(seed),
        })
    }

    /// Create a with-replacement random sampler drawing `num_samples` indices.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `len == 0` or `num_samples == 0`.
    pub fn with_replacement(len: usize, num_samples: usize, seed: u64) -> TrainResult<Self> {
        if len == 0 || num_samples == 0 {
            return Err(TrainError::EmptyParams);
        }
        Ok(Self {
            len,
            num_samples: Some(num_samples),
            rng: LcgRng::new(seed),
        })
    }

    /// Number of indices produced per epoch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_samples.unwrap_or(self.len)
    }

    /// Always `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Draw the next epoch's index ordering, advancing the internal RNG so
    /// successive epochs differ.
    pub fn indices(&mut self) -> Vec<usize> {
        match self.num_samples {
            Some(k) => (0..k)
                .map(|_| (self.rng.next_u64() % self.len as u64) as usize)
                .collect(),
            None => {
                let mut idx: Vec<usize> = (0..self.len).collect();
                // Fisher–Yates shuffle.
                for i in (1..self.len).rev() {
                    let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
                    idx.swap(i, j);
                }
                idx
            }
        }
    }
}

// ─── WeightedRandomSampler ────────────────────────────────────────────────────

/// Samples indices with replacement proportional to non-negative weights.
///
/// Uses an inverse-CDF (alias-free) lookup over the cumulative weight table;
/// `O(log n)` per draw via binary search.
#[derive(Debug, Clone)]
pub struct WeightedRandomSampler {
    /// Cumulative-sum table of the (normalised) weights, length `n`.
    cumulative: Vec<f64>,
    total: f64,
    num_samples: usize,
    rng: LcgRng,
}

impl WeightedRandomSampler {
    /// Build a weighted sampler over `weights` drawing `num_samples` indices.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `weights` is empty or `num_samples == 0`.
    /// * [`TrainError::Internal`] if any weight is negative / NaN, or the total
    ///   weight is zero.
    pub fn new(weights: &[f64], num_samples: usize, seed: u64) -> TrainResult<Self> {
        if weights.is_empty() || num_samples == 0 {
            return Err(TrainError::EmptyParams);
        }
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in weights {
            if w < 0.0 || w.is_nan() {
                return Err(TrainError::Internal {
                    msg: format!("weights must be non-negative and finite, got {w}"),
                });
            }
            acc += w;
            cumulative.push(acc);
        }
        if acc <= 0.0 {
            return Err(TrainError::Internal {
                msg: "total weight must be positive".into(),
            });
        }
        Ok(Self {
            cumulative,
            total: acc,
            num_samples,
            rng: LcgRng::new(seed),
        })
    }

    /// Number of indices produced per epoch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_samples
    }

    /// Always `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Binary-search the cumulative table for the first index whose prefix sum
    /// exceeds `target`.
    fn search(&self, target: f64) -> usize {
        let (mut lo, mut hi) = (0_usize, self.cumulative.len() - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cumulative[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Draw the next epoch's weighted index sequence.
    pub fn indices(&mut self) -> Vec<usize> {
        (0..self.num_samples)
            .map(|_| {
                let target = self.rng.next_f64() * self.total;
                self.search(target)
            })
            .collect()
    }
}

// ─── SubsetRandomSampler ──────────────────────────────────────────────────────

/// Yields a random permutation of a fixed index subset each epoch.
#[derive(Debug, Clone)]
pub struct SubsetRandomSampler {
    subset: Vec<usize>,
    rng: LcgRng,
}

impl SubsetRandomSampler {
    /// Create a subset sampler over the given indices.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] if `subset` is empty.
    pub fn new(subset: Vec<usize>, seed: u64) -> TrainResult<Self> {
        if subset.is_empty() {
            return Err(TrainError::EmptyParams);
        }
        Ok(Self {
            subset,
            rng: LcgRng::new(seed),
        })
    }

    /// Number of indices yielded per epoch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.subset.len()
    }

    /// Always `false`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Draw the next epoch's shuffled subset.
    pub fn indices(&mut self) -> Vec<usize> {
        let mut out = self.subset.clone();
        for i in (1..out.len()).rev() {
            let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
            out.swap(i, j);
        }
        out
    }
}

// ─── BatchSampler ─────────────────────────────────────────────────────────────

/// Groups a flat index sequence into mini-batches.
#[derive(Debug, Clone)]
pub struct BatchSampler {
    batch_size: usize,
    drop_last: bool,
}

impl BatchSampler {
    /// Create a batch grouper.
    ///
    /// * `batch_size` – number of indices per batch (≥ 1).
    /// * `drop_last` – discard a final batch smaller than `batch_size`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::Internal`] if `batch_size == 0`.
    pub fn new(batch_size: usize, drop_last: bool) -> TrainResult<Self> {
        if batch_size == 0 {
            return Err(TrainError::Internal {
                msg: "batch_size must be >= 1".into(),
            });
        }
        Ok(Self {
            batch_size,
            drop_last,
        })
    }

    /// Configured batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Group `indices` into batches.
    #[must_use]
    pub fn batches(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = indices
            .chunks(self.batch_size)
            .map(<[usize]>::to_vec)
            .collect();
        if self.drop_last {
            if let Some(last) = out.last() {
                if last.len() < self.batch_size {
                    out.pop();
                }
            }
        }
        out
    }

    /// Number of batches produced for `n` indices.
    #[must_use]
    pub fn num_batches(&self, n: usize) -> usize {
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_in_order() {
        let s = SequentialSampler::new(5).expect("valid");
        assert_eq!(s.indices(), vec![0, 1, 2, 3, 4]);
        assert_eq!(s.len(), 5);
        assert!(!s.is_empty());
    }

    #[test]
    fn empty_rejected() {
        assert!(matches!(
            SequentialSampler::new(0),
            Err(TrainError::EmptyParams)
        ));
        assert!(matches!(
            RandomSampler::new(0, 1),
            Err(TrainError::EmptyParams)
        ));
    }

    /// A without-replacement random sampler yields a true permutation: every
    /// index appears exactly once.
    #[test]
    fn random_is_permutation() {
        let n = 50;
        let mut s = RandomSampler::new(n, 12345).expect("valid");
        let idx = s.indices();
        assert_eq!(idx.len(), n);
        let mut seen = vec![false; n];
        for &i in &idx {
            assert!(i < n);
            assert!(!seen[i], "index {i} duplicated");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&b| b), "all indices must appear");
    }

    /// Successive epochs of a random sampler differ (the RNG advances).
    #[test]
    fn random_epochs_differ() {
        let mut s = RandomSampler::new(100, 7).expect("valid");
        let a = s.indices();
        let b = s.indices();
        assert_ne!(a, b, "two epochs should differ");
    }

    /// Deterministic for a fixed seed: two samplers built with the same seed
    /// emit identical sequences.
    #[test]
    fn random_deterministic_seed() {
        let mut a = RandomSampler::new(30, 999).expect("valid");
        let mut b = RandomSampler::new(30, 999).expect("valid");
        assert_eq!(a.indices(), b.indices());
    }

    #[test]
    fn random_with_replacement_count() {
        let mut s = RandomSampler::with_replacement(10, 25, 3).expect("valid");
        let idx = s.indices();
        assert_eq!(idx.len(), 25);
        assert!(idx.iter().all(|&i| i < 10));
    }

    /// Weighted sampling honours the weights: a heavily-weighted class dominates
    /// the empirical frequency.
    #[test]
    fn weighted_respects_weights() {
        // Index 2 has 10× the weight of the others.
        let weights = [1.0, 1.0, 10.0, 1.0];
        let n_draws = 20_000;
        let mut s = WeightedRandomSampler::new(&weights, n_draws, 2024).expect("valid");
        let idx = s.indices();
        let mut counts = [0_usize; 4];
        for &i in &idx {
            counts[i] += 1;
        }
        // Expected fraction for index 2 = 10/13 ≈ 0.769.
        let frac2 = counts[2] as f64 / n_draws as f64;
        assert!(
            (frac2 - 10.0 / 13.0).abs() < 0.03,
            "weighted fraction off: {frac2}"
        );
        // The zero-weight-relative classes should each be ≈ 1/13 ≈ 0.077.
        let frac0 = counts[0] as f64 / n_draws as f64;
        assert!((frac0 - 1.0 / 13.0).abs() < 0.03, "frac0 {frac0}");
    }

    /// A zero weight is never sampled.
    #[test]
    fn weighted_skips_zero_weight() {
        let weights = [0.0, 5.0, 0.0, 5.0];
        let mut s = WeightedRandomSampler::new(&weights, 5000, 11).expect("valid");
        let idx = s.indices();
        assert!(
            idx.iter().all(|&i| i == 1 || i == 3),
            "zero-weight classes must never be drawn"
        );
    }

    #[test]
    fn weighted_rejects_bad() {
        assert!(matches!(
            WeightedRandomSampler::new(&[1.0, -1.0], 5, 1),
            Err(TrainError::Internal { .. })
        ));
        assert!(matches!(
            WeightedRandomSampler::new(&[0.0, 0.0], 5, 1),
            Err(TrainError::Internal { .. })
        ));
    }

    /// Subset sampler only ever yields indices from the subset, as a permutation.
    #[test]
    fn subset_permutes_subset() {
        let subset = vec![3, 7, 11, 15];
        let mut s = SubsetRandomSampler::new(subset.clone(), 42).expect("valid");
        let idx = s.indices();
        assert_eq!(idx.len(), subset.len());
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        let mut subset_sorted = subset.clone();
        subset_sorted.sort_unstable();
        assert_eq!(sorted, subset_sorted, "must be a permutation of the subset");
    }

    #[test]
    fn batch_groups_with_remainder() {
        let bs = BatchSampler::new(3, false).expect("valid");
        let batches = bs.batches(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[2], vec![6, 7]); // ragged tail kept
        assert_eq!(bs.num_batches(8), 3);
    }

    #[test]
    fn batch_drops_last() {
        let bs = BatchSampler::new(3, true).expect("valid");
        let batches = bs.batches(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(batches.len(), 2, "ragged tail dropped");
        assert_eq!(bs.num_batches(8), 2);
    }

    #[test]
    fn batch_rejects_zero() {
        assert!(matches!(
            BatchSampler::new(0, false),
            Err(TrainError::Internal { .. })
        ));
    }

    /// End-to-end: a random sampler feeding a drop-last batch sampler yields
    /// full batches covering distinct indices.
    #[test]
    fn random_into_batches() {
        let mut rs = RandomSampler::new(20, 5).expect("valid");
        let bs = BatchSampler::new(4, true).expect("valid");
        let idx = rs.indices();
        let batches = bs.batches(&idx);
        assert_eq!(batches.len(), 5);
        for b in &batches {
            assert_eq!(b.len(), 4);
        }
        // All indices across batches are distinct (random is a permutation).
        let mut flat: Vec<usize> = batches.into_iter().flatten().collect();
        flat.sort_unstable();
        flat.dedup();
        assert_eq!(flat.len(), 20);
    }
}
