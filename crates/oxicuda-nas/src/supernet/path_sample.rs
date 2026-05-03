//! Path sampling strategies for one-shot supernets.
//!
//! Two strategies:
//! * `Uniform` — sample one op per edge uniformly at random.
//! * `FairnessAware` — equalise per-op sample counts across the search.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── PathSampler ─────────────────────────────────────────────────────────────

/// Sampling strategy for the one-shot supernet path sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniformly random op per edge.
    Uniform,
    /// Fairness-aware: prefer under-sampled ops to equalise counts.
    FairnessAware,
}

/// Samples one path (op index per edge) from a supernet.
#[derive(Debug, Clone)]
pub struct PathSampler {
    /// Number of edges.
    pub n_edges: usize,
    /// Number of candidate ops per edge.
    pub n_ops: usize,
    /// Sampling strategy.
    pub strategy: SamplingStrategy,
    /// Per-edge, per-op sample counts (for fairness-aware strategy).
    pub counts: Vec<Vec<u64>>,
}

impl PathSampler {
    /// Construct a new sampler with zero counts.
    #[must_use]
    pub fn new(n_edges: usize, n_ops: usize, strategy: SamplingStrategy) -> Self {
        Self {
            n_edges,
            n_ops,
            strategy,
            counts: vec![vec![0u64; n_ops]; n_edges],
        }
    }

    /// Sample one path: returns `Vec<usize>` of length `n_edges`,
    /// each element is the selected op index for that edge.
    pub fn sample(&mut self, rng: &mut LcgRng) -> NasResult<Vec<usize>> {
        if self.n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        let mut path = Vec::with_capacity(self.n_edges);
        for e in 0..self.n_edges {
            let op_idx = match self.strategy {
                SamplingStrategy::Uniform => rng.next_usize(self.n_ops),
                SamplingStrategy::FairnessAware => {
                    // Pick the op with the minimum count; break ties randomly
                    let min_count = *self.counts[e].iter().min().unwrap_or(&0);
                    let candidates: Vec<usize> = (0..self.n_ops)
                        .filter(|&k| self.counts[e][k] == min_count)
                        .collect();
                    candidates[rng.next_usize(candidates.len())]
                }
            };
            self.counts[e][op_idx] += 1;
            path.push(op_idx);
        }
        Ok(path)
    }

    /// Reset all sample counts to zero.
    pub fn reset_counts(&mut self) {
        for row in &mut self.counts {
            for c in row.iter_mut() {
                *c = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sample_length() {
        let mut rng = LcgRng::new(42);
        let mut ps = PathSampler::new(14, 8, SamplingStrategy::Uniform);
        let path = ps.sample(&mut rng).expect("test invariant: sample path");
        assert_eq!(path.len(), 14);
        assert!(path.iter().all(|&i| i < 8));
    }

    #[test]
    fn fairness_aware_equalises_counts() {
        let mut rng = LcgRng::new(7);
        let mut ps = PathSampler::new(1, 4, SamplingStrategy::FairnessAware);
        // After 100 samples each op should have been picked ~25 times
        for _ in 0..100 {
            ps.sample(&mut rng)
                .expect("test invariant: fairness sample");
        }
        let counts = &ps.counts[0];
        let min_c = *counts.iter().min().unwrap_or(&0);
        let max_c = *counts.iter().max().unwrap_or(&0);
        // All counts within ±1 of each other (perfect round-robin is not guaranteed,
        // but fairness-aware should keep them very close)
        assert!(max_c - min_c <= 1, "counts not balanced: {counts:?}");
    }
}
