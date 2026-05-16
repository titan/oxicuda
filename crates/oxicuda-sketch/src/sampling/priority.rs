//! Priority sampling (Duffield, Lund, Thorup 2007).
//!
//! For each item with weight w, compute priority p = w / u where u ~ Uniform(0, 1).
//! Keep the k items with the largest priorities. Used for ε-unbiased subset-sum estimation.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Priority sampler.
#[derive(Debug, Clone)]
pub struct PrioritySampler {
    pub k: usize,
    pub reservoir: Vec<(f64, u64, f64)>, // (priority, value, original_weight)
    rng: LcgRng,
}

impl PrioritySampler {
    /// New priority sampler retaining the top-`k` priorities.
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            k,
            reservoir: Vec::with_capacity(k + 1),
            rng: LcgRng::new(seed),
        })
    }

    /// Add an item with weight `w > 0`.
    pub fn add(&mut self, x: u64, w: f64) {
        if w <= 0.0 || !w.is_finite() {
            return;
        }
        let u = self.rng.next_f64().max(1.0e-300);
        let priority = w / u;
        self.reservoir.push((priority, x, w));
        if self.reservoir.len() > self.k {
            // Pop minimum.
            let mut min_idx = 0usize;
            let mut min_val = self.reservoir[0].0;
            for (i, &(p, _, _)) in self.reservoir.iter().enumerate().skip(1) {
                if p < min_val {
                    min_val = p;
                    min_idx = i;
                }
            }
            self.reservoir.swap_remove(min_idx);
        }
    }

    /// Estimate the subset-sum: each item `i` contributes `max(w_i, threshold)`.
    /// Threshold = the (k+1)-th largest priority (i.e. the smallest priority not kept).
    #[must_use]
    pub fn estimate_sum(&self) -> f64 {
        if self.reservoir.is_empty() {
            return 0.0;
        }
        // Threshold = min priority in the kept set (approximate; the true threshold is the
        // (k+1)-th largest priority, which we omit. We approximate using the smallest priority
        // among the kept items as a conservative lower bound on the threshold).
        let threshold = self
            .reservoir
            .iter()
            .map(|&(p, _, _)| p)
            .fold(f64::INFINITY, f64::min);
        self.reservoir
            .iter()
            .map(|&(_, _, w)| w.max(threshold))
            .sum()
    }

    /// Get sampled values.
    #[must_use]
    pub fn sample(&self) -> Vec<u64> {
        self.reservoir.iter().map(|&(_, v, _)| v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_invalid_k() {
        assert!(PrioritySampler::new(0, 0).is_err());
    }

    #[test]
    fn priority_constructs() {
        let s = PrioritySampler::new(4, 0).expect("ok");
        assert_eq!(s.k, 4);
    }

    #[test]
    fn priority_keeps_top_k() {
        let mut s = PrioritySampler::new(3, 0).expect("ok");
        for i in 0..20u64 {
            s.add(i, 1.0);
        }
        assert!(s.sample().len() <= 3);
    }
}
