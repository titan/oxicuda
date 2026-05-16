//! Weighted reservoir sampling (Efraimidis, Spirakis 2006).
//!
//! For each item compute key = u^(1/w) where u ~ Uniform\[0,1\]. Keep the `k` items with
//! the largest keys.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Weighted reservoir sampler.
#[derive(Debug, Clone)]
pub struct WeightedReservoirSampler {
    pub k: usize,
    /// Heap of (-key, value) — but we use a sorted Vec for simplicity. Smallest key is replaced.
    pub reservoir: Vec<(f64, u64)>,
    rng: LcgRng,
}

impl WeightedReservoirSampler {
    /// New sampler of capacity `k`.
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            k,
            reservoir: Vec::with_capacity(k),
            rng: LcgRng::new(seed),
        })
    }

    /// Add an item with weight `w > 0`. Items with non-positive weight are ignored.
    pub fn add(&mut self, x: u64, w: f64) {
        if w <= 0.0 || !w.is_finite() {
            return;
        }
        let u = self.rng.next_f64().max(1.0e-300);
        let key = u.ln() / w;
        if self.reservoir.len() < self.k {
            self.reservoir.push((key, x));
            return;
        }
        // Find minimum key.
        let mut min_idx = 0usize;
        let mut min_val = self.reservoir[0].0;
        for (i, &(k, _)) in self.reservoir.iter().enumerate().skip(1) {
            if k < min_val {
                min_val = k;
                min_idx = i;
            }
        }
        if key > min_val {
            self.reservoir[min_idx] = (key, x);
        }
    }

    /// Get the current sample (just the values).
    #[must_use]
    pub fn sample(&self) -> Vec<u64> {
        self.reservoir.iter().map(|(_, v)| *v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrs_constructs() {
        let s = WeightedReservoirSampler::new(8, 0).expect("ok");
        assert_eq!(s.k, 8);
    }

    #[test]
    fn wrs_collects_first_k() {
        let mut s = WeightedReservoirSampler::new(3, 0).expect("ok");
        s.add(1, 1.0);
        s.add(2, 1.0);
        s.add(3, 1.0);
        assert_eq!(s.sample().len(), 3);
    }

    #[test]
    fn wrs_high_weight_bias() {
        // Item with very high weight should be in sample most of the time.
        let mut count = 0;
        let trials = 200;
        for t in 0..trials {
            let mut s = WeightedReservoirSampler::new(2, t as u64).expect("ok");
            s.add(99, 1000.0);
            for i in 0..20u64 {
                s.add(i, 1.0);
            }
            if s.sample().contains(&99) {
                count += 1;
            }
        }
        assert!(
            count > trials * 8 / 10,
            "high-weight item rarely sampled: {count}/{trials}"
        );
    }

    #[test]
    fn wrs_invalid_k() {
        assert!(WeightedReservoirSampler::new(0, 0).is_err());
    }
}
