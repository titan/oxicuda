//! Bernoulli sampling: include each item with fixed probability `p`.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Bernoulli sampler.
#[derive(Debug, Clone)]
pub struct BernoulliSampler {
    pub p: f64,
    pub items: Vec<u64>,
    rng: LcgRng,
}

impl BernoulliSampler {
    /// New sampler with inclusion probability `p ∈ (0, 1]`.
    pub fn new(p: f64, seed: u64) -> SketchResult<Self> {
        if !(0.0 < p && p <= 1.0) {
            return Err(SketchError::InvalidParameter {
                name: "p".to_string(),
                reason: "must be in (0, 1]".to_string(),
            });
        }
        Ok(Self {
            p,
            items: Vec::new(),
            rng: LcgRng::new(seed),
        })
    }

    /// Add an item; include with probability `p`.
    pub fn add(&mut self, x: u64) {
        if self.rng.next_f64() < self.p {
            self.items.push(x);
        }
    }

    /// Get sampled items.
    #[must_use]
    pub fn sample(&self) -> &[u64] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_invalid_p() {
        assert!(BernoulliSampler::new(0.0, 0).is_err());
        assert!(BernoulliSampler::new(2.0, 0).is_err());
    }

    #[test]
    fn bernoulli_rate_close_to_p() {
        let mut b = BernoulliSampler::new(0.25, 11).expect("ok");
        for i in 0..10_000u64 {
            b.add(i);
        }
        let frac = b.sample().len() as f64 / 10_000.0;
        assert!((frac - 0.25).abs() < 0.02, "frac = {frac}");
    }

    #[test]
    fn bernoulli_full_p_keeps_all() {
        let mut b = BernoulliSampler::new(1.0, 11).expect("ok");
        for i in 0..100u64 {
            b.add(i);
        }
        assert_eq!(b.sample().len(), 100);
    }
}
