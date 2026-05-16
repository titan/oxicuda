//! Reservoir sampling (Vitter 1985, Algorithm R).
//!
//! Maintains a fixed-size uniform random sample of size `k` from a stream of unknown length.
//! For element `i` (1-indexed), with prob `k/i` replace a random reservoir slot.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Reservoir sampler with capacity `k`.
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    pub k: usize,
    pub reservoir: Vec<u64>,
    pub n_seen: u64,
    rng: LcgRng,
}

impl ReservoirSampler {
    /// Create a new reservoir of capacity `k`.
    pub fn new(k: usize, seed: u64) -> SketchResult<Self> {
        if k == 0 {
            return Err(SketchError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut rng = LcgRng::new(seed);
        // Warm up the RNG to escape correlated startup transients of the MMIX LCG.
        for _ in 0..8 {
            let _ = rng.next_u64();
        }
        Ok(Self {
            k,
            reservoir: Vec::with_capacity(k),
            n_seen: 0,
            rng,
        })
    }

    /// Process the next item from the stream.
    pub fn add(&mut self, x: u64) {
        self.n_seen += 1;
        if self.reservoir.len() < self.k {
            self.reservoir.push(x);
            return;
        }
        // With probability k / n_seen, replace a random slot.
        // Use the high 32 bits to avoid LCG low-bit periodicity defects.
        let raw = self.rng.next_u64() >> 16;
        let r = raw % self.n_seen;
        if r < self.k as u64 {
            self.reservoir[r as usize] = x;
        }
    }

    /// Current snapshot of the reservoir.
    #[must_use]
    pub fn sample(&self) -> &[u64] {
        &self.reservoir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_constructs() {
        let r = ReservoirSampler::new(8, 0).expect("ok");
        assert_eq!(r.k, 8);
    }

    #[test]
    fn reservoir_invalid_k() {
        assert!(ReservoirSampler::new(0, 0).is_err());
    }

    #[test]
    fn reservoir_collects_first_k() {
        let mut r = ReservoirSampler::new(5, 0).expect("ok");
        for i in 0..5u64 {
            r.add(i);
        }
        assert_eq!(r.sample().len(), 5);
        assert!(r.sample().iter().all(|&v| v < 5));
    }

    #[test]
    fn reservoir_uniform_distribution() {
        // Over many trials, each item should appear approximately uniformly.
        // Use well-spaced seeds derived from a Knuth multiplicative scrambler to
        // avoid LCG-correlated startup transients when seeds differ by 1.
        let trials = 5000usize;
        let n_items = 20usize;
        let k = 5usize;
        let mut counts = vec![0usize; n_items];
        for t in 0..trials {
            let seed = (t as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(1);
            let mut r = ReservoirSampler::new(k, seed).expect("ok");
            for i in 0..n_items as u64 {
                r.add(i);
            }
            for &v in r.sample() {
                counts[v as usize] += 1;
            }
        }
        // Each item should appear roughly trials * k / n_items times.
        let expected = (trials * k) / n_items;
        for &c in &counts {
            let rel = (c as f64 - expected as f64).abs() / (expected as f64);
            assert!(
                rel < 0.25,
                "reservoir non-uniform: count {c}, expected {expected}"
            );
        }
    }
}
