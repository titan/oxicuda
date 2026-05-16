//! HyperLogLog++ cardinality estimator (Heule, Nunkesser, Hall 2013).
//!
//! Improvements over classic HLL:
//! - 6-bit registers (packed conceptually; stored as u8 here for simplicity).
//! - Sparse representation (linear-counting style) for low cardinality.
//! - Empirical bias correction at small cardinalities.
//!
//! We implement the *dense* representation with a coarse bias correction inspired by the
//! Heule paper (piecewise correction in the small-range region). Sparse representation is
//! emulated via an open-addressed sparse-set up to a threshold, then converted to dense.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

const SPARSE_THRESHOLD_FRAC: usize = 4;

/// HyperLogLog++ sketch with optional sparse representation.
#[derive(Debug, Clone)]
pub struct HyperLogLogPlus {
    pub p: u32,
    pub m: usize,
    pub seed: u64,
    /// Dense register array. When sparse, this is empty.
    pub dense: Vec<u8>,
    /// Sparse representation: list of (register_index, value) pairs.
    pub sparse: Vec<(u32, u8)>,
    pub use_sparse: bool,
}

impl HyperLogLogPlus {
    /// Create a new HLL++ sketch with precision `p`, `4 <= p <= 16`.
    pub fn new(p: u32, seed: u64) -> SketchResult<Self> {
        if !(4..=16).contains(&p) {
            return Err(SketchError::InvalidPrecision(p));
        }
        let m = 1usize << p;
        Ok(Self {
            p,
            m,
            seed,
            dense: Vec::new(),
            sparse: Vec::new(),
            use_sparse: true,
        })
    }

    /// Insert a `u64` value.
    pub fn add_u64(&mut self, x: u64) {
        let h = xxh3_64_u64(x, self.seed);
        self.add_hash(h);
    }

    /// Insert a raw 64-bit hash.
    pub fn add_hash(&mut self, h: u64) {
        let idx = (h >> (64 - self.p)) as usize;
        let w = (h << self.p) | (1u64 << self.p.saturating_sub(1));
        let lz = (w.leading_zeros() as u8) + 1;
        let new_val = lz.min(64);
        if self.use_sparse {
            // Linear scan; replace if larger.
            let mut updated = false;
            for entry in self.sparse.iter_mut() {
                if entry.0 as usize == idx {
                    if entry.1 < new_val {
                        entry.1 = new_val;
                    }
                    updated = true;
                    break;
                }
            }
            if !updated {
                self.sparse.push((idx as u32, new_val));
            }
            if self.sparse.len() > self.m / SPARSE_THRESHOLD_FRAC {
                self.convert_to_dense();
            }
        } else {
            if new_val > self.dense[idx] {
                self.dense[idx] = new_val;
            }
        }
    }

    /// Convert from sparse to dense representation.
    fn convert_to_dense(&mut self) {
        let mut dense = vec![0u8; self.m];
        for &(idx, val) in &self.sparse {
            let i = idx as usize;
            if i < self.m && val > dense[i] {
                dense[i] = val;
            }
        }
        self.dense = dense;
        self.sparse.clear();
        self.use_sparse = false;
    }

    /// Estimate distinct cardinality.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;

        if self.use_sparse {
            // Linear counting on the sparse representation when small.
            // V = m - len(sparse)  (number of registers still at 0)
            let v = self.m - self.sparse.len();
            if v == 0 {
                return self.dense_raw_estimate();
            }
            return m * (m / v as f64).ln();
        }

        // Dense path.
        let mut sum = 0.0;
        let mut zero_count = 0usize;
        for &reg in &self.dense {
            sum += 2.0_f64.powi(-(reg as i32));
            if reg == 0 {
                zero_count += 1;
            }
        }
        let alpha = Self::alpha(self.m);
        let raw = alpha * m * m / sum;
        // Bias correction in the small-range zone.
        if raw <= 2.5 * m && zero_count > 0 {
            let lc = m * (m / zero_count as f64).ln();
            // Heule et al. recommend an empirical-bias correction; we use a
            // simplified linear blend: lc when very small, raw otherwise.
            if lc < 2.0 * m {
                return lc;
            }
        }
        raw
    }

    /// Raw HLL estimate without correction (used when sparse-V = 0).
    fn dense_raw_estimate(&self) -> f64 {
        let m = self.m as f64;
        let mut sum = 0.0;
        for &reg in &self.dense {
            sum += 2.0_f64.powi(-(reg as i32));
        }
        let alpha = Self::alpha(self.m);
        alpha * m * m / sum
    }

    /// Alpha constant.
    fn alpha(m: usize) -> f64 {
        match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m as f64),
        }
    }

    /// Merge another HLL++ into this one.
    pub fn merge(&mut self, other: &HyperLogLogPlus) -> SketchResult<()> {
        if self.p != other.p {
            return Err(SketchError::DimensionMismatch {
                a: self.p as usize,
                b: other.p as usize,
            });
        }
        // Force both to dense for merge.
        if self.use_sparse {
            self.convert_to_dense();
        }
        if other.use_sparse {
            // Project other.sparse into dense slots.
            for &(idx, val) in &other.sparse {
                let i = idx as usize;
                if i < self.m && val > self.dense[i] {
                    self.dense[i] = val;
                }
            }
        } else {
            for i in 0..self.m {
                if other.dense[i] > self.dense[i] {
                    self.dense[i] = other.dense[i];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hllp_constructs() {
        let h = HyperLogLogPlus::new(10, 0).expect("ok");
        assert_eq!(h.m, 1024);
        assert!(h.use_sparse);
    }

    #[test]
    fn hllp_invalid_precision() {
        assert!(HyperLogLogPlus::new(3, 0).is_err());
        assert!(HyperLogLogPlus::new(17, 0).is_err());
    }

    #[test]
    fn hllp_sparse_to_dense_conversion() {
        let mut h = HyperLogLogPlus::new(8, 0).expect("ok");
        // Threshold is m/4 = 64 unique slots. Insert many distinct values to force conversion.
        for i in 0..500u64 {
            h.add_u64(i);
        }
        assert!(!h.use_sparse, "should have converted to dense");
    }

    #[test]
    fn hllp_estimate_accuracy() {
        let mut h = HyperLogLogPlus::new(14, 0).expect("ok");
        for i in 0..10_000u64 {
            h.add_u64(i);
        }
        let e = h.estimate();
        let rel = (e - 10_000.0).abs() / 10_000.0;
        assert!(rel < 0.05, "HLL++ relative error {rel}");
    }

    #[test]
    fn hllp_small_cardinality_accurate() {
        let mut h = HyperLogLogPlus::new(12, 0).expect("ok");
        for i in 0..50u64 {
            h.add_u64(i);
        }
        let e = h.estimate();
        let rel = (e - 50.0).abs() / 50.0;
        assert!(rel < 0.1, "small-card HLL++ rel-err {rel}");
    }
}
