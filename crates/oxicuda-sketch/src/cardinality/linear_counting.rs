//! Linear counting cardinality estimator (Whang, Vander-Zanden, Taylor 1990).
//!
//! Uses an m-bit bitmap. After insertions, let V = number of unset bits.
//! Estimate: `n = m * ln(m / V)`.
//! Best when expected cardinality is `<= ~2.5 * m`.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// Linear-counting sketch with an `m`-bit bitmap stored in u64 words.
#[derive(Debug, Clone)]
pub struct LinearCounter {
    pub m: usize,
    pub seed: u64,
    pub bits: Vec<u64>,
}

impl LinearCounter {
    /// Create a new linear counter with `m` bits.
    pub fn new(m: usize, seed: u64) -> SketchResult<Self> {
        if m == 0 {
            return Err(SketchError::InvalidParameter {
                name: "m".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let words = m.div_ceil(64);
        Ok(Self {
            m,
            seed,
            bits: vec![0u64; words],
        })
    }

    /// Insert a `u64` value.
    pub fn add_u64(&mut self, x: u64) {
        let h = xxh3_64_u64(x, self.seed);
        let idx = (h as usize) % self.m;
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// Count the number of set bits (popcount over all words).
    #[must_use]
    pub fn popcount(&self) -> usize {
        let mut s = 0usize;
        for &w in &self.bits {
            s += w.count_ones() as usize;
        }
        // Last word may have padding bits that are always zero (since we only set
        // bits at indices < m).
        s
    }

    /// Number of unset bits among the m valid bits.
    #[must_use]
    pub fn unset_count(&self) -> usize {
        self.m - self.popcount()
    }

    /// Estimate distinct cardinality. Returns `+inf` if the bitmap is full (overflow).
    #[must_use]
    pub fn estimate(&self) -> f64 {
        let v = self.unset_count();
        if v == 0 {
            return f64::INFINITY;
        }
        (self.m as f64) * ((self.m as f64) / v as f64).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lc_constructs() {
        let lc = LinearCounter::new(64, 0).expect("ok");
        assert_eq!(lc.m, 64);
        assert_eq!(lc.bits.len(), 1);
    }

    #[test]
    fn lc_empty_estimate_zero() {
        let lc = LinearCounter::new(1024, 0).expect("ok");
        assert!(lc.estimate() < 1.0);
    }

    #[test]
    fn lc_estimate_close_to_n() {
        let mut lc = LinearCounter::new(2048, 0).expect("ok");
        for i in 0..500u64 {
            lc.add_u64(i);
        }
        let e = lc.estimate();
        let rel = (e - 500.0).abs() / 500.0;
        assert!(rel < 0.15, "LC rel-err {rel}");
    }

    #[test]
    fn lc_zero_m_errs() {
        assert!(LinearCounter::new(0, 0).is_err());
    }
}
