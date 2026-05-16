//! SimHash signatures for cosine similarity (Charikar 2002).
//!
//! For each feature with weight w, hash → ±w; aggregate vector → sign per bit.
//! Cosine similarity ≈ 1 − 2 * hamming / d where `d` is the signature length in bits.

use crate::error::{SketchError, SketchResult};
use crate::hash::xxh3_min::xxh3_64_u64;

/// SimHash signature.
#[derive(Debug, Clone)]
pub struct SimHash {
    pub d: usize,
    pub seed: u64,
    pub accumulator: Vec<f64>,
}

impl SimHash {
    /// Create a new empty SimHash with bit-width `d` (multiple of 64 recommended).
    pub fn new(d: usize, seed: u64) -> SketchResult<Self> {
        if d == 0 {
            return Err(SketchError::InvalidParameter {
                name: "d".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            d,
            seed,
            accumulator: vec![0.0; d],
        })
    }

    /// Add a feature `f` with weight `w`. The hash determines per-bit sign.
    pub fn add_feature(&mut self, f: u64, w: f64) {
        // For each bit of the d-bit hash, accumulate +w if bit=1 else -w.
        // Generate hashed bits across multiple words.
        let words = self.d.div_ceil(64);
        for word in 0..words {
            let h = xxh3_64_u64(f, self.seed.wrapping_add(word as u64));
            for bit in 0..64 {
                let idx = word * 64 + bit;
                if idx >= self.d {
                    break;
                }
                let s = if (h >> bit) & 1 == 1 { w } else { -w };
                self.accumulator[idx] += s;
            }
        }
    }

    /// Finalize: produce a packed bit vector (`Vec<u64>` of length ceil(d/64)) from sign of accumulator.
    #[must_use]
    pub fn signature(&self) -> Vec<u64> {
        let words = self.d.div_ceil(64);
        let mut out = vec![0u64; words];
        for i in 0..self.d {
            if self.accumulator[i] > 0.0 {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }

    /// Estimate cosine similarity between two SimHash sketches of same dimension.
    pub fn cosine_similarity(&self, other: &SimHash) -> SketchResult<f64> {
        if self.d != other.d {
            return Err(SketchError::DimensionMismatch {
                a: self.d,
                b: other.d,
            });
        }
        let sa = self.signature();
        let sb = other.signature();
        let mut hamming = 0usize;
        for i in 0..sa.len() {
            hamming += (sa[i] ^ sb[i]).count_ones() as usize;
        }
        // Truncate trailing padding bits.
        let total_bits = sa.len() * 64;
        let extra = total_bits - self.d;
        // Hamming over extra padded bits is zero by construction.
        let _ = extra;
        let frac = (hamming as f64) / (self.d as f64);
        Ok(1.0 - 2.0 * frac)
    }

    /// Hamming distance to another signature (`Vec<u64>`) of same length.
    #[must_use]
    pub fn hamming_distance(a: &[u64], b: &[u64]) -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x ^ y).count_ones() as usize)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simhash_constructs() {
        let s = SimHash::new(64, 0).expect("ok");
        assert_eq!(s.d, 64);
    }

    #[test]
    fn simhash_invalid_d() {
        assert!(SimHash::new(0, 0).is_err());
    }

    #[test]
    fn simhash_identical_inputs() {
        let mut a = SimHash::new(128, 0).expect("ok");
        let mut b = SimHash::new(128, 0).expect("ok");
        for i in 1..=100u64 {
            a.add_feature(i, 1.0);
            b.add_feature(i, 1.0);
        }
        let cs = a.cosine_similarity(&b).expect("ok");
        assert!((cs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn simhash_signature_packed() {
        let mut s = SimHash::new(128, 0).expect("ok");
        s.add_feature(42, 1.0);
        let sig = s.signature();
        assert_eq!(sig.len(), 2);
    }
}
