//! Cosine LSH using random hyperplanes (SimHash-style).
//!
//! For each of K hyperplanes h_i (drawn from N(0,1)^d), the LSH bit is sign(h_i · x).
//! Two vectors collide (same bit) with probability `(π - θ) / π` where `θ` is the angle.

use crate::error::{SketchError, SketchResult};
use crate::handle::LcgRng;

/// Cosine LSH parameters.
#[derive(Debug, Clone)]
pub struct CosineLsh {
    pub n_bits: usize,
    pub dim: usize,
    pub hyperplanes: Vec<f64>, // n_bits x dim row-major
}

impl CosineLsh {
    /// Construct with `n_bits` hyperplanes in `dim`-dimensional space.
    pub fn new(n_bits: usize, dim: usize, rng: &mut LcgRng) -> SketchResult<Self> {
        if n_bits == 0 || dim == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(n_bits, dim)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        let mut hp = vec![0.0; n_bits * dim];
        for v in hp.iter_mut() {
            *v = rng.next_normal();
        }
        Ok(Self {
            n_bits,
            dim,
            hyperplanes: hp,
        })
    }

    /// Compute the LSH signature for a vector `x` of length `dim`.
    pub fn signature(&self, x: &[f64]) -> SketchResult<Vec<u64>> {
        if x.len() != self.dim {
            return Err(SketchError::DimensionMismatch {
                a: x.len(),
                b: self.dim,
            });
        }
        let words = self.n_bits.div_ceil(64);
        let mut out = vec![0u64; words];
        for b in 0..self.n_bits {
            let mut dot = 0.0;
            for (j, &xj) in x.iter().enumerate().take(self.dim) {
                dot += self.hyperplanes[b * self.dim + j] * xj;
            }
            if dot >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
        Ok(out)
    }

    /// Hamming distance between two signatures.
    #[must_use]
    pub fn hamming_distance(a: &[u64], b: &[u64]) -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x ^ y).count_ones() as usize)
            .sum()
    }

    /// Estimate cosine similarity from hamming distance of signatures.
    /// `cos(θ) ≈ cos(π * hamming / n_bits)`.
    #[must_use]
    pub fn cosine_estimate(&self, ham: usize) -> f64 {
        let theta_estimate = std::f64::consts::PI * (ham as f64) / (self.n_bits as f64);
        theta_estimate.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_lsh_constructs() {
        let mut rng = LcgRng::new(11);
        let c = CosineLsh::new(64, 32, &mut rng).expect("ok");
        assert_eq!(c.n_bits, 64);
    }

    #[test]
    fn cosine_lsh_invalid_params() {
        let mut rng = LcgRng::new(0);
        assert!(CosineLsh::new(0, 4, &mut rng).is_err());
        assert!(CosineLsh::new(4, 0, &mut rng).is_err());
    }

    #[test]
    fn cosine_lsh_same_vector_signature_match() {
        let mut rng = LcgRng::new(11);
        let lsh = CosineLsh::new(128, 32, &mut rng).expect("ok");
        let x: Vec<f64> = (0..32).map(|i| (i as f64) - 16.0).collect();
        let s1 = lsh.signature(&x).expect("ok");
        let s2 = lsh.signature(&x).expect("ok");
        let d = CosineLsh::hamming_distance(&s1, &s2);
        assert_eq!(d, 0);
    }

    #[test]
    fn cosine_lsh_opposite_vectors_far() {
        let mut rng = LcgRng::new(11);
        let lsh = CosineLsh::new(128, 32, &mut rng).expect("ok");
        let x: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| -v).collect();
        let sx = lsh.signature(&x).expect("ok");
        let sy = lsh.signature(&y).expect("ok");
        let d = CosineLsh::hamming_distance(&sx, &sy);
        // Opposite vectors → all bits flipped (with high probability).
        assert!(d > 100, "hamming dist {d}");
    }
}
