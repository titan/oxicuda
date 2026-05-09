use crate::handle::LcgRng;

/// SimHash for cosine similarity / angular distance.
pub struct SimHash {
    /// Random projection vectors `[n_bits, dim]`.
    w: Vec<f32>,
    pub n_bits: usize,
    pub dim: usize,
}

impl SimHash {
    /// Create a new SimHash with `n_bits` random hyperplanes (n_bits ≤ 64).
    #[must_use]
    pub fn new(n_bits: usize, dim: usize, rng: &mut LcgRng) -> Self {
        let n_bits = n_bits.min(64);
        let mut w = vec![0.0_f32; n_bits * dim];
        rng.fill_normal(&mut w);
        Self { w, n_bits, dim }
    }

    /// Compute a `u64` hash by packing sign bits of `W * v`.
    #[must_use]
    pub fn hash(&self, v: &[f32]) -> u64 {
        let mut h: u64 = 0;
        for j in 0..self.n_bits {
            let row = &self.w[j * self.dim..(j + 1) * self.dim];
            let dot: f32 = row.iter().zip(v.iter()).map(|(w, x)| w * x).sum();
            if dot >= 0.0 {
                h |= 1u64 << j;
            }
        }
        h
    }

    /// Hamming distance between two SimHash values.
    #[must_use]
    pub fn hamming_bits(h1: u64, h2: u64) -> u32 {
        (h1 ^ h2).count_ones()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_vector_zero_hamming() {
        let mut rng = LcgRng::new(17);
        let sh = SimHash::new(64, 8, &mut rng);
        let v = vec![1.0_f32, 0.0, -1.0, 0.5, 2.0, -0.5, 1.0, 0.0];
        let h = sh.hash(&v);
        assert_eq!(SimHash::hamming_bits(h, h), 0);
    }
}
