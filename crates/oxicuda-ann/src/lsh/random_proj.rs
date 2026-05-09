use crate::handle::LcgRng;

/// Sign-random-projection LSH for L2 distance.
pub struct RandomProjLsh {
    /// Projection matrix `[n_hashes, dim]` with Gaussian entries.
    w: Vec<f32>,
    pub n_hashes: usize,
    pub dim: usize,
}

impl RandomProjLsh {
    /// Create a new LSH with `n_hashes` random hyperplanes in `dim` dimensions.
    #[must_use]
    pub fn new(n_hashes: usize, dim: usize, rng: &mut LcgRng) -> Self {
        let mut w = vec![0.0_f32; n_hashes * dim];
        rng.fill_normal(&mut w);
        Self { w, n_hashes, dim }
    }

    /// Compute sign-bit hash of `v`. Returns `ceil(n_hashes/32)` packed u32s.
    #[must_use]
    pub fn hash(&self, v: &[f32]) -> Vec<u32> {
        let n_words = self.n_hashes.div_ceil(32);
        let mut bits = vec![0u32; n_words];
        for j in 0..self.n_hashes {
            let row = &self.w[j * self.dim..(j + 1) * self.dim];
            let dot: f32 = row.iter().zip(v.iter()).map(|(w, x)| w * x).sum();
            if dot >= 0.0 {
                bits[j / 32] |= 1u32 << (j % 32);
            }
        }
        bits
    }

    /// Theoretical probability that two points at L2 distance `dist` hash to the
    /// same bucket for a single hyperplane split with bandwidth `w_scale`.
    #[must_use]
    pub fn collision_prob_l2(dist: f32, w_scale: f32) -> f32 {
        if dist <= 0.0 || w_scale <= 0.0 {
            return 1.0;
        }
        let t = dist / w_scale;
        // P = 1 - 2*Φ(-t) - (2/√(2π*t))(1 - exp(-t²/2))  (standard LSH formula)
        // Simpler approximate: 1 - t * (2/π).atan() using standard E[cos] formula
        // Exact: P(collision) = 1 - (1/π)*arccos(... / w)
        // For sign-RP (no width): collision prob = 1 - angle/π
        // Approximation via erf:
        let erf_approx = 1.0 - (t * std::f32::consts::FRAC_2_SQRT_PI * (-t * t).exp()).min(1.0);
        erf_approx.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_vector_identical_hash() {
        let mut rng = LcgRng::new(42);
        let lsh = RandomProjLsh::new(64, 8, &mut rng);
        let v = vec![1.0_f32, -1.0, 0.5, 2.0, -0.3, 1.1, 0.0, 0.9];
        assert_eq!(lsh.hash(&v), lsh.hash(&v));
    }
}
