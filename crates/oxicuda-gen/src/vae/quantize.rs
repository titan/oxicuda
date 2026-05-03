//! VQ-VAE vector quantisation codebook.
//!
//! Implements nearest-neighbour codebook lookup, EMA update for training,
//! and commitment loss computation.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── VqCodebook ───────────────────────────────────────────────────────────────

/// VQ-VAE codebook with exponential moving average (EMA) update.
///
/// The codebook contains `n_codes` embedding vectors, each of dimension
/// `embed_dim`. Quantisation maps each input vector to its nearest codebook
/// entry using L2 distance.
///
/// # Reference
/// Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017.
/// Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", NeurIPS 2019.
#[derive(Debug, Clone)]
pub struct VqCodebook {
    /// Codebook embeddings: row-major `[n_codes × embed_dim]`.
    embeddings: Vec<f32>,
    n_codes: usize,
    embed_dim: usize,
    /// EMA decay factor γ ∈ (0, 1).
    decay: f32,
    /// EMA cluster counts (soft, per code).
    ema_cluster_size: Vec<f32>,
    /// EMA embedding sums (for computing updated embeddings).
    ema_embeddings: Vec<f32>,
}

impl VqCodebook {
    /// Create a new VQ codebook with random uniform initialisation.
    ///
    /// # Arguments
    /// - `n_codes`: Number of codebook entries (must be a power of two >= 2).
    /// - `embed_dim`: Dimensionality of each codebook vector.
    ///
    /// # Errors
    /// - `InvalidCodebookSize` if `n_codes` is not a power of two >= 2
    /// - `EmptyInput` if `embed_dim == 0`
    pub fn new(n_codes: usize, embed_dim: usize) -> GenResult<Self> {
        if embed_dim == 0 {
            return Err(GenError::EmptyInput("embed_dim must be > 0"));
        }
        if n_codes < 2 || !n_codes.is_power_of_two() {
            return Err(GenError::InvalidCodebookSize(n_codes));
        }
        // Use seeded RNG for deterministic initialisation
        let mut rng = LcgRng::new(0xDEAD_BEEF);
        let total = n_codes * embed_dim;
        let embeddings: Vec<f32> = (0..total).map(|_| (rng.next_f32() - 0.5) * 0.1).collect();
        let ema_cluster_size = vec![1.0_f32; n_codes];
        let ema_embeddings = embeddings.clone();
        Ok(Self {
            embeddings,
            n_codes,
            embed_dim,
            decay: 0.99,
            ema_cluster_size,
            ema_embeddings,
        })
    }

    /// Create a codebook with a specified EMA decay rate.
    pub fn with_decay(n_codes: usize, embed_dim: usize, decay: f32) -> GenResult<Self> {
        let mut cb = Self::new(n_codes, embed_dim)?;
        cb.decay = decay.clamp(0.0, 1.0 - 1e-7);
        Ok(cb)
    }

    /// Create a codebook from given embedding weights.
    ///
    /// # Errors
    /// - `InvalidCodebookSize` if n_codes is not a power of two >= 2
    /// - `DimensionMismatch` if `embeddings.len() != n_codes * embed_dim`
    pub fn from_embeddings(
        embeddings: Vec<f32>,
        n_codes: usize,
        embed_dim: usize,
    ) -> GenResult<Self> {
        if n_codes < 2 || !n_codes.is_power_of_two() {
            return Err(GenError::InvalidCodebookSize(n_codes));
        }
        if embeddings.len() != n_codes * embed_dim {
            return Err(GenError::DimensionMismatch {
                expected: n_codes * embed_dim,
                got: embeddings.len(),
            });
        }
        let ema_cluster_size = vec![1.0_f32; n_codes];
        let ema_embeddings = embeddings.clone();
        Ok(Self {
            embeddings,
            n_codes,
            embed_dim,
            decay: 0.99,
            ema_cluster_size,
            ema_embeddings,
        })
    }

    /// Compute L2 distance squared between `z` and codebook entry `k`.
    fn dist_sq(&self, z: &[f32], k: usize) -> f32 {
        let start = k * self.embed_dim;
        let end = start + self.embed_dim;
        z.iter()
            .zip(&self.embeddings[start..end])
            .map(|(&zi, &ei)| {
                let d = zi - ei;
                d * d
            })
            .sum()
    }

    /// Find the nearest codebook entry for a single vector `z`.
    ///
    /// Returns `(quantized_vec, index)`.
    fn nearest(&self, z: &[f32]) -> (Vec<f32>, usize) {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for k in 0..self.n_codes {
            let d = self.dist_sq(z, k);
            if d < best_dist {
                best_dist = d;
                best_idx = k;
            }
        }
        let start = best_idx * self.embed_dim;
        let quantized = self.embeddings[start..start + self.embed_dim].to_vec();
        (quantized, best_idx)
    }

    /// Quantise a batch of vectors.
    ///
    /// # Arguments
    /// - `z`: Flat array of shape `[n_vectors × embed_dim]`.
    ///
    /// # Returns
    /// - `(quantized, indices)`: Quantised vectors and their codebook indices.
    ///
    /// # Errors
    /// - `EmptyInput` if `z` is empty
    /// - `DimensionMismatch` if `z.len()` is not divisible by `embed_dim`
    pub fn quantize(&self, z: &[f32]) -> GenResult<(Vec<f32>, Vec<usize>)> {
        if z.is_empty() {
            return Err(GenError::EmptyInput("z is empty"));
        }
        if z.len() % self.embed_dim != 0 {
            return Err(GenError::DimensionMismatch {
                expected: z.len() - z.len() % self.embed_dim,
                got: z.len(),
            });
        }
        let n = z.len() / self.embed_dim;
        let mut quantized = Vec::with_capacity(z.len());
        let mut indices = Vec::with_capacity(n);
        for i in 0..n {
            let zi = &z[i * self.embed_dim..(i + 1) * self.embed_dim];
            let (q, idx) = self.nearest(zi);
            quantized.extend(q);
            indices.push(idx);
        }
        Ok((quantized, indices))
    }

    /// EMA update of codebook embeddings.
    ///
    /// For each code `k`, gathers all `z` vectors assigned to `k` and
    /// updates:
    /// - `ema_cluster_size[k] ← γ * ema_cluster_size[k] + (1-γ) * count_k`
    /// - `ema_embeddings[k] ← γ * ema_embeddings[k] + (1-γ) * sum_k`
    /// - `embeddings[k] ← ema_embeddings[k] / ema_cluster_size[k]`
    ///
    /// # Errors
    /// - `EmptyInput` if `z` or `indices` are empty
    /// - `DimensionMismatch` on shape mismatch
    pub fn ema_update(&mut self, z: &[f32], indices: &[usize]) -> GenResult<()> {
        if z.is_empty() {
            return Err(GenError::EmptyInput("z is empty"));
        }
        if indices.is_empty() {
            return Err(GenError::EmptyInput("indices is empty"));
        }
        let n = indices.len();
        if z.len() != n * self.embed_dim {
            return Err(GenError::DimensionMismatch {
                expected: n * self.embed_dim,
                got: z.len(),
            });
        }
        let gamma = self.decay;
        // Accumulate cluster counts and embedding sums
        let mut counts = vec![0_u32; self.n_codes];
        let mut sums = vec![0.0_f32; self.n_codes * self.embed_dim];
        for (i, &k) in indices.iter().enumerate() {
            if k < self.n_codes {
                counts[k] += 1;
                let zi = &z[i * self.embed_dim..(i + 1) * self.embed_dim];
                let start = k * self.embed_dim;
                for (j, &v) in zi.iter().enumerate() {
                    sums[start + j] += v;
                }
            }
        }
        // EMA update
        for (k, &count_raw) in counts.iter().enumerate() {
            let count = count_raw as f32;
            self.ema_cluster_size[k] = gamma * self.ema_cluster_size[k] + (1.0 - gamma) * count;
            let start = k * self.embed_dim;
            for j in 0..self.embed_dim {
                self.ema_embeddings[start + j] =
                    gamma * self.ema_embeddings[start + j] + (1.0 - gamma) * sums[start + j];
            }
            // Normalise by smoothed cluster size
            let n_smooth = self.ema_cluster_size[k].max(1e-5);
            for j in 0..self.embed_dim {
                self.embeddings[start + j] = self.ema_embeddings[start + j] / n_smooth;
            }
        }
        Ok(())
    }

    /// Compute the commitment loss for a batch.
    ///
    /// `L_commit = ||z - sg[e]||² / n`
    /// where `sg` denotes stop-gradient (the quantized values are treated as constants).
    ///
    /// # Errors
    /// - `EmptyInput` if inputs are empty
    /// - `DimensionMismatch` if shapes differ
    /// - `NonFiniteCommitmentLoss` if the result is not finite
    pub fn commitment_loss(&self, z: &[f32], quantized: &[f32]) -> GenResult<f32> {
        if z.is_empty() {
            return Err(GenError::EmptyInput("z is empty"));
        }
        if z.len() != quantized.len() {
            return Err(GenError::DimensionMismatch {
                expected: z.len(),
                got: quantized.len(),
            });
        }
        let n = z.len() as f32;
        let loss = z
            .iter()
            .zip(quantized)
            .map(|(&zi, &qi)| {
                let d = zi - qi;
                d * d
            })
            .sum::<f32>()
            / n;
        if !loss.is_finite() {
            return Err(GenError::NonFiniteCommitmentLoss(loss));
        }
        Ok(loss)
    }

    /// Return the number of codebook entries.
    pub fn n_codes(&self) -> usize {
        self.n_codes
    }

    /// Return the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Return the embedding matrix (read-only).
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    /// Return the EMA decay factor.
    pub fn decay(&self) -> f32 {
        self.decay
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_codebook_valid() {
        let cb = VqCodebook::new(16, 64).unwrap();
        assert_eq!(cb.n_codes(), 16);
        assert_eq!(cb.embed_dim(), 64);
        assert_eq!(cb.embeddings().len(), 16 * 64);
    }

    #[test]
    fn invalid_codebook_size_rejected() {
        assert!(VqCodebook::new(3, 64).is_err()); // not power of two
        assert!(VqCodebook::new(1, 64).is_err()); // < 2
        assert!(VqCodebook::new(0, 64).is_err()); // 0
        assert!(VqCodebook::new(4, 64).is_ok()); // valid: 4 = 2^2
    }

    #[test]
    fn quantize_output_shape() {
        let cb = VqCodebook::new(8, 4).unwrap();
        let z = vec![0.1_f32; 8 * 4]; // 8 vectors of dim 4
        let (q, idx) = cb.quantize(&z).unwrap();
        assert_eq!(q.len(), 8 * 4);
        assert_eq!(idx.len(), 8);
    }

    #[test]
    fn quantize_indices_in_range() {
        let cb = VqCodebook::new(16, 8).unwrap();
        let z = vec![0.5_f32; 4 * 8]; // 4 vectors
        let (_, idx) = cb.quantize(&z).unwrap();
        for &i in &idx {
            assert!(i < 16, "index {i} out of range");
        }
    }

    #[test]
    fn nearest_lookup_exact_match() {
        // Insert a known vector into the codebook and check it maps to itself
        let embed_dim = 4;
        let n_codes = 4;
        let mut embeddings = vec![0.0_f32; n_codes * embed_dim];
        // Code 2: [1, 0, 0, 0]
        embeddings[2 * embed_dim] = 1.0;
        let cb = VqCodebook::from_embeddings(embeddings, n_codes, embed_dim).unwrap();
        let z = vec![1.0_f32, 0.0, 0.0, 0.0];
        let (_, idx) = cb.quantize(&z).unwrap();
        assert_eq!(idx[0], 2, "should map to code 2");
    }

    #[test]
    fn commitment_loss_zero_for_quantized() {
        let cb = VqCodebook::new(8, 4).unwrap();
        let z = vec![0.0_f32; 8];
        let commitment = cb.commitment_loss(&z, &z).unwrap();
        assert!(
            commitment.abs() < 1e-7,
            "commitment loss should be 0 for identical: {commitment}"
        );
    }

    #[test]
    fn commitment_loss_positive_for_different() {
        let cb = VqCodebook::new(8, 4).unwrap();
        let z = vec![1.0_f32; 8];
        let q = vec![0.0_f32; 8];
        let commitment = cb.commitment_loss(&z, &q).unwrap();
        assert!(
            commitment > 0.0,
            "commitment loss should be positive: {commitment}"
        );
    }

    #[test]
    fn ema_update_runs() {
        let mut cb = VqCodebook::new(8, 4).unwrap();
        let z = vec![0.5_f32; 4];
        let (_, idx) = cb.quantize(&z).unwrap();
        // Should not error
        cb.ema_update(&z, &idx).unwrap();
    }

    #[test]
    fn quantize_dimension_mismatch() {
        let cb = VqCodebook::new(8, 4).unwrap();
        let z = vec![0.0_f32; 7]; // 7 is not divisible by 4
        assert!(matches!(
            cb.quantize(&z),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn commitment_loss_dimension_mismatch() {
        let cb = VqCodebook::new(8, 4).unwrap();
        let z = vec![1.0_f32; 8];
        let q = vec![0.0_f32; 4];
        assert!(matches!(
            cb.commitment_loss(&z, &q),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn ema_update_changes_embeddings() {
        let mut cb = VqCodebook::with_decay(4, 2, 0.5).unwrap();
        let original = cb.embeddings().to_vec();
        // Quantize and update
        let z = vec![10.0_f32, 10.0, -10.0, -10.0];
        let (_, idx) = cb.quantize(&z).unwrap();
        cb.ema_update(&z, &idx).unwrap();
        let updated = cb.embeddings().to_vec();
        // At least some embeddings should have changed
        let changed = original
            .iter()
            .zip(&updated)
            .any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(changed, "EMA update should change embeddings");
    }
}
