//! Projection head for CLIP embedding normalisation.
//!
//! Maps encoder output embeddings to a shared CLIP embedding space via a
//! linear projection followed by L2 normalisation.  The normalised embeddings
//! are ready for cosine-similarity comparison without an explicit division.

use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── ProjectionWeights ───────────────────────────────────────────────────────

/// Learnable weights for the linear projection layer.
///
/// The weight matrix has layout `[proj_dim × embed_dim]` (row-major):
/// row `p` contains the `embed_dim` weights for output dimension `p`.
pub struct ProjectionWeights {
    /// Weight matrix: flat `[proj_dim × embed_dim]`.
    pub weight: Vec<f32>,
    /// Bias vector: flat `[proj_dim]`.
    pub bias: Vec<f32>,
}

impl ProjectionWeights {
    /// Initialise weights with `N(0, 1/√embed_dim)` and bias with `N(0, 1/√embed_dim)`.
    ///
    /// This is a scaled Gaussian init that keeps the output variance
    /// independent of the input dimension.
    ///
    /// # Errors
    /// Returns [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
    /// Returns [`VisionError::InvalidProjDim`] if `proj_dim == 0`.
    pub fn default_init(embed_dim: usize, proj_dim: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        if embed_dim == 0 {
            return Err(VisionError::InvalidEmbedDim(embed_dim));
        }
        if proj_dim == 0 {
            return Err(VisionError::InvalidProjDim(proj_dim));
        }

        let scale = 1.0 / (embed_dim as f32).sqrt();

        let mut weight = vec![0.0f32; proj_dim * embed_dim];
        rng.fill_normal(&mut weight);
        for v in &mut weight {
            *v *= scale;
        }

        let mut bias = vec![0.0f32; proj_dim];
        rng.fill_normal(&mut bias);
        for v in &mut bias {
            *v *= scale;
        }

        Ok(Self { weight, bias })
    }
}

// ─── ProjectionHead ──────────────────────────────────────────────────────────

/// Linear projection + L2 normalisation.
///
/// Transforms an encoder embedding `x ∈ R^{embed_dim}` to a unit-norm
/// CLIP-space embedding `z ∈ R^{proj_dim}` via:
///
/// ```text
/// z_pre = W · x + b        (linear projection)
/// z     = z_pre / max(‖z_pre‖₂, 1e-12)   (L2 normalisation)
/// ```
pub struct ProjectionHead {
    /// Input dimension (encoder output size).
    pub embed_dim: usize,
    /// Output dimension (CLIP embedding space size).
    pub proj_dim: usize,
    /// Learnable weights.
    pub weights: ProjectionWeights,
}

impl ProjectionHead {
    /// Create a new projection head with default-initialised weights.
    ///
    /// # Errors
    /// Returns [`VisionError::InvalidEmbedDim`] if `embed_dim == 0`.
    /// Returns [`VisionError::InvalidProjDim`] if `proj_dim == 0`.
    pub fn new(embed_dim: usize, proj_dim: usize, rng: &mut LcgRng) -> VisionResult<Self> {
        let weights = ProjectionWeights::default_init(embed_dim, proj_dim, rng)?;
        Ok(Self {
            embed_dim,
            proj_dim,
            weights,
        })
    }

    /// Project a single embedding `x` and L2-normalise the result.
    ///
    /// # Parameters
    /// - `x`: flat `[embed_dim]` input embedding.
    ///
    /// # Returns
    /// Unit-norm `[proj_dim]` embedding.
    ///
    /// # Errors
    /// Returns [`VisionError::DimensionMismatch`] if `x.len() != embed_dim`.
    pub fn project(&self, x: &[f32]) -> VisionResult<Vec<f32>> {
        if x.len() != self.embed_dim {
            return Err(VisionError::DimensionMismatch {
                expected: self.embed_dim,
                got: x.len(),
            });
        }

        let mut z = vec![0.0f32; self.proj_dim];

        // z = W · x + b
        for (p, zp) in z.iter_mut().enumerate() {
            let row_off = p * self.embed_dim;
            let acc: f32 = self.weights.weight[row_off..row_off + self.embed_dim]
                .iter()
                .zip(x.iter())
                .map(|(&w, &xi)| w * xi)
                .sum::<f32>()
                + self.weights.bias[p];
            *zp = acc;
        }

        // L2 normalisation: z /= max(‖z‖₂, 1e-12)
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv_norm = 1.0 / norm.max(1e-12);
        for v in &mut z {
            *v *= inv_norm;
        }

        Ok(z)
    }

    /// Project a batch of embeddings.
    ///
    /// # Parameters
    /// - `x`: flat `[batch × embed_dim]` input embeddings.
    /// - `batch`: number of samples.
    ///
    /// # Returns
    /// Flat `[batch × proj_dim]` unit-norm embeddings.
    ///
    /// # Errors
    /// Returns [`VisionError::DimensionMismatch`] if `x.len() != batch * embed_dim`.
    pub fn project_batch(&self, x: &[f32], batch: usize) -> VisionResult<Vec<f32>> {
        let expected = batch * self.embed_dim;
        if x.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut out = vec![0.0f32; batch * self.proj_dim];

        for b in 0..batch {
            let x_slice = &x[b * self.embed_dim..(b + 1) * self.embed_dim];
            let z = self.project(x_slice)?;
            let out_off = b * self.proj_dim;
            out[out_off..out_off + self.proj_dim].copy_from_slice(&z);
        }

        Ok(out)
    }

    /// Cosine similarity between two vectors `a` and `b`.
    ///
    /// Returns `dot(a, b) / (‖a‖ · ‖b‖ + 1e-12)`.
    ///
    /// If `a` and `b` are already L2-normalised (as produced by [`Self::project`]),
    /// this reduces to their dot product.
    ///
    /// # Errors
    /// Returns [`VisionError::DimensionMismatch`] if `a.len() != b.len()`.
    pub fn cosine_sim(a: &[f32], b: &[f32]) -> VisionResult<f32> {
        if a.len() != b.len() {
            return Err(VisionError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
        let norm_a: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let denom = norm_a * norm_b + 1e-12;

        Ok(dot / denom)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_head(embed_dim: usize, proj_dim: usize, seed: u64) -> ProjectionHead {
        let mut rng = LcgRng::new(seed);
        ProjectionHead::new(embed_dim, proj_dim, &mut rng).expect("valid head")
    }

    fn random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0f32; len];
        rng.fill_normal(&mut v);
        v
    }

    // ── ProjectionWeights ────────────────────────────────────────────────────

    #[test]
    fn weights_correct_sizes() {
        let mut rng = LcgRng::new(1);
        let w = ProjectionWeights::default_init(64, 128, &mut rng).expect("ok");
        assert_eq!(w.weight.len(), 128 * 64, "weight size mismatch");
        assert_eq!(w.bias.len(), 128, "bias size mismatch");
    }

    #[test]
    fn weights_finite_values() {
        let mut rng = LcgRng::new(2);
        let w = ProjectionWeights::default_init(64, 128, &mut rng).expect("ok");
        assert!(w.weight.iter().all(|v| v.is_finite()), "non-finite weights");
        assert!(w.bias.iter().all(|v| v.is_finite()), "non-finite bias");
    }

    #[test]
    fn weights_error_zero_embed_dim() {
        let mut rng = LcgRng::new(3);
        let r = ProjectionWeights::default_init(0, 64, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidEmbedDim(0))));
    }

    #[test]
    fn weights_error_zero_proj_dim() {
        let mut rng = LcgRng::new(4);
        let r = ProjectionWeights::default_init(64, 0, &mut rng);
        assert!(matches!(r, Err(VisionError::InvalidProjDim(0))));
    }

    // ── ProjectionHead::project ──────────────────────────────────────────────

    #[test]
    fn project_output_l2_norm_approx_one() {
        let head = make_head(64, 128, 10);
        let x = random_vec(64, 11);
        let z = head.project(&x).expect("project ok");
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2 norm of projected embedding should be ≈1.0, got {norm}"
        );
    }

    #[test]
    fn project_output_size() {
        let head = make_head(32, 64, 12);
        let x = random_vec(32, 13);
        let z = head.project(&x).expect("project ok");
        assert_eq!(z.len(), 64, "output size should be proj_dim");
    }

    #[test]
    fn project_output_finite() {
        let head = make_head(128, 64, 14);
        let x = random_vec(128, 15);
        let z = head.project(&x).expect("project ok");
        assert!(z.iter().all(|v| v.is_finite()), "output must be finite");
    }

    #[test]
    fn project_error_wrong_input_size() {
        let head = make_head(64, 128, 16);
        let x = random_vec(32, 17); // wrong size: 32 ≠ 64
        let r = head.project(&x);
        assert!(
            matches!(
                r,
                Err(VisionError::DimensionMismatch {
                    expected: 64,
                    got: 32
                })
            ),
            "expected DimensionMismatch(64, 32), got {:?}",
            r
        );
    }

    #[test]
    fn project_zero_input_normalises() {
        // Zero input → linear output is the bias; should still be normalised
        // unless bias is also zero (which is practically impossible for random init).
        let head = make_head(16, 32, 18);
        let x = vec![0.0f32; 16];
        let z = head.project(&x).expect("ok");
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "zero-input projection should still yield unit norm, got {norm}"
        );
    }

    // ── ProjectionHead::project_batch ────────────────────────────────────────

    #[test]
    fn project_batch_output_size() {
        let head = make_head(32, 64, 20);
        let x = random_vec(4 * 32, 21); // batch=4
        let z = head.project_batch(&x, 4).expect("batch project ok");
        assert_eq!(z.len(), 4 * 64, "batch output size mismatch");
    }

    #[test]
    fn project_batch_each_row_unit_norm() {
        let head = make_head(32, 64, 22);
        let x = random_vec(8 * 32, 23); // batch=8
        let z = head.project_batch(&x, 8).expect("ok");
        for i in 0..8 {
            let row = &z[i * 64..(i + 1) * 64];
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "batch row {i} norm = {norm}, expected 1.0"
            );
        }
    }

    #[test]
    fn project_batch_matches_individual() {
        // project_batch(x_all, B) should match B calls to project(x_i).
        let head = make_head(16, 32, 24);
        let x_all = random_vec(3 * 16, 25);
        let z_batch = head.project_batch(&x_all, 3).expect("batch ok");
        for i in 0..3 {
            let xi = &x_all[i * 16..(i + 1) * 16];
            let z_single = head.project(xi).expect("single ok");
            let z_batch_row = &z_batch[i * 32..(i + 1) * 32];
            for (j, (&a, &b)) in z_single.iter().zip(z_batch_row.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "batch vs single at [{i},{j}]: {a} ≠ {b}"
                );
            }
        }
    }

    #[test]
    fn project_batch_error_wrong_total_length() {
        let head = make_head(32, 64, 26);
        let x = random_vec(3 * 32 + 5, 27); // not divisible by 32
        let r = head.project_batch(&x, 3);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    // ── ProjectionHead::cosine_sim ───────────────────────────────────────────

    #[test]
    fn cosine_sim_unit_vector_with_self_is_one() {
        let head = make_head(32, 32, 30);
        let x = random_vec(32, 31);
        let z = head.project(&x).expect("ok");
        let sim = ProjectionHead::cosine_sim(&z, &z).expect("cosine ok");
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "cosine(v, v) should be ≈1.0 for unit-norm v, got {sim}"
        );
    }

    #[test]
    fn cosine_sim_orthogonal_vectors() {
        // Manually construct two orthogonal unit vectors.
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let sim = ProjectionHead::cosine_sim(&a, &b).expect("ok");
        assert!(
            sim.abs() < 1e-6,
            "cosine similarity of orthogonal vectors should be ≈0, got {sim}"
        );
    }

    #[test]
    fn cosine_sim_opposite_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let sim = ProjectionHead::cosine_sim(&a, &b).expect("ok");
        assert!(
            (sim + 1.0).abs() < 1e-5,
            "cosine similarity of opposite vectors should be ≈-1, got {sim}"
        );
    }

    #[test]
    fn cosine_sim_range() {
        // Cosine similarity must lie in [-1, 1].
        let mut rng = LcgRng::new(40);
        for _ in 0..50 {
            let mut a = vec![0.0f32; 64];
            let mut b = vec![0.0f32; 64];
            rng.fill_normal(&mut a);
            rng.fill_normal(&mut b);
            let sim = ProjectionHead::cosine_sim(&a, &b).expect("ok");
            assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&sim),
                "cosine sim out of [-1,1]: {sim}"
            );
        }
    }

    #[test]
    fn cosine_sim_error_length_mismatch() {
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 8];
        let r = ProjectionHead::cosine_sim(&a, &b);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }
}
