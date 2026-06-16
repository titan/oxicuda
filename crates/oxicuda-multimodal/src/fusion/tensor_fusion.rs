//! Tensor Fusion Network outer-product fusion (Zadeh et al. 2017).
//!
//! Implements the multimodal tensor-fusion layer from:
//! Zadeh et al. "Tensor Fusion Network for Multimodal Sentiment Analysis."
//! EMNLP 2017.
//!
//! Each modality embedding is augmented with a constant `1` (the "bias" unit),
//! and the fused representation is the outer product of the augmented vectors.
//! For two modalities `a ∈ R^{d_a}` and `b ∈ R^{d_b}`:
//!
//! ```text
//! ã = [a ; 1] ∈ R^{d_a+1}      b̃ = [b ; 1] ∈ R^{d_b+1}
//! Z = ã ⊗ b̃  ∈ R^{(d_a+1)·(d_b+1)}
//! ```
//!
//! The appended `1`s make the fusion tensor contain, as sub-blocks, the original
//! unimodal features (`a·1`, `1·b`), the bimodal interactions (`a ⊗ b`), and the
//! constant bias term (`1·1`) — capturing both unimodal and cross-modal dynamics
//! in a single representation. An optional learned projection maps the (large)
//! fused tensor down to an output dimension.

use crate::error::{MmResult, MultiModalError};

/// Tensor-fusion operator with an optional output projection.
#[derive(Debug, Clone)]
pub struct TensorFusion {
    /// Modality-A dimension (before the appended bias unit).
    pub d_a: usize,
    /// Modality-B dimension (before the appended bias unit).
    pub d_b: usize,
    /// Output projection `W`: `[fused_dim × d_out]` row-major, where
    /// `fused_dim = (d_a + 1) · (d_b + 1)`.
    pub w_out: Vec<f32>,
    /// Output bias `[d_out]`.
    pub b_out: Vec<f32>,
    /// Projection output dimension.
    pub d_out: usize,
}

impl TensorFusion {
    /// Dimension of the raw fused tensor, `(d_a + 1)·(d_b + 1)`.
    #[must_use]
    pub fn fused_dim(d_a: usize, d_b: usize) -> usize {
        (d_a + 1) * (d_b + 1)
    }

    /// Create a tensor-fusion layer with a zero output projection.
    #[must_use]
    pub fn zeros(d_a: usize, d_b: usize, d_out: usize) -> Self {
        let fused = Self::fused_dim(d_a, d_b);
        Self {
            d_a,
            d_b,
            w_out: vec![0.0_f32; fused * d_out],
            b_out: vec![0.0_f32; d_out],
            d_out,
        }
    }

    /// Compute the raw fused outer-product tensor for a single `(a, b)` pair.
    ///
    /// Returns `[(d_a+1)·(d_b+1)]`, row-major with the augmented-`a` index as the
    /// outer (row) index and augmented-`b` as the inner (column) index.
    ///
    /// # Errors
    /// Returns [`MultiModalError::DimensionMismatch`] when an input length does
    /// not match its declared dimension.
    pub fn fuse_single(&self, a: &[f32], b: &[f32]) -> MmResult<Vec<f32>> {
        if a.len() != self.d_a {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_a,
                got: a.len(),
            });
        }
        if b.len() != self.d_b {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.d_b,
                got: b.len(),
            });
        }
        // Augmented vectors with appended bias unit.
        let mut a_aug = Vec::with_capacity(self.d_a + 1);
        a_aug.extend_from_slice(a);
        a_aug.push(1.0);
        let mut b_aug = Vec::with_capacity(self.d_b + 1);
        b_aug.extend_from_slice(b);
        b_aug.push(1.0);

        let cols = self.d_b + 1;
        let mut z = vec![0.0_f32; a_aug.len() * cols];
        for (i, &ai) in a_aug.iter().enumerate() {
            for (j, &bj) in b_aug.iter().enumerate() {
                z[i * cols + j] = ai * bj;
            }
        }
        Ok(z)
    }

    /// Fuse and project a single `(a, b)` pair → `[d_out]`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::fuse_single`].
    pub fn forward_single(&self, a: &[f32], b: &[f32]) -> MmResult<Vec<f32>> {
        let z = self.fuse_single(a, b)?;
        let fused = z.len();
        let mut out = self.b_out.clone();
        for (o, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for k in 0..fused {
                acc += z[k] * self.w_out[k * self.d_out + o];
            }
            *slot += acc;
        }
        Ok(out)
    }

    /// Batched fuse + project: `a [batch × d_a]`, `b [batch × d_b]` → `[batch × d_out]`.
    ///
    /// # Errors
    /// Returns [`MultiModalError`] on a batch/shape mismatch or empty batch.
    pub fn forward(&self, a: &[f32], b: &[f32], batch: usize) -> MmResult<Vec<f32>> {
        if a.len() != batch * self.d_a {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_a,
                got: a.len(),
            });
        }
        if b.len() != batch * self.d_b {
            return Err(MultiModalError::DimensionMismatch {
                expected: batch * self.d_b,
                got: b.len(),
            });
        }
        if batch == 0 {
            return Err(MultiModalError::InvalidBatchSize);
        }
        let mut out = Vec::with_capacity(batch * self.d_out);
        for bi in 0..batch {
            let a_i = &a[bi * self.d_a..(bi + 1) * self.d_a];
            let b_i = &b[bi * self.d_b..(bi + 1) * self.d_b];
            out.extend_from_slice(&self.forward_single(a_i, b_i)?);
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(MultiModalError::NanEncountered {
                location: "tensor_fusion",
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_dim_formula() {
        assert_eq!(TensorFusion::fused_dim(2, 3), 3 * 4);
        assert_eq!(TensorFusion::fused_dim(0, 0), 1);
    }

    #[test]
    fn fuse_single_shape() {
        let tf = TensorFusion::zeros(2, 3, 5);
        let a = vec![1.0_f32, 2.0];
        let b = vec![3.0_f32, 4.0, 5.0];
        let z = tf.fuse_single(&a, &b).expect("fuse_single should succeed");
        assert_eq!(z.len(), (2 + 1) * (3 + 1));
    }

    #[test]
    fn outer_product_values_correct() {
        // a=[2], b=[3] → a_aug=[2,1], b_aug=[3,1]
        // Z (row-major, cols=2): [2*3, 2*1, 1*3, 1*1] = [6,2,3,1]
        let tf = TensorFusion::zeros(1, 1, 1);
        let z = tf
            .fuse_single(&[2.0], &[3.0])
            .expect("fuse_single should succeed");
        assert_eq!(z, vec![6.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn bias_corner_is_one() {
        // The last element (1·1) is always 1.
        let tf = TensorFusion::zeros(3, 2, 1);
        let a = vec![5.0_f32, 6.0, 7.0];
        let b = vec![8.0_f32, 9.0];
        let z = tf.fuse_single(&a, &b).expect("fuse_single should succeed");
        assert!((z[z.len() - 1] - 1.0).abs() < 1e-9, "bias corner must be 1");
    }

    #[test]
    fn unimodal_subblocks_present() {
        // Row containing a_aug last entry (=1) times b gives the b features.
        // a=[a0], b=[b0]: a_aug=[a0,1], b_aug=[b0,1].
        // Z[1*2+0] = 1*b0 = b0 (unimodal b); Z[0*2+1] = a0*1 = a0 (unimodal a).
        let tf = TensorFusion::zeros(1, 1, 1);
        let z = tf
            .fuse_single(&[0.4], &[0.9])
            .expect("fuse_single should succeed");
        assert!((z[2] - 0.9).abs() < 1e-6, "unimodal b should appear");
        assert!((z[1] - 0.4).abs() < 1e-6, "unimodal a should appear");
    }

    #[test]
    fn forward_output_shape() {
        let tf = TensorFusion::zeros(2, 2, 7);
        let a = vec![0.5_f32; 4 * 2];
        let b = vec![0.3_f32; 4 * 2];
        let out = tf.forward(&a, &b, 4).expect("forward should succeed");
        assert_eq!(out.len(), 4 * 7);
    }

    #[test]
    fn zero_projection_gives_bias() {
        let mut tf = TensorFusion::zeros(2, 2, 3);
        tf.b_out = vec![1.5_f32; 3];
        let a = vec![9.0_f32, 9.0];
        let b = vec![9.0_f32, 9.0];
        let out = tf
            .forward_single(&a, &b)
            .expect("forward_single should succeed");
        for &v in &out {
            assert!((v - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn projection_reads_bias_corner() {
        // Set W so that only the bias corner (last fused index) feeds output 0
        // with weight 2 → output = 2 * (1·1) = 2.
        let d_a = 1;
        let d_b = 1;
        let fused = TensorFusion::fused_dim(d_a, d_b); // 4
        let mut tf = TensorFusion::zeros(d_a, d_b, 1);
        tf.w_out[fused - 1] = 2.0; // last fused row, output col 0
        let out = tf
            .forward_single(&[7.0], &[8.0])
            .expect("forward_single should succeed");
        assert!((out[0] - 2.0).abs() < 1e-6, "out={}", out[0]);
    }

    #[test]
    fn fuse_dim_mismatch_a_errors() {
        let tf = TensorFusion::zeros(2, 3, 4);
        let a = vec![0.0_f32; 1];
        let b = vec![0.0_f32; 3];
        assert!(matches!(
            tf.fuse_single(&a, &b),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fuse_dim_mismatch_b_errors() {
        let tf = TensorFusion::zeros(2, 3, 4);
        let a = vec![0.0_f32; 2];
        let b = vec![0.0_f32; 2];
        assert!(matches!(
            tf.fuse_single(&a, &b),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_zero_errors() {
        let tf = TensorFusion::zeros(2, 3, 4);
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(matches!(
            tf.forward(&a, &b, 0),
            Err(MultiModalError::InvalidBatchSize)
        ));
    }

    #[test]
    fn output_finite() {
        let mut tf = TensorFusion::zeros(3, 3, 4);
        for (i, v) in tf.w_out.iter_mut().enumerate() {
            *v = (i as f32 * 0.01).sin();
        }
        let a = vec![1.0_f32, -2.0, 3.0];
        let b = vec![-1.0_f32, 2.0, -3.0];
        let out = tf.forward(&a, &b, 1).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
