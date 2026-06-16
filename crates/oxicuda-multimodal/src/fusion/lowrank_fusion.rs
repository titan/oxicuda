//! Low-rank Multimodal Fusion — LMF (Liu et al. 2018).
//!
//! Implements the efficient low-rank tensor fusion from:
//! Liu et al. "Efficient Low-rank Multimodal Fusion with Modality-Specific
//! Factors." ACL 2018.
//!
//! Tensor Fusion ([`crate::fusion::tensor_fusion`]) forms the full outer product
//! of augmented modality vectors and projects it with a weight tensor `W`; that
//! tensor is exponentially large. LMF decomposes `W` into `rank` modality-specific
//! low-rank factors and exploits the fact that the projection of an outer product
//! factorises, avoiding ever materialising the big tensor:
//!
//! ```text
//! ã = [a ; 1]   b̃ = [b ; 1]
//! out = ( Σ_{r=1}^{R} (W_a^{(r)} · ã) ⊙ (W_b^{(r)} · b̃) )  +  bias
//! ```
//!
//! where `W_a^{(r)} ∈ R^{d_out × (d_a+1)}`, `W_b^{(r)} ∈ R^{d_out × (d_b+1)}`.
//! This computes the *same* bilinear interaction as full tensor fusion (when the
//! ranks match the tensor rank) at `O(R·d_out·(d_a+d_b))` cost instead of
//! `O(d_out·d_a·d_b)`.

use crate::error::{MmResult, MultiModalError};

/// Low-rank multimodal fusion operator.
#[derive(Debug, Clone)]
pub struct LowRankFusion {
    /// Modality-A factors: `rank` matrices each `[d_out × (d_a+1)]`,
    /// stored contiguously row-major as `[rank × d_out × (d_a+1)]`.
    pub factors_a: Vec<f32>,
    /// Modality-B factors: `[rank × d_out × (d_b+1)]`.
    pub factors_b: Vec<f32>,
    /// Output bias `[d_out]`.
    pub bias: Vec<f32>,
    /// Modality-A dimension (before the appended bias unit).
    pub d_a: usize,
    /// Modality-B dimension (before the appended bias unit).
    pub d_b: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Decomposition rank `R`.
    pub rank: usize,
}

impl LowRankFusion {
    /// Create an LMF layer with zero factors and bias.
    ///
    /// # Errors
    /// Returns [`MultiModalError`] when `rank == 0` or `d_out == 0`.
    pub fn zeros(d_a: usize, d_b: usize, d_out: usize, rank: usize) -> MmResult<Self> {
        if rank == 0 {
            return Err(MultiModalError::InvalidKFactor { k_factor: rank });
        }
        if d_out == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        let a_len = rank * d_out * (d_a + 1);
        let b_len = rank * d_out * (d_b + 1);
        Ok(Self {
            factors_a: vec![0.0_f32; a_len],
            factors_b: vec![0.0_f32; b_len],
            bias: vec![0.0_f32; d_out],
            d_a,
            d_b,
            d_out,
            rank,
        })
    }

    /// Fuse a single `(a, b)` pair → `[d_out]`.
    ///
    /// # Errors
    /// Returns [`MultiModalError::DimensionMismatch`] when an input length does
    /// not match its declared dimension.
    pub fn forward_single(&self, a: &[f32], b: &[f32]) -> MmResult<Vec<f32>> {
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
        let da1 = self.d_a + 1;
        let db1 = self.d_b + 1;

        // Augmented inputs with appended bias unit.
        let mut a_aug = Vec::with_capacity(da1);
        a_aug.extend_from_slice(a);
        a_aug.push(1.0);
        let mut b_aug = Vec::with_capacity(db1);
        b_aug.extend_from_slice(b);
        b_aug.push(1.0);

        let mut out = self.bias.clone();
        for r in 0..self.rank {
            let a_off = r * self.d_out * da1;
            let b_off = r * self.d_out * db1;
            // For each output channel o: (row_a · ã) * (row_b · b̃), summed over rank.
            for o in 0..self.d_out {
                let ra = &self.factors_a[a_off + o * da1..a_off + (o + 1) * da1];
                let rb = &self.factors_b[b_off + o * db1..b_off + (o + 1) * db1];
                let mut pa = 0.0_f32;
                for i in 0..da1 {
                    pa += ra[i] * a_aug[i];
                }
                let mut pb = 0.0_f32;
                for i in 0..db1 {
                    pb += rb[i] * b_aug[i];
                }
                out[o] += pa * pb;
            }
        }
        Ok(out)
    }

    /// Batched fuse: `a [batch × d_a]`, `b [batch × d_b]` → `[batch × d_out]`.
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
                location: "lowrank_fusion",
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_rank_zero_errors() {
        assert!(matches!(
            LowRankFusion::zeros(2, 3, 4, 0),
            Err(MultiModalError::InvalidKFactor { .. })
        ));
    }

    #[test]
    fn zeros_dout_zero_errors() {
        assert!(matches!(
            LowRankFusion::zeros(2, 3, 0, 4),
            Err(MultiModalError::InvalidFeatureDim)
        ));
    }

    #[test]
    fn factor_buffer_sizes() {
        let lmf = LowRankFusion::zeros(2, 3, 5, 4).expect("zeros should succeed");
        assert_eq!(lmf.factors_a.len(), 4 * 5 * (2 + 1));
        assert_eq!(lmf.factors_b.len(), 4 * 5 * (3 + 1));
        assert_eq!(lmf.bias.len(), 5);
    }

    #[test]
    fn output_shape() {
        let lmf = LowRankFusion::zeros(4, 6, 8, 3).expect("zeros should succeed");
        let a = vec![0.5_f32; 2 * 4];
        let b = vec![0.3_f32; 2 * 6];
        let out = lmf.forward(&a, &b, 2).expect("forward should succeed");
        assert_eq!(out.len(), 2 * 8);
    }

    #[test]
    fn zero_factors_give_bias() {
        let mut lmf = LowRankFusion::zeros(2, 2, 3, 2).expect("zeros should succeed");
        lmf.bias = vec![2.5_f32; 3];
        let a = vec![9.0_f32, 9.0];
        let b = vec![9.0_f32, 9.0];
        let out = lmf
            .forward_single(&a, &b)
            .expect("forward_single should succeed");
        for &v in &out {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn single_rank_bias_units_recover_constant() {
        // rank=1, set factor_a row o to pick the bias unit (last index) =1,
        // factor_b row o to pick the bias unit =1 → contribution 1*1 = 1.
        let d_a = 1;
        let d_b = 1;
        let d_out = 1;
        let mut lmf = LowRankFusion::zeros(d_a, d_b, d_out, 1).expect("zeros should succeed");
        let da1 = d_a + 1;
        let db1 = d_b + 1;
        // factors_a[rank0][o0][last] = 1
        lmf.factors_a[da1 - 1] = 1.0;
        lmf.factors_b[db1 - 1] = 1.0;
        let out = lmf
            .forward_single(&[5.0], &[7.0])
            .expect("forward_single should succeed");
        assert!((out[0] - 1.0).abs() < 1e-6, "out={}", out[0]);
    }

    #[test]
    fn matches_full_outer_product_rank1() {
        // With rank 1 and factor_a = full row picking a-component, factor_b
        // picking b-component, LMF computes a_i * b_j — the bimodal interaction.
        let d_a = 1;
        let d_b = 1;
        let d_out = 1;
        let mut lmf = LowRankFusion::zeros(d_a, d_b, d_out, 1).expect("zeros should succeed");
        let da1 = d_a + 1; // 2
        let db1 = d_b + 1; // 2
        // factor_a picks a[0] (index 0), factor_b picks b[0] (index 0)
        lmf.factors_a[0] = 1.0; // ã[0] = a0
        lmf.factors_b[0] = 1.0; // b̃[0] = b0
        let out = lmf
            .forward_single(&[3.0], &[4.0])
            .expect("forward_single should succeed");
        // pa = a0 = 3, pb = b0 = 4 → 12
        assert!((out[0] - 12.0).abs() < 1e-6, "out={}", out[0]);
        // Confirm augmented sizes used.
        assert_eq!(da1 * db1, 4);
    }

    #[test]
    fn rank_accumulates() {
        // Two ranks each contributing 1 (via bias units) → output 2.
        let mut lmf = LowRankFusion::zeros(1, 1, 1, 2).expect("zeros should succeed");
        let da1 = 2;
        let db1 = 2;
        // rank 0 bias-unit picks
        lmf.factors_a[da1 - 1] = 1.0;
        lmf.factors_b[db1 - 1] = 1.0;
        // rank 1 bias-unit picks (offset by one rank block)
        lmf.factors_a[da1 + (da1 - 1)] = 1.0;
        lmf.factors_b[db1 + (db1 - 1)] = 1.0;
        let out = lmf
            .forward_single(&[0.0], &[0.0])
            .expect("forward_single should succeed");
        assert!((out[0] - 2.0).abs() < 1e-6, "out={}", out[0]);
    }

    #[test]
    fn dim_mismatch_a_errors() {
        let lmf = LowRankFusion::zeros(4, 6, 8, 2).expect("zeros should succeed");
        let a = vec![0.0_f32; 3];
        let b = vec![0.0_f32; 6];
        assert!(matches!(
            lmf.forward_single(&a, &b),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dim_mismatch_b_errors() {
        let lmf = LowRankFusion::zeros(4, 6, 8, 2).expect("zeros should succeed");
        let a = vec![0.0_f32; 4];
        let b = vec![0.0_f32; 5];
        assert!(matches!(
            lmf.forward_single(&a, &b),
            Err(MultiModalError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn batch_zero_errors() {
        let lmf = LowRankFusion::zeros(4, 6, 8, 2).expect("zeros should succeed");
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(matches!(
            lmf.forward(&a, &b, 0),
            Err(MultiModalError::InvalidBatchSize)
        ));
    }

    #[test]
    fn output_finite() {
        let mut lmf = LowRankFusion::zeros(3, 3, 4, 2).expect("zeros should succeed");
        for (i, v) in lmf.factors_a.iter_mut().enumerate() {
            *v = (i as f32 * 0.07).sin() * 0.5;
        }
        for (i, v) in lmf.factors_b.iter_mut().enumerate() {
            *v = (i as f32 * 0.05).cos() * 0.5;
        }
        let a = vec![1.0_f32, -2.0, 3.0];
        let b = vec![-1.0_f32, 2.0, -3.0];
        let out = lmf.forward(&a, &b, 1).expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
