//! Concatenation + linear projection fusion.
//!
//! Concatenates two modality embeddings along the feature dimension and applies
//! a learned linear projection to the combined representation.

use crate::error::{MmResult, MultiModalError};

/// Concatenation fusion with linear projection.
///
/// For modalities A (`[d_a]`) and B (`[d_b]`):
/// `fused = W · [a; b] + bias`  where `W` is `[d_out × (d_a + d_b)]`.
#[derive(Debug, Clone)]
pub struct ConcatFusion {
    /// Weight matrix: `[(d_a + d_b) × d_out]` row-major.
    pub weight: Vec<f32>,
    /// Bias: `[d_out]`.
    pub bias: Vec<f32>,
    pub d_a: usize,
    pub d_b: usize,
    pub d_out: usize,
}

impl ConcatFusion {
    /// Create with zero weights.
    #[must_use]
    pub fn zeros(d_a: usize, d_b: usize, d_out: usize) -> Self {
        let d_in = d_a + d_b;
        Self {
            weight: vec![0.0_f32; d_in * d_out],
            bias: vec![0.0_f32; d_out],
            d_a,
            d_b,
            d_out,
        }
    }

    /// Fuse two batched feature matrices.
    ///
    /// - `a`: `[batch × d_a]`
    /// - `b`: `[batch × d_b]`
    ///
    /// Returns `[batch × d_out]`.
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

        let d_in = self.d_a + self.d_b;
        let mut out = vec![0.0_f32; batch * self.d_out];
        for bi in 0..batch {
            // Build concatenated input
            let mut concat = Vec::with_capacity(d_in);
            concat.extend_from_slice(&a[bi * self.d_a..(bi + 1) * self.d_a]);
            concat.extend_from_slice(&b[bi * self.d_b..(bi + 1) * self.d_b]);

            for o in 0..self.d_out {
                let mut acc = self.bias[o];
                for i in 0..d_in {
                    acc += concat[i] * self.weight[i * self.d_out + o];
                }
                out[bi * self.d_out + o] = acc;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concat_fusion_output_shape() {
        let f = ConcatFusion::zeros(4, 8, 16);
        let a = vec![0.5_f32; 3 * 4];
        let b = vec![0.3_f32; 3 * 8];
        let out = f.forward(&a, &b, 3).unwrap();
        assert_eq!(out.len(), 3 * 16);
    }

    #[test]
    fn concat_fusion_zero_weight_gives_bias() {
        let mut f = ConcatFusion::zeros(4, 4, 4);
        f.bias = vec![2.5_f32; 4];
        let a = vec![1.0_f32; 2 * 4];
        let b = vec![1.0_f32; 2 * 4];
        let out = f.forward(&a, &b, 2).unwrap();
        for &v in &out {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn concat_fusion_dimension_error() {
        let f = ConcatFusion::zeros(4, 8, 16);
        let a = vec![0.0_f32; 3 * 5]; // wrong d_a
        let b = vec![0.0_f32; 3 * 8];
        let err = f.forward(&a, &b, 3).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn concat_fusion_batch_error() {
        let f = ConcatFusion::zeros(4, 8, 16);
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let err = f.forward(&a, &b, 0).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidBatchSize));
    }

    #[test]
    fn concat_fusion_linear_check() {
        // W is identity-like over d_in=4 → d_out=4 but concat has 2+2=4 input
        let mut f = ConcatFusion::zeros(2, 2, 4);
        // Set identity: weight[i, o] = 1 if i == o
        for i in 0..4 {
            f.weight[i * 4 + i] = 1.0;
        }
        let a = vec![1.0_f32, 2.0]; // batch=1
        let b = vec![3.0_f32, 4.0];
        let out = f.forward(&a, &b, 1).unwrap();
        // expect [1, 2, 3, 4]
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }
}
