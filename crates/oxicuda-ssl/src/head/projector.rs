//! 2-layer MLP projection head used by SimCLR/BYOL/MoCo v2/v3.
//!
//! Architecture: `Linear(in → hidden) → ReLU → Linear(hidden → out)`.
//! Initialised with Kaiming uniform on the inner Linear, zero bias.

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

/// 2-layer MLP projection head.
#[derive(Debug, Clone)]
pub struct MlpProjector {
    /// Input dim.
    pub in_dim: usize,
    /// Hidden dim.
    pub hidden_dim: usize,
    /// Output (projection) dim.
    pub out_dim: usize,
    /// First layer weights `[hidden × in]` (row-major).
    pub w1: Vec<f32>,
    /// First layer bias `[hidden]`.
    pub b1: Vec<f32>,
    /// Second layer weights `[out × hidden]`.
    pub w2: Vec<f32>,
    /// Second layer bias `[out]`.
    pub b2: Vec<f32>,
}

impl MlpProjector {
    /// New projector with Kaiming-init weights.
    ///
    /// # Errors
    /// [`SslError::InvalidProjectorDim`] if any dim is zero.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        rng: &mut LcgRng,
    ) -> SslResult<Self> {
        if in_dim == 0 || hidden_dim == 0 || out_dim == 0 {
            return Err(SslError::InvalidProjectorDim);
        }
        let scale1 = (2.0_f32 / in_dim as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        rng.fill_normal(&mut w1);
        for v in w1.iter_mut() {
            *v *= scale1;
        }
        let scale2 = (2.0_f32 / hidden_dim as f32).sqrt();
        let mut w2 = vec![0.0_f32; out_dim * hidden_dim];
        rng.fill_normal(&mut w2);
        for v in w2.iter_mut() {
            *v *= scale2;
        }
        Ok(Self {
            in_dim,
            hidden_dim,
            out_dim,
            w1,
            b1: vec![0.0_f32; hidden_dim],
            w2,
            b2: vec![0.0_f32; out_dim],
        })
    }

    /// Forward pass on a single feature vector `[in_dim]` → `[out_dim]`.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] if `x.len() != self.in_dim`.
    pub fn forward(&self, x: &[f32]) -> SslResult<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(SslError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let mut h = vec![0.0_f32; self.hidden_dim];
        for ((hj, b), row) in h
            .iter_mut()
            .zip(self.b1.iter())
            .zip(self.w1.chunks(self.in_dim))
        {
            let mut acc = *b;
            for (w, &xi) in row.iter().zip(x.iter()) {
                acc += w * xi;
            }
            *hj = acc.max(0.0);
        }
        let mut out = vec![0.0_f32; self.out_dim];
        for ((oj, b), row) in out
            .iter_mut()
            .zip(self.b2.iter())
            .zip(self.w2.chunks(self.hidden_dim))
        {
            let mut acc = *b;
            for (w, &hi) in row.iter().zip(h.iter()) {
                acc += w * hi;
            }
            *oj = acc;
        }
        Ok(out)
    }

    /// Forward pass on a batch `[N × in_dim]` → `[N × out_dim]`.
    ///
    /// # Errors
    /// [`SslError::DimensionMismatch`] if `x.len() != n*in_dim`.
    pub fn forward_batch(&self, x: &[f32], n: usize) -> SslResult<Vec<f32>> {
        if x.len() != n * self.in_dim {
            return Err(SslError::DimensionMismatch {
                expected: n * self.in_dim,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(n * self.out_dim);
        for i in 0..n {
            let row = &x[i * self.in_dim..(i + 1) * self.in_dim];
            out.extend_from_slice(&self.forward(row)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projector_construction_correct_shapes() {
        let mut rng = LcgRng::new(0);
        let p = MlpProjector::new(8, 16, 4, &mut rng).expect("new should succeed");
        assert_eq!(p.in_dim, 8);
        assert_eq!(p.hidden_dim, 16);
        assert_eq!(p.out_dim, 4);
        assert_eq!(p.w1.len(), 16 * 8);
        assert_eq!(p.b1.len(), 16);
        assert_eq!(p.w2.len(), 4 * 16);
        assert_eq!(p.b2.len(), 4);
    }

    #[test]
    fn projector_rejects_zero_dim() {
        let mut rng = LcgRng::new(0);
        assert!(MlpProjector::new(0, 4, 4, &mut rng).is_err());
        assert!(MlpProjector::new(4, 0, 4, &mut rng).is_err());
        assert!(MlpProjector::new(4, 4, 0, &mut rng).is_err());
    }

    #[test]
    fn projector_forward_correct_shape() {
        let mut rng = LcgRng::new(0);
        let p = MlpProjector::new(8, 16, 4, &mut rng).expect("new should succeed");
        let x = vec![0.0_f32; 8];
        let y = p.forward(&x).expect("forward should succeed");
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn projector_zero_input_returns_zero_when_no_bias() {
        let mut rng = LcgRng::new(0);
        let p = MlpProjector::new(8, 16, 4, &mut rng).expect("new should succeed");
        let y = p.forward(&[0.0_f32; 8]).expect("forward should succeed");
        for &v in &y {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn projector_forward_rejects_dim_mismatch() {
        let mut rng = LcgRng::new(0);
        let p = MlpProjector::new(8, 16, 4, &mut rng).expect("new should succeed");
        let r = p.forward(&[0.0_f32; 4]);
        assert!(r.is_err());
    }

    #[test]
    fn projector_forward_batch_correct_shape() {
        let mut rng = LcgRng::new(0);
        let p = MlpProjector::new(8, 16, 4, &mut rng).expect("new should succeed");
        let x = vec![0.1_f32; 4 * 8];
        let y = p
            .forward_batch(&x, 4)
            .expect("forward_batch should succeed");
        assert_eq!(y.len(), 4 * 4);
    }

    #[test]
    fn projector_forward_batch_rejects_dim_mismatch() {
        let mut rng = LcgRng::new(0);
        let p = MlpProjector::new(8, 16, 4, &mut rng).expect("new should succeed");
        let r = p.forward_batch(&[0.0_f32; 16], 4);
        assert!(r.is_err());
    }
}
