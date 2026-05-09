//! BYOL / SimSiam predictor head — an additional 2-layer MLP placed only on
//! the *online* branch so the loss is asymmetric.
//!
//! Architecturally identical to [`crate::head::projector::MlpProjector`] but
//! semantically separate: the predictor parameters are *only* updated through
//! the online stream's gradients (the target stream uses an EMA copy of the
//! projector — never a copy of the predictor).

use crate::error::{SslError, SslResult};
use crate::handle::LcgRng;

/// 2-layer MLP predictor head used by BYOL / SimSiam.
#[derive(Debug, Clone)]
pub struct PredictorHead {
    /// Input dim.
    pub in_dim: usize,
    /// Hidden dim.
    pub hidden_dim: usize,
    /// Output dim (same as projector output for both BYOL and SimSiam).
    pub out_dim: usize,
    /// First layer weights `[hidden × in]`.
    pub w1: Vec<f32>,
    /// First layer bias.
    pub b1: Vec<f32>,
    /// Second layer weights `[out × hidden]`.
    pub w2: Vec<f32>,
    /// Second layer bias.
    pub b2: Vec<f32>,
}

impl PredictorHead {
    /// New predictor with Kaiming-init weights.
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

    /// Forward pass on a single feature vector.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_shapes_correct() {
        let mut rng = LcgRng::new(0);
        let p = PredictorHead::new(4, 8, 4, &mut rng).unwrap();
        assert_eq!(p.w1.len(), 8 * 4);
        assert_eq!(p.b1.len(), 8);
        assert_eq!(p.w2.len(), 4 * 8);
        assert_eq!(p.b2.len(), 4);
    }

    #[test]
    fn predictor_forward_correct_dim() {
        let mut rng = LcgRng::new(0);
        let p = PredictorHead::new(4, 8, 4, &mut rng).unwrap();
        let x = vec![0.5_f32; 4];
        let y = p.forward(&x).unwrap();
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn predictor_rejects_zero_dim() {
        let mut rng = LcgRng::new(0);
        assert!(PredictorHead::new(0, 4, 4, &mut rng).is_err());
        assert!(PredictorHead::new(4, 0, 4, &mut rng).is_err());
        assert!(PredictorHead::new(4, 4, 0, &mut rng).is_err());
    }

    #[test]
    fn predictor_rejects_dim_mismatch() {
        let mut rng = LcgRng::new(0);
        let p = PredictorHead::new(4, 8, 4, &mut rng).unwrap();
        let r = p.forward(&[0.0_f32; 5]);
        assert!(r.is_err());
    }
}
