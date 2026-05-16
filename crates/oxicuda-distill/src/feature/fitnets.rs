//! FitNets (Romero et al. 2015) — hint-based intermediate feature distillation.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

/// A simple linear regressor that projects student features into the teacher feature space.
#[derive(Debug, Clone)]
pub struct FitNetsRegressor {
    /// Input (student) feature dimension.
    pub in_dim: usize,
    /// Output (teacher) feature dimension.
    pub out_dim: usize,
    /// Weight matrix W of shape `[out_dim × in_dim]`, stored row-major.
    pub w: Vec<f32>,
    /// Bias vector of shape `[out_dim]`.
    pub b: Vec<f32>,
}

impl FitNetsRegressor {
    /// Construct a new regressor with He-style initialisation.
    ///
    /// Weights are sampled from N(0, 1/√in_dim); biases are zeros.
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = if in_dim == 0 {
            1.0
        } else {
            1.0 / (in_dim as f32).sqrt()
        };
        let mut w = vec![0.0_f32; out_dim * in_dim];
        for wi in w.iter_mut() {
            *wi = rng.next_normal() * scale;
        }
        let b = vec![0.0_f32; out_dim];
        Self {
            in_dim,
            out_dim,
            w,
            b,
        }
    }

    /// Project student features: applies W (in_dim→out_dim) per token plus bias.
    ///
    /// `x` is expected as `[seq_len × in_dim]` flat row-major; output is `[seq_len × out_dim]`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> DistillResult<Vec<f32>> {
        if self.in_dim == 0 || self.out_dim == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "FitNetsRegressor: in_dim or out_dim is zero".into(),
            });
        }
        let expected = seq_len * self.in_dim;
        if x.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; seq_len * self.out_dim];
        for s in 0..seq_len {
            let x_row = &x[s * self.in_dim..(s + 1) * self.in_dim];
            for o in 0..self.out_dim {
                let w_row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
                let dot: f32 = x_row.iter().zip(w_row.iter()).map(|(&a, &b)| a * b).sum();
                out[s * self.out_dim + o] = dot + self.b[o];
            }
        }
        Ok(out)
    }
}

/// Mean squared error between equal-length slices.
#[must_use]
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f32;
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f32>()
        / n
}

/// Compute hint loss: project student features then measure MSE against teacher features.
pub fn hint_loss(
    regressor: &FitNetsRegressor,
    s_feat: &[f32],
    t_feat: &[f32],
) -> DistillResult<f32> {
    if s_feat.is_empty() || t_feat.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if regressor.in_dim == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "in_dim is zero".into(),
        });
    }
    let seq_len = s_feat.len() / regressor.in_dim;
    if seq_len == 0 || !s_feat.len().is_multiple_of(regressor.in_dim) {
        return Err(DistillError::DimensionMismatch {
            expected: seq_len * regressor.in_dim,
            got: s_feat.len(),
        });
    }
    let projected = regressor.forward(s_feat, seq_len)?;
    if projected.len() != t_feat.len() {
        return Err(DistillError::DimensionMismatch {
            expected: projected.len(),
            got: t_feat.len(),
        });
    }
    Ok(mse(&projected, t_feat))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape() {
        let mut rng = LcgRng::new(1);
        let reg = FitNetsRegressor::new(4, 8, &mut rng);
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect(); // seq_len=3, in_dim=4
        let out = reg.forward(&x, 3).unwrap();
        assert_eq!(out.len(), 3 * 8);
    }

    #[test]
    fn mse_identical_is_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(mse(&v, &v) < 1e-10);
    }
}
