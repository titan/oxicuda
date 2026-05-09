use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

pub struct LinearHead {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub n_classes: usize,
    pub feat_dim: usize,
}

impl LinearHead {
    pub fn new(feat_dim: usize, n_classes: usize, rng: &mut LcgRng) -> Self {
        let limit = (6.0_f32 / (feat_dim + n_classes) as f32).sqrt();
        let mut weights = vec![0.0_f32; n_classes * feat_dim];
        for v in weights.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Self {
            weights,
            biases: vec![0.0_f32; n_classes],
            n_classes,
            feat_dim,
        }
    }

    pub fn forward(&self, feat: &[f32]) -> MetaResult<Vec<f32>> {
        if feat.len() != self.feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.feat_dim,
                got: feat.len(),
            });
        }
        let logits: Vec<f32> = (0..self.n_classes)
            .map(|c| {
                let row = &self.weights[c * self.feat_dim..(c + 1) * self.feat_dim];
                row.iter()
                    .zip(feat.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + self.biases[c]
            })
            .collect();
        Ok(logits)
    }

    pub fn param_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    pub fn to_params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_count());
        out.extend_from_slice(&self.weights);
        out.extend_from_slice(&self.biases);
        out
    }

    pub fn from_params(&mut self, params: &[f32]) -> MetaResult<()> {
        if params.len() != self.param_count() {
            return Err(MetaError::DimensionMismatch {
                expected: self.param_count(),
                got: params.len(),
            });
        }
        let wlen = self.weights.len();
        self.weights.copy_from_slice(&params[..wlen]);
        self.biases.copy_from_slice(&params[wlen..]);
        Ok(())
    }
}
