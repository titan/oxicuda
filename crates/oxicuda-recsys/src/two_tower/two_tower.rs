use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

fn dense(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
    (0..fan_out)
        .map(|o| {
            b[o] + w[o * fan_in..(o + 1) * fan_in]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

fn relu_vec(mut x: Vec<f32>) -> Vec<f32> {
    for v in &mut x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    x
}

pub struct TwoTower {
    pub user_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub item_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl TwoTower {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if input_dim == 0 || hidden_dim == 0 || output_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        let build_tower = |rng: &mut LcgRng| -> Vec<(Vec<f32>, Vec<f32>)> {
            let mut layers = Vec::with_capacity(n_layers);
            let mut in_dim = input_dim;
            for layer_idx in 0..n_layers {
                let out_dim = if layer_idx + 1 == n_layers {
                    output_dim
                } else {
                    hidden_dim
                };
                let sc = (2.0 / in_dim as f32).sqrt();
                let w: Vec<f32> = (0..out_dim * in_dim)
                    .map(|_| rng.next_normal() * sc)
                    .collect();
                let b = vec![0.0_f32; out_dim];
                layers.push((w, b));
                in_dim = out_dim;
            }
            layers
        };

        let user_layers = build_tower(rng);
        let item_layers = build_tower(rng);

        Ok(Self {
            user_layers,
            item_layers,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    pub fn encode_user(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        self.mlp_forward(x, &self.user_layers, self.input_dim)
    }

    pub fn encode_item(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        self.mlp_forward(x, &self.item_layers, self.input_dim)
    }

    fn mlp_forward(
        &self,
        x: &[f32],
        layers: &[(Vec<f32>, Vec<f32>)],
        input_dim: usize,
    ) -> RecsysResult<Vec<f32>> {
        let mut current = x.to_vec();
        let mut curr_dim = input_dim;
        for (idx, (w, b)) in layers.iter().enumerate() {
            let out_dim = b.len();
            let out = dense(&current, w, b, curr_dim, out_dim);
            current = if idx + 1 < layers.len() {
                relu_vec(out)
            } else {
                out
            };
            curr_dim = out_dim;
        }
        Ok(current)
    }

    pub fn score(&self, user_x: &[f32], item_x: &[f32]) -> RecsysResult<f32> {
        let u = self.encode_user(user_x)?;
        let i = self.encode_item(item_x)?;
        let dot = u.iter().zip(i.iter()).map(|(&a, &b)| a * b).sum();
        Ok(dot)
    }
}
