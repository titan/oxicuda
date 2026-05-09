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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct WideDeep {
    pub wide_w: Vec<f32>,
    pub deep_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
}

impl WideDeep {
    pub fn new(
        input_dim: usize,
        deep_hidden_dims: &[usize],
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if input_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        let wide_w: Vec<f32> = (0..input_dim).map(|_| rng.next_normal() * 0.01).collect();

        let mut deep_layers = Vec::new();
        let mut in_dim = input_dim;
        for &out_dim in deep_hidden_dims {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_normal() * sc)
                .collect();
            let b = vec![0.0_f32; out_dim];
            deep_layers.push((w, b));
            in_dim = out_dim;
        }
        // Final scalar output
        {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * sc).collect();
            let b = vec![0.0_f32; 1];
            deep_layers.push((w, b));
        }

        Ok(Self {
            wide_w,
            deep_layers,
            input_dim,
        })
    }

    pub fn forward(&self, x: &[f32]) -> RecsysResult<f32> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        // Wide: linear dot product
        let wide_val: f32 = x
            .iter()
            .zip(self.wide_w.iter())
            .map(|(&xi, &wi)| xi * wi)
            .sum();

        // Deep: MLP with ReLU
        let mut deep_cur = x.to_vec();
        let mut cur_dim = self.input_dim;
        for (idx, (w, b)) in self.deep_layers.iter().enumerate() {
            let out_dim = b.len();
            let mut out = dense(&deep_cur, w, b, cur_dim, out_dim);
            if idx + 1 < self.deep_layers.len() {
                for v in &mut out {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            deep_cur = out;
            cur_dim = out_dim;
        }
        let deep_val = deep_cur.first().copied().unwrap_or(0.0);

        Ok(sigmoid(wide_val + deep_val))
    }
}
