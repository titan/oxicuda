use crate::error::{RlhfError, RlhfResult};
use crate::handle::LcgRng;

pub struct RewardModel {
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    dims: Vec<usize>,
}

impl RewardModel {
    pub fn new(dims: &[usize], rng: &mut LcgRng) -> RlhfResult<Self> {
        if dims.len() < 2 {
            return Err(RlhfError::Internal {
                msg: "dims must have at least 2 entries (input and output)".into(),
            });
        }
        let layers = dims
            .windows(2)
            .map(|pair| {
                let (in_d, out_d) = (pair[0], pair[1]);
                let scale = (2.0_f32 / in_d as f32).sqrt();
                let weights = (0..in_d * out_d)
                    .map(|_| {
                        let (a, _) = rng.next_normal_pair();
                        a * scale
                    })
                    .collect::<Vec<f32>>();
                let bias = vec![0.0_f32; out_d];
                (weights, bias)
            })
            .collect();
        Ok(Self {
            layers,
            dims: dims.to_vec(),
        })
    }

    pub fn forward(&self, x: &[f32]) -> RlhfResult<f32> {
        if x.len() != self.dims[0] {
            return Err(RlhfError::DimensionMismatch {
                expected: self.dims[0],
                got: x.len(),
            });
        }
        let mut current = x.to_vec();
        for (layer_idx, (weights, bias)) in self.layers.iter().enumerate() {
            let in_d = self.dims[layer_idx];
            let out_d = self.dims[layer_idx + 1];
            let mut next = vec![0.0_f32; out_d];
            for (j, (b, n)) in bias.iter().zip(next.iter_mut()).enumerate() {
                let row_start = j * in_d;
                let dot: f32 = weights[row_start..row_start + in_d]
                    .iter()
                    .zip(current.iter())
                    .map(|(&w, &c)| w * c)
                    .sum();
                *n = dot + b;
                if layer_idx + 1 < self.layers.len() {
                    *n = n.max(0.0);
                }
            }
            current = next;
        }
        let out = current[0];
        if out.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(out)
    }
}
