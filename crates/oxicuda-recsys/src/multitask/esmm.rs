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

fn mlp_forward(x: &[f32], layers: &[(Vec<f32>, Vec<f32>)], input_dim: usize) -> f32 {
    let mut cur = x.to_vec();
    let mut cur_dim = input_dim;
    for (idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = b.len();
        let mut out = dense(&cur, w, b, cur_dim, out_dim);
        if idx + 1 < layers.len() {
            for v in &mut out {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        cur = out;
        cur_dim = out_dim;
    }
    cur.first().copied().unwrap_or(0.0)
}

/// Entire Space Multi-Task Model (ESMM).
///
/// Models pCTR and pCVR jointly; pCTCVR = pCTR * pCVR to address selection bias.
pub struct Esmm {
    pub ctr_tower: Vec<(Vec<f32>, Vec<f32>)>,
    pub cvr_tower: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
}

impl Esmm {
    pub fn new(input_dim: usize, hidden_dims: &[usize], rng: &mut LcgRng) -> RecsysResult<Self> {
        if input_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        let build_tower = |rng: &mut LcgRng| -> Vec<(Vec<f32>, Vec<f32>)> {
            let mut layers = Vec::new();
            let mut in_dim = input_dim;
            for &out_dim in hidden_dims {
                let sc = (2.0 / in_dim as f32).sqrt();
                let w: Vec<f32> = (0..out_dim * in_dim)
                    .map(|_| rng.next_normal() * sc)
                    .collect();
                layers.push((w, vec![0.0_f32; out_dim]));
                in_dim = out_dim;
            }
            // Final scalar
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * sc).collect();
            layers.push((w, vec![0.0_f32; 1]));
            layers
        };

        let ctr_tower = build_tower(rng);
        let cvr_tower = build_tower(rng);

        Ok(Self {
            ctr_tower,
            cvr_tower,
            input_dim,
        })
    }

    /// Returns (pCTR, pCVR, pCTCVR).
    pub fn forward(&self, x: &[f32]) -> RecsysResult<(f32, f32, f32)> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let p_ctr = sigmoid(mlp_forward(x, &self.ctr_tower, self.input_dim));
        let p_cvr = sigmoid(mlp_forward(x, &self.cvr_tower, self.input_dim));
        let p_ctcvr = p_ctr * p_cvr;
        Ok((p_ctr, p_cvr, p_ctcvr))
    }
}
