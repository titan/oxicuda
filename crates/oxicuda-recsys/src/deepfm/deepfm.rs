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

pub struct DeepFm {
    pub field_dims: Vec<usize>,
    pub emb_dim: usize,
    /// Per-field embedding tables: embeddings[field][field_val * emb_dim .. (field_val+1)*emb_dim]
    pub embeddings: Vec<Vec<f32>>,
    pub linear_w: Vec<f32>,
    pub deep_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub deep_input_dim: usize,
}

impl DeepFm {
    pub fn new(
        field_dims: Vec<usize>,
        emb_dim: usize,
        deep_dims: &[usize],
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if field_dims.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        if emb_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: emb_dim });
        }
        let n_fields = field_dims.len();
        let scale = (1.0 / emb_dim as f32).sqrt();

        let embeddings: Vec<Vec<f32>> = field_dims
            .iter()
            .map(|&dim| {
                (0..dim * emb_dim)
                    .map(|_| rng.next_normal() * scale)
                    .collect()
            })
            .collect();

        let linear_w: Vec<f32> = field_dims
            .iter()
            .flat_map(|&dim| {
                (0..dim)
                    .map(|_| rng.next_normal() * 0.01)
                    .collect::<Vec<_>>()
            })
            .collect();

        let deep_input_dim = n_fields * emb_dim;
        let mut deep_layers = Vec::new();
        let mut in_dim = deep_input_dim;
        for &out_dim in deep_dims {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_normal() * sc)
                .collect();
            let b = vec![0.0_f32; out_dim];
            deep_layers.push((w, b));
            in_dim = out_dim;
        }
        // Final scalar layer
        {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * sc).collect();
            let b = vec![0.0_f32; 1];
            deep_layers.push((w, b));
        }

        Ok(Self {
            field_dims,
            emb_dim,
            embeddings,
            linear_w,
            deep_layers,
            deep_input_dim,
        })
    }

    pub fn forward(&self, field_ids: &[usize]) -> RecsysResult<f32> {
        if field_ids.len() != self.field_dims.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: self.field_dims.len(),
                got: field_ids.len(),
            });
        }
        let n_fields = self.field_dims.len();
        let d = self.emb_dim;

        // Validate field ids
        for (f, (&id, &dim)) in field_ids.iter().zip(self.field_dims.iter()).enumerate() {
            if id >= dim {
                return Err(RecsysError::Internal {
                    msg: format!("field {f}: id {id} >= dim {dim}"),
                });
            }
        }

        // Linear term: sum_i w[field_i][id_i]
        let mut linear_offset = 0usize;
        let linear_val: f32 = field_ids
            .iter()
            .zip(self.field_dims.iter())
            .map(|(&id, &dim)| {
                let v = self.linear_w[linear_offset + id];
                linear_offset += dim;
                v
            })
            .sum();

        // Gather field embeddings
        let embs: Vec<&[f32]> = field_ids
            .iter()
            .enumerate()
            .map(|(f, &id)| &self.embeddings[f][id * d..(id + 1) * d])
            .collect();

        // FM 2nd order: 0.5 * (||sum e_i||^2 - sum ||e_i||^2)
        let mut sum_emb = vec![0.0_f32; d];
        let mut sum_sq = 0.0_f32;
        for &e in &embs {
            for (k, &ek) in e.iter().enumerate() {
                sum_emb[k] += ek;
            }
            sum_sq += e.iter().map(|&v| v * v).sum::<f32>();
        }
        let sum_sq_emb: f32 = sum_emb.iter().map(|&v| v * v).sum();
        let fm_val = 0.5 * (sum_sq_emb - sum_sq);

        // Deep: MLP over concatenated embeddings
        let concat: Vec<f32> = (0..n_fields)
            .flat_map(|f| embs[f].iter().copied())
            .collect();
        let mut deep_cur = concat;
        let mut cur_dim = self.deep_input_dim;
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

        Ok(sigmoid(linear_val + fm_val + deep_val))
    }
}
