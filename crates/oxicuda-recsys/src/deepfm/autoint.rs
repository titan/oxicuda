use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / (sum + 1e-10);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// AutoInt: interaction modelling with self-attention.
/// Each attention layer stores (Wq, Wk, Wv): [emb_dim x emb_dim] each.
pub struct AutoInt {
    pub field_dims: Vec<usize>,
    pub emb_dim: usize,
    pub embeddings: Vec<Vec<f32>>,
    /// (Wq, Wk, Wv) per attention layer
    pub attn_layers: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub output_w: Vec<f32>,
    pub output_b: f32,
}

impl AutoInt {
    pub fn new(
        field_dims: Vec<usize>,
        emb_dim: usize,
        n_attn_layers: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if field_dims.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        if emb_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: emb_dim });
        }
        let scale = (1.0 / emb_dim as f32).sqrt();
        let embeddings: Vec<Vec<f32>> = field_dims
            .iter()
            .map(|&dim| {
                (0..dim * emb_dim)
                    .map(|_| rng.next_normal() * scale)
                    .collect()
            })
            .collect();

        let attn_sc = (1.0 / (emb_dim * emb_dim) as f32).sqrt();
        let attn_layers: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..n_attn_layers)
            .map(|_| {
                let wq: Vec<f32> = (0..emb_dim * emb_dim)
                    .map(|_| rng.next_normal() * attn_sc)
                    .collect();
                let wk: Vec<f32> = (0..emb_dim * emb_dim)
                    .map(|_| rng.next_normal() * attn_sc)
                    .collect();
                let wv: Vec<f32> = (0..emb_dim * emb_dim)
                    .map(|_| rng.next_normal() * attn_sc)
                    .collect();
                (wq, wk, wv)
            })
            .collect();

        let n_fields = field_dims.len();
        let out_sc = (1.0 / (n_fields * emb_dim) as f32).sqrt();
        let output_w: Vec<f32> = (0..n_fields * emb_dim)
            .map(|_| rng.next_normal() * out_sc)
            .collect();

        Ok(Self {
            field_dims,
            emb_dim,
            embeddings,
            attn_layers,
            output_w,
            output_b: 0.0,
        })
    }

    pub fn forward(&self, field_ids: &[usize]) -> RecsysResult<f32> {
        if field_ids.len() != self.field_dims.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: self.field_dims.len(),
                got: field_ids.len(),
            });
        }
        for (f, (&id, &dim)) in field_ids.iter().zip(self.field_dims.iter()).enumerate() {
            if id >= dim {
                return Err(RecsysError::Internal {
                    msg: format!("field {f}: id {id} >= dim {dim}"),
                });
            }
        }

        let n_fields = self.field_dims.len();
        let d = self.emb_dim;

        // Stack field embeddings: [n_fields x d]
        let mut x: Vec<f32> = field_ids
            .iter()
            .enumerate()
            .flat_map(|(f, &id)| self.embeddings[f][id * d..(id + 1) * d].iter().copied())
            .collect();

        let scale = 1.0 / (d as f32).sqrt();

        // Apply self-attention layers
        for (wq, wk, wv) in &self.attn_layers {
            // Q, K, V: [n_fields x d] each
            let q = matvec_batch(&x, wq, n_fields, d, d);
            let k = matvec_batch(&x, wk, n_fields, d, d);
            let v = matvec_batch(&x, wv, n_fields, d, d);

            // Attention scores: [n_fields x n_fields]
            let mut attn_scores = vec![0.0_f32; n_fields * n_fields];
            for i in 0..n_fields {
                for j in 0..n_fields {
                    attn_scores[i * n_fields + j] = q[i * d..(i + 1) * d]
                        .iter()
                        .zip(k[j * d..(j + 1) * d].iter())
                        .map(|(&qi, &kj)| qi * kj)
                        .sum::<f32>()
                        * scale;
                }
                softmax_inplace(&mut attn_scores[i * n_fields..(i + 1) * n_fields]);
            }

            // Output: [n_fields x d] = attn * V
            let mut out = vec![0.0_f32; n_fields * d];
            for i in 0..n_fields {
                for j in 0..n_fields {
                    let a = attn_scores[i * n_fields + j];
                    for (k_idx, &vk) in v[j * d..(j + 1) * d].iter().enumerate() {
                        out[i * d + k_idx] += a * vk;
                    }
                }
            }
            // Residual connection
            for (xv, ov) in x.iter_mut().zip(out.iter()) {
                *xv = (*xv + *ov).max(0.0);
            }
        }

        // Mean-pool across fields
        let mut pooled = vec![0.0_f32; n_fields * d];
        pooled.copy_from_slice(&x);

        // Output: dot with output_w + b
        let logit = self.output_b
            + pooled
                .iter()
                .zip(self.output_w.iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum::<f32>();

        Ok(sigmoid(logit))
    }
}

/// Apply weight matrix W [d_out x d_in] to each row of X [n x d_in] -> [n x d_out]
fn matvec_batch(x: &[f32], w: &[f32], n: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * d_out];
    for row in 0..n {
        for col in 0..d_out {
            out[row * d_out + col] = w[col * d_in..(col + 1) * d_in]
                .iter()
                .zip(x[row * d_in..(row + 1) * d_in].iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum();
        }
    }
    out
}
