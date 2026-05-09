use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

fn layer_norm(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .zip(g.iter().zip(b.iter()))
        .map(|(&xi, (&gi, &bi))| (xi - mean) * inv_std * gi + bi)
        .collect()
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

pub struct SasLayer {
    /// Multi-head self-attention weights: \[d x d\] for Q, K, V, O
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    /// FFN: w1 \[4d x d\], b1 \[4d\], w2 \[d x 4d\], b2 \[d\]
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub ln1_g: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
}

impl SasLayer {
    pub fn new(emb_dim: usize, rng: &mut LcgRng) -> Self {
        let sc = (1.0 / emb_dim as f32).sqrt();
        let ffn_dim = 4 * emb_dim;
        let ffn_sc = (2.0 / emb_dim as f32).sqrt();
        Self {
            wq: (0..emb_dim * emb_dim)
                .map(|_| rng.next_normal() * sc)
                .collect(),
            wk: (0..emb_dim * emb_dim)
                .map(|_| rng.next_normal() * sc)
                .collect(),
            wv: (0..emb_dim * emb_dim)
                .map(|_| rng.next_normal() * sc)
                .collect(),
            wo: (0..emb_dim * emb_dim)
                .map(|_| rng.next_normal() * sc)
                .collect(),
            w1: (0..ffn_dim * emb_dim)
                .map(|_| rng.next_normal() * ffn_sc)
                .collect(),
            b1: vec![0.0_f32; ffn_dim],
            w2: (0..emb_dim * ffn_dim)
                .map(|_| rng.next_normal() * ffn_sc)
                .collect(),
            b2: vec![0.0_f32; emb_dim],
            ln1_g: vec![1.0_f32; emb_dim],
            ln1_b: vec![0.0_f32; emb_dim],
            ln2_g: vec![1.0_f32; emb_dim],
            ln2_b: vec![0.0_f32; emb_dim],
        }
    }
}

pub struct SasRec {
    pub n_items: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub item_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub attn_layers: Vec<SasLayer>,
}

impl SasRec {
    pub fn new(
        n_items: usize,
        emb_dim: usize,
        n_heads: usize,
        n_layers: usize,
        max_seq_len: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if emb_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: emb_dim });
        }
        let sc = (1.0 / emb_dim as f32).sqrt();
        let item_emb: Vec<f32> = (0..n_items * emb_dim)
            .map(|_| rng.next_normal() * sc)
            .collect();
        let pos_emb: Vec<f32> = (0..max_seq_len * emb_dim)
            .map(|_| rng.next_normal() * sc)
            .collect();
        let attn_layers: Vec<SasLayer> =
            (0..n_layers).map(|_| SasLayer::new(emb_dim, rng)).collect();

        Ok(Self {
            n_items,
            emb_dim,
            n_heads,
            n_layers,
            item_emb,
            pos_emb,
            attn_layers,
        })
    }

    pub fn forward(&self, item_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        if item_ids.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        for &id in item_ids {
            if id >= self.n_items {
                return Err(RecsysError::UnknownItem { id });
            }
        }

        let seq_len = item_ids.len();
        let d = self.emb_dim;

        // Embed items + positional embeddings
        let mut h: Vec<f32> = item_ids
            .iter()
            .enumerate()
            .flat_map(|(pos, &id)| {
                let item_e = &self.item_emb[id * d..(id + 1) * d];
                let pos_e_start = pos.min(self.pos_emb.len() / d - 1) * d;
                let pos_e = &self.pos_emb[pos_e_start..pos_e_start + d];
                item_e
                    .iter()
                    .zip(pos_e.iter())
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>()
            })
            .collect();

        // Apply transformer layers
        for layer in &self.attn_layers {
            h = self.apply_layer(&h, layer, seq_len)?;
        }

        // Last position output as query against all item embeddings
        let last = &h[(seq_len - 1) * d..seq_len * d];
        let logits: Vec<f32> = (0..self.n_items)
            .map(|item| {
                self.item_emb[item * d..(item + 1) * d]
                    .iter()
                    .zip(last.iter())
                    .map(|(&e, &q)| e * q)
                    .sum()
            })
            .collect();

        Ok(logits)
    }

    fn apply_layer(&self, h: &[f32], layer: &SasLayer, seq_len: usize) -> RecsysResult<Vec<f32>> {
        let d = self.emb_dim;
        let scale = 1.0 / (d as f32).sqrt();

        // Multi-head causal self-attention (single-head for simplicity when n_heads=1)
        let q = matmul_rows(h, &layer.wq, seq_len, d, d);
        let k = matmul_rows(h, &layer.wk, seq_len, d, d);
        let v = matmul_rows(h, &layer.wv, seq_len, d, d);

        let mut attn_out = vec![0.0_f32; seq_len * d];
        for i in 0..seq_len {
            let mut scores: Vec<f32> = (0..=i)
                .map(|j| {
                    q[i * d..(i + 1) * d]
                        .iter()
                        .zip(k[j * d..(j + 1) * d].iter())
                        .map(|(&qi, &kj)| qi * kj)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            softmax_inplace(&mut scores);

            for (j, &a) in scores.iter().enumerate() {
                for (k_idx, &vk) in v[j * d..(j + 1) * d].iter().enumerate() {
                    attn_out[i * d + k_idx] += a * vk;
                }
            }
        }

        // Project with Wo
        let proj = matmul_rows(&attn_out, &layer.wo, seq_len, d, d);

        // Residual + LayerNorm 1
        let mut h_after_attn = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            let residual: Vec<f32> = h[pos * d..(pos + 1) * d]
                .iter()
                .zip(proj[pos * d..(pos + 1) * d].iter())
                .map(|(&hv, &pv)| hv + pv)
                .collect();
            let normed = layer_norm(&residual, &layer.ln1_g, &layer.ln1_b);
            h_after_attn[pos * d..(pos + 1) * d].copy_from_slice(&normed);
        }

        // FFN: two-layer with GELU-like activation (using tanh approx)
        let ffn_dim = 4 * d;
        let mut h_after_ffn = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            let x = &h_after_attn[pos * d..(pos + 1) * d];
            // First linear: [ffn_dim x d]
            let mut mid: Vec<f32> = (0..ffn_dim)
                .map(|o| {
                    layer.b1[o]
                        + layer.w1[o * d..(o + 1) * d]
                            .iter()
                            .zip(x.iter())
                            .map(|(&w, &xi)| w * xi)
                            .sum::<f32>()
                })
                .collect();
            // ReLU
            for v in &mut mid {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            // Second linear: [d x ffn_dim]
            let out: Vec<f32> = (0..d)
                .map(|o| {
                    layer.b2[o]
                        + layer.w2[o * ffn_dim..(o + 1) * ffn_dim]
                            .iter()
                            .zip(mid.iter())
                            .map(|(&w, &mi)| w * mi)
                            .sum::<f32>()
                })
                .collect();
            let residual2: Vec<f32> = x.iter().zip(out.iter()).map(|(&hv, &ov)| hv + ov).collect();
            let normed2 = layer_norm(&residual2, &layer.ln2_g, &layer.ln2_b);
            h_after_ffn[pos * d..(pos + 1) * d].copy_from_slice(&normed2);
        }

        Ok(h_after_ffn)
    }
}

/// Multiply each row of X [n x d_in] by W^T where W is [d_out x d_in] -> [n x d_out]
fn matmul_rows(x: &[f32], w: &[f32], n: usize, d_in: usize, d_out: usize) -> Vec<f32> {
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
