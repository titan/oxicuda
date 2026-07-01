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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(seed: u64) -> SasRec {
        let mut rng = LcgRng::new(seed);
        SasRec::new(10, 8, 2, 2, 16, &mut rng).expect("construction ok")
    }

    #[test]
    fn rejects_invalid_construction() {
        let mut rng = LcgRng::new(1);
        assert!(
            SasRec::new(0, 8, 1, 1, 8, &mut rng).is_err(),
            "n_items=0 must fail"
        );
        assert!(
            SasRec::new(10, 0, 1, 1, 8, &mut rng).is_err(),
            "emb_dim=0 must fail"
        );
    }

    #[test]
    fn output_shape_and_finiteness() {
        let model = make_model(2);
        let logits = model.forward(&[0, 1, 2]).expect("forward ok");
        assert_eq!(
            logits.len(),
            10,
            "forward must return exactly n_items=10 logits"
        );
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}]={v} must be finite");
        }
    }

    #[test]
    fn forward_rejects_empty_and_oob() {
        let model = make_model(3);
        assert!(model.forward(&[]).is_err(), "empty input must fail");
        assert!(
            model.forward(&[10]).is_err(),
            "id=10 >= n_items=10 must fail"
        );
        // Boundary: id=9 is valid (n_items=10 so 0..9 inclusive).
        assert!(model.forward(&[9]).is_ok(), "id=9 must succeed");
    }

    #[test]
    fn determinism() {
        // Two models from the same seed must produce bit-identical outputs.
        let model_a = make_model(5);
        let model_b = make_model(5);
        let seq = [0usize, 1, 2, 3, 4];
        let out_a = model_a.forward(&seq).expect("fwd a");
        let out_b = model_b.forward(&seq).expect("fwd b");
        for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
            assert_eq!(
                a, b,
                "logit[{i}] must be bit-identical across same-seed models"
            );
        }
    }

    #[test]
    fn single_item_sequence() {
        // A length-1 sequence is valid: position 0 attends only to itself.
        let model = make_model(6);
        let logits = model.forward(&[0]).expect("single-item sequence ok");
        assert_eq!(logits.len(), 10, "n_items logits expected");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite for a single-item sequence"
        );
    }

    #[test]
    fn order_sensitivity_permutation() {
        // SASRec has positional encoding and causal attention; permuting items changes
        // the final logits because the sequential order is baked into the representation.
        let model = make_model(7);
        let out_ab = model.forward(&[0, 1]).expect("fwd [0,1]");
        let out_ba = model.forward(&[1, 0]).expect("fwd [1,0]");
        let differs = out_ab
            .iter()
            .zip(out_ba.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "permuting sequence items must change logits (order-sensitive causal model)"
        );
    }

    #[test]
    fn causal_prefix_change_propagates_to_final_output() {
        // The causal mask lets position L-1 attend to ALL earlier positions 0..=L-1.
        // Changing item at position 0 (prefix) in a length-3 sequence must therefore
        // change the final logit (which comes from position 2 attending back to position 0).
        let mut rng = LcgRng::new(100);
        let mut model = SasRec::new(10, 8, 1, 1, 16, &mut rng).expect("construction ok");
        // Zero positional embeddings so only item identity contributes to attention
        // key/query vectors, isolating the causal propagation effect.
        for v in &mut model.pos_emb {
            *v = 0.0;
        }
        let out_012 = model.forward(&[0, 1, 2]).expect("fwd [0,1,2]");
        // Change only position 0 (item 0 → item 3); positions 1 and 2 are unchanged.
        let out_312 = model.forward(&[3, 1, 2]).expect("fwd [3,1,2]");
        let differs = out_012
            .iter()
            .zip(out_312.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "changing prefix item must propagate through causal attention chain to final output"
        );
    }
}
