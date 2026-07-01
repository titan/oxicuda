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

pub struct BertLayer {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub ln1_g: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
}

impl BertLayer {
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

pub struct Bert4Rec {
    pub n_items: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub item_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    /// Special \[MASK\] token embedding
    pub mask_emb: Vec<f32>,
    pub attn_layers: Vec<BertLayer>,
}

/// Token id used as mask sentinel (n_items means mask).
const MASK_TOKEN: usize = usize::MAX;

impl Bert4Rec {
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
        let mask_emb: Vec<f32> = (0..emb_dim).map(|_| rng.next_normal() * sc).collect();
        let attn_layers: Vec<BertLayer> = (0..n_layers)
            .map(|_| BertLayer::new(emb_dim, rng))
            .collect();

        Ok(Self {
            n_items,
            emb_dim,
            n_heads,
            n_layers,
            item_emb,
            pos_emb,
            mask_emb,
            attn_layers,
        })
    }

    pub fn mask_sequence(
        &self,
        item_ids: &[usize],
        mask_ratio: f32,
        rng: &mut LcgRng,
    ) -> Vec<usize> {
        item_ids
            .iter()
            .map(|&id| {
                if rng.next_f32() < mask_ratio {
                    MASK_TOKEN
                } else {
                    id
                }
            })
            .collect()
    }

    fn embed_sequence(&self, masked_ids: &[usize]) -> Vec<f32> {
        let d = self.emb_dim;
        let seq_len = masked_ids.len();
        let mut h = vec![0.0_f32; seq_len * d];
        for (pos, &id) in masked_ids.iter().enumerate() {
            let item_e: &[f32] = if id == MASK_TOKEN {
                &self.mask_emb
            } else if id < self.n_items {
                &self.item_emb[id * d..(id + 1) * d]
            } else {
                &self.mask_emb
            };
            let pos_start = pos.min(self.pos_emb.len() / d - 1) * d;
            let pos_e = &self.pos_emb[pos_start..pos_start + d];
            for (k, (&ie, &pe)) in item_e.iter().zip(pos_e.iter()).enumerate() {
                h[pos * d + k] = ie + pe;
            }
        }
        h
    }

    fn apply_layer(&self, h: &[f32], layer: &BertLayer, seq_len: usize) -> Vec<f32> {
        let d = self.emb_dim;
        let scale = 1.0 / (d as f32).sqrt();

        let q = matmul_rows(h, &layer.wq, seq_len, d, d);
        let k = matmul_rows(h, &layer.wk, seq_len, d, d);
        let v = matmul_rows(h, &layer.wv, seq_len, d, d);

        // Bidirectional attention (no causal mask)
        let mut attn_out = vec![0.0_f32; seq_len * d];
        for i in 0..seq_len {
            let mut scores: Vec<f32> = (0..seq_len)
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

        let proj = matmul_rows(&attn_out, &layer.wo, seq_len, d, d);

        // Residual + LN1
        let ffn_dim = 4 * d;
        let mut h_attn = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            let res: Vec<f32> = h[pos * d..(pos + 1) * d]
                .iter()
                .zip(proj[pos * d..(pos + 1) * d].iter())
                .map(|(&hv, &pv)| hv + pv)
                .collect();
            let normed = layer_norm(&res, &layer.ln1_g, &layer.ln1_b);
            h_attn[pos * d..(pos + 1) * d].copy_from_slice(&normed);
        }

        // FFN
        let mut h_ffn = vec![0.0_f32; seq_len * d];
        for pos in 0..seq_len {
            let x = &h_attn[pos * d..(pos + 1) * d];
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
            for v in &mut mid {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
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
            let res2: Vec<f32> = x.iter().zip(out.iter()).map(|(&hv, &ov)| hv + ov).collect();
            let normed2 = layer_norm(&res2, &layer.ln2_g, &layer.ln2_b);
            h_ffn[pos * d..(pos + 1) * d].copy_from_slice(&normed2);
        }

        h_ffn
    }

    /// Run bidirectional BERT-style forward pass on masked sequence.
    /// Returns logit vectors (one per position).
    pub fn forward_masked(&self, masked_ids: &[usize]) -> RecsysResult<Vec<Vec<f32>>> {
        if masked_ids.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        for &id in masked_ids {
            if id != MASK_TOKEN && id >= self.n_items {
                return Err(RecsysError::UnknownItem { id });
            }
        }

        let seq_len = masked_ids.len();
        let d = self.emb_dim;

        let mut h = self.embed_sequence(masked_ids);

        for layer in &self.attn_layers {
            h = self.apply_layer(&h, layer, seq_len);
        }

        // For each position compute logits over all items
        let logits: Vec<Vec<f32>> = (0..seq_len)
            .map(|pos| {
                let h_pos = &h[pos * d..(pos + 1) * d];
                (0..self.n_items)
                    .map(|item| {
                        self.item_emb[item * d..(item + 1) * d]
                            .iter()
                            .zip(h_pos.iter())
                            .map(|(&e, &q)| e * q)
                            .sum()
                    })
                    .collect()
            })
            .collect();

        Ok(logits)
    }
}

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

    fn make_model(seed: u64) -> Bert4Rec {
        let mut rng = LcgRng::new(seed);
        Bert4Rec::new(10, 8, 2, 2, 16, &mut rng).expect("construction ok")
    }

    #[test]
    fn rejects_invalid_construction() {
        let mut rng = LcgRng::new(1);
        assert!(
            Bert4Rec::new(0, 8, 1, 1, 8, &mut rng).is_err(),
            "n_items=0 must fail"
        );
        assert!(
            Bert4Rec::new(10, 0, 1, 1, 8, &mut rng).is_err(),
            "emb_dim=0 must fail"
        );
    }

    #[test]
    fn forward_masked_output_shape() {
        let model = make_model(2);
        // Mix of real items and the mask sentinel (usize::MAX == MASK_TOKEN)
        let seq = [0usize, 1, usize::MAX, 3];
        let logits = model.forward_masked(&seq).expect("forward ok");
        assert_eq!(logits.len(), 4, "one logit vec per position");
        for (pos, row) in logits.iter().enumerate() {
            assert_eq!(
                row.len(),
                10,
                "position {pos}: logit vec must have n_items=10 entries"
            );
        }
    }

    #[test]
    fn forward_masked_rejects_empty_and_oob() {
        let model = make_model(3);
        // Empty sequence
        assert!(model.forward_masked(&[]).is_err(), "empty input must fail");
        // id=10 >= n_items=10 and not the mask sentinel → UnknownItem
        assert!(
            model.forward_masked(&[10]).is_err(),
            "out-of-bounds item must fail"
        );
        // usize::MAX is the MASK_TOKEN sentinel and must be accepted
        assert!(
            model.forward_masked(&[usize::MAX]).is_ok(),
            "mask sentinel (usize::MAX) must be accepted"
        );
    }

    #[test]
    fn logits_are_finite() {
        let model = make_model(4);
        let logits = model
            .forward_masked(&[0, 1, usize::MAX, 2])
            .expect("forward ok");
        for (pos, row) in logits.iter().enumerate() {
            for (item, &v) in row.iter().enumerate() {
                assert!(v.is_finite(), "logit[{pos}][{item}]={v} must be finite");
            }
        }
    }

    #[test]
    fn determinism_across_identical_models() {
        // Two models initialised from the same seed must produce bit-identical outputs.
        let model_a = make_model(5);
        let model_b = make_model(5);
        let seq = [0usize, 1, usize::MAX, 3, 2];
        let out_a = model_a.forward_masked(&seq).expect("fwd a");
        let out_b = model_b.forward_masked(&seq).expect("fwd b");
        for (pos, (row_a, row_b)) in out_a.iter().zip(out_b.iter()).enumerate() {
            for (item, (&a, &b)) in row_a.iter().zip(row_b.iter()).enumerate() {
                assert_eq!(a, b, "logit[{pos}][{item}] must be bit-identical");
            }
        }
    }

    #[test]
    fn mask_sequence_length_and_zero_ratio() {
        let model = make_model(6);
        let items = vec![0usize, 1, 2, 3, 4];
        let mut rng = LcgRng::new(42);
        // Length must be preserved regardless of ratio.
        let masked_half = model.mask_sequence(&items, 0.5, &mut rng);
        assert_eq!(masked_half.len(), items.len(), "length must be preserved");
        // ratio=0.0: next_f32() is always >= 0, so no item is ever masked.
        let mut rng2 = LcgRng::new(99);
        let masked_none = model.mask_sequence(&items, 0.0, &mut rng2);
        assert_eq!(
            masked_none, items,
            "ratio=0.0 must leave all items unchanged"
        );
    }

    #[test]
    fn mask_sequence_full_ratio_all_masked() {
        let model = make_model(7);
        let items = vec![0usize, 1, 2, 3, 4];
        let mut rng = LcgRng::new(7);
        // ratio=1.0: next_f32() ∈ [0, 1), so next_f32() < 1.0 is always true.
        let masked = model.mask_sequence(&items, 1.0, &mut rng);
        for (i, &m) in masked.iter().enumerate() {
            assert_eq!(
                m,
                usize::MAX,
                "ratio=1.0: position {i} must become mask sentinel (usize::MAX)"
            );
        }
    }

    #[test]
    fn bidirectional_future_item_affects_earlier_positions() {
        // BERT4Rec uses full (non-causal) attention: every position attends to every
        // other position. Changing a future item (position 2) must change logits at
        // an earlier position (position 0) because position 0 attends to position 2.
        let mut rng = LcgRng::new(42);
        let mut model = Bert4Rec::new(10, 8, 1, 1, 16, &mut rng).expect("construction ok");
        // Zero positional embeddings so only item identity drives attention keys/values.
        for v in &mut model.pos_emb {
            *v = 0.0;
        }
        // Sequences identical at positions 0 and 1, differing only at position 2.
        let logits_a = model.forward_masked(&[0, usize::MAX, 2]).expect("fwd a");
        let logits_b = model.forward_masked(&[0, usize::MAX, 3]).expect("fwd b");
        // Position-0 logits must differ (item 2 vs item 3 at position 2 changes k[2]/v[2]).
        let differs = logits_a[0]
            .iter()
            .zip(logits_b[0].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "bidirectional attention: changing a future item must influence position-0 logits"
        );
    }
}
