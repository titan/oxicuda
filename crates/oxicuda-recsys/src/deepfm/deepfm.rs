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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn tiny_model(rng: &mut LcgRng) -> DeepFm {
        DeepFm::new(vec![4, 5, 3], 4, &[8, 4], rng).expect("must build")
    }

    #[test]
    fn output_in_unit_interval() {
        let mut rng = make_rng(1);
        let model = tiny_model(&mut rng);
        let p = model.forward(&[1, 2, 0]).expect("forward must succeed");
        assert!((0.0..=1.0).contains(&p), "output {p} not in [0,1]");
    }

    #[test]
    fn deterministic_same_seed() {
        let mut rng = make_rng(2);
        let model = tiny_model(&mut rng);
        let p1 = model.forward(&[0, 0, 0]).expect("must succeed");
        let p2 = model.forward(&[0, 0, 0]).expect("must succeed");
        assert_eq!(p1, p2, "same input must yield identical output");
    }

    #[test]
    fn finite_output() {
        let mut rng = make_rng(3);
        let model = tiny_model(&mut rng);
        let p = model.forward(&[3, 4, 2]).expect("forward must succeed");
        assert!(p.is_finite(), "output must be finite, got {p}");
    }

    /// DeepFM FM 2nd-order term is 0.5·(‖Σe_i‖² – Σ‖e_i‖²), which for two
    /// fields equals e₀·e₁ (sum of all pairwise dot products). Build a 2-field
    /// model, zero the linear and deep weights, install known embeddings, and
    /// verify that forward ≈ sigmoid(e₀·e₁).
    #[test]
    fn fm_second_order_matches_closed_form() {
        let mut rng = make_rng(99);
        // 2 fields, each cardinality 2, embedding dim 2, no hidden deep layers.
        let mut model = DeepFm::new(vec![2, 2], 2, &[], &mut rng).expect("must build");

        // Zero linear weights (total size = 2+2 = 4 entries).
        for v in &mut model.linear_w {
            *v = 0.0;
        }
        // Zero the single final deep scalar layer (weights size = deep_input_dim = 4).
        for (w, b) in &mut model.deep_layers {
            for v in w.iter_mut() {
                *v = 0.0;
            }
            for v in b.iter_mut() {
                *v = 0.0;
            }
        }
        // field 0 val 0 → e₀=[1,2];  field 1 val 0 → e₁=[3,4]
        model.embeddings[0] = vec![1.0, 2.0, 0.0, 0.0];
        model.embeddings[1] = vec![3.0, 4.0, 0.0, 0.0];

        // Analytic: e₀·e₁ = 1·3 + 2·4 = 11.0
        // FM = 0.5·(‖[4,6]‖² – (‖[1,2]‖² + ‖[3,4]‖²)) = 0.5·(52 – 30) = 11.0
        let expected_fm = 11.0_f32;
        let expected_p = 1.0 / (1.0 + (-expected_fm).exp());

        let p = model.forward(&[0, 0]).expect("forward must succeed");
        assert!(
            (p - expected_p).abs() < 1e-5,
            "FM closed form: got {p}, want {expected_p}"
        );
    }

    #[test]
    fn wrong_field_count_errors() {
        let mut rng = make_rng(4);
        let model = tiny_model(&mut rng); // 3 fields
        let err = model.forward(&[0, 0]); // only 2 → mismatch
        assert!(matches!(err, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn field_id_out_of_bounds_errors() {
        let mut rng = make_rng(5);
        let model = DeepFm::new(vec![3, 3], 2, &[4], &mut rng).expect("must build");
        // field 0 has cardinality 3; id=3 is out of bounds → Internal error
        let err = model.forward(&[3, 0]);
        assert!(matches!(err, Err(RecsysError::Internal { .. })));
    }

    #[test]
    fn empty_field_dims_rejected() {
        let mut rng = make_rng(6);
        let err = DeepFm::new(vec![], 4, &[8], &mut rng);
        assert!(matches!(err, Err(RecsysError::EmptyInput)));
    }

    #[test]
    fn zero_emb_dim_rejected() {
        let mut rng = make_rng(7);
        let err = DeepFm::new(vec![3, 3], 0, &[8], &mut rng);
        assert!(matches!(err, Err(RecsysError::InvalidEmbeddingDim { .. })));
    }
}
