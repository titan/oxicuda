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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn tiny_model(rng: &mut LcgRng) -> AutoInt {
        AutoInt::new(vec![4, 5, 3], 4, 2, rng).expect("must build")
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

    /// Set Wq=0, Wk=0 so all attention scores are 0 → softmax is uniform (1/n_fields per row).
    /// Set Wv=I so V=X (embeddings unchanged). Compute the expected post-residual pooled
    /// representation analytically and verify forward() matches to within 1e-5.
    ///
    /// This simultaneously tests:
    ///   • softmax gives non-negative weights summing to 1 (uniform case)
    ///   • residual: x_new[i] = (x[i] + attn_out[i]).max(0)
    ///   • output dot product with output_w
    #[test]
    fn attention_uniform_qk_zero_identity_v_matches_analytic() {
        let mut rng = make_rng(55);
        let field_dims = vec![2, 2]; // 2 fields, cardinality 2 each
        let emb_dim = 2;
        let mut model = AutoInt::new(field_dims, emb_dim, 1, &mut rng).expect("must build");

        // Known embeddings: field 0 val 0 → [1,2], field 1 val 0 → [3,4]
        model.embeddings[0] = vec![1.0, 2.0, 0.0, 0.0];
        model.embeddings[1] = vec![3.0, 4.0, 0.0, 0.0];

        let d = emb_dim;
        let zeros = vec![0.0_f32; d * d];
        let mut identity = vec![0.0_f32; d * d];
        for i in 0..d {
            identity[i * d + i] = 1.0;
        }

        model.attn_layers[0].0 = zeros.clone(); // Wq = 0  → Q = 0
        model.attn_layers[0].1 = zeros.clone(); // Wk = 0  → K = 0
        model.attn_layers[0].2 = identity; //      Wv = I  → V = X

        // Derivation (n_fields=2, d=2):
        //   Q=K=0 ⟹ attn_score[i,j]=0 ∀i,j ⟹ softmax row = [0.5, 0.5]
        //   V = X: v0=[1,2], v1=[3,4]
        //   attn_out[0] = 0.5·v0 + 0.5·v1 = [2.0, 3.0]
        //   attn_out[1] = 0.5·v0 + 0.5·v1 = [2.0, 3.0]
        //   residual+ReLU:
        //     x_new[0..2] = (x0 + attn_out[0]).max(0) = ([1+2,2+3]).max(0) = [3.0, 5.0]
        //     x_new[2..4] = (x1 + attn_out[1]).max(0) = ([3+2,4+3]).max(0) = [5.0, 7.0]
        let expected_pooled = [3.0_f32, 5.0, 5.0, 7.0];
        let expected_logit = model.output_b
            + expected_pooled
                .iter()
                .zip(model.output_w.iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum::<f32>();
        let expected_p = 1.0 / (1.0 + (-expected_logit).exp());

        let p = model.forward(&[0, 0]).expect("forward must succeed");
        assert!(
            (p - expected_p).abs() < 1e-5,
            "analytic attention+residual: got {p}, want {expected_p}"
        );
    }

    /// Verify the residual connection is active: the output with residual (x + attn_out)
    /// must differ from the output of pure attention only (attn_out alone, without adding x).
    /// Uses the same zero-Wq/Wk, identity-Wv setup so the expected pure-attention pooled
    /// representation can be computed analytically as [2,3,2,3].
    #[test]
    fn residual_changes_output_versus_attention_only() {
        let mut rng = make_rng(56);
        let field_dims = vec![2, 2];
        let emb_dim = 2;
        let mut model = AutoInt::new(field_dims, emb_dim, 1, &mut rng).expect("must build");

        model.embeddings[0] = vec![1.0, 2.0, 0.0, 0.0];
        model.embeddings[1] = vec![3.0, 4.0, 0.0, 0.0];

        let d = emb_dim;
        let zeros = vec![0.0_f32; d * d];
        let mut identity = vec![0.0_f32; d * d];
        for i in 0..d {
            identity[i * d + i] = 1.0;
        }
        model.attn_layers[0].0 = zeros.clone();
        model.attn_layers[0].1 = zeros.clone();
        model.attn_layers[0].2 = identity;

        let p_with_residual = model.forward(&[0, 0]).expect("forward must succeed");

        // Without residual the pooled representation would be attn_out.max(0) = [2,3,2,3].
        // (attn_out[i] = mean(X) = 0.5·[1,2]+0.5·[3,4] = [2,3] for both rows; all positive.)
        let attn_only_pooled = [2.0_f32, 3.0, 2.0, 3.0];
        let logit_no_residual = model.output_b
            + attn_only_pooled
                .iter()
                .zip(model.output_w.iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum::<f32>();
        let p_no_residual = 1.0 / (1.0 + (-logit_no_residual).exp());

        // The actual pooled is [3,5,5,7] (with residual) vs [2,3,2,3] (without).
        // Unless output_w is exactly orthogonal to [1,2,3,4], they differ.
        assert!(
            (p_with_residual - p_no_residual).abs() > 1e-6,
            "residual must change output: with={p_with_residual}, without={p_no_residual}"
        );
    }

    #[test]
    fn wrong_field_count_errors() {
        let mut rng = make_rng(4);
        let model = tiny_model(&mut rng); // 3 fields
        let err = model.forward(&[0, 0]); // 2 fields given → mismatch
        assert!(matches!(err, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_field_dims_rejected() {
        let mut rng = make_rng(5);
        let err = AutoInt::new(vec![], 4, 2, &mut rng);
        assert!(matches!(err, Err(RecsysError::EmptyInput)));
    }

    #[test]
    fn zero_emb_dim_rejected() {
        let mut rng = make_rng(6);
        let err = AutoInt::new(vec![3, 3], 0, 2, &mut rng);
        assert!(matches!(err, Err(RecsysError::InvalidEmbeddingDim { .. })));
    }
}
