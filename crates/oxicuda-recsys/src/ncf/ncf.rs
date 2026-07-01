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

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Neural Collaborative Filtering: fuses GMF and MLP towers.
pub struct Ncf {
    pub n_users: usize,
    pub n_items: usize,
    pub emb_dim: usize,
    pub mlp_dims: Vec<usize>,
    pub user_emb: Vec<f32>,
    pub item_emb: Vec<f32>,
    pub gmf_user_emb: Vec<f32>,
    pub gmf_item_emb: Vec<f32>,
    /// (weight, bias) for each MLP layer
    pub mlp_weights: Vec<(Vec<f32>, Vec<f32>)>,
    pub output_w: Vec<f32>,
    pub output_b: f32,
}

impl Ncf {
    pub fn new(
        n_users: usize,
        n_items: usize,
        emb_dim: usize,
        mlp_dims: Vec<usize>,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: n_users });
        }
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if emb_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: emb_dim });
        }
        let scale = (1.0 / emb_dim as f32).sqrt();
        let init_emb = |size: usize, rng: &mut LcgRng| -> Vec<f32> {
            (0..size).map(|_| rng.next_normal() * scale).collect()
        };

        let user_emb = init_emb(n_users * emb_dim, rng);
        let item_emb = init_emb(n_items * emb_dim, rng);
        let gmf_user_emb = init_emb(n_users * emb_dim, rng);
        let gmf_item_emb = init_emb(n_items * emb_dim, rng);

        // MLP input dim = 2 * emb_dim (concat of user and item)
        let mut layer_dims = vec![2 * emb_dim];
        layer_dims.extend_from_slice(&mlp_dims);

        let mut mlp_weights = Vec::with_capacity(mlp_dims.len());
        for window in layer_dims.windows(2) {
            let (fan_in, fan_out) = (window[0], window[1]);
            let sc = (2.0 / fan_in as f32).sqrt();
            let w: Vec<f32> = (0..fan_out * fan_in)
                .map(|_| rng.next_normal() * sc)
                .collect();
            let b = vec![0.0_f32; fan_out];
            mlp_weights.push((w, b));
        }

        let mlp_out_dim = mlp_dims.last().copied().unwrap_or(emb_dim);
        // output_w size = gmf_dim + mlp_out_dim
        let out_dim = emb_dim + mlp_out_dim;
        let out_sc = (2.0 / out_dim as f32).sqrt();
        let output_w: Vec<f32> = (0..out_dim).map(|_| rng.next_normal() * out_sc).collect();

        Ok(Self {
            n_users,
            n_items,
            emb_dim,
            mlp_dims,
            user_emb,
            item_emb,
            gmf_user_emb,
            gmf_item_emb,
            mlp_weights,
            output_w,
            output_b: 0.0,
        })
    }

    pub fn forward(&self, user: usize, item: usize) -> RecsysResult<f32> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        if item >= self.n_items {
            return Err(RecsysError::UnknownItem { id: item });
        }
        let d = self.emb_dim;

        // GMF branch: element-wise product
        let gmf_out: Vec<f32> = self.gmf_user_emb[user * d..(user + 1) * d]
            .iter()
            .zip(self.gmf_item_emb[item * d..(item + 1) * d].iter())
            .map(|(&u, &i)| u * i)
            .collect();

        // MLP branch: concat and pass through layers
        let mut mlp_input: Vec<f32> = self.user_emb[user * d..(user + 1) * d]
            .iter()
            .chain(self.item_emb[item * d..(item + 1) * d].iter())
            .copied()
            .collect();

        let mut current_dim = 2 * d;
        for (w, b) in &self.mlp_weights {
            let out_dim = b.len();
            let mut out = dense(&mlp_input, w, b, current_dim, out_dim);
            relu(&mut out);
            mlp_input = out;
            current_dim = out_dim;
        }

        // Concatenate GMF output and MLP output
        let combined: Vec<f32> = gmf_out.iter().chain(mlp_input.iter()).copied().collect();
        let logit: f32 = self.output_b
            + combined
                .iter()
                .zip(self.output_w.iter())
                .map(|(&c, &w)| c * w)
                .sum::<f32>();

        Ok(sigmoid(logit))
    }

    pub fn bce_loss(&self, user: usize, item: usize, label: f32) -> RecsysResult<f32> {
        let pred = self.forward(user, item)?;
        let pred = pred.clamp(1e-7, 1.0 - 1e-7);
        Ok(-label * pred.ln() - (1.0 - label) * (1.0 - pred).ln())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid_params_succeeds() {
        let mut rng = LcgRng::new(42);
        let model = Ncf::new(5, 8, 4, vec![8, 4], &mut rng).expect("new should succeed");
        assert_eq!(model.n_users, 5);
        assert_eq!(model.n_items, 8);
        assert_eq!(model.emb_dim, 4);
        // Two MLP layers: (2*4→8) and (8→4)
        assert_eq!(model.mlp_weights.len(), 2);
        // output_w length = emb_dim + mlp_out_dim = 4 + 4 = 8
        assert_eq!(model.output_w.len(), 4 + 4);
    }

    #[test]
    fn new_zero_users_returns_err() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Ncf::new(0, 5, 4, vec![8], &mut rng),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn new_zero_dim_returns_err() {
        let mut rng = LcgRng::new(2);
        assert!(matches!(
            Ncf::new(3, 5, 0, vec![8], &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn forward_output_in_open_unit_interval() {
        // sigmoid never returns exactly 0 or 1 for finite logit.
        let mut rng = LcgRng::new(77);
        let model = Ncf::new(4, 6, 8, vec![16, 8], &mut rng).expect("new should succeed");
        for u in 0..4 {
            for i in 0..6 {
                let s = model.forward(u, i).expect("forward ok");
                assert!(
                    s > 0.0 && s < 1.0,
                    "sigmoid output {s} must be in (0,1) for u={u} i={i}"
                );
            }
        }
    }

    #[test]
    fn forward_deterministic_with_same_seed() {
        // Two identically seeded models must produce bit-identical scores.
        let mut rng_a = LcgRng::new(123);
        let model_a = Ncf::new(3, 4, 4, vec![8, 4], &mut rng_a).expect("new ok a");
        let mut rng_b = LcgRng::new(123);
        let model_b = Ncf::new(3, 4, 4, vec![8, 4], &mut rng_b).expect("new ok b");

        for u in 0..3 {
            for i in 0..4 {
                let sa = model_a.forward(u, i).expect("forward ok a");
                let sb = model_b.forward(u, i).expect("forward ok b");
                assert_eq!(sa, sb, "forward({u},{i}) not deterministic: {sa} vs {sb}");
            }
        }
    }

    #[test]
    fn forward_unknown_user_or_item_returns_err() {
        let mut rng = LcgRng::new(5);
        let model = Ncf::new(3, 4, 4, vec![8], &mut rng).expect("new should succeed");
        assert!(matches!(
            model.forward(3, 0),
            Err(RecsysError::UnknownUser { .. })
        ));
        assert!(matches!(
            model.forward(0, 4),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn bce_loss_finite_and_nonneg_for_pos_and_neg_labels() {
        let mut rng = LcgRng::new(88);
        let model = Ncf::new(4, 5, 6, vec![12, 6], &mut rng).expect("new should succeed");
        for u in 0..4 {
            for i in 0..5 {
                let loss_pos = model.bce_loss(u, i, 1.0).expect("bce ok pos");
                let loss_neg = model.bce_loss(u, i, 0.0).expect("bce ok neg");
                assert!(
                    loss_pos.is_finite() && loss_pos >= 0.0,
                    "bce_loss pos: {loss_pos} at u={u} i={i}"
                );
                assert!(
                    loss_neg.is_finite() && loss_neg >= 0.0,
                    "bce_loss neg: {loss_neg} at u={u} i={i}"
                );
            }
        }
    }

    #[test]
    fn gmf_branch_element_wise_product_contributes_to_logit() {
        // Zero out output_w for the MLP tail so only the GMF head reaches the
        // logit.  Then the expected output is sigmoid(Σ_k gu_k * gi_k * w_k).
        let mut rng = LcgRng::new(200);
        let d = 4usize;
        let mlp_out = 4usize;
        let mut model = Ncf::new(2, 2, d, vec![mlp_out], &mut rng).expect("new should succeed");

        // Zero MLP portion of output weights; keep GMF portion intact.
        for w in model.output_w[d..].iter_mut() {
            *w = 0.0;
        }
        model.output_b = 0.0;

        let u = 0usize;
        let i = 0usize;
        let expected_logit: f32 = model.gmf_user_emb[u * d..(u + 1) * d]
            .iter()
            .zip(model.gmf_item_emb[i * d..(i + 1) * d].iter())
            .zip(model.output_w[..d].iter())
            .map(|((&gu, &gi), &w)| gu * gi * w)
            .sum();
        let expected = 1.0 / (1.0 + (-expected_logit).exp());
        let got = model.forward(u, i).expect("forward ok");
        assert!(
            (got - expected).abs() < 1e-5,
            "GMF branch mismatch: expected {expected:.7}, got {got:.7}"
        );
    }
}
