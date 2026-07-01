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

fn relu_vec(mut x: Vec<f32>) -> Vec<f32> {
    for v in &mut x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    x
}

pub struct TwoTower {
    pub user_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub item_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl TwoTower {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if input_dim == 0 || hidden_dim == 0 || output_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        let build_tower = |rng: &mut LcgRng| -> Vec<(Vec<f32>, Vec<f32>)> {
            let mut layers = Vec::with_capacity(n_layers);
            let mut in_dim = input_dim;
            for layer_idx in 0..n_layers {
                let out_dim = if layer_idx + 1 == n_layers {
                    output_dim
                } else {
                    hidden_dim
                };
                let sc = (2.0 / in_dim as f32).sqrt();
                let w: Vec<f32> = (0..out_dim * in_dim)
                    .map(|_| rng.next_normal() * sc)
                    .collect();
                let b = vec![0.0_f32; out_dim];
                layers.push((w, b));
                in_dim = out_dim;
            }
            layers
        };

        let user_layers = build_tower(rng);
        let item_layers = build_tower(rng);

        Ok(Self {
            user_layers,
            item_layers,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    pub fn encode_user(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        self.mlp_forward(x, &self.user_layers, self.input_dim)
    }

    pub fn encode_item(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        self.mlp_forward(x, &self.item_layers, self.input_dim)
    }

    fn mlp_forward(
        &self,
        x: &[f32],
        layers: &[(Vec<f32>, Vec<f32>)],
        input_dim: usize,
    ) -> RecsysResult<Vec<f32>> {
        let mut current = x.to_vec();
        let mut curr_dim = input_dim;
        for (idx, (w, b)) in layers.iter().enumerate() {
            let out_dim = b.len();
            let out = dense(&current, w, b, curr_dim, out_dim);
            current = if idx + 1 < layers.len() {
                relu_vec(out)
            } else {
                out
            };
            curr_dim = out_dim;
        }
        Ok(current)
    }

    pub fn score(&self, user_x: &[f32], item_x: &[f32]) -> RecsysResult<f32> {
        let u = self.encode_user(user_x)?;
        let i = self.encode_item(item_x)?;
        let dot = u.iter().zip(i.iter()).map(|(&a, &b)| a * b).sum();
        Ok(dot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn construction_succeeds() {
        let mut rng = make_rng();
        let model = TwoTower::new(4, 8, 3, 2, &mut rng).expect("model construction should succeed");
        assert_eq!(model.input_dim, 4);
        assert_eq!(model.hidden_dim, 8);
        assert_eq!(model.output_dim, 3);
        assert_eq!(model.user_layers.len(), 2);
        assert_eq!(model.item_layers.len(), 2);
        // Layer 0: 4→8 (hidden), layer 1: 8→3 (output)
        assert_eq!(model.user_layers[0].0.len(), 8 * 4);
        assert_eq!(model.user_layers[0].1.len(), 8);
        assert_eq!(model.user_layers[1].0.len(), 3 * 8);
        assert_eq!(model.user_layers[1].1.len(), 3);
    }

    #[test]
    fn err_zero_dims() {
        let mut rng = make_rng();
        assert!(matches!(
            TwoTower::new(0, 8, 3, 1, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
        let mut rng2 = LcgRng::new(43);
        assert!(matches!(
            TwoTower::new(4, 0, 3, 1, &mut rng2),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_dimension_mismatch() {
        let mut rng = make_rng();
        let model = TwoTower::new(4, 8, 3, 1, &mut rng).expect("model construction should succeed");
        // input is only 3 elements but input_dim=4
        assert!(matches!(
            model.encode_user(&[1.0_f32, 2.0, 3.0]),
            Err(RecsysError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            model.encode_item(&[1.0_f32, 2.0, 3.0]),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn encode_identity_weights_exact() {
        // 1-layer model: input_dim=2, output_dim=2 (no hidden dimension used).
        // Set W = I₂, b = 0 for both towers.  The final layer has no relu,
        // so encode(x) must equal x exactly.
        //
        // score([3, 5], [2, 7]) = 3·2 + 5·7 = 41.
        let mut rng = make_rng();
        let mut model =
            TwoTower::new(2, 4, 2, 1, &mut rng).expect("model construction should succeed");
        let identity_w = vec![1.0_f32, 0.0, 0.0, 1.0];
        let zero_b = vec![0.0_f32, 0.0];
        model.user_layers = vec![(identity_w.clone(), zero_b.clone())];
        model.item_layers = vec![(identity_w, zero_b)];

        let user_in = [3.0_f32, 5.0];
        let item_in = [2.0_f32, 7.0];
        let u_emb = model
            .encode_user(&user_in)
            .expect("encode_user should succeed");
        let i_emb = model
            .encode_item(&item_in)
            .expect("encode_item should succeed");

        let eps = 1e-5_f32;
        assert!((u_emb[0] - 3.0).abs() < eps, "u_emb[0]={}", u_emb[0]);
        assert!((u_emb[1] - 5.0).abs() < eps, "u_emb[1]={}", u_emb[1]);
        assert!((i_emb[0] - 2.0).abs() < eps, "i_emb[0]={}", i_emb[0]);
        assert!((i_emb[1] - 7.0).abs() < eps, "i_emb[1]={}", i_emb[1]);

        let score = model
            .score(&user_in, &item_in)
            .expect("score should succeed");
        assert!((score - 41.0).abs() < eps, "score={score}, expected 41.0");
    }

    #[test]
    fn score_equals_manual_dot_of_tower_outputs() {
        // Verify score() == dot(encode_user(), encode_item()) for arbitrary weights.
        let mut rng = make_rng();
        let model = TwoTower::new(4, 6, 3, 2, &mut rng).expect("model construction should succeed");
        let user_in = [0.1_f32, 0.2, 0.3, 0.4];
        let item_in = [0.5_f32, 0.6, 0.7, 0.8];

        let u_emb = model
            .encode_user(&user_in)
            .expect("encode_user should succeed");
        let i_emb = model
            .encode_item(&item_in)
            .expect("encode_item should succeed");
        let manual_dot: f32 = u_emb.iter().zip(i_emb.iter()).map(|(&a, &b)| a * b).sum();
        let score = model
            .score(&user_in, &item_in)
            .expect("score should succeed");
        let eps = 1e-5_f32;
        assert!(
            (score - manual_dot).abs() < eps,
            "score={score} must equal manual dot={manual_dot}"
        );
    }

    #[test]
    fn encode_output_has_correct_dim() {
        let mut rng = make_rng();
        let model = TwoTower::new(4, 8, 3, 2, &mut rng).expect("model construction should succeed");
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let u = model.encode_user(&x).expect("encode_user should succeed");
        let i = model.encode_item(&x).expect("encode_item should succeed");
        assert_eq!(u.len(), model.output_dim);
        assert_eq!(i.len(), model.output_dim);
    }

    #[test]
    fn score_ordering_by_alignment() {
        // With identity weights encode(x) = x, so score = dot(user, item).
        // A parallel item must score strictly higher than an orthogonal one.
        let mut rng = make_rng();
        let mut model =
            TwoTower::new(2, 4, 2, 1, &mut rng).expect("model construction should succeed");
        let identity_w = vec![1.0_f32, 0.0, 0.0, 1.0];
        let zero_b = vec![0.0_f32, 0.0];
        model.user_layers = vec![(identity_w.clone(), zero_b.clone())];
        model.item_layers = vec![(identity_w, zero_b)];

        let user = [1.0_f32, 0.0];
        let item_parallel = [2.0_f32, 0.0]; // dot = 2.0
        let item_orthogonal = [0.0_f32, 1.0]; // dot = 0.0

        let score_hi = model
            .score(&user, &item_parallel)
            .expect("score should succeed");
        let score_lo = model
            .score(&user, &item_orthogonal)
            .expect("score should succeed");
        assert!(
            score_hi > score_lo,
            "parallel item (score={score_hi}) must beat orthogonal item (score={score_lo})"
        );
    }
}
