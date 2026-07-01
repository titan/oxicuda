use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

fn leaky_relu(x: f32) -> f32 {
    if x >= 0.0 { x } else { 0.01 * x }
}

pub struct Ngcf {
    pub n_users: usize,
    pub n_items: usize,
    pub emb_dim: usize,
    pub n_layers: usize,
    pub user_emb: Vec<f32>,
    pub item_emb: Vec<f32>,
    /// Per-layer (W1, W2): each [emb_dim x emb_dim]
    pub weights: Vec<(Vec<f32>, Vec<f32>)>,
    /// Concatenated multi-layer user embeddings [n_users x (n_layers+1)*emb_dim]
    pub user_final: Vec<f32>,
    /// Concatenated multi-layer item embeddings [n_items x (n_layers+1)*emb_dim]
    pub item_final: Vec<f32>,
}

impl Ngcf {
    pub fn new(
        n_users: usize,
        n_items: usize,
        emb_dim: usize,
        n_layers: usize,
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
        let w_scale = (2.0 / emb_dim as f32).sqrt();

        let user_emb: Vec<f32> = (0..n_users * emb_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let item_emb: Vec<f32> = (0..n_items * emb_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();

        let weights: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
            .map(|_| {
                let w1: Vec<f32> = (0..emb_dim * emb_dim)
                    .map(|_| rng.next_normal() * w_scale)
                    .collect();
                let w2: Vec<f32> = (0..emb_dim * emb_dim)
                    .map(|_| rng.next_normal() * w_scale)
                    .collect();
                (w1, w2)
            })
            .collect();

        let user_final = user_emb.clone();
        let item_final = item_emb.clone();

        Ok(Self {
            n_users,
            n_items,
            emb_dim,
            n_layers,
            user_emb,
            item_emb,
            weights,
            user_final,
            item_final,
        })
    }

    pub fn propagate(&mut self, edges: &[(usize, usize)]) -> RecsysResult<()> {
        if edges.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }

        let d = self.emb_dim;

        let mut deg_u = vec![0usize; self.n_users];
        let mut deg_i = vec![0usize; self.n_items];
        for &(u, i) in edges {
            if u < self.n_users && i < self.n_items {
                deg_u[u] += 1;
                deg_i[i] += 1;
            }
        }

        let mut cur_user = self.user_emb.clone();
        let mut cur_item = self.item_emb.clone();

        // Collect all layer outputs for concatenation
        let mut user_concat: Vec<f32> = cur_user.clone();
        let mut item_concat: Vec<f32> = cur_item.clone();

        for (w1, w2) in &self.weights {
            let mut next_user = vec![0.0_f32; self.n_users * d];
            let mut next_item = vec![0.0_f32; self.n_items * d];

            // Aggregate neighborhood embeddings
            let mut agg_user = vec![0.0_f32; self.n_users * d];
            let mut agg_item = vec![0.0_f32; self.n_items * d];

            for &(u, i) in edges {
                if u >= self.n_users || i >= self.n_items {
                    continue;
                }
                let du = deg_u[u] as f32;
                let di = deg_i[i] as f32;
                if du < 1.0 || di < 1.0 {
                    continue;
                }
                let norm_u = 1.0 / du.sqrt();
                let norm_i = 1.0 / di.sqrt();

                // Aggregate from items to user: D_u^{-1/2} * e_i
                for k in 0..d {
                    agg_user[u * d + k] += norm_u * cur_item[i * d + k];
                }
                // Aggregate from users to item: D_i^{-1/2} * e_u
                for k in 0..d {
                    agg_item[i * d + k] += norm_i * cur_user[u * d + k];
                }
            }

            // W1 * agg_emb + W2 * (cur_emb ⊙ agg_emb) for each user
            for u in 0..self.n_users {
                let agg = &agg_user[u * d..(u + 1) * d];
                let cur = &cur_user[u * d..(u + 1) * d];

                // hadamard product
                let hadamard: Vec<f32> = cur.iter().zip(agg.iter()).map(|(&c, &a)| c * a).collect();

                for out_k in 0..d {
                    let w1_part: f32 = w1[out_k * d..(out_k + 1) * d]
                        .iter()
                        .zip(agg.iter())
                        .map(|(&w, &a)| w * a)
                        .sum();
                    let w2_part: f32 = w2[out_k * d..(out_k + 1) * d]
                        .iter()
                        .zip(hadamard.iter())
                        .map(|(&w, &h)| w * h)
                        .sum();
                    next_user[u * d + out_k] = leaky_relu(w1_part + w2_part);
                }
            }

            for i in 0..self.n_items {
                let agg = &agg_item[i * d..(i + 1) * d];
                let cur = &cur_item[i * d..(i + 1) * d];
                let hadamard: Vec<f32> = cur.iter().zip(agg.iter()).map(|(&c, &a)| c * a).collect();

                for out_k in 0..d {
                    let w1_part: f32 = w1[out_k * d..(out_k + 1) * d]
                        .iter()
                        .zip(agg.iter())
                        .map(|(&w, &a)| w * a)
                        .sum();
                    let w2_part: f32 = w2[out_k * d..(out_k + 1) * d]
                        .iter()
                        .zip(hadamard.iter())
                        .map(|(&w, &h)| w * h)
                        .sum();
                    next_item[i * d + out_k] = leaky_relu(w1_part + w2_part);
                }
            }

            // Concatenate layer outputs
            for (c, &n) in user_concat.iter_mut().zip(next_user.iter()) {
                let _ = n; // handled below by extending
                let _ = c;
            }
            user_concat.extend_from_slice(&next_user);
            item_concat.extend_from_slice(&next_item);

            cur_user = next_user;
            cur_item = next_item;
        }

        self.user_final = user_concat;
        self.item_final = item_concat;

        Ok(())
    }

    pub fn score(&self, user: usize, item: usize) -> f32 {
        if user >= self.n_users || item >= self.n_items {
            return 0.0;
        }
        let total_dim = self.user_final.len() / self.n_users;
        let u_emb = &self.user_final[user * total_dim..(user + 1) * total_dim];
        let i_emb = &self.item_final[item * total_dim..(item + 1) * total_dim];
        u_emb.iter().zip(i_emb.iter()).map(|(&a, &b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn small_model(rng: &mut LcgRng) -> Ngcf {
        Ngcf::new(5, 10, 4, 2, rng).expect("model construction should succeed")
    }

    #[test]
    fn construction_succeeds() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert_eq!(model.n_users, 5);
        assert_eq!(model.n_items, 10);
        assert_eq!(model.emb_dim, 4);
        assert_eq!(model.n_layers, 2);
        assert_eq!(model.user_emb.len(), 5 * 4);
        assert_eq!(model.item_emb.len(), 10 * 4);
        assert_eq!(model.weights.len(), 2);
        for (w1, w2) in &model.weights {
            assert_eq!(w1.len(), 4 * 4);
            assert_eq!(w2.len(), 4 * 4);
        }
    }

    #[test]
    fn err_n_users_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            Ngcf::new(0, 10, 4, 1, &mut rng),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn err_n_items_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            Ngcf::new(5, 0, 4, 1, &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    #[test]
    fn err_emb_dim_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            Ngcf::new(5, 10, 0, 1, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_propagate_empty_edges() {
        let mut rng = make_rng();
        let mut model = small_model(&mut rng);
        assert!(matches!(
            model.propagate(&[]),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn propagate_output_shape() {
        // After n_layers=2 propagation the concatenated final vectors must
        // have exactly n_users * emb_dim * (n_layers + 1) entries.
        let mut rng = make_rng();
        let mut model = Ngcf::new(3, 5, 4, 2, &mut rng).expect("model construction should succeed");
        let edges = vec![(0, 0), (0, 1), (1, 2), (2, 3)];
        model.propagate(&edges).expect("propagate should succeed");
        assert_eq!(model.user_final.len(), 3 * 4 * (2 + 1));
        assert_eq!(model.item_final.len(), 5 * 4 * (2 + 1));
    }

    #[test]
    fn propagate_identity_weights_closed_form() {
        // Single user, single item, 1 layer, identity W1 = W2 = I₂.
        // user_emb = [1, 2], item_emb = [3, 4], edge (0, 0).
        //
        // deg_u[0]=1, deg_i[0]=1  →  norm_u = norm_i = 1.
        // agg_user = item_emb = [3, 4]
        // agg_item = user_emb = [1, 2]
        // hadamard_user = [1·3, 2·4] = [3, 8]
        // hadamard_item = [3·1, 4·2] = [3, 8]
        //
        // next_user[k] = leaky_relu(I·agg[k] + I·hadamard[k])
        //   k=0 → leaky_relu(3 + 3) = 6
        //   k=1 → leaky_relu(4 + 8) = 12
        //
        // next_item[k] = leaky_relu(I·agg_item[k] + I·hadamard_item[k])
        //   k=0 → leaky_relu(1 + 3) = 4
        //   k=1 → leaky_relu(2 + 8) = 10
        //
        // user_final = [1, 2, 6, 12]   (concat layer0 || layer1)
        // item_final = [3, 4, 4, 10]
        // score(0,0) = 1·3 + 2·4 + 6·4 + 12·10 = 3 + 8 + 24 + 120 = 155
        let mut rng = make_rng();
        let mut model = Ngcf::new(1, 1, 2, 1, &mut rng).expect("model construction should succeed");
        model.user_emb = vec![1.0_f32, 2.0];
        model.item_emb = vec![3.0_f32, 4.0];
        // 2×2 identity matrices, stored row-major
        model.weights = vec![(vec![1.0_f32, 0.0, 0.0, 1.0], vec![1.0_f32, 0.0, 0.0, 1.0])];
        model
            .propagate(&[(0, 0)])
            .expect("propagate should succeed");

        let eps = 1e-5_f32;
        let expected_user_final = [1.0_f32, 2.0, 6.0, 12.0];
        let expected_item_final = [3.0_f32, 4.0, 4.0, 10.0];
        for (got, exp) in model.user_final.iter().zip(expected_user_final.iter()) {
            assert!(
                (got - exp).abs() < eps,
                "user_final mismatch: got {got}, expected {exp}"
            );
        }
        for (got, exp) in model.item_final.iter().zip(expected_item_final.iter()) {
            assert!(
                (got - exp).abs() < eps,
                "item_final mismatch: got {got}, expected {exp}"
            );
        }
        let score = model.score(0, 0);
        assert!(
            (score - 155.0_f32).abs() < eps,
            "score mismatch: got {score}, expected 155.0"
        );
    }

    #[test]
    fn score_oob_returns_zero() {
        let mut rng = make_rng();
        let model = small_model(&mut rng);
        assert_eq!(model.score(model.n_users, 0), 0.0_f32);
        assert_eq!(model.score(0, model.n_items), 0.0_f32);
    }
}
