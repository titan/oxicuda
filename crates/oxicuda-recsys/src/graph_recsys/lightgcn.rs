use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

pub struct LightGcn {
    pub n_users: usize,
    pub n_items: usize,
    pub emb_dim: usize,
    pub n_layers: usize,
    pub user_emb: Vec<f32>,
    pub item_emb: Vec<f32>,
}

impl LightGcn {
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
        let user_emb: Vec<f32> = (0..n_users * emb_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let item_emb: Vec<f32> = (0..n_items * emb_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();

        Ok(Self {
            n_users,
            n_items,
            emb_dim,
            n_layers,
            user_emb,
            item_emb,
        })
    }

    pub fn propagate(&mut self, edges: &[(usize, usize)]) -> RecsysResult<()> {
        if edges.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }

        let d = self.emb_dim;

        // Degree computation
        let mut deg_u = vec![0usize; self.n_users];
        let mut deg_i = vec![0usize; self.n_items];
        for &(u, i) in edges {
            if u < self.n_users && i < self.n_items {
                deg_u[u] += 1;
                deg_i[i] += 1;
            }
        }

        // Store layer-wise embeddings to average at the end
        let mut all_user_layers: Vec<Vec<f32>> = vec![self.user_emb.clone()];
        let mut all_item_layers: Vec<Vec<f32>> = vec![self.item_emb.clone()];

        let mut cur_user = self.user_emb.clone();
        let mut cur_item = self.item_emb.clone();

        for _ in 0..self.n_layers {
            let mut next_user = vec![0.0_f32; self.n_users * d];
            let mut next_item = vec![0.0_f32; self.n_items * d];

            for &(u, i) in edges {
                if u >= self.n_users || i >= self.n_items {
                    continue;
                }
                let du = deg_u[u] as f32;
                let di = deg_i[i] as f32;
                if du < 1.0 || di < 1.0 {
                    continue;
                }
                let norm = 1.0 / (du * di).sqrt();

                for k in 0..d {
                    next_user[u * d + k] += norm * cur_item[i * d + k];
                    next_item[i * d + k] += norm * cur_user[u * d + k];
                }
            }

            all_user_layers.push(next_user.clone());
            all_item_layers.push(next_item.clone());
            cur_user = next_user;
            cur_item = next_item;
        }

        // Final = mean of all layer embeddings
        let n_layers_total = all_user_layers.len() as f32;
        let mut final_user = vec![0.0_f32; self.n_users * d];
        let mut final_item = vec![0.0_f32; self.n_items * d];

        for layer_emb in &all_user_layers {
            for (fv, &lv) in final_user.iter_mut().zip(layer_emb.iter()) {
                *fv += lv;
            }
        }
        for layer_emb in &all_item_layers {
            for (fv, &lv) in final_item.iter_mut().zip(layer_emb.iter()) {
                *fv += lv;
            }
        }

        let inv = 1.0 / n_layers_total;
        for v in &mut final_user {
            *v *= inv;
        }
        for v in &mut final_item {
            *v *= inv;
        }

        self.user_emb = final_user;
        self.item_emb = final_item;

        Ok(())
    }

    pub fn score(&self, user: usize, item: usize) -> f32 {
        if user >= self.n_users || item >= self.n_items {
            return 0.0;
        }
        let d = self.emb_dim;
        self.user_emb[user * d..(user + 1) * d]
            .iter()
            .zip(self.item_emb[item * d..(item + 1) * d].iter())
            .map(|(&u, &i)| u * i)
            .sum()
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
        let model = LightGcn::new(4, 8, 6, 3, &mut rng).expect("model construction should succeed");
        assert_eq!(model.n_users, 4);
        assert_eq!(model.n_items, 8);
        assert_eq!(model.emb_dim, 6);
        assert_eq!(model.n_layers, 3);
        assert_eq!(model.user_emb.len(), 4 * 6);
        assert_eq!(model.item_emb.len(), 8 * 6);
    }

    #[test]
    fn err_n_users_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            LightGcn::new(0, 8, 6, 1, &mut rng),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn err_n_items_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            LightGcn::new(4, 0, 6, 1, &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    #[test]
    fn err_emb_dim_zero() {
        let mut rng = make_rng();
        assert!(matches!(
            LightGcn::new(4, 8, 0, 1, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn err_propagate_empty_edges() {
        let mut rng = make_rng();
        let mut model =
            LightGcn::new(4, 8, 6, 1, &mut rng).expect("model construction should succeed");
        assert!(matches!(
            model.propagate(&[]),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn propagate_single_edge_closed_form() {
        // 1-user, 1-item, dim=2, 1 layer.
        // user_emb=[1,0], item_emb=[0,1], edge (0,0).
        // deg_u[0]=1, deg_i[0]=1  →  norm = 1/√(1·1) = 1.
        //
        // Layer 1:
        //   next_user[0] = 1·item_emb[0] = [0, 1]
        //   next_item[0] = 1·user_emb[0] = [1, 0]
        //
        // Final = mean of 2 layers (layer0 + layer1):
        //   final_user[0] = ([1+0]/2, [0+1]/2) = [0.5, 0.5]
        //   final_item[0] = ([0+1]/2, [1+0]/2) = [0.5, 0.5]
        //
        // score(0,0) = 0.5·0.5 + 0.5·0.5 = 0.5
        let mut rng = make_rng();
        let mut model =
            LightGcn::new(1, 1, 2, 1, &mut rng).expect("model construction should succeed");
        model.user_emb = vec![1.0_f32, 0.0];
        model.item_emb = vec![0.0_f32, 1.0];
        model
            .propagate(&[(0, 0)])
            .expect("propagate should succeed");

        let eps = 1e-5_f32;
        assert!(
            (model.user_emb[0] - 0.5).abs() < eps,
            "user_emb[0]={}, expected 0.5",
            model.user_emb[0]
        );
        assert!(
            (model.user_emb[1] - 0.5).abs() < eps,
            "user_emb[1]={}, expected 0.5",
            model.user_emb[1]
        );
        assert!(
            (model.item_emb[0] - 0.5).abs() < eps,
            "item_emb[0]={}, expected 0.5",
            model.item_emb[0]
        );
        assert!(
            (model.item_emb[1] - 0.5).abs() < eps,
            "item_emb[1]={}, expected 0.5",
            model.item_emb[1]
        );

        let score = model.score(0, 0);
        assert!((score - 0.5).abs() < eps, "score={score}, expected 0.5");
    }

    #[test]
    fn propagate_symmetric_norm_two_users_one_item() {
        // 2-users, 1-item, dim=2, 1 layer.
        // user0=[1,0], user1=[0,1], item0=[1,1], edges [(0,0),(1,0)].
        // deg_u[0]=1, deg_u[1]=1, deg_i[0]=2
        //   → norm for both edges = 1/√(1·2) = 1/√2.
        //
        // Layer 1:
        //   next_user[0] = (1/√2)·[1,1]
        //   next_user[1] = (1/√2)·[1,1]
        //   next_item[0] = (1/√2)·[1,0] + (1/√2)·[0,1] = [1/√2, 1/√2]
        //
        // Final (mean of 2 layers):
        //   final_user[0] = ((1 + 1/√2)/2, (0 + 1/√2)/2)
        //   final_user[1] = ((0 + 1/√2)/2, (1 + 1/√2)/2)
        //   final_item[0] = ((1 + 1/√2)/2, (1 + 1/√2)/2)
        let mut rng = make_rng();
        let mut model =
            LightGcn::new(2, 1, 2, 1, &mut rng).expect("model construction should succeed");
        model.user_emb = vec![1.0_f32, 0.0, 0.0, 1.0];
        model.item_emb = vec![1.0_f32, 1.0];
        model
            .propagate(&[(0, 0), (1, 0)])
            .expect("propagate should succeed");

        let sq2 = 2.0_f32.sqrt();
        let eps = 1e-5_f32;

        let exp_u0_0 = (1.0 + 1.0 / sq2) / 2.0;
        let exp_u0_1 = (0.0 + 1.0 / sq2) / 2.0;
        let exp_u1_0 = (0.0 + 1.0 / sq2) / 2.0;
        let exp_u1_1 = (1.0 + 1.0 / sq2) / 2.0;
        let exp_i0_v = (1.0 + 1.0 / sq2) / 2.0;

        assert!(
            (model.user_emb[0] - exp_u0_0).abs() < eps,
            "user0[0]: got {}, expected {exp_u0_0}",
            model.user_emb[0]
        );
        assert!(
            (model.user_emb[1] - exp_u0_1).abs() < eps,
            "user0[1]: got {}, expected {exp_u0_1}",
            model.user_emb[1]
        );
        assert!(
            (model.user_emb[2] - exp_u1_0).abs() < eps,
            "user1[0]: got {}, expected {exp_u1_0}",
            model.user_emb[2]
        );
        assert!(
            (model.user_emb[3] - exp_u1_1).abs() < eps,
            "user1[1]: got {}, expected {exp_u1_1}",
            model.user_emb[3]
        );
        assert!(
            (model.item_emb[0] - exp_i0_v).abs() < eps,
            "item0[0]: got {}, expected {exp_i0_v}",
            model.item_emb[0]
        );
        assert!(
            (model.item_emb[1] - exp_i0_v).abs() < eps,
            "item0[1]: got {}, expected {exp_i0_v}",
            model.item_emb[1]
        );
    }

    #[test]
    fn score_oob_returns_zero() {
        let mut rng = make_rng();
        let model = LightGcn::new(4, 8, 6, 1, &mut rng).expect("model construction should succeed");
        assert_eq!(model.score(model.n_users, 0), 0.0_f32);
        assert_eq!(model.score(0, model.n_items), 0.0_f32);
    }
}
