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
