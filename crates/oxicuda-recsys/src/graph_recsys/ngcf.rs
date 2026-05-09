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
