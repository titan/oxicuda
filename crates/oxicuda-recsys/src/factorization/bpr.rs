use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

pub struct Bpr {
    pub n_users: usize,
    pub n_items: usize,
    pub dim: usize,
    pub user_emb: Vec<f32>,
    pub item_emb: Vec<f32>,
    pub lr: f32,
    pub reg: f32,
}

impl Bpr {
    pub fn new(
        n_users: usize,
        n_items: usize,
        dim: usize,
        lr: f32,
        reg: f32,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: n_users });
        }
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: dim });
        }
        let scale = (1.0 / dim as f32).sqrt();
        let mut user_emb = vec![0.0_f32; n_users * dim];
        let mut item_emb = vec![0.0_f32; n_items * dim];
        for v in &mut user_emb {
            *v = rng.next_normal() * scale;
        }
        for v in &mut item_emb {
            *v = rng.next_normal() * scale;
        }
        Ok(Self {
            n_users,
            n_items,
            dim,
            user_emb,
            item_emb,
            lr,
            reg,
        })
    }

    /// Returns average BPR loss over the triplets.
    pub fn train_step(&mut self, triplets: &[(usize, usize, usize)]) -> f32 {
        if triplets.is_empty() {
            return 0.0;
        }
        let d = self.dim;
        let mut total_loss = 0.0_f32;

        for &(u, i_pos, i_neg) in triplets {
            if u >= self.n_users || i_pos >= self.n_items || i_neg >= self.n_items {
                continue;
            }
            let x_ui: f32 = self.user_emb[u * d..(u + 1) * d]
                .iter()
                .zip(self.item_emb[i_pos * d..(i_pos + 1) * d].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let x_uj: f32 = self.user_emb[u * d..(u + 1) * d]
                .iter()
                .zip(self.item_emb[i_neg * d..(i_neg + 1) * d].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            let x_uij = x_ui - x_uj;
            let sigma = sigmoid(x_uij);
            let grad_factor = 1.0 - sigma;

            total_loss -= (sigma + 1e-10).ln();

            let u_start = u * d;
            let ip_start = i_pos * d;
            let in_start = i_neg * d;

            // Gradient updates (must clone slices to avoid borrow issues)
            let u_emb: Vec<f32> = self.user_emb[u_start..u_start + d].to_vec();
            let ip_emb: Vec<f32> = self.item_emb[ip_start..ip_start + d].to_vec();
            let in_emb: Vec<f32> = self.item_emb[in_start..in_start + d].to_vec();

            for (k, (&ip_k, &in_k)) in ip_emb.iter().zip(in_emb.iter()).enumerate() {
                self.user_emb[u_start + k] +=
                    self.lr * (grad_factor * (ip_k - in_k) - self.reg * u_emb[k]);
            }
            for (k, &u_k) in u_emb.iter().enumerate() {
                self.item_emb[ip_start + k] += self.lr * (grad_factor * u_k - self.reg * ip_emb[k]);
                self.item_emb[in_start + k] +=
                    self.lr * (-grad_factor * u_k - self.reg * in_emb[k]);
            }
        }

        total_loss / triplets.len() as f32
    }

    pub fn score(&self, user: usize, item: usize) -> f32 {
        if user >= self.n_users || item >= self.n_items {
            return 0.0;
        }
        let d = self.dim;
        self.user_emb[user * d..(user + 1) * d]
            .iter()
            .zip(self.item_emb[item * d..(item + 1) * d].iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
