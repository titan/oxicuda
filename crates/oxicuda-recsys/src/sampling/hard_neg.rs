use std::collections::BTreeSet;

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

pub struct HardNegSampler {
    /// Per-user dot scores against all items: [n_users x n_items]
    pub scores: Vec<f32>,
    pub n_items: usize,
    pub n_users: usize,
}

impl HardNegSampler {
    pub fn new(n_users: usize, n_items: usize) -> RecsysResult<Self> {
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: n_users });
        }
        Ok(Self {
            scores: vec![0.0_f32; n_users * n_items],
            n_items,
            n_users,
        })
    }

    pub fn update_scores(
        &mut self,
        user: usize,
        user_emb: &[f32],
        item_embs: &[f32],
    ) -> RecsysResult<()> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        let d = user_emb.len();
        if d == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d });
        }
        if item_embs.len() != self.n_items * d {
            return Err(RecsysError::DimensionMismatch {
                expected: self.n_items * d,
                got: item_embs.len(),
            });
        }
        for item in 0..self.n_items {
            let score: f32 = user_emb
                .iter()
                .zip(item_embs[item * d..(item + 1) * d].iter())
                .map(|(&u, &e)| u * e)
                .sum();
            self.scores[user * self.n_items + item] = score;
        }
        Ok(())
    }

    /// Sample from the top-20% scoring non-positive items for the given user.
    pub fn sample(
        &self,
        user: usize,
        user_positives: &BTreeSet<usize>,
        rng: &mut LcgRng,
    ) -> RecsysResult<usize> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        let user_scores = &self.scores[user * self.n_items..(user + 1) * self.n_items];

        // Collect non-positive item scores
        let mut candidates: Vec<(usize, f32)> = user_scores
            .iter()
            .enumerate()
            .filter(|(item, _)| !user_positives.contains(item))
            .map(|(item, &s)| (item, s))
            .collect();

        if candidates.is_empty() {
            return Err(RecsysError::NoNegativeAvailable { user });
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-20%
        let top_k = ((candidates.len() as f32 * 0.2).ceil() as usize).max(1);
        let pool = &candidates[..top_k];

        // Random sample from pool
        let idx = (rng.next_u32() as usize) % pool.len();
        Ok(pool[idx].0)
    }
}
