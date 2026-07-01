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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use std::collections::BTreeSet;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn construction_succeeds() {
        let s = HardNegSampler::new(3, 10).expect("construction should succeed");
        assert_eq!(s.n_users, 3);
        assert_eq!(s.n_items, 10);
        assert_eq!(s.scores.len(), 30);
        assert!(s.scores.iter().all(|&v| v == 0.0_f32));
    }

    #[test]
    fn err_invalid_counts() {
        assert!(matches!(
            HardNegSampler::new(0, 10),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
        assert!(matches!(
            HardNegSampler::new(3, 0),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    #[test]
    fn update_scores_computes_correct_dot_products() {
        // user_emb = [1, 0]
        // item0=[2,0] → dot=2, item1=[0,3] → dot=0, item2=[4,0] → dot=4
        let mut s = HardNegSampler::new(1, 3).expect("construction should succeed");
        let user_emb = [1.0_f32, 0.0];
        let item_embs = [2.0_f32, 0.0, 0.0, 3.0, 4.0, 0.0];
        s.update_scores(0, &user_emb, &item_embs)
            .expect("update_scores should succeed");

        let eps = 1e-6_f32;
        assert!(
            (s.scores[0] - 2.0).abs() < eps,
            "item0 score={}",
            s.scores[0]
        );
        assert!(
            (s.scores[1] - 0.0).abs() < eps,
            "item1 score={}",
            s.scores[1]
        );
        assert!(
            (s.scores[2] - 4.0).abs() < eps,
            "item2 score={}",
            s.scores[2]
        );
    }

    #[test]
    fn sample_never_returns_positive_item() {
        // item0 has the highest score but is the positive; sampled item must != 0.
        let mut s = HardNegSampler::new(1, 6).expect("construction should succeed");
        s.scores = vec![10.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0];
        let positives: BTreeSet<usize> = std::iter::once(0).collect();
        let mut rng = make_rng();
        for _ in 0..20 {
            let neg = s
                .sample(0, &positives, &mut rng)
                .expect("sample should succeed");
            assert_ne!(neg, 0, "sampled negative must not equal positive item 0");
        }
    }

    #[test]
    fn sample_always_from_hardest_when_pool_size_one() {
        // n_items=6, positives={0}.
        // Non-positive items: 1(5.0), 2(4.0), 3(3.0), 4(2.0), 5(1.0) → 5 candidates.
        // top_k = ceil(5 × 0.20) = 1 → pool = {item1 (score 5.0)}.
        // Any RNG call returns pool[0] = item1.
        let mut s = HardNegSampler::new(1, 6).expect("construction should succeed");
        s.scores = vec![10.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0];
        let positives: BTreeSet<usize> = std::iter::once(0).collect();
        let mut rng = make_rng();
        for _ in 0..10 {
            let neg = s
                .sample(0, &positives, &mut rng)
                .expect("sample should succeed");
            assert_eq!(
                neg, 1,
                "top-20% of 5 candidates is exactly {{item1}}; got item {neg}"
            );
        }
    }

    #[test]
    fn sample_deterministic_with_fixed_seed() {
        // Two RNGs seeded identically must produce the same sequence of samples.
        let mut s = HardNegSampler::new(1, 20).expect("construction should succeed");
        // Scores: item i → (20 - i), so item0 has score 20 (highest).
        for i in 0..20 {
            s.scores[i] = (20 - i) as f32;
        }
        let positives: BTreeSet<usize> = std::iter::once(0).collect();
        let mut rng_a = LcgRng::new(77);
        let mut rng_b = LcgRng::new(77);
        for _ in 0..8 {
            let a = s
                .sample(0, &positives, &mut rng_a)
                .expect("sample A should succeed");
            let b = s
                .sample(0, &positives, &mut rng_b)
                .expect("sample B should succeed");
            assert_eq!(a, b, "same seed must yield identical samples");
        }
    }

    #[test]
    fn no_negative_available_when_all_items_positive() {
        let mut s = HardNegSampler::new(1, 3).expect("construction should succeed");
        s.scores = vec![1.0_f32, 2.0, 3.0];
        let positives: BTreeSet<usize> = [0, 1, 2].iter().copied().collect();
        let mut rng = make_rng();
        assert!(matches!(
            s.sample(0, &positives, &mut rng),
            Err(RecsysError::NoNegativeAvailable { .. })
        ));
    }
}
