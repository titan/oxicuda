use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

pub struct Als {
    pub n_users: usize,
    pub n_items: usize,
    pub dim: usize,
    pub user_emb: Vec<f32>,
    pub item_emb: Vec<f32>,
    pub lambda: f32,
}

impl Als {
    pub fn new(
        n_users: usize,
        n_items: usize,
        dim: usize,
        lambda: f32,
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
            lambda,
        })
    }

    pub fn fit(
        &mut self,
        interactions: &[(usize, usize, f32)],
        n_iters: usize,
    ) -> RecsysResult<()> {
        if interactions.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        const ALPHA: f32 = 40.0;
        let d = self.dim;

        for _iter in 0..n_iters {
            // Update user embeddings
            for u in 0..self.n_users {
                let user_ints: Vec<(usize, f32)> = interactions
                    .iter()
                    .filter(|&&(uid, _, _)| uid == u)
                    .map(|&(_, iid, r)| (iid, r))
                    .collect();

                let mut a = vec![0.0_f32; d * d];
                let mut b = vec![0.0_f32; d];

                // Regularizer on diagonal
                for k in 0..d {
                    a[k * d + k] = self.lambda;
                }

                for (iid, r) in &user_ints {
                    let c = 1.0 + ALPHA * r;
                    let e = &self.item_emb[iid * d..(iid + 1) * d];
                    for (ki, &ei) in e.iter().enumerate() {
                        for (kj, &ej) in e.iter().enumerate() {
                            a[ki * d + kj] += c * ei * ej;
                        }
                        b[ki] += c * ei;
                    }
                }

                let solution = gauss_jordan(&a, &b, d)?;
                self.user_emb[u * d..(u + 1) * d].copy_from_slice(&solution);
            }

            // Update item embeddings
            for i in 0..self.n_items {
                let item_ints: Vec<(usize, f32)> = interactions
                    .iter()
                    .filter(|&&(_, iid, _)| iid == i)
                    .map(|&(uid, _, r)| (uid, r))
                    .collect();

                let mut a = vec![0.0_f32; d * d];
                let mut b = vec![0.0_f32; d];

                for k in 0..d {
                    a[k * d + k] = self.lambda;
                }

                for (uid, r) in &item_ints {
                    let c = 1.0 + ALPHA * r;
                    let e = &self.user_emb[uid * d..(uid + 1) * d];
                    for (ki, &ei) in e.iter().enumerate() {
                        for (kj, &ej) in e.iter().enumerate() {
                            a[ki * d + kj] += c * ei * ej;
                        }
                        b[ki] += c * ei;
                    }
                }

                let solution = gauss_jordan(&a, &b, d)?;
                self.item_emb[i * d..(i + 1) * d].copy_from_slice(&solution);
            }
        }
        Ok(())
    }

    pub fn score(&self, user: usize, item: usize) -> RecsysResult<f32> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        if item >= self.n_items {
            return Err(RecsysError::UnknownItem { id: item });
        }
        let d = self.dim;
        let dot = self.user_emb[user * d..(user + 1) * d]
            .iter()
            .zip(self.item_emb[item * d..(item + 1) * d].iter())
            .map(|(&u, &i)| u * i)
            .sum();
        Ok(dot)
    }

    pub fn top_k(&self, user: usize, k: usize) -> RecsysResult<Vec<usize>> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        if k == 0 || k > self.n_items {
            return Err(RecsysError::InvalidK { k, n: self.n_items });
        }
        let mut scores: Vec<(usize, f32)> = (0..self.n_items)
            .map(|item| {
                let s = self.score(user, item).unwrap_or(f32::NEG_INFINITY);
                (item, s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores.iter().take(k).map(|&(id, _)| id).collect())
    }
}

/// Gauss-Jordan elimination to solve A x = b for x (dim x dim system).
fn gauss_jordan(a: &[f32], b: &[f32], d: usize) -> RecsysResult<Vec<f32>> {
    // Build augmented matrix [A | b]
    let mut aug: Vec<f32> = vec![0.0; d * (d + 1)];
    for row in 0..d {
        for col in 0..d {
            aug[row * (d + 1) + col] = a[row * d + col];
        }
        aug[row * (d + 1) + d] = b[row];
    }

    for col in 0..d {
        // Find pivot
        let pivot_row = (col..d)
            .max_by(|&r1, &r2| {
                aug[r1 * (d + 1) + col]
                    .abs()
                    .partial_cmp(&aug[r2 * (d + 1) + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| RecsysError::Internal {
                msg: "no pivot row".into(),
            })?;

        // Swap full rows col and pivot_row in the augmented matrix.
        for k in 0..=(d) {
            let tmp_col = aug[col * (d + 1) + k];
            let tmp_piv = aug[pivot_row * (d + 1) + k];
            aug[col * (d + 1) + k] = tmp_piv;
            aug[pivot_row * (d + 1) + k] = tmp_col;
        }

        let piv = aug[col * (d + 1) + col];
        if piv.abs() < 1e-12 {
            continue;
        }
        let inv_piv = 1.0 / piv;
        for k in 0..=(d) {
            aug[col * (d + 1) + k] *= inv_piv;
        }

        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = aug[row * (d + 1) + col];
            if factor.abs() < 1e-15 {
                continue;
            }
            for k in 0..=(d) {
                let val = factor * aug[col * (d + 1) + k];
                aug[row * (d + 1) + k] -= val;
            }
        }
    }

    Ok((0..d).map(|row| aug[row * (d + 1) + d]).collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// WMF objective: Σ_{observed} c*(1-u·v)² + λ(||U||²+||V||²).
    /// Each ALS alternating step solves its subproblem exactly, so this must
    /// be non-increasing across iterations.
    fn wmf_obj(model: &Als, interactions: &[(usize, usize, f32)]) -> f32 {
        const ALPHA: f32 = 40.0;
        let d = model.dim;
        let mut obj = 0.0_f32;
        for &(u, i, r) in interactions {
            let c = 1.0 + ALPHA * r;
            let pred = model.score(u, i).expect("score ok");
            obj += c * (1.0 - pred).powi(2);
        }
        for chunk in model.user_emb.chunks(d) {
            obj += model.lambda * chunk.iter().map(|&v| v * v).sum::<f32>();
        }
        for chunk in model.item_emb.chunks(d) {
            obj += model.lambda * chunk.iter().map(|&v| v * v).sum::<f32>();
        }
        obj
    }

    #[test]
    fn new_valid_params_succeeds() {
        let mut rng = LcgRng::new(42);
        let model = Als::new(4, 6, 8, 0.01, &mut rng).expect("new should succeed");
        assert_eq!(model.user_emb.len(), 4 * 8);
        assert_eq!(model.item_emb.len(), 6 * 8);
        assert_eq!(model.n_users, 4);
        assert_eq!(model.n_items, 6);
    }

    #[test]
    fn new_zero_dim_returns_err() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Als::new(4, 6, 0, 0.01, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn new_zero_users_returns_err() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Als::new(0, 6, 4, 0.01, &mut rng),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn fit_empty_interactions_returns_err() {
        let mut rng = LcgRng::new(2);
        let mut model = Als::new(4, 6, 4, 0.01, &mut rng).expect("new should succeed");
        assert!(matches!(
            model.fit(&[], 1),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn score_out_of_bounds_returns_err() {
        let mut rng = LcgRng::new(3);
        let model = Als::new(3, 5, 4, 0.01, &mut rng).expect("new should succeed");
        assert!(matches!(
            model.score(3, 0),
            Err(RecsysError::UnknownUser { .. })
        ));
        assert!(matches!(
            model.score(0, 5),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn wmf_objective_non_increasing_across_als_steps() {
        // ALS exactly solves each subproblem so the global objective must
        // be non-increasing per alternating step.
        let interactions: Vec<(usize, usize, f32)> = vec![
            (0, 0, 1.0),
            (0, 1, 0.5),
            (1, 0, 0.3),
            (1, 2, 1.0),
            (2, 1, 0.8),
            (2, 3, 0.6),
        ];
        let mut rng = LcgRng::new(2024);
        let mut model = Als::new(3, 4, 3, 0.01, &mut rng).expect("new should succeed");
        let mut prev = wmf_obj(&model, &interactions);
        for step in 1..=10usize {
            model.fit(&interactions, 1).expect("fit should succeed");
            let obj = wmf_obj(&model, &interactions);
            assert!(
                obj <= prev + 1e-3,
                "WMF objective increased at step {step}: {prev:.6} -> {obj:.6}"
            );
            prev = obj;
        }
    }

    #[test]
    fn interacted_items_score_higher_than_unobserved_after_convergence() {
        // User 0 sees items {0,1}; user 1 sees items {2,3}.
        // After convergence, interacted items must outscore unobserved ones.
        let interactions: Vec<(usize, usize, f32)> =
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)];
        let mut rng = LcgRng::new(99);
        let mut model = Als::new(2, 4, 2, 0.001, &mut rng).expect("new should succeed");
        model.fit(&interactions, 30).expect("fit should succeed");

        let s00 = model.score(0, 0).expect("score ok");
        let s01 = model.score(0, 1).expect("score ok");
        let s02 = model.score(0, 2).expect("score ok");
        let s03 = model.score(0, 3).expect("score ok");
        let obs_avg = (s00 + s01) / 2.0;
        let unobs_avg = (s02 + s03) / 2.0;
        assert!(
            obs_avg > unobs_avg,
            "observed avg {obs_avg:.4} should exceed unobserved avg {unobs_avg:.4}"
        );
    }

    #[test]
    fn top_k_length_and_descending_order() {
        let mut rng = LcgRng::new(55);
        let interactions: Vec<(usize, usize, f32)> = vec![(0, 0, 1.0), (0, 2, 0.5), (1, 1, 1.0)];
        let mut model = Als::new(2, 4, 4, 0.01, &mut rng).expect("new should succeed");
        model.fit(&interactions, 3).expect("fit should succeed");
        let top = model.top_k(0, 3).expect("top_k should succeed");
        assert_eq!(top.len(), 3);
        for &id in &top {
            assert!(id < 4, "item id {id} out of bounds");
        }
        let scores: Vec<f32> = top
            .iter()
            .map(|&i| model.score(0, i).expect("score ok"))
            .collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-6,
                "top_k not sorted descending: {} < {}",
                w[0],
                w[1]
            );
        }
    }
}
