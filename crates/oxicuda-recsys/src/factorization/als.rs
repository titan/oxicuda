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

        aug.swap(col * (d + 1), pivot_row * (d + 1)); // swap full rows
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
