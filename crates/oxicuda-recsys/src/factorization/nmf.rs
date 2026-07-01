use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-10;

pub struct Nmf {
    pub n_users: usize,
    pub n_items: usize,
    pub dim: usize,
    /// W matrix: [n_users x dim]
    pub w: Vec<f32>,
    /// H matrix: [dim x n_items]
    pub h: Vec<f32>,
}

impl Nmf {
    pub fn fit(
        data: &[(usize, usize, f32)],
        n_users: usize,
        n_items: usize,
        dim: usize,
        n_iters: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if data.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        if n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: n_users });
        }
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: dim });
        }

        // Initialize W and H with small positive values
        let mut w: Vec<f32> = (0..n_users * dim)
            .map(|_| rng.next_f32() * 0.1 + 0.01)
            .collect();
        let mut h: Vec<f32> = (0..dim * n_items)
            .map(|_| rng.next_f32() * 0.1 + 0.01)
            .collect();

        // Build dense V matrix from sparse data
        let mut v = vec![0.0_f32; n_users * n_items];
        for &(u, i, r) in data {
            if u < n_users && i < n_items {
                v[u * n_items + i] = r;
            }
        }

        for _ in 0..n_iters {
            // Update H: H <- H * (W^T V) / (W^T W H + eps)
            // W^T W: [dim x dim]
            let wtw = matmul_t1(&w, &w, n_users, dim, dim);
            // W^T V: [dim x n_items]
            let wtv = matmul_t1(&w, &v, n_users, dim, n_items);
            // W^T W H: [dim x n_items]
            let wtwh = matmul(&wtw, &h, dim, dim, n_items);

            for (k, (h_k, (wtv_k, wtwh_k))) in
                h.iter_mut().zip(wtv.iter().zip(wtwh.iter())).enumerate()
            {
                let _ = k;
                *h_k *= (*wtv_k + EPS) / (*wtwh_k + EPS);
            }

            // Update W: W <- W * (V H^T) / (W H H^T + eps)
            // H H^T: [dim x dim]
            let hht = matmul_t2(&h, &h, dim, n_items, dim);
            // V H^T: [n_users x dim]
            let vht = matmul_t2(&v, &h, n_users, n_items, dim);
            // W H H^T: [n_users x dim]
            let whht = matmul(&w, &hht, n_users, dim, dim);

            for (w_k, (vht_k, whht_k)) in w.iter_mut().zip(vht.iter().zip(whht.iter())) {
                *w_k *= (*vht_k + EPS) / (*whht_k + EPS);
            }
        }

        Ok(Self {
            n_users,
            n_items,
            dim,
            w,
            h,
        })
    }

    pub fn score(&self, user: usize, item: usize) -> RecsysResult<f32> {
        if user >= self.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        if item >= self.n_items {
            return Err(RecsysError::UnknownItem { id: item });
        }
        let d = self.dim;
        // W[user, :] . H[:, item]
        let dot = self.w[user * d..(user + 1) * d]
            .iter()
            .enumerate()
            .map(|(k, &wk)| wk * self.h[k * self.n_items + item])
            .sum();
        Ok(dot)
    }
}

/// A (m x k) * B (k x n) -> C (m x n)
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            c[row * n + col] = (0..k).map(|p| a[row * k + p] * b[p * n + col]).sum();
        }
    }
    c
}

/// A^T (k x m) * B (m x n) -> C (k x n), A is (m x k)
fn matmul_t1(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; k * n];
    for row in 0..k {
        for col in 0..n {
            c[row * n + col] = (0..m).map(|p| a[p * k + row] * b[p * n + col]).sum();
        }
    }
    c
}

/// A (m x k) * B^T (k x n), B is (n x k) -> C (m x n)
fn matmul_t2(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            c[row * n + col] = (0..k).map(|p| a[row * k + p] * b[col * k + p]).sum();
        }
    }
    c
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Frobenius squared error ||V_dense - W H||_F^2.
    fn frob_sq_err(model: &Nmf, v_dense: &[f32]) -> f32 {
        let mut err = 0.0_f32;
        for u in 0..model.n_users {
            for i in 0..model.n_items {
                let v_ui = v_dense[u * model.n_items + i];
                let pred = model.score(u, i).expect("score ok");
                let d = v_ui - pred;
                err += d * d;
            }
        }
        err
    }

    /// Build a dense [n_users × n_items] rank-1 matrix: V[u,i] = u_vec[u]*v_vec[i].
    fn dense_rank1(u_vec: &[f32], v_vec: &[f32]) -> Vec<f32> {
        let n = u_vec.len();
        let m = v_vec.len();
        (0..n)
            .flat_map(|u| (0..m).map(move |i| u_vec[u] * v_vec[i]))
            .collect()
    }

    #[test]
    fn nmf_empty_data_returns_err() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Nmf::fit(&[], 3, 3, 2, 10, &mut rng),
            Err(RecsysError::EmptyInteraction)
        ));
    }

    #[test]
    fn nmf_zero_dim_returns_err() {
        let mut rng = LcgRng::new(2);
        let data = vec![(0usize, 0usize, 1.0_f32)];
        assert!(matches!(
            Nmf::fit(&data, 2, 2, 0, 5, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn nmf_factors_nonnegative_after_many_updates() {
        // The defining invariant of NMF multiplicative updates: W,H >= 0 always.
        let mut rng = LcgRng::new(42);
        let data: Vec<(usize, usize, f32)> = vec![
            (0, 0, 1.0),
            (0, 1, 0.5),
            (1, 0, 0.3),
            (1, 2, 0.8),
            (2, 1, 0.6),
            (2, 2, 1.0),
        ];
        let model = Nmf::fit(&data, 3, 3, 4, 100, &mut rng).expect("fit should succeed");
        for (idx, &w) in model.w.iter().enumerate() {
            assert!(w >= 0.0, "W[{idx}] = {w} is negative");
        }
        for (idx, &h) in model.h.iter().enumerate() {
            assert!(h >= 0.0, "H[{idx}] = {h} is negative");
        }
    }

    #[test]
    fn nmf_reconstruction_error_decreases_with_more_iterations() {
        // Identical seed → identical W,H init → more iters must yield lower
        // (or equal) Frobenius error (Lee & Seung multiplicative-update guarantee).
        let u_vec = [0.8_f32, 0.5, 1.0, 0.3];
        let v_vec = [1.0_f32, 0.6, 0.4, 0.9];
        let dense = dense_rank1(&u_vec, &v_vec);
        let n_users = 4;
        let n_items = 4;
        // Build data from raw arrays (Copy) to avoid moving `dense` into closure.
        let data: Vec<(usize, usize, f32)> = (0..n_users)
            .flat_map(|u| (0..n_items).map(move |i| (u, i, u_vec[u] * v_vec[i])))
            .collect();

        let milestones = [0usize, 5, 20, 80];
        let mut prev_err = f32::MAX;
        for &n_iters in &milestones {
            let mut rng = LcgRng::new(111);
            let model = Nmf::fit(&data, n_users, n_items, 2, n_iters, &mut rng)
                .expect("fit should succeed");
            let err = frob_sq_err(&model, &dense);
            assert!(
                err <= prev_err + 1e-3,
                "error increased from {prev_err:.6} to {err:.6} at n_iters={n_iters}"
            );
            prev_err = err;
        }
    }

    #[test]
    fn nmf_rank1_input_converges_to_small_reconstruction_error() {
        // A rank-1 non-negative matrix is exactly factorisable by NMF with
        // dim=2.  After enough iterations the reconstruction should be close.
        let u_vec = [1.0_f32, 0.7, 0.5];
        let v_vec = [0.8_f32, 1.0, 0.6];
        let dense = dense_rank1(&u_vec, &v_vec);
        let n_users = 3;
        let n_items = 3;
        // Build data from raw arrays (Copy) to avoid moving `dense` into closure.
        let data: Vec<(usize, usize, f32)> = (0..n_users)
            .flat_map(|u| (0..n_items).map(move |i| (u, i, u_vec[u] * v_vec[i])))
            .collect();

        let mut rng = LcgRng::new(7);
        let model =
            Nmf::fit(&data, n_users, n_items, 2, 400, &mut rng).expect("fit should succeed");
        let err = frob_sq_err(&model, &dense);
        assert!(
            err < 0.05,
            "rank-1 NMF should converge to small error, got {err:.6}"
        );
    }

    #[test]
    fn nmf_scores_finite_and_nonneg_for_all_pairs() {
        // W >= 0 and H >= 0 implies WH >= 0 entry-wise.
        let mut rng = LcgRng::new(13);
        let data: Vec<(usize, usize, f32)> = vec![(0, 0, 1.0), (1, 1, 0.5), (2, 2, 0.8)];
        let model = Nmf::fit(&data, 3, 3, 3, 20, &mut rng).expect("fit should succeed");
        for u in 0..3 {
            for i in 0..3 {
                let s = model.score(u, i).expect("score ok");
                assert!(s.is_finite(), "score({u},{i}) = {s} not finite");
                assert!(s >= -1e-7, "score({u},{i}) = {s} < 0 (W,H >= 0 so WH >= 0)");
            }
        }
    }
}
