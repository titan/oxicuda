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
