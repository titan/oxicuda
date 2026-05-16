//! Laplacian Eigenmaps (Belkin & Niyogi, 2003).
//!
//! 1. Build kNN graph (symmetric).
//! 2. Edge weight `W_ij = exp(-||x_i - x_j||^2 / sigma)` for adjacent pairs.
//! 3. Degree matrix `D_ii = sum_j W_ij`.
//! 4. Generalised eigenproblem `L v = lambda D v`, where `L = D - W`.
//! 5. Drop the smallest (constant) eigenvector and take the next `d`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::neighbor::knn_brute::knn_brute;

/// Laplacian Eigenmaps result.
pub struct LapEigResult {
    pub embedding: Vec<f64>,
    pub eigenvalues: Vec<f64>,
}

/// Fit Laplacian Eigenmaps.
pub fn laplacian_eigenmaps_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_neighbors: usize,
    n_components: usize,
    sigma: f64,
) -> ManifoldResult<LapEigResult> {
    if n_samples == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n_samples * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, dim],
            got: vec![x.len()],
        });
    }
    if n_components == 0 || n_components + 1 > n_samples {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n_samples}"),
        });
    }
    let n = n_samples;
    let k = n_neighbors;
    let (idx, d2) = knn_brute(x, n, dim, k)?;
    let mut w_mat: Vec<f64> = vec![0.0; n * n];
    let two_sigma2 = 2.0 * sigma * sigma;
    for i in 0..n {
        for jj in 0..k {
            let nb = idx[i * k + jj];
            let val = (-d2[i * k + jj] / two_sigma2).exp();
            // Make symmetric
            w_mat[i * n + nb] = w_mat[i * n + nb].max(val);
            w_mat[nb * n + i] = w_mat[nb * n + i].max(val);
        }
    }
    let mut d: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += w_mat[i * n + j];
        }
        d[i] = s;
    }
    // Normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    let mut l_sym = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let denom = (d[i].max(1e-14) * d[j].max(1e-14)).sqrt();
            l_sym[i * n + j] = -w_mat[i * n + j] / denom;
        }
        l_sym[i * n + i] += 1.0;
    }
    // Symmetrise
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (l_sym[i * n + j] + l_sym[j * n + i]);
            l_sym[i * n + j] = v;
            l_sym[j * n + i] = v;
        }
    }
    let (eigvals, eigvecs) = jacobi_eigh(&l_sym, n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigvals[a]
            .partial_cmp(&eigvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Embedding: u_i = v_i / sqrt(D_ii), drop the first (constant) eigenvector.
    let mut embedding = vec![0.0; n * n_components];
    let mut emb_eigvals = vec![0.0; n_components];
    for c in 0..n_components {
        let col = order[c + 1];
        emb_eigvals[c] = eigvals[col];
        for r in 0..n {
            let inv_sqrt = 1.0 / d[r].max(1e-14).sqrt();
            embedding[r * n_components + c] = eigvecs[r * n + col] * inv_sqrt;
        }
    }
    Ok(LapEigResult {
        embedding,
        eigenvalues: emb_eigvals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplacian_eigenmaps_runs() {
        let n = 10;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = 0.5 * (i as f64);
        }
        let r = laplacian_eigenmaps_fit(&x, n, dim, 3, 1, 1.0).expect("ok");
        assert_eq!(r.embedding.len(), n);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }
}
