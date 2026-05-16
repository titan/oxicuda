//! Modified Locally Linear Embedding (MLLE), Zhang & Wang 2007.
//!
//! Uses multiple weight vectors per neighbourhood (the small eigenvectors of the
//! Gram matrix `Z Z^T`) to construct a more robust local geometry. The resulting
//! sparse matrix `M` follows the same form as in LLE.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::neighbor::knn_brute::knn_brute;

/// MLLE result.
pub struct MlleResult {
    pub embedding: Vec<f64>,
    pub eigenvalues: Vec<f64>,
}

/// Fit MLLE.
pub fn mlle_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_neighbors: usize,
    n_components: usize,
    reg: f64,
) -> ManifoldResult<MlleResult> {
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
    if n_neighbors <= n_components {
        return Err(ManifoldError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: format!("must exceed n_components={n_components}"),
        });
    }
    let n = n_samples;
    let k = n_neighbors;
    let (idx, _d) = knn_brute(x, n, dim, k)?;
    // Construct M directly from per-neighbourhood multi-weight projections.
    let mut m = vec![0.0; n * n];
    let mut z = vec![0.0; k * dim];
    let mut cov = vec![0.0; k * k];
    let s = k - n_components;
    let s_actual = s.max(1);
    for i in 0..n {
        for jj in 0..k {
            let nb = idx[i * k + jj];
            for d in 0..dim {
                z[jj * dim + d] = x[nb * dim + d] - x[i * dim + d];
            }
        }
        for a in 0..k {
            for b in a..k {
                let mut acc = 0.0;
                for d in 0..dim {
                    acc += z[a * dim + d] * z[b * dim + d];
                }
                cov[a * k + b] = acc;
                cov[b * k + a] = acc;
            }
        }
        // Regularise
        let mut tr = 0.0;
        for j in 0..k {
            tr += cov[j * k + j];
        }
        let alpha = (reg * tr / k as f64).max(1.0e-10);
        for j in 0..k {
            cov[j * k + j] += alpha;
        }
        let (eigvals, eigvecs) = jacobi_eigh(&cov, k)?;
        // Sort ascending to take smallest s_actual eigvecs as the null-space basis
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| {
            eigvals[a]
                .partial_cmp(&eigvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Use eigvecs corresponding to smallest s_actual eigvalues as orthonormal weight basis.
        // Per Zhang-Wang, also use a weight that sums to 1 (LLE-style). Average across the basis.
        // Construct H matrix: V (k x s_actual). Build (I - W^T) H H^T (I - W) contribution.
        // Simplified: aggregate sum w_alpha * w_alpha^T from each basis column, plus normalise sums.
        let mut wmat = vec![0.0; k * s_actual];
        for col in 0..s_actual {
            let ce = order[col];
            // Normalise so sum-of-entries is 1 if possible, else use raw
            let mut sum = 0.0;
            for r in 0..k {
                sum += eigvecs[r * k + ce];
            }
            if sum.abs() < 1.0e-12 {
                sum = 1.0;
            }
            for r in 0..k {
                wmat[r * s_actual + col] = eigvecs[r * k + ce] / sum;
            }
        }
        // For each weight col, accumulate contribution into M
        for col in 0..s_actual {
            // Build (1, -w_1, ..., -w_k) vector; here it's the i-th row + neighbours
            // M[i,i] += 1, M[i, nb_j] -= w_j, M[nb_j, i] -= w_j, M[nb_a, nb_b] += w_a w_b
            m[i * n + i] += 1.0;
            for j in 0..k {
                let nb = idx[i * k + j];
                let w = wmat[j * s_actual + col];
                m[i * n + nb] -= w;
                m[nb * n + i] -= w;
                m[nb * n + nb] += w * w;
                for j2 in 0..k {
                    if j2 == j {
                        continue;
                    }
                    let nb2 = idx[i * k + j2];
                    let w2 = wmat[j2 * s_actual + col];
                    m[nb * n + nb2] += w * w2;
                }
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    let (eigvals, eigvecs) = jacobi_eigh(&m, n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigvals[a]
            .partial_cmp(&eigvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut embedding = vec![0.0; n * n_components];
    let mut emb_eigvals = vec![0.0; n_components];
    for c in 0..n_components {
        let col = order[c + 1];
        emb_eigvals[c] = eigvals[col];
        for r in 0..n {
            embedding[r * n_components + c] = eigvecs[r * n + col];
        }
    }
    Ok(MlleResult {
        embedding,
        eigenvalues: emb_eigvals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlle_runs() {
        let n = 10;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            let t = i as f64 * 0.2;
            x[i * dim] = t.cos();
            x[i * dim + 1] = t.sin();
            x[i * dim + 2] = t;
        }
        let r = mlle_fit(&x, n, dim, 4, 2, 1e-3).expect("ok");
        assert_eq!(r.embedding.len(), n * 2);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }
}
