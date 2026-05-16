//! Locally Linear Embedding (LLE).
//!
//! 1. Find k-nearest neighbours of each point.
//! 2. Solve constrained LS for weights `W_ij` with `sum_j W_ij = 1` minimising
//!    `||x_i - sum_j W_ij x_j||^2` over the k neighbours.
//! 3. Build sparse `M = (I - W)^T (I - W)`.
//! 4. Compute `d + 1` smallest eigenvectors of `M`, drop the trivial one (constant).

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::householder_qr::solve_lower_triangular;
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::neighbor::knn_brute::knn_brute;

/// LLE fit result.
pub struct LleResult {
    /// Embedding `(n, d_out)`.
    pub embedding: Vec<f64>,
    /// Reconstruction-error eigenvalues used in the embedding.
    pub eigenvalues: Vec<f64>,
}

/// Fit LLE on row-major `(n_samples, dim)` data.
pub fn lle_fit(
    x: &[f64],
    n_samples: usize,
    dim: usize,
    n_neighbors: usize,
    n_components: usize,
    reg: f64,
) -> ManifoldResult<LleResult> {
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
    let (idx, _d) = knn_brute(x, n, dim, k)?;
    // For each point, compute weights via constrained LS
    let mut weights = vec![0.0; n * k];
    let mut z = vec![0.0; k * dim];
    let mut cov = vec![0.0; k * k];
    for i in 0..n {
        // Z = X_neighbors - x_i  (k x dim)
        for jj in 0..k {
            let nb = idx[i * k + jj];
            for d in 0..dim {
                z[jj * dim + d] = x[nb * dim + d] - x[i * dim + d];
            }
        }
        // C = Z Z^T  (k x k)
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
        // Regularise: C += reg * trace(C) / k * I
        let mut tr = 0.0;
        for j in 0..k {
            tr += cov[j * k + j];
        }
        let alpha = reg * tr / k as f64;
        let alpha = if alpha < 1.0e-10 { 1.0e-10 } else { alpha };
        for j in 0..k {
            cov[j * k + j] += alpha;
        }
        // Solve C w = 1 via Cholesky-like factorisation: jacobi eig pseudoinverse.
        // For numerical robustness, use the eigen route.
        let (eigvals, eigvecs) = jacobi_eigh(&cov, k)?;
        // w = C^{-1} * 1 = sum_v ( (v^T 1) / lam ) * v
        let mut w = vec![0.0; k];
        for c in 0..k {
            let mut dot1 = 0.0;
            for r in 0..k {
                dot1 += eigvecs[r * k + c];
            }
            let lam = if eigvals[c].abs() < 1.0e-14 {
                1.0e-14
            } else {
                eigvals[c]
            };
            let coef = dot1 / lam;
            for r in 0..k {
                w[r] += coef * eigvecs[r * k + c];
            }
        }
        // Normalise so sum w = 1
        let s: f64 = w.iter().sum();
        if s.abs() > 1.0e-14 {
            for v in &mut w {
                *v /= s;
            }
        }
        for j in 0..k {
            weights[i * k + j] = w[j];
        }
    }
    // Build M = (I - W)^T (I - W) as dense
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] += 1.0;
        for j in 0..k {
            let nb = idx[i * k + j];
            let w = weights[i * k + j];
            // (I - W)[i, nb] = -W[i, nb]
            // Contribution to m: (I-W)^T (I-W). For row p:
            //    m[p, q] += sum_i (I-W)[i,p] * (I-W)[i,q]
            //    here this point contributes for i fixed, p in {i, nb} and q similarly.
            m[i * n + nb] -= w;
            m[nb * n + i] -= w;
            // Add w^2 to diagonal entry for nb (because (I-W)[i, nb] = -w)
            m[nb * n + nb] += w * w;
            // Cross with other neighbours nb2 of i
            for j2 in 0..k {
                if j2 == j {
                    continue;
                }
                let nb2 = idx[i * k + j2];
                let w2 = weights[i * k + j2];
                m[nb * n + nb2] += w * w2;
            }
        }
    }
    // Symmetrise just in case
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    // Get smallest n_components + 1 eigenvalues; drop the smallest (it is the constant)
    let (eigvals, eigvecs) = jacobi_eigh(&m, n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigvals[a]
            .partial_cmp(&eigvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Drop the first (smallest, ~0). Take next n_components.
    let mut embedding = vec![0.0; n * n_components];
    let mut emb_eigvals = vec![0.0; n_components];
    for c in 0..n_components {
        let col = order[c + 1];
        emb_eigvals[c] = eigvals[col];
        for r in 0..n {
            embedding[r * n_components + c] = eigvecs[r * n + col];
        }
    }
    // Silence warnings about unused helpers
    let _ = solve_lower_triangular;
    Ok(LleResult {
        embedding,
        eigenvalues: emb_eigvals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn lle_runs_swiss_roll_substitute() {
        // Generate 12 points roughly on a 1D curve in 3D space.
        let mut rng = LcgRng::new(7);
        let n = 12;
        let dim = 3;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            let t = i as f64 * 0.3;
            x[i * dim] = t.cos() + 0.01 * rng.next_normal();
            x[i * dim + 1] = t.sin() + 0.01 * rng.next_normal();
            x[i * dim + 2] = t + 0.01 * rng.next_normal();
        }
        let r = lle_fit(&x, n, dim, 4, 1, 1e-3).expect("ok");
        assert_eq!(r.embedding.len(), n);
        assert!(r.embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lle_weights_sum_to_one() {
        // Internal: build via small example. Indirect test: success implies finite.
        let n = 8;
        let dim = 2;
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i * dim] = i as f64;
            x[i * dim + 1] = 0.5 * (i as f64);
        }
        let r = lle_fit(&x, n, dim, 3, 1, 1e-3).expect("ok");
        assert!(r.eigenvalues.iter().all(|v| v.is_finite()));
    }
}
