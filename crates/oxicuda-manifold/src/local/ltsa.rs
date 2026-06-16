//! Local Tangent Space Alignment (LTSA), Zhang & Zha 2005.
//!
//! LTSA recovers a global low-dimensional parametrisation by aligning the local
//! tangent-space coordinates of overlapping neighbourhoods. For every point `i`:
//!
//! 1. Gather its neighbourhood `N_i` (the point itself plus its `k - 1` nearest
//!    neighbours, `k` members total) and centre it (`X_i`, `k x D`).
//! 2. Estimate the local tangent coordinates via local PCA: the top-`d` left singular
//!    vectors `Q_i` of the centred neighbourhood (`k x d`, computed from the `k x k`
//!    Gram matrix).
//! 3. Form `G_i = [ (1/sqrt(k)) 1_k | Q_i ]` (`k x (d+1)`) and the local alignment
//!    contribution `W_i = I_k - G_i G_i^T` (`k x k`).
//! 4. Accumulate the global alignment matrix `B[N_i, N_i] += W_i`.
//! 5. The embedding is the `d` eigenvectors of `B` for the smallest nonzero
//!    eigenvalues, skipping the single null vector (the constant function).
//!
//! For data sampled from an affine image of a `d`-dimensional parameter space, LTSA
//! recovers the parameters up to an affine transform essentially exactly.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::neighbor::knn_brute::knn_brute;

/// Fit LTSA on row-major `(n_samples, n_features)` data.
///
/// Returns the embedding as a row-major `n_samples x n_components` matrix.
///
/// `n_neighbors` must exceed `n_components` (a tangent basis of `d + 1` columns must
/// fit inside the neighbourhood) and must not exceed `n_samples`.
pub fn ltsa(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    n_neighbors: usize,
) -> ManifoldResult<Vec<f64>> {
    if n_samples == 0 || n_features == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n_samples * n_features {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![data.len()],
        });
    }
    if n_components == 0 || n_components + 1 > n_samples {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must be in 1..{n_samples}"),
        });
    }
    if n_components > n_features {
        return Err(ManifoldError::InvalidParameter {
            name: "n_components".into(),
            reason: format!("must not exceed n_features={n_features}, got {n_components}"),
        });
    }
    let d = n_components;
    // The neighbourhood must accommodate the constant + d tangent directions.
    let min_k = d + 2;
    if n_neighbors < min_k {
        return Err(ManifoldError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: format!(
                "LTSA requires n_neighbors >= n_components + 2 = {min_k}, got {n_neighbors}"
            ),
        });
    }
    if n_neighbors > n_samples {
        return Err(ManifoldError::KNeighborsTooLarge {
            k: n_neighbors,
            n: n_samples,
        });
    }
    let n = n_samples;
    let dim = n_features;
    let k = n_neighbors;

    // Neighbourhood = the point itself + its (k - 1) nearest neighbours.
    let (knn_idx, _knn_d) = knn_brute(data, n, dim, k - 1)?;
    let mut neigh = vec![0usize; n * k];
    for i in 0..n {
        neigh[i * k] = i;
        for jj in 0..(k - 1) {
            neigh[i * k + 1 + jj] = knn_idx[i * (k - 1) + jj];
        }
    }

    // Global alignment matrix B (n x n), symmetric.
    let mut b = vec![0.0f64; n * n];

    // Reusable per-neighbourhood scratch.
    let mut z = vec![0.0f64; k * dim]; // centred neighbourhood (k x dim)
    let mut gram = vec![0.0f64; k * k]; // Z Z^T (k x k)
    // g holds [ (1/sqrt(k)) 1 | Q_i ]  (k x (d+1)).
    let cols = d + 1;
    let mut g = vec![0.0f64; k * cols];

    let inv_sqrt_k = 1.0 / (k as f64).sqrt();

    for i in 0..n {
        // Gather and centre the neighbourhood.
        for jj in 0..k {
            let nb = neigh[i * k + jj];
            for c in 0..dim {
                z[jj * dim + c] = data[nb * dim + c];
            }
        }
        for c in 0..dim {
            let mut mean = 0.0;
            for jj in 0..k {
                mean += z[jj * dim + c];
            }
            mean /= k as f64;
            for jj in 0..k {
                z[jj * dim + c] -= mean;
            }
        }

        // Local PCA via the k x k Gram matrix. The top-d eigenvectors are the local
        // tangent coordinates Q_i (left singular vectors of the centred neighbourhood).
        for a in 0..k {
            for bb in a..k {
                let mut acc = 0.0;
                for c in 0..dim {
                    acc += z[a * dim + c] * z[bb * dim + c];
                }
                gram[a * k + bb] = acc;
                gram[bb * k + a] = acc;
            }
        }
        let (w, v) = jacobi_eigh(&gram, k)?;
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&p, &q| w[q].partial_cmp(&w[p]).unwrap_or(std::cmp::Ordering::Equal));

        // Build G_i = [ (1/sqrt(k)) 1 | Q_i ]. The Gram eigenvectors are already
        // orthonormal, so G_i has orthonormal columns and W_i = I - G_i G_i^T is the
        // orthogonal projector onto the complement of span(G_i).
        for row in 0..k {
            g[row * cols] = inv_sqrt_k;
        }
        for col in 0..d {
            let src = order[col];
            for row in 0..k {
                g[row * cols + 1 + col] = v[row * k + src];
            }
        }

        // Accumulate B[N_i, N_i] += (I - G_i G_i^T).
        for a in 0..k {
            let na = neigh[i * k + a];
            for bb in 0..k {
                let nb = neigh[i * k + bb];
                // (G_i G_i^T)[a, bb]
                let mut gg = 0.0;
                for c in 0..cols {
                    gg += g[a * cols + c] * g[bb * cols + c];
                }
                let w_ab = if a == bb { 1.0 - gg } else { -gg };
                b[na * n + nb] += w_ab;
            }
        }
    }

    // Symmetrise B.
    for a in 0..n {
        for bb in (a + 1)..n {
            let avg = 0.5 * (b[a * n + bb] + b[bb * n + a]);
            b[a * n + bb] = avg;
            b[bb * n + a] = avg;
        }
    }

    // Smallest eigenvectors of B; drop the bottom null vector (constant).
    let (eigvals, eigvecs) = jacobi_eigh(&b, n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &bb| {
        eigvals[a]
            .partial_cmp(&eigvals[bb])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut embedding = vec![0.0f64; n * d];
    for c in 0..d {
        let col = order[c + 1];
        for r in 0..n {
            embedding[r * d + c] = eigvecs[r * n + col];
        }
    }

    // Centre then rescale to identity covariance (unique up to an orthogonal map).
    for c in 0..d {
        let mut mean = 0.0;
        for r in 0..n {
            mean += embedding[r * d + c];
        }
        mean /= n as f64;
        for r in 0..n {
            embedding[r * d + c] -= mean;
        }
    }
    normalize_to_identity_covariance(&mut embedding, n, d)?;
    Ok(embedding)
}

/// Rescale a centred `n x d` embedding so its covariance is the identity:
/// computes `R = Y^T Y`, then `Y <- Y R^{-1/2} sqrt(n)`.
fn normalize_to_identity_covariance(y: &mut [f64], n: usize, d: usize) -> ManifoldResult<()> {
    let mut r = vec![0.0f64; d * d];
    for a in 0..d {
        for b in a..d {
            let mut acc = 0.0;
            for row in 0..n {
                acc += y[row * d + a] * y[row * d + b];
            }
            r[a * d + b] = acc;
            r[b * d + a] = acc;
        }
    }
    let (w, v) = jacobi_eigh(&r, d)?;
    let mut r_inv_sqrt = vec![0.0f64; d * d];
    for a in 0..d {
        for b in 0..d {
            let mut acc = 0.0;
            for c in 0..d {
                let lam = w[c];
                if lam <= 1e-12 {
                    continue;
                }
                acc += v[a * d + c] * v[b * d + c] / lam.sqrt();
            }
            r_inv_sqrt[a * d + b] = acc;
        }
    }
    let scale = (n as f64).sqrt();
    let mut out = vec![0.0f64; n * d];
    for row in 0..n {
        for b in 0..d {
            let mut acc = 0.0;
            for a in 0..d {
                acc += y[row * d + a] * r_inv_sqrt[a * d + b];
            }
            out[row * d + b] = acc * scale;
        }
    }
    y.copy_from_slice(&out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for (x, y) in a.iter().zip(b) {
            let dx = x - ma;
            let dy = y - mb;
            cov += dx * dy;
            va += dx * dx;
            vb += dy * dy;
        }
        if va <= 1e-30 || vb <= 1e-30 {
            return 0.0;
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    fn pairwise(emb: &[f64], n: usize, d: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0;
                for c in 0..d {
                    let v = emb[i * d + c] - emb[j * d + c];
                    s += v * v;
                }
                out.push(s.sqrt());
            }
        }
        out
    }

    /// (a) Linear 2D-in-3D data: LTSA recovers it up to an affine map essentially
    /// exactly, so the pairwise-distance correlation should be very high (> 0.99).
    #[test]
    fn ltsa_recovers_linear_affine_tight() {
        let side = 8;
        let n = side * side;
        let dim = 3;
        let mut data = vec![0.0; n * dim];
        let mut params = vec![0.0; n * 2];
        for a in 0..side {
            for b in 0..side {
                let idx = a * side + b;
                let t0 = a as f64 / (side - 1) as f64;
                let t1 = b as f64 / (side - 1) as f64;
                params[idx * 2] = t0;
                params[idx * 2 + 1] = t1;
                data[idx * dim] = t0;
                data[idx * dim + 1] = t1;
                data[idx * dim + 2] = 0.3 * t0 + 0.2 * t1;
            }
        }
        let emb = ltsa(&data, n, dim, 2, 8).expect("ltsa ok");
        assert_eq!(emb.len(), n * 2);
        assert!(emb.iter().all(|v| v.is_finite()));
        let pd_emb = pairwise(&emb, n, 2);
        let pd_true = pairwise(&params, n, 2);
        let corr = pearson(&pd_emb, &pd_true).abs();
        assert!(
            corr > 0.99,
            "LTSA linear affine correlation too low: {corr}"
        );
    }

    /// (b) An S-curve-like 2D manifold in 3D should preserve neighbourhoods.
    #[test]
    fn ltsa_scurve_preserves_neighbors() {
        use crate::metrics::metrics::trustworthiness;
        let n_t = 14;
        let n_u = 5;
        let n = n_t * n_u;
        let dim = 3;
        let mut data = vec![0.0; n * dim];
        let mut params = vec![0.0; n * 2];
        for it in 0..n_t {
            let t =
                -std::f64::consts::PI + 2.0 * std::f64::consts::PI * it as f64 / (n_t - 1) as f64;
            for iu in 0..n_u {
                let u = iu as f64 / (n_u - 1) as f64;
                let idx = it * n_u + iu;
                data[idx * dim] = t.sin();
                data[idx * dim + 1] = u;
                data[idx * dim + 2] = t.signum() * (t.cos() - 1.0);
                params[idx * 2] = t;
                params[idx * 2 + 1] = u;
            }
        }
        let emb = ltsa(&data, n, dim, 2, 12).expect("ltsa ok");
        assert!(emb.iter().all(|v| v.is_finite()));
        let tw = trustworthiness(&params, &emb, n, 2, 2, 6).expect("tw ok");
        assert!(tw > 0.85, "trustworthiness too low: {tw}");
    }

    /// (c) Parameter-validation errors.
    #[test]
    fn ltsa_param_errors() {
        let n = 10;
        let dim = 3;
        let data = vec![0.0; n * dim];
        // k too small: for d=2, min_k = 4.
        assert!(ltsa(&data, n, dim, 2, 3).is_err());
        // n_components exceeds n_features.
        let small = vec![0.0; 10 * 2];
        assert!(ltsa(&small, 10, 2, 3, 8).is_err());
        // Empty input.
        assert!(ltsa(&[], 0, 0, 1, 3).is_err());
        // k larger than n.
        assert!(ltsa(&data, n, dim, 2, n + 1).is_err());
    }
}
