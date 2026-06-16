//! Hessian Locally Linear Embedding (HLLE), Donoho & Grimes 2003.
//!
//! HLLE estimates, for every point, the Hessian of the manifold expressed in local
//! tangent coordinates, then seeks an embedding whose coordinate functions have a
//! vanishing local Hessian (i.e. functions that are locally linear in the intrinsic
//! parametrisation). Concretely:
//!
//! 1. For each point `i`, gather its neighbourhood `N_i` (the point itself plus its
//!    `k - 1` nearest neighbours, `k` members total) and centre it.
//! 2. Estimate the local tangent coordinates `M_i` (`k x d`) as the top-`d` principal
//!    directions of the centred neighbourhood (local PCA via the `k x k` Gram matrix).
//! 3. Form the Hessian design matrix `Y_i` whose columns are
//!    `[ 1 | M_i (d linear) | the d(d+1)/2 quadratic cross-products ]`,
//!    orthonormalise it (Householder QR), and take the last `dp = d(d+1)/2`
//!    orthonormal columns as the local Hessian estimator `H_i` (`k x dp`).
//! 4. Accumulate the global symmetric matrix `Phi[N_i, N_i] += H_i H_i^T`.
//! 5. The embedding is the `d` eigenvectors of `Phi` belonging to the smallest
//!    eigenvalues, skipping the bottom null eigenvector (the constant function).
//!
//! The eigenvectors are renormalised so that the embedding has identity covariance
//! (the standard `Y <- Y (Y^T Y)^{-1/2} sqrt(n)` rescaling), making the output unique
//! up to an orthogonal transform.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::householder_qr::householder_qr;
use crate::linalg::jacobi_eig::jacobi_eigh;
use crate::neighbor::knn_brute::knn_brute;

/// Fit Hessian LLE on row-major `(n_samples, n_features)` data.
///
/// Returns the embedding as a row-major `n_samples x n_components` matrix.
///
/// `n_neighbors` must satisfy `n_neighbors > n_components * (n_components + 1) / 2 +
/// n_components`; this guarantees the neighbourhood is large enough to fit the
/// constant, linear, and quadratic terms of the local Hessian design.
pub fn hessian_lle(
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
    let dp = d * (d + 1) / 2;
    // Minimum neighbourhood size: constant (1) + linear (d) + quadratic (dp).
    let min_k = 1 + d + dp;
    if n_neighbors < min_k {
        return Err(ManifoldError::InvalidParameter {
            name: "n_neighbors".into(),
            reason: format!(
                "Hessian LLE requires n_neighbors >= 1 + n_components + \
                 n_components*(n_components+1)/2 = {min_k}, got {n_neighbors}"
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
    // Materialise the per-point neighbour list including self (length k).
    let mut neigh = vec![0usize; n * k];
    for i in 0..n {
        neigh[i * k] = i;
        for jj in 0..(k - 1) {
            neigh[i * k + 1 + jj] = knn_idx[i * (k - 1) + jj];
        }
    }

    // Width of the Hessian design matrix.
    let width = 1 + d + dp;

    // Global accumulation matrix Phi (n x n), symmetric.
    let mut phi = vec![0.0f64; n * n];

    // Reusable per-neighbourhood scratch.
    let mut z = vec![0.0f64; k * dim]; // centred neighbourhood (k x dim)
    let mut gram = vec![0.0f64; k * k]; // Z Z^T (k x k)
    let mut tangent = vec![0.0f64; k * d]; // local tangent coords M_i (k x d)
    let mut design = vec![0.0f64; k * width]; // [1 | M_i | quadratic]

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

        // Local PCA via the k x k Gram matrix G = Z Z^T. Its top-d eigenvectors are
        // the tangent coordinates (left singular vectors of Z) — exactly the local
        // tangent parametrisation we need, regardless of the ambient dimension.
        for a in 0..k {
            for b in a..k {
                let mut acc = 0.0;
                for c in 0..dim {
                    acc += z[a * dim + c] * z[b * dim + c];
                }
                gram[a * k + b] = acc;
                gram[b * k + a] = acc;
            }
        }
        let (mut w, mut v) = jacobi_eigh(&gram, k)?;
        // Select the d largest eigenvalues' eigenvectors as tangent coordinates.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&p, &q| w[q].partial_cmp(&w[p]).unwrap_or(std::cmp::Ordering::Equal));
        for col in 0..d {
            let src = order[col];
            for row in 0..k {
                tangent[row * d + col] = v[row * k + src];
            }
        }
        // Avoid unused-mut warnings on the moved-out buffers.
        let _ = &mut w;
        let _ = &mut v;

        // Build the Hessian design matrix:
        //   column 0          : constant 1
        //   columns 1..=d     : linear tangent coords
        //   columns d+1..     : quadratic cross-products (incl. squares)
        for row in 0..k {
            design[row * width] = 1.0;
            for c in 0..d {
                design[row * width + 1 + c] = tangent[row * d + c];
            }
            let mut col = 1 + d;
            for a in 0..d {
                for b in a..d {
                    design[row * width + col] = tangent[row * d + a] * tangent[row * d + b];
                    col += 1;
                }
            }
        }

        // Orthonormalise the design columns. The last dp orthonormal columns span the
        // local Hessian estimator H_i (k x dp).
        let (q, _r) = householder_qr(&design, k, width)?;
        // H_i = columns [width - dp .. width) of q (k x dp).
        // Accumulate Phi[N_i, N_i] += H_i H_i^T.
        let h_off = width - dp;
        for a in 0..k {
            let na = neigh[i * k + a];
            for b in 0..k {
                let nb = neigh[i * k + b];
                let mut acc = 0.0;
                for h in 0..dp {
                    acc += q[a * width + h_off + h] * q[b * width + h_off + h];
                }
                phi[na * n + nb] += acc;
            }
        }
    }

    // Symmetrise Phi to cancel floating-point asymmetry.
    for a in 0..n {
        for b in (a + 1)..n {
            let avg = 0.5 * (phi[a * n + b] + phi[b * n + a]);
            phi[a * n + b] = avg;
            phi[b * n + a] = avg;
        }
    }

    // Eigenvectors of Phi for the smallest eigenvalues. The very smallest (~0)
    // corresponds to the constant function and is dropped.
    let (eigvals, eigvecs) = jacobi_eigh(&phi, n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigvals[a]
            .partial_cmp(&eigvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Skip the bottom null vector, take the next d eigenvectors.
    let mut embedding = vec![0.0f64; n * d];
    for c in 0..d {
        let col = order[c + 1];
        for r in 0..n {
            embedding[r * d + c] = eigvecs[r * n + col];
        }
    }

    // Renormalise so the embedding has identity covariance:
    //   R = Y^T Y, then Y <- Y R^{-1/2} sqrt(n).
    // First, centre each column (eigenvectors of Phi are orthogonal to the constant,
    // so they are already ~zero-mean; subtract the empirical mean to be safe).
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
    // R = Y^T Y (d x d).
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
    // R^{-1/2} = V diag(1/sqrt(lam)) V^T.
    let (w, v) = jacobi_eigh(&r, d)?;
    let mut r_inv_sqrt = vec![0.0f64; d * d];
    for a in 0..d {
        for b in 0..d {
            let mut acc = 0.0;
            for c in 0..d {
                let lam = w[c];
                if lam <= 1e-12 {
                    // Degenerate embedding direction; treat as already unit-scaled.
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

    /// Pearson correlation between two equal-length slices.
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

    /// Flattened upper-triangular pairwise Euclidean distances of a row-major `n x d`.
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

    /// (a) Linear 2D-in-3D data must be recovered up to an affine map: the pairwise
    /// distances of the embedding correlate strongly with those of the true params.
    #[test]
    fn hlle_recovers_linear_affine() {
        // 2D grid of parameters t in [0,1]^2, mapped linearly into 3D.
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
        let emb = hessian_lle(&data, n, dim, 2, 8).expect("hlle ok");
        assert_eq!(emb.len(), n * 2);
        assert!(emb.iter().all(|v| v.is_finite()));
        let pd_emb = pairwise(&emb, n, 2);
        let pd_true = pairwise(&params, n, 2);
        let corr = pearson(&pd_emb, &pd_true).abs();
        assert!(corr > 0.95, "linear affine correlation too low: {corr}");
    }

    /// (b) An S-curve-like 2D manifold in 3D should yield an embedding that preserves
    /// local neighbourhoods of the intrinsic parameters.
    #[test]
    fn hlle_scurve_preserves_neighbors() {
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
        let emb = hessian_lle(&data, n, dim, 2, 12).expect("hlle ok");
        assert!(emb.iter().all(|v| v.is_finite()));
        // Trustworthiness between the intrinsic params and the embedding should be high.
        let tw = trustworthiness(&params, &emb, n, 2, 2, 6).expect("tw ok");
        assert!(tw > 0.85, "trustworthiness too low: {tw}");
    }

    /// (c) Parameter-validation errors.
    #[test]
    fn hlle_param_errors() {
        let n = 10;
        let dim = 3;
        let data = vec![0.0; n * dim];
        // k too small: for d=2, min_k = 1 + 2 + 3 = 6.
        assert!(hessian_lle(&data, n, dim, 2, 5).is_err());
        // n_components exceeds n_features.
        let small = vec![0.0; 10 * 2];
        assert!(hessian_lle(&small, 10, 2, 3, 8).is_err());
        // Empty input.
        assert!(hessian_lle(&[], 0, 0, 1, 3).is_err());
        // k larger than n.
        assert!(hessian_lle(&data, n, dim, 2, n + 1).is_err());
    }

    /// Collinear / degenerate neighbourhoods must not panic (QR + ridge-free guard).
    #[test]
    fn hlle_degenerate_neighborhood_is_finite() {
        // All points on a 1D line embedded in 3D — local neighbourhoods are rank-1.
        let n = 12;
        let dim = 3;
        let mut data = vec![0.0; n * dim];
        for i in 0..n {
            let t = i as f64;
            data[i * dim] = t;
            data[i * dim + 1] = 2.0 * t;
            data[i * dim + 2] = -t;
        }
        // d = 1 needs min_k = 1 + 1 + 1 = 3.
        let emb = hessian_lle(&data, n, dim, 1, 4).expect("hlle ok");
        assert!(emb.iter().all(|v| v.is_finite()));
    }
}
