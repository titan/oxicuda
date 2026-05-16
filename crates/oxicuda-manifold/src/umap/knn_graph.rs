//! kNN graph construction and per-row sigma/rho fitting for UMAP.

use crate::error::{ManifoldError, ManifoldResult};
use crate::neighbor::knn_brute::knn_brute;

/// Build kNN distances (Euclidean) from row-major data.
///
/// Returns `(indices, distances)` of shape `(n, k)`.
pub fn build_knn_distances(
    x: &[f64],
    n: usize,
    dim: usize,
    k: usize,
) -> ManifoldResult<(Vec<usize>, Vec<f64>)> {
    let (idx, d2) = knn_brute(x, n, dim, k)?;
    let d: Vec<f64> = d2.iter().map(|v| v.sqrt()).collect();
    Ok((idx, d))
}

/// Smooth kNN distances: for each row find `sigma_i` and `rho_i = d[i, 0]` (nearest neighbour)
/// satisfying `sum_j exp(-(d[i,j] - rho_i)/sigma_i) = log2(k)`.
///
/// Returns `(sigmas, rhos)` each of length `n`.
pub fn smooth_knn_distances(
    knn_dist: &[f64],
    n: usize,
    k: usize,
    n_iter: usize,
    tol: f64,
) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    if knn_dist.len() != n * k {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, k],
            got: vec![knn_dist.len()],
        });
    }
    let target = (k as f64).log2();
    let mut sigmas = vec![1.0; n];
    let mut rhos = vec![0.0; n];
    for i in 0..n {
        // rho is the smallest non-zero distance.
        let mut rho = 0.0_f64;
        for j in 0..k {
            let dij = knn_dist[i * k + j];
            if dij > 0.0 {
                rho = dij;
                break;
            }
        }
        rhos[i] = rho;
        // Binary search for sigma so that sum_j max(0, exp(-(d_ij - rho)/sigma)) = log2 k
        let mut lo = 0.0_f64;
        let mut hi = f64::INFINITY;
        let mut mid = 1.0_f64;
        for _ in 0..n_iter {
            let mut s = 0.0;
            for j in 0..k {
                let dij = knn_dist[i * k + j];
                let arg = (dij - rho).max(0.0);
                if mid > 0.0 {
                    s += (-arg / mid).exp();
                }
            }
            if (s - target).abs() < tol {
                break;
            }
            if s > target {
                hi = mid;
                mid = 0.5 * (lo + hi);
            } else {
                lo = mid;
                if hi == f64::INFINITY {
                    mid *= 2.0;
                } else {
                    mid = 0.5 * (lo + hi);
                }
            }
        }
        sigmas[i] = mid.max(1e-12);
    }
    Ok((sigmas, rhos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_returns_finite() {
        // 4 rows, k=3
        let d = vec![0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 0.05, 0.1, 0.4, 0.3, 0.7, 0.8];
        let (s, r) = smooth_knn_distances(&d, 4, 3, 32, 1e-5).expect("ok");
        for v in s.iter().chain(r.iter()) {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }
}
