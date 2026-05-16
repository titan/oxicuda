//! Pairwise distance matrix computations and k-NN graph construction.
//!
//! All distances operate on row-major point clouds: `points[i * n_dims + d]` is
//! the d-th coordinate of point i.

use crate::error::{TdaError, TdaResult};

/// Validate a point-cloud slice.
fn validate_points(points: &[f64], n_dims: usize) -> TdaResult<usize> {
    if n_dims == 0 {
        return Err(TdaError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if points.is_empty() {
        return Err(TdaError::EmptyPointCloud);
    }
    if !points.len().is_multiple_of(n_dims) {
        return Err(TdaError::DimensionMismatch {
            expected: (points.len() / n_dims) * n_dims,
            got: points.len(),
        });
    }
    Ok(points.len() / n_dims)
}

/// Pairwise **squared** Euclidean distance matrix (n×n, row-major).
///
/// `result[i * n + j] = Σ_d (p_i[d] - p_j[d])²`
pub fn pairwise_euclidean_sq(points: &[f64], n_dims: usize) -> TdaResult<Vec<f64>> {
    let n = validate_points(points, n_dims)?;
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0_f64;
            for d in 0..n_dims {
                let diff = points[i * n_dims + d] - points[j * n_dims + d];
                sq += diff * diff;
            }
            dist[i * n + j] = sq;
            dist[j * n + i] = sq;
        }
    }
    Ok(dist)
}

/// Pairwise Euclidean distance matrix (n×n, row-major).
pub fn pairwise_euclidean(points: &[f64], n_dims: usize) -> TdaResult<Vec<f64>> {
    let sq = pairwise_euclidean_sq(points, n_dims)?;
    Ok(sq.into_iter().map(|v| v.sqrt()).collect())
}

/// Pairwise Manhattan (L1) distance matrix (n×n, row-major).
pub fn pairwise_manhattan(points: &[f64], n_dims: usize) -> TdaResult<Vec<f64>> {
    let n = validate_points(points, n_dims)?;
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut l1 = 0.0_f64;
            for d in 0..n_dims {
                l1 += (points[i * n_dims + d] - points[j * n_dims + d]).abs();
            }
            dist[i * n + j] = l1;
            dist[j * n + i] = l1;
        }
    }
    Ok(dist)
}

/// k-Nearest-Neighbour graph: for each of the n points, return the indices of its k closest
/// neighbours (excluding itself), sorted by distance ascending.
pub fn knn_graph(dist: &[f64], n_pts: usize, k: usize) -> TdaResult<Vec<Vec<usize>>> {
    if n_pts == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if dist.len() != n_pts * n_pts {
        return Err(TdaError::DimensionMismatch {
            expected: n_pts * n_pts,
            got: dist.len(),
        });
    }
    let actual_k = k.min(n_pts.saturating_sub(1));
    let mut result = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let mut neighbours: Vec<(usize, u64)> = (0..n_pts)
            .filter(|&j| j != i)
            .map(|j| (j, dist[i * n_pts + j].to_bits()))
            .collect();
        neighbours.sort_unstable_by_key(|&(_, d)| d);
        result.push(
            neighbours
                .into_iter()
                .take(actual_k)
                .map(|(j, _)| j)
                .collect(),
        );
    }
    Ok(result)
}

/// Convert a raw point cloud to a symmetric Euclidean distance matrix.
///
/// Equivalent to `pairwise_euclidean`, provided as a convenience alias.
pub fn points_to_distance_matrix(points: &[f64], n_dims: usize) -> TdaResult<Vec<f64>> {
    pairwise_euclidean(points, n_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_sq_identity() {
        let pts = vec![0.0f64, 0.0, 3.0, 4.0];
        let d = pairwise_euclidean_sq(&pts, 2).expect("ok");
        // dist²(p0, p0) = 0
        assert!((d[0]).abs() < 1e-12);
        // dist²(p0, p1) = 9+16 = 25
        assert!((d[1] - 25.0).abs() < 1e-10);
        // symmetric
        assert!((d[1] - d[2]).abs() < 1e-12);
    }

    #[test]
    fn knn_selects_closest() {
        let pts = vec![0.0f64, 0.0, 1.0, 0.0, 2.0, 0.0];
        let dist = pairwise_euclidean(&pts, 2).expect("ok");
        let knn = knn_graph(&dist, 3, 1).expect("ok");
        // Point 1 (at x=1) is closest to both point 0 and point 2
        assert_eq!(knn[0][0], 1);
        assert_eq!(knn[2][0], 1);
    }
}
