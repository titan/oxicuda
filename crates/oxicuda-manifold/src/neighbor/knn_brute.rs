//! Brute-force k-nearest-neighbours search.

use crate::error::{ManifoldError, ManifoldResult};

/// Compute squared Euclidean distance between two row-major vectors of length `dim`.
fn sq_dist(a: &[f64], b: &[f64], dim: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..dim {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Brute-force kNN on row-major data of shape `(n, dim)`.
///
/// Returns `(indices, distances)` of shape `(n, k)`. The query point itself is excluded.
pub fn knn_brute(
    x: &[f64],
    n: usize,
    dim: usize,
    k: usize,
) -> ManifoldResult<(Vec<usize>, Vec<f64>)> {
    if n == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if x.len() != n * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, dim],
            got: vec![x.len()],
        });
    }
    if k == 0 || k >= n {
        return Err(ManifoldError::KNeighborsTooLarge { k, n });
    }
    let mut idx_out = vec![0usize; n * k];
    let mut dist_out = vec![0.0f64; n * k];
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
    for i in 0..n {
        buf.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = sq_dist(&x[i * dim..i * dim + dim], &x[j * dim..j * dim + dim], dim);
            buf.push((d, j));
        }
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for kk in 0..k {
            idx_out[i * k + kk] = buf[kk].1;
            dist_out[i * k + kk] = buf[kk].0;
        }
    }
    Ok((idx_out, dist_out))
}

/// Brute-force kNN given a precomputed `(n, n)` distance matrix (row-major).
///
/// Distances of a point to itself are ignored.
pub fn knn_brute_from_distance_matrix(
    d: &[f64],
    n: usize,
    k: usize,
) -> ManifoldResult<(Vec<usize>, Vec<f64>)> {
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if d.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![d.len()],
        });
    }
    if k == 0 || k >= n {
        return Err(ManifoldError::KNeighborsTooLarge { k, n });
    }
    let mut idx_out = vec![0usize; n * k];
    let mut dist_out = vec![0.0f64; n * k];
    let mut buf: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
    for i in 0..n {
        buf.clear();
        for j in 0..n {
            if j == i {
                continue;
            }
            buf.push((d[i * n + j], j));
        }
        buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for kk in 0..k {
            idx_out[i * k + kk] = buf[kk].1;
            dist_out[i * k + kk] = buf[kk].0;
        }
    }
    Ok((idx_out, dist_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_basic() {
        // 4 points on a line: 0, 1, 2, 3
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let (idx, _d) = knn_brute(&x, 4, 1, 2).expect("ok");
        // closest two for point 1: {0, 2}
        let n1 = [idx[2], idx[3]];
        assert!(n1.contains(&0) && n1.contains(&2));
    }

    #[test]
    fn knn_too_large_errors() {
        let x = vec![0.0, 1.0, 2.0];
        let r = knn_brute(&x, 3, 1, 10);
        assert!(r.is_err());
    }

    #[test]
    fn knn_from_matrix() {
        let n = 3;
        let d = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
        let (idx, _) = knn_brute_from_distance_matrix(&d, n, 1).expect("ok");
        assert_eq!(idx[0], 1);
        assert_eq!(idx[2], 1);
    }
}
