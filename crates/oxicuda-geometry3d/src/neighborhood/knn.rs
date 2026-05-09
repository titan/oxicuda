//! Brute-force k-nearest neighbors search.

use crate::error::{Geom3dError, Geom3dResult};

/// Brute-force k-nearest neighbors.
///
/// `queries`: `[nq×3]`, `points`: `[np×3]`.
/// Returns `(indices: [nq×k], sq_dists: [nq×k])` row-major.
pub fn knn(
    queries: &[f32],
    nq: usize,
    points: &[f32],
    np: usize,
    k: usize,
) -> Geom3dResult<(Vec<usize>, Vec<f32>)> {
    if nq == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if np == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if queries.len() != nq * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: nq * 3,
            got: queries.len(),
        });
    }
    if points.len() != np * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: np * 3,
            got: points.len(),
        });
    }
    if k > np {
        return Err(Geom3dError::InvalidK { k, n: np });
    }
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut all_indices = vec![0usize; nq * k];
    let mut all_sq_dists = vec![0.0_f32; nq * k];

    for qi in 0..nq {
        let qx = queries[qi * 3];
        let qy = queries[qi * 3 + 1];
        let qz = queries[qi * 3 + 2];

        // Compute all distances
        let mut dists: Vec<(f32, usize)> = points
            .chunks_exact(3)
            .enumerate()
            .map(|(pi, p)| {
                let dx = p[0] - qx;
                let dy = p[1] - qy;
                let dz = p[2] - qz;
                (dx * dx + dy * dy + dz * dz, pi)
            })
            .collect();

        // Partial sort to get k smallest
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for j in 0..k {
            all_indices[qi * k + j] = dists[j].1;
            all_sq_dists[qi * k + j] = dists[j].0;
        }
    }

    Ok((all_indices, all_sq_dists))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_basic() {
        let pts: Vec<f32> = (0..5).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let queries = vec![2.1_f32, 0.0, 0.0];
        let (idx, dists) = knn(&queries, 1, &pts, 5, 2).unwrap();
        // nearest to 2.1 should be 2 and 3
        assert!(idx[0] == 2 || idx[0] == 3);
        assert!(dists[0] <= dists[1]);
    }

    #[test]
    fn knn_empty_query_error() {
        let pts = vec![1.0_f32, 0.0, 0.0];
        assert!(knn(&[], 0, &pts, 1, 1).is_err());
    }

    #[test]
    fn knn_k_exceeds_n_error() {
        let pts: Vec<f32> = (0..3).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let q = vec![0.0_f32, 0.0, 0.0];
        assert_eq!(
            knn(&q, 1, &pts, 3, 5),
            Err(Geom3dError::InvalidK { k: 5, n: 3 })
        );
    }

    #[test]
    fn knn_sorted_distances() {
        let pts: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let q = vec![4.5_f32, 0.0, 0.0];
        let (_, dists) = knn(&q, 1, &pts, 10, 5).unwrap();
        for w in dists.windows(2) {
            assert!(w[0] <= w[1], "distances should be sorted ascending");
        }
    }

    #[test]
    fn knn_self_distance_zero() {
        let pts: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_, dists) = knn(&pts, 2, &pts, 2, 1).unwrap();
        assert!(dists[0] < 1e-6);
        assert!(dists[1] < 1e-6);
    }
}
