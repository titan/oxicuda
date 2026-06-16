//! Radius-limited ball query for point clouds.

use crate::error::{Geom3dError, Geom3dResult};

/// Radius-limited ball query.
///
/// Returns `(indices: [nq × k_max], counts: [nq])` with sentinel `usize::MAX`
/// for empty slots.
pub fn ball_query(
    queries: &[f32],
    nq: usize,
    points: &[f32],
    np: usize,
    k_max: usize,
    radius: f32,
) -> Geom3dResult<(Vec<usize>, Vec<usize>)> {
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
    if radius <= 0.0 || !radius.is_finite() {
        return Err(Geom3dError::InvalidRadius { radius });
    }

    let r_sq = radius * radius;
    let mut indices = vec![usize::MAX; nq * k_max];
    let mut counts = vec![0usize; nq];

    for qi in 0..nq {
        let qx = queries[qi * 3];
        let qy = queries[qi * 3 + 1];
        let qz = queries[qi * 3 + 2];
        let mut cnt = 0usize;

        for pi in 0..np {
            if cnt >= k_max {
                break;
            }
            let dx = points[pi * 3] - qx;
            let dy = points[pi * 3 + 1] - qy;
            let dz = points[pi * 3 + 2] - qz;
            let d_sq = dx * dx + dy * dy + dz * dz;
            if d_sq < r_sq {
                indices[qi * k_max + cnt] = pi;
                cnt += 1;
            }
        }
        counts[qi] = cnt;
    }

    Ok((indices, counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ball_query_basic() {
        let pts: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let q = vec![4.5_f32, 0.0, 0.0];
        let (idx, cnt) = ball_query(&q, 1, &pts, 10, 10, 2.0).expect("ball_query should succeed");
        assert_eq!(cnt[0], 4); // 3,4,5,6 are within 2.0
        assert!(idx[..cnt[0]].iter().all(|&i| i != usize::MAX));
    }

    #[test]
    fn ball_query_invalid_radius_error() {
        let pts = vec![1.0_f32, 0.0, 0.0];
        let q = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(
            ball_query(&q, 1, &pts, 1, 1, 0.0),
            Err(Geom3dError::InvalidRadius { radius: 0.0 })
        );
        assert_eq!(
            ball_query(&q, 1, &pts, 1, 1, -1.0),
            Err(Geom3dError::InvalidRadius { radius: -1.0 })
        );
    }

    #[test]
    fn ball_query_empty_sentinels() {
        let pts: Vec<f32> = vec![10.0, 10.0, 10.0];
        let q = vec![0.0_f32, 0.0, 0.0];
        let (idx, cnt) = ball_query(&q, 1, &pts, 1, 5, 1.0).expect("ball_query should succeed");
        assert_eq!(cnt[0], 0);
        assert!(idx.iter().all(|&i| i == usize::MAX));
    }

    #[test]
    fn ball_query_k_max_respected() {
        let pts: Vec<f32> = (0..20).flat_map(|_| vec![0.0_f32, 0.0, 0.0]).collect();
        let q = vec![0.0_f32, 0.0, 0.0];
        let k = 5;
        let (_, cnt) = ball_query(&q, 1, &pts, 20, k, 1.0).expect("ball_query should succeed");
        assert!(cnt[0] <= k, "count must not exceed k_max");
    }
}
