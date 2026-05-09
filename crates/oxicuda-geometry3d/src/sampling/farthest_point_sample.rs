//! Farthest Point Sampling (FPS) for point clouds.

use crate::error::{Geom3dError, Geom3dResult};

/// Farthest Point Sampling: selects `m` points from `n` 3D points.
///
/// `points`: flat row-major `[n×3]` of f32 (x0,y0,z0, x1,y1,z1,...).
/// Returns indices of length m. Deterministic: `idx[0] = 0`.
///
/// Algorithm: maintain `dist[i] = f32::INFINITY`. For k=1..m: update
/// `dist[i] = min(dist[i], sq_dist(points[i], points[selected[k-1]]))`,
/// pick `argmax(dist)` (first occurrence on tie). Return selected indices.
pub fn farthest_point_sample(points: &[f32], n: usize, m: usize) -> Geom3dResult<Vec<usize>> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if points.len() != n * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * 3,
            got: points.len(),
        });
    }
    if m > n {
        return Err(Geom3dError::InvalidSampleCount {
            requested: m,
            available: n,
        });
    }
    if m == 0 {
        return Ok(Vec::new());
    }

    let mut selected = Vec::with_capacity(m);
    let mut dist = vec![f32::INFINITY; n];

    selected.push(0usize);
    dist[0] = 0.0;

    for _k in 1..m {
        let last = *selected
            .last()
            .ok_or_else(|| Geom3dError::Internal("selected list empty unexpectedly".to_string()))?;
        let lx = points[last * 3];
        let ly = points[last * 3 + 1];
        let lz = points[last * 3 + 2];

        // Update distances
        for i in 0..n {
            let dx = points[i * 3] - lx;
            let dy = points[i * 3 + 1] - ly;
            let dz = points[i * 3 + 2] - lz;
            let d = dx * dx + dy * dy + dz * dz;
            if d < dist[i] {
                dist[i] = d;
            }
        }

        // Pick argmax(dist)
        let mut best_idx = 0usize;
        let mut best_dist = f32::NEG_INFINITY;
        for (i, &d) in dist.iter().enumerate() {
            if d > best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        selected.push(best_idx);
        dist[best_idx] = 0.0;
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(n: usize) -> Vec<f32> {
        let mut pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            pts.push(i as f32);
            pts.push(0.0);
            pts.push(0.0);
        }
        pts
    }

    #[test]
    fn fps_empty_cloud_error() {
        let pts: Vec<f32> = vec![];
        assert_eq!(
            farthest_point_sample(&pts, 0, 1),
            Err(Geom3dError::EmptyPointCloud)
        );
    }

    #[test]
    fn fps_dimension_mismatch() {
        let pts = vec![1.0_f32, 2.0]; // only 2 elements, not 3
        assert_eq!(
            farthest_point_sample(&pts, 1, 1),
            Err(Geom3dError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn fps_m_exceeds_n() {
        let pts = make_grid(5);
        assert_eq!(
            farthest_point_sample(&pts, 5, 6),
            Err(Geom3dError::InvalidSampleCount {
                requested: 6,
                available: 5
            })
        );
    }

    #[test]
    fn fps_m_zero_returns_empty() {
        let pts = make_grid(5);
        let r = farthest_point_sample(&pts, 5, 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn fps_m_one_returns_index_zero() {
        let pts = make_grid(5);
        let r = farthest_point_sample(&pts, 5, 1).unwrap();
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn fps_selects_correct_count() {
        let pts = make_grid(100);
        let r = farthest_point_sample(&pts, 100, 10).unwrap();
        assert_eq!(r.len(), 10);
    }

    #[test]
    fn fps_all_distinct() {
        let pts = make_grid(50);
        let r = farthest_point_sample(&pts, 50, 20).unwrap();
        let mut sorted = r.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 20, "FPS must return distinct indices");
    }

    #[test]
    fn fps_indices_in_range() {
        let pts = make_grid(30);
        let r = farthest_point_sample(&pts, 30, 15).unwrap();
        assert!(r.iter().all(|&i| i < 30));
    }

    #[test]
    fn fps_first_is_zero() {
        let pts = make_grid(20);
        let r = farthest_point_sample(&pts, 20, 5).unwrap();
        assert_eq!(r[0], 0, "FPS first selected must be index 0");
    }

    #[test]
    fn fps_on_linear_grid_selects_extremes() {
        // On a 1D line 0..10, FPS should pick well-spread points
        let n = 10;
        let mut pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            pts.push(i as f32);
            pts.push(0.0);
            pts.push(0.0);
        }
        let r = farthest_point_sample(&pts, n, 3).unwrap();
        assert_eq!(r.len(), 3);
        // Should select 0, 9 (farthest from 0), and 4 or 5 (farthest from both)
        assert!(r.contains(&0));
        assert!(r.contains(&9), "Expected 9 in FPS result, got {:?}", r);
    }

    #[test]
    fn fps_deterministic() {
        let pts = make_grid(50);
        let r1 = farthest_point_sample(&pts, 50, 10).unwrap();
        let r2 = farthest_point_sample(&pts, 50, 10).unwrap();
        assert_eq!(r1, r2, "FPS must be deterministic");
    }

    #[test]
    fn fps_3d_spread() {
        // 8 corners of a unit cube
        let pts: Vec<f32> = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let r = farthest_point_sample(&pts, 8, 4).unwrap();
        // Should select well-spread corners
        assert_eq!(r.len(), 4);
        let mut sorted = r.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
    }
}
