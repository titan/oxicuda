//! Grid-based kNN / uniform spatial hashing for point clouds.
//!
//! A [`SpatialHashGrid`] buckets every point into a uniform 3-D grid keyed by
//! integer cell coordinates. For (roughly) uniform-density clouds this gives
//! near-constant-time neighbourhood queries and outperforms a brute-force scan
//! for large `n`, serving as the spatial-hashing fallback for kNN / radius
//! search.
//!
//! # Distance conventions
//!
//! * [`SpatialHashGrid::knn`] returns `(index, squared_distance)` pairs sorted
//!   by ascending distance, matching the crate kNN convention used by
//!   [`crate::neighborhood::kd_tree::KdTree`] and
//!   [`crate::neighborhood::knn::knn`] (distances are **squared**).
//! * [`SpatialHashGrid::radius_search`] uses **plain Euclidean** distance and
//!   returns the indices of every point within `<= radius` of the query.
//!
//! # Exactness of the expanding-ring kNN
//!
//! The kNN search starts at the query's own cell and visits expanding cubic
//! "rings" of cells at increasing Chebyshev radius `r = 0, 1, 2, …`. Any point
//! in a cell at ring `r` is at least `(r - 1) * cell_size` away along the
//! dominant axis from the query; conversely, the *closest possible* point in
//! any cell of ring `r` is at least `(r - 1) * cell_size` from the query, so a
//! point not yet examined (i.e. in ring `r + 1` or beyond) is at least
//! `r * cell_size` away. Once the bounded heap holds `k` candidates and the
//! current k-th-best distance is `<= r * cell_size`, no farther ring can
//! improve the result and the search terminates — guaranteeing exactness.

use crate::error::{Geom3dError, Geom3dResult};
use std::collections::HashMap;

/// Configuration for [`SpatialHashGrid`].
#[derive(Debug, Clone, PartialEq)]
pub struct GridKnnConfig {
    /// Edge length of each cubic grid cell; must be `> 0`.
    pub cell_size: f32,
}

/// Uniform spatial-hash grid over a flat row-major `[n×3]` point cloud.
#[derive(Debug)]
pub struct SpatialHashGrid {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
    points: Vec<f32>,
    n: usize,
}

impl SpatialHashGrid {
    /// Build a spatial-hash grid over `n` 3-D points (flat row-major `[n×3]`).
    ///
    /// # Errors
    ///
    /// * [`Geom3dError::EmptyPointCloud`] if `n == 0`.
    /// * [`Geom3dError::DimensionMismatch`] if `points.len() != n * 3`.
    /// * [`Geom3dError::InvalidVoxelSize`] if `cell_size <= 0` or non-finite.
    pub fn build(points: &[f32], n: usize, cfg: GridKnnConfig) -> Geom3dResult<Self> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if points.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: points.len(),
            });
        }
        if cfg.cell_size <= 0.0 || !cfg.cell_size.is_finite() {
            return Err(Geom3dError::InvalidVoxelSize {
                voxel_size: cfg.cell_size,
            });
        }

        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        let inv = 1.0_f32 / cfg.cell_size;
        for i in 0..n {
            let cx = (points[i * 3] * inv).floor() as i32;
            let cy = (points[i * 3 + 1] * inv).floor() as i32;
            let cz = (points[i * 3 + 2] * inv).floor() as i32;
            cells.entry((cx, cy, cz)).or_default().push(i);
        }

        Ok(Self {
            cells,
            cell_size: cfg.cell_size,
            points: points.to_vec(),
            n,
        })
    }

    /// Integer cell coordinate of point `p`:
    /// `(floor(x/c), floor(y/c), floor(z/c))`.
    #[must_use]
    pub fn cell_of(&self, p: &[f32; 3]) -> (i32, i32, i32) {
        let inv = 1.0_f32 / self.cell_size;
        (
            (p[0] * inv).floor() as i32,
            (p[1] * inv).floor() as i32,
            (p[2] * inv).floor() as i32,
        )
    }

    /// Return the `k` nearest points to `query` as `(index, squared_distance)`
    /// sorted by ascending distance.
    ///
    /// If `k` exceeds the number of points, all points are returned. Distances
    /// are **squared** Euclidean to match the crate kNN convention.
    ///
    /// # Errors
    ///
    /// [`Geom3dError::InvalidK`] if `k == 0`.
    pub fn knn(&self, query: &[f32; 3], k: usize) -> Geom3dResult<Vec<(usize, f32)>> {
        if k == 0 {
            return Err(Geom3dError::InvalidK { k, n: self.n });
        }
        let kk = k.min(self.n);
        let (qcx, qcy, qcz) = self.cell_of(query);

        // Bounded max-heap (worst at index 0).
        let mut heap: Vec<(f32, usize)> = Vec::with_capacity(kk + 1);

        let mut ring = 0_i32;
        loop {
            self.scan_ring(query, qcx, qcy, qcz, ring, kk, &mut heap);

            // Termination: once full, the next ring's minimum reachable
            // distance is `ring * cell_size`. If that already exceeds the
            // k-th best, no farther cell can improve the result.
            if heap.len() >= kk {
                let worst = heap.first().map_or(f32::INFINITY, |h| h.0);
                let min_next = ring as f32 * self.cell_size;
                if min_next * min_next > worst {
                    break;
                }
            }

            ring += 1;
            // Safety bound: rings cannot exceed the populated extent. Once a
            // ring (and several beyond it for the full guard) is entirely
            // empty and the heap is full, the guard above stops us. The hard
            // cap prevents unbounded looping on degenerate inputs.
            if ring > Self::MAX_RING {
                break;
            }
        }

        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(heap.into_iter().map(|(d, i)| (i, d)).collect())
    }

    /// Return every point index within `radius` (Euclidean, inclusive) of
    /// `query`.
    ///
    /// # Errors
    ///
    /// [`Geom3dError::InvalidRadius`] if `radius` is negative or non-finite.
    pub fn radius_search(&self, query: &[f32; 3], radius: f32) -> Geom3dResult<Vec<usize>> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(Geom3dError::InvalidRadius { radius });
        }
        let r_sq = radius * radius;
        let (qcx, qcy, qcz) = self.cell_of(query);
        let reach = (radius / self.cell_size).ceil() as i32;

        let mut out = Vec::new();
        for dz in -reach..=reach {
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    if let Some(bucket) = self.cells.get(&(qcx + dx, qcy + dy, qcz + dz)) {
                        for &idx in bucket {
                            if self.point_sq_dist(idx, query) <= r_sq {
                                out.push(idx);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the grid indexes zero points (always `false` after a successful
    /// [`SpatialHashGrid::build`], which rejects empty clouds).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of occupied cells (useful for diagnostics / density estimates).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Hard cap on ring expansion to guard against degenerate / non-finite
    /// inputs (a few thousand cells along an axis is far beyond any realistic
    /// uniform-density use).
    const MAX_RING: i32 = 4096;

    /// Squared distance from `query` to point `idx`.
    #[inline]
    fn point_sq_dist(&self, idx: usize, query: &[f32; 3]) -> f32 {
        let base = idx * 3;
        let dx = self.points[base] - query[0];
        let dy = self.points[base + 1] - query[1];
        let dz = self.points[base + 2] - query[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Scan exactly the cells whose Chebyshev distance from the centre cell is
    /// `ring` (i.e. the surface shell of the `(2*ring+1)^3` cube), feeding the
    /// bounded heap. `ring == 0` scans the single centre cell.
    #[allow(clippy::too_many_arguments)]
    fn scan_ring(
        &self,
        query: &[f32; 3],
        qcx: i32,
        qcy: i32,
        qcz: i32,
        ring: i32,
        k: usize,
        heap: &mut Vec<(f32, usize)>,
    ) {
        if ring == 0 {
            self.scan_cell(query, qcx, qcy, qcz, k, heap);
            return;
        }
        for dz in -ring..=ring {
            let on_z_face = dz.abs() == ring;
            for dy in -ring..=ring {
                let on_y_face = dy.abs() == ring;
                if on_z_face || on_y_face {
                    // Whole x-row lies on the shell.
                    for dx in -ring..=ring {
                        self.scan_cell(query, qcx + dx, qcy + dy, qcz + dz, k, heap);
                    }
                } else {
                    // Interior of this (y,z) plane: only the two x extremes.
                    self.scan_cell(query, qcx - ring, qcy + dy, qcz + dz, k, heap);
                    self.scan_cell(query, qcx + ring, qcy + dy, qcz + dz, k, heap);
                }
            }
        }
    }

    /// Insert every point of one cell into the bounded heap.
    fn scan_cell(
        &self,
        query: &[f32; 3],
        cx: i32,
        cy: i32,
        cz: i32,
        k: usize,
        heap: &mut Vec<(f32, usize)>,
    ) {
        if let Some(bucket) = self.cells.get(&(cx, cy, cz)) {
            for &idx in bucket {
                let d = self.point_sq_dist(idx, query);
                heap_consider(heap, k, d, idx);
            }
        }
    }
}

/// Insert a candidate into the bounded max-heap (largest at index 0).
#[inline]
fn heap_consider(heap: &mut Vec<(f32, usize)>, k: usize, d: f32, idx: usize) {
    if heap.len() < k {
        heap.push((d, idx));
        heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    } else if let Some(worst) = heap.first() {
        if d < worst.0 {
            heap[0] = (d, idx);
            heap.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cloud(n: usize) -> Vec<f32> {
        let mut pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            let a = ((i * 2654435761) % 1000) as f32 / 100.0;
            let b = ((i * 40503 + 7) % 1000) as f32 / 100.0;
            let c = ((i * 2246822519usize) % 1000) as f32 / 100.0;
            pts.push(a);
            pts.push(b);
            pts.push(c);
        }
        pts
    }

    fn cfg(cell: f32) -> GridKnnConfig {
        GridKnnConfig { cell_size: cell }
    }

    fn brute_knn(pts: &[f32], n: usize, q: &[f32; 3], k: usize) -> Vec<(usize, f32)> {
        let mut d: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let dx = pts[i * 3] - q[0];
                let dy = pts[i * 3 + 1] - q[1];
                let dz = pts[i * 3 + 2] - q[2];
                (dx * dx + dy * dy + dz * dz, i)
            })
            .collect();
        d.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        d.into_iter().take(k).map(|(dd, i)| (i, dd)).collect()
    }

    fn brute_radius(pts: &[f32], n: usize, q: &[f32; 3], radius: f32) -> Vec<usize> {
        let r_sq = radius * radius;
        let mut out = Vec::new();
        for i in 0..n {
            let dx = pts[i * 3] - q[0];
            let dy = pts[i * 3 + 1] - q[1];
            let dz = pts[i * 3 + 2] - q[2];
            if dx * dx + dy * dy + dz * dz <= r_sq {
                out.push(i);
            }
        }
        out
    }

    #[test]
    fn knn_matches_brute_force_k1_k3() {
        let n = 200;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        let q = [4.0_f32, 5.0, 3.0];
        for &k in &[1usize, 3] {
            let got = grid.knn(&q, k).unwrap();
            let exp = brute_knn(&pts, n, &q, k);
            assert_eq!(got.len(), exp.len());
            for ((gi, gd), (ei, ed)) in got.iter().zip(exp.iter()) {
                assert_eq!(gi, ei, "knn index mismatch k={k}");
                assert!((gd - ed).abs() < 1e-5, "knn dist mismatch k={k}");
            }
        }
    }

    #[test]
    fn knn_matches_brute_force_various_cells() {
        let n = 150;
        let pts = make_cloud(n);
        let q = [2.3_f32, 7.1, 4.4];
        for &cell in &[0.5_f32, 1.0, 2.5, 5.0] {
            let grid = SpatialHashGrid::build(&pts, n, cfg(cell)).unwrap();
            let got = grid.knn(&q, 5).unwrap();
            let exp = brute_knn(&pts, n, &q, 5);
            for ((gi, _), (ei, _)) in got.iter().zip(exp.iter()) {
                assert_eq!(gi, ei, "cell {cell} knn mismatch");
            }
        }
    }

    #[test]
    fn radius_search_matches_brute_force() {
        let n = 200;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        let q = [5.0_f32, 5.0, 5.0];
        for &radius in &[1.0_f32, 2.5, 4.0] {
            let mut got = grid.radius_search(&q, radius).unwrap();
            got.sort_unstable();
            let mut exp = brute_radius(&pts, n, &q, radius);
            exp.sort_unstable();
            assert_eq!(got, exp, "radius {radius} mismatch");
        }
    }

    #[test]
    fn cell_of_known_coords() {
        let pts = vec![0.0_f32, 0.0, 0.0];
        let grid = SpatialHashGrid::build(&pts, 1, cfg(2.0)).unwrap();
        assert_eq!(grid.cell_of(&[0.0, 0.0, 0.0]), (0, 0, 0));
        assert_eq!(grid.cell_of(&[3.0, 5.0, 1.5]), (1, 2, 0));
        assert_eq!(grid.cell_of(&[4.0, 4.0, 4.0]), (2, 2, 2));
    }

    #[test]
    fn cell_of_negative_coords_floor() {
        let pts = vec![0.0_f32, 0.0, 0.0];
        let grid = SpatialHashGrid::build(&pts, 1, cfg(1.0)).unwrap();
        // floor(-0.5) == -1, floor(-1.0) == -1, floor(-1.5) == -2
        assert_eq!(grid.cell_of(&[-0.5, -1.0, -1.5]), (-1, -1, -2));
        assert_eq!(grid.cell_of(&[-0.001, -2.999, -3.0]), (-1, -3, -3));
    }

    #[test]
    fn len_and_is_empty() {
        let n = 17;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        assert_eq!(grid.len(), n);
        assert!(!grid.is_empty());
        assert!(grid.cell_count() >= 1);
    }

    #[test]
    fn empty_cloud_is_err() {
        let err = SpatialHashGrid::build(&[], 0, cfg(1.0)).unwrap_err();
        assert_eq!(err, Geom3dError::EmptyPointCloud);
    }

    #[test]
    fn knn_k_larger_than_n_returns_n() {
        let n = 6;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        let res = grid.knn(&[3.0, 3.0, 3.0], 100).unwrap();
        assert_eq!(res.len(), n);
    }

    #[test]
    fn knn_exact_when_nn_in_neighbor_cell() {
        // Query cell is empty of the true NN; nearest point sits in an
        // adjacent cell. Ring expansion must still find it.
        let pts = vec![
            0.1_f32, 0.1, 0.1, // cell (0,0,0)
            1.05, 0.5, 0.5, // cell (1,0,0) — true NN of a query near x=1
            5.0, 5.0, 5.0, // far away
        ];
        let grid = SpatialHashGrid::build(&pts, 3, cfg(1.0)).unwrap();
        // Query at (0.95, 0.5, 0.5): own cell (0,0,0) holds index 0 at dist^2
        // ~0.85, but the true NN is index 1 in cell (1,0,0) at dist^2 = 0.01.
        let res = grid.knn(&[0.95, 0.5, 0.5], 1).unwrap();
        assert_eq!(res[0].0, 1, "must find NN in the neighbouring cell");
        assert!(res[0].1 < 0.02);
    }

    #[test]
    fn knn_points_sharing_cell() {
        // Several points in the same cell.
        let pts = vec![
            0.10_f32, 0.10, 0.10, 0.20, 0.20, 0.20, 0.30, 0.30, 0.30, 0.40, 0.40, 0.40,
        ];
        let grid = SpatialHashGrid::build(&pts, 4, cfg(1.0)).unwrap();
        assert_eq!(grid.cell_count(), 1, "all share one cell");
        let got = grid.knn(&[0.0, 0.0, 0.0], 3).unwrap();
        let exp = brute_knn(&pts, 4, &[0.0, 0.0, 0.0], 3);
        for ((gi, _), (ei, _)) in got.iter().zip(exp.iter()) {
            assert_eq!(gi, ei);
        }
    }

    #[test]
    fn deterministic_queries() {
        let n = 120;
        let pts = make_cloud(n);
        let a = SpatialHashGrid::build(&pts, n, cfg(1.5)).unwrap();
        let b = SpatialHashGrid::build(&pts, n, cfg(1.5)).unwrap();
        let q = [3.3_f32, 4.4, 5.5];
        assert_eq!(a.knn(&q, 7).unwrap(), b.knn(&q, 7).unwrap());
        let mut ra = a.radius_search(&q, 2.0).unwrap();
        let mut rb = b.radius_search(&q, 2.0).unwrap();
        ra.sort_unstable();
        rb.sort_unstable();
        assert_eq!(ra, rb);
    }

    #[test]
    fn knn_k1_single_nearest() {
        let n = 64;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        let q = [6.6_f32, 1.2, 8.8];
        let got = grid.knn(&q, 1).unwrap();
        let exp = brute_knn(&pts, n, &q, 1);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, exp[0].0);
        assert!((got[0].1 - exp[0].1).abs() < 1e-5);
    }

    #[test]
    fn radius_zero_at_exact_point() {
        let pts = vec![1.0_f32, 2.0, 3.0, 7.0, 8.0, 9.0];
        let grid = SpatialHashGrid::build(&pts, 2, cfg(1.0)).unwrap();
        let res = grid.radius_search(&[1.0, 2.0, 3.0], 0.0).unwrap();
        assert_eq!(res, vec![0]);
    }

    #[test]
    fn err_points_length_mismatch() {
        let pts = vec![0.0_f32, 0.0, 0.0];
        let err = SpatialHashGrid::build(&pts, 2, cfg(1.0)).unwrap_err();
        assert_eq!(
            err,
            Geom3dError::DimensionMismatch {
                expected: 6,
                got: 3
            }
        );
    }

    #[test]
    fn err_cell_size_non_positive() {
        let pts = make_cloud(4);
        let e0 = SpatialHashGrid::build(&pts, 4, cfg(0.0)).unwrap_err();
        assert_eq!(e0, Geom3dError::InvalidVoxelSize { voxel_size: 0.0 });
        let en = SpatialHashGrid::build(&pts, 4, cfg(-1.0)).unwrap_err();
        assert_eq!(en, Geom3dError::InvalidVoxelSize { voxel_size: -1.0 });
    }

    #[test]
    fn err_k_zero() {
        let n = 10;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        assert_eq!(
            grid.knn(&[0.0, 0.0, 0.0], 0),
            Err(Geom3dError::InvalidK { k: 0, n })
        );
    }

    #[test]
    fn err_radius_negative() {
        let n = 10;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        assert_eq!(
            grid.radius_search(&[0.0, 0.0, 0.0], -2.0),
            Err(Geom3dError::InvalidRadius { radius: -2.0 })
        );
    }

    #[test]
    fn sparse_cloud_finds_nn_via_ring_expansion() {
        // Points far apart with large empty regions between cells.
        let pts = vec![
            0.0_f32, 0.0, 0.0, // cell (0,0,0)
            50.0, 0.0, 0.0, // cell (50,0,0)
            0.0, 80.0, 0.0, // cell (0,80,0)
            100.0, 100.0, 100.0,
        ];
        let grid = SpatialHashGrid::build(&pts, 4, cfg(1.0)).unwrap();
        // Query near the second point; many empty rings lie between it and the
        // populated cell, but ring expansion (capped) must still locate it.
        let q = [48.0_f32, 1.0, 0.0];
        let res = grid.knn(&q, 1).unwrap();
        assert_eq!(res[0].0, 1, "must find the only nearby point");
        let exp = brute_knn(&pts, 4, &q, 1);
        assert_eq!(res[0].0, exp[0].0);
    }

    #[test]
    fn two_equidistant_points_stable() {
        // Two points exactly equidistant from the query along ±x.
        let pts = vec![-1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let grid = SpatialHashGrid::build(&pts, 3, cfg(1.0)).unwrap();
        let res = grid.knn(&[0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(res.len(), 2);
        // Both nearest have equal squared distance 1.0.
        assert!((res[0].1 - 1.0).abs() < 1e-5);
        assert!((res[1].1 - 1.0).abs() < 1e-5);
        let mut idxs = vec![res[0].0, res[1].0];
        idxs.sort_unstable();
        assert_eq!(idxs, vec![0, 1]);
    }

    #[test]
    fn knn_matches_kdtree() {
        use crate::neighborhood::kd_tree::KdTree;
        let n = 130;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        let kd = KdTree::build(&pts, n).unwrap();
        let q = [4.2_f32, 6.6, 2.2];
        let g = grid.knn(&q, 6).unwrap();
        let k = kd.knn(q, 6).unwrap();
        for ((gi, gd), (ki, kd_d)) in g.iter().zip(k.iter()) {
            assert_eq!(gi, ki);
            assert!((gd - kd_d).abs() < 1e-5);
        }
    }

    #[test]
    fn radius_search_empty_far_away() {
        let n = 30;
        let pts = make_cloud(n);
        let grid = SpatialHashGrid::build(&pts, n, cfg(1.0)).unwrap();
        let res = grid.radius_search(&[1000.0, 1000.0, 1000.0], 1.0).unwrap();
        assert!(res.is_empty());
    }
}
