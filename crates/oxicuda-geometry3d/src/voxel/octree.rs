//! Octree-based hierarchical voxelisation for very large point clouds.
//!
//! An [`Octree`] recursively subdivides an axis-aligned bounding box (AABB)
//! into eight equal octants until each leaf contains at most
//! `max_points_per_leaf` points or the maximum depth is reached. This gives
//! logarithmic-depth spatial queries (radius / kNN) with AABB-based pruning,
//! which is dramatically faster than a brute-force scan on large, spatially
//! coherent clouds.
//!
//! # Distance conventions
//!
//! * [`Octree::query_radius`] uses **plain Euclidean** distance and returns the
//!   indices of every point whose distance to the query centre is
//!   `<= radius`.
//! * [`Octree::query_knn`] returns `(index, squared_distance)` pairs sorted by
//!   ascending distance, matching the crate-wide kNN convention used by
//!   [`crate::neighborhood::kd_tree::KdTree`] and
//!   [`crate::neighborhood::knn::knn`] (distances are **squared**).

use crate::error::{Geom3dError, Geom3dResult};

/// Configuration controlling octree construction.
#[derive(Debug, Clone, PartialEq)]
pub struct OctreeConfig {
    /// Maximum subdivision depth. `0` produces a single leaf holding all points.
    pub max_depth: usize,
    /// A node with this many points or fewer becomes a leaf (when depth allows).
    pub max_points_per_leaf: usize,
    /// Minimum corner of the root AABB (inclusive).
    pub bounds_min: [f32; 3],
    /// Maximum corner of the root AABB (inclusive).
    pub bounds_max: [f32; 3],
}

/// A node in the octree: either a leaf holding point indices or an internal
/// node with up to eight children.
#[derive(Debug, Clone)]
pub enum OctreeNode {
    /// Terminal node storing the indices of the points it contains.
    Leaf {
        /// Indices (into the original point cloud) contained in this leaf.
        point_indices: Vec<usize>,
    },
    /// Internal node with eight octant children plus its centre / half-extent.
    Internal {
        /// Children indexed by octant bitmask `(x>=cx)|((y>=cy)<<1)|((z>=cz)<<2)`.
        children: Box<[Option<OctreeNode>; 8]>,
        /// Centre of this node's AABB.
        center: [f32; 3],
        /// Half-extent of this node's AABB along each axis.
        half_size: [f32; 3],
    },
}

/// A hierarchical octree built over a flat row-major `[n×3]` point cloud.
#[derive(Debug)]
pub struct Octree {
    root: OctreeNode,
    config: OctreeConfig,
    points: Vec<f32>,
    n: usize,
}

impl Octree {
    /// Build an octree over `n` 3-D points (`points` is flat row-major `[n×3]`).
    ///
    /// # Errors
    ///
    /// * [`Geom3dError::EmptyPointCloud`] if `n == 0`.
    /// * [`Geom3dError::DimensionMismatch`] if `points.len() != n * 3`.
    /// * [`Geom3dError::InvalidTopology`] if any `bounds_min[d] >= bounds_max[d]`.
    pub fn build(points: &[f32], n: usize, cfg: OctreeConfig) -> Geom3dResult<Self> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if points.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: points.len(),
            });
        }
        for d in 0..3 {
            let ordered = matches!(
                cfg.bounds_min[d].partial_cmp(&cfg.bounds_max[d]),
                Some(std::cmp::Ordering::Less)
            );
            if !ordered || !cfg.bounds_min[d].is_finite() || !cfg.bounds_max[d].is_finite() {
                return Err(Geom3dError::InvalidTopology {
                    reason: "bounds_min must be strictly less than bounds_max on every axis",
                });
            }
        }

        let center = [
            0.5 * (cfg.bounds_min[0] + cfg.bounds_max[0]),
            0.5 * (cfg.bounds_min[1] + cfg.bounds_max[1]),
            0.5 * (cfg.bounds_min[2] + cfg.bounds_max[2]),
        ];
        let half_size = [
            0.5 * (cfg.bounds_max[0] - cfg.bounds_min[0]),
            0.5 * (cfg.bounds_max[1] - cfg.bounds_min[1]),
            0.5 * (cfg.bounds_max[2] - cfg.bounds_min[2]),
        ];

        let all_indices: Vec<usize> = (0..n).collect();
        let root = build_node(
            points,
            all_indices,
            center,
            half_size,
            0,
            cfg.max_depth,
            cfg.max_points_per_leaf,
        );

        Ok(Self {
            root,
            config: cfg,
            points: points.to_vec(),
            n,
        })
    }

    /// Return every point index within `radius` (Euclidean, inclusive) of
    /// `center`.
    ///
    /// # Errors
    ///
    /// [`Geom3dError::InvalidRadius`] if `radius` is negative or non-finite.
    pub fn query_radius(&self, center: &[f32; 3], radius: f32) -> Geom3dResult<Vec<usize>> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(Geom3dError::InvalidRadius { radius });
        }
        let r_sq = radius * radius;
        let mut out = Vec::new();
        radius_recurse(&self.root, &self.points, center, r_sq, &mut out);
        Ok(out)
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
    pub fn query_knn(&self, query: &[f32; 3], k: usize) -> Geom3dResult<Vec<(usize, f32)>> {
        if k == 0 {
            return Err(Geom3dError::InvalidK { k, n: self.n });
        }
        let kk = k.min(self.n);
        // Bounded max-heap as a Vec; entry 0 holds the current worst (largest).
        let mut heap: Vec<(f32, usize)> = Vec::with_capacity(kk + 1);
        knn_recurse(&self.root, &self.points, query, kk, &mut heap);
        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(heap.into_iter().map(|(d, i)| (i, d)).collect())
    }

    /// Total number of nodes (leaves + internal) in the tree.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        count_nodes(&self.root)
    }

    /// Actual depth of the tree (a single-leaf tree has depth `0`).
    #[must_use]
    pub fn depth(&self) -> usize {
        node_depth(&self.root)
    }

    /// Number of leaf nodes in the tree.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        count_leaves(&self.root)
    }

    /// Return the configuration used to build this tree.
    #[must_use]
    pub fn config(&self) -> &OctreeConfig {
        &self.config
    }

    /// Number of points indexed by this tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the tree indexes zero points (always `false` after a successful
    /// [`Octree::build`], which rejects empty clouds).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Octant bitmask for `p` relative to the node centre.
#[inline]
fn octant_of(p: &[f32], base: usize, center: &[f32; 3]) -> usize {
    let bx = usize::from(p[base] >= center[0]);
    let by = usize::from(p[base + 1] >= center[1]);
    let bz = usize::from(p[base + 2] >= center[2]);
    bx | (by << 1) | (bz << 2)
}

/// Centre of child octant `octant` given the parent centre / half-size.
#[inline]
fn child_center(center: &[f32; 3], child_half: &[f32; 3], octant: usize) -> [f32; 3] {
    let sx = if octant & 1 == 0 { -1.0 } else { 1.0 };
    let sy = if octant & 2 == 0 { -1.0 } else { 1.0 };
    let sz = if octant & 4 == 0 { -1.0 } else { 1.0 };
    [
        center[0] + sx * child_half[0],
        center[1] + sy * child_half[1],
        center[2] + sz * child_half[2],
    ]
}

/// Whether every point referenced by `indices` has identical coordinates.
/// Used to halt subdivision on coincident points (which can never be split).
fn all_coincident(points: &[f32], indices: &[usize]) -> bool {
    let Some(&first) = indices.first() else {
        return true;
    };
    let base = first * 3;
    let (x0, y0, z0) = (points[base], points[base + 1], points[base + 2]);
    indices.iter().skip(1).all(|&idx| {
        let b = idx * 3;
        points[b] == x0 && points[b + 1] == y0 && points[b + 2] == z0
    })
}

/// Recursively build a node from the given indices spanning the AABB defined by
/// `center` / `half_size`.
fn build_node(
    points: &[f32],
    indices: Vec<usize>,
    center: [f32; 3],
    half_size: [f32; 3],
    depth: usize,
    max_depth: usize,
    max_points_per_leaf: usize,
) -> OctreeNode {
    if indices.len() <= max_points_per_leaf || depth >= max_depth {
        return OctreeNode::Leaf {
            point_indices: indices,
        };
    }

    // Avoid infinite recursion when all points are coincident: subdivision can
    // never separate identical coordinates, so terminate as a leaf. (Distinct
    // points that merely share an octant at this level are still subdivided —
    // they separate at a finer level, bounded by `max_depth`.)
    if all_coincident(points, &indices) {
        return OctreeNode::Leaf {
            point_indices: indices,
        };
    }

    let child_half = [half_size[0] * 0.5, half_size[1] * 0.5, half_size[2] * 0.5];
    let mut buckets: [Vec<usize>; 8] = Default::default();
    for &idx in &indices {
        let oct = octant_of(points, idx * 3, &center);
        buckets[oct].push(idx);
    }

    let mut children: [Option<OctreeNode>; 8] = Default::default();
    for (oct, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let cc = child_center(&center, &child_half, oct);
        children[oct] = Some(build_node(
            points,
            bucket,
            cc,
            child_half,
            depth + 1,
            max_depth,
            max_points_per_leaf,
        ));
    }

    OctreeNode::Internal {
        children: Box::new(children),
        center,
        half_size,
    }
}

/// Squared distance from `query` to point `idx`.
#[inline]
fn point_sq_dist(points: &[f32], idx: usize, query: &[f32; 3]) -> f32 {
    let base = idx * 3;
    let dx = points[base] - query[0];
    let dy = points[base + 1] - query[1];
    let dz = points[base + 2] - query[2];
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from `query` to the closest point on the AABB centred at
/// `center` with the given `half_size`. Returns `0` if `query` is inside.
#[inline]
fn aabb_closest_sq_dist(center: &[f32; 3], half_size: &[f32; 3], query: &[f32; 3]) -> f32 {
    let mut acc = 0.0_f32;
    for d in 0..3 {
        let lo = center[d] - half_size[d];
        let hi = center[d] + half_size[d];
        let q = query[d];
        let diff = if q < lo {
            lo - q
        } else if q > hi {
            q - hi
        } else {
            0.0
        };
        acc += diff * diff;
    }
    acc
}

/// Recursive radius collection with AABB pruning.
fn radius_recurse(
    node: &OctreeNode,
    points: &[f32],
    center: &[f32; 3],
    r_sq: f32,
    out: &mut Vec<usize>,
) {
    match node {
        OctreeNode::Leaf { point_indices } => {
            for &idx in point_indices {
                // Inclusive Euclidean test; compare squared to avoid sqrt.
                if point_sq_dist(points, idx, center) <= r_sq {
                    out.push(idx);
                }
            }
        }
        OctreeNode::Internal {
            children,
            center: node_center,
            half_size,
        } => {
            // Prune if the sphere cannot reach this node's AABB at all.
            if aabb_closest_sq_dist(node_center, half_size, center) > r_sq {
                return;
            }
            let child_half = [half_size[0] * 0.5, half_size[1] * 0.5, half_size[2] * 0.5];
            for (oct, child) in children.iter().enumerate() {
                if let Some(child_node) = child {
                    let cc = child_center(node_center, &child_half, oct);
                    if aabb_closest_sq_dist(&cc, &child_half, center) <= r_sq {
                        radius_recurse(child_node, points, center, r_sq, out);
                    }
                }
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

/// Current k-th best squared distance (`+inf` while the heap is not yet full).
#[inline]
fn heap_worst(heap: &[(f32, usize)], k: usize) -> f32 {
    if heap.len() < k {
        f32::INFINITY
    } else {
        heap.first().map_or(f32::INFINITY, |h| h.0)
    }
}

/// Recursive kNN with AABB pruning against the current k-th best distance.
fn knn_recurse(
    node: &OctreeNode,
    points: &[f32],
    query: &[f32; 3],
    k: usize,
    heap: &mut Vec<(f32, usize)>,
) {
    match node {
        OctreeNode::Leaf { point_indices } => {
            for &idx in point_indices {
                let d = point_sq_dist(points, idx, query);
                heap_consider(heap, k, d, idx);
            }
        }
        OctreeNode::Internal {
            children,
            center: node_center,
            half_size,
        } => {
            let child_half = [half_size[0] * 0.5, half_size[1] * 0.5, half_size[2] * 0.5];
            // Visit children ordered by proximity to the query for tighter
            // pruning: closer octants populate the heap first.
            let mut order: [(f32, usize); 8] = [(f32::INFINITY, 0); 8];
            for (oct, child) in children.iter().enumerate() {
                let dist = if child.is_some() {
                    let cc = child_center(node_center, &child_half, oct);
                    aabb_closest_sq_dist(&cc, &child_half, query)
                } else {
                    f32::INFINITY
                };
                order[oct] = (dist, oct);
            }
            order.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (dist, oct) in order {
                if dist > heap_worst(heap, k) {
                    break;
                }
                if let Some(child_node) = &children[oct] {
                    knn_recurse(child_node, points, query, k, heap);
                }
            }
        }
    }
}

/// Total node count (this node plus all descendants).
fn count_nodes(node: &OctreeNode) -> usize {
    match node {
        OctreeNode::Leaf { .. } => 1,
        OctreeNode::Internal { children, .. } => {
            1 + children
                .iter()
                .filter_map(|c| c.as_ref())
                .map(count_nodes)
                .sum::<usize>()
        }
    }
}

/// Leaf count below (and including) this node.
fn count_leaves(node: &OctreeNode) -> usize {
    match node {
        OctreeNode::Leaf { .. } => 1,
        OctreeNode::Internal { children, .. } => children
            .iter()
            .filter_map(|c| c.as_ref())
            .map(count_leaves)
            .sum::<usize>(),
    }
}

/// Depth of the subtree (a leaf has depth `0`).
fn node_depth(node: &OctreeNode) -> usize {
    match node {
        OctreeNode::Leaf { .. } => 0,
        OctreeNode::Internal { children, .. } => {
            1 + children
                .iter()
                .filter_map(|c| c.as_ref())
                .map(node_depth)
                .max()
                .unwrap_or(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small cloud spread inside the unit cube.
    fn make_cloud(n: usize) -> Vec<f32> {
        // Cheap deterministic pseudo-spread without external RNG.
        let mut pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            let a = ((i * 2654435761) % 1000) as f32 / 1000.0;
            let b = ((i * 40503 + 7) % 1000) as f32 / 1000.0;
            let c = ((i * 2246822519usize) % 1000) as f32 / 1000.0;
            pts.push(a);
            pts.push(b);
            pts.push(c);
        }
        pts
    }

    fn unit_cfg(max_depth: usize, max_pts: usize) -> OctreeConfig {
        OctreeConfig {
            max_depth,
            max_points_per_leaf: max_pts,
            bounds_min: [0.0, 0.0, 0.0],
            bounds_max: [1.0, 1.0, 1.0],
        }
    }

    fn brute_radius(pts: &[f32], n: usize, center: &[f32; 3], radius: f32) -> Vec<usize> {
        let r_sq = radius * radius;
        let mut out = Vec::new();
        for i in 0..n {
            let dx = pts[i * 3] - center[0];
            let dy = pts[i * 3 + 1] - center[1];
            let dz = pts[i * 3 + 2] - center[2];
            if dx * dx + dy * dy + dz * dz <= r_sq {
                out.push(i);
            }
        }
        out
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

    #[test]
    fn build_all_indices_present_once() {
        let n = 200;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(8, 4)).unwrap();
        let radius = brute_radius(&pts, n, &[0.5, 0.5, 0.5], 100.0);
        // Sanity: brute force inside huge radius is all indices.
        assert_eq!(radius.len(), n);
        // Collect leaf indices via a giant radius query (covers whole cube).
        let mut all = tree.query_radius(&[0.5, 0.5, 0.5], 100.0).unwrap();
        all.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(all, expected, "every index must appear exactly once");
    }

    #[test]
    fn build_leaf_partition_exact() {
        // Directly walk the tree gathering leaf indices to confirm partition.
        let n = 64;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(6, 2)).unwrap();
        let mut collected = Vec::new();
        fn walk(node: &OctreeNode, out: &mut Vec<usize>) {
            match node {
                OctreeNode::Leaf { point_indices } => out.extend_from_slice(point_indices),
                OctreeNode::Internal { children, .. } => {
                    for c in children.iter().flatten() {
                        walk(c, out);
                    }
                }
            }
        }
        walk(&tree.root, &mut collected);
        collected.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn query_radius_matches_brute_force() {
        let n = 150;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(8, 3)).unwrap();
        let center = [0.4_f32, 0.6, 0.5];
        for &radius in &[0.1_f32, 0.25, 0.5] {
            let mut got = tree.query_radius(&center, radius).unwrap();
            got.sort_unstable();
            let mut exp = brute_radius(&pts, n, &center, radius);
            exp.sort_unstable();
            assert_eq!(got, exp, "radius {radius} mismatch");
        }
    }

    #[test]
    fn query_knn_matches_brute_force() {
        let n = 120;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(8, 2)).unwrap();
        let q = [0.55_f32, 0.45, 0.5];
        for &k in &[1usize, 3, 5, 10] {
            let got = tree.query_knn(&q, k).unwrap();
            let exp = brute_knn(&pts, n, &q, k);
            assert_eq!(got.len(), exp.len());
            for ((gi, gd), (ei, ed)) in got.iter().zip(exp.iter()) {
                assert_eq!(gi, ei, "knn index mismatch at k={k}");
                assert!((gd - ed).abs() < 1e-6, "knn dist mismatch at k={k}");
            }
        }
    }

    #[test]
    fn query_knn_sorted_ascending() {
        let n = 80;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(6, 3)).unwrap();
        let res = tree.query_knn(&[0.3, 0.3, 0.3], 7).unwrap();
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "knn must be ascending");
        }
    }

    #[test]
    fn depth_at_most_max_depth() {
        let n = 256;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(4, 1)).unwrap();
        assert!(tree.depth() <= 4, "depth {} exceeds max", tree.depth());
    }

    #[test]
    fn counts_are_positive() {
        let n = 50;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(5, 4)).unwrap();
        assert!(tree.leaf_count() >= 1);
        assert!(tree.n_nodes() >= 1);
        assert_eq!(tree.len(), n);
        assert!(!tree.is_empty());
    }

    #[test]
    fn single_point_knn_zero_distance() {
        let pts = vec![0.5_f32, 0.5, 0.5];
        let tree = Octree::build(&pts, 1, unit_cfg(4, 1)).unwrap();
        let res = tree.query_knn(&[0.5, 0.5, 0.5], 1).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        assert!(res[0].1 < 1e-9);
        // A single-point cloud is one leaf at depth 0.
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.n_nodes(), 1);
    }

    #[test]
    fn knn_k_larger_than_n_returns_n() {
        let n = 7;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(4, 2)).unwrap();
        let res = tree.query_knn(&[0.5, 0.5, 0.5], 100).unwrap();
        assert_eq!(res.len(), n);
    }

    #[test]
    fn radius_zero_at_exact_point() {
        let pts = vec![0.2_f32, 0.3, 0.4, 0.7, 0.8, 0.9];
        let tree = Octree::build(&pts, 2, unit_cfg(4, 1)).unwrap();
        let res = tree.query_radius(&[0.2, 0.3, 0.4], 0.0).unwrap();
        assert_eq!(res, vec![0]);
    }

    #[test]
    fn deterministic_build() {
        let n = 100;
        let pts = make_cloud(n);
        let a = Octree::build(&pts, n, unit_cfg(6, 3)).unwrap();
        let b = Octree::build(&pts, n, unit_cfg(6, 3)).unwrap();
        assert_eq!(a.n_nodes(), b.n_nodes());
        assert_eq!(a.depth(), b.depth());
        assert_eq!(a.leaf_count(), b.leaf_count());
        let q = [0.33_f32, 0.66, 0.5];
        assert_eq!(a.query_knn(&q, 5).unwrap(), b.query_knn(&q, 5).unwrap());
    }

    #[test]
    fn points_on_octant_boundary_not_lost() {
        // Points lying exactly on the centre planes of the root.
        let pts = vec![
            0.5_f32, 0.5, 0.5, // dead centre
            0.5, 0.25, 0.75, // on x-plane
            0.25, 0.5, 0.25, // on y-plane
            0.75, 0.75, 0.5, // on z-plane
        ];
        let tree = Octree::build(&pts, 4, unit_cfg(6, 1)).unwrap();
        let mut all = tree.query_radius(&[0.5, 0.5, 0.5], 100.0).unwrap();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3]);
    }

    #[test]
    fn err_points_length_mismatch() {
        let pts = vec![0.0_f32, 0.0, 0.0];
        let err = Octree::build(&pts, 2, unit_cfg(4, 1)).unwrap_err();
        assert_eq!(
            err,
            Geom3dError::DimensionMismatch {
                expected: 6,
                got: 3
            }
        );
    }

    #[test]
    fn err_empty_cloud() {
        let err = Octree::build(&[], 0, unit_cfg(4, 1)).unwrap_err();
        assert_eq!(err, Geom3dError::EmptyPointCloud);
    }

    #[test]
    fn err_k_zero() {
        let pts = make_cloud(10);
        let tree = Octree::build(&pts, 10, unit_cfg(4, 2)).unwrap();
        assert_eq!(
            tree.query_knn(&[0.0, 0.0, 0.0], 0),
            Err(Geom3dError::InvalidK { k: 0, n: 10 })
        );
    }

    #[test]
    fn err_radius_negative() {
        let pts = make_cloud(10);
        let tree = Octree::build(&pts, 10, unit_cfg(4, 2)).unwrap();
        assert_eq!(
            tree.query_radius(&[0.0, 0.0, 0.0], -1.0),
            Err(Geom3dError::InvalidRadius { radius: -1.0 })
        );
    }

    #[test]
    fn err_bounds_min_ge_max() {
        let pts = make_cloud(4);
        let bad = OctreeConfig {
            max_depth: 4,
            max_points_per_leaf: 1,
            bounds_min: [0.0, 0.0, 0.0],
            bounds_max: [1.0, 0.0, 1.0], // y collapsed
        };
        assert!(Octree::build(&pts, 4, bad).is_err());
    }

    #[test]
    fn empty_far_radius_query() {
        let n = 30;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(6, 2)).unwrap();
        // Query far away from the unit cube.
        let res = tree.query_radius(&[100.0, 100.0, 100.0], 1.0).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn max_points_per_leaf_one_forces_subdivision() {
        // Two points clustered in the SAME first-level octant (both well below
        // centre 0.5 on every axis) that only separate at a finer level, plus
        // two others. Leaf capacity 1 must therefore subdivide beyond depth 1.
        let pts = vec![
            0.10_f32, 0.10, 0.10, // octant 0 (low,low,low)
            0.20, 0.20, 0.20, // octant 0 as well -> needs deeper split
            0.90, 0.90, 0.90, // octant 7
            0.90, 0.10, 0.10, // octant 1
        ];
        let tree = Octree::build(&pts, 4, unit_cfg(10, 1)).unwrap();
        assert!(tree.depth() > 1, "clustered points must subdivide deeply");
        assert!(tree.leaf_count() >= 2);
    }

    #[test]
    fn max_depth_zero_single_leaf() {
        let n = 40;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(0, 1)).unwrap();
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.n_nodes(), 1);
        // Queries must still work on a single leaf.
        let res = tree.query_knn(&[0.5, 0.5, 0.5], 3).unwrap();
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn duplicate_points_terminate() {
        // All identical points would collapse into one octant forever without
        // the single-octant leaf guard.
        let pts = vec![0.5_f32; 30]; // 10 identical points
        let tree = Octree::build(&pts, 10, unit_cfg(20, 1)).unwrap();
        let res = tree.query_knn(&[0.5, 0.5, 0.5], 10).unwrap();
        assert_eq!(res.len(), 10);
        assert!(res.iter().all(|&(_, d)| d < 1e-9));
    }

    #[test]
    fn config_accessor_roundtrip() {
        let n = 12;
        let pts = make_cloud(n);
        let cfg = unit_cfg(5, 3);
        let tree = Octree::build(&pts, n, cfg.clone()).unwrap();
        assert_eq!(tree.config(), &cfg);
    }

    #[test]
    fn knn_matches_kdtree_on_random_cloud() {
        // Cross-check against the existing KdTree for confidence.
        use crate::neighborhood::kd_tree::KdTree;
        let n = 90;
        let pts = make_cloud(n);
        let tree = Octree::build(&pts, n, unit_cfg(8, 2)).unwrap();
        let kd = KdTree::build(&pts, n).unwrap();
        let q = [0.41_f32, 0.59, 0.5];
        let oct = tree.query_knn(&q, 4).unwrap();
        let kdr = kd.knn(q, 4).unwrap();
        for ((oi, od), (ki, kd_d)) in oct.iter().zip(kdr.iter()) {
            assert_eq!(oi, ki);
            assert!((od - kd_d).abs() < 1e-6);
        }
    }
}
