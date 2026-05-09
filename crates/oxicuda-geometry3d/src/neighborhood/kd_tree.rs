//! KD-tree for 3D point cloud nearest neighbor search.

use crate::error::{Geom3dError, Geom3dResult};

/// A node in the KD-tree.
#[derive(Debug, Clone)]
enum KdNode {
    Leaf {
        idx: usize,
    },
    Split {
        axis: u8,
        value: f32,
        left: usize,
        right: usize,
    },
}

/// A KD-tree built over a 3D point cloud.
#[derive(Debug)]
pub struct KdTree {
    nodes: Vec<KdNode>,
    points_ref: Vec<f32>,
}

impl KdTree {
    /// Build a KD-tree from `n` 3D points.
    ///
    /// `points`: flat row-major `[n×3]`.
    pub fn build(points: &[f32], n: usize) -> Geom3dResult<Self> {
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if points.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: points.len(),
            });
        }

        let mut nodes = Vec::new();
        let mut point_indices: Vec<usize> = (0..n).collect();

        build_recursive(points, &mut point_indices, 0, n, 0, &mut nodes);

        Ok(Self {
            nodes,
            points_ref: points.to_vec(),
        })
    }

    /// Find the nearest point to `query`.
    ///
    /// Returns `(index, sq_dist)`.
    pub fn nearest(&self, query: [f32; 3]) -> Geom3dResult<(usize, f32)> {
        if self.nodes.is_empty() {
            return Err(Geom3dError::EmptyPointCloud);
        }
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        search_nearest(
            &self.nodes,
            &self.points_ref,
            0,
            query,
            &mut best_idx,
            &mut best_dist,
        );
        Ok((best_idx, best_dist))
    }

    /// Find `k` nearest points to `query`.
    ///
    /// Returns sorted `Vec<(index, sq_dist)>` from nearest to farthest.
    pub fn knn(&self, query: [f32; 3], k: usize) -> Geom3dResult<Vec<(usize, f32)>> {
        if self.nodes.is_empty() {
            return Err(Geom3dError::EmptyPointCloud);
        }
        let n = self.points_ref.len() / 3;
        if k > n {
            return Err(Geom3dError::InvalidK { k, n });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        // Use a max-heap maintained as Vec<(f32, usize)> sorted by distance descending
        let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k + 1);

        search_knn(&self.nodes, &self.points_ref, 0, query, k, &mut heap);

        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(heap.into_iter().map(|(d, i)| (i, d)).collect())
    }

    /// Find all points within `radius` of `query`.
    ///
    /// Returns unsorted `Vec<(index, sq_dist)>`.
    pub fn radius_search(&self, query: [f32; 3], radius: f32) -> Geom3dResult<Vec<(usize, f32)>> {
        if self.nodes.is_empty() {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if radius <= 0.0 || !radius.is_finite() {
            return Err(Geom3dError::InvalidRadius { radius });
        }
        let r_sq = radius * radius;
        let mut results = Vec::new();
        search_radius(&self.nodes, &self.points_ref, 0, query, r_sq, &mut results);
        Ok(results)
    }
}

/// Recursive KD-tree build. Returns root node index.
fn build_recursive(
    points: &[f32],
    indices: &mut [usize],
    start: usize,
    end: usize,
    depth: usize,
    nodes: &mut Vec<KdNode>,
) -> usize {
    let count = end - start;
    if count == 1 {
        let node_idx = nodes.len();
        nodes.push(KdNode::Leaf {
            idx: indices[start],
        });
        return node_idx;
    }

    // Choose axis and sort by median
    let axis = (depth % 3) as u8;

    // Sort indices[start..end] by the chosen axis coordinate
    let sub = &mut indices[start..end];
    sub.sort_unstable_by(|&a, &b| {
        let av = points[a * 3 + axis as usize];
        let bv = points[b * 3 + axis as usize];
        av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = start + count / 2;
    let split_value = points[indices[mid] * 3 + axis as usize];

    // Reserve slot for this split node
    let node_idx = nodes.len();
    nodes.push(KdNode::Leaf { idx: 0 }); // placeholder

    let left = build_recursive(points, indices, start, mid, depth + 1, nodes);
    let right = build_recursive(points, indices, mid, end, depth + 1, nodes);

    nodes[node_idx] = KdNode::Split {
        axis,
        value: split_value,
        left,
        right,
    };
    node_idx
}

fn sq_dist(points: &[f32], idx: usize, query: [f32; 3]) -> f32 {
    let dx = points[idx * 3] - query[0];
    let dy = points[idx * 3 + 1] - query[1];
    let dz = points[idx * 3 + 2] - query[2];
    dx * dx + dy * dy + dz * dz
}

fn search_nearest(
    nodes: &[KdNode],
    points: &[f32],
    node_idx: usize,
    query: [f32; 3],
    best_idx: &mut usize,
    best_dist: &mut f32,
) {
    match &nodes[node_idx] {
        KdNode::Leaf { idx } => {
            let d = sq_dist(points, *idx, query);
            if d < *best_dist {
                *best_dist = d;
                *best_idx = *idx;
            }
        }
        KdNode::Split {
            axis,
            value,
            left,
            right,
        } => {
            let q_val = query[*axis as usize];
            let diff = q_val - value;
            let (near, far) = if diff <= 0.0 {
                (*left, *right)
            } else {
                (*right, *left)
            };

            search_nearest(nodes, points, near, query, best_idx, best_dist);

            if diff * diff < *best_dist {
                search_nearest(nodes, points, far, query, best_idx, best_dist);
            }
        }
    }
}

fn search_knn(
    nodes: &[KdNode],
    points: &[f32],
    node_idx: usize,
    query: [f32; 3],
    k: usize,
    heap: &mut Vec<(f32, usize)>,
) {
    match &nodes[node_idx] {
        KdNode::Leaf { idx } => {
            let d = sq_dist(points, *idx, query);
            if heap.len() < k {
                heap.push((d, *idx));
                // bubble up to maintain max at end
                heap.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else if let Some(worst) = heap.first() {
                if d < worst.0 {
                    heap[0] = (d, *idx);
                    heap.sort_unstable_by(|a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }
        KdNode::Split {
            axis,
            value,
            left,
            right,
        } => {
            let q_val = query[*axis as usize];
            let diff = q_val - value;
            let (near, far) = if diff <= 0.0 {
                (*left, *right)
            } else {
                (*right, *left)
            };

            search_knn(nodes, points, near, query, k, heap);

            let worst_dist = heap.first().map(|h| h.0).unwrap_or(f32::INFINITY);
            if heap.len() < k || diff * diff < worst_dist {
                search_knn(nodes, points, far, query, k, heap);
            }
        }
    }
}

fn search_radius(
    nodes: &[KdNode],
    points: &[f32],
    node_idx: usize,
    query: [f32; 3],
    r_sq: f32,
    results: &mut Vec<(usize, f32)>,
) {
    match &nodes[node_idx] {
        KdNode::Leaf { idx } => {
            let d = sq_dist(points, *idx, query);
            if d < r_sq {
                results.push((*idx, d));
            }
        }
        KdNode::Split {
            axis,
            value,
            left,
            right,
        } => {
            let q_val = query[*axis as usize];
            let diff = q_val - value;
            let (near, far) = if diff <= 0.0 {
                (*left, *right)
            } else {
                (*right, *left)
            };

            search_radius(nodes, points, near, query, r_sq, results);
            if diff * diff < r_sq {
                search_radius(nodes, points, far, query, r_sq, results);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect()
    }

    #[test]
    fn kdtree_build_empty_error() {
        assert!(KdTree::build(&[], 0).is_err());
    }

    #[test]
    fn kdtree_nearest_single() {
        let pts = vec![1.0_f32, 2.0, 3.0];
        let tree = KdTree::build(&pts, 1).unwrap();
        let (idx, d) = tree.nearest([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(idx, 0);
        assert!(d < 1e-6);
    }

    #[test]
    fn kdtree_nearest_correct() {
        let pts = make_line(10);
        let tree = KdTree::build(&pts, 10).unwrap();
        let (idx, _) = tree.nearest([3.1, 0.0, 0.0]).unwrap();
        assert_eq!(idx, 3, "Nearest to 3.1 should be index 3");
    }

    #[test]
    fn kdtree_nearest_matches_brute_force() {
        let n = 50;
        let pts: Vec<f32> = (0..n)
            .flat_map(|i| vec![i as f32 * 0.3, (i % 5) as f32, (i % 3) as f32])
            .collect();
        let tree = KdTree::build(&pts, n).unwrap();
        let query = [7.2_f32, 2.1, 0.9];

        let (tree_idx, _) = tree.nearest(query).unwrap();

        // Brute force
        let bf_idx = (0..n)
            .min_by(|&a, &b| {
                let da = {
                    let dx = pts[a * 3] - query[0];
                    let dy = pts[a * 3 + 1] - query[1];
                    let dz = pts[a * 3 + 2] - query[2];
                    dx * dx + dy * dy + dz * dz
                };
                let db = {
                    let dx = pts[b * 3] - query[0];
                    let dy = pts[b * 3 + 1] - query[1];
                    let dz = pts[b * 3 + 2] - query[2];
                    dx * dx + dy * dy + dz * dz
                };
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        assert_eq!(tree_idx, bf_idx, "KD-tree nearest must match brute force");
    }

    #[test]
    fn kdtree_knn_correct_count() {
        let pts = make_line(20);
        let tree = KdTree::build(&pts, 20).unwrap();
        let result = tree.knn([9.5, 0.0, 0.0], 5).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn kdtree_knn_sorted() {
        let pts = make_line(20);
        let tree = KdTree::build(&pts, 20).unwrap();
        let result = tree.knn([5.5, 0.0, 0.0], 4).unwrap();
        for w in result.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "KNN results must be sorted by distance ascending"
            );
        }
    }

    #[test]
    fn kdtree_knn_k_exceeds_n_error() {
        let pts = make_line(5);
        let tree = KdTree::build(&pts, 5).unwrap();
        assert_eq!(
            tree.knn([0.0, 0.0, 0.0], 10),
            Err(Geom3dError::InvalidK { k: 10, n: 5 })
        );
    }

    #[test]
    fn kdtree_radius_search_correct() {
        let pts = make_line(10);
        let tree = KdTree::build(&pts, 10).unwrap();
        let results = tree.radius_search([4.5, 0.0, 0.0], 1.6).unwrap();
        // Should find 3,4,5,6 (within radius 1.6 of 4.5)
        let mut found: Vec<usize> = results.iter().map(|&(i, _)| i).collect();
        found.sort_unstable();
        assert_eq!(
            found,
            vec![3, 4, 5, 6],
            "Expected indices 3,4,5,6 in radius"
        );
    }

    #[test]
    fn kdtree_radius_search_invalid_radius() {
        let pts = vec![0.0_f32, 0.0, 0.0];
        let tree = KdTree::build(&pts, 1).unwrap();
        assert!(tree.radius_search([0.0, 0.0, 0.0], -1.0).is_err());
    }

    #[test]
    fn kdtree_large_build_nearest() {
        let n = 200;
        let pts: Vec<f32> = (0..n)
            .flat_map(|i| vec![(i as f32).sin(), (i as f32).cos(), i as f32 * 0.01])
            .collect();
        let tree = KdTree::build(&pts, n).unwrap();
        let q = [0.5, 0.5, 0.5];
        let (idx, sq_d) = tree.nearest(q).unwrap();

        // Verify brute force agrees
        let bf_idx = (0..n)
            .min_by(|&a, &b| {
                let da = {
                    let dx = pts[a * 3] - q[0];
                    let dy = pts[a * 3 + 1] - q[1];
                    let dz = pts[a * 3 + 2] - q[2];
                    dx * dx + dy * dy + dz * dz
                };
                let db = {
                    let dx = pts[b * 3] - q[0];
                    let dy = pts[b * 3 + 1] - q[1];
                    let dz = pts[b * 3 + 2] - q[2];
                    dx * dx + dy * dy + dz * dz
                };
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        assert_eq!(idx, bf_idx);
        assert!(sq_d.is_finite());
    }
}
