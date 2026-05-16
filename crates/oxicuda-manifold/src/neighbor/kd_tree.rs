//! Axis-aligned KD-tree with median split.
//!
//! Designed for low-dimensional kNN. For very high dimensions, prefer brute force
//! or ball trees.

use crate::error::{ManifoldError, ManifoldResult};

/// A node in the KD-tree.
#[derive(Debug, Clone)]
pub struct KdTreeNode {
    pub idx: usize,
    pub axis: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

/// KD-tree over row-major data.
#[derive(Debug, Clone)]
pub struct KdTree {
    pub points: Vec<f64>,
    pub n: usize,
    pub dim: usize,
    pub nodes: Vec<KdTreeNode>,
    pub root: Option<usize>,
}

impl KdTree {
    /// Build a KD-tree from row-major data of shape `(n, dim)`.
    pub fn build(points: &[f64], n: usize, dim: usize) -> ManifoldResult<Self> {
        if n == 0 || dim == 0 {
            return Err(ManifoldError::EmptyInput);
        }
        if points.len() != n * dim {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![n, dim],
                got: vec![points.len()],
            });
        }
        let mut indices: Vec<usize> = (0..n).collect();
        let mut tree = KdTree {
            points: points.to_vec(),
            n,
            dim,
            nodes: Vec::new(),
            root: None,
        };
        tree.root = tree.build_recursive(&mut indices, 0);
        Ok(tree)
    }

    fn build_recursive(&mut self, indices: &mut [usize], depth: usize) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % self.dim;
        // Sort by current axis
        indices.sort_by(|&a, &b| {
            self.points[a * self.dim + axis]
                .partial_cmp(&self.points[b * self.dim + axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = indices.len() / 2;
        let median = indices[mid];
        let (left_slice, right_slice) = indices.split_at_mut(mid);
        let right_slice = &mut right_slice[1..]; // skip median
        let left = self.build_recursive(left_slice, depth + 1);
        let right = self.build_recursive(right_slice, depth + 1);
        self.nodes.push(KdTreeNode {
            idx: median,
            axis,
            left,
            right,
        });
        Some(self.nodes.len() - 1)
    }

    fn sq_dist_to(&self, p: &[f64], idx: usize) -> f64 {
        let mut s = 0.0;
        for (d, pd) in p.iter().enumerate().take(self.dim) {
            let v = *pd - self.points[idx * self.dim + d];
            s += v * v;
        }
        s
    }

    /// Search for the k nearest neighbours of `query`.
    /// Returns `(indices, sq_distances)` sorted ascending by distance.
    pub fn knn(&self, query: &[f64], k: usize) -> ManifoldResult<(Vec<usize>, Vec<f64>)> {
        if query.len() != self.dim {
            return Err(ManifoldError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![query.len()],
            });
        }
        if k == 0 || k > self.n {
            return Err(ManifoldError::KNeighborsTooLarge { k, n: self.n });
        }
        // bounded priority queue of size k (max-heap by distance => keep smallest)
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        self.recurse_knn(query, self.root, k, &mut heap);
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let idx_out = heap.iter().map(|t| t.1).collect();
        let d_out = heap.into_iter().map(|t| t.0).collect();
        Ok((idx_out, d_out))
    }

    fn recurse_knn(
        &self,
        q: &[f64],
        node_id: Option<usize>,
        k: usize,
        heap: &mut Vec<(f64, usize)>,
    ) {
        let Some(nid) = node_id else { return };
        let n = &self.nodes[nid];
        let d = self.sq_dist_to(q, n.idx);
        Self::heap_push(heap, k, (d, n.idx));
        let axis = n.axis;
        let diff = q[axis] - self.points[n.idx * self.dim + axis];
        let (first, second) = if diff < 0.0 {
            (n.left, n.right)
        } else {
            (n.right, n.left)
        };
        self.recurse_knn(q, first, k, heap);
        let worst = heap.iter().map(|t| t.0).fold(f64::NEG_INFINITY, f64::max);
        if heap.len() < k || diff * diff < worst {
            self.recurse_knn(q, second, k, heap);
        }
    }

    fn heap_push(heap: &mut Vec<(f64, usize)>, k: usize, item: (f64, usize)) {
        if heap.len() < k {
            heap.push(item);
        } else {
            // find current max
            let (mut max_idx, mut max_val) = (0usize, f64::NEG_INFINITY);
            for (i, t) in heap.iter().enumerate() {
                if t.0 > max_val {
                    max_val = t.0;
                    max_idx = i;
                }
            }
            if item.0 < max_val {
                heap[max_idx] = item;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_query() {
        let n = 5;
        let dim = 2;
        let pts = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let tree = KdTree::build(&pts, n, dim).expect("ok");
        let q = vec![0.5, 0.5];
        let (idx, _d) = tree.knn(&q, 1).expect("ok");
        assert_eq!(idx[0], 4); // (0.5, 0.5)
    }

    #[test]
    fn knn_k_neighbours() {
        let pts = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let tree = KdTree::build(&pts, 5, 2).expect("ok");
        let (idx, _d) = tree.knn(&[2.0, 2.0], 3).expect("ok");
        assert!(idx.contains(&2));
    }
}
