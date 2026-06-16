//! Tree-Wasserstein distance (f64 API).
//!
//! When the ground metric is a *tree metric* — distances given by path lengths
//! on a weighted tree — the Wasserstein-1 distance admits a closed form that is
//! computed in a single linear pass, with no transport LP at all (Le et al.,
//! NeurIPS 2019, "Tree-Sliced Variants of Wasserstein Distances").  For a rooted
//! tree with edge weights `w_e`,
//!
//! ```text
//! W₁^T(a, b) = Σ_e  w_e · | Σ_{v ∈ subtree(e)} ( a_v − b_v ) |,
//! ```
//!
//! where `subtree(e)` is the set of leaves hanging below the child endpoint of
//! edge `e`.  Intuitively each edge must carry the *net* mass imbalance of its
//! subtree, and the optimal cost is the weighted sum of these flows.  This makes
//! the tree-Wasserstein distance a true metric that lower-bounds the Euclidean
//! `W₁` whenever the tree is built from a spanning structure of the support, and
//! is the backbone of *tree-sliced* Wasserstein (average over many random
//! trees).
//!
//! ## Tree representation
//! The tree on `n` nodes is given by parent pointers `parent[v]` (with
//! `parent[root] = usize::MAX`) and the weight `edge_w[v]` of the edge joining
//! `v` to its parent (`edge_w[root]` is ignored).  Mass distributions `a`, `b`
//! are supplied per node; interior nodes may legitimately carry zero mass.
//!
//! References:
//! - Le, T., Yamada, M., Fukumizu, K., & Cuturi, M. (2019). *Tree-Sliced
//!   Variants of Wasserstein Distances.* NeurIPS.
//! - Indyk, P. & Thaper, N. (2003). *Fast image retrieval via embeddings.*

use crate::error::{OtError, OtResult};

/// A weighted rooted tree over `n` nodes for tree-Wasserstein computations.
#[derive(Debug, Clone)]
pub struct WeightedTree {
    /// Number of nodes.
    pub n: usize,
    /// `parent[v]` = parent index of node `v`; the root uses `usize::MAX`.
    pub parent: Vec<usize>,
    /// `edge_w[v]` = non-negative weight of the edge `(v, parent[v])`; the
    /// root's entry is ignored.
    pub edge_w: Vec<f64>,
}

impl WeightedTree {
    /// Construct a tree from parent pointers and edge weights, validating that
    /// it is acyclic, connected, and has exactly one root.
    ///
    /// # Errors
    /// - [`OtError::EmptyInput`] if `n == 0`.
    /// - [`OtError::IncompatibleLength`] if the arrays have the wrong length.
    /// - [`OtError::Internal`] if there is not exactly one root, an out-of-range
    ///   parent, a negative weight, or a cycle.
    pub fn new(n: usize, parent: Vec<usize>, edge_w: Vec<f64>) -> OtResult<Self> {
        if n == 0 {
            return Err(OtError::EmptyInput);
        }
        if parent.len() != n || edge_w.len() != n {
            return Err(OtError::IncompatibleLength {
                a: parent.len(),
                b: n,
            });
        }
        let mut roots = 0_usize;
        for (v, &p) in parent.iter().enumerate() {
            if p == usize::MAX {
                roots += 1;
            } else {
                if p >= n {
                    return Err(OtError::Internal {
                        msg: format!("tree: parent {p} of node {v} out of range"),
                    });
                }
                if p == v {
                    return Err(OtError::Internal {
                        msg: format!("tree: node {v} is its own parent"),
                    });
                }
                if !(edge_w[v] >= 0.0 && edge_w[v].is_finite()) {
                    return Err(OtError::Internal {
                        msg: format!("tree: edge weight {} at node {v} invalid", edge_w[v]),
                    });
                }
            }
        }
        if roots != 1 {
            return Err(OtError::Internal {
                msg: format!("tree: expected exactly one root, found {roots}"),
            });
        }
        // Cycle check: every node must reach the root within n steps.
        for start in 0..n {
            let mut cur = start;
            let mut steps = 0_usize;
            while cur != usize::MAX {
                cur = parent[cur];
                steps += 1;
                if steps > n {
                    return Err(OtError::Internal {
                        msg: "tree: cycle detected in parent pointers".into(),
                    });
                }
            }
        }
        Ok(Self { n, parent, edge_w })
    }

    /// Return a topological order (children before parents) of the nodes, so a
    /// single forward sweep accumulates subtree masses bottom-up.
    fn post_order(&self) -> Vec<usize> {
        // Depth of each node from the root; sort descending by depth.
        let mut depth = vec![0_usize; self.n];
        for (v, dv) in depth.iter_mut().enumerate() {
            let mut cur = v;
            let mut d = 0_usize;
            while self.parent[cur] != usize::MAX {
                cur = self.parent[cur];
                d += 1;
            }
            *dv = d;
        }
        let mut order: Vec<usize> = (0..self.n).collect();
        order.sort_by(|&u, &v| depth[v].cmp(&depth[u]));
        order
    }
}

/// Compute the closed-form tree-Wasserstein-1 distance between two node-mass
/// distributions on a [`WeightedTree`].
///
/// `a` and `b` must each sum to the same total mass (within tolerance).
///
/// # Errors
/// - [`OtError::IncompatibleLength`] if `a` or `b` is not of length `tree.n`.
/// - [`OtError::NegativeWeight`] if any mass entry is negative.
/// - [`OtError::MassImbalance`] if the two distributions carry different totals.
pub fn tree_wasserstein(tree: &WeightedTree, a: &[f64], b: &[f64]) -> OtResult<f64> {
    let n = tree.n;
    if a.len() != n {
        return Err(OtError::IncompatibleLength { a: a.len(), b: n });
    }
    if b.len() != n {
        return Err(OtError::IncompatibleLength { a: b.len(), b: n });
    }
    if a.iter().chain(b).any(|&v| v < 0.0) {
        return Err(OtError::NegativeWeight);
    }
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    if (sum_a - sum_b).abs() > 1e-5 {
        return Err(OtError::MassImbalance {
            sum_a: sum_a as f32,
            sum_b: sum_b as f32,
        });
    }

    // Net signed mass per node; accumulate up to parents in post order.
    let mut sub: Vec<f64> = a.iter().zip(b).map(|(ai, bi)| ai - bi).collect();
    let order = tree.post_order();
    let mut cost = 0.0_f64;
    for &v in &order {
        let p = tree.parent[v];
        if p == usize::MAX {
            continue; // root has no upward edge
        }
        // Edge (v, parent) must carry |net subtree mass at v|.
        cost += tree.edge_w[v] * sub[v].abs();
        sub[p] += sub[v];
    }
    Ok(cost)
}

/// Build a *balanced binary tree* over `n` leaves (the canonical "quadtree"-style
/// structure used by tree-sliced Wasserstein) with a uniform per-level edge
/// weight, returning the tree together with the leaf node indices in input
/// order.
///
/// The returned tree has the `n` leaves as the first `n` node indices, so a
/// leaf-mass vector of length `n` can be zero-padded to the tree's node count.
///
/// # Errors
/// - [`OtError::EmptyInput`] if `n == 0`.
/// - [`OtError::Internal`] if `level_weight` is not a positive finite number.
pub fn balanced_binary_tree(n: usize, level_weight: f64) -> OtResult<(WeightedTree, Vec<usize>)> {
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    if !(level_weight > 0.0 && level_weight.is_finite()) {
        return Err(OtError::Internal {
            msg: format!("balanced tree: level_weight must be > 0, got {level_weight}"),
        });
    }
    // Leaves occupy indices 0..n. Internal nodes are appended level by level.
    let leaves: Vec<usize> = (0..n).collect();
    let mut parent: Vec<usize> = vec![usize::MAX; n];
    let mut edge_w: Vec<f64> = vec![level_weight; n];

    let mut current: Vec<usize> = leaves.clone();
    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len().div_ceil(2));
        let mut i = 0_usize;
        while i < current.len() {
            if i + 1 < current.len() {
                // Create an internal parent for the pair (current[i], current[i+1]).
                let idx = parent.len();
                parent.push(usize::MAX);
                edge_w.push(level_weight);
                parent[current[i]] = idx;
                parent[current[i + 1]] = idx;
                next.push(idx);
                i += 2;
            } else {
                // Odd node carried up unchanged.
                next.push(current[i]);
                i += 1;
            }
        }
        current = next;
    }
    // The single remaining node is the root.
    let root = current[0];
    parent[root] = usize::MAX;
    let total = parent.len();
    let tree = WeightedTree::new(total, parent, edge_w)?;
    Ok((tree, leaves))
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A simple chain 0 - 1 - 2 with unit edges; tree distance d(0,2) = 2.
    fn chain3() -> WeightedTree {
        // node 0 child of 1, node 1 child of 2 (root). Edge weights 1.
        WeightedTree::new(3, vec![1, 2, usize::MAX], vec![1.0, 1.0, 0.0]).expect("ok")
    }

    #[test]
    fn chain_two_diracs() {
        // Mass 1 at leaf 0 vs mass 1 at node 2 → must traverse both edges → 2.
        let t = chain3();
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 0.0, 1.0];
        let d = tree_wasserstein(&t, &a, &b).expect("ok");
        assert!((d - 2.0).abs() < 1e-12, "d={d}");
    }

    #[test]
    fn chain_adjacent() {
        // Mass at node 0 vs node 1 → one edge → 1.
        let t = chain3();
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 1.0, 0.0];
        let d = tree_wasserstein(&t, &a, &b).expect("ok");
        assert!((d - 1.0).abs() < 1e-12, "d={d}");
    }

    #[test]
    fn zero_on_equal() {
        let t = chain3();
        let a = vec![0.5_f64, 0.2, 0.3];
        let d = tree_wasserstein(&t, &a, &a).expect("ok");
        assert!(d.abs() < 1e-12, "d={d}");
    }

    #[test]
    fn star_tree_split_mass() {
        // Star: leaves 0,1,2 all child of root 3, edge weights 1,2,3.
        let t =
            WeightedTree::new(4, vec![3, 3, 3, usize::MAX], vec![1.0, 2.0, 3.0, 0.0]).expect("ok");
        // Move 1 unit from leaf 0 to leaf 2: path cost = w0 + w2 = 1 + 3 = 4.
        let a = vec![1.0_f64, 0.0, 0.0, 0.0];
        let b = vec![0.0_f64, 0.0, 1.0, 0.0];
        let d = tree_wasserstein(&t, &a, &b).expect("ok");
        assert!((d - 4.0).abs() < 1e-12, "d={d}");
    }

    #[test]
    fn symmetry() {
        let t = chain3();
        let a = vec![0.7_f64, 0.0, 0.3];
        let b = vec![0.1_f64, 0.6, 0.3];
        let dab = tree_wasserstein(&t, &a, &b).expect("ok");
        let dba = tree_wasserstein(&t, &b, &a).expect("ok");
        assert!((dab - dba).abs() < 1e-12, "ab={dab}, ba={dba}");
    }

    #[test]
    fn triangle_inequality() {
        // d(a,c) ≤ d(a,b) + d(b,c) for three distributions on the chain.
        let t = chain3();
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 1.0, 0.0];
        let c = vec![0.0_f64, 0.0, 1.0];
        let dac = tree_wasserstein(&t, &a, &c).expect("ok");
        let dab = tree_wasserstein(&t, &a, &b).expect("ok");
        let dbc = tree_wasserstein(&t, &b, &c).expect("ok");
        assert!(dac <= dab + dbc + 1e-12, "ac={dac}, ab+bc={}", dab + dbc);
    }

    #[test]
    fn balanced_tree_construction() {
        let (tree, leaves) = balanced_binary_tree(4, 1.0).expect("ok");
        assert_eq!(leaves, vec![0, 1, 2, 3]);
        // 4 leaves + 2 mid + 1 root = 7 nodes.
        assert_eq!(tree.n, 7);
        // Moving mass between sibling leaves costs 2 (up one edge, down one).
        let mut a = vec![0.0_f64; tree.n];
        let mut b = vec![0.0_f64; tree.n];
        a[0] = 1.0;
        b[1] = 1.0;
        let d = tree_wasserstein(&tree, &a, &b).expect("ok");
        assert!((d - 2.0).abs() < 1e-12, "sibling d={d}");
    }

    #[test]
    fn balanced_tree_far_leaves_cost_more() {
        let (tree, _) = balanced_binary_tree(4, 1.0).expect("ok");
        let mut a = vec![0.0_f64; tree.n];
        let mut b_sib = vec![0.0_f64; tree.n];
        let mut b_far = vec![0.0_f64; tree.n];
        a[0] = 1.0;
        b_sib[1] = 1.0; // sibling of leaf 0
        b_far[3] = 1.0; // in the other half of the tree
        let d_sib = tree_wasserstein(&tree, &a, &b_sib).expect("ok");
        let d_far = tree_wasserstein(&tree, &a, &b_far).expect("ok");
        assert!(d_far > d_sib, "far={d_far} should exceed sib={d_sib}");
    }

    #[test]
    fn odd_leaf_count_tree() {
        let (tree, leaves) = balanced_binary_tree(5, 0.5).expect("ok");
        assert_eq!(leaves.len(), 5);
        let mut a = vec![0.0_f64; tree.n];
        let mut b = vec![0.0_f64; tree.n];
        a[0] = 1.0;
        b[4] = 1.0;
        let d = tree_wasserstein(&tree, &a, &b).expect("ok");
        assert!(d > 0.0 && d.is_finite(), "d={d}");
    }

    #[test]
    fn distance_nonneg_and_finite() {
        let t = chain3();
        let a = vec![0.3_f64, 0.3, 0.4];
        let b = vec![0.5_f64, 0.1, 0.4];
        let d = tree_wasserstein(&t, &a, &b).expect("ok");
        assert!(d >= 0.0 && d.is_finite(), "d={d}");
    }

    #[test]
    fn rejects_mass_imbalance() {
        let t = chain3();
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 0.0, 2.0];
        assert!(matches!(
            tree_wasserstein(&t, &a, &b),
            Err(OtError::MassImbalance { .. })
        ));
    }

    #[test]
    fn rejects_negative_mass() {
        let t = chain3();
        let a = vec![1.5_f64, -0.5, 0.0];
        let b = vec![0.0_f64, 0.0, 1.0];
        assert!(matches!(
            tree_wasserstein(&t, &a, &b),
            Err(OtError::NegativeWeight)
        ));
    }

    #[test]
    fn invalid_tree_rejected() {
        // Two roots.
        let res = WeightedTree::new(3, vec![usize::MAX, usize::MAX, 0], vec![0.0, 1.0, 1.0]);
        assert!(matches!(res, Err(OtError::Internal { .. })));
        // Cycle: 0 -> 1 -> 0.
        let res2 = WeightedTree::new(2, vec![1, 0], vec![1.0, 1.0]);
        assert!(matches!(res2, Err(OtError::Internal { .. })));
    }
}
