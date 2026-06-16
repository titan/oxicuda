//! Gomory-Hu cut tree (Gusfield's 1990 simplified construction).
//!
//! # Overview
//!
//! Given an **undirected** weighted graph `G` on `n` vertices, the *Gomory-Hu tree*
//! is a weighted tree `T` on the same vertex set with exactly `n − 1` edges such that
//! for **every** pair of vertices `(s, t)`, the minimum-weight edge on the unique
//! `s`–`t` path in `T` equals the value of a minimum `s`–`t` cut in `G`.
//!
//! The classical Gomory-Hu construction (1961) requires graph contractions. Gusfield
//! (1990, *Very Simple Methods for All Pairs Network Flow Analysis*) showed the same
//! tree can be built with `n − 1` max-flow computations performed **directly on the
//! original graph** — no contractions, no Steiner trees. That is what we implement here.
//!
//! ## Gusfield's algorithm
//!
//! ```text
//! parent[i] ← 0          for all i in 1..n        (tree rooted at vertex 0)
//! weight[i] ← 0          for all i
//! for s in 1..n:
//!     t ← parent[s]
//!     (f, S) ← min-cut(s, t)          # S = side of the cut containing s
//!     weight[s] ← f
//!     for v in 0..n:                  # re-parent vertices that fell on s's side
//!         if v ≠ s and v in S and parent[v] == t:
//!             parent[v] ← s
//!     if parent[t] in S:              # Gusfield's extra re-parent step
//!         parent[s] ← parent[t]
//!         parent[t] ← s
//!         weight[s] ← weight[t]
//!         weight[t] ← f
//! ```
//!
//! The resulting tree is encoded by the `(parent, weight)` arrays: for `i ≥ 1` there is
//! a tree edge `{i, parent[i]}` of weight `weight[i]`. Vertex `0` is the root.
//!
//! Min-cuts are obtained by reusing the crate's [`min_cut_from_max_flow`], which runs
//! Edmonds-Karp on a symmetric [`FlowNetwork`] and returns both the cut value and the
//! source-side vertex set.

use crate::error::{GraphalgError, GraphalgResult};
use crate::max_flow::edmonds_karp::FlowNetwork;
use crate::max_flow::min_cut::min_cut_from_max_flow;

/// A Gomory-Hu cut tree for an undirected weighted graph.
///
/// The tree is rooted at vertex `0`. For every non-root vertex `i`, `parent[i]` is its
/// parent in the tree and `weight[i]` is the weight of the tree edge `{i, parent[i]}`,
/// which equals the minimum `i`–`parent[i]` cut value in the original graph.
#[derive(Debug, Clone)]
pub struct GomoryHuTree {
    /// Number of vertices (same as the input graph).
    pub n: usize,
    /// `parent[i]` = parent of vertex `i` in the tree; `parent[0] == 0` (the root).
    pub parent: Vec<usize>,
    /// `weight[i]` = weight of the tree edge `{i, parent[i]}`; `weight[0] == 0.0`.
    pub weight: Vec<f64>,
}

impl GomoryHuTree {
    /// The `n − 1` undirected tree edges as `(u, v, weight)` triples (`u < v`).
    pub fn tree_edges(&self) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::with_capacity(self.n.saturating_sub(1));
        for i in 1..self.n {
            let u = i.min(self.parent[i]);
            let v = i.max(self.parent[i]);
            edges.push((u, v, self.weight[i]));
        }
        edges
    }

    /// Minimum `s`–`t` cut value, read off as the lightest edge on the tree path
    /// from `s` to `t`.
    ///
    /// Equal vertices yield `0.0`. Two vertices in different tree components (which can
    /// only happen if the original graph is disconnected, producing one or more
    /// zero-weight tree edges) also yield a `0.0` minimum on the path.
    ///
    /// # Errors
    /// - [`GraphalgError::IndexOutOfBounds`] if `s` or `t` is not a valid vertex.
    pub fn min_cut(&self, s: usize, t: usize) -> GraphalgResult<f64> {
        if s >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: s,
                len: self.n,
            });
        }
        if t >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: t,
                len: self.n,
            });
        }
        if s == t {
            return Ok(0.0);
        }

        // Depth of each node from the root, plus its parent, lets us walk both endpoints
        // up to their lowest common ancestor while tracking the minimum edge weight.
        let depth = self.depths();
        let (mut a, mut b) = (s, t);
        let mut best = f64::INFINITY;

        // Bring the deeper node up first.
        while depth[a] > depth[b] {
            best = best.min(self.weight[a]);
            a = self.parent[a];
        }
        while depth[b] > depth[a] {
            best = best.min(self.weight[b]);
            b = self.parent[b];
        }
        // Now ascend together until they meet at the LCA.
        while a != b {
            best = best.min(self.weight[a]);
            a = self.parent[a];
            best = best.min(self.weight[b]);
            b = self.parent[b];
        }
        Ok(best)
    }

    /// Depth (number of edges to the root) of every vertex.
    fn depths(&self) -> Vec<usize> {
        let mut depth = vec![0usize; self.n];
        // The parent of any node has a strictly smaller index in Gusfield's tree is
        // *not* guaranteed, so resolve depths by repeatedly walking to the root. With
        // memoization this is linear overall.
        let mut resolved = vec![false; self.n];
        resolved[0] = true;
        for start in 1..self.n {
            // Collect the chain up to the first resolved ancestor.
            let mut chain = Vec::new();
            let mut cur = start;
            while !resolved[cur] {
                chain.push(cur);
                cur = self.parent[cur];
            }
            let mut d = depth[cur];
            for &node in chain.iter().rev() {
                d += 1;
                depth[node] = d;
                resolved[node] = true;
            }
        }
        depth
    }
}

/// Build a symmetric [`FlowNetwork`] from an undirected weighted edge list.
///
/// Each undirected edge `{u, v}` with weight `w` contributes capacity `w` to both
/// `u→v` and `v→u`, which is the standard reduction for undirected min-cut via max-flow.
fn build_undirected_network(
    n: usize,
    edges: &[(usize, usize, f64)],
) -> GraphalgResult<FlowNetwork> {
    let mut net = FlowNetwork::new(n);
    for &(u, v, w) in edges {
        if u >= n || v >= n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: n,
            });
        }
        if !w.is_finite() {
            return Err(GraphalgError::InvalidEdgeWeight(format!(
                "edge ({u},{v}) weight not finite: {w}"
            )));
        }
        if w < 0.0 {
            return Err(GraphalgError::NegativeWeight {
                edge: (u, v),
                weight: w,
            });
        }
        if u == v {
            // Self-loops never cross any cut; ignore them.
            continue;
        }
        net.add_edge(u, v, w)?;
        net.add_edge(v, u, w)?;
    }
    Ok(net)
}

/// Construct the Gomory-Hu cut tree of an undirected weighted graph using Gusfield's
/// `n − 1` max-flow computations.
///
/// `edges` is an undirected weighted edge list: each `(u, v, w)` denotes an undirected
/// edge of non-negative weight `w`. Parallel edges are summed (their capacities add),
/// self-loops are ignored.
///
/// # Errors
/// - [`GraphalgError::IndexOutOfBounds`] if any endpoint is `>= n`.
/// - [`GraphalgError::NegativeWeight`] if any edge weight is negative.
/// - [`GraphalgError::InvalidEdgeWeight`] if any edge weight is non-finite.
pub fn gomory_hu_tree(n: usize, edges: &[(usize, usize, f64)]) -> GraphalgResult<GomoryHuTree> {
    if n == 0 {
        return Ok(GomoryHuTree {
            n: 0,
            parent: Vec::new(),
            weight: Vec::new(),
        });
    }
    if n == 1 {
        return Ok(GomoryHuTree {
            n: 1,
            parent: vec![0],
            weight: vec![0.0],
        });
    }

    let net = build_undirected_network(n, edges)?;

    // Gusfield: tree rooted at 0.
    let mut parent = vec![0usize; n];
    let mut weight = vec![0.0f64; n];

    for s in 1..n {
        let t = parent[s];
        let cut = min_cut_from_max_flow(&net, s, t)?;
        let f = cut.value;
        weight[s] = f;

        // Membership test on the source side (the side containing `s`).
        let mut in_source = vec![false; n];
        for &v in &cut.source_side {
            in_source[v] = true;
        }

        // Re-parent every other vertex that shares parent `t` and landed on `s`'s side.
        for v in 0..n {
            if v != s && parent[v] == t && in_source[v] {
                parent[v] = s;
            }
        }

        // Gusfield's additional adjustment: if `parent[t]` lies on `s`'s side, rotate.
        let pt = parent[t];
        if in_source[pt] {
            parent[s] = pt;
            parent[t] = s;
            weight[s] = weight[t];
            weight[t] = f;
        }
    }

    Ok(GomoryHuTree { n, parent, weight })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::max_flow::edmonds_karp::FlowNetwork;
    use crate::max_flow::min_cut::min_cut_from_max_flow;

    /// Direct min `s`–`t` cut via max-flow on a fresh symmetric network — the oracle.
    fn direct_min_cut(n: usize, edges: &[(usize, usize, f64)], s: usize, t: usize) -> f64 {
        let mut net = FlowNetwork::new(n);
        for &(u, v, w) in edges {
            if u == v {
                continue;
            }
            net.add_edge(u, v, w).expect("add ok");
            net.add_edge(v, u, w).expect("add ok");
        }
        min_cut_from_max_flow(&net, s, t).expect("cut ok").value
    }

    /// Verify: for EVERY ordered pair, the tree path-minimum equals the oracle cut.
    fn assert_all_pairs(n: usize, edges: &[(usize, usize, f64)]) {
        let tree = gomory_hu_tree(n, edges).expect("tree ok");
        for s in 0..n {
            for t in 0..n {
                if s == t {
                    continue;
                }
                let tree_cut = tree.min_cut(s, t).expect("min_cut ok");
                let oracle = direct_min_cut(n, edges, s, t);
                assert!(
                    (tree_cut - oracle).abs() < 1e-6,
                    "pair ({s},{t}): tree={tree_cut} oracle={oracle}"
                );
            }
        }
    }

    #[test]
    fn all_pairs_match_on_weighted_graph() {
        // The classic 6-node weighted graph from Gomory & Hu's 1961 paper.
        let edges = [
            (0, 1, 1.0),
            (0, 2, 7.0),
            (1, 2, 1.0),
            (1, 3, 3.0),
            (1, 4, 2.0),
            (2, 4, 4.0),
            (3, 4, 1.0),
            (3, 5, 6.0),
            (4, 5, 2.0),
        ];
        assert_all_pairs(6, &edges);
    }

    #[test]
    fn all_pairs_match_on_dense_small() {
        // A small dense graph (K4 with varied weights) — every pair checked.
        let edges = [
            (0, 1, 3.0),
            (0, 2, 1.0),
            (0, 3, 5.0),
            (1, 2, 4.0),
            (1, 3, 2.0),
            (2, 3, 6.0),
        ];
        assert_all_pairs(4, &edges);
    }

    #[test]
    fn tree_has_n_minus_one_edges_and_is_connected() {
        let edges = [
            (0, 1, 2.0),
            (1, 2, 3.0),
            (2, 3, 4.0),
            (3, 0, 5.0),
            (0, 2, 1.0),
        ];
        let n = 4;
        let tree = gomory_hu_tree(n, &edges).expect("tree ok");
        let tree_edges = tree.tree_edges();
        assert_eq!(tree_edges.len(), n - 1);

        // Connectivity: union-find over the n-1 tree edges must form a single component.
        let mut comp: Vec<usize> = (0..n).collect();
        fn find(comp: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while comp[r] != r {
                r = comp[r];
            }
            let mut c = x;
            while comp[c] != c {
                let nxt = comp[c];
                comp[c] = r;
                c = nxt;
            }
            r
        }
        for &(u, v, _) in &tree_edges {
            let (ru, rv) = (find(&mut comp, u), find(&mut comp, v));
            comp[ru] = rv;
        }
        let root = find(&mut comp, 0);
        for v in 0..n {
            assert_eq!(find(&mut comp, v), root, "vertex {v} not connected in tree");
        }
    }

    #[test]
    fn simple_cycle_all_pairs() {
        // A cycle of unit weights: every min cut should be 2 (two edges must be cut).
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
        let n = 4;
        assert_all_pairs(n, &edges);
        let tree = gomory_hu_tree(n, &edges).expect("tree ok");
        for s in 0..n {
            for t in 0..n {
                if s != t {
                    assert!((tree.min_cut(s, t).expect("ok") - 2.0).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn disconnected_components_give_zero_cut() {
        // Two disjoint edges: {0-1} and {2-3}. Cuts across components are 0.
        let edges = [(0, 1, 5.0), (2, 3, 7.0)];
        let n = 4;
        let tree = gomory_hu_tree(n, &edges).expect("tree ok");
        // Across components → 0.
        assert!((tree.min_cut(0, 2).expect("ok")).abs() < 1e-9);
        assert!((tree.min_cut(1, 3).expect("ok")).abs() < 1e-9);
        // Within a component → the single edge weight.
        assert!((tree.min_cut(0, 1).expect("ok") - 5.0).abs() < 1e-9);
        assert!((tree.min_cut(2, 3).expect("ok") - 7.0).abs() < 1e-9);
        // And cross-check the whole thing against the oracle.
        assert_all_pairs(n, &edges);
    }

    #[test]
    fn cut_is_symmetric() {
        let edges = [
            (0, 1, 2.0),
            (1, 2, 5.0),
            (2, 3, 1.0),
            (0, 3, 4.0),
            (1, 3, 3.0),
        ];
        let n = 4;
        let tree = gomory_hu_tree(n, &edges).expect("tree ok");
        for s in 0..n {
            for t in 0..n {
                let st = tree.min_cut(s, t).expect("ok");
                let ts = tree.min_cut(t, s).expect("ok");
                assert!((st - ts).abs() < 1e-12, "asymmetry ({s},{t}): {st} vs {ts}");
            }
        }
    }

    #[test]
    fn single_edge_graph() {
        let edges = [(0, 1, 9.0)];
        let tree = gomory_hu_tree(2, &edges).expect("tree ok");
        assert_eq!(tree.tree_edges().len(), 1);
        assert!((tree.min_cut(0, 1).expect("ok") - 9.0).abs() < 1e-9);
        assert!((tree.min_cut(1, 0).expect("ok") - 9.0).abs() < 1e-9);
    }

    #[test]
    fn tiny_two_node_no_edges() {
        let tree = gomory_hu_tree(2, &[]).expect("tree ok");
        // No edges → cut is 0.
        assert!((tree.min_cut(0, 1).expect("ok")).abs() < 1e-9);
        assert_eq!(tree.tree_edges().len(), 1);
    }

    #[test]
    fn single_node_graph() {
        let tree = gomory_hu_tree(1, &[]).expect("tree ok");
        assert_eq!(tree.n, 1);
        assert_eq!(tree.tree_edges().len(), 0);
        assert!((tree.min_cut(0, 0).expect("ok")).abs() < 1e-9);
    }

    #[test]
    fn empty_graph() {
        let tree = gomory_hu_tree(0, &[]).expect("tree ok");
        assert_eq!(tree.n, 0);
        assert_eq!(tree.tree_edges().len(), 0);
    }

    #[test]
    fn same_vertex_is_zero() {
        let edges = [(0, 1, 3.0), (1, 2, 4.0)];
        let tree = gomory_hu_tree(3, &edges).expect("tree ok");
        assert!((tree.min_cut(1, 1).expect("ok")).abs() < 1e-12);
    }

    #[test]
    fn parallel_edges_sum() {
        // Two parallel edges 0-1 of weight 2 and 3 → combined cut should be 5.
        let edges = [(0, 1, 2.0), (0, 1, 3.0)];
        let tree = gomory_hu_tree(2, &edges).expect("tree ok");
        assert!((tree.min_cut(0, 1).expect("ok") - 5.0).abs() < 1e-9);
    }

    #[test]
    fn self_loops_ignored() {
        let edges = [(0, 0, 100.0), (0, 1, 4.0), (1, 1, 50.0)];
        let tree = gomory_hu_tree(2, &edges).expect("tree ok");
        assert!((tree.min_cut(0, 1).expect("ok") - 4.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_negative_weight() {
        let edges = [(0, 1, -1.0)];
        assert!(matches!(
            gomory_hu_tree(2, &edges),
            Err(GraphalgError::NegativeWeight { .. })
        ));
    }

    #[test]
    fn rejects_oob_endpoint() {
        let edges = [(0, 5, 1.0)];
        assert!(matches!(
            gomory_hu_tree(3, &edges),
            Err(GraphalgError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn min_cut_rejects_oob_query() {
        let tree = gomory_hu_tree(3, &[(0, 1, 1.0), (1, 2, 1.0)]).expect("ok");
        assert!(tree.min_cut(0, 9).is_err());
        assert!(tree.min_cut(9, 0).is_err());
    }

    #[test]
    fn star_graph_all_pairs() {
        // Star centered at 0 with leaf weights 1,2,3. min cut between two leaves is the
        // smaller of their two spoke weights; between center and a leaf is the spoke.
        let edges = [(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)];
        let n = 4;
        assert_all_pairs(n, &edges);
    }

    #[test]
    fn path_graph_all_pairs() {
        let edges = [(0, 1, 4.0), (1, 2, 2.0), (2, 3, 6.0), (3, 4, 5.0)];
        assert_all_pairs(5, &edges);
    }
}
