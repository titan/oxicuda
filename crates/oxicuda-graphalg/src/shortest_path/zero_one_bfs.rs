//! 0-1 BFS — single-source shortest paths on graphs whose edge weights are
//! restricted to `{0, 1}`.
//!
//! When every edge has weight 0 or 1, Dijkstra's `O(E log V)` heap can be
//! replaced by a double-ended queue giving `O(V + E)`: relaxing a **0-weight**
//! edge pushes the neighbour to the *front* of the deque, while a **1-weight**
//! edge pushes to the *back*. The deque therefore always holds vertices of at
//! most two consecutive distance levels, so it is processed in non-decreasing
//! distance order exactly as a priority queue would be — but in linear time.
//!
//! This is the standard tool for grid/state-graph problems where some
//! transitions are "free" and others "cost one".
//!
//! ## References
//! - Bard, E. & others; folklore "0-1 BFS" (see CP-Algorithms,
//!   "0-1 BFS"). Equivalent to Dial's algorithm with two buckets.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

/// Result of a 0-1 BFS run from a single source.
#[derive(Debug, Clone)]
pub struct ZeroOneBfsOutput {
    /// `dist[v]` = shortest distance from source (`u64::MAX` if unreachable).
    pub dist: Vec<u64>,
    /// `parent[v]` = predecessor of `v` (`usize::MAX` if none; source is its own).
    pub parent: Vec<usize>,
}

/// Compute single-source shortest paths on a graph with `{0, 1}` edge weights.
///
/// Edge weights are read from the [`WeightedGraph`]; each must equal `0.0` or
/// `1.0` (within a small tolerance).
///
/// # Errors
/// - [`GraphalgError::SourceOutOfRange`] if `source >= g.n`.
/// - [`GraphalgError::InvalidEdgeWeight`] if any edge weight is not 0 or 1.
pub fn zero_one_bfs(g: &WeightedGraph, source: usize) -> GraphalgResult<ZeroOneBfsOutput> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    // Validate the {0,1} restriction.
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            let is_zero = w.abs() < 1e-9;
            let is_one = (w - 1.0).abs() < 1e-9;
            if !(is_zero || is_one) {
                return Err(GraphalgError::InvalidEdgeWeight(format!(
                    "edge ({u},{v}) weight {w} is not 0 or 1"
                )));
            }
        }
    }

    let n = g.n;
    let mut dist = vec![u64::MAX; n];
    let mut parent = vec![usize::MAX; n];
    dist[source] = 0;
    parent[source] = source;

    let mut dq: VecDeque<usize> = VecDeque::new();
    dq.push_front(source);

    while let Some(u) = dq.pop_front() {
        let du = dist[u];
        for &(v, w) in g.neighbors(u)? {
            // `w` is guaranteed 0 or 1 by the validation above.
            let weight = if w < 0.5 { 0u64 } else { 1u64 };
            let nd = du.saturating_add(weight);
            if nd < dist[v] {
                dist[v] = nd;
                parent[v] = u;
                if weight == 0 {
                    dq.push_front(v);
                } else {
                    dq.push_back(v);
                }
            }
        }
    }

    Ok(ZeroOneBfsOutput { dist, parent })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ones_is_plain_bfs() {
        // Path 0 -1 -2 -3 with unit edges → distances 0,1,2,3.
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist, vec![0, 1, 2, 3]);
    }

    #[test]
    fn zero_edge_is_free() {
        // 0 -(1)-> 1 -(0)-> 2; distance to 2 equals distance to 1.
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 0.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist[1], 1);
        assert_eq!(out.dist[2], 1, "0-weight edge should be free");
    }

    #[test]
    fn prefers_cheaper_zero_path() {
        // Two ways to reach 3: direct 0->3 cost 1, or 0->1->2->3 with weights
        // 0,0,0 cost 0. The 0-cost route must win.
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 3, 1.0).expect("ok");
        g.add_edge(0, 1, 0.0).expect("ok");
        g.add_edge(1, 2, 0.0).expect("ok");
        g.add_edge(2, 3, 0.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist[3], 0, "should take the all-zero path");
    }

    #[test]
    fn source_distance_zero() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist[0], 0);
        assert_eq!(out.parent[0], 0);
    }

    #[test]
    fn unreachable_is_max() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist[2], u64::MAX);
        assert_eq!(out.parent[2], usize::MAX);
    }

    #[test]
    fn rejects_non_binary_weight() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, 2.0).expect("ok");
        assert!(matches!(
            zero_one_bfs(&g, 0).unwrap_err(),
            GraphalgError::InvalidEdgeWeight(_)
        ));
    }

    #[test]
    fn source_out_of_range() {
        let g = WeightedGraph::new(3);
        assert!(matches!(
            zero_one_bfs(&g, 9).unwrap_err(),
            GraphalgError::SourceOutOfRange { .. }
        ));
    }

    #[test]
    fn matches_unit_bfs_on_grid() {
        // Undirected unit-weight ring 0-1-2-3-4-0; from 0 distances are
        // 0,1,2,2,1.
        let mut g = WeightedGraph::new(5);
        g.add_undirected_edge(0, 1, 1.0).expect("ok");
        g.add_undirected_edge(1, 2, 1.0).expect("ok");
        g.add_undirected_edge(2, 3, 1.0).expect("ok");
        g.add_undirected_edge(3, 4, 1.0).expect("ok");
        g.add_undirected_edge(4, 0, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist, vec![0, 1, 2, 2, 1]);
    }

    #[test]
    fn mixed_weights_correct() {
        // 0 -(0)-> 1 -(1)-> 2 -(0)-> 3 -(1)-> 4.  dist = 0,0,1,1,2.
        let mut g = WeightedGraph::new(5);
        g.add_edge(0, 1, 0.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(2, 3, 0.0).expect("ok");
        g.add_edge(3, 4, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn parent_chain_reconstructs_path() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        // Walk parents back from 3 to the source.
        let mut path = vec![3usize];
        let mut cur = 3usize;
        while cur != 0 {
            cur = out.parent[cur];
            path.push(cur);
        }
        path.reverse();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn deterministic() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 0.0).expect("ok");
        g.add_edge(0, 2, 1.0).expect("ok");
        g.add_edge(1, 3, 1.0).expect("ok");
        g.add_edge(2, 3, 0.0).expect("ok");
        let a = zero_one_bfs(&g, 0).expect("ok");
        let b = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(a.dist, b.dist);
    }

    #[test]
    fn self_loop_ignored() {
        // A 0-weight self loop must not break termination or distances.
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 0, 0.0).expect("ok");
        g.add_edge(0, 1, 1.0).expect("ok");
        let out = zero_one_bfs(&g, 0).expect("ok");
        assert_eq!(out.dist[0], 0);
        assert_eq!(out.dist[1], 1);
    }
}
