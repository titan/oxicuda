//! Betweenness centrality (Brandes 2001).
//!
//! The betweenness centrality of a node `v` measures how often `v` lies on a
//! shortest path between two other nodes:
//! ```text
//! C_B(v) = Σ_{s ≠ v ≠ t}  σ_{st}(v) / σ_{st}
//! ```
//! where `σ_{st}` is the number of shortest `s → t` paths and `σ_{st}(v)` is the
//! number of those that pass through `v`.
//!
//! Brandes' algorithm computes all pairwise contributions in `O(n·m)` time (for
//! unweighted graphs) without ever materialising the `O(n²)` pair table. For
//! each source `s` it
//! 1. runs a BFS that records, for every node, its shortest-path distance, the
//!    number of shortest paths `σ` from `s`, and the set of *predecessors* on
//!    those paths, and
//! 2. accumulates dependencies `δ_s(v)` by walking the BFS-discovered nodes in
//!    order of *non-increasing* distance, using the recurrence
//!    ```text
//!    δ_s(v) = Σ_{w : v ∈ pred(w)} (σ_v / σ_w) · (1 + δ_s(w)).
//!    ```
//!
//! The adjacency list is interpreted as a **directed** graph. For an undirected
//! graph supply a symmetric adjacency list; the returned scores then follow the
//! usual convention of including each unordered pair twice (the values are *not*
//! halved here, matching the directed accumulation).
//!
//! Reference: U. Brandes, "A faster algorithm for betweenness centrality",
//! Journal of Mathematical Sociology, 2001.

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};

/// Compute the (shortest-path) betweenness centrality of every node.
///
/// `adj[u]` lists the out-neighbours of node `u`. The result has length
/// `n_nodes`; `result[v]` is the betweenness score of node `v`. Scores are
/// non-negative.
///
/// # Errors
/// - [`GraphError::EmptyGraph`] when `n_nodes == 0`.
/// - [`GraphError::InvalidPlan`] when an adjacency entry references a node
///   `>= n_nodes`.
pub fn betweenness_centrality(adj: &[Vec<usize>], n_nodes: usize) -> GraphResult<Vec<f64>> {
    if n_nodes == 0 {
        return Err(GraphError::EmptyGraph);
    }
    for (u, heads) in adj.iter().enumerate() {
        for &w in heads {
            if w >= n_nodes {
                return Err(GraphError::InvalidPlan(format!(
                    "adjacency edge {u} -> {w} references node >= n_nodes={n_nodes}"
                )));
            }
        }
    }

    let mut centrality = vec![0.0f64; n_nodes];

    // Reusable per-source scratch buffers.
    let mut dist = vec![-1i64; n_nodes];
    let mut sigma = vec![0.0f64; n_nodes];
    let mut delta = vec![0.0f64; n_nodes];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    // Stack of nodes in order of discovery (non-decreasing distance).
    let mut order: Vec<usize> = Vec::with_capacity(n_nodes);
    let mut queue: VecDeque<usize> = VecDeque::new();

    for s in 0..n_nodes {
        // Reset scratch.
        for v in 0..n_nodes {
            dist[v] = -1;
            sigma[v] = 0.0;
            delta[v] = 0.0;
            preds[v].clear();
        }
        order.clear();
        queue.clear();

        dist[s] = 0;
        sigma[s] = 1.0;
        queue.push_back(s);

        // BFS: shortest-path counting + predecessor recording.
        while let Some(v) = queue.pop_front() {
            order.push(v);
            if let Some(neighbours) = adj.get(v) {
                for &w in neighbours {
                    // First time `w` is found: set its distance, enqueue it.
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }
                    // `v` lies on a shortest path to `w`.
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        preds[w].push(v);
                    }
                }
            }
        }

        // Dependency accumulation in reverse BFS order.
        while let Some(w) = order.pop() {
            let coeff = (1.0 + delta[w]) / sigma[w];
            for &v in &preds[w] {
                delta[v] += sigma[v] * coeff;
            }
            if w != s {
                centrality[w] += delta[w];
            }
        }
    }

    Ok(centrality)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric (undirected) adjacency list from unordered edges.
    fn undirected(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        adj
    }

    // 1. All scores are non-negative.
    #[test]
    fn nonneg() {
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
        let c = betweenness_centrality(&adj, 5).expect("betweenness_centrality should succeed");
        for &v in &c {
            assert!(v >= 0.0, "negative betweenness {v}");
        }
    }

    // 2. In a star, the centre has the strictly highest betweenness.
    #[test]
    fn star_center_highest() {
        // Node 0 is the hub of a 4-leaf star.
        let adj = undirected(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let c = betweenness_centrality(&adj, 5).expect("betweenness_centrality should succeed");
        for leaf in 1..5 {
            assert!(
                c[0] > c[leaf],
                "centre {} should exceed leaf {} = {}",
                c[0],
                leaf,
                c[leaf]
            );
            assert!(
                (c[leaf]).abs() < 1e-12,
                "leaf {leaf} should be 0, got {}",
                c[leaf]
            );
        }
    }

    // 3. In a path graph, the middle node has the highest betweenness.
    #[test]
    fn path_graph_middle_highest() {
        // 0 - 1 - 2 - 3 - 4 ; node 2 is the centre.
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let c = betweenness_centrality(&adj, 5).expect("betweenness_centrality should succeed");
        assert!(c[2] > c[1] && c[2] > c[3], "middle {} not the max", c[2]);
        assert!(c[1] > c[0] && c[3] > c[4], "endpoints should be lowest");
    }

    // 4. In a complete graph every node has equal (zero) betweenness, since all
    //    shortest paths are direct edges.
    #[test]
    fn complete_graph_zero() {
        let n = 4;
        let mut edges = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                edges.push((a, b));
            }
        }
        let adj = undirected(n, &edges);
        let c = betweenness_centrality(&adj, n).expect("betweenness_centrality should succeed");
        for &v in &c {
            assert!(
                v.abs() < 1e-12,
                "complete-graph betweenness {v} should be 0"
            );
        }
    }

    // 5. n_nodes == 0 errors.
    #[test]
    fn n_nodes_0_error() {
        let adj: Vec<Vec<usize>> = vec![];
        let err = betweenness_centrality(&adj, 0);
        assert!(matches!(err, Err(GraphError::EmptyGraph)), "got {err:?}");
    }

    // 6. An isolated node has zero betweenness.
    #[test]
    fn isolated_node_zero() {
        // Node 3 has no edges.
        let adj = undirected(4, &[(0, 1), (1, 2)]);
        let c = betweenness_centrality(&adj, 4).expect("betweenness_centrality should succeed");
        assert!(c[3].abs() < 1e-12, "isolated node betweenness {}", c[3]);
    }

    // 7. Output shape matches n_nodes.
    #[test]
    fn output_shape() {
        let adj = undirected(6, &[(0, 1), (2, 3), (4, 5)]);
        let c = betweenness_centrality(&adj, 6).expect("betweenness_centrality should succeed");
        assert_eq!(c.len(), 6);
    }

    // 8. A single node has zero betweenness.
    #[test]
    fn single_node_zero() {
        let adj = vec![vec![]];
        let c = betweenness_centrality(&adj, 1).expect("betweenness_centrality should succeed");
        assert_eq!(c.len(), 1);
        assert!(c[0].abs() < 1e-12, "single-node betweenness {}", c[0]);
    }

    // 9. Symmetric structure yields symmetric values (the two arms of a path are
    //    mirror images).
    #[test]
    fn symmetric_values() {
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let c = betweenness_centrality(&adj, 5).expect("betweenness_centrality should succeed");
        assert!((c[0] - c[4]).abs() < 1e-9, "endpoints {} vs {}", c[0], c[4]);
        assert!((c[1] - c[3]).abs() < 1e-9, "inner {} vs {}", c[1], c[3]);
    }

    // 10. Out-of-range adjacency reference errors.
    #[test]
    fn out_of_range_edge_error() {
        let adj = vec![vec![9]]; // node 9 >= n_nodes = 3
        let err = betweenness_centrality(&adj, 3);
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(_))),
            "got {err:?}"
        );
    }
}
