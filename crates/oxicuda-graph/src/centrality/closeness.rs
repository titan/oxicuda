//! Closeness centrality (Bavelas 1950; Wasserman-Faust normalisation).
//!
//! The closeness centrality of a node `v` rewards nodes that are, on average,
//! a small number of hops away from all other reachable nodes:
//! ```text
//! C(v) = (r_v - 1) / Σ_{u reachable from v} d(v, u)   ·   (r_v - 1) / (n - 1)
//! ```
//! where `d(v, u)` is the shortest-path distance, `r_v` is the number of nodes
//! reachable from `v` (including `v` itself), and `n` is the total number of
//! nodes. The leading factor is the classic inverse-mean-distance closeness; the
//! trailing `(r_v - 1)/(n - 1)` is the **Wasserman-Faust** correction that
//! down-weights nodes which can only reach a small fraction of a disconnected
//! graph, so that scores remain comparable across components. On a connected
//! graph (`r_v = n`) the correction is `1` and the formula reduces to the
//! textbook `(n - 1) / Σ d(v, u)`.
//!
//! Distances are computed with one breadth-first search per source, giving an
//! overall `O(n · (n + m))` running time for an unweighted directed graph.
//!
//! References:
//! - A. Bavelas, "Communication patterns in task-oriented groups", 1950.
//! - S. Wasserman & K. Faust, "Social Network Analysis", 1994 (disconnected-graph
//!   normalisation).

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};

/// Compute the closeness centrality of every node.
///
/// `adj[u]` lists the out-neighbours of node `u`. The result has length
/// `n_nodes`; `result[v] ∈ [0, 1]` is the (Wasserman-Faust normalised) closeness
/// of node `v`. A node that cannot reach any other node receives a score of `0`.
///
/// # Errors
/// - [`GraphError::EmptyGraph`] when `n_nodes == 0`.
/// - [`GraphError::InvalidPlan`] when an adjacency entry references a node
///   `>= n_nodes`.
pub fn closeness_centrality(adj: &[Vec<usize>], n_nodes: usize) -> GraphResult<Vec<f64>> {
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
    let mut dist = vec![-1i64; n_nodes];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for s in 0..n_nodes {
        for d in dist.iter_mut() {
            *d = -1;
        }
        queue.clear();
        dist[s] = 0;
        queue.push_back(s);

        let mut total_dist: f64 = 0.0;
        let mut reachable: usize = 1; // includes `s`

        while let Some(v) = queue.pop_front() {
            if let Some(neighbours) = adj.get(v) {
                for &w in neighbours {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        total_dist += dist[w] as f64;
                        reachable += 1;
                        queue.push_back(w);
                    }
                }
            }
        }

        // A node that reaches no one (or a 1-node graph) has closeness 0.
        if reachable > 1 && total_dist > 0.0 {
            let inv_mean = (reachable as f64 - 1.0) / total_dist;
            let wf = (reachable as f64 - 1.0) / (n_nodes as f64 - 1.0);
            centrality[s] = inv_mean * wf;
        }
    }

    Ok(centrality)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn undirected(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        adj
    }

    // 1. All scores lie in [0, 1].
    #[test]
    fn in_unit_range() {
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let c = closeness_centrality(&adj, 5).expect("closeness_centrality should succeed");
        for &v in &c {
            assert!(
                (0.0..=1.0 + 1e-12).contains(&v),
                "out-of-range closeness {v}"
            );
        }
    }

    // 2. In a star, the centre is the closest to everyone.
    #[test]
    fn star_center_highest() {
        let adj = undirected(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let c = closeness_centrality(&adj, 5).expect("closeness_centrality should succeed");
        for leaf in 1..5 {
            assert!(
                c[0] > c[leaf],
                "centre {} should exceed leaf {} = {}",
                c[0],
                leaf,
                c[leaf]
            );
        }
        // Centre reaches every other node in 1 hop: closeness = 1.
        assert!((c[0] - 1.0).abs() < 1e-12, "centre closeness {} != 1", c[0]);
    }

    // 3. In a path graph the middle node has the highest closeness.
    #[test]
    fn path_graph_middle_highest() {
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let c = closeness_centrality(&adj, 5).expect("closeness_centrality should succeed");
        assert!(c[2] > c[1] && c[2] > c[3], "middle {} not the max", c[2]);
        assert!(c[1] > c[0] && c[3] > c[4], "endpoints should be lowest");
    }

    // 4. In a complete graph all closeness values equal 1.
    #[test]
    fn complete_graph_uniform() {
        let n = 4;
        let mut edges = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                edges.push((a, b));
            }
        }
        let adj = undirected(n, &edges);
        let c = closeness_centrality(&adj, n).expect("closeness_centrality should succeed");
        for &v in &c {
            assert!((v - 1.0).abs() < 1e-12, "complete-graph closeness {v} != 1");
        }
    }

    // 5. n_nodes == 0 errors.
    #[test]
    fn n_nodes_0_error() {
        let adj: Vec<Vec<usize>> = vec![];
        let err = closeness_centrality(&adj, 0);
        assert!(matches!(err, Err(GraphError::EmptyGraph)), "got {err:?}");
    }

    // 6. An isolated node has zero closeness.
    #[test]
    fn isolated_node_zero() {
        let adj = undirected(4, &[(0, 1), (1, 2)]);
        let c = closeness_centrality(&adj, 4).expect("closeness_centrality should succeed");
        assert!(c[3].abs() < 1e-12, "isolated node closeness {}", c[3]);
    }

    // 7. Output shape matches n_nodes.
    #[test]
    fn output_shape() {
        let adj = undirected(6, &[(0, 1), (2, 3), (4, 5)]);
        let c = closeness_centrality(&adj, 6).expect("closeness_centrality should succeed");
        assert_eq!(c.len(), 6);
    }

    // 8. A single node has zero closeness (reaches nobody).
    #[test]
    fn single_node_zero() {
        let adj = vec![vec![]];
        let c = closeness_centrality(&adj, 1).expect("closeness_centrality should succeed");
        assert_eq!(c.len(), 1);
        assert!(c[0].abs() < 1e-12, "single-node closeness {}", c[0]);
    }

    // 9. Symmetric structure yields symmetric values.
    #[test]
    fn symmetric_values() {
        let adj = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let c = closeness_centrality(&adj, 5).expect("closeness_centrality should succeed");
        assert!((c[0] - c[4]).abs() < 1e-9, "endpoints {} vs {}", c[0], c[4]);
        assert!((c[1] - c[3]).abs() < 1e-9, "inner {} vs {}", c[1], c[3]);
    }

    // 10. The Wasserman-Faust correction down-weights a node trapped in a small
    //     component relative to one in a large component of the same graph.
    #[test]
    fn disconnected_components_normalised() {
        // Component A: 0-1-2 (3 nodes). Component B: 3-4 (2 nodes).
        let adj = undirected(5, &[(0, 1), (1, 2), (3, 4)]);
        let c = closeness_centrality(&adj, 5).expect("closeness_centrality should succeed");
        // Node 1 (hub of the larger component) reaches 2 others; node 3 reaches 1.
        // The WF normalisation makes node 1 strictly more central than node 3.
        assert!(
            c[1] > c[3],
            "large-component hub {} <= small-component {}",
            c[1],
            c[3]
        );
        for &v in &c {
            assert!(v >= 0.0, "negative closeness {v}");
        }
    }

    // 11. Out-of-range adjacency reference errors.
    #[test]
    fn out_of_range_edge_error() {
        let adj = vec![vec![7]];
        let err = closeness_centrality(&adj, 3);
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(_))),
            "got {err:?}"
        );
    }
}
