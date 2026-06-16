//! PageRank centrality (Brin & Page 1998).
//!
//! PageRank ranks the nodes of a directed graph by the stationary
//! distribution of a random walk that, with probability `damping`, follows an
//! out-edge chosen uniformly at random and, with probability `1 - damping`,
//! teleports to a uniformly random node.
//!
//! The score vector `r` is the fixed point of
//! ```text
//! r[i] = (1 - d) / n  +  d * ( Σ_{j → i} r[j] / outdeg[j]  +  (1/n) Σ_{k dangling} r[k] )
//! ```
//! where the second inner sum redistributes the mass held by *dangling* nodes
//! (nodes with no out-edges) uniformly across the graph, which keeps the total
//! probability mass conserved at 1.
//!
//! The fixed point is approximated by **power iteration**: starting from the
//! uniform distribution `r = 1/n`, the update above is applied until the
//! L1 change between successive iterates falls below `tol` (convergence) or
//! `max_iter` iterations have elapsed.
//!
//! Reference: S. Brin and L. Page, "The anatomy of a large-scale hypertextual
//! Web search engine", Computer Networks and ISDN Systems, 1998.

use crate::error::{GraphError, GraphResult};

/// Hyper-parameters for the PageRank power iteration.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor `d` (probability of following an out-edge rather than
    /// teleporting). The classic value is `0.85`. Must lie in `[0, 1]`.
    pub damping: f64,
    /// Maximum number of power-iteration sweeps.
    pub max_iter: usize,
    /// Convergence tolerance on the L1 distance between successive iterates.
    pub tol: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iter: 100,
            tol: 1e-9,
        }
    }
}

/// Compute the PageRank score of every node in a directed graph.
///
/// `adj[j]` lists the heads of the out-edges of node `j` (i.e. `j → adj[j][*]`).
/// The returned vector has length `n_nodes`, holds non-negative scores that sum
/// to `1`, and is ordered so that `result[i]` is the score of node `i`.
///
/// # Errors
/// - [`GraphError::EmptyGraph`] when `n_nodes == 0`.
/// - [`GraphError::InvalidPlan`] when `damping` is outside `[0, 1]` or when an
///   adjacency entry references a node `>= n_nodes`.
pub fn pagerank(adj: &[Vec<usize>], n_nodes: usize, cfg: &PageRankConfig) -> GraphResult<Vec<f64>> {
    if n_nodes == 0 {
        return Err(GraphError::EmptyGraph);
    }
    if !(0.0..=1.0).contains(&cfg.damping) || !cfg.damping.is_finite() {
        return Err(GraphError::InvalidPlan(format!(
            "damping must be in [0, 1], got {}",
            cfg.damping
        )));
    }
    // Validate adjacency references. Nodes beyond `adj.len()` are treated as
    // isolated (no out-edges), which is permitted.
    for (j, heads) in adj.iter().enumerate() {
        for &i in heads {
            if i >= n_nodes {
                return Err(GraphError::InvalidPlan(format!(
                    "adjacency edge {j} -> {i} references node >= n_nodes={n_nodes}"
                )));
            }
        }
    }

    let n_f = n_nodes as f64;
    let teleport = (1.0 - cfg.damping) / n_f;

    // Out-degree of each node (0 for dangling / out-of-range nodes).
    let mut outdeg = vec![0usize; n_nodes];
    for (j, heads) in adj.iter().enumerate().take(n_nodes) {
        outdeg[j] = heads.len();
    }

    let mut r = vec![1.0 / n_f; n_nodes];
    let mut next = vec![0.0; n_nodes];

    for _ in 0..cfg.max_iter {
        // Mass currently held by dangling nodes is redistributed uniformly.
        let mut dangling_mass = 0.0;
        for i in 0..n_nodes {
            if outdeg[i] == 0 {
                dangling_mass += r[i];
            }
        }
        let base = teleport + cfg.damping * dangling_mass / n_f;
        for slot in next.iter_mut() {
            *slot = base;
        }
        // Push each node's rank to its out-neighbours.
        for (j, heads) in adj.iter().enumerate().take(n_nodes) {
            if heads.is_empty() {
                continue;
            }
            let share = cfg.damping * r[j] / heads.len() as f64;
            for &i in heads {
                next[i] += share;
            }
        }

        // L1 convergence check.
        let mut delta = 0.0;
        for i in 0..n_nodes {
            delta += (next[i] - r[i]).abs();
        }
        std::mem::swap(&mut r, &mut next);
        if delta < cfg.tol {
            break;
        }
    }

    // Renormalise to guard against accumulated floating-point drift.
    let sum: f64 = r.iter().sum();
    if sum > 0.0 {
        for v in &mut r {
            *v /= sum;
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PageRankConfig {
        PageRankConfig::default()
    }

    // 1. Scores sum to 1.
    #[test]
    fn sums_to_1() {
        let adj = vec![vec![1, 2], vec![2], vec![0]];
        let r = pagerank(&adj, 3, &cfg()).expect("value should be present");
        let s: f64 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum = {s}");
    }

    // 2. All scores strictly positive (teleport guarantees this).
    #[test]
    fn all_positive() {
        let adj = vec![vec![1], vec![2], vec![0]];
        let r = pagerank(&adj, 3, &cfg()).expect("value should be present");
        for &v in &r {
            assert!(v > 0.0, "non-positive score {v}");
        }
    }

    // 3. n_nodes == 0 errors.
    #[test]
    fn n_nodes_0_error() {
        let adj: Vec<Vec<usize>> = vec![];
        let err = pagerank(&adj, 0, &cfg());
        assert!(matches!(err, Err(GraphError::EmptyGraph)), "got {err:?}");
    }

    // 4. A hub pointed at by many nodes earns a higher rank.
    #[test]
    fn hub_higher_rank() {
        // Nodes 1,2,3 all point to node 0; node 0 points back to 1.
        let adj = vec![vec![1], vec![0], vec![0], vec![0]];
        let r = pagerank(&adj, 4, &cfg()).expect("value should be present");
        assert!(
            r[0] > r[1] && r[0] > r[2] && r[0] > r[3],
            "hub {} should exceed {:?}",
            r[0],
            &r[1..]
        );
    }

    // 5. A symmetric (vertex-transitive) ring yields a uniform distribution.
    #[test]
    fn symmetric_graph_uniform() {
        // Bidirectional 4-cycle: every node has identical structure.
        let adj = vec![vec![1, 3], vec![0, 2], vec![1, 3], vec![2, 0]];
        let r = pagerank(&adj, 4, &cfg()).expect("value should be present");
        for &v in &r {
            assert!((v - 0.25).abs() < 1e-6, "score {v} not ~0.25");
        }
    }

    // 6. A dangling node (no out-edges) is handled without losing mass.
    #[test]
    fn dangling_node_handled() {
        // Node 2 is dangling.
        let adj = vec![vec![1], vec![2], vec![]];
        let r = pagerank(&adj, 3, &cfg()).expect("value should be present");
        let s: f64 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum = {s}");
        for &v in &r {
            assert!(v > 0.0, "score {v} should be positive");
        }
    }

    // 7. damping == 0 collapses to the uniform teleport distribution.
    #[test]
    fn damping_0_uniform() {
        let adj = vec![vec![1], vec![0], vec![0], vec![1]];
        let c = PageRankConfig {
            damping: 0.0,
            ..cfg()
        };
        let r = pagerank(&adj, 4, &c).expect("pagerank should succeed");
        for &v in &r {
            assert!((v - 0.25).abs() < 1e-9, "score {v} not uniform");
        }
    }

    // 8. Power iteration converges (running with a tight tolerance succeeds and
    //    re-running with more iterations does not change the result materially).
    #[test]
    fn converges() {
        let adj = vec![vec![1, 2], vec![2], vec![0], vec![0, 1]];
        let r1 = pagerank(&adj, 4, &cfg()).expect("value should be present");
        let c2 = PageRankConfig {
            max_iter: 1000,
            tol: 1e-12,
            ..cfg()
        };
        let r2 = pagerank(&adj, 4, &c2).expect("pagerank should succeed");
        for i in 0..4 {
            assert!(
                (r1[i] - r2[i]).abs() < 1e-6,
                "node {i}: {} vs {}",
                r1[i],
                r2[i]
            );
        }
    }

    // 9. A single node receives the entire mass.
    #[test]
    fn single_node() {
        let adj = vec![vec![]];
        let r = pagerank(&adj, 1, &cfg()).expect("value should be present");
        assert_eq!(r.len(), 1);
        assert!((r[0] - 1.0).abs() < 1e-12, "score {}", r[0]);
    }

    // 10. Out-of-range adjacency reference errors.
    #[test]
    fn out_of_range_edge_error() {
        let adj = vec![vec![5]]; // node 5 >= n_nodes = 2
        let err = pagerank(&adj, 2, &cfg());
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(_))),
            "got {err:?}"
        );
    }

    // 11. Invalid damping errors.
    #[test]
    fn invalid_damping_error() {
        let adj = vec![vec![1], vec![0]];
        let c = PageRankConfig {
            damping: 1.5,
            ..cfg()
        };
        let err = pagerank(&adj, 2, &c);
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(_))),
            "got {err:?}"
        );
    }
}
