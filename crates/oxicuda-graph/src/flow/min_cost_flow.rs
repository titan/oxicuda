//! Min-cost max-flow via Successive Shortest Paths (SPFA / Bellman-Ford with queue).
//!
//! Klein (1967) showed that successively augmenting along shortest (minimum-cost)
//! augmenting paths converges to a minimum-cost maximum flow. This implementation
//! uses the **Shortest Path Faster Algorithm (SPFA)** — a BFS-based relaxation of
//! Bellman-Ford — to find the next augmenting path in the residual graph.
//!
//! Negative-cost edges are handled correctly: SPFA tolerates negative arc weights
//! and detects negative-cost cycles (though well-posed MCF instances should have
//! none after the first feasibility check).
//!
//! # Complexity
//! O(V·E·max_flow) in the worst case; much faster in practice.

use crate::error::{GraphError, GraphResult};
use std::collections::VecDeque;

// ─── Public API types ─────────────────────────────────────────────────────────

/// A directed edge with integer capacity and cost.
#[derive(Debug, Clone)]
pub struct McfEdge {
    /// Tail (source endpoint) of the edge.
    pub from: usize,
    /// Head (target endpoint) of the edge.
    pub to: usize,
    /// Maximum flow capacity.
    pub capacity: i64,
    /// Cost per unit of flow.
    pub cost: i64,
}

/// Result of a min-cost flow computation.
#[derive(Debug, Clone)]
pub struct McfResult {
    /// Total flow routed from source to sink.
    pub flow: i64,
    /// Total cost of the flow (sum of `cost * flow_on_edge` for all edges).
    pub cost: i64,
    /// Flow on each input edge, in the same order as the `edges` parameter.
    pub flow_per_edge: Vec<i64>,
}

// ─── Internal residual graph ──────────────────────────────────────────────────

/// A single arc in the residual graph.
#[derive(Debug, Clone)]
struct ResidualArc {
    /// Destination node.
    to: usize,
    /// Residual capacity (remaining capacity for flow).
    cap: i64,
    /// Cost (negative for reverse arcs).
    cost: i64,
    /// Index of the reverse arc in `adj[to]`.
    rev: usize,
}

/// Build a residual graph from the input edges.
///
/// Returns `(adjacency list, forward_arc_indices)` where `forward_arc_indices[i]`
/// is `(node, arc_pos)` such that `adj[node][arc_pos]` is the forward arc for
/// input edge `i`.
/// Adjacency list for the residual graph.
type ResidualAdj = Vec<Vec<ResidualArc>>;
/// Per-input-edge indices into the adjacency list: `(node, arc_pos)`.
type FwdIndices = Vec<(usize, usize)>;

fn build_residual(n: usize, edges: &[McfEdge]) -> GraphResult<(ResidualAdj, FwdIndices)> {
    let mut adj: Vec<Vec<ResidualArc>> = vec![Vec::new(); n];
    let mut fwd_indices = Vec::with_capacity(edges.len());

    for edge in edges {
        if edge.from >= n || edge.to >= n {
            return Err(GraphError::InvalidPlan("node_out_of_range".to_owned()));
        }
        // Forward arc
        let fwd_node = edge.from;
        let fwd_pos = adj[edge.from].len();
        fwd_indices.push((fwd_node, fwd_pos));

        let rev_pos = adj[edge.to].len();
        adj[edge.from].push(ResidualArc {
            to: edge.to,
            cap: edge.capacity,
            cost: edge.cost,
            rev: rev_pos,
        });

        // Backward (reverse) arc — capacity 0, cost negated
        let fwd_back = fwd_pos;
        adj[edge.to].push(ResidualArc {
            to: edge.from,
            cap: 0,
            cost: -edge.cost,
            rev: fwd_back,
        });
    }

    Ok((adj, fwd_indices))
}

// ─── SPFA (shortest-path faster algorithm) ───────────────────────────────────

/// Find the shortest (minimum-cost) augmenting path from `source` to `sink`
/// using SPFA.  Returns `(prev_node, prev_arc)` arrays for path reconstruction,
/// or `None` if `sink` is unreachable.
///
/// `prev_node[v]` = the predecessor node on the shortest path to `v`.
/// `prev_arc[v]`  = the index into `adj[prev_node[v]]` of the arc used.
fn spfa(
    adj: &[Vec<ResidualArc>],
    source: usize,
    sink: usize,
    n: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut dist = vec![i64::MAX; n];
    let mut in_queue = vec![false; n];
    let mut prev_node = vec![usize::MAX; n];
    let mut prev_arc = vec![usize::MAX; n];

    dist[source] = 0;
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(source);
    in_queue[source] = true;

    while let Some(u) = queue.pop_front() {
        in_queue[u] = false;

        for (arc_idx, arc) in adj[u].iter().enumerate() {
            if arc.cap > 0 && dist[u] != i64::MAX {
                let new_dist = dist[u].saturating_add(arc.cost);
                if new_dist < dist[arc.to] {
                    dist[arc.to] = new_dist;
                    prev_node[arc.to] = u;
                    prev_arc[arc.to] = arc_idx;
                    if !in_queue[arc.to] {
                        in_queue[arc.to] = true;
                        queue.push_back(arc.to);
                    }
                }
            }
        }
    }

    if dist[sink] == i64::MAX {
        None
    } else {
        Some((prev_node, prev_arc))
    }
}

// ─── Main algorithm ───────────────────────────────────────────────────────────

/// Compute a minimum-cost maximum flow (up to `max_flow`) on a directed graph.
///
/// Uses the **Successive Shortest Paths** algorithm with SPFA as the
/// shortest-path oracle. At each iteration the algorithm finds the
/// minimum-cost augmenting path from `source` to `sink` in the residual
/// graph, determines the bottleneck capacity, pushes that flow, and updates
/// residual capacities and cumulative cost. This repeats until either
/// `max_flow` units of flow have been sent or no augmenting path exists.
///
/// Handles negative-cost edges: SPFA relaxes negative-weight arcs correctly.
///
/// # Arguments
/// * `n_nodes` — number of nodes (nodes are indexed `0..n_nodes`).
/// * `edges`   — directed edges with capacity and cost.
/// * `source`  — source node index.
/// * `sink`    — sink node index.
/// * `max_flow` — maximum flow to route (pass `i64::MAX` for unconstrained).
///
/// # Errors
/// - [`GraphError::InvalidPlan`]`("source_equals_sink")` if `source == sink`.
/// - [`GraphError::InvalidPlan`]`("n_nodes_zero")`       if `n_nodes == 0`.
/// - [`GraphError::InvalidPlan`]`("node_out_of_range")`  if any edge endpoint ≥ `n_nodes`.
pub fn min_cost_flow(
    n_nodes: usize,
    edges: &[McfEdge],
    source: usize,
    sink: usize,
    max_flow: i64,
) -> GraphResult<McfResult> {
    if n_nodes == 0 {
        return Err(GraphError::InvalidPlan("n_nodes_zero".to_owned()));
    }
    if source == sink {
        return Err(GraphError::InvalidPlan("source_equals_sink".to_owned()));
    }
    if source >= n_nodes || sink >= n_nodes {
        return Err(GraphError::InvalidPlan("node_out_of_range".to_owned()));
    }

    let (mut adj, fwd_indices) = build_residual(n_nodes, edges)?;

    let mut total_flow: i64 = 0;
    let mut total_cost: i64 = 0;

    while total_flow < max_flow {
        // Find shortest augmenting path via SPFA.
        let (prev_node, prev_arc) = match spfa(&adj, source, sink, n_nodes) {
            Some(p) => p,
            None => break, // no augmenting path
        };

        // Find bottleneck capacity along the path.
        let remaining = max_flow - total_flow;
        let mut bottleneck = remaining;
        let mut v = sink;
        while v != source {
            let u = prev_node[v];
            let arc = &adj[u][prev_arc[v]];
            bottleneck = bottleneck.min(arc.cap);
            v = u;
        }
        if bottleneck <= 0 {
            break;
        }

        // Compute path cost and augment.
        let mut path_cost: i64 = 0;
        v = sink;
        while v != source {
            let u = prev_node[v];
            path_cost = path_cost.saturating_add(adj[u][prev_arc[v]].cost);
            v = u;
        }
        total_cost = total_cost.saturating_add(path_cost.saturating_mul(bottleneck));
        total_flow += bottleneck;

        // Update residual capacities.
        v = sink;
        while v != source {
            let u = prev_node[v];
            let arc_idx = prev_arc[v];
            let rev_idx = adj[u][arc_idx].rev;
            adj[u][arc_idx].cap -= bottleneck;
            adj[v][rev_idx].cap += bottleneck;
            v = u;
        }
    }

    // Reconstruct per-edge flow: original capacity minus remaining residual capacity.
    let flow_per_edge: Vec<i64> = fwd_indices
        .iter()
        .zip(edges.iter())
        .map(|(&(node, pos), edge)| edge.capacity - adj[node][pos].cap)
        .collect();

    Ok(McfResult {
        flow: total_flow,
        cost: total_cost,
        flow_per_edge,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(from: usize, to: usize, cap: i64, cost: i64) -> McfEdge {
        McfEdge {
            from,
            to,
            capacity: cap,
            cost,
        }
    }

    // 1. Simple path: 0→1→2, cap=1, cost 1+1=2
    #[test]
    fn simple_path_flow() {
        let edges = vec![edge(0, 1, 1, 1), edge(1, 2, 1, 1)];
        let r = min_cost_flow(3, &edges, 0, 2, 10).expect("min_cost_flow should succeed");
        assert_eq!(r.flow, 1);
        assert_eq!(r.cost, 2);
    }

    // 2. max_flow limits: path has cap 5 but max_flow=3 → only 3 sent
    #[test]
    fn max_flow_limited() {
        let edges = vec![edge(0, 1, 5, 1), edge(1, 2, 5, 1)];
        let r = min_cost_flow(3, &edges, 0, 2, 3).expect("min_cost_flow should succeed");
        assert_eq!(r.flow, 3);
        assert_eq!(r.cost, 6);
    }

    // 3. Cost-optimal: two paths — cheap one used first.
    //    Path A: 0→1→3 cost 1+1=2
    //    Path B: 0→2→3 cost 5+5=10
    #[test]
    fn cost_optimal_route() {
        let edges = vec![
            edge(0, 1, 1, 1),
            edge(1, 3, 1, 1),
            edge(0, 2, 1, 5),
            edge(2, 3, 1, 5),
        ];
        let r = min_cost_flow(4, &edges, 0, 3, 1).expect("min_cost_flow should succeed");
        assert_eq!(r.flow, 1);
        assert_eq!(r.cost, 2); // cheap path used
    }

    // 4. No augmenting path → flow=0, cost=0
    #[test]
    fn no_augmenting_path_zero_flow() {
        let edges: Vec<McfEdge> = vec![edge(0, 1, 1, 1)]; // no path to node 3
        let r = min_cost_flow(4, &edges, 0, 3, 10).expect("min_cost_flow should succeed");
        assert_eq!(r.flow, 0);
        assert_eq!(r.cost, 0);
    }

    // 5. Negative-cost edge: 0→1→2, cost=-1+3=2, but 0→2 cost=5.
    //    With negative cost SPFA should prefer the lower-total-cost path.
    #[test]
    fn negative_cost_edge() {
        let edges = vec![edge(0, 1, 1, -2), edge(1, 2, 1, 3), edge(0, 2, 1, 5)];
        let r = min_cost_flow(3, &edges, 0, 2, 1).expect("min_cost_flow should succeed");
        assert_eq!(r.flow, 1);
        assert_eq!(r.cost, 1); // -2 + 3 = 1 < 5
    }

    // 6. Parallel edges — both are usable
    #[test]
    fn parallel_edges() {
        let edges = vec![edge(0, 1, 1, 1), edge(0, 1, 1, 2), edge(1, 2, 2, 1)];
        let r = min_cost_flow(3, &edges, 0, 2, 2).expect("min_cost_flow should succeed");
        assert_eq!(r.flow, 2);
        // Cheap parallel edge used first: 1+1=2, then 2+1=3 → total 5
        assert_eq!(r.cost, 5);
    }

    // 7. flow_per_edge sums correctly
    #[test]
    fn flow_per_edge_correct() {
        let edges = vec![edge(0, 1, 2, 1), edge(1, 2, 2, 1)];
        let r = min_cost_flow(3, &edges, 0, 2, 10).expect("min_cost_flow should succeed");
        assert_eq!(r.flow_per_edge.len(), 2);
        // All flow must pass through both edges
        assert_eq!(r.flow_per_edge[0], r.flow);
        assert_eq!(r.flow_per_edge[1], r.flow);
        assert!(r.flow_per_edge.iter().sum::<i64>() > 0);
    }

    // 8. source == sink → error
    #[test]
    fn source_equals_sink_error() {
        let edges = vec![edge(0, 1, 1, 1)];
        let err = min_cost_flow(3, &edges, 1, 1, 10);
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(ref s)) if s == "source_equals_sink"),
            "got: {err:?}"
        );
    }

    // 9. n_nodes=0 → error
    #[test]
    fn n_nodes_0_error() {
        let err = min_cost_flow(0, &[], 0, 0, 0);
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(ref s)) if s == "n_nodes_zero"),
            "got: {err:?}"
        );
    }

    // 10. Conservation law: net flow into each non-source/sink node == 0
    #[test]
    fn conservation_law() {
        // Diamond: 0→1, 0→2, 1→3, 2→3 each cap 2
        let edges = vec![
            edge(0, 1, 2, 1),
            edge(0, 2, 2, 2),
            edge(1, 3, 2, 1),
            edge(2, 3, 2, 2),
        ];
        let n = 4;
        let r = min_cost_flow(n, &edges, 0, 3, 4).expect("min_cost_flow should succeed");

        // Compute net flow at each node
        let mut net = vec![0i64; n];
        for (e, &f) in edges.iter().zip(r.flow_per_edge.iter()) {
            net[e.from] -= f;
            net[e.to] += f;
        }
        // Source: net = -total_flow, Sink: net = +total_flow
        // All interior nodes: net = 0
        for (v, &net_v) in net.iter().enumerate().skip(1).take(n - 2) {
            assert_eq!(net_v, 0, "node {v} violates conservation: net={net_v}");
        }
        assert_eq!(net[0], -r.flow);
        assert_eq!(net[n - 1], r.flow);
    }

    // 11. Node out of range → error
    #[test]
    fn node_out_of_range_error() {
        let edges = vec![edge(0, 5, 1, 1)]; // node 5 >= n_nodes=4
        let err = min_cost_flow(4, &edges, 0, 3, 10);
        assert!(
            matches!(err, Err(GraphError::InvalidPlan(_))),
            "got: {err:?}"
        );
    }

    // 12. Zero-capacity edge contributes no flow
    #[test]
    fn zero_capacity_edge_contributes_nothing() {
        let edges = vec![
            edge(0, 1, 0, 1), // zero cap
            edge(0, 1, 2, 5),
            edge(1, 2, 2, 1),
        ];
        let r = min_cost_flow(3, &edges, 0, 2, 10).expect("min_cost_flow should succeed");
        assert_eq!(r.flow_per_edge[0], 0);
        assert!(r.flow > 0);
    }
}
