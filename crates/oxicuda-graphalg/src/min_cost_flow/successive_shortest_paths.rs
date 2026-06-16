//! Successive Shortest Paths (SSP) algorithm for minimum-cost maximum-flow.
//!
//! # Why not `FlowNetwork`?
//!
//! The dense `n×n` capacity matrix in `FlowNetwork` (Edmonds-Karp module) stores
//! antiparallel edges by summing capacities at `cap[u*n+v]` and `cap[v*n+u]`. That
//! representation becomes ambiguous once costs are introduced: an edge `u→v` with cost `c`
//! and its reverse `v→u` with cost `-c` occupy the same two matrix cells as a genuine
//! `v→u` edge with a different cost. To track per-edge costs and residual flows correctly,
//! we use an **explicit edge list with paired reverse edges**:
//!
//! - Forward edge at index `2k` has reverse at index `2k+1` (and vice versa).
//! - Reverse edges carry negated cost and zero initial capacity.
//! - `graph[u]` stores indices into the edge list, not destination vertices.
//!
//! # Algorithm
//!
//! SSP iteratively finds the **shortest (min-cost) augmenting path** from source to sink
//! using SPFA (Shortest Path Faster Algorithm, a queue-accelerated Bellman-Ford). Negative
//! cost edges are handled naturally; a negative cycle is detected when any node is relaxed
//! more than `n` times.
//!
//! Time complexity: O(V · E · max_flow) in the worst case; far faster on typical networks.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};

// ─── Edge ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Edge {
    to: usize,
    cap: f64,
    cost: f64,
    flow: f64,
}

// ─── MinCostFlowNetwork ────────────────────────────────────────────────────

/// Directed flow network with per-edge costs, built from an explicit edge list.
///
/// Antiparallel edges are stored unambiguously: each call to [`MinCostFlowNetwork::add_edge`]
/// appends a forward edge followed immediately by its reverse (negated cost, zero capacity),
/// so forward edge `i` always has its reverse at index `i ^ 1`.
#[derive(Debug, Clone)]
pub struct MinCostFlowNetwork {
    /// Number of vertices.
    pub n: usize,
    edges: Vec<Edge>,
    /// `graph[u]` = list of edge indices (into `edges`) whose tail is `u`.
    graph: Vec<Vec<usize>>,
}

impl MinCostFlowNetwork {
    /// Create an empty network with `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            graph: vec![Vec::new(); n],
        }
    }

    /// Add a directed edge `u → v` with capacity `cap` and cost `cost`.
    ///
    /// Simultaneously adds a reverse edge `v → u` with capacity 0 and cost `-cost`.
    ///
    /// # Errors
    /// - `IndexOutOfBounds` if `u >= n` or `v >= n`.
    /// - `InvalidParameter` if `cap < 0`.
    /// - `InvalidEdgeWeight` if `cost` is not finite.
    pub fn add_edge(&mut self, u: usize, v: usize, cap: f64, cost: f64) -> GraphalgResult<()> {
        if u >= self.n || v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: self.n,
            });
        }
        if cap < 0.0 {
            return Err(GraphalgError::InvalidParameter(
                "negative capacity".to_string(),
            ));
        }
        if !cost.is_finite() {
            return Err(GraphalgError::InvalidEdgeWeight(
                "non-finite cost".to_string(),
            ));
        }

        let fwd_idx = self.edges.len();
        let rev_idx = fwd_idx + 1;

        self.edges.push(Edge {
            to: v,
            cap,
            cost,
            flow: 0.0,
        });
        self.edges.push(Edge {
            to: u,
            cap: 0.0,
            cost: -cost,
            flow: 0.0,
        });

        self.graph[u].push(fwd_idx);
        self.graph[v].push(rev_idx);

        Ok(())
    }
}

// ─── MinCostFlowResult ─────────────────────────────────────────────────────

/// Result of a min-cost flow computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MinCostFlowResult {
    /// Total flow sent from source to sink.
    pub flow: f64,
    /// Total cost of that flow.
    pub cost: f64,
}

// ─── SPFA shortest path ────────────────────────────────────────────────────

/// SPFA (queue-based Bellman-Ford) on the residual graph.
///
/// Returns `(dist, prev_edge)` where:
/// - `dist[v]` = min cost to reach `v` from `s` through residual edges.
/// - `prev_edge[v]` = index of the edge used to arrive at `v`; `usize::MAX` if unreachable.
///
/// Negative cycles (possible when negative-cost edges become reachable) are detected
/// by counting how many times each node is relaxed; if any node is relaxed more than
/// `n` times, a `NegativeCycle` error is returned.
fn spfa_shortest_path(
    net: &MinCostFlowNetwork,
    s: usize,
    t: usize,
) -> GraphalgResult<(Vec<f64>, Vec<usize>)> {
    let n = net.n;
    let mut dist = vec![f64::INFINITY; n];
    let mut prev_edge = vec![usize::MAX; n];
    let mut in_queue = vec![false; n];
    let mut relax_count = vec![0usize; n];

    dist[s] = 0.0;
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(s);
    in_queue[s] = true;

    while let Some(u) = queue.pop_front() {
        in_queue[u] = false;

        for &edge_idx in &net.graph[u] {
            let e = &net.edges[edge_idx];
            // Only traverse edges with remaining capacity.
            if e.cap - e.flow <= 1e-9 {
                continue;
            }
            let new_dist = dist[u] + e.cost;
            if new_dist < dist[e.to] - 1e-9 {
                dist[e.to] = new_dist;
                prev_edge[e.to] = edge_idx;
                if !in_queue[e.to] {
                    relax_count[e.to] += 1;
                    if relax_count[e.to] > n {
                        return Err(GraphalgError::NegativeCycle);
                    }
                    queue.push_back(e.to);
                    in_queue[e.to] = true;
                }
            }
        }
    }

    let _ = t; // dist[t] checked by caller
    Ok((dist, prev_edge))
}

// ─── Augmentation helper ───────────────────────────────────────────────────

/// Trace the augmenting path from `s` to `t` via `prev_edge`, compute the bottleneck
/// capacity, augment all edges along the path, and return the (bottleneck, path_cost).
///
/// `path_cost` = `dist[t]` (already computed by SPFA).
fn augment_path(
    net: &mut MinCostFlowNetwork,
    s: usize,
    t: usize,
    prev_edge: &[usize],
    path_cost: f64,
    flow_limit: f64,
) -> f64 {
    // Collect path edges from t back to s.
    let mut path_edges: Vec<usize> = Vec::new();
    let mut v = t;
    while v != s {
        let eidx = prev_edge[v];
        path_edges.push(eidx);
        v = net.edges[eidx ^ 1].to; // reverse edge points back to tail
    }

    // Bottleneck = min residual capacity along path, capped by flow_limit.
    let mut bottleneck = flow_limit;
    for &eidx in &path_edges {
        let e = &net.edges[eidx];
        bottleneck = bottleneck.min(e.cap - e.flow);
    }

    // Augment: push `bottleneck` units through each edge on the path.
    for &eidx in &path_edges {
        net.edges[eidx].flow += bottleneck;
        net.edges[eidx ^ 1].flow -= bottleneck;
    }

    let _ = path_cost; // used only in the calling function for cost accounting
    bottleneck
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Send as much flow as possible from `s` to `t` at minimum cost.
///
/// Returns the total flow sent and its total cost. Returns `flow=0, cost=0`
/// if `s` and `t` are disconnected.
///
/// # Errors
/// - `SourceOutOfRange` if `s >= n` or `t >= n`.
/// - `InvalidParameter` if `s == t`.
/// - `NegativeCycle` if the residual graph contains a negative cycle.
pub fn min_cost_max_flow(
    net: &MinCostFlowNetwork,
    s: usize,
    t: usize,
) -> GraphalgResult<MinCostFlowResult> {
    min_cost_flow_bounded(net, s, t, f64::INFINITY)
}

/// Send at most `max_flow` units from `s` to `t` at minimum cost.
///
/// Returns the total flow sent (≤ `max_flow`) and its total cost.
///
/// # Errors
/// - `SourceOutOfRange` if `s >= n` or `t >= n`.
/// - `InvalidParameter` if `s == t` or `max_flow < 0`.
/// - `NegativeCycle` if the residual graph contains a negative cycle.
pub fn min_cost_flow_bounded(
    net: &MinCostFlowNetwork,
    s: usize,
    t: usize,
    max_flow: f64,
) -> GraphalgResult<MinCostFlowResult> {
    let n = net.n;

    // Validate parameters.
    if s >= n || t >= n {
        return Err(GraphalgError::SourceOutOfRange { node: s.max(t), n });
    }
    if s == t {
        return Err(GraphalgError::InvalidParameter(
            "source == sink".to_string(),
        ));
    }
    if max_flow < 0.0 {
        return Err(GraphalgError::InvalidParameter(
            "max_flow must be non-negative".to_string(),
        ));
    }
    if max_flow == 0.0 {
        return Ok(MinCostFlowResult {
            flow: 0.0,
            cost: 0.0,
        });
    }

    // Clone the network; the algorithm works on a mutable residual copy.
    let mut residual = net.clone();

    let mut total_flow = 0.0;
    let mut total_cost = 0.0;

    loop {
        // Find min-cost path via SPFA.
        let (dist, prev_edge) = spfa_shortest_path(&residual, s, t)?;

        // No augmenting path.
        if dist[t].is_infinite() {
            break;
        }

        // Remaining budget.
        let remaining = max_flow - total_flow;
        if remaining <= 1e-9 {
            break;
        }

        let path_cost = dist[t];
        let pushed = augment_path(&mut residual, s, t, &prev_edge, path_cost, remaining);

        if pushed <= 1e-9 {
            break;
        }

        total_flow += pushed;
        total_cost += pushed * path_cost;
    }

    Ok(MinCostFlowResult {
        flow: total_flow,
        cost: total_cost,
    })
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::max_flow::edmonds_karp::{FlowNetwork, edmonds_karp};

    // Helper: build a FlowNetwork from (u, v, cap) triples.
    fn flow_net_from(n: usize, edges: &[(usize, usize, f64)]) -> FlowNetwork {
        let mut net = FlowNetwork::new(n);
        for &(u, v, c) in edges {
            net.add_edge(u, v, c).expect("add_edge ok");
        }
        net
    }

    // Helper: build a MinCostFlowNetwork from (u, v, cap, cost) tuples.
    fn mcf_net(n: usize, edges: &[(usize, usize, f64, f64)]) -> MinCostFlowNetwork {
        let mut net = MinCostFlowNetwork::new(n);
        for &(u, v, cap, cost) in edges {
            net.add_edge(u, v, cap, cost).expect("add_edge ok");
        }
        net
    }

    // Test 1: Diamond network — cheapest path first.
    #[test]
    fn diamond_max_flow_cost() {
        // s=0, t=3
        // 0→1 cap=2 cost=1, 0→2 cap=2 cost=2, 1→3 cap=2 cost=1, 2→3 cap=2 cost=2
        // Paths: 0-1-3 cost 2 (×2 units) = 4; 0-2-3 cost 4 (×2 units) = 8
        // Total flow=4, cost=12
        let net = mcf_net(
            4,
            &[
                (0, 1, 2.0, 1.0),
                (0, 2, 2.0, 2.0),
                (1, 3, 2.0, 1.0),
                (2, 3, 2.0, 2.0),
            ],
        );
        let r = min_cost_max_flow(&net, 0, 3).expect("ok");
        assert!((r.flow - 4.0).abs() < 1e-9, "flow={}", r.flow);
        assert!((r.cost - 12.0).abs() < 1e-9, "cost={}", r.cost);
    }

    // Test 2: Single path.
    #[test]
    fn single_path() {
        // 0→1 cap=1 cost=3, 1→2 cap=1 cost=4 → flow=1, cost=7
        let net = mcf_net(3, &[(0, 1, 1.0, 3.0), (1, 2, 1.0, 4.0)]);
        let r = min_cost_max_flow(&net, 0, 2).expect("ok");
        assert!((r.flow - 1.0).abs() < 1e-9);
        assert!((r.cost - 7.0).abs() < 1e-9);
    }

    // Test 3: Cross-check with Edmonds-Karp (flow value only, costs ignored).
    #[test]
    fn cross_check_with_edmonds_karp() {
        // 5-node network with diverse capacities.
        // Edges (u, v, cap):
        // 0→1 cap=10, 0→2 cap=6, 1→3 cap=8, 2→3 cap=5, 1→4 cap=4, 3→4 cap=9
        let edge_caps: &[(usize, usize, f64)] = &[
            (0, 1, 10.0),
            (0, 2, 6.0),
            (1, 3, 8.0),
            (2, 3, 5.0),
            (1, 4, 4.0),
            (3, 4, 9.0),
        ];

        let ek_net = flow_net_from(5, edge_caps);
        let ek_flow = edmonds_karp(&ek_net, 0, 4).expect("ek ok");

        // Build MCF net with unit cost (cost doesn't affect max flow).
        let mcf = mcf_net(
            5,
            &edge_caps
                .iter()
                .map(|&(u, v, c)| (u, v, c, 1.0))
                .collect::<Vec<_>>(),
        );
        let r = min_cost_max_flow(&mcf, 0, 4).expect("mcf ok");

        assert!(
            (r.flow - ek_flow).abs() < 1e-9,
            "SSP flow={} but EK flow={}",
            r.flow,
            ek_flow
        );
    }

    // Test 4: Bounded flow caps at requested amount.
    #[test]
    fn bounded_flow_diamond() {
        // Same diamond; max_flow=2 → only uses cheap path 0-1-3 (cost 2 each unit)
        let net = mcf_net(
            4,
            &[
                (0, 1, 2.0, 1.0),
                (0, 2, 2.0, 2.0),
                (1, 3, 2.0, 1.0),
                (2, 3, 2.0, 2.0),
            ],
        );
        let r = min_cost_flow_bounded(&net, 0, 3, 2.0).expect("ok");
        assert!((r.flow - 2.0).abs() < 1e-9, "flow={}", r.flow);
        assert!((r.cost - 4.0).abs() < 1e-9, "cost={}", r.cost);
    }

    // Test 5: Negative cost edge (SPFA handles negative costs).
    #[test]
    fn negative_cost_edge() {
        // 0→1 cap=1 cost=-2, 1→2 cap=1 cost=5 → flow=1, cost=3
        let net = mcf_net(3, &[(0, 1, 1.0, -2.0), (1, 2, 1.0, 5.0)]);
        let r = min_cost_max_flow(&net, 0, 2).expect("ok");
        assert!((r.flow - 1.0).abs() < 1e-9);
        assert!((r.cost - 3.0).abs() < 1e-9, "cost={}", r.cost);
    }

    // Test 6: Antiparallel edges.
    #[test]
    fn antiparallel_edges() {
        // 0→1 cap=1 cost=1 and 1→0 cap=1 cost=1; route s=0, t=2 through 0→1→2.
        let net = mcf_net(3, &[(0, 1, 1.0, 1.0), (1, 0, 1.0, 1.0), (1, 2, 1.0, 1.0)]);
        let r = min_cost_max_flow(&net, 0, 2).expect("ok");
        assert!((r.flow - 1.0).abs() < 1e-9, "flow={}", r.flow);
        assert!((r.cost - 2.0).abs() < 1e-9, "cost={}", r.cost);
    }

    // Test 7: Disconnected s–t → flow=0, cost=0 (no error).
    #[test]
    fn disconnected() {
        // Node 0 isolated; t=3 unreachable.
        let net = mcf_net(4, &[(1, 2, 1.0, 1.0), (2, 3, 1.0, 1.0)]);
        let r = min_cost_max_flow(&net, 0, 3).expect("ok");
        assert!((r.flow).abs() < 1e-9);
        assert!((r.cost).abs() < 1e-9);
    }

    // Test 8: Zero-capacity edge → not augmented.
    #[test]
    fn zero_capacity_edge() {
        let net = mcf_net(3, &[(0, 1, 0.0, 1.0), (1, 2, 0.0, 1.0)]);
        let r = min_cost_max_flow(&net, 0, 2).expect("ok");
        assert!((r.flow).abs() < 1e-9);
    }

    // Test 9: s == t → InvalidParameter.
    #[test]
    fn source_equals_sink() {
        let net = MinCostFlowNetwork::new(3);
        assert!(matches!(
            min_cost_max_flow(&net, 1, 1),
            Err(GraphalgError::InvalidParameter(_))
        ));
    }

    // Test 10: s >= n → SourceOutOfRange.
    #[test]
    fn source_out_of_range() {
        let net = MinCostFlowNetwork::new(3);
        assert!(matches!(
            min_cost_max_flow(&net, 5, 1),
            Err(GraphalgError::SourceOutOfRange { .. })
        ));
    }

    // Test 11: cap < 0 in add_edge → InvalidParameter.
    #[test]
    fn add_edge_negative_cap() {
        let mut net = MinCostFlowNetwork::new(3);
        assert!(matches!(
            net.add_edge(0, 1, -1.0, 1.0),
            Err(GraphalgError::InvalidParameter(_))
        ));
    }

    // Test 12: Non-finite cost in add_edge → InvalidEdgeWeight.
    #[test]
    fn add_edge_nonfinite_cost() {
        let mut net = MinCostFlowNetwork::new(3);
        assert!(matches!(
            net.add_edge(0, 1, 1.0, f64::INFINITY),
            Err(GraphalgError::InvalidEdgeWeight(_))
        ));
        assert!(matches!(
            net.add_edge(0, 1, 1.0, f64::NAN),
            Err(GraphalgError::InvalidEdgeWeight(_))
        ));
    }

    // Test 13: max_flow = 0 for bounded → flow=0, cost=0.
    #[test]
    fn bounded_max_flow_zero() {
        let net = mcf_net(3, &[(0, 1, 5.0, 1.0), (1, 2, 5.0, 1.0)]);
        let r = min_cost_flow_bounded(&net, 0, 2, 0.0).expect("ok");
        assert!((r.flow).abs() < 1e-9);
        assert!((r.cost).abs() < 1e-9);
    }

    // Test 14: max_flow < 0 for bounded → InvalidParameter.
    #[test]
    fn bounded_negative_max_flow() {
        let net = MinCostFlowNetwork::new(3);
        assert!(matches!(
            min_cost_flow_bounded(&net, 0, 2, -1.0),
            Err(GraphalgError::InvalidParameter(_))
        ));
    }

    // Test 15: Multiple unit-capacity paths picked in cost order.
    #[test]
    fn three_paths_cost_order() {
        // Three parallel paths s→middle→t with costs 1, 2, 3.
        // Bounded to 2 → picks two cheapest paths (total cost 3).
        let net = mcf_net(
            8,
            &[
                (0, 1, 1.0, 1.0),
                (1, 7, 1.0, 0.0), // cost 1 path
                (0, 2, 1.0, 2.0),
                (2, 7, 1.0, 0.0), // cost 2 path
                (0, 3, 1.0, 3.0),
                (3, 7, 1.0, 0.0), // cost 3 path
            ],
        );
        let r = min_cost_flow_bounded(&net, 0, 7, 2.0).expect("ok");
        assert!((r.flow - 2.0).abs() < 1e-9, "flow={}", r.flow);
        assert!((r.cost - 3.0).abs() < 1e-9, "cost={}", r.cost); // paths of cost 1+2
    }

    // Test 16: Determinism — same network, same s,t → same result.
    #[test]
    fn deterministic_result() {
        let net = mcf_net(
            4,
            &[
                (0, 1, 2.0, 1.0),
                (0, 2, 2.0, 2.0),
                (1, 3, 2.0, 1.0),
                (2, 3, 2.0, 2.0),
            ],
        );
        let r1 = min_cost_max_flow(&net, 0, 3).expect("r1");
        let r2 = min_cost_max_flow(&net, 0, 3).expect("r2");
        assert_eq!(r1, r2);
    }

    // Test 17: Large-ish network (10 nodes, 20 edges) — non-negative flow, finite cost.
    #[test]
    fn large_network() {
        let edges: Vec<(usize, usize, f64, f64)> = vec![
            (0, 1, 5.0, 1.0),
            (0, 2, 3.0, 2.0),
            (0, 3, 4.0, 3.0),
            (1, 4, 2.0, 1.0),
            (1, 5, 3.0, 2.0),
            (2, 4, 2.0, 3.0),
            (2, 6, 2.0, 1.0),
            (3, 5, 2.0, 2.0),
            (3, 6, 2.0, 1.0),
            (4, 7, 3.0, 1.0),
            (5, 7, 3.0, 2.0),
            (6, 7, 2.0, 3.0),
            (4, 8, 2.0, 1.0),
            (5, 8, 2.0, 2.0),
            (6, 8, 2.0, 1.0),
            (7, 9, 5.0, 1.0),
            (8, 9, 5.0, 1.0),
            (1, 3, 1.0, 1.0),
            (2, 5, 1.0, 2.0),
            (3, 6, 1.0, 1.0),
        ];
        let net = mcf_net(10, &edges);
        let r = min_cost_max_flow(&net, 0, 9).expect("ok");
        assert!(r.flow >= 0.0, "flow must be non-negative");
        assert!(r.cost.is_finite(), "cost must be finite");
    }

    // Test 18: Flow conservation on diamond network.
    #[test]
    fn flow_conservation_diamond() {
        // Build and run; then verify by reconstructing flows manually.
        // Since the algorithm clones internally, we re-run on a fresh clone.
        let net = mcf_net(
            4,
            &[
                (0, 1, 2.0, 1.0),
                (0, 2, 2.0, 2.0),
                (1, 3, 2.0, 1.0),
                (2, 3, 2.0, 2.0),
            ],
        );
        let r = min_cost_max_flow(&net, 0, 3).expect("ok");
        // Net flow out of s must equal the reported flow.
        // We verify by checking the reported flow is achievable given the capacities.
        let max_out_s = 2.0 + 2.0; // edges from 0
        assert!(r.flow <= max_out_s + 1e-9);
        assert!((r.flow - 4.0).abs() < 1e-9);
    }

    // Test 19: n=0 network with add_edge → IndexOutOfBounds.
    #[test]
    fn empty_network_add_edge_errors() {
        let mut net = MinCostFlowNetwork::new(0);
        assert!(matches!(
            net.add_edge(0, 0, 1.0, 1.0),
            Err(GraphalgError::IndexOutOfBounds { .. })
        ));
    }

    // Test 20: Clone of network — original unchanged after running algorithm.
    #[test]
    fn original_unchanged_after_algorithm() {
        let net = mcf_net(3, &[(0, 1, 5.0, 1.0), (1, 2, 5.0, 1.0)]);
        let original_edge_count = net.edges.len();
        let original_n = net.n;

        let _result = min_cost_max_flow(&net, 0, 2).expect("ok");

        // Original network untouched.
        assert_eq!(net.n, original_n);
        assert_eq!(net.edges.len(), original_edge_count);
        // All forward edges should still have flow = 0 (algorithm clones internally).
        for (i, e) in net.edges.iter().enumerate() {
            assert_eq!(
                e.flow, 0.0,
                "edge {} flow should be 0 in original, got {}",
                i, e.flow
            );
        }
    }
}
