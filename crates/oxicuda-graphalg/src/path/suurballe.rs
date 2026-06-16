//! Suurballe's algorithm for a pair of vertex-disjoint shortest paths.
//!
//! # Overview
//!
//! Given a directed graph with non-negative edge costs, a source `s`, and a target `t`,
//! Suurballe's algorithm (Suurballe 1974; Suurballe & Tarjan 1984) finds **two paths
//! from `s` to `t` that share no vertex other than `s` and `t`** and whose **total cost
//! is minimum** over all such pairs.
//!
//! ## Method
//!
//! 1. Run **Dijkstra** from `s`, obtaining shortest-path distances `d[v]` and the first
//!    shortest path `P₁`.
//! 2. Re-cost every edge with the **reduced cost** `w'(u→v) = w(u→v) + d[u] − d[v]`. By
//!    the triangle inequality `d[v] ≤ d[u] + w(u→v)`, every reduced cost is ≥ 0, so a
//!    second Dijkstra is valid. Reduced costs are **0 exactly on shortest-path-tree edges**.
//! 3. In the residual graph, **reverse each edge of `P₁`** (its reduced cost is 0, so the
//!    reversed arc also has cost 0). This lets the second path "cancel" against the first.
//! 4. To force **vertex** disjointness (not just edge disjointness), every interior vertex
//!    `v` is **split** into `vᵢₙ → vₒᵤₜ` with unit capacity; the reversal of `P₁` also
//!    reverses these split arcs, so the second search may reuse an interior vertex of `P₁`
//!    only by travelling *backwards* through it — which, after recombination, yields two
//!    genuinely vertex-disjoint paths.
//! 5. Run **Dijkstra** again from `s` to `t` on the reduced-cost residual graph to get the
//!    augmenting walk `P₂`.
//! 6. **Recombine**: superimpose `P₁` and `P₂`, cancel every edge traversed once forward
//!    and once backward, and decompose the remaining arcs into two vertex-disjoint `s`→`t`
//!    paths.
//!
//! The total cost of the recovered pair equals `2·d[t] +` (reduced length of `P₂`) `= `
//! the cost of a minimum-cost flow of value 2 from `s` to `t` under unit vertex capacities.
//!
//! This module reuses the crate's [`dijkstra`](mod@crate::shortest_path::dijkstra) building
//! block in spirit; for the reduced-cost residual searches it runs a dedicated binary-heap
//! Dijkstra over the **split** node space, which `WeightedGraph` does not natively model.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

/// A pair of vertex-disjoint shortest paths from `s` to `t`.
#[derive(Debug, Clone)]
pub struct DisjointPaths {
    /// The first path as a vertex sequence `[s, …, t]`.
    pub path_a: Vec<usize>,
    /// The second path as a vertex sequence `[s, …, t]`, vertex-disjoint from `path_a`
    /// apart from the shared endpoints `s` and `t`.
    pub path_b: Vec<usize>,
    /// Combined cost of both paths (sum of all their edge weights).
    pub total_cost: f64,
}

// ── Split-graph node numbering ───────────────────────────────────────────────
//
// Interior vertex v (v ≠ s, v ≠ t) becomes two split nodes:
//   in(v)  = 2*v
//   out(v) = 2*v + 1
// connected by a unit-capacity arc in(v) → out(v) of cost 0.
//
// Source and target are *not* split (they may be shared): we use a single node each,
//   node(s) = 2*s + 1   (acts purely as an "out")
//   node(t) = 2*t       (acts purely as an "in")
// so that an external edge x→s would arrive at in(s) and x→t arrives at in(t), while
// s→y leaves from out(s). Treating s as out-only and t as in-only matches the fact that
// no path needs to *pass through* s or t internally.

#[inline]
fn node_in(v: usize) -> usize {
    2 * v
}
#[inline]
fn node_out(v: usize) -> usize {
    2 * v + 1
}

#[derive(Debug, Clone, Copy)]
struct ResEdge {
    to: usize,
    rev: usize,
    cap: i64,
    cost: f64,
}

struct Residual {
    adj: Vec<Vec<usize>>,
    edges: Vec<ResEdge>,
}

impl Residual {
    fn new(num_nodes: usize) -> Self {
        Self {
            adj: vec![Vec::new(); num_nodes],
            edges: Vec::new(),
        }
    }

    /// Add a directed arc `u→v` with `cap` units at `cost`, and its zero-capacity reverse.
    fn add(&mut self, u: usize, v: usize, cap: i64, cost: f64) {
        let a = self.edges.len();
        let b = a + 1;
        self.edges.push(ResEdge {
            to: v,
            rev: b,
            cap,
            cost,
        });
        self.edges.push(ResEdge {
            to: u,
            rev: a,
            cap: 0,
            cost: -cost,
        });
        self.adj[u].push(a);
        self.adj[v].push(b);
    }
}

// ── Dijkstra heap item ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    dist: f64,
    node: usize,
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// Plain Dijkstra over the residual graph using *raw* (already-reduced) edge costs,
/// only traversing arcs with positive residual capacity. Returns `(dist, prev_edge)`.
fn dijkstra_residual(res: &Residual, src: usize, num_nodes: usize) -> (Vec<f64>, Vec<usize>) {
    let mut dist = vec![f64::INFINITY; num_nodes];
    let mut prev_edge = vec![usize::MAX; num_nodes];
    dist[src] = 0.0;
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    heap.push(HeapItem {
        dist: 0.0,
        node: src,
    });
    while let Some(HeapItem { dist: d, node: u }) = heap.pop() {
        if d > dist[u] + 1e-12 {
            continue;
        }
        for &eid in &res.adj[u] {
            let e = res.edges[eid];
            if e.cap <= 0 {
                continue;
            }
            // Reduced costs are clamped at 0 to absorb floating-point noise.
            let step = if e.cost < 0.0 { 0.0 } else { e.cost };
            let nd = d + step;
            if nd + 1e-12 < dist[e.to] {
                dist[e.to] = nd;
                prev_edge[e.to] = eid;
                heap.push(HeapItem {
                    dist: nd,
                    node: e.to,
                });
            }
        }
    }
    (dist, prev_edge)
}

/// Find two vertex-disjoint shortest `s`→`t` paths of minimum total cost.
///
/// `graph` must have non-negative edge weights. On success the two returned paths share
/// only `s` and `t`. If fewer than two vertex-disjoint paths exist (e.g. `t` is reachable
/// only through a cut vertex / bridge), a [`GraphalgError::NoSolution`] is returned.
///
/// # Errors
/// - [`GraphalgError::SourceOutOfRange`] if `s` or `t` is out of range.
/// - [`GraphalgError::InvalidParameter`] if `s == t`.
/// - [`GraphalgError::NegativeWeight`] if any edge weight is negative.
/// - [`GraphalgError::NoSolution`] if no two vertex-disjoint `s`→`t` paths exist.
pub fn suurballe_vertex_disjoint(
    graph: &WeightedGraph,
    s: usize,
    t: usize,
) -> GraphalgResult<DisjointPaths> {
    let n = graph.n;
    if s >= n || t >= n {
        return Err(GraphalgError::SourceOutOfRange { node: s.max(t), n });
    }
    if s == t {
        return Err(GraphalgError::InvalidParameter(
            "source must differ from target".to_string(),
        ));
    }
    // Validate weights up front.
    for u in 0..n {
        for &(v, w) in graph.neighbors(u)? {
            if w < 0.0 {
                return Err(GraphalgError::NegativeWeight {
                    edge: (u, v),
                    weight: w,
                });
            }
        }
    }

    // Step 1 — Dijkstra from s on the *original* graph for potentials d[·].
    let d = dijkstra_potentials(graph, s)?;
    if d[t].is_infinite() {
        return Err(GraphalgError::NoSolution(
            "target unreachable from source".to_string(),
        ));
    }

    // Step 2/3/4 — build the reduced-cost split residual graph.
    let num_nodes = 2 * n;
    let mut res = Residual::new(num_nodes);

    // Interior split arcs in(v)→out(v), unit capacity, cost 0. (Skip s and t.)
    for v in 0..n {
        if v == s || v == t {
            continue;
        }
        res.add(node_in(v), node_out(v), 1, 0.0);
    }

    // Original edges, lifted to the split graph with reduced cost.
    // An edge u→v leaves u's "out" and enters v's "in".
    for u in 0..n {
        for &(v, w) in graph.neighbors(u)? {
            if u == v {
                continue; // self loops are irrelevant
            }
            // Skip edges that cannot lie on any shortest-structure path: those whose tail
            // is unreachable. Their reduced cost would be undefined (∞ potential).
            if d[u].is_infinite() {
                continue;
            }
            // Edges pointing *into* s or *out of* t cannot help a forward s→t path and would
            // corrupt the single-node treatment of s/t; drop them.
            if v == s || u == t {
                continue;
            }
            let from = node_out(u);
            let to = node_in(v);
            // Reduced cost; clamp tiny negatives to 0.
            let mut rc = w + d[u] - d[v];
            if rc < 0.0 {
                rc = 0.0;
            }
            res.add(from, to, 1, rc);
        }
    }

    let src = node_out(s);
    let dst = node_in(t);

    // Step 4a — first augmenting path (this is the shortest path itself, reduced length 0).
    let (_, prev1) = dijkstra_residual(&res, src, num_nodes);
    if prev1[dst] == usize::MAX {
        return Err(GraphalgError::NoSolution(
            "target unreachable in residual graph".to_string(),
        ));
    }
    augment(&mut res, src, dst, &prev1);

    // Step 4b — second augmenting path.
    let (_, prev2) = dijkstra_residual(&res, src, num_nodes);
    if prev2[dst] == usize::MAX {
        return Err(GraphalgError::NoSolution(
            "no second vertex-disjoint path exists".to_string(),
        ));
    }
    augment(&mut res, src, dst, &prev2);

    // Step 6 — decompose the flow (two units, s→t) into two vertex-disjoint paths.
    let (path_a, path_b) = decompose_two_paths(&mut res, s, t, n)?;

    // Total cost = sum of the *original* edge weights actually used.
    let total_cost = path_cost(graph, &path_a)? + path_cost(graph, &path_b)?;

    Ok(DisjointPaths {
        path_a,
        path_b,
        total_cost,
    })
}

/// Dijkstra purely for vertex potentials `d[·]` from `src` on the original graph.
fn dijkstra_potentials(graph: &WeightedGraph, src: usize) -> GraphalgResult<Vec<f64>> {
    let n = graph.n;
    let mut dist = vec![f64::INFINITY; n];
    dist[src] = 0.0;
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    heap.push(HeapItem {
        dist: 0.0,
        node: src,
    });
    while let Some(HeapItem { dist: dd, node: u }) = heap.pop() {
        if dd > dist[u] + 1e-12 {
            continue;
        }
        for &(v, w) in graph.neighbors(u)? {
            let nd = dd + w;
            if nd + 1e-12 < dist[v] {
                dist[v] = nd;
                heap.push(HeapItem { dist: nd, node: v });
            }
        }
    }
    Ok(dist)
}

/// Push one unit of flow along the `prev_edge` chain from `src` to `dst`.
fn augment(res: &mut Residual, src: usize, dst: usize, prev_edge: &[usize]) {
    let mut v = dst;
    while v != src {
        let eid = prev_edge[v];
        res.edges[eid].cap -= 1;
        let rev = res.edges[eid].rev;
        res.edges[rev].cap += 1;
        v = res.edges[rev].to;
    }
}

/// Is edge `eid` a *forward* arc carrying net flow?
///
/// Forward arcs are the even-indexed ones (each `add` pushes forward then reverse). A unit
/// forward arc carries net flow exactly when its reverse arc currently has positive residual
/// capacity (i.e. the unit it received from augmentation has not been cancelled back).
fn carries_flow(edges: &[ResEdge], eid: usize) -> bool {
    if eid % 2 != 0 {
        return false;
    }
    let rev = edges[eid].rev;
    edges[rev].cap > 0
}

/// Decompose the unit-flow residual into two vertex-disjoint `s`→`t` paths.
///
/// After two augmentations, the *used* forward arcs (those whose residual capacity dropped
/// to 0) carry exactly the flow. We follow them from `s`, peeling off one path at a time.
fn decompose_two_paths(
    res: &mut Residual,
    s: usize,
    t: usize,
    n: usize,
) -> GraphalgResult<(Vec<usize>, Vec<usize>)> {
    // Build per-node lists of *outgoing used* original arcs in the split graph.
    // An original edge u→v lives as out(u) → in(v); the split arc in(v)→out(v) is internal.
    // A forward arc carried flow iff its current cap is below its initial cap, i.e. the
    // reverse arc now has positive cap. We detect "used forward arc" as: the edge has
    // even index (forward arcs were always added first) and its cap decreased.
    //
    // To recover original-vertex paths we walk the split graph but only emit a vertex when
    // we traverse an out(u)→in(v) arc.
    let num_nodes = 2 * n;
    // next[node] = index into adj to resume from when peeling paths.
    let mut used_next: Vec<usize> = vec![0; num_nodes];

    let src = node_out(s);
    let dst = node_in(t);

    let mut paths: Vec<Vec<usize>> = Vec::new();

    for _ in 0..2 {
        let mut path_vertices: Vec<usize> = vec![s];
        let mut cur = src;
        let mut guard = 0usize;
        let limit = num_nodes * 4 + 8;
        loop {
            guard += 1;
            if guard > limit {
                return Err(GraphalgError::NoSolution(
                    "path decomposition did not terminate".to_string(),
                ));
            }
            if cur == dst {
                break;
            }
            // Find the next unused flow-carrying arc out of `cur`.
            let mut advanced = false;
            while used_next[cur] < res.adj[cur].len() {
                let eid = res.adj[cur][used_next[cur]];
                used_next[cur] += 1;
                if carries_flow(&res.edges, eid) {
                    // Consume one unit so we never reuse this arc in the other path.
                    let rev = res.edges[eid].rev;
                    res.edges[rev].cap -= 1;
                    let to = res.edges[eid].to;
                    // Emit a vertex only for cross arcs out(u)→in(v).
                    if cur % 2 == 1 && to % 2 == 0 {
                        let v = to / 2;
                        path_vertices.push(v);
                    }
                    cur = to;
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                return Err(GraphalgError::NoSolution(
                    "incomplete vertex-disjoint path pair".to_string(),
                ));
            }
        }
        paths.push(path_vertices);
    }

    let path_a = paths.remove(0);
    let path_b = paths.remove(0);

    // Sanity: both must start at s and end at t.
    if path_a.first() != Some(&s)
        || path_a.last() != Some(&t)
        || path_b.first() != Some(&s)
        || path_b.last() != Some(&t)
    {
        return Err(GraphalgError::NoSolution(
            "recovered paths are malformed".to_string(),
        ));
    }
    Ok((path_a, path_b))
}

/// Sum the original edge weights along a vertex path.
fn path_cost(graph: &WeightedGraph, path: &[usize]) -> GraphalgResult<f64> {
    let mut total = 0.0;
    for w in path.windows(2) {
        let (u, v) = (w[0], w[1]);
        let mut best: Option<f64> = None;
        for &(nb, weight) in graph.neighbors(u)? {
            if nb == v {
                best = Some(best.map_or(weight, |b: f64| b.min(weight)));
            }
        }
        match best {
            Some(c) => total += c,
            None => {
                return Err(GraphalgError::NoSolution(format!(
                    "reconstructed edge ({u},{v}) absent from graph"
                )));
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::min_cost_flow::successive_shortest_paths::{
        MinCostFlowNetwork, min_cost_flow_bounded,
    };

    fn wgraph(n: usize, edges: &[(usize, usize, f64)]) -> WeightedGraph {
        let mut g = WeightedGraph::new(n);
        for &(u, v, w) in edges {
            g.add_edge(u, v, w).expect("add ok");
        }
        g
    }

    /// Min-cost-flow oracle for *vertex*-disjoint pair cost: split interior vertices with
    /// unit capacity and push exactly 2 units s→t at minimum cost.
    fn mcf_vertex_disjoint_cost(
        n: usize,
        edges: &[(usize, usize, f64)],
        s: usize,
        t: usize,
    ) -> Option<f64> {
        // Node numbering identical to the algorithm's split convention.
        let mut net = MinCostFlowNetwork::new(2 * n);
        for v in 0..n {
            if v == s || v == t {
                continue;
            }
            net.add_edge(2 * v, 2 * v + 1, 1.0, 0.0).expect("ok");
        }
        for &(u, v, w) in edges {
            if u == v || v == s || u == t {
                continue;
            }
            net.add_edge(2 * u + 1, 2 * v, 1.0, w).expect("ok");
        }
        let src = 2 * s + 1;
        let dst = 2 * t;
        let r = min_cost_flow_bounded(&net, src, dst, 2.0).expect("mcf ok");
        if (r.flow - 2.0).abs() < 1e-9 {
            Some(r.cost)
        } else {
            None
        }
    }

    fn assert_vertex_disjoint(dp: &DisjointPaths, s: usize, t: usize) {
        use std::collections::HashSet;
        let interior_a: HashSet<usize> = dp
            .path_a
            .iter()
            .copied()
            .filter(|&v| v != s && v != t)
            .collect();
        let interior_b: HashSet<usize> = dp
            .path_b
            .iter()
            .copied()
            .filter(|&v| v != s && v != t)
            .collect();
        assert!(
            interior_a.is_disjoint(&interior_b),
            "paths share an interior vertex: {:?} vs {:?}",
            dp.path_a,
            dp.path_b
        );
        // Each path is itself simple.
        let mut seen_a = HashSet::new();
        for &v in &dp.path_a {
            assert!(seen_a.insert(v), "path_a revisits {v}");
        }
        let mut seen_b = HashSet::new();
        for &v in &dp.path_b {
            assert!(seen_b.insert(v), "path_b revisits {v}");
        }
        assert_eq!(dp.path_a.first(), Some(&s));
        assert_eq!(dp.path_a.last(), Some(&t));
        assert_eq!(dp.path_b.first(), Some(&s));
        assert_eq!(dp.path_b.last(), Some(&t));
    }

    #[test]
    fn two_parallel_paths_diamond() {
        // s=0, t=3, two disjoint 2-hop routes via 1 and via 2.
        let edges = [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 1.0)];
        let g = wgraph(4, &edges);
        let dp = suurballe_vertex_disjoint(&g, 0, 3).expect("ok");
        assert_vertex_disjoint(&dp, 0, 3);
        assert!((dp.total_cost - 4.0).abs() < 1e-9, "cost={}", dp.total_cost);
    }

    #[test]
    fn min_total_cost_matches_mcf_oracle() {
        // Asymmetric weights so the "two shortest" greedily is wrong; Suurballe must pick
        // the globally cheapest *pair*.
        let edges = [
            (0, 1, 1.0),
            (1, 4, 1.0),
            (0, 2, 2.0),
            (2, 4, 2.0),
            (0, 3, 3.0),
            (3, 4, 3.0),
            (1, 2, 1.0),
        ];
        let g = wgraph(5, &edges);
        let dp = suurballe_vertex_disjoint(&g, 0, 4).expect("ok");
        assert_vertex_disjoint(&dp, 0, 4);
        let oracle = mcf_vertex_disjoint_cost(5, &edges, 0, 4).expect("oracle has 2 paths");
        assert!(
            (dp.total_cost - oracle).abs() < 1e-6,
            "suurballe={} oracle={}",
            dp.total_cost,
            oracle
        );
    }

    #[test]
    fn matches_oracle_on_grid() {
        // 3x3-ish DAG grid; multiple disjoint routes with varied weights.
        let edges = [
            (0, 1, 2.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (1, 4, 3.0),
            (2, 4, 1.0),
            (2, 5, 2.0),
            (3, 6, 2.0),
            (4, 6, 1.0),
            (4, 7, 2.0),
            (5, 7, 1.0),
            (6, 8, 1.0),
            (7, 8, 2.0),
        ];
        let n = 9;
        let g = wgraph(n, &edges);
        let dp = suurballe_vertex_disjoint(&g, 0, 8).expect("ok");
        assert_vertex_disjoint(&dp, 0, 8);
        let oracle = mcf_vertex_disjoint_cost(n, &edges, 0, 8).expect("oracle");
        assert!(
            (dp.total_cost - oracle).abs() < 1e-6,
            "suurballe={} oracle={}",
            dp.total_cost,
            oracle
        );
    }

    #[test]
    fn fails_when_only_a_bridge_connects() {
        // s → a → t with a a cut-vertex: only ONE vertex-disjoint path exists.
        let edges = [(0, 1, 1.0), (1, 2, 1.0)];
        let g = wgraph(3, &edges);
        assert!(matches!(
            suurballe_vertex_disjoint(&g, 0, 2),
            Err(GraphalgError::NoSolution(_))
        ));
    }

    #[test]
    fn fails_when_target_unreachable() {
        let edges = [(0, 1, 1.0)];
        let g = wgraph(3, &edges); // node 2 isolated
        assert!(matches!(
            suurballe_vertex_disjoint(&g, 0, 2),
            Err(GraphalgError::NoSolution(_))
        ));
    }

    #[test]
    fn fails_with_single_direct_edge_only() {
        // Only the direct edge s→t exists; no second disjoint path.
        let edges = [(0, 1, 5.0)];
        let g = wgraph(2, &edges);
        assert!(matches!(
            suurballe_vertex_disjoint(&g, 0, 1),
            Err(GraphalgError::NoSolution(_))
        ));
    }

    #[test]
    fn two_disjoint_with_a_shared_cut_attempt() {
        // Build a graph where a naive "remove first shortest path then re-search" would
        // fail but Suurballe's reversal succeeds. Classic trap graph.
        //   0->1 (1), 1->2 (1), 2->5 (1)        [shortest path 0-1-2-5 cost 3]
        //   0->3 (2), 3->2 (1), 1->4 (2), 4->5 (1)
        let edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 5, 1.0),
            (0, 3, 2.0),
            (3, 2, 1.0),
            (1, 4, 2.0),
            (4, 5, 1.0),
        ];
        let n = 6;
        let g = wgraph(n, &edges);
        let dp = suurballe_vertex_disjoint(&g, 0, 5).expect("ok");
        assert_vertex_disjoint(&dp, 0, 5);
        let oracle = mcf_vertex_disjoint_cost(n, &edges, 0, 5).expect("oracle");
        assert!(
            (dp.total_cost - oracle).abs() < 1e-6,
            "suurballe={} oracle={}",
            dp.total_cost,
            oracle
        );
    }

    #[test]
    fn reduced_costs_are_nonnegative() {
        // Directly assert the reduced-cost transformation property used in step 2.
        let edges = [
            (0, 1, 4.0),
            (0, 2, 1.0),
            (2, 1, 1.0),
            (1, 3, 1.0),
            (2, 3, 5.0),
        ];
        let g = wgraph(4, &edges);
        let d = dijkstra_potentials(&g, 0).expect("ok");
        for u in 0..g.n {
            if d[u].is_infinite() {
                continue;
            }
            for &(v, w) in g.neighbors(u).expect("nb") {
                if d[v].is_infinite() {
                    continue;
                }
                let rc = w + d[u] - d[v];
                assert!(rc >= -1e-9, "reduced cost {rc} negative on {u}->{v}");
            }
        }
    }

    #[test]
    fn rejects_negative_weight() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, -2.0).expect("add");
        g.add_edge(1, 2, 1.0).expect("add");
        assert!(matches!(
            suurballe_vertex_disjoint(&g, 0, 2),
            Err(GraphalgError::NegativeWeight { .. })
        ));
    }

    #[test]
    fn rejects_source_equals_target() {
        let g = wgraph(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        assert!(matches!(
            suurballe_vertex_disjoint(&g, 1, 1),
            Err(GraphalgError::InvalidParameter(_))
        ));
    }

    #[test]
    fn rejects_out_of_range() {
        let g = wgraph(3, &[(0, 1, 1.0)]);
        assert!(matches!(
            suurballe_vertex_disjoint(&g, 0, 9),
            Err(GraphalgError::SourceOutOfRange { .. })
        ));
    }

    #[test]
    fn k1_reduces_to_dijkstra_shortest_path() {
        // The first augmenting path before the disjointness reversal equals Dijkstra's
        // shortest path. We verify by checking that the cheaper of the two returned paths
        // has cost equal to the Dijkstra distance to t.
        use crate::shortest_path::dijkstra::dijkstra;
        let edges = [
            (0, 1, 1.0),
            (1, 3, 1.0),
            (0, 2, 5.0),
            (2, 3, 1.0),
            (0, 3, 9.0),
        ];
        let g = wgraph(4, &edges);
        let sp = dijkstra(&g, 0).expect("dij");
        let dp = suurballe_vertex_disjoint(&g, 0, 3).expect("ok");
        let ca = path_cost(&g, &dp.path_a).expect("ca");
        let cb = path_cost(&g, &dp.path_b).expect("cb");
        let cheaper = ca.min(cb);
        assert!(
            (cheaper - sp.dist[3]).abs() < 1e-9,
            "cheaper path {cheaper} != dijkstra {}",
            sp.dist[3]
        );
    }

    #[test]
    fn total_cost_is_sum_of_path_costs() {
        let edges = [(0, 1, 2.0), (1, 3, 3.0), (0, 2, 4.0), (2, 3, 1.0)];
        let g = wgraph(4, &edges);
        let dp = suurballe_vertex_disjoint(&g, 0, 3).expect("ok");
        let ca = path_cost(&g, &dp.path_a).expect("ca");
        let cb = path_cost(&g, &dp.path_b).expect("cb");
        assert!((dp.total_cost - (ca + cb)).abs() < 1e-12);
        assert!(
            (dp.total_cost - 10.0).abs() < 1e-9,
            "cost={}",
            dp.total_cost
        );
    }

    #[test]
    fn three_disjoint_available_picks_cheapest_two() {
        // Three disjoint routes of cost 2, 4, 6; the pair must be 2+4 = 6.
        let edges = [
            (0, 1, 1.0),
            (1, 7, 1.0),
            (0, 2, 2.0),
            (2, 7, 2.0),
            (0, 3, 3.0),
            (3, 7, 3.0),
        ];
        let n = 8;
        let g = wgraph(n, &edges);
        let dp = suurballe_vertex_disjoint(&g, 0, 7).expect("ok");
        assert_vertex_disjoint(&dp, 0, 7);
        assert!((dp.total_cost - 6.0).abs() < 1e-9, "cost={}", dp.total_cost);
        let oracle = mcf_vertex_disjoint_cost(n, &edges, 0, 7).expect("oracle");
        assert!((dp.total_cost - oracle).abs() < 1e-6);
    }
}
