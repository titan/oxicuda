//! Delta-stepping single-source shortest paths (Meyer & Sanders 2003).
//!
//! Delta-stepping is a bucket-based label-correcting SSSP algorithm that bridges
//! Dijkstra (`Δ → 0`, fully label-setting) and Bellman-Ford (`Δ → ∞`, fully
//! label-correcting). It groups tentative distances into buckets of width `Δ`
//! and processes the lowest non-empty bucket at a time, which exposes the
//! parallelism exploited by GPU/cluster implementations while remaining a clean
//! sequential algorithm here.
//!
//! ## Algorithm
//!
//! Edges are classified as **light** (`weight ≤ Δ`) or **heavy** (`weight > Δ`).
//! For the current bucket `B[i]`:
//!
//! 1. Repeatedly relax all *light* edges out of the bucket's vertices,
//!    re-inserting any vertex that lands back in `B[i]` (this settles all
//!    vertices whose final distance lies in `[iΔ, (i+1)Δ)`).
//! 2. Then relax all *heavy* edges out of the vertices that were removed from
//!    `B[i]` during step 1 (these can only target later buckets).
//!
//! A relaxation `relax(v, x)` moves `v` from its old bucket to bucket
//! `⌊x / Δ⌋` whenever `x` improves `tent[v]`.
//!
//! Rejects negative edge weights (buckets assume non-negative distances), like
//! Dijkstra.
//!
//! ## References
//! - Meyer, U. & Sanders, P. (2003). "Δ-stepping: a parallelizable shortest path
//!   algorithm." Journal of Algorithms 49(1), 114–152.

use std::collections::HashSet;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

/// Result of a delta-stepping run from a single source.
#[derive(Debug, Clone)]
pub struct DeltaSteppingOutput {
    /// `dist[v]` = shortest-path distance from the source (∞ if unreachable).
    pub dist: Vec<f64>,
    /// `parent[v]` = predecessor of `v` on a shortest path (`usize::MAX` if none;
    /// the source is its own parent).
    pub parent: Vec<usize>,
}

/// Compute single-source shortest paths via delta-stepping with bucket width `delta`.
///
/// # Errors
/// - [`GraphalgError::SourceOutOfRange`] if `source >= g.n`.
/// - [`GraphalgError::InvalidParameter`] if `delta` is not finite and positive.
/// - [`GraphalgError::NegativeWeight`] if any edge weight is negative.
pub fn delta_stepping(
    g: &WeightedGraph,
    source: usize,
    delta: f64,
) -> GraphalgResult<DeltaSteppingOutput> {
    if source >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source,
            n: g.n,
        });
    }
    if !(delta.is_finite() && delta > 0.0) {
        return Err(GraphalgError::InvalidParameter(format!(
            "delta must be finite and > 0, got {delta}"
        )));
    }
    // Validate non-negativity up front.
    for (u, adj) in g.edges.iter().enumerate() {
        for &(v, w) in adj {
            if w < 0.0 {
                return Err(GraphalgError::NegativeWeight {
                    edge: (u, v),
                    weight: w,
                });
            }
        }
    }

    let n = g.n;
    let mut tent = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];
    // The bucket index each vertex currently sits in (usize::MAX = none).
    let mut bucket_of = vec![usize::MAX; n];
    // Sparse bucket array: index → set of vertices.
    let mut buckets: Vec<HashSet<usize>> = Vec::new();

    // relax(v, x, p): place v into the correct bucket if x improves tent[v].
    // Implemented inline (closures can't easily borrow all of these mutably).
    fn relax(
        v: usize,
        x: f64,
        p: usize,
        delta: f64,
        tent: &mut [f64],
        parent: &mut [usize],
        bucket_of: &mut [usize],
        buckets: &mut Vec<HashSet<usize>>,
    ) {
        if x < tent[v] {
            // Remove from its old bucket, if any.
            let old = bucket_of[v];
            if old != usize::MAX {
                if let Some(b) = buckets.get_mut(old) {
                    b.remove(&v);
                }
            }
            let new_idx = (x / delta).floor() as usize;
            if new_idx >= buckets.len() {
                buckets.resize(new_idx + 1, HashSet::new());
            }
            buckets[new_idx].insert(v);
            bucket_of[v] = new_idx;
            tent[v] = x;
            parent[v] = p;
        }
    }

    // `relax` performs the `x < tent[v]` test, so leave `tent[source]` at ∞ and
    // let the initial relaxation set it to 0 and insert it into bucket 0.
    relax(
        source,
        0.0,
        source,
        delta,
        &mut tent,
        &mut parent,
        &mut bucket_of,
        &mut buckets,
    );

    let mut i = 0usize;
    while i < buckets.len() {
        // Skip empty buckets.
        if buckets[i].is_empty() {
            i += 1;
            continue;
        }

        // `settled` collects every vertex removed from B[i] across the light
        // phase; their heavy edges are relaxed afterwards.
        let mut settled: Vec<usize> = Vec::new();

        // ── Light-edge phase: iterate until B[i] is empty ────────────────────
        while !buckets[i].is_empty() {
            // Snapshot the current contents of B[i] and clear it.
            let current: Vec<usize> = buckets[i].drain().collect();
            for &u in &current {
                // Mark as removed from this bucket for the heavy phase.
                bucket_of[u] = usize::MAX;
                settled.push(u);
            }
            // Relax light edges; vertices may re-enter B[i].
            for &u in &current {
                let du = tent[u];
                for &(v, w) in g.neighbors(u)? {
                    if w <= delta {
                        relax(
                            v,
                            du + w,
                            u,
                            delta,
                            &mut tent,
                            &mut parent,
                            &mut bucket_of,
                            &mut buckets,
                        );
                    }
                }
            }
        }

        // ── Heavy-edge phase: relax heavy edges from settled vertices ────────
        for &u in &settled {
            let du = tent[u];
            for &(v, w) in g.neighbors(u)? {
                if w > delta {
                    relax(
                        v,
                        du + w,
                        u,
                        delta,
                        &mut tent,
                        &mut parent,
                        &mut bucket_of,
                        &mut buckets,
                    );
                }
            }
        }
        // Do not advance `i`: heavy relaxations may have re-filled B[i] with a
        // vertex whose improved distance landed back in this bucket. Loop again;
        // it terminates because distances only decrease and are bounded below.
        if buckets[i].is_empty() {
            i += 1;
        }
    }

    Ok(DeltaSteppingOutput { dist: tent, parent })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shortest_path::dijkstra::dijkstra;

    #[test]
    fn simple_chain() {
        // 0 ->1(1) ->2(1) ->3(1).
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let out = delta_stepping(&g, 0, 1.0).expect("ok");
        assert!((out.dist[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn matches_dijkstra_small() {
        // 0->1(1), 0->2(4), 1->2(2), 1->3(5), 2->3(1).
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 4.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(1, 3, 5.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let dj = dijkstra(&g, 0).expect("ok");
        for &delta in &[0.5_f64, 1.0, 2.0, 10.0] {
            let ds = delta_stepping(&g, 0, delta).expect("ok");
            for v in 0..g.n {
                assert!(
                    (ds.dist[v] - dj.dist[v]).abs() < 1e-9,
                    "delta={delta} v={v}: {} vs {}",
                    ds.dist[v],
                    dj.dist[v]
                );
            }
        }
    }

    #[test]
    fn matches_dijkstra_undirected_grid() {
        // A small undirected weighted graph with mixed light/heavy edges.
        let mut g = WeightedGraph::new(6);
        g.add_undirected_edge(0, 1, 0.3).expect("ok");
        g.add_undirected_edge(1, 2, 2.5).expect("ok");
        g.add_undirected_edge(0, 3, 1.0).expect("ok");
        g.add_undirected_edge(3, 4, 0.7).expect("ok");
        g.add_undirected_edge(4, 5, 3.2).expect("ok");
        g.add_undirected_edge(2, 5, 0.4).expect("ok");
        g.add_undirected_edge(1, 4, 1.1).expect("ok");
        let dj = dijkstra(&g, 0).expect("ok");
        let ds = delta_stepping(&g, 0, 1.0).expect("ok");
        for v in 0..g.n {
            assert!(
                (ds.dist[v] - dj.dist[v]).abs() < 1e-9,
                "v={v}: {} vs {}",
                ds.dist[v],
                dj.dist[v]
            );
        }
    }

    #[test]
    fn source_distance_zero() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 2.0).expect("ok");
        let out = delta_stepping(&g, 0, 1.0).expect("ok");
        assert!((out.dist[0] - 0.0).abs() < 1e-12);
        assert_eq!(out.parent[0], 0);
    }

    #[test]
    fn unreachable_is_infinity() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).expect("ok");
        // Node 2 is unreachable.
        let out = delta_stepping(&g, 0, 1.0).expect("ok");
        assert!(out.dist[2].is_infinite());
        assert_eq!(out.parent[2], usize::MAX);
    }

    #[test]
    fn rejects_negative_weight() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, -1.0).expect("ok");
        assert!(matches!(
            delta_stepping(&g, 0, 1.0).unwrap_err(),
            GraphalgError::NegativeWeight { .. }
        ));
    }

    #[test]
    fn rejects_bad_delta() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, 1.0).expect("ok");
        assert!(delta_stepping(&g, 0, 0.0).is_err());
        assert!(delta_stepping(&g, 0, -1.0).is_err());
        assert!(delta_stepping(&g, 0, f64::INFINITY).is_err());
    }

    #[test]
    fn source_out_of_range() {
        let g = WeightedGraph::new(3);
        assert!(matches!(
            delta_stepping(&g, 5, 1.0).unwrap_err(),
            GraphalgError::SourceOutOfRange { .. }
        ));
    }

    #[test]
    fn parent_pointers_form_tree() {
        // Each reachable non-source vertex must have a valid parent whose
        // distance + edge weight equals the vertex's distance.
        let mut g = WeightedGraph::new(5);
        g.add_edge(0, 1, 2.0).expect("ok");
        g.add_edge(0, 2, 5.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(2, 3, 2.0).expect("ok");
        g.add_edge(1, 4, 7.0).expect("ok");
        let out = delta_stepping(&g, 0, 1.5).expect("ok");
        for v in 0..g.n {
            if v == 0 || out.dist[v].is_infinite() {
                continue;
            }
            let p = out.parent[v];
            assert!(p != usize::MAX, "v={v} has no parent");
            // Find the edge (p, v).
            let w = g
                .neighbors(p)
                .expect("ok")
                .iter()
                .find(|&&(t, _)| t == v)
                .map(|&(_, w)| w)
                .expect("parent edge exists");
            assert!(
                (out.dist[p] + w - out.dist[v]).abs() < 1e-9,
                "tree broken at v={v}"
            );
        }
    }

    #[test]
    fn large_delta_acts_like_bellman_ford() {
        // With Δ larger than any edge, all edges are "light"; result must still
        // be correct.
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 2, 1.0).expect("ok");
        g.add_edge(0, 2, 5.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let dj = dijkstra(&g, 0).expect("ok");
        let ds = delta_stepping(&g, 0, 1000.0).expect("ok");
        for v in 0..g.n {
            assert!((ds.dist[v] - dj.dist[v]).abs() < 1e-9, "v={v}");
        }
    }

    #[test]
    fn small_delta_acts_like_dijkstra() {
        // Tiny Δ forces many buckets (label-setting behaviour).
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 4.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let dj = dijkstra(&g, 0).expect("ok");
        let ds = delta_stepping(&g, 0, 0.1).expect("ok");
        for v in 0..g.n {
            assert!((ds.dist[v] - dj.dist[v]).abs() < 1e-9, "v={v}");
        }
    }

    #[test]
    fn deterministic() {
        let mut g = WeightedGraph::new(5);
        g.add_undirected_edge(0, 1, 1.0).expect("ok");
        g.add_undirected_edge(1, 2, 2.0).expect("ok");
        g.add_undirected_edge(2, 3, 1.5).expect("ok");
        g.add_undirected_edge(3, 4, 0.5).expect("ok");
        let a = delta_stepping(&g, 0, 1.0).expect("ok");
        let b = delta_stepping(&g, 0, 1.0).expect("ok");
        assert_eq!(a.dist, b.dist);
    }
}
