//! Harmonic centrality `Σ_{u ≠ v} 1 / d(v, u)` (unweighted graph).
//!
//! Harmonic centrality (Marchiori & Latora 2000) is the sum of the reciprocals
//! of the shortest-path distances from a vertex to every other vertex. Unlike
//! closeness centrality — which sums distances and therefore requires a
//! connected graph — unreachable vertices simply contribute `1/∞ = 0`, so
//! harmonic centrality is well-defined on **disconnected** graphs.
//!
//! ```text
//! H(v) = Σ_{u ≠ v, d(v,u) < ∞} 1 / d(v, u)
//! ```
//!
//! Distances are computed by a BFS from each source (treating the graph as
//! unweighted). The optional normalised variant divides by `n − 1` so values
//! lie in `[0, 1]`.
//!
//! ## References
//! - Marchiori, M. & Latora, V. (2000). "Harmony in the small-world."
//!   Physica A 285(3–4), 539–546.
//! - Boldi, P. & Vigna, S. (2014). "Axioms for centrality." Internet Math.

use std::collections::VecDeque;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

/// Compute the (raw) harmonic centrality of every vertex.
///
/// Returns a length-`n` vector `H` where `H[v] = Σ_{u≠v} 1/d(v,u)` over
/// vertices `u` reachable from `v` (BFS distance, unweighted).
///
/// # Errors
/// Propagates [`AdjacencyList::neighbors`] index errors (cannot occur for a
/// well-formed graph; surfaced for robustness).
pub fn harmonic_centrality(g: &AdjacencyList) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    let mut hc = vec![0.0f64; n];
    for s in 0..n {
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            for &v in g.neighbors(u)? {
                if dist[v] < 0 {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        let mut acc = 0.0f64;
        for &d in dist.iter() {
            if d > 0 {
                acc += 1.0 / d as f64;
            }
        }
        hc[s] = acc;
    }
    Ok(hc)
}

/// Compute the **normalised** harmonic centrality (`H(v) / (n − 1)`).
///
/// Values lie in `[0, 1]`; `1.0` is attained by a vertex adjacent to every
/// other vertex. For `n ≤ 1` all entries are `0`.
///
/// # Errors
/// As [`harmonic_centrality`].
pub fn harmonic_centrality_normalized(g: &AdjacencyList) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    let mut hc = harmonic_centrality(g)?;
    if n > 1 {
        let denom = (n - 1) as f64;
        for v in hc.iter_mut() {
            *v /= denom;
        }
    }
    Ok(hc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_graph_centre_value() {
        // Path 0 - 1 - 2. From node 1: 1/1 + 1/1 = 2.
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        let h = harmonic_centrality(&g).expect("ok");
        assert!((h[1] - 2.0).abs() < 1e-12, "h[1] = {}", h[1]);
        // From node 0: 1/1 (to 1) + 1/2 (to 2) = 1.5.
        assert!((h[0] - 1.5).abs() < 1e-12, "h[0] = {}", h[0]);
    }

    #[test]
    fn complete_graph_is_n_minus_1() {
        // K_4: every vertex is at distance 1 from the other 3 → H = 3.
        let n = 4;
        let mut g = AdjacencyList::new(n);
        for i in 0..n {
            for j in i + 1..n {
                g.add_undirected_edge(i, j).expect("ok");
            }
        }
        let h = harmonic_centrality(&g).expect("ok");
        for (v, &val) in h.iter().enumerate() {
            assert!((val - 3.0).abs() < 1e-12, "vertex {v}: {val}");
        }
    }

    #[test]
    fn disconnected_graph_well_defined() {
        // Two disjoint edges: 0-1 and 2-3. Each vertex reaches exactly one other
        // at distance 1 → H = 1 everywhere (no infinities).
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(2, 3).expect("ok");
        let h = harmonic_centrality(&g).expect("ok");
        for (v, &val) in h.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-12, "vertex {v}: {val}");
            assert!(val.is_finite());
        }
    }

    #[test]
    fn isolated_vertex_is_zero() {
        // Vertex 2 is isolated.
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        let h = harmonic_centrality(&g).expect("ok");
        assert!((h[2] - 0.0).abs() < 1e-12, "isolated h[2] = {}", h[2]);
    }

    #[test]
    fn empty_graph() {
        let g = AdjacencyList::new(0);
        let h = harmonic_centrality(&g).expect("ok");
        assert!(h.is_empty());
    }

    #[test]
    fn single_vertex() {
        let g = AdjacencyList::new(1);
        let h = harmonic_centrality(&g).expect("ok");
        assert_eq!(h.len(), 1);
        assert!((h[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_in_unit_range() {
        let n = 5;
        let mut g = AdjacencyList::new(n);
        for i in 0..n {
            for j in i + 1..n {
                g.add_undirected_edge(i, j).expect("ok");
            }
        }
        let h = harmonic_centrality_normalized(&g).expect("ok");
        // Complete graph → all normalised values are exactly 1.
        for &val in &h {
            assert!((val - 1.0).abs() < 1e-12, "val = {val}");
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn directed_asymmetry() {
        // Directed star out of 0: 0->1, 0->2. From 0 both are reachable (H=2);
        // from 1 nothing is reachable (H=0).
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(0, 2).expect("ok");
        let h = harmonic_centrality(&g).expect("ok");
        assert!((h[0] - 2.0).abs() < 1e-12, "h[0] = {}", h[0]);
        assert!((h[1] - 0.0).abs() < 1e-12, "h[1] = {}", h[1]);
        assert!((h[2] - 0.0).abs() < 1e-12, "h[2] = {}", h[2]);
    }

    #[test]
    fn deterministic() {
        let mut g = AdjacencyList::new(6);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(2, 3).expect("ok");
        g.add_undirected_edge(3, 4).expect("ok");
        g.add_undirected_edge(4, 5).expect("ok");
        let a = harmonic_centrality(&g).expect("ok");
        let b = harmonic_centrality(&g).expect("ok");
        assert_eq!(a, b);
    }

    #[test]
    fn all_values_finite_and_nonneg() {
        // A graph with a mix of reachable and unreachable pairs.
        let mut g = AdjacencyList::new(5);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(3, 4).expect("ok");
        let h = harmonic_centrality(&g).expect("ok");
        assert!(h.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn star_centre_higher_than_leaf() {
        // Undirected star: centre 0 connected to 1,2,3.
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        g.add_undirected_edge(0, 3).expect("ok");
        let h = harmonic_centrality(&g).expect("ok");
        // Centre: 1/1 * 3 = 3. Leaf: 1/1 (centre) + 1/2 + 1/2 = 2.
        assert!((h[0] - 3.0).abs() < 1e-12, "centre {}", h[0]);
        assert!((h[1] - 2.0).abs() < 1e-12, "leaf {}", h[1]);
        assert!(h[0] > h[1]);
    }
}
