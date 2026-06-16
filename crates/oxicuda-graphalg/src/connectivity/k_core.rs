//! k-core decomposition via the Batagelj-Zaversnik linear-time bucket method.
//!
//! The **k-core** of a graph is the maximal subgraph in which every vertex has
//! degree at least `k`. The **core number** of a vertex `v` is the largest `k`
//! such that `v` belongs to the k-core. The **degeneracy** of the graph is the
//! maximum core number over all vertices (equivalently, the smallest value `d`
//! such that every subgraph has a vertex of degree `<= d`).
//!
//! # Directed vs undirected
//!
//! Core numbers are an **undirected** notion. The crate's [`AdjacencyList`] is
//! directed by default, so this module first materialises the *symmetric*
//! adjacency (an edge `u -> v` contributes the undirected relation `{u, v}`),
//! deduplicating parallel edges and ignoring self-loops. As a consequence a
//! graph built with [`AdjacencyList::add_undirected_edge`] and a graph built
//! with a single directed edge per pair yield identical core numbers.
//!
//! # Algorithm (Batagelj & Zaversnik, 2003), O(V + E)
//!
//! Vertices are bucket-sorted by current degree into the `vert` array. The
//! arrays `bin` (start offset of each degree bucket), `pos` (position of each
//! vertex inside `vert`), and `vert` (vertices ordered by current degree) are
//! maintained so that the minimum-degree vertex can be extracted in O(1) and a
//! neighbour's degree decremented (with its bucket boundary advanced) in O(1).
//! Processing vertices left-to-right, the degree a vertex has at the moment it
//! is processed is exactly its core number, and the processing order is a
//! degeneracy ordering.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

/// Result of a full k-core decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KCoreResult {
    /// `core_number[v]` is the core number of vertex `v`.
    pub core_number: Vec<usize>,
    /// The graph degeneracy: `max(core_number)` (`0` for an empty graph).
    pub degeneracy: usize,
}

/// Build the symmetric (undirected) adjacency of a directed [`AdjacencyList`].
///
/// Parallel edges are collapsed and self-loops dropped, so `sym[u]` is the set
/// of distinct neighbours of `u` in the underlying undirected graph.
fn symmetric_adjacency(graph: &AdjacencyList) -> GraphalgResult<Vec<Vec<usize>>> {
    let n = graph.n;
    // First gather the raw symmetric neighbour multiset: for every directed edge
    // u -> v (v != u) record v on u's list AND u on v's list. This makes the
    // result undirected whether the input stored one or both directions, but it
    // may contain duplicates (e.g. when both u -> v and v -> u are present).
    let mut raw: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        for &v in graph.neighbors(u)? {
            if v == u {
                continue; // self-loops do not contribute to undirected degree
            }
            raw[u].push(v);
            raw[v].push(u);
        }
    }
    // Deduplicate each vertex's neighbour list independently, in O(deg), using a
    // per-vertex marker keyed by the owning vertex id.
    let mut sym: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut seen: Vec<usize> = vec![usize::MAX; n];
    for u in 0..n {
        for &v in &raw[u] {
            if seen[v] != u {
                seen[v] = u;
                sym[u].push(v);
            }
        }
    }
    Ok(sym)
}

/// Internal Batagelj-Zaversnik core: given the symmetric adjacency, return the
/// per-vertex core numbers together with the degeneracy (removal) ordering.
fn batagelj_zaversnik(sym: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>) {
    let n = sym.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // deg[v] = current degree of v (starts at undirected degree).
    let mut deg: Vec<usize> = sym.iter().map(|adj| adj.len()).collect();
    let max_deg = deg.iter().copied().max().unwrap_or(0);

    // bin[d] = number of vertices with degree d, later turned into start offsets.
    let mut bin: Vec<usize> = vec![0usize; max_deg + 1];
    for &d in &deg {
        bin[d] += 1;
    }
    // Prefix-sum so bin[d] becomes the starting index of bucket d in `vert`.
    let mut start = 0usize;
    for d in 0..=max_deg {
        let count = bin[d];
        bin[d] = start;
        start += count;
    }

    // pos[v] = index of v inside `vert`; vert = vertices ordered by degree.
    let mut pos: Vec<usize> = vec![0usize; n];
    let mut vert: Vec<usize> = vec![0usize; n];
    for v in 0..n {
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        bin[deg[v]] += 1;
    }
    // Restore bin to bucket starts (we advanced each by its count above).
    for d in (1..=max_deg).rev() {
        bin[d] = bin[d - 1];
    }
    bin[0] = 0;

    // Process vertices in increasing current-degree order.
    let mut core: Vec<usize> = vec![0usize; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let v = vert[i];
        core[v] = deg[v];
        order.push(v);
        // "Remove" v: every still-later neighbour u with deg[u] > deg[v] is
        // shifted down one bucket (its degree decreases by one).
        for &u in &sym[v] {
            if deg[u] > deg[v] {
                let du = deg[u];
                let pu = pos[u];
                let pw = bin[du];
                let w = vert[pw];
                if u != w {
                    // Swap u to the front of its bucket so we can shrink it.
                    vert[pu] = w;
                    vert[pw] = u;
                    pos[w] = pu;
                    pos[u] = pw;
                }
                // Advance the start of bucket `du`, shrinking it, and move u into
                // bucket `du - 1`.
                bin[du] += 1;
                deg[u] = du - 1;
            }
        }
    }

    (core, order)
}

/// Compute a full k-core decomposition: per-vertex core numbers and degeneracy.
///
/// Runs in O(V + E). An empty graph yields an empty `core_number` vector and a
/// degeneracy of `0`.
pub fn k_core_decomposition(graph: &AdjacencyList) -> GraphalgResult<KCoreResult> {
    let sym = symmetric_adjacency(graph)?;
    let (core_number, _order) = batagelj_zaversnik(&sym);
    let degeneracy = core_number.iter().copied().max().unwrap_or(0);
    Ok(KCoreResult {
        core_number,
        degeneracy,
    })
}

/// Return only the per-vertex core numbers (`core_number[v]`).
pub fn core_numbers(graph: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    Ok(k_core_decomposition(graph)?.core_number)
}

/// Return the graph degeneracy (maximum core number; `0` for an empty graph).
pub fn degeneracy(graph: &AdjacencyList) -> GraphalgResult<usize> {
    Ok(k_core_decomposition(graph)?.degeneracy)
}

/// Return a degeneracy ordering: the vertices in the order they are removed by
/// the Batagelj-Zaversnik peeling (repeatedly extracting a minimum-degree
/// vertex). The vector is a permutation of `0..graph.n`.
pub fn degeneracy_ordering(graph: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let sym = symmetric_adjacency(graph)?;
    let (_core, order) = batagelj_zaversnik(&sym);
    Ok(order)
}

/// Return the vertices of the k-core: all vertices whose core number is `>= k`.
///
/// The returned vector is sorted ascending. By definition these vertices induce
/// the maximal subgraph with minimum degree `>= k`, and the sets are nested:
/// `k_core_subgraph(k) ⊆ k_core_subgraph(k - 1)`.
pub fn k_core_subgraph(graph: &AdjacencyList, k: usize) -> GraphalgResult<Vec<usize>> {
    let core = core_numbers(graph)?;
    Ok((0..core.len()).filter(|&v| core[v] >= k).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an undirected clique K_n.
    fn clique(n: usize) -> AdjacencyList {
        let mut g = AdjacencyList::new(n);
        for u in 0..n {
            for v in (u + 1)..n {
                g.add_undirected_edge(u, v).expect("edge");
            }
        }
        g
    }

    /// Build an undirected path 0-1-2-...-(n-1).
    fn path(n: usize) -> AdjacencyList {
        let mut g = AdjacencyList::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_undirected_edge(i, i + 1).expect("edge");
        }
        g
    }

    /// Build an undirected cycle C_n.
    fn cycle(n: usize) -> AdjacencyList {
        let mut g = AdjacencyList::new(n);
        for i in 0..n {
            g.add_undirected_edge(i, (i + 1) % n).expect("edge");
        }
        g
    }

    /// Build an undirected star: center 0 with `leaves` leaves.
    fn star(leaves: usize) -> AdjacencyList {
        let mut g = AdjacencyList::new(leaves + 1);
        for leaf in 1..=leaves {
            g.add_undirected_edge(0, leaf).expect("edge");
        }
        g
    }

    #[test]
    fn clique_k5_all_core_four() {
        let g = clique(5);
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![4, 4, 4, 4, 4]);
        assert_eq!(res.degeneracy, 4);
    }

    #[test]
    fn path_p5_all_core_one() {
        let g = path(5);
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![1, 1, 1, 1, 1]);
        assert_eq!(res.degeneracy, 1);
    }

    #[test]
    fn cycle_c5_all_core_two() {
        let g = cycle(5);
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![2, 2, 2, 2, 2]);
        assert_eq!(res.degeneracy, 2);
    }

    #[test]
    fn star_all_core_one() {
        let g = star(4);
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![1, 1, 1, 1, 1]);
        assert_eq!(res.degeneracy, 1);
    }

    #[test]
    fn tree_degeneracy_one() {
        // A small binary-ish tree: 0-1, 0-2, 1-3, 1-4, 2-5.
        let mut g = AdjacencyList::new(6);
        for (u, v) in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)] {
            g.add_undirected_edge(u, v).expect("edge");
        }
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.degeneracy, 1);
        assert!(res.core_number.iter().all(|&c| c == 1));
    }

    #[test]
    fn triangle_plus_pendant() {
        // Triangle {0,1,2} plus a pendant edge 3-2.
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("edge");
        g.add_undirected_edge(1, 2).expect("edge");
        g.add_undirected_edge(0, 2).expect("edge");
        g.add_undirected_edge(3, 2).expect("edge");
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![2, 2, 2, 1]);
        assert_eq!(res.degeneracy, 2);
    }

    #[test]
    fn directed_single_edge_equals_undirected() {
        // Graph built with single directed edges must give the same core
        // numbers as the undirected version (symmetric-adjacency handling).
        let mut directed = AdjacencyList::new(3);
        directed.add_edge(0, 1).expect("edge");
        directed.add_edge(1, 2).expect("edge");
        directed.add_edge(2, 0).expect("edge");
        let res = core_numbers(&directed).expect("kcore");
        assert_eq!(res, vec![2, 2, 2]); // it is C_3, all core 2
    }

    #[test]
    fn parallel_edges_deduplicated() {
        // Adding the same undirected edge twice must not inflate degree.
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("edge");
        g.add_undirected_edge(0, 1).expect("edge");
        g.add_undirected_edge(1, 2).expect("edge");
        let res = core_numbers(&g).expect("kcore");
        assert_eq!(res, vec![1, 1, 1]);
    }

    #[test]
    fn self_loops_ignored() {
        let mut g = AdjacencyList::new(2);
        g.add_undirected_edge(0, 0).expect("edge"); // self loop
        g.add_undirected_edge(0, 1).expect("edge");
        let res = core_numbers(&g).expect("kcore");
        assert_eq!(res, vec![1, 1]);
    }

    #[test]
    fn k_core_subgraph_nesting() {
        // Use a graph with a spread of core numbers: triangle + pendant.
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("edge");
        g.add_undirected_edge(1, 2).expect("edge");
        g.add_undirected_edge(0, 2).expect("edge");
        g.add_undirected_edge(3, 2).expect("edge");
        let degen = degeneracy(&g).expect("degeneracy");
        for k in 1..=(degen + 1) {
            let hi = k_core_subgraph(&g, k).expect("subgraph");
            let lo = k_core_subgraph(&g, k - 1).expect("subgraph");
            // Every vertex in the k-core is also in the (k-1)-core.
            for &v in &hi {
                assert!(lo.contains(&v), "nesting violated at k={k} vertex {v}");
            }
        }
        // Concretely: 2-core is the triangle, 1-core is everything.
        assert_eq!(k_core_subgraph(&g, 2).expect("sg"), vec![0, 1, 2]);
        assert_eq!(k_core_subgraph(&g, 1).expect("sg"), vec![0, 1, 2, 3]);
        assert_eq!(k_core_subgraph(&g, 0).expect("sg"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn degeneracy_ordering_is_permutation() {
        let g = clique(5);
        let mut order = degeneracy_ordering(&g).expect("order");
        assert_eq!(order.len(), 5);
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn empty_graph() {
        let g = AdjacencyList::new(0);
        let res = k_core_decomposition(&g).expect("kcore");
        assert!(res.core_number.is_empty());
        assert_eq!(res.degeneracy, 0);
        assert!(degeneracy_ordering(&g).expect("order").is_empty());
        assert!(k_core_subgraph(&g, 1).expect("sg").is_empty());
    }

    #[test]
    fn single_vertex() {
        let g = AdjacencyList::new(1);
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![0]);
        assert_eq!(res.degeneracy, 0);
        assert_eq!(degeneracy_ordering(&g).expect("order"), vec![0]);
        assert_eq!(k_core_subgraph(&g, 0).expect("sg"), vec![0]);
        assert!(k_core_subgraph(&g, 1).expect("sg").is_empty());
    }

    #[test]
    fn isolated_vertices_core_zero() {
        // Edge 0-1 plus two isolated vertices 2, 3.
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("edge");
        let res = core_numbers(&g).expect("kcore");
        assert_eq!(res, vec![1, 1, 0, 0]);
        assert_eq!(degeneracy(&g).expect("d"), 1);
    }

    #[test]
    fn complete_bipartite_k33() {
        // K_{3,3}: parts {0,1,2} and {3,4,5}. Every vertex has degree 3 and the
        // whole graph is 3-degenerate (it is 3-regular and has no vertex of
        // degree < 3 in any 3-core... actually K_{3,3} core number is 3).
        let mut g = AdjacencyList::new(6);
        for u in 0..3 {
            for v in 3..6 {
                g.add_undirected_edge(u, v).expect("edge");
            }
        }
        let res = k_core_decomposition(&g).expect("kcore");
        assert_eq!(res.core_number, vec![3, 3, 3, 3, 3, 3]);
        assert_eq!(res.degeneracy, 3);
    }
}
