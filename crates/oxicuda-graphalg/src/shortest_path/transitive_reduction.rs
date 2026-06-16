//! Transitive reduction of a directed acyclic graph.
//!
//! The transitive reduction of a DAG `G` is the unique minimal digraph `R` on
//! the same vertices with the same reachability relation (transitive closure)
//! as `G`. For DAGs it is exactly the *covering relation* of the reachability
//! partial order: an edge `u -> v` is kept iff there is no intermediate vertex
//! that already routes `u` to `v`.
//!
//! Only DAGs are accepted — on a graph containing a cycle the reduction is not
//! unique, so [`transitive_reduction`] returns [`GraphalgError::NotADag`](crate::error::GraphalgError::NotADag). Cycle
//! detection reuses the crate's Kahn topological sort.
//!
//! # Algorithm
//!
//! Compute reachability with [`transitive_closure`] (irreflexive). Edge
//! `u -> v` is **redundant** iff `u` has another direct successor `w` (`w != v`,
//! `w != u`) from which `v` is reachable — then `u -> w ->* v` already provides
//! the connection. The reduction keeps precisely the non-redundant edges. Each
//! original `u -> v` pair is considered once (parallel edges and self-loops are
//! dropped, matching the covering-relation semantics on a DAG).

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;
use crate::shortest_path::transitive_closure::transitive_closure;
use crate::topological::kahn::topo_sort_kahn;

/// Compute the transitive reduction of a DAG.
///
/// Returns a new [`AdjacencyList`] on the same `n` vertices containing only the
/// covering edges. Returns [`GraphalgError::NotADag`](crate::error::GraphalgError::NotADag) if the input has a cycle.
///
/// The output preserves the reachability relation of the input: the transitive
/// closures of the reduction and the original are identical.
pub fn transitive_reduction(graph: &AdjacencyList) -> GraphalgResult<AdjacencyList> {
    let n = graph.n;

    // Reject cyclic input: the reduction is only unique on DAGs. Kahn returns
    // NotADag precisely when a cycle is present.
    topo_sort_kahn(graph)?;

    // Reachability under paths of length >= 1.
    let tc = transitive_closure(graph, false)?;
    let reach = tc.matrix();

    // Deduplicate the direct successors of each vertex (ignore self-loops and
    // parallel edges) so each covering candidate is examined once.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut seen: Vec<usize> = vec![usize::MAX; n];
    for u in 0..n {
        for &v in graph.neighbors(u)? {
            if v == u {
                continue;
            }
            if seen[v] != u {
                seen[v] = u;
                succ[u].push(v);
            }
        }
    }

    let mut reduced = AdjacencyList::new(n);
    for u in 0..n {
        for &v in &succ[u] {
            // Edge u -> v is redundant iff some OTHER direct successor w of u can
            // reach v (w != v, w != u). Keep the edge only if none can.
            let mut redundant = false;
            for &w in &succ[u] {
                if w == v || w == u {
                    continue;
                }
                if reach[w * n + v] {
                    redundant = true;
                    break;
                }
            }
            if !redundant {
                reduced.add_edge(u, v)?;
            }
        }
    }

    Ok(reduced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::GraphalgError;

    /// Collect the directed edges of an adjacency list as a sorted set.
    fn edge_set(g: &AdjacencyList) -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = Vec::new();
        for u in 0..g.n {
            for &v in &g.edges[u] {
                e.push((u, v));
            }
        }
        e.sort_unstable();
        e.dedup();
        e
    }

    #[test]
    fn drops_single_shortcut() {
        // 0->1, 1->2, 0->2 : reduction must drop 0->2.
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("edge");
        g.add_edge(1, 2).expect("edge");
        g.add_edge(0, 2).expect("edge");
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(edge_set(&r), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn diamond_drops_long_shortcut() {
        // Diamond 0->1, 0->2, 1->3, 2->3 plus redundant 0->3.
        let mut g = AdjacencyList::new(4);
        for (u, v) in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)] {
            g.add_edge(u, v).expect("edge");
        }
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(edge_set(&r), vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn already_reduced_chain_unchanged() {
        // A simple chain is its own reduction.
        let mut g = AdjacencyList::new(4);
        for i in 0..3 {
            g.add_edge(i, i + 1).expect("edge");
        }
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(edge_set(&r), edge_set(&g));
    }

    #[test]
    fn already_reduced_tree_unchanged() {
        // An out-tree (every covering edge is essential) is unchanged.
        let mut g = AdjacencyList::new(7);
        for (u, v) in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)] {
            g.add_edge(u, v).expect("edge");
        }
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(edge_set(&r), edge_set(&g));
    }

    #[test]
    fn reduction_preserves_closure() {
        // Defining property: closure(reduction) == closure(original).
        let mut g = AdjacencyList::new(6);
        for (u, v) in [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 5),
            (1, 3),
            (1, 4),
            (2, 4),
            (3, 5),
            (4, 5),
            (2, 5),
        ] {
            g.add_edge(u, v).expect("edge");
        }
        let r = transitive_reduction(&g).expect("reduction");
        let cg = transitive_closure(&g, false).expect("closure orig");
        let cr = transitive_closure(&r, false).expect("closure reduced");
        assert_eq!(cg.matrix(), cr.matrix());
        // And the reduction has no more edges than the original.
        assert!(r.num_edges() <= g.num_edges());
    }

    #[test]
    fn cyclic_graph_is_not_a_dag() {
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("edge");
        g.add_edge(1, 2).expect("edge");
        g.add_edge(2, 0).expect("edge");
        assert!(matches!(
            transitive_reduction(&g),
            Err(GraphalgError::NotADag)
        ));
    }

    #[test]
    fn hasse_diagram_of_divisibility_poset() {
        // Poset on {1,2,3,4,6,12} ordered by divisibility, indexed
        // 0:1, 1:2, 2:3, 3:4, 4:6, 5:12. Full order relation -> covering relation.
        // Cover edges: 1|2, 1|3, 2|4, 2|6, 3|6, 4|12, 6|12.
        let labels = [1usize, 2, 3, 4, 6, 12];
        let mut g = AdjacencyList::new(6);
        for i in 0..6 {
            for j in 0..6 {
                if i != j && labels[j] % labels[i] == 0 {
                    g.add_edge(i, j).expect("edge"); // full strict order relation
                }
            }
        }
        let r = transitive_reduction(&g).expect("reduction");
        let expected = vec![
            (0, 1), // 1 -> 2
            (0, 2), // 1 -> 3
            (1, 3), // 2 -> 4
            (1, 4), // 2 -> 6
            (2, 4), // 3 -> 6
            (3, 5), // 4 -> 12
            (4, 5), // 6 -> 12
        ];
        assert_eq!(edge_set(&r), expected);
        // Closure preserved as well (covering relation regenerates the order).
        let cg = transitive_closure(&g, false).expect("c orig");
        let cr = transitive_closure(&r, false).expect("c reduced");
        assert_eq!(cg.matrix(), cr.matrix());
    }

    #[test]
    fn empty_graph_reduces_to_empty() {
        let g = AdjacencyList::new(0);
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(r.n, 0);
        assert_eq!(r.num_edges(), 0);
    }

    #[test]
    fn single_vertex_no_edges() {
        let g = AdjacencyList::new(1);
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(r.n, 1);
        assert_eq!(r.num_edges(), 0);
    }

    #[test]
    fn parallel_edges_collapsed() {
        // Duplicate edge 0->1 should appear once in the reduction.
        let mut g = AdjacencyList::new(2);
        g.add_edge(0, 1).expect("edge");
        g.add_edge(0, 1).expect("edge");
        let r = transitive_reduction(&g).expect("reduction");
        assert_eq!(edge_set(&r), vec![(0, 1)]);
        assert_eq!(r.num_edges(), 1);
    }
}
