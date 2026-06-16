//! Transitive closure of a directed graph.
//!
//! The transitive closure of a digraph `G` is the relation `R` where `R[u][v]`
//! holds iff there is a directed path from `u` to `v` in `G`. With
//! `reflexive = true` the diagonal `R[v][v]` is forced true regardless of paths;
//! with `reflexive = false` the diagonal is true only when `v` lies on a cycle
//! (so a path of length `>= 1` returns to it).
//!
//! Two independent constructions are provided so they can cross-check:
//!   * [`transitive_closure`] — boolean Floyd-Warshall, `O(V^3)`.
//!   * [`transitive_closure_bfs`] — one BFS per source, `O(V * (V + E))`.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// Dense boolean reachability relation for an `n`-vertex digraph.
///
/// The matrix is stored row-major: entry `(u, v)` lives at index `u * n + v`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitiveClosure {
    /// Number of vertices.
    pub n: usize,
    /// Row-major `n * n` reachability bits.
    reach: Vec<bool>,
    /// Whether the closure was computed with the reflexive diagonal forced on.
    reflexive: bool,
}

impl TransitiveClosure {
    /// Is `v` reachable from `u`? Returns [`GraphalgError::IndexOutOfBounds`] if
    /// either index is `>= n`.
    pub fn reachable(&self, u: usize, v: usize) -> GraphalgResult<bool> {
        if u >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n,
            });
        }
        if v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: v,
                len: self.n,
            });
        }
        Ok(self.reach[u * self.n + v])
    }

    /// Borrow the underlying row-major reachability matrix (`n * n` bits).
    pub fn matrix(&self) -> &[bool] {
        &self.reach
    }

    /// Whether this closure was built with the reflexive diagonal forced on.
    pub fn is_reflexive(&self) -> bool {
        self.reflexive
    }
}

/// Transitive closure via boolean Floyd-Warshall, `O(V^3)`.
///
/// `R[i][j]` is initialised to `edge(i -> j)` (plus the diagonal when
/// `reflexive`), then for every intermediate `k` it is updated by
/// `R[i][j] |= R[i][k] & R[k][j]`.
pub fn transitive_closure(
    graph: &AdjacencyList,
    reflexive: bool,
) -> GraphalgResult<TransitiveClosure> {
    let n = graph.n;
    let mut reach = vec![false; n * n];

    // Seed with direct edges.
    for u in 0..n {
        for &v in graph.neighbors(u)? {
            reach[u * n + v] = true;
        }
    }
    if reflexive {
        for v in 0..n {
            reach[v * n + v] = true;
        }
    }

    // Floyd-Warshall closure: for each pivot k, connect i->k->j.
    for k in 0..n {
        // Splitting the borrow: row k is read-only inside the i-loop.
        for i in 0..n {
            if reach[i * n + k] {
                let base_i = i * n;
                let base_k = k * n;
                for j in 0..n {
                    if reach[base_k + j] {
                        reach[base_i + j] = true;
                    }
                }
            }
        }
    }

    Ok(TransitiveClosure {
        n,
        reach,
        reflexive,
    })
}

/// Transitive closure via one breadth-first search per source,
/// `O(V * (V + E))`.
///
/// From each source `s`, BFS marks every vertex reachable through a path of
/// length `>= 1`; those bits form row `s`. When `reflexive` is set the diagonal
/// is forced true afterwards (otherwise `R[s][s]` is true only if a cycle
/// brings the search back to `s`).
pub fn transitive_closure_bfs(
    graph: &AdjacencyList,
    reflexive: bool,
) -> GraphalgResult<TransitiveClosure> {
    let n = graph.n;
    let mut reach = vec![false; n * n];
    let mut visited = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for s in 0..n {
        // Reset the per-source scratch state.
        for v in visited.iter_mut() {
            *v = false;
        }
        queue.clear();

        // Seed the queue with the direct successors of s (length-1 reach). We do
        // NOT mark s as visited up front, so a cycle returning to s correctly
        // sets R[s][s] even without the reflexive flag.
        for &v in graph.neighbors(s)? {
            if !visited[v] {
                visited[v] = true;
                reach[s * n + v] = true;
                queue.push_back(v);
            }
        }
        while let Some(u) = queue.pop_front() {
            for &w in graph.neighbors(u)? {
                if !visited[w] {
                    visited[w] = true;
                    reach[s * n + w] = true;
                    queue.push_back(w);
                }
            }
        }

        if reflexive {
            reach[s * n + s] = true;
        }
    }

    Ok(TransitiveClosure {
        n,
        reach,
        reflexive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directed chain 0 -> 1 -> 2 -> 3.
    fn chain(n: usize) -> AdjacencyList {
        let mut g = AdjacencyList::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1).expect("edge");
        }
        g
    }

    #[test]
    fn chain_upper_triangular_irreflexive() {
        let g = chain(4);
        let tc = transitive_closure(&g, false).expect("tc");
        for i in 0..4 {
            for j in 0..4 {
                let expected = i < j; // strictly upper triangular
                assert_eq!(tc.reachable(i, j).expect("reach"), expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn chain_upper_triangular_reflexive() {
        let g = chain(4);
        let tc = transitive_closure(&g, true).expect("tc");
        for i in 0..4 {
            for j in 0..4 {
                let expected = i <= j; // upper triangular incl. diagonal
                assert_eq!(tc.reachable(i, j).expect("reach"), expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn small_dag_full_matrix() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3.
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("edge");
        g.add_edge(0, 2).expect("edge");
        g.add_edge(1, 3).expect("edge");
        g.add_edge(2, 3).expect("edge");
        let tc = transitive_closure(&g, false).expect("tc");
        // 0 reaches 1,2,3; 1 reaches 3; 2 reaches 3; 3 reaches nothing.
        #[rustfmt::skip]
        let expected: Vec<bool> = vec![
            false, true,  true,  true,
            false, false, false, true,
            false, false, false, true,
            false, false, false, false,
        ];
        assert_eq!(tc.matrix(), expected.as_slice());
    }

    #[test]
    fn fw_and_bfs_agree_on_several_graphs() {
        // Several deterministic digraphs; both methods must produce identical
        // matrices for both reflexive settings.
        let mut graphs: Vec<AdjacencyList> = Vec::new();

        // g1: chain.
        graphs.push(chain(6));

        // g2: a DAG with cross edges.
        let mut g2 = AdjacencyList::new(5);
        for (u, v) in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (0, 4)] {
            g2.add_edge(u, v).expect("edge");
        }
        graphs.push(g2);

        // g3: contains a cycle 0->1->2->0 plus a tail to 3.
        let mut g3 = AdjacencyList::new(4);
        for (u, v) in [(0, 1), (1, 2), (2, 0), (2, 3)] {
            g3.add_edge(u, v).expect("edge");
        }
        graphs.push(g3);

        // g4: two disjoint components 0->1 and 2->3->4.
        let mut g4 = AdjacencyList::new(5);
        for (u, v) in [(0, 1), (2, 3), (3, 4)] {
            g4.add_edge(u, v).expect("edge");
        }
        graphs.push(g4);

        // g5: deterministic pseudo-random-ish digraph (LCG-style index walk).
        let mut g5 = AdjacencyList::new(8);
        let mut state = 1u64;
        for _ in 0..20 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as usize % 8;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = (state >> 33) as usize % 8;
            g5.add_edge(u, v).expect("edge");
        }
        graphs.push(g5);

        for g in &graphs {
            for &reflexive in &[false, true] {
                let a = transitive_closure(g, reflexive).expect("fw");
                let b = transitive_closure_bfs(g, reflexive).expect("bfs");
                assert_eq!(a.matrix(), b.matrix(), "mismatch (reflexive={reflexive})");
            }
        }
    }

    #[test]
    fn cycle_everyone_reaches_everyone() {
        // 0 -> 1 -> 2 -> 0.
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("edge");
        g.add_edge(1, 2).expect("edge");
        g.add_edge(2, 0).expect("edge");
        let tc = transitive_closure(&g, false).expect("tc");
        for i in 0..3 {
            for j in 0..3 {
                assert!(tc.reachable(i, j).expect("reach"), "({i},{j})");
            }
        }
        // Diagonal true even without reflexive flag, due to the cycle.
        for v in 0..3 {
            assert!(tc.reachable(v, v).expect("reach"));
        }
        // BFS variant agrees on the cyclic diagonal.
        let tb = transitive_closure_bfs(&g, false).expect("bfs");
        for v in 0..3 {
            assert!(tb.reachable(v, v).expect("reach"));
        }
    }

    #[test]
    fn reflexive_flag_respected_with_isolated_vertex() {
        // 0 -> 1, vertex 2 isolated, no self-loops.
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("edge");

        let irreflexive = transitive_closure(&g, false).expect("tc");
        // Without the flag, no diagonal entries (no cycles / self-loops).
        for v in 0..3 {
            assert!(!irreflexive.reachable(v, v).expect("reach"), "diag {v}");
        }

        let reflexive = transitive_closure(&g, true).expect("tc");
        for v in 0..3 {
            assert!(reflexive.reachable(v, v).expect("reach"), "diag {v}");
        }
        // The isolated vertex 2 reaches only itself (and only when reflexive).
        assert!(!reflexive.reachable(2, 0).expect("reach"));
        assert!(!reflexive.reachable(2, 1).expect("reach"));
    }

    #[test]
    fn reachable_bounds_check() {
        let g = chain(3);
        let tc = transitive_closure(&g, false).expect("tc");
        assert!(matches!(
            tc.reachable(3, 0),
            Err(GraphalgError::IndexOutOfBounds { index: 3, len: 3 })
        ));
        assert!(matches!(
            tc.reachable(0, 5),
            Err(GraphalgError::IndexOutOfBounds { index: 5, len: 3 })
        ));
    }

    #[test]
    fn empty_graph_closure() {
        let g = AdjacencyList::new(0);
        let tc = transitive_closure(&g, true).expect("tc");
        assert_eq!(tc.n, 0);
        assert!(tc.matrix().is_empty());
        let tb = transitive_closure_bfs(&g, true).expect("bfs");
        assert!(tb.matrix().is_empty());
    }

    #[test]
    fn is_reflexive_reported() {
        let g = chain(2);
        assert!(transitive_closure(&g, true).expect("tc").is_reflexive());
        assert!(!transitive_closure(&g, false).expect("tc").is_reflexive());
    }
}
