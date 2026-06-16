//! Bron-Kerbosch algorithm for enumerating all maximal cliques in an undirected graph.
//!
//! Three public entry points are provided:
//!
//! - [`bron_kerbosch`]: basic BK with Tomita 2006 pivot selection.
//! - [`bron_kerbosch_degeneracy`]: BK with Eppstein 2010 degeneracy-order outer loop
//!   (faster in practice on sparse graphs; produces the same clique set).
//! - [`maximum_clique`]: returns the single largest clique (first found on tie).
//!
//! # References
//! - Bron & Kerbosch, "Algorithm 457: Finding all cliques of an undirected graph," CACM 1973.
//! - Tomita, Tanaka & Takahashi, "The worst-case time complexity for generating all maximal
//!   cliques and computational experiments," TCS 2006.
//! - Eppstein, Löffler & Strash, "Listing all maximal cliques in sparse graphs in near-optimal
//!   time," ISAAC 2010.

use std::collections::HashSet;

use crate::error::GraphalgResult;
use crate::repr::AdjacencyList;

// ─── Internal adjacency set representation ─────────────────────────────────

/// Convert an `AdjacencyList` to a per-vertex `HashSet<usize>` of neighbors.
///
/// For undirected graphs (built with `add_undirected_edge`) this captures both
/// directions. Self-loops are excluded to avoid degenerate cliques of the form `{v, v}`.
fn build_adj_sets(g: &AdjacencyList) -> Vec<HashSet<usize>> {
    (0..g.n)
        .map(|u| {
            g.edges[u]
                .iter()
                .copied()
                .filter(|&v| v != u) // exclude self-loops
                .collect()
        })
        .collect()
}

// ─── Tomita pivot selection ─────────────────────────────────────────────────

/// Choose the pivot vertex `u ∈ P ∪ X` that maximises `|P ∩ N(u)|` (Tomita 2006).
///
/// The pivot minimises the number of recursive calls by maximising the number of
/// candidates eliminated from `P` in each branch.
fn pick_pivot(p: &HashSet<usize>, x: &HashSet<usize>, adj: &[HashSet<usize>]) -> usize {
    let mut best_u = usize::MAX;
    let mut best_count = 0usize;

    for &u in p.iter().chain(x.iter()) {
        let count = p.intersection(&adj[u]).count();
        if best_u == usize::MAX || count > best_count {
            best_count = count;
            best_u = u;
        }
    }

    best_u
}

// ─── Core BK recursion ─────────────────────────────────────────────────────

/// Recursive BK step with Tomita pivot.
///
/// `r` — current clique being grown.
/// `p` — candidates (vertices connected to all of `r`).
/// `x` — already-processed vertices (connected to all of `r`).
fn bk_pivot(
    r: &mut Vec<usize>,
    p: &mut HashSet<usize>,
    x: &mut HashSet<usize>,
    adj: &[HashSet<usize>],
    result: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        let mut clique = r.clone();
        clique.sort_unstable();
        result.push(clique);
        return;
    }

    if p.is_empty() {
        return;
    }

    let pivot = pick_pivot(p, x, adj);

    // Candidates = P \ N(pivot) — must collect first to avoid borrow issues.
    let mut candidates: Vec<usize> = p
        .iter()
        .copied()
        .filter(|v| !adj[pivot].contains(v))
        .collect();
    candidates.sort_unstable(); // deterministic ordering

    for v in candidates {
        // New P = P ∩ N(v)
        let new_p: HashSet<usize> = p.intersection(&adj[v]).copied().collect();
        // New X = X ∩ N(v)
        let new_x: HashSet<usize> = x.intersection(&adj[v]).copied().collect();

        r.push(v);
        bk_pivot(r, &mut { new_p }, &mut { new_x }, adj, result);
        r.pop();

        p.remove(&v);
        x.insert(v);
    }
}

// ─── Degeneracy ordering ───────────────────────────────────────────────────

/// Compute a degeneracy ordering of `g`: repeatedly remove the vertex of minimum
/// degree among remaining vertices.
///
/// Returns `(ordering, position)` where `position[v]` is the index at which `v`
/// appears in `ordering`.
fn compute_degeneracy_ordering(g: &AdjacencyList) -> (Vec<usize>, Vec<usize>) {
    let n = g.n;
    let mut degrees: Vec<usize> = (0..n).map(|u| g.edges[u].len()).collect();
    let mut removed = vec![false; n];
    let mut ordering = Vec::with_capacity(n);

    for _ in 0..n {
        // Find the non-removed vertex with minimum degree.
        let v = (0..n)
            .filter(|&u| !removed[u])
            .min_by_key(|&u| degrees[u])
            .unwrap_or(0); // n > 0 is guaranteed by caller

        ordering.push(v);
        removed[v] = true;

        // Update degrees of v's neighbors.
        for &w in &g.edges[v] {
            if !removed[w] && w != v {
                degrees[w] = degrees[w].saturating_sub(1);
            }
        }
    }

    let mut position = vec![0usize; n];
    for (idx, &v) in ordering.iter().enumerate() {
        position[v] = idx;
    }

    (ordering, position)
}

// ─── Public API ────────────────────────────────────────────────────────────

/// Enumerate all maximal cliques via Bron-Kerbosch with Tomita 2006 pivot selection.
///
/// Returns a vector of cliques; each clique is a **sorted** vector of vertex indices.
/// An empty graph (`g.n == 0`) returns an empty vector (not an error).
pub fn bron_kerbosch(g: &AdjacencyList) -> GraphalgResult<Vec<Vec<usize>>> {
    if g.n == 0 {
        return Ok(Vec::new());
    }

    let adj = build_adj_sets(g);
    let p: HashSet<usize> = (0..g.n).collect();
    let x: HashSet<usize> = HashSet::new();

    let mut result = Vec::new();
    bk_pivot(&mut Vec::new(), &mut { p }, &mut { x }, &adj, &mut result);

    // Sort the result set for determinism (sort by length desc, then lexicographic).
    result.sort_unstable();
    Ok(result)
}

/// Enumerate all maximal cliques via Bron-Kerbosch with Eppstein 2010 degeneracy-order
/// outer loop.
///
/// Produces the **same clique set** as [`bron_kerbosch`] but is faster in practice on
/// sparse graphs because the degeneracy order limits the size of the candidate set `P`
/// at each top-level call.
pub fn bron_kerbosch_degeneracy(g: &AdjacencyList) -> GraphalgResult<Vec<Vec<usize>>> {
    if g.n == 0 {
        return Ok(Vec::new());
    }

    let adj = build_adj_sets(g);
    let (ordering, position) = compute_degeneracy_ordering(g);

    let mut result = Vec::new();

    for &v in &ordering {
        let pos_v = position[v];

        // P = later neighbors of v (those appearing after v in the ordering) ∩ N(v)
        let p: HashSet<usize> = adj[v]
            .iter()
            .copied()
            .filter(|&u| position[u] > pos_v)
            .collect();

        // X = earlier neighbors of v (those appearing before v in the ordering)
        let x: HashSet<usize> = adj[v]
            .iter()
            .copied()
            .filter(|&u| position[u] < pos_v)
            .collect();

        let mut r = vec![v];
        bk_pivot(&mut r, &mut { p }, &mut { x }, &adj, &mut result);
    }

    result.sort_unstable();
    Ok(result)
}

/// Return the single maximum clique (largest by size; first found lexicographically on tie).
///
/// Returns an empty vector for an empty graph; never returns an error.
pub fn maximum_clique(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    if g.n == 0 {
        return Ok(Vec::new());
    }
    let all = bron_kerbosch(g)?;
    if all.is_empty() {
        return Ok(Vec::new());
    }
    // Find the clique with maximum size; ties broken by earlier occurrence (lexicographic
    // order already maintained by bron_kerbosch's sort_unstable on sorted cliques).
    let best = all.into_iter().max_by_key(|c| c.len()).unwrap_or_default();
    Ok(best)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build AdjacencyList from undirected edges.
    fn undirected(n: usize, edges: &[(usize, usize)]) -> AdjacencyList {
        let mut g = AdjacencyList::new(n);
        for &(u, v) in edges {
            g.add_undirected_edge(u, v).expect("add ok");
        }
        g
    }

    // Programmatic check: every pair within each clique must be adjacent.
    fn check_cliques_genuine(g: &AdjacencyList, cliques: &[Vec<usize>]) {
        let adj = build_adj_sets(g);
        for clique in cliques {
            for i in 0..clique.len() {
                for j in (i + 1)..clique.len() {
                    assert!(
                        adj[clique[i]].contains(&clique[j]),
                        "Edge ({},{}) missing — clique {:?} is not a clique",
                        clique[i],
                        clique[j],
                        clique
                    );
                }
            }
        }
    }

    // Programmatic check: no outside vertex is adjacent to all clique members.
    fn check_cliques_maximal(g: &AdjacencyList, cliques: &[Vec<usize>]) {
        let adj = build_adj_sets(g);
        let clique_set: Vec<HashSet<usize>> = cliques
            .iter()
            .map(|c| c.iter().copied().collect())
            .collect();
        for (cidx, clique) in cliques.iter().enumerate() {
            let members = &clique_set[cidx];
            for v in 0..g.n {
                if members.contains(&v) {
                    continue;
                }
                // v adjacent to all members?
                let adj_to_all = clique.iter().all(|&u| adj[v].contains(&u));
                assert!(
                    !adj_to_all,
                    "Vertex {} can extend clique {:?} — not maximal",
                    v, clique
                );
            }
        }
    }

    // Test 1: K3 triangle.
    #[test]
    fn k3_triangle() {
        let g = undirected(3, &[(0, 1), (1, 2), (0, 2)]);
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(cliques, vec![vec![0usize, 1, 2]]);
    }

    // Test 2: K_5 complete graph.
    #[test]
    fn k5_complete() {
        let g = undirected(
            5,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 4),
            ],
        );
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(cliques, vec![vec![0usize, 1, 2, 3, 4]]);
    }

    // Test 3: Edgeless graph n=4 → 4 singletons.
    #[test]
    fn edgeless_n4() {
        let g = AdjacencyList::new(4);
        let mut cliques = bron_kerbosch(&g).expect("ok");
        cliques.sort_unstable();
        let expected: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(cliques, expected);
    }

    // Test 4: Path 0-1-2 → cliques {0,1} and {1,2}.
    #[test]
    fn path_three_nodes() {
        let g = undirected(3, &[(0, 1), (1, 2)]);
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(cliques.len(), 2);
        assert!(cliques.contains(&vec![0usize, 1]));
        assert!(cliques.contains(&vec![1usize, 2]));
    }

    // Test 5: Two disjoint triangles.
    #[test]
    fn two_disjoint_triangles() {
        let g = undirected(
            6,
            &[
                (0, 1),
                (1, 2),
                (0, 2), // first triangle
                (3, 4),
                (4, 5),
                (3, 5), // second triangle
            ],
        );
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(cliques.len(), 2);
        assert!(cliques.contains(&vec![0usize, 1, 2]));
        assert!(cliques.contains(&vec![3usize, 4, 5]));
    }

    // Test 6: C4 (4-cycle) → 4 edge-cliques.
    #[test]
    fn c4_cycle() {
        let g = undirected(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(
            cliques.len(),
            4,
            "C4 should have 4 maximal cliques, got {:?}",
            cliques
        );
        for c in &cliques {
            assert_eq!(c.len(), 2, "Each C4 clique should be an edge: {:?}", c);
        }
    }

    // Test 7: C5 (5-cycle) → 5 edge-cliques.
    #[test]
    fn c5_cycle() {
        let g = undirected(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]);
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(
            cliques.len(),
            5,
            "C5 should have 5 maximal cliques, got {:?}",
            cliques
        );
        for c in &cliques {
            assert_eq!(c.len(), 2);
        }
    }

    // Test 8: maximum_clique picks size-3 over size-2.
    #[test]
    fn maximum_clique_picks_triangle() {
        // Triangle 0-1-2 plus isolated edge 3-4.
        let g = undirected(5, &[(0, 1), (1, 2), (0, 2), (3, 4)]);
        let mc = maximum_clique(&g).expect("ok");
        assert_eq!(
            mc.len(),
            3,
            "max clique should be the triangle, got {:?}",
            mc
        );
    }

    // Test 9: All returned cliques are genuine cliques (load-bearing).
    #[test]
    fn all_cliques_genuine() {
        // Petersen-like graph with some triangles for variety.
        let g = undirected(
            7,
            &[
                (0, 1),
                (1, 2),
                (0, 2), // triangle
                (3, 4),
                (4, 5),
                (3, 5), // triangle
                (0, 3),
                (2, 5),
                (1, 6), // cross edges
            ],
        );
        let cliques = bron_kerbosch(&g).expect("ok");
        check_cliques_genuine(&g, &cliques);
    }

    // Test 10: All returned cliques are maximal (load-bearing).
    #[test]
    fn all_cliques_maximal() {
        let g = undirected(
            7,
            &[
                (0, 1),
                (1, 2),
                (0, 2),
                (3, 4),
                (4, 5),
                (3, 5),
                (0, 3),
                (2, 5),
                (1, 6),
            ],
        );
        let cliques = bron_kerbosch(&g).expect("ok");
        check_cliques_maximal(&g, &cliques);
    }

    // Test 11: Plain-pivot vs degeneracy return identical clique sets (load-bearing).
    #[test]
    fn pivot_vs_degeneracy_identical() {
        let g = undirected(
            7,
            &[
                (0, 1),
                (1, 2),
                (0, 2),
                (3, 4),
                (4, 5),
                (3, 5),
                (0, 3),
                (2, 5),
                (1, 6),
            ],
        );
        let mut c1 = bron_kerbosch(&g).expect("pivot");
        let mut c2 = bron_kerbosch_degeneracy(&g).expect("degeneracy");
        c1.sort_unstable();
        c2.sort_unstable();
        assert_eq!(
            c1, c2,
            "pivot and degeneracy must return the same clique set"
        );
    }

    // Test 12: n=1 → returns [vec![0]].
    #[test]
    fn single_vertex() {
        let g = AdjacencyList::new(1);
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(cliques, vec![vec![0usize]]);
    }

    // Test 13: n=0 → returns [].
    #[test]
    fn empty_graph() {
        let g = AdjacencyList::new(0);
        let cliques = bron_kerbosch(&g).expect("ok");
        assert!(cliques.is_empty());
    }

    // Test 14: Self-loop — algorithm does not produce {v, v} cliques.
    #[test]
    fn self_loop_no_degenerate_clique() {
        let mut g = AdjacencyList::new(3);
        // add_undirected_edge(0,0) adds only one entry (u==v branch).
        g.add_undirected_edge(0, 0).expect("self-loop ok");
        g.add_undirected_edge(0, 1).expect("ok");
        let cliques = bron_kerbosch(&g).expect("ok");
        for c in &cliques {
            // No clique should have duplicate vertices.
            let unique: HashSet<usize> = c.iter().copied().collect();
            assert_eq!(
                unique.len(),
                c.len(),
                "clique has duplicate vertex: {:?}",
                c
            );
        }
    }

    // Test 15: Determinism — same graph same result.
    #[test]
    fn deterministic() {
        let g = undirected(6, &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]);
        let r1 = bron_kerbosch(&g).expect("r1");
        let r2 = bron_kerbosch(&g).expect("r2");
        assert_eq!(r1, r2);
    }

    // Test 16: maximum_clique on empty n=0 graph → Ok(vec![]).
    #[test]
    fn maximum_clique_empty_graph() {
        let g = AdjacencyList::new(0);
        let mc = maximum_clique(&g).expect("ok");
        assert!(mc.is_empty());
    }

    // Test 17: maximum_clique on K5 → 5-element clique.
    #[test]
    fn maximum_clique_k5() {
        let g = undirected(
            5,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 4),
            ],
        );
        let mc = maximum_clique(&g).expect("ok");
        assert_eq!(mc.len(), 5);
    }

    // Test 18: maximum_clique on path → 2-element clique.
    #[test]
    fn maximum_clique_path() {
        let g = undirected(4, &[(0, 1), (1, 2), (2, 3)]);
        let mc = maximum_clique(&g).expect("ok");
        assert_eq!(mc.len(), 2);
    }

    // Test 19: Graph with known clique number 3 (three triangles sharing edges).
    #[test]
    fn clique_number_three() {
        // Three triangles sharing vertex 0: {0,1,2}, {0,2,3}, {0,3,4}
        let g = undirected(
            5,
            &[
                (0, 1),
                (0, 2),
                (1, 2), // {0,1,2}
                (0, 3),
                (2, 3), // {0,2,3}
                (0, 4),
                (3, 4), // {0,3,4}
            ],
        );
        let mc = maximum_clique(&g).expect("ok");
        assert_eq!(mc.len(), 3, "max clique size should be 3, got {:?}", mc);
    }

    // Test 20: All returned cliques are sorted ascending.
    #[test]
    fn all_cliques_sorted_ascending() {
        let g = undirected(6, &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3)]);
        let cliques = bron_kerbosch(&g).expect("ok");
        for c in &cliques {
            let sorted: Vec<usize> = {
                let mut tmp = c.clone();
                tmp.sort_unstable();
                tmp
            };
            assert_eq!(c, &sorted, "clique not sorted: {:?}", c);
        }
    }

    // Test 21: K10 complete graph → exactly 1 clique of size 10.
    #[test]
    fn k10_complete() {
        let mut g = AdjacencyList::new(10);
        for i in 0..10 {
            for j in (i + 1)..10 {
                g.add_undirected_edge(i, j).expect("ok");
            }
        }
        let cliques = bron_kerbosch(&g).expect("ok");
        assert_eq!(cliques.len(), 1, "K10 should have exactly 1 maximal clique");
        assert_eq!(cliques[0].len(), 10);
    }

    // Test 22: Petersen graph — all cliques have size ≤ 2 (no triangle).
    #[test]
    fn petersen_no_triangle() {
        // Outer pentagon: 0-1-2-3-4-0
        // Inner pentagram: 5-7-9-6-8-5
        // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        let g = undirected(
            10,
            &[
                // outer pentagon
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                // inner pentagram
                (5, 7),
                (7, 9),
                (9, 6),
                (6, 8),
                (8, 5),
                // spokes
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
            ],
        );
        let cliques = bron_kerbosch(&g).expect("ok");
        for c in &cliques {
            assert!(
                c.len() <= 2,
                "Petersen graph has no triangle; found clique of size {}: {:?}",
                c.len(),
                c
            );
        }
    }
}
