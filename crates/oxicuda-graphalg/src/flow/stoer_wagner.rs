//! Stoer-Wagner global minimum cut.
//!
//! # Overview
//!
//! The *global minimum cut* of a connected undirected weighted graph `G` is the partition
//! of its vertices into two non-empty sets `(A, B)` that minimises the total weight of
//! edges crossing between `A` and `B`. Unlike an `s`–`t` min cut, there is **no fixed pair
//! of terminals** — the algorithm finds the globally cheapest way to split the graph.
//!
//! Stoer & Wagner (1997, *A Simple Min-Cut Algorithm*) compute it **without any max-flow
//! computation** in `O(V³)` time (or `O(V·E + V²·log V)` with a heap). Each of the `n − 1`
//! *phases* performs a **maximum adjacency search**: starting from an arbitrary vertex, it
//! repeatedly adds the vertex most tightly connected to the growing set. The last vertex
//! added in a phase, `t`, together with the second-to-last, `s`, yields a *cut-of-the-phase*
//! whose value is the total weight of edges incident to `t` at that moment — this is exactly
//! a minimum `s`–`t` cut. The phase then **merges** `s` and `t` into a single super-vertex
//! and repeats. The smallest cut-of-the-phase over all phases is the global minimum cut.
//!
//! ## Why it works
//!
//! The cut-of-the-phase is a minimum `s`–`t` cut for the phase's terminals `s, t`. Any
//! global min cut either separates `s` from `t` (in which case the cut-of-the-phase is no
//! larger) or keeps them together (in which case merging them is safe). Inducting over the
//! merges proves the minimum cut-of-the-phase equals the global minimum cut.
//!
//! This implementation maintains a **dense weight matrix** that is contracted in place as
//! super-vertices are merged, mirroring the structure of the crate's [`FlowNetwork`](crate::max_flow::edmonds_karp::FlowNetwork) but
//! specialised for the symmetric, mergeable bookkeeping Stoer-Wagner needs.

use crate::error::{GraphalgError, GraphalgResult};

/// Result of a global minimum cut computation.
#[derive(Debug, Clone)]
pub struct GlobalMinCut {
    /// Weight of the global minimum cut.
    pub value: f64,
    /// One side of the cut (a non-empty proper subset of the vertices). The other side is
    /// its complement. Vertices are original indices.
    pub partition: Vec<usize>,
}

/// Compute the global minimum cut of an undirected weighted graph via Stoer-Wagner.
///
/// `edges` is an undirected weighted edge list: each `(u, v, w)` is an undirected edge of
/// non-negative weight `w`. Parallel edges sum; self-loops are ignored.
///
/// Requires `n >= 2`. The graph need not be connected: a disconnected graph has a global
/// minimum cut of weight `0` (any component versus the rest), which is reported faithfully.
///
/// # Errors
/// - [`GraphalgError::InvalidParameter`] if `n < 2`.
/// - [`GraphalgError::IndexOutOfBounds`] if any endpoint is `>= n`.
/// - [`GraphalgError::NegativeWeight`] if any edge weight is negative.
/// - [`GraphalgError::InvalidEdgeWeight`] if any edge weight is non-finite.
pub fn stoer_wagner_min_cut(
    n: usize,
    edges: &[(usize, usize, f64)],
) -> GraphalgResult<GlobalMinCut> {
    if n < 2 {
        return Err(GraphalgError::InvalidParameter(
            "global min cut needs at least 2 vertices".to_string(),
        ));
    }

    // Dense symmetric adjacency weight matrix.
    let mut w = vec![0.0f64; n * n];
    let put = |w: &mut [f64], a: usize, b: usize, x: f64| {
        w[a * n + b] += x;
        w[b * n + a] += x;
    };
    for &(u, v, weight) in edges {
        if u >= n || v >= n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: n,
            });
        }
        if !weight.is_finite() {
            return Err(GraphalgError::InvalidEdgeWeight(format!(
                "edge ({u},{v}) weight not finite: {weight}"
            )));
        }
        if weight < 0.0 {
            return Err(GraphalgError::NegativeWeight {
                edge: (u, v),
                weight,
            });
        }
        if u == v {
            continue;
        }
        put(&mut w, u, v, weight);
    }

    // `vertices[i]` is the set of original vertices represented by super-vertex slot `i`.
    let mut groups: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
    // `active[i]` indicates slot `i` is still an un-merged super-vertex.
    let mut active = vec![true; n];
    let mut remaining = n;

    let mut best_value = f64::INFINITY;
    let mut best_partition: Vec<usize> = Vec::new();

    // n - 1 phases.
    while remaining > 1 {
        // Maximum adjacency search over currently-active slots.
        // weights_into[i] = total weight from slot i into the growing set A.
        let mut weights_into = vec![0.0f64; n];
        let mut added = vec![false; n];

        let mut prev = usize::MAX; // second-to-last added (s)
        let mut last = usize::MAX; // last added (t)

        for _ in 0..remaining {
            // Pick the active, not-yet-added slot with maximum connection to A.
            let mut sel = usize::MAX;
            let mut sel_w = f64::NEG_INFINITY;
            for i in 0..n {
                if active[i] && !added[i] && weights_into[i] > sel_w {
                    sel_w = weights_into[i];
                    sel = i;
                }
            }
            // `sel` is always found because there are exactly `remaining` candidates.
            if sel == usize::MAX {
                return Err(GraphalgError::NumericalInstability(
                    "maximum adjacency search failed to select a vertex".to_string(),
                ));
            }
            added[sel] = true;
            prev = last;
            last = sel;
            // Relax: every other active not-added slot gains weight w[sel][j].
            for j in 0..n {
                if active[j] && !added[j] {
                    weights_into[j] += w[sel * n + j];
                }
            }
        }

        // Cut-of-the-phase value = weight from `last` (t) into A just before it was added,
        // which is `weights_into[last]` at the moment of its selection. Since the relax
        // step for `last` adds nothing relevant (it was the final pick), `weights_into[last]`
        // still holds its connection weight to everything else (= all of A).
        let cut_value = weights_into[last];

        if cut_value < best_value {
            best_value = cut_value;
            // The cut isolates the original vertices inside super-vertex `last`.
            best_partition = groups[last].clone();
        }

        // Merge `last` (t) into `prev` (s): s absorbs t.
        if prev == usize::MAX {
            return Err(GraphalgError::NumericalInstability(
                "phase produced no merge pair".to_string(),
            ));
        }
        for j in 0..n {
            if j != prev && j != last {
                let add = w[last * n + j];
                w[prev * n + j] += add;
                w[j * n + prev] += add;
            }
        }
        active[last] = false;
        let absorbed = std::mem::take(&mut groups[last]);
        groups[prev].extend(absorbed);
        remaining -= 1;
    }

    // Normalise the reported partition to the smaller, canonical side.
    if best_partition.len() * 2 > n {
        let in_set: Vec<bool> = {
            let mut m = vec![false; n];
            for &v in &best_partition {
                m[v] = true;
            }
            m
        };
        best_partition = (0..n).filter(|&v| !in_set[v]).collect();
    }
    best_partition.sort_unstable();

    Ok(GlobalMinCut {
        value: best_value,
        partition: best_partition,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force global min cut by enumerating all 2^(n-1) − 1 non-trivial bipartitions.
    /// Fixes vertex 0 on side A to avoid counting each cut twice. Only for tiny `n`.
    fn brute_force_min_cut(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
        let mut best = f64::INFINITY;
        // side bitmask over vertices 1..n; vertex 0 always on side A (bit clear).
        // `mask == 0` would leave side B empty (every vertex on A), which is not a valid
        // 2-partition, so we start from 1.
        for mask in 1u32..(1u32 << (n - 1)) {
            // membership: vertex 0 -> A; vertex k (k>=1) -> B iff bit (k-1) set.
            let side = |v: usize| -> bool {
                if v == 0 {
                    false
                } else {
                    (mask >> (v - 1)) & 1 == 1
                }
            };
            // Both sides are non-empty: vertex 0 is always on side A, and `mask >= 1`
            // guarantees at least one vertex on side B.
            let mut cut = 0.0;
            for &(u, v, w) in edges {
                if u == v {
                    continue;
                }
                if side(u) != side(v) {
                    cut += w;
                }
            }
            if cut < best {
                best = cut;
            }
        }
        best
    }

    fn assert_cut_value_correct(n: usize, edges: &[(usize, usize, f64)]) -> GlobalMinCut {
        let gmc = stoer_wagner_min_cut(n, edges).expect("ok");
        let bf = brute_force_min_cut(n, edges);
        assert!(
            (gmc.value - bf).abs() < 1e-6,
            "stoer-wagner={} brute={}",
            gmc.value,
            bf
        );
        // The reported partition must actually realise that value.
        let mut in_set = vec![false; n];
        for &v in &gmc.partition {
            in_set[v] = true;
        }
        assert!(!gmc.partition.is_empty(), "partition empty");
        assert!(gmc.partition.len() < n, "partition is everything");
        let mut realised = 0.0;
        for &(u, v, w) in edges {
            if u != v && in_set[u] != in_set[v] {
                realised += w;
            }
        }
        assert!(
            (realised - gmc.value).abs() < 1e-6,
            "partition realises {realised}, reported {}",
            gmc.value
        );
        gmc
    }

    #[test]
    fn canonical_stoer_wagner_example() {
        // The 8-vertex graph from Stoer & Wagner's paper; known global min cut = 4.
        let edges = [
            (0, 1, 2.0),
            (0, 4, 3.0),
            (1, 2, 3.0),
            (1, 4, 2.0),
            (1, 5, 2.0),
            (2, 3, 4.0),
            (2, 6, 2.0),
            (3, 6, 2.0),
            (3, 7, 2.0),
            (4, 5, 3.0),
            (5, 6, 1.0),
            (6, 7, 3.0),
        ];
        let gmc = assert_cut_value_correct(8, &edges);
        assert!((gmc.value - 4.0).abs() < 1e-9, "value={}", gmc.value);
    }

    #[test]
    fn triangle_unit_weights() {
        // Triangle: every cut isolates one vertex, value 2.
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let gmc = assert_cut_value_correct(3, &edges);
        assert!((gmc.value - 2.0).abs() < 1e-9);
    }

    #[test]
    fn path_graph_cut_is_lightest_edge() {
        // Path 0-1-2-3 with weights 5,1,5 → min cut = 1 (the middle edge).
        let edges = [(0, 1, 5.0), (1, 2, 1.0), (2, 3, 5.0)];
        let gmc = assert_cut_value_correct(4, &edges);
        assert!((gmc.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn single_bridge_min_cut() {
        // Two triangles joined by one light bridge → the bridge is the global min cut.
        let edges = [
            (0, 1, 10.0),
            (1, 2, 10.0),
            (0, 2, 10.0),
            (3, 4, 10.0),
            (4, 5, 10.0),
            (3, 5, 10.0),
            (2, 3, 1.0), // bridge
        ];
        let gmc = assert_cut_value_correct(6, &edges);
        assert!((gmc.value - 1.0).abs() < 1e-9, "value={}", gmc.value);
        // The partition must separate {0,1,2} from {3,4,5}.
        let mut in_set = vec![false; 6];
        for &v in &gmc.partition {
            in_set[v] = true;
        }
        assert_eq!(in_set[0], in_set[1]);
        assert_eq!(in_set[1], in_set[2]);
        assert_ne!(in_set[2], in_set[3]);
    }

    #[test]
    fn disconnected_graph_zero_cut() {
        // {0-1} and {2-3} disjoint → global min cut = 0.
        let edges = [(0, 1, 7.0), (2, 3, 9.0)];
        let gmc = stoer_wagner_min_cut(4, &edges).expect("ok");
        assert!(gmc.value.abs() < 1e-9, "value={}", gmc.value);
    }

    #[test]
    fn two_vertices_single_edge() {
        let edges = [(0, 1, 3.5)];
        let gmc = stoer_wagner_min_cut(2, &edges).expect("ok");
        assert!((gmc.value - 3.5).abs() < 1e-9);
        assert_eq!(gmc.partition.len(), 1);
    }

    #[test]
    fn two_vertices_no_edge() {
        let gmc = stoer_wagner_min_cut(2, &[]).expect("ok");
        assert!(gmc.value.abs() < 1e-9);
    }

    #[test]
    fn parallel_edges_sum() {
        // Two parallel edges 0-1 (2.0 and 4.0); single cut weight 6.
        let edges = [(0, 1, 2.0), (0, 1, 4.0)];
        let gmc = stoer_wagner_min_cut(2, &edges).expect("ok");
        assert!((gmc.value - 6.0).abs() < 1e-9, "value={}", gmc.value);
    }

    #[test]
    fn self_loops_ignored() {
        let edges = [(0, 0, 100.0), (0, 1, 2.0), (1, 1, 50.0)];
        let gmc = stoer_wagner_min_cut(2, &edges).expect("ok");
        assert!((gmc.value - 2.0).abs() < 1e-9);
    }

    #[test]
    fn complete_graph_k4_unit() {
        // K4 unit weights: min cut isolates one vertex → degree 3.
        let edges = [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ];
        let gmc = assert_cut_value_correct(4, &edges);
        assert!((gmc.value - 3.0).abs() < 1e-9);
    }

    #[test]
    fn weighted_random_small_matches_brute_force() {
        // A handful of fixed weighted graphs, all cross-checked against brute force.
        type Case<'a> = (usize, &'a [(usize, usize, f64)]);
        let cases: &[Case<'_>] = &[
            (
                5,
                &[
                    (0, 1, 2.0),
                    (1, 2, 3.0),
                    (2, 3, 1.0),
                    (3, 4, 4.0),
                    (0, 4, 2.0),
                    (1, 3, 5.0),
                ],
            ),
            (
                6,
                &[
                    (0, 1, 3.0),
                    (0, 2, 1.0),
                    (1, 2, 2.0),
                    (2, 3, 4.0),
                    (3, 4, 1.0),
                    (4, 5, 3.0),
                    (3, 5, 2.0),
                    (1, 4, 1.0),
                ],
            ),
            (
                5,
                &[
                    (0, 1, 6.0),
                    (0, 2, 6.0),
                    (1, 2, 6.0),
                    (2, 3, 1.0),
                    (3, 4, 6.0),
                    (1, 4, 1.0),
                ],
            ),
        ];
        for &(n, edges) in cases {
            assert_cut_value_correct(n, edges);
        }
    }

    #[test]
    fn star_graph_min_cut_is_lightest_spoke() {
        // Star with spoke weights 1,2,3 → min cut = 1 (isolate the leaf with weight 1).
        let edges = [(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0)];
        let gmc = assert_cut_value_correct(4, &edges);
        assert!((gmc.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_too_few_vertices() {
        assert!(matches!(
            stoer_wagner_min_cut(1, &[]),
            Err(GraphalgError::InvalidParameter(_))
        ));
        assert!(matches!(
            stoer_wagner_min_cut(0, &[]),
            Err(GraphalgError::InvalidParameter(_))
        ));
    }

    #[test]
    fn rejects_negative_weight() {
        assert!(matches!(
            stoer_wagner_min_cut(3, &[(0, 1, -1.0)]),
            Err(GraphalgError::NegativeWeight { .. })
        ));
    }

    #[test]
    fn rejects_oob_endpoint() {
        assert!(matches!(
            stoer_wagner_min_cut(3, &[(0, 9, 1.0)]),
            Err(GraphalgError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_weight() {
        assert!(matches!(
            stoer_wagner_min_cut(3, &[(0, 1, f64::INFINITY)]),
            Err(GraphalgError::InvalidEdgeWeight(_))
        ));
    }

    #[test]
    fn partition_is_smaller_side() {
        // 4-clique plus a pendant; the pendant alone is the min cut, partition = {pendant}.
        let edges = [
            (0, 1, 5.0),
            (0, 2, 5.0),
            (0, 3, 5.0),
            (1, 2, 5.0),
            (1, 3, 5.0),
            (2, 3, 5.0),
            (3, 4, 1.0),
        ];
        let gmc = stoer_wagner_min_cut(5, &edges).expect("ok");
        assert!((gmc.value - 1.0).abs() < 1e-9);
        assert_eq!(gmc.partition, vec![4]);
    }
}
