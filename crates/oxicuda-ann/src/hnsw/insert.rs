use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::graph::HnswGraph;
use crate::handle::LcgRng;

/// Insert vector `v` into `graph`, returning the new node id.
pub fn hnsw_insert(graph: &mut HnswGraph, v: &[f32], rng: &mut LcgRng) -> u32 {
    let m_l = 1.0 / (graph.m as f32).ln();
    let level = (-(rng.next_f32().max(f32::EPSILON).ln()) * m_l) as usize;
    let n_layers = level + 1;

    let new_id = graph.add_node(v, n_layers);

    if graph.n_nodes() == 1 {
        graph.entry_point = Some(new_id);
        graph.max_layer = level;
        return new_id;
    }

    let ep_id = match graph.entry_point {
        Some(e) => e,
        None => {
            graph.entry_point = Some(new_id);
            return new_id;
        }
    };

    let max_layer = graph.max_layer;

    // Greedy search from top layer down to level+1 (ef=1)
    let mut ep = ep_id;
    for lc in (level + 1..=max_layer).rev() {
        let candidates = search_layer_greedy(graph, v, ep, 1, lc);
        if let Some(&(_, closest)) = candidates.first() {
            ep = closest;
        }
    }

    // From min(level, max_layer) down to 0: collect ef_construction candidates
    let connect_from = level.min(max_layer);
    for lc in (0..=connect_from).rev() {
        let ef_c = graph.ef_construction;
        let mut candidates = search_layer_greedy(graph, v, ep, ef_c, lc);
        candidates
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let cap = if lc == 0 { graph.m_max0 } else { graph.m };
        let selected = select_neighbors_heuristic(graph, v, &candidates, cap);

        graph.set_neighbors(new_id, lc, selected.clone());

        // Bidirectional links
        for &nbr in &selected {
            let mut nbr_nbrs: Vec<u32> = graph.get_neighbors(nbr, lc).to_vec();
            nbr_nbrs.push(new_id);
            let max_nbr = if lc == 0 { graph.m_max0 } else { graph.m };
            if nbr_nbrs.len() > max_nbr {
                // Prune
                let nbr_vec: Vec<f32> = graph.get_vector(nbr).to_vec();
                let cands: Vec<(f32, u32)> =
                    nbr_nbrs.iter().map(|&c| (graph.l2_sq(nbr, c), c)).collect();
                let pruned = select_neighbors_heuristic(graph, &nbr_vec, &cands, max_nbr);
                graph.set_neighbors(nbr, lc, pruned);
            } else {
                graph.set_neighbors(nbr, lc, nbr_nbrs);
            }
        }

        if !candidates.is_empty() {
            ep = candidates[0].1;
        }
    }

    if level > max_layer {
        graph.entry_point = Some(new_id);
        graph.max_layer = level;
    }

    new_id
}

/// Greedy beam search in one layer; returns `(dist_sq, node_id)` sorted ascending.
pub fn search_layer_greedy(
    graph: &HnswGraph,
    query: &[f32],
    ep: u32,
    ef: usize,
    layer: usize,
) -> Vec<(f32, u32)> {
    let ep_d = graph.l2_sq_query(query, ep);

    // candidates: min-heap by distance
    let mut candidates: BinaryHeap<Reverse<(ordered_float::OrderedF32, u32)>> = BinaryHeap::new();
    // result: max-heap by distance (bounded ef)
    let mut result: BinaryHeap<(ordered_float::OrderedF32, u32)> = BinaryHeap::new();
    let mut visited: HashSet<u32> = HashSet::new();

    candidates.push(Reverse((ordered_float::OrderedF32(ep_d), ep)));
    result.push((ordered_float::OrderedF32(ep_d), ep));
    visited.insert(ep);

    while let Some(Reverse((ordered_float::OrderedF32(c_dist), c_id))) = candidates.pop() {
        let worst = result
            .peek()
            .map_or(f32::INFINITY, |(ordered_float::OrderedF32(d), _)| *d);
        if c_dist > worst && result.len() >= ef {
            break;
        }

        for &nbr in graph.get_neighbors(c_id, layer) {
            if visited.contains(&nbr) {
                continue;
            }
            visited.insert(nbr);
            let d = graph.l2_sq_query(query, nbr);
            let worst_now = result
                .peek()
                .map_or(f32::INFINITY, |(ordered_float::OrderedF32(w), _)| *w);
            if d < worst_now || result.len() < ef {
                candidates.push(Reverse((ordered_float::OrderedF32(d), nbr)));
                result.push((ordered_float::OrderedF32(d), nbr));
                if result.len() > ef {
                    result.pop();
                }
            }
        }
    }

    let mut out: Vec<(f32, u32)> = result
        .into_iter()
        .map(|(ordered_float::OrderedF32(d), id)| (d, id))
        .collect();
    out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Distance-diverse neighbor selection heuristic.
fn select_neighbors_heuristic(
    graph: &HnswGraph,
    _query: &[f32],
    candidates: &[(f32, u32)],
    m: usize,
) -> Vec<u32> {
    let mut result: Vec<u32> = Vec::with_capacity(m);
    for &(d_cq, c) in candidates {
        if result.len() >= m {
            break;
        }
        // Accept c if it is closer to query than to any already-selected neighbor
        let closer_to_existing = result.iter().any(|&r| {
            let d_cr = graph.l2_sq(c, r);
            d_cr < d_cq
        });
        if !closer_to_existing {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{hnsw_insert, search_layer_greedy};
    use crate::handle::LcgRng;
    use crate::hnsw::graph::HnswGraph;

    fn make_graph(dim: usize, m: usize, ef: usize) -> HnswGraph {
        HnswGraph::new(dim, m, ef, ef)
    }

    /// Brute-force L2-sq nearest-neighbor list from `query` to all nodes, ascending.
    fn brute_force_sorted(graph: &HnswGraph, query: &[f32]) -> Vec<(f32, u32)> {
        let mut dists: Vec<(f32, u32)> = (0..graph.n_nodes() as u32)
            .map(|id| (graph.l2_sq_query(query, id), id))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists
    }

    #[test]
    fn insert_single_sets_entry_point() {
        let mut g = make_graph(2, 4, 16);
        let mut rng = LcgRng::new(42);
        let id = hnsw_insert(&mut g, &[0.0_f32, 0.0], &mut rng);
        assert_eq!(g.n_nodes(), 1);
        assert_eq!(id, 0);
        assert_eq!(
            g.entry_point,
            Some(0),
            "single insertion must set entry_point"
        );
    }

    #[test]
    fn insert_n_nodes_count() {
        let mut g = make_graph(2, 4, 32);
        let mut rng = LcgRng::new(99);
        for i in 0..20_u32 {
            hnsw_insert(&mut g, &[i as f32, (i * 2) as f32], &mut rng);
        }
        assert_eq!(g.n_nodes(), 20, "n_nodes must equal number of insertions");
    }

    #[test]
    fn every_node_has_at_least_layer0() {
        let mut g = make_graph(2, 4, 32);
        let mut rng = LcgRng::new(17);
        for i in 0..25_u32 {
            hnsw_insert(&mut g, &[i as f32, -(i as f32)], &mut rng);
        }
        for id in 0..g.n_nodes() as u32 {
            assert!(
                !g.layers[id as usize].is_empty(),
                "node {id} must have at least layer-0 slot"
            );
        }
    }

    #[test]
    fn degree_bound_layer0_respected() {
        let m = 4_usize;
        let mut g = HnswGraph::new(2, m, 32, 10);
        let mut rng = LcgRng::new(7);
        for i in 0..30_u32 {
            hnsw_insert(&mut g, &[i as f32 * 0.5, (30 - i) as f32 * 0.5], &mut rng);
        }
        let cap = g.m_max0;
        for id in 0..g.n_nodes() as u32 {
            let deg = g.get_neighbors(id, 0).len();
            assert!(
                deg <= cap,
                "node {id} layer-0 degree {deg} exceeds m_max0={cap}"
            );
        }
    }

    #[test]
    fn degree_bound_upper_layers_respected() {
        let m = 4_usize;
        let mut g = HnswGraph::new(2, m, 32, 10);
        let mut rng = LcgRng::new(13);
        for i in 0..40_u32 {
            hnsw_insert(&mut g, &[i as f32, 0.0], &mut rng);
        }
        for id in 0..g.n_nodes() as u32 {
            let n_layers = g.layers[id as usize].len();
            for lc in 1..n_layers {
                let deg = g.get_neighbors(id, lc).len();
                assert!(deg <= m, "node {id} layer-{lc} degree {deg} exceeds m={m}");
            }
        }
    }

    #[test]
    fn entry_point_lives_at_max_layer() {
        let mut g = make_graph(2, 4, 32);
        let mut rng = LcgRng::new(31);
        for i in 0..30_u32 {
            hnsw_insert(&mut g, &[i as f32 * 0.3, i as f32 * 0.7], &mut rng);
        }
        let ep = g
            .entry_point
            .expect("entry_point must be Some after insertions");
        let ep_layers = g.layers[ep as usize].len();
        assert!(
            ep_layers > g.max_layer,
            "entry_point id={ep} has {ep_layers} layer slots but max_layer={}",
            g.max_layer
        );
    }

    #[test]
    fn search_layer_greedy_exact_on_complete_graph() {
        // Build a complete graph at layer 0 so the greedy search has no dead ends.
        let mut g = HnswGraph::new(2, 8, 16, 4);
        let points: &[[f32; 2]] = &[
            [0.0, 0.0],  // id 0 — closest to query [0.05, 0]
            [1.0, 0.0],  // id 1
            [2.0, 0.0],  // id 2
            [10.0, 0.0], // id 3
            [11.0, 0.0], // id 4
        ];
        for p in points {
            g.add_node(p, 1);
        }
        // Connect every node to all others at layer 0.
        for i in 0..5_u32 {
            let nbrs: Vec<u32> = (0..5_u32).filter(|&j| j != i).collect();
            g.set_neighbors(i, 0, nbrs);
        }
        g.entry_point = Some(0);
        g.max_layer = 0;

        // Distances from query [0.05, 0]: 0→0.0025, 1→0.9025, 2→3.8025, 3→98.7, 4→119.4
        let q = [0.05_f32, 0.0];
        let results = search_layer_greedy(&g, &q, 0, 3, 0);

        assert!(
            !results.is_empty(),
            "search must return at least one result"
        );
        assert_eq!(
            results[0].1, 0,
            "top result must be node 0; got {:?}",
            results
        );
        assert_eq!(
            results[1].1, 1,
            "second result must be node 1; got {:?}",
            results
        );
        assert_eq!(
            results[2].1, 2,
            "third result must be node 2; got {:?}",
            results
        );
    }

    #[test]
    fn search_layer_greedy_distances_are_finite_and_sorted() {
        let mut g = make_graph(2, 4, 32);
        let mut rng = LcgRng::new(55);
        for i in 0..15_u32 {
            hnsw_insert(&mut g, &[i as f32, (15 - i) as f32], &mut rng);
        }
        let ep = g.entry_point.expect("entry_point set");
        let q = [7.0_f32, 8.0];
        let results = search_layer_greedy(&g, &q, ep, 5, 0);
        assert!(!results.is_empty());
        for &(d, _) in &results {
            assert!(
                d.is_finite() && d >= 0.0,
                "distance must be finite non-neg: {d}"
            );
        }
        for w in results.windows(2) {
            assert!(
                w[0].0 <= w[1].0,
                "results must be sorted ascending: {:?}",
                results
            );
        }
    }

    #[test]
    fn search_finds_exact_nearest_on_brute_force_comparable_graph() {
        // Build HNSW index and verify that the greedy search top-1 at layer 0
        // (starting from the entry_point) matches the brute-force nearest neighbour
        // when the clusters are well-separated.
        let m = 6_usize;
        let mut g = HnswGraph::new(2, m, 64, 10);
        let mut rng = LcgRng::new(55);

        // Two tight clusters separated by 1000 units. Within each cluster the gap is 0.1.
        // Any greedy search starting anywhere should reach the correct cluster
        // via the inter-layer descent before the layer-0 search.
        let cluster_a: Vec<[f32; 2]> = (0..8).map(|j| [j as f32 * 0.1, 0.0_f32]).collect();
        let cluster_b: Vec<[f32; 2]> = (0..8).map(|j| [1000.0_f32 + j as f32 * 0.1, 0.0]).collect();
        for p in cluster_a.iter().chain(cluster_b.iter()) {
            hnsw_insert(&mut g, p, &mut rng);
        }

        // Query deep inside cluster A; brute-force nearest is one of the first 8 nodes.
        let q = [0.35_f32, 0.0];
        let bf = brute_force_sorted(&g, &q);
        let bf_best_id = bf[0].1;
        assert!(
            bf_best_id < 8,
            "brute-force nearest to cluster-A query should be in cluster A, got {bf_best_id}"
        );

        // Do the multi-layer descent just like hnsw_insert does.
        let mut ep = g.entry_point.expect("entry_point must be set");
        for lc in (1..=g.max_layer).rev() {
            let cands = search_layer_greedy(&g, &q, ep, 1, lc);
            if let Some(&(_, closest)) = cands.first() {
                ep = closest;
            }
        }
        let results = search_layer_greedy(&g, &q, ep, 5, 0);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].1, bf_best_id,
            "layer-0 search top-1={} must match brute-force top-1={}",
            results[0].1, bf_best_id
        );
    }
}

// Minimal ordered float wrapper to use in BinaryHeap
mod ordered_float {
    #[derive(Clone, Copy, PartialEq)]
    pub struct OrderedF32(pub f32);

    impl PartialOrd for OrderedF32 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Eq for OrderedF32 {}
    impl Ord for OrderedF32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}
