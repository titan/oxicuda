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
