use super::graph::HnswGraph;
use super::insert::search_layer_greedy;
use crate::error::{AnnError, AnnResult};

/// Search the HNSW graph for the `k` approximate nearest neighbors of `query`.
pub fn hnsw_search(graph: &HnswGraph, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
    if graph.n_nodes() == 0 {
        return Err(AnnError::IndexEmpty);
    }
    if query.len() != graph.dim {
        return Err(AnnError::DimensionMismatch {
            expected: graph.dim,
            got: query.len(),
        });
    }

    let ep = match graph.entry_point {
        Some(e) => e,
        None => return Err(AnnError::IndexEmpty),
    };

    // Greedy descent from top layer to layer 1 (ef=1)
    let mut cur_ep = ep;
    for lc in (1..=graph.max_layer).rev() {
        let candidates = search_layer_greedy(graph, query, cur_ep, 1, lc);
        if let Some(&(_, closest)) = candidates.first() {
            cur_ep = closest;
        }
    }

    // Layer 0: beam search with ef candidates
    let ef = graph.ef.max(k);
    let candidates = search_layer_greedy(graph, query, cur_ep, ef, 0);

    let actual_k = k.min(candidates.len());
    if actual_k == 0 {
        return Err(AnnError::InvalidK {
            k,
            n: graph.n_nodes(),
        });
    }

    let result: Vec<(u32, f32)> = candidates
        .into_iter()
        .take(actual_k)
        .map(|(d, id)| (id, d))
        .collect();

    Ok(result)
}
