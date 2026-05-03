//! Derive a discrete architecture from continuous DARTS arch parameters.
//!
//! For each intermediate node, the top-2 incoming edges (by max softmax weight,
//! excluding `Zero`) are selected.  If a node has no non-Zero ops with positive
//! weight the function returns `Err(NoFeasibleArchitecture)`.

use crate::darts::cell::DartsCell;
use crate::error::{NasError, NasResult};
use crate::ops::mixed_op::softmax;
use crate::ops::primitives::OpKind;

// ─── DiscretizedCell ─────────────────────────────────────────────────────────

/// A cell architecture derived from continuous DARTS arch params.
#[derive(Debug, Clone)]
pub struct DiscretizedCell {
    /// For each intermediate node: a list of `(source_node_idx, OpKind)`.
    pub edges: Vec<Vec<(usize, OpKind)>>,
    /// Whether this was a reduction cell.
    pub reduction: bool,
}

/// Derive a discrete cell from a continuous `DartsCell`.
///
/// For each intermediate node `j`:
/// 1. Collect all incoming edges (each has `n_ops` arch params).
/// 2. For each edge compute `softmax(α)` and find the best non-Zero op.
/// 3. Rank incoming edges by their best non-Zero op weight; keep top-2.
/// 4. If all edges have Zero as best op → return `NoFeasibleArchitecture`.
pub fn derive_discrete_cell(cell: &DartsCell) -> NasResult<DiscretizedCell> {
    let mut node_edges: Vec<Vec<(usize, OpKind)>> = Vec::with_capacity(cell.n_nodes);
    let mut edge_idx = 0usize;

    for dest in 0..cell.n_nodes {
        let n_incoming = dest + 2;

        // Collect (src, best_non_zero_weight, best_non_zero_op) for each incoming edge
        let mut edge_scores: Vec<(usize, f32, OpKind)> = Vec::with_capacity(n_incoming);

        for src in 0..n_incoming {
            let mixed = &cell.mixed_ops[edge_idx];
            let ws = softmax(&mixed.arch_params);

            // Find best non-Zero op
            let best = mixed
                .op_kinds
                .iter()
                .zip(ws.iter())
                .filter(|&(&op, _)| op != OpKind::Zero)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let (best_weight, best_op) = match best {
                Some((&op, &w)) => (w, op),
                None => {
                    // all ops are Zero — treat weight as 0
                    (0.0, OpKind::Zero)
                }
            };
            edge_scores.push((src, best_weight, best_op));
            edge_idx += 1;
        }

        // Sort by weight descending, pick top-2
        edge_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top2: Vec<(usize, OpKind)> = edge_scores
            .iter()
            .take(2)
            .filter(|(_, w, op)| *w > 0.0 || *op != OpKind::Zero)
            .map(|&(src, _, op)| (src, op))
            .collect();

        if top2.is_empty() {
            return Err(NasError::NoFeasibleArchitecture);
        }
        node_edges.push(top2);
    }

    Ok(DiscretizedCell {
        edges: node_edges,
        reduction: cell.reduction,
    })
}

// ─── DiscretizedNetwork ──────────────────────────────────────────────────────

/// A discretized network architecture.
#[derive(Debug, Clone)]
pub struct DiscretizedNetwork {
    /// Discrete normal cell template.
    pub normal_cell: DiscretizedCell,
    /// Discrete reduction cell template.
    pub reduction_cell: DiscretizedCell,
    /// Number of layers.
    pub n_layers: usize,
    /// Initial channel count.
    pub init_channels: usize,
}

/// Derive a discrete network from two continuous DARTS cells (normal + reduction).
pub fn derive_network(
    normal: &DartsCell,
    reduction: &DartsCell,
    n_layers: usize,
    init_channels: usize,
) -> NasResult<DiscretizedNetwork> {
    let normal_cell = derive_discrete_cell(normal)?;
    let reduction_cell = derive_discrete_cell(reduction)?;
    Ok(DiscretizedNetwork {
        normal_cell,
        reduction_cell,
        n_layers,
        init_channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn derive_cell_has_edges_per_node() {
        let mut rng = LcgRng::new(42);
        let cell = DartsCell::new(4, 8, false, &mut rng);
        let disc = derive_discrete_cell(&cell).expect("test invariant: derive cell");
        assert_eq!(disc.edges.len(), 4);
        for edges in &disc.edges {
            assert!(!edges.is_empty(), "each node must have at least one edge");
        }
    }

    #[test]
    fn derive_network_ok() {
        let mut rng = LcgRng::new(7);
        let normal = DartsCell::new(4, 8, false, &mut rng);
        let reduction = DartsCell::new(4, 8, true, &mut rng);
        let net =
            derive_network(&normal, &reduction, 8, 16).expect("test invariant: derive network");
        assert_eq!(net.n_layers, 8);
    }
}
