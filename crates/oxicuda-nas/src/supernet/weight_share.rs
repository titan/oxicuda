//! Weight-sharing supernet: all paths share a single set of weights.
//!
//! The supernet is a one-shot model where each op in each edge shares its
//! weights regardless of which path is sampled.  During inference, only the
//! sampled path is activated, but the same weight tensors are used for all ops.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::primitives::{OpKind, OpWeights};

// ─── Supernet ─────────────────────────────────────────────────────────────────

/// One-shot supernet with shared weights for all candidate operations on each edge.
///
/// Layout:
/// * `shared_weights[edge][op]` — one `OpWeights` per (edge, op) pair.
/// * `n_edges` — total number of edges across all cells.
/// * `n_ops` — number of candidate operations per edge.
#[derive(Debug, Clone)]
pub struct Supernet {
    /// Shared weight pool: `[n_edges][n_ops]`.
    pub shared_weights: Vec<Vec<OpWeights>>,
    /// Number of edges.
    pub n_edges: usize,
    /// Number of candidate ops per edge.
    pub n_ops: usize,
    /// Input channel count.
    pub in_ch: usize,
    /// Kernel size used to size convolutional weights.
    pub kernel: usize,
}

impl Supernet {
    /// Build a supernet with randomly initialised shared weights.
    ///
    /// # Arguments
    /// * `n_edges` — total number of edges
    /// * `n_ops` — candidate ops per edge
    /// * `in_ch`, `out_ch` — channel dimensions
    /// * `kernel` — max kernel size (5 for SepConv5x5)
    #[must_use]
    pub fn new(
        n_edges: usize,
        n_ops: usize,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let mut shared_weights = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let mut edge_weights = Vec::with_capacity(n_ops);
            for _ in 0..n_ops {
                edge_weights.push(OpWeights::random(in_ch, out_ch, kernel, rng));
            }
            shared_weights.push(edge_weights);
        }
        Self {
            shared_weights,
            n_edges,
            n_ops,
            in_ch,
            kernel,
        }
    }

    /// Forward pass through a single edge using the shared weights for a given op.
    ///
    /// # Arguments
    /// * `edge_idx` — which edge to use
    /// * `op_idx` — which op to use (index into the op list)
    /// * `op_kind` — the operation variant
    /// * `input`, `in_ch`, `h`, `w`, `out_ch` — feature map layout
    pub fn forward_edge(
        &self,
        edge_idx: usize,
        op_idx: usize,
        op_kind: OpKind,
        input: &[f32],
        in_ch: usize,
        h: usize,
        w: usize,
        out_ch: usize,
    ) -> NasResult<Vec<f32>> {
        let edge_weights = self
            .shared_weights
            .get(edge_idx)
            .ok_or(NasError::InvalidRank {
                rank: edge_idx,
                dim: self.n_edges,
            })?;
        let op_weights = edge_weights.get(op_idx).ok_or(NasError::InvalidRank {
            rank: op_idx,
            dim: self.n_ops,
        })?;
        op_kind.forward_cpu(input, in_ch, h, w, out_ch, op_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supernet_forward_edge_ok() {
        let mut rng = LcgRng::new(42);
        let sn = Supernet::new(14, 8, 4, 4, 5, &mut rng);
        let input = vec![0.5_f32; 4 * 8 * 8];
        let out = sn
            .forward_edge(0, 1, OpKind::Identity, &input, 4, 8, 8, 4)
            .expect("test invariant: supernet forward");
        assert_eq!(out.len(), 4 * 8 * 8);
    }

    #[test]
    fn supernet_invalid_edge_errors() {
        let mut rng = LcgRng::new(1);
        let sn = Supernet::new(2, 4, 4, 4, 3, &mut rng);
        let input = vec![1.0_f32; 4 * 4 * 4];
        let result = sn.forward_edge(99, 0, OpKind::Zero, &input, 4, 4, 4, 4);
        assert!(result.is_err());
    }
}
