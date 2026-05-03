//! DARTS cell: normal and reduction cells with continuous architecture parameters.
//!
//! A DARTS cell has 2 input nodes and `n_nodes` intermediate nodes.  Node `j`
//! (0-indexed) receives one edge from each of the `j + 2` preceding nodes
//! (2 inputs + j intermediates).  Each edge is a `MixedOp`.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::mixed_op::MixedOp;
use crate::ops::primitives::{OpKind, OpWeights};

// ─── DartsCell ───────────────────────────────────────────────────────────────

/// A single DARTS cell (normal or reduction).
///
/// Topology: 2 input nodes (indices 0 and 1) + `n_nodes` intermediate nodes.
/// Node `j` (0-indexed intermediate) receives edges from all `i ∈ [0, j+2)`.
/// Total edges = `Σ_{j=0}^{n_nodes-1} (j+2)`.
#[derive(Debug, Clone)]
pub struct DartsCell {
    /// Number of intermediate nodes.
    pub n_nodes: usize,
    /// Whether this is a reduction cell (stride-2).
    pub reduction: bool,
    /// Mixed operations, one per edge (in row-major order: outer=dest, inner=src).
    pub mixed_ops: Vec<MixedOp>,
}

impl DartsCell {
    /// Construct a DARTS cell.
    ///
    /// Architecture parameters are initialised from `N(0, 0.01)`.
    #[must_use]
    pub fn new(n_nodes: usize, _in_ch: usize, reduction: bool, rng: &mut LcgRng) -> Self {
        let ops = OpKind::all().to_vec();
        let n_edges = (0..n_nodes).map(|j| j + 2).sum::<usize>();
        let mut mixed_ops = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            mixed_ops.push(MixedOp::new(ops.clone(), rng));
        }
        Self {
            n_nodes,
            reduction,
            mixed_ops,
        }
    }

    /// Total number of edges in this cell.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        (0..self.n_nodes).map(|j| j + 2).sum()
    }

    /// Return a reference to the `MixedOp` for edge `(src, dest)`.
    ///
    /// `dest` is the 0-indexed intermediate node; `src` is any node with index < dest+2.
    /// Edge index is computed as `Σ_{d=0}^{dest-1} (d+2) + src`.
    pub fn edge(&self, dest: usize, src: usize) -> NasResult<&MixedOp> {
        if dest >= self.n_nodes {
            return Err(NasError::InvalidRank {
                rank: dest,
                dim: self.n_nodes,
            });
        }
        if src >= dest + 2 {
            return Err(NasError::InvalidRank {
                rank: src,
                dim: dest + 2,
            });
        }
        let base = (0..dest).map(|d| d + 2).sum::<usize>();
        let idx = base + src;
        self.mixed_ops.get(idx).ok_or(NasError::InvalidArchEncoding)
    }

    /// CPU reference forward pass.
    ///
    /// Returns the concatenation of all intermediate node outputs:
    /// `[n_nodes * in_ch * H * W]`.
    ///
    /// For a normal cell, spatial dims are preserved.
    ///
    /// # Arguments
    /// * `inputs` — exactly 2 input feature maps, each `[in_ch * H * W]`
    /// * `in_ch`, `h`, `w` — spatial dims of each input
    /// * `op_weights` — `[n_edges][n_ops]` weight tensors
    pub fn forward_cpu(
        &self,
        inputs: &[Vec<f32>],
        in_ch: usize,
        h: usize,
        w: usize,
        op_weights: &[Vec<OpWeights>],
    ) -> NasResult<Vec<f32>> {
        if inputs.len() != 2 {
            return Err(NasError::DimensionMismatch {
                expected: 2,
                got: inputs.len(),
            });
        }
        let n_edges = self.n_edges();
        if op_weights.len() != n_edges {
            return Err(NasError::DimensionMismatch {
                expected: n_edges,
                got: op_weights.len(),
            });
        }

        // All nodes: 0 = input[0], 1 = input[1], 2..= intermediate
        let mut node_outputs: Vec<Vec<f32>> = Vec::with_capacity(self.n_nodes + 2);
        node_outputs.push(inputs[0].clone());
        node_outputs.push(inputs[1].clone());

        let out_size = in_ch * h * w;
        let mut edge_idx = 0usize;

        for dest in 0..self.n_nodes {
            let n_incoming = dest + 2;
            let mut node_out = vec![0.0_f32; out_size];

            for src_feat in node_outputs.iter().take(n_incoming) {
                let mixed = &self.mixed_ops[edge_idx];
                let edge_weights = &op_weights[edge_idx];

                let op_out = mixed.forward_cpu(src_feat, in_ch, h, w, in_ch, edge_weights)?;

                if op_out.len() != out_size {
                    return Err(NasError::DimensionMismatch {
                        expected: out_size,
                        got: op_out.len(),
                    });
                }
                for (r, &o) in node_out.iter_mut().zip(op_out.iter()) {
                    *r += o;
                }
                edge_idx += 1;
            }
            node_outputs.push(node_out);
        }

        // Concatenate the last n_nodes node outputs
        let start = 2; // skip the 2 input nodes
        let mut concat = Vec::with_capacity(self.n_nodes * out_size);
        for node in &node_outputs[start..] {
            concat.extend_from_slice(node);
        }
        Ok(concat)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn darts_cell_n_edges_4_nodes() {
        let mut rng = LcgRng::new(42);
        let cell = DartsCell::new(4, 8, false, &mut rng);
        // 2+3+4+5 = 14
        assert_eq!(cell.n_edges(), 14);
        assert_eq!(cell.mixed_ops.len(), 14);
    }

    #[test]
    fn edge_accessor_valid() {
        let mut rng = LcgRng::new(1);
        let cell = DartsCell::new(4, 8, false, &mut rng);
        // dest=0, src=0 → edge 0
        assert!(cell.edge(0, 0).is_ok());
        assert!(cell.edge(0, 1).is_ok());
        // dest=0, src=2 → out of range
        assert!(cell.edge(0, 2).is_err());
    }
}
