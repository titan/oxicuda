//! Search space definitions for NAS: cell topology and network structure.

use crate::error::{NasError, NasResult};
use crate::ops::primitives::OpKind;

// ─── CellSpace ───────────────────────────────────────────────────────────────

/// Search space for a single cell.
///
/// In DARTS, each cell has `n_nodes` intermediate nodes with 2 extra input nodes,
/// giving `n_nodes * (n_nodes + 3) / 2` directed edges total
/// (each node `j ∈ [0, n_nodes)` receives edges from all `i < j+2` nodes).
#[derive(Debug, Clone)]
pub struct CellSpace {
    /// Number of intermediate nodes (default 4 in DARTS).
    pub n_nodes: usize,
    /// Candidate operations per edge.
    pub op_kinds: Vec<OpKind>,
}

impl CellSpace {
    /// Create the default DARTS cell search space: 4 nodes, 8 ops.
    #[must_use]
    pub fn darts_default() -> Self {
        Self {
            n_nodes: 4,
            op_kinds: OpKind::all().to_vec(),
        }
    }

    /// Validate the cell space parameters.
    pub fn validate(&self) -> NasResult<()> {
        if self.n_nodes < 1 {
            return Err(NasError::InvalidNumNodes {
                min: 1,
                got: self.n_nodes,
            });
        }
        if self.op_kinds.is_empty() {
            return Err(NasError::InvalidNumOps);
        }
        Ok(())
    }

    /// Number of edges in the cell: each intermediate node `j` receives edges
    /// from all `j + 2` preceding nodes (2 inputs + `j` intermediates).
    #[must_use]
    pub fn n_edges(&self) -> usize {
        // node 0 ← inputs 0, 1           → 2 edges
        // node 1 ← inputs 0, 1, node 0   → 3 edges
        // ...
        // node j ← j+2 inputs
        (0..self.n_nodes).map(|j| j + 2).sum()
    }

    /// Number of candidate ops per edge.
    #[must_use]
    pub fn n_ops(&self) -> usize {
        self.op_kinds.len()
    }
}

// ─── SearchSpace ─────────────────────────────────────────────────────────────

/// Full network search space combining cell and network hyperparameters.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Cell topology.
    pub cell: CellSpace,
    /// Number of layers (stacked cells) in the network.
    pub n_layers: usize,
    /// Initial number of channels after the stem convolution.
    pub init_channels: usize,
}

impl SearchSpace {
    /// Construct with validation.
    pub fn new(cell: CellSpace, n_layers: usize, init_channels: usize) -> NasResult<Self> {
        cell.validate()?;
        if n_layers == 0 {
            return Err(NasError::InvalidNumNodes { min: 1, got: 0 });
        }
        if init_channels == 0 {
            return Err(NasError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            cell,
            n_layers,
            init_channels,
        })
    }

    /// DARTS default: 8 layers, 16 init channels, 4-node cell, 8 ops.
    #[must_use]
    pub fn darts_default() -> Self {
        Self {
            cell: CellSpace::darts_default(),
            n_layers: 8,
            init_channels: 16,
        }
    }
}

// ─── NetworkSpace ─────────────────────────────────────────────────────────────

/// Macro-architecture space: how cells are arranged into a network.
#[derive(Debug, Clone)]
pub struct NetworkSpace {
    /// Positions (layer indices) of reduction cells (all others are normal).
    pub reduction_positions: Vec<usize>,
    /// Total number of layers.
    pub n_layers: usize,
    /// Number of output classes for the final classifier.
    pub n_classes: usize,
}

impl NetworkSpace {
    /// DARTS default: reduction at layers 2 and 5 (out of 8).
    #[must_use]
    pub fn darts_default(n_layers: usize, n_classes: usize) -> Self {
        let third = n_layers / 3;
        let two_thirds = 2 * n_layers / 3;
        Self {
            reduction_positions: vec![third, two_thirds],
            n_layers,
            n_classes,
        }
    }

    /// Return true if layer `i` is a reduction cell.
    #[must_use]
    pub fn is_reduction(&self, layer: usize) -> bool {
        self.reduction_positions.contains(&layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_space_n_edges_4_nodes() {
        let cs = CellSpace::darts_default();
        // nodes 0..4: edges = 2+3+4+5 = 14
        assert_eq!(cs.n_edges(), 14);
    }

    #[test]
    fn search_space_validate_ok() {
        let ss = SearchSpace::darts_default();
        assert_eq!(ss.n_layers, 8);
        assert_eq!(ss.init_channels, 16);
    }

    #[test]
    fn network_space_reduction_positions() {
        let ns = NetworkSpace::darts_default(8, 10);
        assert!(ns.is_reduction(2));
        assert!(ns.is_reduction(5));
        assert!(!ns.is_reduction(0));
    }
}
