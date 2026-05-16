//! Tensor-network diagnostics: bond dimensions, entanglement entropy, Schmidt
//! spectrum, and fidelity.

pub mod metrics;

pub use metrics::{
    bond_dimension, entanglement_entropy, fidelity, max_bond_dimension, mps_overlap,
    schmidt_spectrum,
};
