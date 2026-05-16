//! Matrix Product State (MPS) representation.
//!
//! An MPS represents a quantum state on `L` sites as the contraction of `L` rank-3
//! tensors `M[s]` of shape `(D_l, d, D_r)`. The left and right virtual bonds at the
//! boundary are 1 by convention. The physical dimension `d` is the local Hilbert space
//! size (e.g. `d=2` for qubits).

pub mod canonical;
pub mod mps;
pub mod tensor;
pub mod truncation;

pub use canonical::{Canonicalization, left_canonicalize, mixed_canonicalize, right_canonicalize};
pub use mps::Mps;
pub use tensor::MpsTensor;
pub use truncation::{BondTruncationResult, bond_truncate, svd_truncate};
