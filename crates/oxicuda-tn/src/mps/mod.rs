//! Matrix Product State (MPS) representation.
//!
//! An MPS represents a quantum state on `L` sites as the contraction of `L` rank-3
//! tensors `M[s]` of shape `(D_l, d, D_r)`. The left and right virtual bonds at the
//! boundary are 1 by convention. The physical dimension `d` is the local Hilbert space
//! size (e.g. `d=2` for qubits).
//!
//! Also exposes [`itebd`]: infinite-system TEBD in the Vidal Γ–Λ canonical form for
//! translationally-invariant 1D systems, and [`isometry_tn`]: 2D Isometric Tensor
//! Network states (isoTNS) with the Moses Move (Zaletel-Pollmann 2020).

pub mod canonical;
pub mod isometry_tn;
pub mod itebd;
pub mod mps;
pub mod symmetric;
pub mod tensor;
pub mod truncation;

pub use canonical::{Canonicalization, left_canonicalize, mixed_canonicalize, right_canonicalize};
pub use isometry_tn::{
    FatMpsColumn, FatTensor, IsoTnsTensor, IsometryTn, MosesMoveResult, TripartiteSplit,
    moses_move_column, tripartite_split,
};
pub use itebd::{
    ItedbConfig, ItedbResult, ItedbState, heisenberg_hamiltonian_2site, itebd_energy,
    itebd_heisenberg, itebd_run, mat_exp_4x4,
};
pub use mps::Mps;
pub use symmetric::{
    Qn, QnBlock, SymMps, SymMpsConfig, SymMpsTensor, block_svd, sym_mps_left_canonicalize,
    sym_mps_local_expectation, sym_mps_norm, sym_mps_random, sym_mps_to_dense,
};
pub use tensor::MpsTensor;
pub use truncation::{BondTruncationResult, bond_truncate, svd_truncate};
