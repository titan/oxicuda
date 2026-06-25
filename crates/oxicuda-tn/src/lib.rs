//! `oxicuda-tn` — Tensor Networks for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-tn
//! ├── svd/          — Dense Jacobi SVD (foundation for canonicalization, truncation)
//! ├── mps/          — Matrix Product State: tensors, canonicalization, truncation
//! ├── mpo/          — Matrix Product Operator and MPO·MPS contraction
//! ├── dmrg/         — Two-site DMRG ground-state solver with Lanczos
//! ├── tebd/         — Time-Evolving Block Decimation with Suzuki-Trotter splittings
//! ├── peps/         — 2D Projected Entangled Pair States with boundary-MPS contraction
//! ├── tt/           — Tensor-Train (Oseledets) decomposition: TT-SVD and TT-cross
//! ├── tucker/       — HOSVD and HOOI Tucker decompositions
//! ├── cp/           — CP / PARAFAC decomposition via alternating least squares
//! ├── contraction/  — Generic einsum and greedy contraction-path optimisation
//! └── metrics/      — Bond dimensions, entanglement entropy, Schmidt spectrum, fidelity
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod blocksparse;
pub mod circuits;
pub mod contraction;
pub mod cp;
pub mod dmrg;
pub mod error;
pub mod handle;
pub mod mera;
pub mod metrics;
pub mod mpo;
pub mod mps;
pub mod optim;
pub mod peps;
pub mod ptx_kernels;
pub mod svd;
pub mod tebd;
pub mod tr;
pub mod tree;
pub mod trg;
pub mod tt;
pub mod tucker;

pub use blocksparse::{BlockKey, BlockSparseTensor};
pub use circuits::{Circuit, CircuitConfig, CircuitGate, compile_circuit_to_tebd_gates};
pub use contraction::path_optimal::{
    ContractionPathConfig, DpEntry, OptimalPath, TensorSpec, build_index_dims, compare_with_greedy,
    contraction_flops, contraction_result_indices, greedy_flops, optimal_contraction_path,
};
pub use error::{TnError, TnResult};
pub use handle::{LcgRng, SmVersion, TnHandle};
pub use mera::MeraLayer;
pub use metrics::{
    LoschmidtConfig, LoschmidtResult, ReturnProbResult, StructureFactorConfig,
    StructureFactorResult, SzOperator, loschmidt_echo, mpo_expectation_value,
    mps_inner_product as loschmidt_mps_inner_product, operator_matrix, return_probability,
    static_structure_factor,
};
pub use mps::isometry_tn::{
    FatMpsColumn, FatTensor, IsoTnsTensor, IsometryTn, MosesMoveResult, TripartiteSplit,
    moses_move_column, tripartite_split,
};
pub use mps::symmetric::{
    Qn, QnBlock, SymMps, SymMpsConfig, SymMpsTensor, block_svd, sym_mps_left_canonicalize,
    sym_mps_local_expectation, sym_mps_norm, sym_mps_random, sym_mps_to_dense,
};
pub use optim::{
    FixedRankManifold, RiemannianTn, RiemannianTnConfig, RiemannianTnMethod, TnPoint, TnResultData,
    eckart_young_objective, low_rank_completion_egrad, low_rank_completion_objective,
    low_rank_egrad, low_rank_objective,
};
pub use peps::ctmrg::{
    CtmrgConfig, CtmrgEnv, CtmrgResult, ctmrg_expectation, ctmrg_init, ctmrg_norm_per_site,
    ctmrg_run, ctmrg_step_down, ctmrg_step_right,
};
pub use peps::simple_update::PepsTensor as SuPepsTensor;
pub use peps::simple_update::{
    PepsLambdas, SimpleUpdateConfig, SimpleUpdateResult, simple_update_energy, simple_update_init,
    simple_update_run, simple_update_step_h, simple_update_step_v,
};
pub use tr::{TrCore, TrTensor, tr_svd};
pub use tree::{TreeTensorNetwork, TtnNode};
pub use trg::ising::{ising_tensor, onsager_log_z_per_site};
pub use trg::{LatticeTensor, trg_partition_log, trg_step};

#[cfg(test)]
mod e2e_tests;
