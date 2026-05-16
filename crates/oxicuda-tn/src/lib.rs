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

pub mod contraction;
pub mod cp;
pub mod dmrg;
pub mod error;
pub mod handle;
pub mod metrics;
pub mod mpo;
pub mod mps;
pub mod peps;
pub mod ptx_kernels;
pub mod svd;
pub mod tebd;
pub mod tt;
pub mod tucker;

pub use error::{TnError, TnResult};
pub use handle::{LcgRng, SmVersion, TnHandle};

#[cfg(test)]
mod e2e_tests;
