//! Iterative sparse linear solvers.
//!
//! Provides matrix-free iterative methods for solving large sparse linear systems
//! `A * x = b`. The solvers accept a closure `spmv: F` that computes the
//! sparse matrix-vector product `y = A * x`, enabling use with any sparse
//! matrix format or even matrix-free operators.
//!
//! # Solvers
//!
//! - **CG** ([`cg`]): Conjugate Gradient for symmetric positive definite systems.
//! - **MINRES** ([`minres`]): Minimal Residual for symmetric (possibly indefinite) systems.
//! - **BiCGSTAB** ([`bicgstab`]): Biconjugate Gradient Stabilized for non-symmetric systems.
//! - **QMR** ([`qmr`]): Quasi-Minimal Residual for non-symmetric systems (needs `Aᵀ`).
//! - **GMRES(m)** ([`gmres`]): Generalized Minimal Residual with restart for general systems.
//! - **LSQR** ([`lsqr`]): Iterative least-squares `min ‖A·x − b‖₂` for rectangular `A`.
//! - **Direct** ([`direct`]): Direct sparse solver via dense LU (for small-to-medium systems).
//!
//! # Sparse direct factorizations
//!
//! - **Left-looking LU** ([`superlu_left_looking`]): Gilbert–Peierls / SuperLU
//!   column-by-column sparse LU with partial pivoting and supernode detection.
//! - **PARDISO-compatible** ([`pardiso_compat`]): phased (analysis / factorize /
//!   solve) sparse direct solver with nested-dissection reordering.

pub mod bicgstab;
pub mod cg;
pub mod direct;
pub mod direct_factorization;
pub mod fgmres;
pub mod gmres;
pub mod lsqr;
pub mod minres;
pub mod nested_dissection;
pub mod pardiso_compat;
pub mod preconditioned;
pub mod qmr;
pub mod superlu_left_looking;

pub use bicgstab::{BiCgStabConfig, bicgstab_solve};
pub use cg::{CgConfig, cg_solve};
pub use direct::prefer_direct_solver;
pub use direct_factorization::{
    EliminationTree, MultifrontalLUSolver, SupernodalCholeskySolver, SupernodalStructure,
    SymbolicFactorization, sparse_cholesky_solve, sparse_lu_solve,
};
pub use fgmres::{FgmresConfig, fgmres};
pub use gmres::{GmresConfig, gmres_solve};
pub use lsqr::{LsqrConfig, lsqr_solve};
pub use minres::{MinresConfig, minres_solve};
pub use nested_dissection::{
    AdjacencyGraph, NestedDissectionOrdering, OrderingQuality, Permutation,
};
pub use pardiso_compat::{PardisoCompatSolver, Phase, pardiso_solve};
pub use preconditioned::{
    IdentityPreconditioner, IterativeSolverResult, JacobiPreconditioner, PcgConfig, PgmresConfig,
    Preconditioner, preconditioned_cg, preconditioned_gmres,
};
pub use qmr::{QmrConfig, qmr_solve};
pub use superlu_left_looking::{LeftLookingLu, left_looking_lu_solve};
