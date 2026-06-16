//! Incomplete factorization preconditioners for iterative solvers.
//!
//! Preconditioners approximate the inverse of a sparse matrix to accelerate
//! iterative methods such as conjugate gradient (CG) and GMRES.
//!
//! ## Available preconditioners
//!
//! - [`fn@ilu0`] -- Incomplete LU factorization with zero fill-in (ILU(0)).
//!   Suitable for general non-symmetric systems.
//! - [`fn@ic0`] -- Incomplete Cholesky factorization with zero fill-in (IC(0)).
//!   Suitable for symmetric positive-definite (SPD) systems.
//!
//! Both preconditioners use a level-set parallel approach: rows are grouped
//! into dependency levels, and all rows within a level are processed in
//! parallel on the GPU.

pub mod amg;
pub mod graph_coloring;
pub mod ic0;
pub mod ick;
pub mod ilu0;
pub mod iluk;

pub use amg::{AmgHierarchy, AmgLevel, AmgOptions, amg_setup, amg_solve, amg_v_cycle};
pub use graph_coloring::GraphColoring;
pub use ic0::ic0;
pub use ick::{IncompleteCholeskyK, ic_k};
pub use ilu0::ilu0;
pub use iluk::{IlukConfig, IlukFactorization};
