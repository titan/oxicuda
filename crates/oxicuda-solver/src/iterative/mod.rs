//! Dense iterative Krylov solvers.
//!
//! These solvers operate directly on a *dense*, row-major coefficient matrix
//! `A ∈ ℝ^{n×n}` passed as an `&[f64]` slice, returning a self-contained
//! [`gmres::GmresResult`]. They complement the matrix-free, closure-based
//! solvers in [`crate::sparse`] (which take an `spmv` operator instead of an
//! explicit matrix) and the ILU(0) preconditioner that accelerates them.
//!
//! | Module | Method | Reference |
//! |--------|--------|-----------|
//! | [`mod@gmres`]    | Restarted GMRES(m)             | Saad & Schultz 1986   |
//! | [`mod@bicgstab`] | Biconjugate Gradient Stabilized | van der Vorst 1992  |
//! | [`ilu0`]     | Incomplete LU, zero fill-in    | Saad 2003 §10.3       |
//!
//! The `gmres`/`bicgstab` configuration and result types live in their own
//! modules to avoid colliding with the identically named (but differently
//! shaped) types in [`crate::sparse`]; import them explicitly, e.g.
//! `use oxicuda_solver::iterative::gmres::{gmres, GmresConfig};`.

pub mod bicgstab;
pub mod gmres;
pub mod ilu0;

pub use bicgstab::{BicgstabConfig, bicgstab};
pub use gmres::{GmresConfig, GmresResult, gmres};
pub use ilu0::Ilu0;
