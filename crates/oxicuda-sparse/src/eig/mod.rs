//! Sparse eigenvalue solvers.
//!
//! This module provides iterative eigensolvers for large sparse symmetric
//! matrices. Unlike the Krylov building blocks in [`ops::krylov`](crate::ops::krylov)
//! (which generate the kernels for Lanczos/Arnoldi tridiagonalisation), the
//! solvers here drive a complete host-side iteration to convergence and return
//! converged eigenpairs.
//!
//! - [`mod@dense_sym`] -- an inline cyclic-Jacobi solver for the small dense
//!   symmetric eigenproblems that arise in projected (Rayleigh-Ritz) subspaces.
//! - [`mod@lobpcg`] -- the Locally Optimal Block Preconditioned Conjugate
//!   Gradient method for the smallest eigenpairs of a sparse SPD matrix.
//! - [`mod@shift_invert`] -- shift-invert power iteration for the eigenvalue
//!   nearest a target shift (interior eigenvalues).

pub mod dense_sym;
pub mod lobpcg;
pub mod shift_invert;

pub use dense_sym::jacobi_eigh;
pub use lobpcg::{LobpcgResult, lobpcg};
pub use shift_invert::{ShiftInvertResult, shift_invert};
