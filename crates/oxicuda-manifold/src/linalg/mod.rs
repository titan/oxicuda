//! Linear-algebra primitives for `oxicuda-manifold`.
//!
//! All routines operate on row-major `Vec<f64>` matrices.
//! - [`jacobi_eig`] cyclic Jacobi eigendecomposition for symmetric matrices.
//! - [`power_iter`] dominant eigenpair extraction.
//! - [`lanczos`] Krylov tridiagonalization for extreme eigenvalues.
//! - [`householder_qr()`] reduced QR factorization.

pub mod householder_qr;
pub mod jacobi_eig;
pub mod lanczos;
pub mod power_iter;

pub use householder_qr::{householder_qr, polar_orthogonal, solve_lower_triangular};
pub use jacobi_eig::{jacobi_eigh, sort_eigen_descending};
pub use lanczos::{LanczosResult, lanczos_smallest_eig};
pub use power_iter::{power_iteration, power_iteration_deflated};
