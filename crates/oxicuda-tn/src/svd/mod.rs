//! Dense singular value decomposition.
//!
//! The implementation uses one-sided Jacobi sweeps over column pairs until off-diagonals
//! of `A^T A` fall below tolerance. This converges quadratically near the limit and is
//! robust for the matrix sizes encountered in MPS bond updates (≤50×50 routinely).

pub mod svd_dense;

pub use svd_dense::{SvdResult, svd_jacobi, svd_jacobi_full};
