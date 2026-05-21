//! Dense singular value decomposition.
//!
//! The implementation uses one-sided Jacobi sweeps over column pairs until off-diagonals
//! of `A^T A` fall below tolerance. This converges quadratically near the limit and is
//! robust for the matrix sizes encountered in MPS bond updates (≤50×50 routinely).
//!
//! The [`mod@randomised_svd`] module provides a Halko-Martinsson-Tropp 2011 randomized SVD
//! that scales to larger matrices by first finding a low-dimensional approximate range.

pub mod randomised_svd;
pub mod svd_dense;
pub mod svd_householder;

pub use randomised_svd::{RsvdConfig, randomised_svd};
pub use svd_dense::{SvdResult, svd_jacobi, svd_jacobi_full};
pub use svd_householder::{svd_householder, svd_householder_truncated};
