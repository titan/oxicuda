//! SSM core submodules.
//!
//! - [`discretize`] — Convert continuous-time `(A, B)` to discrete `(Ā, B̄)`.
//! - [`parallel_scan`] — Associative prefix scan for efficient state computation.
//! - [`ssm_kernel`] — Full forward-pass SSM kernel operating on discrete parameters.

pub mod discretize;
pub mod parallel_scan;
pub mod ssm_kernel;
