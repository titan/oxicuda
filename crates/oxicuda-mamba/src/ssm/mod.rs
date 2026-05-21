//! SSM core submodules.
//!
//! - [`discretize`] — Convert continuous-time `(A, B)` to discrete `(Ā, B̄)`.
//! - [`parallel_scan`] — Associative prefix scan for efficient state computation.
//! - [`ssm_kernel`] — Full forward-pass SSM kernel operating on discrete parameters.
//! - [`hippo_variants`] — HiPPO-LegT and HiPPO-FOUT alternative polynomial projection matrices.

pub mod discretize;
pub mod hippo_variants;
pub mod parallel_scan;
pub mod ssm_kernel;

pub use hippo_variants::{
    HippoFou, HippoFouConfig, HippoLegT, HippoLegTConfig, HippoMatrix, compare_hippo_variants,
    hippo_legs_matrix,
};
