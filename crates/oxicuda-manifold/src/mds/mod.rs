//! Multidimensional Scaling (MDS).
//!
//! - [`classical_mds()`] Torgerson's classical (metric) MDS via double centering + eigendecomp.
//! - [`smacof`] Stress majorization (SMACOF) for metric MDS.

pub mod classical_mds;
pub mod smacof;

pub use classical_mds::{ClassicalMdsResult, classical_mds};
pub use smacof::{SmacofResult, smacof_mds};
