//! Multidimensional Scaling (MDS).
//!
//! - [`classical_mds()`] Torgerson's classical (metric) MDS via double centering + eigendecomp.
//! - [`smacof`] Stress majorization (SMACOF) for metric MDS.
//! - [`mod@nonmetric_mds`] Non-metric (ordinal) MDS via isotonic regression (Kruskal, 1964).

pub mod classical_mds;
pub mod nonmetric_mds;
pub mod smacof;

pub use classical_mds::{ClassicalMdsResult, classical_mds};
pub use nonmetric_mds::{NonmetricMdsResult, nonmetric_mds, pava};
pub use smacof::{SmacofResult, smacof_mds};
