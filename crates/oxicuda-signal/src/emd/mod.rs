//! Empirical Mode Decomposition (EMD).
//!
//! Adaptive decomposition of a signal into Intrinsic Mode Functions (IMFs)
//! using iterative sifting with natural cubic spline envelope interpolation,
//! plus Hilbert transform and instantaneous frequency utilities.
pub mod emd;
pub use emd::{EmdConfig, EmdResult, emd, emd_energy, hilbert_transform, instantaneous_frequency};
