//! Matrix Product State (MPS) simulator.
//!
//! See [`simulator`] for the [`MatrixProductState`] type, its
//! [`MpsConfig`], and the self-contained complex SVD used for bond truncation.

pub mod simulator;

pub use simulator::{MatrixProductState, MpsConfig};
