//! Differential privacy primitives for federated learning.
//!
//! Provides mechanisms, accountants, and aggregation protocols that satisfy
//! (ε, δ)-differential privacy or Rényi DP guarantees.

pub mod gaussian;
pub mod laplacian;
pub mod moments;
pub mod pate;
pub mod rdp;
