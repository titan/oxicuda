//! Differential privacy primitives for federated learning.
//!
//! Provides mechanisms, accountants, and aggregation protocols that satisfy
//! (ε, δ)-differential privacy or Rényi DP guarantees.

pub mod dp_ftrl;
pub mod gaussian;
pub mod laplacian;
pub mod moments;
pub mod pate;
pub mod randomized_response;
pub mod rdp;

pub use dp_ftrl::{DpFtrl, DpFtrlConfig, DpFtrlResult, DpFtrlState};
pub use randomized_response::{RandomizedResponse, RandomizedResponseConfig};
