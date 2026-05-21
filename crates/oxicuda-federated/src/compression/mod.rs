//! Communication compression for federated learning.
//!
//! Reduces the communication cost of federated learning by compressing
//! gradient updates using sparsification, quantization, or low-rank
//! approximation.

pub mod powersgd;
pub mod quantize;
pub mod randomk;
pub mod signed_sgd;
pub mod sketch;
pub mod ternary;
pub mod topk;

pub use signed_sgd::{SignedSgd, SignedSgdConfig, SignedSgdState, SignedSgdUpdate};
pub use sketch::{CountSketch, CountSketchConfig, RandomHadamard};
pub use ternary::{TernaryCompressor, TernaryConfig, TernaryEncoded, TernaryMode};
