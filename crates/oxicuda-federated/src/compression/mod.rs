//! Communication compression for federated learning.
//!
//! Reduces the communication cost of federated learning by compressing
//! gradient updates using sparsification, quantization, or low-rank
//! approximation.

pub mod powersgd;
pub mod quantize;
pub mod randomk;
pub mod topk;
