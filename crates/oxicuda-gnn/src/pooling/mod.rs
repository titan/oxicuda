//! Graph pooling operations.

pub mod diff_pool;
pub mod global_pool;
pub mod sag_pool;
pub mod topk_pool;

pub use sag_pool::{SagPool, SagPoolResult};
