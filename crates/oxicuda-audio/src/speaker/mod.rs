//! Speaker embedding and temporal pooling modules.

pub mod attentive_pool;
pub mod stats_pool;
pub mod x_vector;

pub use attentive_pool::AttentivePool;
pub use stats_pool::stats_pool;
pub use x_vector::{XVectorConfig, XVectorTdnn};
