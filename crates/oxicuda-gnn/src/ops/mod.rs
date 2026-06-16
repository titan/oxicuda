//! Graph neural network operation primitives.

pub mod scatter_softmax;

pub use scatter_softmax::{scatter_add, scatter_mean, scatter_softmax};
