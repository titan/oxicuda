//! Generic tensor contraction.

pub mod einsum;
pub mod path;

pub use einsum::{LabelledTensor, einsum_binary};
pub use path::{ContractionPath, greedy_path};
