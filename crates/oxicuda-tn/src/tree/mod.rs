//! Tree Tensor Networks (TTN).
//!
//! Hierarchical binary-tree tensor networks for 1D quantum states whose
//! entanglement follows a tree-like (logarithmic-depth) structure.

pub mod tree_tn;

pub use tree_tn::{TreeTensorNetwork, TtnNode};
