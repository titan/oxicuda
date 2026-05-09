//! Regularization-based continual learning methods.
//!
//! These methods prevent catastrophic forgetting by adding regularization
//! terms to the loss that penalize changes to important parameters.

pub mod ewc;
pub mod mas;
pub mod si;
