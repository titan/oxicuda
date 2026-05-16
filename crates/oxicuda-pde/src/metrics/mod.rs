//! Convergence metrics: L2 norm, H1 seminorm, max-norm, convergence-order estimation.

pub mod metrics;

pub use metrics::{convergence_order, h1_seminorm_1d, l2_norm_1d, max_norm};
