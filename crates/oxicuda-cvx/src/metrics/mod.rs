//! Convergence and KKT-residual metrics.

pub mod metrics;

pub use metrics::{convergence_rate, dual_residual, duality_gap, kkt_residual, primal_residual};
