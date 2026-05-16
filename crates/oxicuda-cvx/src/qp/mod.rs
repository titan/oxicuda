//! Quadratic programming.

pub mod active_set_qp;
pub mod primal_dual_qp;

pub use active_set_qp::active_set_qp;
pub use primal_dual_qp::primal_dual_qp;
