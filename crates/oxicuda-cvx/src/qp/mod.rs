//! Quadratic programming.

pub mod active_set_qp;
pub mod mehrotra_qp;
pub mod osqp;
pub mod primal_dual_qp;

pub use active_set_qp::active_set_qp;
pub use mehrotra_qp::{MehrotraQpResult, mehrotra_qp};
pub use osqp::{Osqp, OsqpConfig, OsqpResult};
pub use primal_dual_qp::primal_dual_qp;
