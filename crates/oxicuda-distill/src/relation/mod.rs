//! Relation-based distillation methods.

pub mod cc;
pub mod corr_congruence;
pub mod crd;
pub mod graph_distill;
pub mod rkd;

pub use corr_congruence::{CcKernel, cc_loss, correlation_matrix, l2_normalize_rows};
