//! Relation-based distillation methods.

pub mod cc;
pub mod corr_congruence;
pub mod crd;
pub mod crd_proj;
pub mod graph_distill;
pub mod rkd;
pub mod rkd_full;

pub use corr_congruence::{CcKernel, cc_loss, correlation_matrix, l2_normalize_rows};
pub use crd_proj::{CrdProjConfig, CrdProjectionHead, crd_proj_loss};
pub use rkd_full::{full_angle_loss, full_rkd_loss, full_triplet_count};
