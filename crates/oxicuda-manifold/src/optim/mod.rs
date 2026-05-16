//! Optimisation on Riemannian manifolds.
//!
//! - [`riemannian_sgd`] generic Riemannian SGD step (project gradient, retract back).
//! - [`retraction`]    retraction helpers.

pub mod retraction;
pub mod riemannian_sgd;

pub use retraction::{retract_polar_spd, retract_qr_stiefel};
pub use riemannian_sgd::{RsgdConfig, rsgd_step_spd, rsgd_step_stiefel};
