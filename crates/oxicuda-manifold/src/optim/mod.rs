//! Optimisation on Riemannian manifolds.
//!
//! - [`riemannian_sgd`]   generic Riemannian SGD step (project gradient, retract back).
//! - [`mod@riemannian_adam`]  Riemannian Adam on Stiefel / Grassmann manifolds.
//! - [`retraction`]       retraction helpers.

pub mod retraction;
pub mod riemannian_adam;
pub mod riemannian_sgd;

pub use retraction::{retract_polar_spd, retract_qr_stiefel};
pub use riemannian_adam::{
    ManifoldType, RiemannianAdamConfig, RiemannianAdamResult, RiemannianAdamState, gradient_norm,
    project_tangent_ambient, retract_ambient, riemannian_adam, vec_axpby, vec_hadamard,
    vec_sqrt_eps,
};
pub use riemannian_sgd::{RsgdConfig, rsgd_step_spd, rsgd_step_stiefel};
