//! Riemannian geometry primitives.
//!
//! - [`stiefel`]    Stiefel manifold St(n, p): orthonormal n x p frames.
//! - [`grassmann`]  Grassmann manifold Gr(n, p): p-dimensional subspaces.
//! - [`spd`]        Symmetric Positive-Definite cone SPD(n) with affine-invariant metric.
//! - [`hyperbolic_poincare`]  Poincaré ball model of hyperbolic space.

pub mod grassmann;
pub mod hyperbolic_poincare;
pub mod spd;
pub mod stiefel;

pub use grassmann::{grassmann_distance, grassmann_project_tangent, grassmann_retract};
pub use hyperbolic_poincare::{mobius_add, poincare_distance, poincare_project};
pub use spd::{spd_distance, spd_exp, spd_log, spd_project_symmetric};
pub use stiefel::{stiefel_project_tangent, stiefel_retract_qr};
