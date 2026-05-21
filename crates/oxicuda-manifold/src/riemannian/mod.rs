//! Riemannian geometry primitives.
//!
//! - [`stiefel`]              Stiefel manifold St(n, p): orthonormal n x p frames.
//! - [`grassmann`]            Grassmann manifold Gr(n, p): p-dimensional subspaces.
//! - [`spd`]                  Symmetric Positive-Definite cone SPD(n) with affine-invariant metric.
//! - [`spd_bures`]            Bures-Wasserstein geometry on SPD(n) (optimal transport metric).
//! - [`mod@spd_kmeans`]       Fréchet mean and Riemannian k-means on SPD(n).
//! - [`hyperbolic_poincare`]  Poincaré ball model of hyperbolic space.
//! - [`hyperbolic_lorentz`]   Lorentz (hyperboloid) model of hyperbolic space.
//! - [`so_n`]                 Special Orthogonal Group SO(n) with matrix-exponential retraction.

pub mod grassmann;
pub mod hyperbolic_lorentz;
pub mod hyperbolic_poincare;
pub mod so_n;
pub mod spd;
pub mod spd_bures;
pub mod spd_kmeans;
pub mod stiefel;

pub use grassmann::{grassmann_distance, grassmann_project_tangent, grassmann_retract};
pub use hyperbolic_lorentz::{
    lorentz_distance, lorentz_exp, lorentz_from_poincare, lorentz_inner, lorentz_log,
    lorentz_mobius_add, lorentz_norm_sq, lorentz_origin, lorentz_project_tangent,
    lorentz_to_poincare,
};
pub use hyperbolic_poincare::{mobius_add, poincare_distance, poincare_project};
pub use so_n::{
    so_2_rotation, so_n_check, so_n_distance, so_n_geodesic, so_n_identity, so_n_inner, so_n_log,
    so_n_norm, so_n_project_tangent, so_n_random, so_n_retract_cayley, so_n_retract_expm,
    so_n_retract_qr, so_n_riemannian_gradient,
};
pub use spd::{spd_distance, spd_exp, spd_log, spd_project_symmetric};
pub use spd_bures::{
    bures_distance, bures_exp, bures_frechet_mean, bures_geodesic, bures_geometric_mean, bures_log,
    spd_inv, spd_inv_sqrt, spd_sqrt,
};
pub use spd_kmeans::{
    FrechetMeanConfig, FrechetMeanResult, SpdKmeansConfig, SpdKmeansResult, spd_frechet_mean,
    spd_kmeans,
};
pub use stiefel::{stiefel_project_tangent, stiefel_retract_qr};
