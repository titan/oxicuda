//! Discontinuous Galerkin (DG) methods.

pub mod br2_elliptic;
pub mod dg1d;
pub mod dg_2d;
pub mod limiter_2d;

pub use br2_elliptic::{BR2_FACES_PER_ELEMENT, Br2Elliptic, DEFAULT_BR2_PENALTY};
pub use dg_2d::{Dg2dSpace, DgBoundary, DgFlux, dg_2d_advect, dg_2d_burgers};
pub use dg1d::{Dg1dSpace, lgl_nodes, lgl_weights};
pub use limiter_2d::{
    limit_bounds, limit_minmod, minmod, minmod_bounded_closure, minmod_closure, minmod_tvb,
};
