//! Boundary condition types and helpers.

pub mod dirichlet;
pub mod neumann;
pub mod periodic;
pub mod robin;

pub use dirichlet::DirichletBc;
pub use neumann::NeumannBc;
pub use periodic::{
    enforce_periodic_endpoint, periodic_first_derivative, periodic_laplacian_1d,
    periodic_laplacian_2d, wrap_index,
};
pub use robin::RobinBc;

/// Generic boundary condition kind tagging a face/edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BcKind {
    Dirichlet,
    Neumann,
    Robin,
    Periodic,
}
