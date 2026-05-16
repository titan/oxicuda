//! Boundary condition types and helpers.

pub mod dirichlet;
pub mod neumann;
pub mod robin;

pub use dirichlet::DirichletBc;
pub use neumann::NeumannBc;
pub use robin::RobinBc;

/// Generic boundary condition kind tagging a face/edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BcKind {
    Dirichlet,
    Neumann,
    Robin,
}
