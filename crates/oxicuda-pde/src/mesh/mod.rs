//! Mesh data structures for finite-difference and finite-element solvers.

pub mod delaunay;
pub mod mesh1d;
pub mod mesh2d;
pub mod triangulation;
pub mod unstructured;

pub use delaunay::{Delaunay, DelaunayResult};
pub use mesh1d::Mesh1d;
pub use mesh2d::Mesh2d;
pub use triangulation::TriMesh2d;
pub use unstructured::{ElementKind, UnstructuredMesh};
