//! Boundary matrix, column reduction, and persistence pair extraction.

pub mod boundary;
pub mod persistent;
pub mod reduction;

pub use boundary::BoundaryMatrix;
pub use persistent::{PersistencePair, extract_persistence_pairs};
pub use reduction::reduce_boundary_matrix;
