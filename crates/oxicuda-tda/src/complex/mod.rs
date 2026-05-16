//! Simplicial complex data structures: simplex, complex, filtration.

pub mod complex;
pub mod filtration;
pub mod simplex;

pub use complex::SimplicialComplex;
pub use filtration::{FilteredSimplex, Filtration};
pub use simplex::Simplex;
