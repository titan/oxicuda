//! Discontinuous Galerkin (DG) methods.

pub mod dg1d;

pub use dg1d::{Dg1dSpace, lgl_nodes, lgl_weights};
