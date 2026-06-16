//! Disjoint-path algorithms.
//!
//! - [`suurballe`] — Suurballe's algorithm for a pair of vertex-disjoint shortest paths
//!   of minimum total cost (Dijkstra + reduced-cost residual search).

pub mod suurballe;

pub use suurballe::{DisjointPaths, suurballe_vertex_disjoint};
