//! Visibility structures for path planning.
//!
//! Currently provides the visibility graph (Lozano-Pérez / de Berg) with Dijkstra shortest
//! paths for navigating around polygonal obstacles.

pub mod visibility_graph;

pub use visibility_graph::{VisibilityGraph, visible};
