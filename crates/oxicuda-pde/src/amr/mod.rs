//! Adaptive Mesh Refinement (AMR) for 2D quadtree hierarchies.
//!
//! * [`octree`] — a quadtree cell tree with refine / coarsen, leaf iteration,
//!   level tracking, face-neighbour queries, and 2:1 balance enforcement.
//! * [`error_estimator`] — jump / gradient refinement indicators plus Dörfler
//!   (fixed-fraction) and threshold marking strategies.
//!
//! The quadtree is the 2D analogue of an octree; the same algorithms extend to
//! 3D unchanged by using eight children per cell.

pub mod error_estimator;
pub mod octree;

pub use error_estimator::{
    Indicators, MarkedCells, dorfler_mark, gradient_indicator, jump_indicator, threshold_mark,
};
pub use octree::{Aabb, CHILDREN_PER_CELL, Cell, Quadtree};
