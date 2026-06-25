//! Matrix sketching algorithms.

pub mod frequent_directions;
pub mod graph_sketch;

pub use frequent_directions::FrequentDirections;
pub use graph_sketch::{Edge, GraphSketch, GraphSketchConfig, SparsifiedGraph};
