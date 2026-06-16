//! Polygon triangulation and Delaunay triangulation.

pub mod bowyer_watson_delaunay;
pub mod constrained_delaunay;
pub mod ear_clipping;
pub mod monotone;

pub use bowyer_watson_delaunay::{Triangle, bowyer_watson};
pub use constrained_delaunay::constrained_delaunay;
pub use ear_clipping::ear_clipping;
pub use monotone::{make_monotone, triangulate_monotone, triangulate_simple};
