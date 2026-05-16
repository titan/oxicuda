//! Voronoi diagrams: Fortune's sweepline + Delaunay dual.

pub mod fortune_sweepline;
pub mod voronoi_from_delaunay;

pub use fortune_sweepline::{VoronoiDiagram, VoronoiEdge, fortune_voronoi};
pub use voronoi_from_delaunay::voronoi_from_delaunay;
