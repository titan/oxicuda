//! `oxicuda-geom2d` - 2D Computational Geometry for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-geom2d
//! |-- primitives/      - Point, Vector, Line, Segment, Ray, Circle, Aabb, Polygon
//! |-- predicate/       - Orientation, in-circle, dot/cross, robust signs
//! |-- intersection/    - Segment-segment, line-line, segment-polygon, circle-* intersection
//! |-- containment/     - Point-in-polygon (winding/ray-cast), in convex polygon, in circle
//! |-- hull/            - Graham scan, Andrew monotone chain, QuickHull, Jarvis march, Chan
//! |-- triangulation/   - Ear clipping, Bowyer-Watson Delaunay, constrained Delaunay
//! |-- voronoi/         - Fortune sweepline, Voronoi from Delaunay dual
//! |-- clipping/        - Sutherland-Hodgman, Weiler-Atherton, Cohen-Sutherland, Liang-Barsky
//! |-- polygon_ops/     - Shoelace area, centroid, perimeter, convexity, offset, Minkowski sum
//! |-- closest_pair/    - Brute force O(n^2), divide-and-conquer O(n log n)
//! |-- enclosing/       - Welzl smallest circle, AABB, rotating calipers diameter/width
//! |-- sweepline/       - Bentley-Ottmann segment intersection sweep
//! |-- point_location/  - Slab method, trapezoidal map
//! |-- index/           - 2D KD-tree, R-tree (STR bulk load), Quadtree
//! |-- metrics/         - Euclidean, Manhattan, Chebyshev, angle, signed area
//! `-- ptx_kernels      - 7 GPU kernels for batched geometry primitives
//! ```
//!
//! All algorithms are implemented in pure Rust with no external dependencies beyond `thiserror`.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

pub mod clipping;
pub mod closest_pair;
pub mod containment;
pub mod enclosing;
pub mod error;
pub mod handle;
pub mod hull;
pub mod index;
pub mod intersection;
pub mod metrics;
pub mod point_location;
pub mod polygon_ops;
pub mod predicate;
pub mod primitives;
pub mod ptx_kernels;
pub mod sweepline;
pub mod triangulation;
pub mod voronoi;

pub use error::{Geom2dError, Geom2dResult};
pub use handle::{Geom2dHandle, LcgRng, SmVersion};

#[cfg(test)]
mod e2e_tests;
