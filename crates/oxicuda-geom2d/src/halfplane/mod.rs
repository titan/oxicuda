//! Half-plane intersection.
//!
//! Computes the intersection of a finite set of half-planes
//! `{ (x, y) : a*x + b*y <= c }` and returns the resulting convex feasible
//! region as a polygon, or an explicit `Empty` / `Unbounded` marker.

pub mod half_plane_intersection;

pub use half_plane_intersection::{HalfPlane, HalfPlaneRegion, half_plane_intersection};
