//! 2D alpha shapes (Edelsbrunner-Kirkpatrick-Seidel 1983).
//!
//! An alpha shape is a generalisation of the convex hull parameterised by a
//! radius `alpha`. It is built from the Delaunay triangulation of the point set;
//! the `alpha`-complex retains Delaunay triangles whose circumradius does not
//! exceed `alpha`, and its boundary is the set of edges incident to at most one
//! retained triangle (plus exposed "singular" edges admitted by the edge-circle
//! criterion).

pub mod alpha_shape;

pub use alpha_shape::{AlphaShape, alpha_shape, alpha_shape_auto};
