//! Polygon operations: area, centroid, perimeter, convexity, offset, Minkowski sum.

pub mod area_shoelace;
pub mod centroid;
pub mod convexity_test;
pub mod minkowski_sum;
pub mod perimeter;
pub mod polygon_offset;

pub use area_shoelace::{area_shoelace, signed_area_shoelace};
pub use centroid::polygon_centroid;
pub use convexity_test::is_convex;
pub use minkowski_sum::minkowski_sum;
pub use perimeter::polygon_perimeter;
pub use polygon_offset::polygon_offset;
