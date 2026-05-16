//! Intersection routines for lines, segments, circles, and polygons.

pub mod circle_circle;
pub mod circle_segment;
pub mod line_line;
pub mod segment_polygon;
pub mod segment_segment;

pub use circle_circle::{CircleCircleIntersection, intersect_circles};
pub use circle_segment::{CircleSegmentIntersection, intersect_circle_segment};
pub use line_line::{LineLineIntersection, intersect_lines};
pub use segment_polygon::intersect_segment_polygon;
pub use segment_segment::{SegmentSegmentIntersection, intersect_segments};
