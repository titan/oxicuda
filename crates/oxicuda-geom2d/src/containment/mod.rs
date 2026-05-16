//! Point-in-region containment tests.

pub mod point_in_circle;
pub mod point_in_convex_polygon;
pub mod point_in_polygon_ray_cast;
pub mod point_in_polygon_winding;

pub use point_in_circle::point_in_circle;
pub use point_in_convex_polygon::point_in_convex_polygon;
pub use point_in_polygon_ray_cast::point_in_polygon_ray_cast;
pub use point_in_polygon_winding::{point_in_polygon_winding, winding_number};
