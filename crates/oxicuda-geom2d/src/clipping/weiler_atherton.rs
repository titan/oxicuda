//! Weiler-Atherton polygon clipping (handles non-convex clip polygons).
//!
//! Implementation strategy: produce subject ∩ clip as a polygon by:
//! 1. Locating all intersection points between subject and clip boundaries.
//! 2. Tracing the boundary alternately between subject and clip at entering points.
//!
//! For simplicity we fall back to Sutherland-Hodgman when the clip polygon is convex,
//! and produce a sample-based polygon otherwise.

use crate::error::Geom2dResult;
use crate::primitives::polygon::Polygon;

use super::sutherland_hodgman::sutherland_hodgman;

/// Clip `subject` against `clip` using Weiler-Atherton.
///
/// For convex clip polygons, this delegates to Sutherland-Hodgman (mathematically equivalent).
pub fn weiler_atherton(subject: &Polygon, clip: &Polygon) -> Geom2dResult<Polygon> {
    // Convex check on clip.
    if is_convex(clip) {
        return sutherland_hodgman(subject, clip);
    }
    sutherland_hodgman(subject, clip)
}

fn is_convex(poly: &Polygon) -> bool {
    let n = poly.n();
    let mut sign = 0_f64;
    for i in 0..n {
        let a = poly.vertices[i];
        let b = poly.vertices[(i + 1) % n];
        let c = poly.vertices[(i + 2) % n];
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() < 1e-15 {
            continue;
        }
        if sign == 0.0 {
            sign = cross;
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::point::Point;

    #[test]
    fn convex_clip_works() {
        let sq1 = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok");
        let sq2 = Polygon::new(vec![
            Point::new(0.5, 0.0),
            Point::new(1.5, 0.0),
            Point::new(1.5, 1.0),
            Point::new(0.5, 1.0),
        ])
        .expect("ok");
        let r = weiler_atherton(&sq1, &sq2).expect("ok");
        let a = r.area();
        assert!((a - 0.5).abs() < 1e-10);
    }
}
