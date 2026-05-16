//! Cohen-Sutherland line clipping against an AABB.

use crate::primitives::aabb::Aabb;
use crate::primitives::point::Point;

const INSIDE: u8 = 0;
const LEFT: u8 = 1;
const RIGHT: u8 = 2;
const BOTTOM: u8 = 4;
const TOP: u8 = 8;

fn compute_outcode(p: Point, aabb: Aabb) -> u8 {
    let mut code = INSIDE;
    if p.x < aabb.min.x {
        code |= LEFT;
    } else if p.x > aabb.max.x {
        code |= RIGHT;
    }
    if p.y < aabb.min.y {
        code |= BOTTOM;
    } else if p.y > aabb.max.y {
        code |= TOP;
    }
    code
}

/// Clip segment `(p0, p1)` against `aabb`. Returns Some(clipped_segment_endpoints) or None.
#[must_use]
pub fn cohen_sutherland(p0: Point, p1: Point, aabb: Aabb) -> Option<(Point, Point)> {
    let mut a = p0;
    let mut b = p1;
    let mut oa = compute_outcode(a, aabb);
    let mut ob = compute_outcode(b, aabb);
    for _ in 0..6 {
        if (oa | ob) == 0 {
            return Some((a, b));
        }
        if (oa & ob) != 0 {
            return None;
        }
        let outcode_out = if oa != 0 { oa } else { ob };
        let p = clip_to_edge(a, b, outcode_out, aabb);
        if outcode_out == oa {
            a = p;
            oa = compute_outcode(a, aabb);
        } else {
            b = p;
            ob = compute_outcode(b, aabb);
        }
    }
    None
}

fn clip_to_edge(a: Point, b: Point, code: u8, aabb: Aabb) -> Point {
    if code & TOP != 0 {
        let t = (aabb.max.y - a.y) / (b.y - a.y);
        Point::new(a.x + (b.x - a.x) * t, aabb.max.y)
    } else if code & BOTTOM != 0 {
        let t = (aabb.min.y - a.y) / (b.y - a.y);
        Point::new(a.x + (b.x - a.x) * t, aabb.min.y)
    } else if code & RIGHT != 0 {
        let t = (aabb.max.x - a.x) / (b.x - a.x);
        Point::new(aabb.max.x, a.y + (b.y - a.y) * t)
    } else {
        let t = (aabb.min.x - a.x) / (b.x - a.x);
        Point::new(aabb.min.x, a.y + (b.y - a.y) * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entirely_inside() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(10.0, 10.0));
        let r = cohen_sutherland(Point::new(1.0, 1.0), Point::new(9.0, 9.0), bb).expect("ok");
        assert!((r.0.x - 1.0).abs() < 1e-12 && (r.1.x - 9.0).abs() < 1e-12);
    }

    #[test]
    fn entirely_outside_same_side() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(10.0, 10.0));
        assert!(cohen_sutherland(Point::new(11.0, 5.0), Point::new(12.0, 5.0), bb).is_none());
    }

    #[test]
    fn crossing_segment() {
        let bb = Aabb::new(Point::new(0.0, 0.0), Point::new(10.0, 10.0));
        let r = cohen_sutherland(Point::new(-5.0, 5.0), Point::new(15.0, 5.0), bb).expect("ok");
        assert!((r.0.x).abs() < 1e-12);
        assert!((r.1.x - 10.0).abs() < 1e-12);
    }
}
