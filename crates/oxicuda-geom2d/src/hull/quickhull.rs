//! Divide-and-conquer QuickHull convex hull.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

/// Compute the convex hull in CCW order via QuickHull.
pub fn quickhull(pts: &[Point]) -> Geom2dResult<Vec<Point>> {
    if pts.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: pts.len(),
        });
    }
    // Find extreme points
    let mut leftmost = 0;
    let mut rightmost = 0;
    for i in 1..pts.len() {
        if pts[i].x < pts[leftmost].x || (pts[i].x == pts[leftmost].x && pts[i].y < pts[leftmost].y)
        {
            leftmost = i;
        }
        if pts[i].x > pts[rightmost].x
            || (pts[i].x == pts[rightmost].x && pts[i].y > pts[rightmost].y)
        {
            rightmost = i;
        }
    }
    let a = pts[leftmost];
    let b = pts[rightmost];
    if a == b {
        return Err(Geom2dError::DegeneratePolygon(
            "all points coincident".into(),
        ));
    }
    let mut upper_pts: Vec<Point> = Vec::new();
    let mut lower_pts: Vec<Point> = Vec::new();
    for &p in pts {
        if p == a || p == b {
            continue;
        }
        let s = orient_value(a, b, p);
        if s > 0.0 {
            upper_pts.push(p);
        } else if s < 0.0 {
            lower_pts.push(p);
        }
    }
    let mut hull = Vec::new();
    hull.push(a);
    find_hull(&upper_pts, a, b, &mut hull);
    hull.push(b);
    find_hull(&lower_pts, b, a, &mut hull);
    if hull.len() < 3 {
        return Err(Geom2dError::DegeneratePolygon("collinear input".into()));
    }
    Ok(hull)
}

fn find_hull(pts: &[Point], a: Point, b: Point, hull: &mut Vec<Point>) {
    if pts.is_empty() {
        return;
    }
    // Farthest point from line a-b on the same side.
    let mut farthest = pts[0];
    let mut best = orient_value(a, b, pts[0]).abs();
    for &p in &pts[1..] {
        let d = orient_value(a, b, p).abs();
        if d > best {
            best = d;
            farthest = p;
        }
    }
    // Split remaining points into outside(a, farthest) and outside(farthest, b)
    let mut s1: Vec<Point> = Vec::new();
    let mut s2: Vec<Point> = Vec::new();
    for &p in pts {
        if p == farthest {
            continue;
        }
        if orient_value(a, farthest, p) > 0.0 {
            s1.push(p);
        } else if orient_value(farthest, b, p) > 0.0 {
            s2.push(p);
        }
    }
    find_hull(&s1, a, farthest, hull);
    hull.push(farthest);
    find_hull(&s2, farthest, b, hull);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_square_hull() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(0.5, 0.5),
        ];
        let h = quickhull(&pts).expect("ok");
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn triangle_hull() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
        ];
        let h = quickhull(&pts).expect("ok");
        assert_eq!(h.len(), 3);
    }
}
