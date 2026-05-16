//! Jarvis march (gift wrapping) convex hull. O(n h) where h = hull size.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

/// Compute the convex hull in CCW order via gift wrapping.
pub fn jarvis_march(pts: &[Point]) -> Geom2dResult<Vec<Point>> {
    if pts.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: pts.len(),
        });
    }
    let mut start = 0;
    for i in 1..pts.len() {
        if pts[i].x < pts[start].x || (pts[i].x == pts[start].x && pts[i].y < pts[start].y) {
            start = i;
        }
    }
    let mut hull = Vec::new();
    let mut current = start;
    loop {
        hull.push(pts[current]);
        let mut next = (current + 1) % pts.len();
        for i in 0..pts.len() {
            if i == current {
                continue;
            }
            let o = orient_value(pts[current], pts[next], pts[i]);
            if o < 0.0
                || (o == 0.0
                    && pts[current].distance_sq(pts[i]) > pts[current].distance_sq(pts[next]))
            {
                next = i;
            }
        }
        current = next;
        if current == start {
            break;
        }
        if hull.len() > pts.len() {
            return Err(Geom2dError::NumericalInstability(
                "Jarvis march did not terminate".into(),
            ));
        }
    }
    if hull.len() < 3 {
        return Err(Geom2dError::DegeneratePolygon("collinear input".into()));
    }
    Ok(hull)
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
        let h = jarvis_march(&pts).expect("ok");
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn triangle_hull() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(0.0, 4.0),
        ];
        let h = jarvis_march(&pts).expect("ok");
        assert_eq!(h.len(), 3);
    }
}
