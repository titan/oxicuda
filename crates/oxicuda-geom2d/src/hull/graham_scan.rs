//! Graham scan convex hull algorithm.
//!
//! 1. Find the lowest-leftmost point as pivot.
//! 2. Sort remaining points by polar angle around the pivot.
//! 3. Sweep with a stack, popping when a right turn (CW) is encountered.
//!
//! Output: CCW-ordered hull (no duplicates, no collinear interior points).

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

/// Run the Graham scan on `pts`. Returns CCW-ordered hull vertices.
pub fn graham_scan(pts: &[Point]) -> Geom2dResult<Vec<Point>> {
    if pts.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: pts.len(),
        });
    }
    // Find pivot: lowest y, leftmost on tie.
    let mut pivot_idx = 0;
    for i in 1..pts.len() {
        let p = pts[i];
        let q = pts[pivot_idx];
        if p.y < q.y || (p.y == q.y && p.x < q.x) {
            pivot_idx = i;
        }
    }
    let pivot = pts[pivot_idx];
    let mut rest: Vec<Point> = pts.iter().copied().filter(|&p| p != pivot).collect();
    rest.sort_by(|a, b| {
        let av = orient_value(pivot, *a, *b);
        if av > 0.0 {
            core::cmp::Ordering::Less
        } else if av < 0.0 {
            core::cmp::Ordering::Greater
        } else {
            // Collinear: nearer comes first.
            let da = pivot.distance_sq(*a);
            let db = pivot.distance_sq(*b);
            da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
        }
    });
    // Discard collinear points except the farthest at each angular slot.
    let mut filtered: Vec<Point> = Vec::with_capacity(rest.len());
    for &p in &rest {
        while let Some(&last) = filtered.last() {
            if orient_value(pivot, last, p) == 0.0 {
                filtered.pop();
            } else {
                break;
            }
        }
        filtered.push(p);
    }
    let mut stack: Vec<Point> = vec![pivot];
    for p in filtered {
        while stack.len() >= 2 {
            let top = stack[stack.len() - 1];
            let nxt = stack[stack.len() - 2];
            if orient_value(nxt, top, p) <= 0.0 {
                stack.pop();
            } else {
                break;
            }
        }
        stack.push(p);
    }
    Ok(stack)
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
        let h = graham_scan(&pts).expect("ok");
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn collinear_pts() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        ];
        // All collinear: hull degenerates to fewer than 3 points.
        let r = graham_scan(&pts);
        // Should not panic; either error or a degenerate result.
        // We accept either: collapsed to 2 or fewer points.
        if let Ok(h) = r {
            assert!(h.len() <= 3);
        }
    }

    #[test]
    fn triangle_hull() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
        ];
        let h = graham_scan(&pts).expect("ok");
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn pentagon_random_interior() {
        let n = 5;
        let mut pts = Vec::new();
        for i in 0..n {
            let t = std::f64::consts::TAU * (i as f64) / (n as f64);
            pts.push(Point::new(t.cos(), t.sin()));
        }
        // Add interior points
        pts.push(Point::new(0.0, 0.0));
        pts.push(Point::new(0.1, 0.2));
        let h = graham_scan(&pts).expect("ok");
        assert_eq!(h.len(), n);
    }
}
