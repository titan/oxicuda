//! Andrew's monotone chain convex hull. O(n log n).
//!
//! 1. Sort points by (x, y).
//! 2. Build the lower hull by sweeping left-to-right.
//! 3. Build the upper hull by sweeping right-to-left.
//! 4. Concatenate, removing the duplicated endpoints.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

/// Compute the convex hull in CCW order.
pub fn andrew_monotone_chain(pts: &[Point]) -> Geom2dResult<Vec<Point>> {
    if pts.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: pts.len(),
        });
    }
    let mut pts = pts.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(core::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| a == b);
    let n = pts.len();
    if n < 3 {
        return Err(Geom2dError::DegeneratePolygon(
            "fewer than 3 distinct points".into(),
        ));
    }
    let mut lower: Vec<Point> = Vec::with_capacity(n);
    for &p in &pts {
        while lower.len() >= 2 {
            let a = lower[lower.len() - 2];
            let b = lower[lower.len() - 1];
            if orient_value(a, b, p) <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }
    let mut upper: Vec<Point> = Vec::with_capacity(n);
    for &p in pts.iter().rev() {
        while upper.len() >= 2 {
            let a = upper[upper.len() - 2];
            let b = upper[upper.len() - 1];
            if orient_value(a, b, p) <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.len() < 3 {
        return Err(Geom2dError::DegeneratePolygon("collinear hull".into()));
    }
    Ok(lower)
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
        let h = andrew_monotone_chain(&pts).expect("ok");
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn duplicated_points() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(0.5, 0.5),
        ];
        let h = andrew_monotone_chain(&pts).expect("ok");
        assert_eq!(h.len(), 3);
    }
}
