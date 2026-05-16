//! Diameter (largest pairwise distance) of a point set via rotating calipers on the hull.

use crate::error::{Geom2dError, Geom2dResult};
use crate::hull::andrew_monotone_chain::andrew_monotone_chain;
use crate::primitives::point::Point;

/// Compute the diameter of `pts`: the maximum distance between any two points.
///
/// Uses the convex hull + antipodal pairs technique for O(n log n).
pub fn rotating_calipers_diameter(pts: &[Point]) -> Geom2dResult<(usize, usize, f64)> {
    if pts.len() < 2 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 2,
            got: pts.len(),
        });
    }
    if pts.len() == 2 {
        return Ok((0, 1, pts[0].distance(pts[1])));
    }
    let hull = match andrew_monotone_chain(pts) {
        Ok(h) => h,
        Err(_) => {
            // Collinear: just return the extreme pair.
            return brute_force_diameter(pts);
        }
    };
    let h = hull.len();
    if h < 2 {
        return brute_force_diameter(pts);
    }
    if h == 2 {
        return Ok((
            find_index(pts, hull[0]),
            find_index(pts, hull[1]),
            hull[0].distance(hull[1]),
        ));
    }
    let mut best = 0.0_f64;
    let mut best_i = 0_usize;
    let mut best_j = 0_usize;
    let mut k = 1_usize;
    // Find initial antipodal point: max area triangle.
    while cross_area(hull[0], hull[1], hull[(k + 1) % h]) > cross_area(hull[0], hull[1], hull[k]) {
        k += 1;
        if k >= h {
            k = 1;
            break;
        }
    }
    let mut j = k;
    for i in 0..h {
        loop {
            let ni = (i + 1) % h;
            let nj = (j + 1) % h;
            if cross_area(hull[i], hull[ni], hull[nj]) > cross_area(hull[i], hull[ni], hull[j]) {
                j = nj;
            } else {
                break;
            }
        }
        let d = hull[i].distance_sq(hull[j]);
        if d > best {
            best = d;
            best_i = find_index(pts, hull[i]);
            best_j = find_index(pts, hull[j]);
        }
        let nj = (j + 1) % h;
        let d2 = hull[i].distance_sq(hull[nj]);
        if d2 > best {
            best = d2;
            best_i = find_index(pts, hull[i]);
            best_j = find_index(pts, hull[nj]);
        }
    }
    Ok((best_i, best_j, best.sqrt()))
}

fn cross_area(a: Point, b: Point, c: Point) -> f64 {
    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
}

fn find_index(pts: &[Point], p: Point) -> usize {
    let mut best = 0_usize;
    let mut bd = pts[0].distance_sq(p);
    for (i, &q) in pts.iter().enumerate().skip(1) {
        let d = q.distance_sq(p);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

fn brute_force_diameter(pts: &[Point]) -> Geom2dResult<(usize, usize, f64)> {
    let mut best = 0.0_f64;
    let mut a = 0_usize;
    let mut b = 1_usize;
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d = pts[i].distance_sq(pts[j]);
            if d > best {
                best = d;
                a = i;
                b = j;
            }
        }
    }
    Ok((a, b, best.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_square_diameter_root_two() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let (_, _, d) = rotating_calipers_diameter(&pts).expect("ok");
        assert!((d - 2_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn two_points() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(3.0, 4.0)];
        let (_, _, d) = rotating_calipers_diameter(&pts).expect("ok");
        assert!((d - 5.0).abs() < 1e-10);
    }
}
