//! Welzl's randomized smallest enclosing circle (Move-To-Front variant).
//!
//! Expected linear-time algorithm. Returns the minimum-radius circle that contains all input points.

use crate::error::{Geom2dError, Geom2dResult};
use crate::handle::LcgRng;
use crate::primitives::circle::Circle;
use crate::primitives::point::Point;

/// Smallest enclosing circle of `pts`.
pub fn welzl_smallest_circle(pts: &[Point], seed: u64) -> Geom2dResult<Circle> {
    if pts.is_empty() {
        return Err(Geom2dError::EmptyInput);
    }
    let mut shuffled: Vec<Point> = pts.to_vec();
    let mut rng = LcgRng::new(seed);
    // Fisher-Yates shuffle for randomness.
    for i in (1..shuffled.len()).rev() {
        let j = rng.next_usize(i + 1);
        shuffled.swap(i, j);
    }
    let mut c = Circle::new(shuffled[0], 0.0);
    for i in 1..shuffled.len() {
        if !c.contains_eq(shuffled[i]) {
            c = Circle::new(shuffled[i], 0.0);
            for j in 0..i {
                if !c.contains_eq(shuffled[j]) {
                    c = Circle::from_two_points(shuffled[i], shuffled[j]);
                    for k in 0..j {
                        if !c.contains_eq(shuffled[k]) {
                            c = match Circle::from_three_points(
                                shuffled[i],
                                shuffled[j],
                                shuffled[k],
                            ) {
                                Some(cc) => cc,
                                None => {
                                    // Collinear triple: take the two extreme points as a diameter.
                                    let p1 = shuffled[i];
                                    let p2 = shuffled[j];
                                    let p3 = shuffled[k];
                                    let d12 = p1.distance_sq(p2);
                                    let d13 = p1.distance_sq(p3);
                                    let d23 = p2.distance_sq(p3);
                                    if d12 >= d13 && d12 >= d23 {
                                        Circle::from_two_points(p1, p2)
                                    } else if d13 >= d23 {
                                        Circle::from_two_points(p1, p3)
                                    } else {
                                        Circle::from_two_points(p2, p3)
                                    }
                                }
                            };
                        }
                    }
                }
            }
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_square_radius_root2_over_two() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let c = welzl_smallest_circle(&pts, 7).expect("ok");
        let r = 2_f64.sqrt() / 2.0;
        assert!((c.radius - r).abs() < 1e-10);
        assert!((c.center.x - 0.5).abs() < 1e-10);
        assert!((c.center.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn single_point() {
        let pts = vec![Point::new(3.0, 4.0)];
        let c = welzl_smallest_circle(&pts, 0).expect("ok");
        assert!(c.radius < 1e-12);
    }

    #[test]
    fn collinear_three() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
        ];
        let c = welzl_smallest_circle(&pts, 1).expect("ok");
        assert!((c.center.x - 1.0).abs() < 1e-10);
        assert!((c.radius - 1.0).abs() < 1e-10);
    }
}
