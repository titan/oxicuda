//! Brute-force O(n^2) closest pair.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;

/// Returns the (index_a, index_b, distance) of the closest pair.
pub fn closest_pair_brute(pts: &[Point]) -> Geom2dResult<(usize, usize, f64)> {
    if pts.len() < 2 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 2,
            got: pts.len(),
        });
    }
    let mut best = f64::INFINITY;
    let mut a = 0usize;
    let mut b = 1usize;
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d = pts[i].distance_sq(pts[j]);
            if d < best {
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
    fn unit_square_distance_one() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 0.0),
        ];
        let (_, _, d) = closest_pair_brute(&pts).expect("ok");
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn too_few_errs() {
        let pts = vec![Point::new(0.0, 0.0)];
        assert!(closest_pair_brute(&pts).is_err());
    }
}
