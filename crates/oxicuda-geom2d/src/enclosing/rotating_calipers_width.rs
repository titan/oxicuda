//! Width of a point set: minimum distance between two parallel lines enclosing it.

use crate::error::{Geom2dError, Geom2dResult};
use crate::hull::andrew_monotone_chain::andrew_monotone_chain;
use crate::primitives::point::Point;

/// Width of a point set: min over hull edges of `max signed distance` from that edge.
pub fn rotating_calipers_width(pts: &[Point]) -> Geom2dResult<f64> {
    if pts.len() < 2 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 2,
            got: pts.len(),
        });
    }
    if pts.len() == 2 {
        return Ok(0.0);
    }
    let hull = match andrew_monotone_chain(pts) {
        Ok(h) => h,
        Err(_) => return Ok(0.0),
    };
    let h = hull.len();
    if h < 3 {
        return Ok(0.0);
    }
    let mut best = f64::INFINITY;
    for i in 0..h {
        let a = hull[i];
        let b = hull[(i + 1) % h];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-15 {
            continue;
        }
        let mut max_d = 0.0_f64;
        for &p in &hull {
            let d = ((p.x - a.x) * dy - (p.y - a.y) * dx).abs() / len;
            if d > max_d {
                max_d = d;
            }
        }
        if max_d < best {
            best = max_d;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_square_width_one() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let w = rotating_calipers_width(&pts).expect("ok");
        assert!((w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn thin_rectangle() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let w = rotating_calipers_width(&pts).expect("ok");
        assert!((w - 1.0).abs() < 1e-10);
    }
}
