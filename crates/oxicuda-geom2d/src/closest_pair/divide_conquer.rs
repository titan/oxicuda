//! Divide-and-conquer O(n log n) closest pair.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;

/// Returns `(idx_a, idx_b, distance)` for the closest pair.
pub fn closest_pair_dc(pts: &[Point]) -> Geom2dResult<(usize, usize, f64)> {
    if pts.len() < 2 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 2,
            got: pts.len(),
        });
    }
    let mut indexed: Vec<(usize, Point)> = pts.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        a.1.x
            .partial_cmp(&b.1.x)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(
                a.1.y
                    .partial_cmp(&b.1.y)
                    .unwrap_or(core::cmp::Ordering::Equal),
            )
    });
    let (a, b, d_sq) = rec(&indexed);
    Ok((a, b, d_sq.sqrt()))
}

fn rec(pts: &[(usize, Point)]) -> (usize, usize, f64) {
    let n = pts.len();
    if n <= 3 {
        let mut best = f64::INFINITY;
        let mut ia = pts[0].0;
        let mut ib = pts[1].0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = pts[i].1.distance_sq(pts[j].1);
                if d < best {
                    best = d;
                    ia = pts[i].0;
                    ib = pts[j].0;
                }
            }
        }
        return (ia, ib, best);
    }
    let mid = n / 2;
    let mid_x = pts[mid].1.x;
    let (left, right) = pts.split_at(mid);
    let (la, lb, ld) = rec(left);
    let (ra, rb, rd) = rec(right);
    let (mut ba, mut bb, mut bd) = if ld < rd { (la, lb, ld) } else { (ra, rb, rd) };
    let d = bd.sqrt();
    let mut strip: Vec<(usize, Point)> = Vec::new();
    for &(idx, p) in pts {
        if (p.x - mid_x).abs() <= d {
            strip.push((idx, p));
        }
    }
    strip.sort_by(|a, b| {
        a.1.y
            .partial_cmp(&b.1.y)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    for i in 0..strip.len() {
        let mut j = i + 1;
        while j < strip.len() && strip[j].1.y - strip[i].1.y < d {
            let ds = strip[i].1.distance_sq(strip[j].1);
            if ds < bd {
                bd = ds;
                ba = strip[i].0;
                bb = strip[j].0;
            }
            j += 1;
        }
    }
    (ba, bb, bd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closest_pair::brute_force::closest_pair_brute;
    use crate::handle::LcgRng;

    #[test]
    fn square_distance_one() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 0.0),
        ];
        let (_, _, d) = closest_pair_dc(&pts).expect("ok");
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matches_brute() {
        let mut r = LcgRng::new(42);
        let n = 30;
        let pts: Vec<Point> = (0..n)
            .map(|_| Point::new(r.next_f64() * 10.0, r.next_f64() * 10.0))
            .collect();
        let (_, _, d1) = closest_pair_brute(&pts).expect("ok");
        let (_, _, d2) = closest_pair_dc(&pts).expect("ok");
        assert!((d1 - d2).abs() < 1e-12);
    }
}
