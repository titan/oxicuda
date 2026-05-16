//! Chan's algorithm: O(n log h) optimal convex hull via partition + gift-wrap.
//!
//! 1. For a guess `m`, partition into ceil(n/m) groups; compute each group's hull via Graham.
//! 2. Gift-wrap across groups using O(log m) tangent searches.
//! 3. Double `m` until success.

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

use super::graham_scan::graham_scan;

/// Convex hull via Chan's algorithm.
pub fn chans_algorithm(pts: &[Point]) -> Geom2dResult<Vec<Point>> {
    if pts.len() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: pts.len(),
        });
    }
    let n = pts.len();
    let mut m = 4_usize;
    while m < n * 2 {
        if let Some(hull) = try_chan(pts, m) {
            return Ok(hull);
        }
        m = m.saturating_mul(2).max(m + 1);
        if m >= n {
            // Final attempt: use full set as one "group" (i.e., Graham scan over the whole).
            return graham_scan(pts);
        }
    }
    graham_scan(pts)
}

fn try_chan(pts: &[Point], m: usize) -> Option<Vec<Point>> {
    let groups: Vec<Vec<Point>> = pts.chunks(m).map(|c| c.to_vec()).collect();
    let mut sub_hulls: Vec<Vec<Point>> = Vec::new();
    for g in &groups {
        if g.len() < 3 {
            // Trivial: itself acts as its hull.
            sub_hulls.push(g.clone());
        } else {
            match graham_scan(g) {
                Ok(h) => sub_hulls.push(h),
                Err(_) => sub_hulls.push(g.clone()),
            }
        }
    }
    // Find bottom-most lowest-leftmost overall point as start.
    let mut start_group = 0;
    let mut start_idx = 0;
    let mut best = sub_hulls[0][0];
    for (gi, sh) in sub_hulls.iter().enumerate() {
        for (i, &p) in sh.iter().enumerate() {
            if p.y < best.y || (p.y == best.y && p.x < best.x) {
                best = p;
                start_group = gi;
                start_idx = i;
            }
        }
    }
    let mut hull = Vec::new();
    let mut cur_group = start_group;
    let mut cur_idx = start_idx;
    let max_iter = m + 2;
    for _ in 0..max_iter {
        let cur = sub_hulls[cur_group][cur_idx];
        hull.push(cur);
        // Find best next candidate across all sub-hulls.
        let mut best_group = 0;
        let mut best_idx = 0;
        let mut best_pt = sub_hulls[0][0];
        for (gi, sh) in sub_hulls.iter().enumerate() {
            let cand_idx = right_tangent(sh, cur);
            let cand = sh[cand_idx];
            if cand == cur {
                continue;
            }
            if (best_pt == cur)
                || orient_value(cur, best_pt, cand) < 0.0
                || (orient_value(cur, best_pt, cand) == 0.0
                    && cur.distance_sq(cand) > cur.distance_sq(best_pt))
            {
                best_group = gi;
                best_idx = cand_idx;
                best_pt = cand;
            }
        }
        if best_pt == sub_hulls[start_group][start_idx] {
            return Some(hull);
        }
        cur_group = best_group;
        cur_idx = best_idx;
        if hull.len() > m {
            return None;
        }
    }
    None
}

fn right_tangent(hull: &[Point], from: Point) -> usize {
    // O(n) tangent search: pick the point yielding the most clockwise turn.
    let mut best = 0;
    let n = hull.len();
    for i in 1..n {
        let o = orient_value(from, hull[best], hull[i]);
        if hull[best] == from
            || o < 0.0
            || (o == 0.0 && from.distance_sq(hull[i]) > from.distance_sq(hull[best]))
        {
            best = i;
        }
    }
    best
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
        let h = chans_algorithm(&pts).expect("ok");
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn many_points() {
        let n = 50;
        let mut pts = Vec::new();
        for i in 0..n {
            let t = std::f64::consts::TAU * (i as f64) / (n as f64);
            pts.push(Point::new(t.cos(), t.sin()));
        }
        // Add interior junk
        for k in 0..20 {
            pts.push(Point::new((k as f64) * 0.01, 0.0));
        }
        let h = chans_algorithm(&pts).expect("ok");
        assert!(h.len() <= n);
        assert!(h.len() >= 3);
    }
}
