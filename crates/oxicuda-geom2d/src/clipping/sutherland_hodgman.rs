//! Sutherland-Hodgman polygon clipping (clip subject against a convex clip polygon).
//!
//! Iteratively clip the subject polygon against each edge of the convex clip polygon.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;

/// Clip the (possibly non-convex) `subject` polygon against the **convex** `clip` polygon.
/// Returns the intersection polygon, possibly empty.
pub fn sutherland_hodgman(subject: &Polygon, clip: &Polygon) -> Geom2dResult<Polygon> {
    if subject.n() < 3 || clip.n() < 3 {
        return Err(Geom2dError::NotEnoughPoints {
            needed: 3,
            got: subject.n().min(clip.n()),
        });
    }
    let mut out: Vec<Point> = subject.vertices.clone();
    let m = clip.n();
    for i in 0..m {
        if out.is_empty() {
            break;
        }
        let a = clip.vertices[i];
        let b = clip.vertices[(i + 1) % m];
        let input = out.clone();
        out.clear();
        let len = input.len();
        for j in 0..len {
            let cur = input[j];
            let prev = input[(j + len - 1) % len];
            let cur_in = inside_left_of(a, b, cur);
            let prev_in = inside_left_of(a, b, prev);
            if cur_in {
                if !prev_in {
                    if let Some(ip) = line_intersect(prev, cur, a, b) {
                        out.push(ip);
                    }
                }
                out.push(cur);
            } else if prev_in {
                if let Some(ip) = line_intersect(prev, cur, a, b) {
                    out.push(ip);
                }
            }
        }
    }
    if out.len() < 3 {
        return Ok(Polygon {
            vertices: vec![Point::ORIGIN, Point::ORIGIN, Point::ORIGIN],
        });
    }
    Polygon::new(out)
}

fn inside_left_of(a: Point, b: Point, p: Point) -> bool {
    (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= -1e-12
}

fn line_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> Option<Point> {
    let dx1 = p2.x - p1.x;
    let dy1 = p2.y - p1.y;
    let dx2 = p4.x - p3.x;
    let dy2 = p4.y - p3.y;
    let denom = dx1 * dy2 - dy1 * dx2;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((p3.x - p1.x) * dy2 - (p3.y - p1.y) * dx2) / denom;
    Some(Point::new(p1.x + t * dx1, p1.y + t * dy1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Polygon {
        Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ])
        .expect("ok")
    }

    #[test]
    fn clip_square_by_offset_square() {
        let sq1 = unit_square();
        let sq2 = Polygon::new(vec![
            Point::new(0.5, 0.5),
            Point::new(1.5, 0.5),
            Point::new(1.5, 1.5),
            Point::new(0.5, 1.5),
        ])
        .expect("ok");
        let r = sutherland_hodgman(&sq1, &sq2).expect("ok");
        let a = r.area();
        assert!((a - 0.25).abs() < 1e-10);
    }

    #[test]
    fn clip_disjoint_empty() {
        let sq1 = unit_square();
        let sq2 = Polygon::new(vec![
            Point::new(10.0, 10.0),
            Point::new(11.0, 10.0),
            Point::new(11.0, 11.0),
            Point::new(10.0, 11.0),
        ])
        .expect("ok");
        let r = sutherland_hodgman(&sq1, &sq2).expect("ok");
        assert!(r.area() < 1e-12);
    }
}
