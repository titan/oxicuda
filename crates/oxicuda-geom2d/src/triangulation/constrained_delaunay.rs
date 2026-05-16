//! Constrained Delaunay triangulation.
//!
//! Builds an unconstrained Delaunay triangulation via Bowyer-Watson, then flips diagonals
//! to honor required edges (segments that must appear as triangle edges).

use crate::error::{Geom2dError, Geom2dResult};
use crate::predicate::in_circle::in_circle_signed;
use crate::predicate::orientation::orient_value;
use crate::primitives::point::Point;

use super::bowyer_watson_delaunay::{Triangle, bowyer_watson};

/// Compute a Delaunay triangulation that respects the given mandatory edges.
///
/// Each `(i, j)` in `edges` must appear as a triangle edge in the output.
/// Uses an iterative flip strategy.
pub fn constrained_delaunay(
    pts: &[Point],
    edges: &[(usize, usize)],
) -> Geom2dResult<Vec<Triangle>> {
    let mut tris = bowyer_watson(pts)?;
    if edges.is_empty() {
        return Ok(tris);
    }
    let max_iter = 8 * tris.len() * edges.len() + 16;
    for _ in 0..max_iter {
        let mut all_present = true;
        for &(a, b) in edges {
            if !contains_edge(&tris, a, b) {
                if let Some(flip) = find_flip(&tris, pts, a, b) {
                    apply_flip(&mut tris, pts, flip);
                    all_present = false;
                    break;
                } else {
                    return Err(Geom2dError::DegeneratePolygon(
                        "cannot enforce constraint edge".into(),
                    ));
                }
            }
        }
        if all_present {
            // Run a Delaunay legalization pass to restore the property where possible
            // (but avoid breaking required edges).
            legalize(&mut tris, pts, edges);
            return Ok(tris);
        }
    }
    Err(Geom2dError::NotConverged { iter: max_iter })
}

fn contains_edge(tris: &[Triangle], a: usize, b: usize) -> bool {
    for t in tris {
        if (t.a == a && t.b == b)
            || (t.a == b && t.b == a)
            || (t.b == a && t.c == b)
            || (t.b == b && t.c == a)
            || (t.c == a && t.a == b)
            || (t.c == b && t.a == a)
        {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, Copy)]
struct FlipPlan {
    t1: usize,
    t2: usize,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
}

fn find_flip(
    tris: &[Triangle],
    pts: &[Point],
    target_a: usize,
    target_b: usize,
) -> Option<FlipPlan> {
    // Look for a pair of adjacent triangles whose diagonal crosses target_a-target_b.
    let pa = pts[target_a];
    let pb = pts[target_b];
    for i in 0..tris.len() {
        for j in (i + 1)..tris.len() {
            if let Some(plan) = shared_diagonal(tris[i], tris[j], i, j) {
                let (c, d) = (plan.c, plan.d);
                let pc = pts[c];
                let pd = pts[d];
                if crosses(pa, pb, pc, pd) {
                    return Some(plan);
                }
            }
        }
    }
    None
}

fn shared_diagonal(t1: Triangle, t2: Triangle, i1: usize, i2: usize) -> Option<FlipPlan> {
    let v1 = [t1.a, t1.b, t1.c];
    let v2 = [t2.a, t2.b, t2.c];
    let mut shared: Vec<usize> = Vec::new();
    for x in v1 {
        if v2.contains(&x) {
            shared.push(x);
        }
    }
    if shared.len() != 2 {
        return None;
    }
    let a = shared[0];
    let b = shared[1];
    let c = v1.iter().copied().find(|&x| x != a && x != b)?;
    let d = v2.iter().copied().find(|&x| x != a && x != b)?;
    Some(FlipPlan {
        t1: i1,
        t2: i2,
        a,
        b,
        c,
        d,
    })
}

fn crosses(p1: Point, p2: Point, q1: Point, q2: Point) -> bool {
    let o1 = orient_value(p1, p2, q1);
    let o2 = orient_value(p1, p2, q2);
    let o3 = orient_value(q1, q2, p1);
    let o4 = orient_value(q1, q2, p2);
    (o1 * o2 < 0.0) && (o3 * o4 < 0.0)
}

fn apply_flip(tris: &mut Vec<Triangle>, pts: &[Point], plan: FlipPlan) {
    // Replace diagonal (a, b) with diagonal (c, d); produces two new triangles (a, c, d), (b, d, c).
    let new_t1 = Triangle::new_ccw(plan.a, plan.c, plan.d, pts);
    let new_t2 = Triangle::new_ccw(plan.b, plan.d, plan.c, pts);
    let (lo, hi) = if plan.t1 < plan.t2 {
        (plan.t1, plan.t2)
    } else {
        (plan.t2, plan.t1)
    };
    tris.remove(hi);
    tris.remove(lo);
    tris.push(new_t1);
    tris.push(new_t2);
}

fn legalize(tris: &mut Vec<Triangle>, pts: &[Point], constraint_edges: &[(usize, usize)]) {
    let mut changed = true;
    let mut iter = 0;
    while changed && iter < tris.len() * tris.len() + 8 {
        changed = false;
        iter += 1;
        for i in 0..tris.len() {
            for j in (i + 1)..tris.len() {
                if let Some(plan) = shared_diagonal(tris[i], tris[j], i, j) {
                    let pa = pts[plan.a];
                    let pb = pts[plan.b];
                    let pc = pts[plan.c];
                    let pd = pts[plan.d];
                    if is_constrained(plan.a, plan.b, constraint_edges) {
                        continue;
                    }
                    if in_circle_signed(pa, pc, pb, pd) > 1e-12 {
                        // Convex quadrilateral? Confirm c, d are on opposite sides of a-b.
                        let oc = orient_value(pa, pb, pc);
                        let od = orient_value(pa, pb, pd);
                        if oc * od < 0.0 {
                            apply_flip(tris, pts, plan);
                            changed = true;
                            break;
                        }
                    }
                }
            }
            if changed {
                break;
            }
        }
    }
}

fn is_constrained(a: usize, b: usize, constraint_edges: &[(usize, usize)]) -> bool {
    constraint_edges
        .iter()
        .any(|&(u, v)| (u == a && v == b) || (u == b && v == a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_constraints_matches_delaunay() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let dt = constrained_delaunay(&pts, &[]).expect("ok");
        assert_eq!(dt.len(), 2);
    }

    #[test]
    fn one_constraint_present() {
        // Pick a configuration where the constraint is achievable via diagonal flip.
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.5),
            Point::new(2.0, 0.0),
            Point::new(1.0, 1.0),
        ];
        // Bowyer-Watson on these 4 points produces 2 triangles using diagonal (1,3).
        // Constraint (0, 2) is the other diagonal — should be enforceable by flipping.
        let cdt = constrained_delaunay(&pts, &[(0, 2)]).expect("ok");
        assert!(contains_edge(&cdt, 0, 2));
    }
}
