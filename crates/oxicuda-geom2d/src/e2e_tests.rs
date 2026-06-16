//! End-to-end integration tests for `oxicuda-geom2d`.

use crate::clipping::sutherland_hodgman::sutherland_hodgman;
use crate::closest_pair::brute_force::closest_pair_brute;
use crate::closest_pair::divide_conquer::closest_pair_dc;
use crate::containment::point_in_polygon_ray_cast::point_in_polygon_ray_cast;
use crate::containment::point_in_polygon_winding::point_in_polygon_winding;
use crate::enclosing::welzl_smallest_circle::welzl_smallest_circle;
use crate::handle::LcgRng;
use crate::hull::andrew_monotone_chain::andrew_monotone_chain;
use crate::hull::graham_scan::graham_scan;
use crate::hull::quickhull::quickhull;
use crate::index::kd_tree_2d::KdTree2d;
use crate::intersection::segment_segment::{SegmentSegmentIntersection, intersect_segments};
use crate::polygon_ops::centroid::polygon_centroid;
use crate::predicate::orientation::{Orientation, orient};
use crate::primitives::point::Point;
use crate::primitives::polygon::Polygon;
use crate::primitives::segment::Segment;
use crate::ptx_kernels::{
    convex_hull_step_ptx, cross_product_ptx, kd_tree_traverse_ptx, orientation_test_ptx,
    point_in_aabb_ptx, polygon_area_ptx, segment_intersection_ptx,
};
use crate::sweepline::bentley_ottmann::bentley_ottmann;
use crate::triangulation::bowyer_watson_delaunay::bowyer_watson;
use crate::voronoi::fortune_sweepline::fortune_voronoi;

fn unit_square() -> Polygon {
    Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
    ])
    .expect("ok")
}

// 1. CCW orientation for (0,0), (1,0), (0,1)
#[test]
fn e2e_orientation_ccw() {
    let o = orient(
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
    );
    assert_eq!(o, Orientation::Ccw);
}

// 2. Point-in-polygon (winding) — unit square contains (0.5, 0.5)
#[test]
fn e2e_winding_inside_center() {
    assert!(point_in_polygon_winding(
        &unit_square(),
        Point::new(0.5, 0.5)
    ));
}

// 3. Point-in-polygon (winding) — unit square does not contain (2, 2)
#[test]
fn e2e_winding_outside_far() {
    assert!(!point_in_polygon_winding(
        &unit_square(),
        Point::new(2.0, 2.0)
    ));
}

// 4. Point-in-polygon (ray cast) — same as winding
#[test]
fn e2e_ray_cast_inside_center() {
    assert!(point_in_polygon_ray_cast(
        &unit_square(),
        Point::new(0.5, 0.5)
    ));
    assert!(!point_in_polygon_ray_cast(
        &unit_square(),
        Point::new(2.0, 2.0)
    ));
}

// 5. Graham, Andrew, QuickHull agree on 5 points
#[test]
fn e2e_hulls_agree() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(0.5, 0.5),
    ];
    let g = graham_scan(&pts).expect("ok");
    let a = andrew_monotone_chain(&pts).expect("ok");
    let q = quickhull(&pts).expect("ok");
    assert_eq!(g.len(), 4);
    assert_eq!(a.len(), 4);
    assert_eq!(q.len(), 4);
}

// 6. Segment-segment intersection of two diagonals at (1,1)
#[test]
fn e2e_seg_seg_intersection() {
    let s1 = Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 2.0));
    let s2 = Segment::new(Point::new(0.0, 2.0), Point::new(2.0, 0.0));
    match intersect_segments(s1, s2) {
        SegmentSegmentIntersection::Point(p) => {
            assert!((p.x - 1.0).abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
        }
        other => panic!("expected Point at (1,1), got {other:?}"),
    }
}

// 7. Shoelace area of unit square = 1
#[test]
fn e2e_shoelace_area_unit_square() {
    assert!((unit_square().area() - 1.0).abs() < 1e-15);
}

// 8. Centroid of unit square = (0.5, 0.5)
#[test]
fn e2e_centroid_unit_square() {
    let c = polygon_centroid(&unit_square()).expect("ok");
    assert!((c.x - 0.5).abs() < 1e-12 && (c.y - 0.5).abs() < 1e-12);
}

// 9. Welzl smallest circle of unit square has radius sqrt(2)/2
#[test]
fn e2e_welzl_unit_square() {
    let pts = unit_square().vertices.clone();
    let c = welzl_smallest_circle(&pts, 7).expect("ok");
    let r = 2_f64.sqrt() / 2.0;
    assert!((c.radius - r).abs() < 1e-10);
}

// 10. Bowyer-Watson rejects 3 collinear points as degenerate
#[test]
fn e2e_bowyer_collinear_degenerate() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(2.0, 0.0),
    ];
    assert!(bowyer_watson(&pts).is_err());
}

// 11. Bowyer-Watson on 4 general-position points returns 2 triangles
#[test]
fn e2e_bowyer_4_general_2_triangles() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
    ];
    let t = bowyer_watson(&pts).expect("ok");
    assert_eq!(t.len(), 2);
}

// 12. Closest pair of unit-square corners = 1
#[test]
fn e2e_closest_pair_unit_square() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
        Point::new(1.0, 0.0),
    ];
    let (_, _, d_b) = closest_pair_brute(&pts).expect("ok");
    let (_, _, d_d) = closest_pair_dc(&pts).expect("ok");
    assert!((d_b - 1.0).abs() < 1e-12);
    assert!((d_d - 1.0).abs() < 1e-12);
}

// 13. Sutherland-Hodgman clipping a square against an offset square
#[test]
fn e2e_sutherland_hodgman_offset_squares() {
    let sq1 = unit_square();
    let sq2 = Polygon::new(vec![
        Point::new(0.5, 0.5),
        Point::new(1.5, 0.5),
        Point::new(1.5, 1.5),
        Point::new(0.5, 1.5),
    ])
    .expect("ok");
    let r = sutherland_hodgman(&sq1, &sq2).expect("ok");
    assert!((r.area() - 0.25).abs() < 1e-10);
}

// 14. Fortune Voronoi on 2 points returns the perpendicular bisector
#[test]
fn e2e_voronoi_two_sites_bisector() {
    let s = vec![Point::new(-1.0, 0.0), Point::new(1.0, 0.0)];
    let d = fortune_voronoi(&s).expect("ok");
    assert_eq!(d.edges.len(), 1);
    let p = d.edges[0].p_a.expect("endpoint");
    assert!(p.x.abs() < 1e-12);
}

// 15. Bentley-Ottmann finds intersections in cross pattern
#[test]
fn e2e_bentley_ottmann_cross() {
    // Four segments forming distinct pairwise intersections (not concurrent).
    let segs = vec![
        Segment::new(Point::new(0.0, 0.0), Point::new(4.0, 4.0)),
        Segment::new(Point::new(0.0, 4.0), Point::new(4.0, 0.0)),
        Segment::new(Point::new(0.5, -1.0), Point::new(0.5, 5.0)),
        Segment::new(Point::new(-1.0, 3.0), Point::new(5.0, 3.0)),
    ];
    let pts = bentley_ottmann(&segs);
    assert!(pts.len() >= 4);
}

// 16. KD-tree kNN agrees with brute force on n=20 points
#[test]
fn e2e_kdtree_agrees_with_brute() {
    let mut r = LcgRng::new(7);
    let n = 20;
    let pts: Vec<Point> = (0..n)
        .map(|_| Point::new(r.next_f64() * 10.0, r.next_f64() * 10.0))
        .collect();
    let kd = KdTree2d::build(&pts);
    let q = Point::new(5.0, 5.0);
    let knn = kd.knn(q, 1);
    let (b_idx, _, _) = closest_pair_brute_for_query(&pts, q);
    assert_eq!(knn[0].0, b_idx);
}

fn closest_pair_brute_for_query(pts: &[Point], q: Point) -> (usize, f64, f64) {
    let mut idx = 0;
    let mut bd = pts[0].distance_sq(q);
    for (i, &p) in pts.iter().enumerate().skip(1) {
        let d = p.distance_sq(q);
        if d < bd {
            bd = d;
            idx = i;
        }
    }
    (idx, bd, bd.sqrt())
}

type PtxKernel = (&'static str, fn(u32) -> String);

// 17. All 7 PTX kernels produce non-empty PTX for every SM
#[test]
fn e2e_ptx_all_kernels_all_sm() {
    let kernels: [PtxKernel; 7] = [
        ("orientation_test", orientation_test_ptx),
        ("cross_product", cross_product_ptx),
        ("point_in_aabb", point_in_aabb_ptx),
        ("segment_intersection", segment_intersection_ptx),
        ("convex_hull_step", convex_hull_step_ptx),
        ("kd_tree_traverse", kd_tree_traverse_ptx),
        ("polygon_area", polygon_area_ptx),
    ];
    for sm in [75u32, 80, 86, 89, 90, 100] {
        for (n, f) in &kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "kernel {n} sm={sm} empty");
            assert!(s.contains(".visible .entry"));
        }
    }
}

// 18. Welzl on collinear points still returns valid circle
#[test]
fn e2e_welzl_collinear() {
    let pts = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(2.0, 0.0),
    ];
    let c = welzl_smallest_circle(&pts, 1).expect("ok");
    assert!((c.center.x - 1.0).abs() < 1e-10);
    assert!((c.radius - 1.0).abs() < 1e-10);
}

// 19. Voronoi from 4-site square has at least 1 vertex
#[test]
fn e2e_voronoi_four_sites_has_vertex() {
    let s = vec![
        Point::new(0.0, 0.0),
        Point::new(2.0, 0.0),
        Point::new(2.0, 2.0),
        Point::new(0.0, 2.0),
    ];
    let d = fortune_voronoi(&s).expect("ok");
    assert!(!d.vertices.is_empty());
}

// 20. Closest-pair DC and brute force agree on random n=30
#[test]
fn e2e_closest_pair_dc_brute_agree() {
    let mut r = LcgRng::new(13);
    let n = 30;
    let pts: Vec<Point> = (0..n)
        .map(|_| Point::new(r.next_f64() * 10.0, r.next_f64() * 10.0))
        .collect();
    let (_, _, d_b) = closest_pair_brute(&pts).expect("ok");
    let (_, _, d_d) = closest_pair_dc(&pts).expect("ok");
    assert!((d_b - d_d).abs() < 1e-12);
}

// 21. Greiner-Hormann area identity vs shoelace: A∩B + A∪B == A + B exactly.
#[test]
fn e2e_greiner_hormann_area_identity() {
    let a = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(3.0, 0.0),
        Point::new(3.0, 3.0),
        Point::new(0.0, 3.0),
    ])
    .expect("ok");
    let b = Polygon::new(vec![
        Point::new(1.5, 1.5),
        Point::new(4.5, 1.5),
        Point::new(4.5, 4.5),
        Point::new(1.5, 4.5),
    ])
    .expect("ok");
    let inter = crate::clipping::greiner_hormann::intersection(&a, &b).expect("ok");
    let uni = crate::clipping::greiner_hormann::union(&a, &b).expect("ok");
    let lhs = crate::clipping::greiner_hormann::filled_area_of_rings(&inter)
        + crate::clipping::greiner_hormann::filled_area_of_rings(&uni);
    let rhs = a.area() + b.area();
    assert!((lhs - rhs).abs() < 1e-9, "lhs={lhs}, rhs={rhs}");
}

// 22. Greiner-Hormann intersection agrees with Sutherland-Hodgman for a convex
//     clip (cross-validation against the existing convex clipper).
#[test]
fn e2e_greiner_hormann_matches_sutherland_hodgman() {
    let subj = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(2.0, 0.0),
        Point::new(2.0, 2.0),
        Point::new(0.0, 2.0),
    ])
    .expect("ok");
    let clip = Polygon::new(vec![
        Point::new(1.0, 1.0),
        Point::new(3.0, 1.0),
        Point::new(3.0, 3.0),
        Point::new(1.0, 3.0),
    ])
    .expect("ok");
    let gh = crate::clipping::greiner_hormann::intersection(&subj, &clip).expect("ok");
    let sh = sutherland_hodgman(&subj, &clip).expect("ok");
    let gh_area: f64 = gh.iter().map(Polygon::area).sum();
    assert!(
        (gh_area - sh.area()).abs() < 1e-9,
        "gh={gh_area}, sh={}",
        sh.area()
    );
}

// 23. Alpha-shape with very large alpha recovers the convex-hull boundary
//     vertices exactly (cross-check against Andrew's monotone chain). Uses a
//     general-position seed (no points collinear on a hull edge).
#[test]
fn e2e_alpha_shape_recovers_hull() {
    let mut r = LcgRng::new(11);
    let pts: Vec<Point> = (0..50)
        .map(|_| Point::new(r.next_f64() * 10.0, r.next_f64() * 10.0))
        .collect();
    let shape = crate::alpha_shape::alpha_shape(&pts, 1.0e9).expect("ok");
    let mut boundary_v: Vec<usize> = shape
        .boundary_edges
        .iter()
        .flat_map(|e| [e[0], e[1]])
        .collect();
    boundary_v.sort_unstable();
    boundary_v.dedup();

    let hull = andrew_monotone_chain(&pts).expect("ok");
    let mut hull_idx: Vec<usize> = hull
        .iter()
        .map(|hp| {
            pts.iter()
                .position(|p| (p.x - hp.x).abs() < 1e-9 && (p.y - hp.y).abs() < 1e-9)
                .expect("hull point is input")
        })
        .collect();
    hull_idx.sort_unstable();
    hull_idx.dedup();
    assert_eq!(boundary_v, hull_idx);
}

// 24. Half-plane intersection of a polygon's CCW edges reconstructs that polygon
//     (area matches shoelace; cross-check with the hull of the same points).
#[test]
fn e2e_half_plane_reconstructs_polygon() {
    use crate::halfplane::{HalfPlane, HalfPlaneRegion, half_plane_intersection};
    let verts = [
        Point::new(0.0, 0.0),
        Point::new(4.0, 0.0),
        Point::new(4.0, 3.0),
        Point::new(0.0, 3.0),
    ];
    let mut planes = Vec::new();
    for i in 0..verts.len() {
        let from = verts[i];
        let to = verts[(i + 1) % verts.len()];
        planes.push(HalfPlane::from_directed_edge(from, to).expect("edge"));
    }
    match half_plane_intersection(&planes).expect("ok") {
        HalfPlaneRegion::Polygon(p) => {
            assert!((p.area() - 12.0).abs() < 1e-7);
            // Every vertex satisfies every constraint.
            for v in &p.vertices {
                for h in &planes {
                    assert!(h.c - (h.a * v.x + h.b * v.y) >= -1e-6);
                }
            }
        }
        other => panic!("expected bounded polygon, got {other:?}"),
    }
}
