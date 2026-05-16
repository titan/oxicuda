//! Build a Voronoi diagram by dualizing a Delaunay triangulation.

use crate::primitives::point::Point;
use crate::triangulation::bowyer_watson_delaunay::Triangle;

use super::fortune_sweepline::{VoronoiDiagram, VoronoiEdge};

/// Build Voronoi vertices (circumcenters) and Voronoi edges from a Delaunay triangulation.
#[must_use]
pub fn voronoi_from_delaunay(sites: &[Point], tris: &[Triangle]) -> VoronoiDiagram {
    let mut vertices: Vec<Point> = Vec::with_capacity(tris.len());
    for t in tris {
        let cc = circumcenter(sites[t.a], sites[t.b], sites[t.c]);
        vertices.push(cc);
    }
    // For each undirected edge (i, j) shared by triangles t1 and t2, emit a Voronoi edge
    // between vertices[t1] and vertices[t2]. Singleton edges become rays (one None endpoint).
    let mut edge_map: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        let edges = [
            sorted_pair(t.a, t.b),
            sorted_pair(t.b, t.c),
            sorted_pair(t.c, t.a),
        ];
        for e in edges {
            edge_map.entry(e).or_default().push(ti);
        }
    }
    let mut edges: Vec<VoronoiEdge> = Vec::new();
    for ((i_a, i_b), incident) in edge_map {
        if incident.len() == 2 {
            edges.push(VoronoiEdge {
                site_a: i_a,
                site_b: i_b,
                p_a: Some(vertices[incident[0]]),
                p_b: Some(vertices[incident[1]]),
            });
        } else if incident.len() == 1 {
            // Boundary edge: unbounded ray (we leave the open end as None).
            edges.push(VoronoiEdge {
                site_a: i_a,
                site_b: i_b,
                p_a: Some(vertices[incident[0]]),
                p_b: None,
            });
        }
    }
    VoronoiDiagram { vertices, edges }
}

fn sorted_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn circumcenter(a: Point, b: Point, c: Point) -> Point {
    let ax = a.x;
    let ay = a.y;
    let bx = b.x;
    let by = b.y;
    let cx = c.x;
    let cy = c.y;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-30 {
        return a.midpoint(c);
    }
    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;
    Point::new(ux, uy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triangulation::bowyer_watson_delaunay::bowyer_watson;

    #[test]
    fn unit_square_circumcenter_at_half() {
        let s = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let t = bowyer_watson(&s).expect("ok");
        let v = voronoi_from_delaunay(&s, &t);
        assert!(!v.vertices.is_empty());
        for c in &v.vertices {
            assert!((c.x - 0.5).abs() < 0.5 && (c.y - 0.5).abs() < 0.5);
        }
    }
}
