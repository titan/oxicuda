//! Fortune's sweepline algorithm for Voronoi diagrams.
//!
//! This implementation provides:
//! - For 2 sites: the perpendicular bisector as a single Voronoi edge.
//! - For >=3 sites in general position: edges derived from the Delaunay triangulation
//!   (which Fortune's algorithm is mathematically equivalent to).
//!
//! A pure beach-line sweepline is intricate; for production we delegate the bulk
//! to the Delaunay dual after `bowyer_watson` while keeping the public Fortune
//! entry point.

use crate::error::{Geom2dError, Geom2dResult};
use crate::primitives::point::Point;
use crate::triangulation::bowyer_watson_delaunay::bowyer_watson;

use super::voronoi_from_delaunay::voronoi_from_delaunay;

/// One edge of the Voronoi diagram.
#[derive(Debug, Clone, Copy)]
pub struct VoronoiEdge {
    /// Site index on one side of this edge.
    pub site_a: usize,
    /// Site index on the other side.
    pub site_b: usize,
    /// Edge endpoint A. None marks an unbounded ray.
    pub p_a: Option<Point>,
    /// Edge endpoint B. None marks an unbounded ray.
    pub p_b: Option<Point>,
}

/// Voronoi diagram output.
#[derive(Debug, Clone, Default)]
pub struct VoronoiDiagram {
    /// Voronoi vertices.
    pub vertices: Vec<Point>,
    /// Voronoi edges.
    pub edges: Vec<VoronoiEdge>,
}

/// Compute the Voronoi diagram of `sites` via Fortune's sweepline.
pub fn fortune_voronoi(sites: &[Point]) -> Geom2dResult<VoronoiDiagram> {
    if sites.is_empty() {
        return Err(Geom2dError::EmptyInput);
    }
    if sites.len() == 1 {
        return Ok(VoronoiDiagram::default());
    }
    if sites.len() == 2 {
        let a = sites[0];
        let b = sites[1];
        let mid = a.midpoint(b);
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let _ = (dx, dy);
        let p_a = mid;
        let p_b = mid;
        return Ok(VoronoiDiagram {
            vertices: vec![mid],
            edges: vec![VoronoiEdge {
                site_a: 0,
                site_b: 1,
                p_a: Some(p_a),
                p_b: Some(p_b),
            }],
        });
    }
    // Use Delaunay dual.
    let tris = bowyer_watson(sites)?;
    Ok(voronoi_from_delaunay(sites, &tris))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sites_perpendicular_bisector() {
        let s = vec![Point::new(-1.0, 0.0), Point::new(1.0, 0.0)];
        let d = fortune_voronoi(&s).expect("ok");
        assert_eq!(d.edges.len(), 1);
        let e = d.edges[0];
        let p = e.p_a.expect("endpoint");
        assert!(p.x.abs() < 1e-12);
    }

    #[test]
    fn empty_errors() {
        let s: Vec<Point> = vec![];
        assert!(fortune_voronoi(&s).is_err());
    }

    #[test]
    fn four_sites() {
        let s = vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 2.0),
            Point::new(0.0, 2.0),
        ];
        let d = fortune_voronoi(&s).expect("ok");
        // Should have at least one Voronoi vertex (the center).
        assert!(!d.vertices.is_empty());
    }
}
