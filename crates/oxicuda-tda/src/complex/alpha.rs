//! Alpha complex filtration via incremental Bowyer-Watson Delaunay triangulation (2D).
//!
//! The alpha complex is a sub-complex of the Delaunay triangulation that is
//! topologically equivalent to the union of balls of radius `r` around each point,
//! yet far sparser than the corresponding Vietoris-Rips complex.
//!
//! **Filtration values:**
//! - 0-simplices (vertices): value 0.
//! - 1-simplices (edges): circumradius of the circumscribed circle of the edge (= half
//!   the edge length).
//! - 2-simplices (triangles): circumradius of the circumscribed circle of the triangle.
//!
//! The Delaunay triangulation is computed with the incremental Bowyer-Watson algorithm:
//! a super-triangle is inserted first, then points are added one by one, and at the end
//! all triangles that share a vertex with the super-triangle are removed.

use crate::error::{TdaError, TdaResult};

// ── 2D primitives ──────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two 2-D points.
#[inline]
fn dist_sq(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

/// Euclidean distance between two 2-D points.
#[inline]
fn dist2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    dist_sq(ax, ay, bx, by).sqrt()
}

/// Circumcircle centre and radius of triangle (ax,ay),(bx,by),(cx,cy).
///
/// Returns `None` when the points are collinear (determinant ≈ 0).
fn circumcircle(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> Option<(f32, f32, f32)> {
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        return None;
    }
    let ax2 = ax * ax + ay * ay;
    let bx2 = bx * bx + by * by;
    let cx2 = cx * cx + cy * cy;
    let ux = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / d;
    let uy = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / d;
    let r = dist2(ux, uy, ax, ay);
    Some((ux, uy, r))
}

// ── Internal triangle representation ──────────────────────────────────────────

/// A triangle in the Bowyer-Watson triangulation, stored as vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Triangle {
    v: [usize; 3],
}

impl Triangle {
    fn new(a: usize, b: usize, c: usize) -> Self {
        Self { v: [a, b, c] }
    }

    /// Return the three directed edges (in canonical `(min,max)` form for easy dedup).
    fn edges(self) -> [(usize, usize); 3] {
        let [a, b, c] = self.v;
        [
            (a.min(b), a.max(b)),
            (b.min(c), b.max(c)),
            (a.min(c), a.max(c)),
        ]
    }

    fn contains_vertex(self, idx: usize) -> bool {
        self.v[0] == idx || self.v[1] == idx || self.v[2] == idx
    }
}

// ── Bowyer-Watson incremental Delaunay ─────────────────────────────────────────

/// Run the Bowyer-Watson algorithm on `pts` (length `n*2`, row-major).
///
/// Super-triangle vertices are appended at indices `n`, `n+1`, `n+2`.
/// Returns the list of triangles with only original vertex indices (super-triangle
/// triangles are discarded).
fn bowyer_watson(pts: &[(f32, f32)]) -> Vec<Triangle> {
    let n = pts.len();

    // Compute a bounding box for the super-triangle.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for &(x, y) in pts {
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let dx = (max_x - min_x).max(1e-6);
    let dy = (max_y - min_y).max(1e-6);
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;

    // Super-triangle vertices (indices n, n+1, n+2).
    let st0 = (mid_x - 20.0 * delta_max, mid_y - delta_max);
    let st1 = (mid_x, mid_y + 20.0 * delta_max);
    let st2 = (mid_x + 20.0 * delta_max, mid_y - delta_max);

    // Augmented point list: original + super-triangle.
    let mut all_pts: Vec<(f32, f32)> = pts.to_vec();
    all_pts.push(st0);
    all_pts.push(st1);
    all_pts.push(st2);
    let st_a = n;
    let st_b = n + 1;
    let st_c = n + 2;

    let mut triangles: Vec<Triangle> = vec![Triangle::new(st_a, st_b, st_c)];

    for (pi, &(px, py)) in pts.iter().enumerate() {
        // Find all triangles whose circumcircle contains point pi.
        let mut bad: Vec<usize> = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            let [a, b, c] = tri.v;
            let (ax, ay) = all_pts[a];
            let (bx, by) = all_pts[b];
            let (cx, cy) = all_pts[c];
            if let Some((ux, uy, r)) = circumcircle(ax, ay, bx, by, cx, cy)
                && dist_sq(ux, uy, px, py) <= r * r + 1e-10
            {
                bad.push(ti);
            }
        }

        // Collect the boundary polygon: edges that appear exactly once.
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for &ti in &bad {
            for e in triangles[ti].edges() {
                *edge_count.entry(e).or_insert(0) += 1;
            }
        }
        let boundary: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|(_, cnt)| *cnt == 1)
            .map(|(e, _)| e)
            .collect();

        // Remove bad triangles (in reverse order to preserve indices).
        let mut sorted_bad = bad.clone();
        sorted_bad.sort_unstable_by(|a, b| b.cmp(a));
        for ti in sorted_bad {
            triangles.swap_remove(ti);
        }

        // Re-triangulate with the new point.
        for (ea, eb) in boundary {
            triangles.push(Triangle::new(pi, ea, eb));
        }
    }

    // Remove triangles that contain a super-triangle vertex.
    triangles.retain(|t| {
        !t.contains_vertex(st_a) && !t.contains_vertex(st_b) && !t.contains_vertex(st_c)
    });

    triangles
}

// ── Alpha filtration ───────────────────────────────────────────────────────────

/// Configuration for building an [`AlphaFiltration`].
#[derive(Debug, Clone)]
pub struct AlphaConfig {
    /// Maximum filtration value; simplices with a larger circumradius are pruned.
    pub max_radius: f32,
    /// Maximum simplex dimension to include (typically 2 for 2D alpha complex).
    pub max_dim: usize,
}

/// An Alpha complex filtration: simplices paired with their circumscribed-circle-radius
/// filtration value, sorted ascending.
///
/// The filtration values are:
/// - Vertices (0-simplex): 0.
/// - Edges (1-simplex): half the edge length (circumradius of the 2-point set).
/// - Triangles (2-simplex): circumradius of the triangle.
#[derive(Debug, Clone)]
pub struct AlphaFiltration {
    simplices: Vec<(Vec<usize>, f32)>,
    cfg: AlphaConfig,
}

impl AlphaFiltration {
    /// Build the alpha filtration from a flat row-major 2D point cloud.
    ///
    /// `points` has length `n * 2` (point `i` at `points[2*i], points[2*i+1]`).
    ///
    /// # Errors
    /// - [`TdaError::EmptyPointCloud`] if `n == 0`.
    /// - [`TdaError::DimensionMismatch`] if `points.len() != n * 2`.
    /// - [`TdaError::ParameterOutOfRange`] if `max_radius < 0`.
    /// - [`TdaError::NanFiltrationValue`] if any coordinate is NaN or infinite.
    pub fn build(points: &[f32], n: usize, cfg: &AlphaConfig) -> TdaResult<Self> {
        if n == 0 || points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        if points.len() != n * 2 {
            return Err(TdaError::DimensionMismatch {
                expected: n * 2,
                got: points.len(),
            });
        }
        if cfg.max_radius < 0.0 {
            return Err(TdaError::ParameterOutOfRange(
                "max_radius must be non-negative".to_owned(),
            ));
        }
        for &v in points {
            if !v.is_finite() {
                return Err(TdaError::NanFiltrationValue);
            }
        }

        // Convert to (x, y) pairs.
        let pts: Vec<(f32, f32)> = (0..n).map(|i| (points[2 * i], points[2 * i + 1])).collect();

        let mut simplices: Vec<(Vec<usize>, f32)> = Vec::new();

        // 0-simplices: all vertices at value 0.
        for i in 0..n {
            simplices.push((vec![i], 0.0));
        }

        if cfg.max_dim == 0 {
            // Sorted order is already satisfied (all value 0).
            return Ok(Self {
                simplices,
                cfg: cfg.clone(),
            });
        }

        // For n < 3 there are no triangles; handle degenerate cases directly.
        // n == 1: no edges.
        // n == 2: exactly one edge.
        if n == 2 {
            if cfg.max_dim >= 1 {
                let (ax, ay) = pts[0];
                let (bx, by) = pts[1];
                let val = dist2(ax, ay, bx, by) * 0.5;
                if val <= cfg.max_radius {
                    simplices.push((vec![0, 1], val));
                }
            }
            simplices.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.len().cmp(&b.0.len()))
                    .then_with(|| a.0.cmp(&b.0))
            });
            return Ok(Self {
                simplices,
                cfg: cfg.clone(),
            });
        }

        // Run Bowyer-Watson to get the Delaunay triangles.
        let triangles = bowyer_watson(&pts);

        // Collect all Delaunay edges from triangles.
        let mut delaunay_edges: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for tri in &triangles {
            for e in tri.edges() {
                delaunay_edges.insert(e);
            }
        }

        // Build a map: edge -> list of triangles that share it (for coloring edges with
        // the circumradius of their containing triangle when they are interior).
        // For the alpha filtration, every Delaunay edge gets filtration value = half-edge-length.
        // (Using the circumradius of the edge = half-distance is standard for alpha complexes.)

        // 1-simplices: all Delaunay edges.
        if cfg.max_dim >= 1 {
            for &(a, b) in &delaunay_edges {
                let (ax, ay) = pts[a];
                let (bx, by) = pts[b];
                let val = dist2(ax, ay, bx, by) * 0.5;
                if val <= cfg.max_radius {
                    simplices.push((vec![a, b], val));
                }
            }
        }

        // 2-simplices: Delaunay triangles with their circumradius.
        if cfg.max_dim >= 2 {
            for tri in &triangles {
                let mut v = tri.v;
                v.sort_unstable();
                let (ax, ay) = pts[v[0]];
                let (bx, by) = pts[v[1]];
                let (cx, cy) = pts[v[2]];
                let val = if let Some((_, _, r)) = circumcircle(ax, ay, bx, by, cx, cy) {
                    r
                } else {
                    // Degenerate collinear triangle — use longest half-edge.
                    let e0 = dist2(ax, ay, bx, by);
                    let e1 = dist2(bx, by, cx, cy);
                    let e2 = dist2(ax, ay, cx, cy);
                    e0.max(e1).max(e2) * 0.5
                };
                if val <= cfg.max_radius {
                    simplices.push((v.to_vec(), val));
                }
            }
        }

        // Sort ascending by value; ties: lower dimension first, then lexicographic.
        simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        Ok(Self {
            simplices,
            cfg: cfg.clone(),
        })
    }

    /// The simplices sorted ascending by filtration value.
    pub fn simplices(&self) -> &[(Vec<usize>, f32)] {
        &self.simplices
    }

    /// Total number of simplices in the filtration.
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// The largest filtration value present (0 if no simplices or only vertices).
    pub fn max_value(&self) -> f32 {
        self.simplices
            .iter()
            .map(|(_, v)| *v)
            .fold(0.0_f32, f32::max)
    }

    /// The configuration this filtration was built with.
    pub fn config(&self) -> &AlphaConfig {
        &self.cfg
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> AlphaConfig {
        AlphaConfig {
            max_radius: 1e6,
            max_dim: 2,
        }
    }

    // 1. All vertices appear at filtration value 0.
    #[test]
    fn vertices_at_zero() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.5, 1.0];
        let filt = AlphaFiltration::build(&pts, 3, &default_cfg()).expect("build");
        let n_zero_verts = filt
            .simplices()
            .iter()
            .filter(|(v, val)| v.len() == 1 && val.abs() < 1e-6)
            .count();
        assert_eq!(n_zero_verts, 3, "all 3 vertices must be at value 0");
    }

    // 2. Two-point case: single edge value = half-length.
    #[test]
    fn edge_filtration_half_length_two_points() {
        // Two points distance 4 apart.
        let pts = vec![0.0_f32, 0.0, 4.0, 0.0];
        let filt = AlphaFiltration::build(&pts, 2, &default_cfg()).expect("build");
        let edge = filt
            .simplices()
            .iter()
            .find(|(v, _)| v.len() == 2)
            .expect("edge");
        assert!(
            (edge.1 - 2.0).abs() < 1e-5,
            "edge value should be 2 (half of 4)"
        );
    }

    // 3. Filtration is sorted ascending.
    #[test]
    fn sorted_ascending() {
        let pts = vec![0.0_f32, 0.0, 2.0, 0.0, 1.0, 1.5, 3.0, 1.0];
        let filt = AlphaFiltration::build(&pts, 4, &default_cfg()).expect("build");
        for w in filt.simplices().windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-6,
                "filtration not sorted at {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
    }

    // 4. max_radius pruning removes high-value simplices.
    #[test]
    fn max_radius_prunes_simplices() {
        let pts = vec![0.0_f32, 0.0, 10.0, 0.0];
        let cfg = AlphaConfig {
            max_radius: 1.0,
            max_dim: 2,
        };
        let filt = AlphaFiltration::build(&pts, 2, &cfg).expect("build");
        // Edge has half-length = 5 > 1.0, so only vertices remain.
        assert!(
            filt.simplices().iter().all(|(v, _)| v.len() == 1),
            "high-radius edge should be pruned"
        );
        assert_eq!(filt.n_simplices(), 2);
    }

    // 5. Equilateral triangle: circumradius = side / sqrt(3).
    #[test]
    fn equilateral_triangle_circumradius() {
        let s = 2.0_f32;
        let h = s * (3.0_f32).sqrt() / 2.0;
        let pts = vec![0.0_f32, 0.0, s, 0.0, s * 0.5, h];
        let filt = AlphaFiltration::build(&pts, 3, &default_cfg()).expect("build");
        let tri = filt
            .simplices()
            .iter()
            .find(|(v, _)| v.len() == 3)
            .expect("triangle");
        let expected = s / (3.0_f32).sqrt();
        assert!(
            (tri.1 - expected).abs() < 1e-3,
            "circumradius {} != expected {}",
            tri.1,
            expected
        );
    }

    // 6. Build is deterministic (same input -> same output).
    #[test]
    fn deterministic() {
        let pts = vec![0.1_f32, 0.2, 0.9, 0.3, 0.5, 0.8, 0.3, 0.7, 0.7, 0.1];
        let a = AlphaFiltration::build(&pts, 5, &default_cfg()).expect("build a");
        let b = AlphaFiltration::build(&pts, 5, &default_cfg()).expect("build b");
        assert_eq!(a.simplices(), b.simplices(), "must be deterministic");
    }

    // 7. Error: empty point cloud.
    #[test]
    fn error_empty_point_cloud() {
        let pts: Vec<f32> = vec![];
        assert!(AlphaFiltration::build(&pts, 0, &default_cfg()).is_err());
    }

    // 8. Error: wrong length (n * 2 != pts.len()).
    #[test]
    fn error_wrong_length() {
        let pts = vec![0.0_f32, 0.0, 1.0]; // 3 values, n=2 wants 4
        let result = AlphaFiltration::build(&pts, 2, &default_cfg());
        assert!(result.is_err());
    }

    // 9. Error: negative max_radius.
    #[test]
    fn error_negative_max_radius() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0];
        let cfg = AlphaConfig {
            max_radius: -1.0,
            max_dim: 2,
        };
        assert!(AlphaFiltration::build(&pts, 2, &cfg).is_err());
    }

    // 10. max_dim=0 returns only vertices.
    #[test]
    fn max_dim_zero_vertices_only() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.5, 1.0];
        let cfg = AlphaConfig {
            max_radius: 1e6,
            max_dim: 0,
        };
        let filt = AlphaFiltration::build(&pts, 3, &cfg).expect("build");
        assert_eq!(filt.n_simplices(), 3);
        assert!(filt.simplices().iter().all(|(v, _)| v.len() == 1));
    }

    // 11. Single point: one vertex at value 0, no edges.
    #[test]
    fn single_point() {
        let pts = vec![1.5_f32, 2.5];
        let filt = AlphaFiltration::build(&pts, 1, &default_cfg()).expect("build");
        assert_eq!(filt.n_simplices(), 1);
        assert_eq!(filt.simplices()[0].0, vec![0]);
        assert!((filt.simplices()[0].1).abs() < 1e-6);
    }

    // 12. n_simplices and max_value accessors work.
    #[test]
    fn accessors_work() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let filt = AlphaFiltration::build(&pts, 3, &default_cfg()).expect("build");
        assert!(filt.n_simplices() > 0);
        assert!(filt.max_value() >= 0.0);
        assert_eq!(filt.config().max_dim, 2);
    }
}
