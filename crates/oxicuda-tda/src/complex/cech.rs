//! Čech filtration via the minimum enclosing ball (MEB) radius per simplex.
//!
//! The Čech complex at scale `r` contains a simplex `σ` whenever the closed balls of
//! radius `r` centred at the vertices of `σ` have a common intersection.  By Helly's
//! theorem (in the nerve sense) this is equivalent to the minimum enclosing ball of the
//! vertex set having radius ≤ `r`.  We therefore assign every simplex a filtration value
//! equal to the radius of the MEB of its vertices, which yields the *Čech filtration*.
//!
//! The MEB is computed with the deterministic Bădoiu–Clarkson iterative algorithm
//! (Bădoiu & Clarkson, "Smaller core-sets for balls", 2003): starting from the centroid,
//! the candidate centre is repeatedly nudged towards the currently farthest point with a
//! shrinking step `1/(i+2)`.  No randomness is used, so results are fully reproducible.
//! Exact closed forms are used for the trivial 1-point and 2-point cases.

use crate::error::{TdaError, TdaResult};

/// Number of Bădoiu–Clarkson refinement iterations.
///
/// The Bădoiu–Clarkson recurrence converges to the true MEB radius at rate `O(1/iter)`.
/// 1000 iterations bring the radius to within roughly `1e-3` of optimal for the small
/// vertex sets enumerated in a Čech complex (a simplex has at most `max_dim + 1`
/// vertices), while remaining negligibly cheap and fully deterministic.  The estimated
/// centre converges somewhat more slowly than the radius, which is intrinsic to the
/// algorithm.
const BC_ITERATIONS: usize = 1000;

/// Compute the minimum enclosing ball of a point set.
///
/// `points` is a slice of points, each represented as a `Vec<f32>` of identical length
/// (the ambient dimension).  Returns `(center, radius)` where `center` has the same
/// dimension as the inputs.
///
/// # Algorithm
/// - 0 points: error (`EmptyPointCloud`).
/// - 1 point: the point itself with radius `0`.
/// - 2 points: the midpoint with radius equal to half the distance.
/// - ≥ 3 points: the deterministic Bădoiu–Clarkson iteration seeded at the centroid.
///
/// # Errors
/// Returns [`TdaError::EmptyPointCloud`] for an empty list and
/// [`TdaError::DimensionMismatch`] if the points do not all share the same dimension or
/// have zero dimension.
pub fn minimum_enclosing_ball(points: &[Vec<f32>]) -> TdaResult<(Vec<f32>, f32)> {
    if points.is_empty() {
        return Err(TdaError::EmptyPointCloud);
    }
    let dim = points[0].len();
    if dim == 0 {
        return Err(TdaError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    for p in points {
        if p.len() != dim {
            return Err(TdaError::DimensionMismatch {
                expected: dim,
                got: p.len(),
            });
        }
    }

    // Exact closed form: single point.
    if points.len() == 1 {
        return Ok((points[0].clone(), 0.0));
    }

    // Exact closed form: two points -> midpoint, half-distance.
    if points.len() == 2 {
        let mut center = vec![0.0_f32; dim];
        let mut dist_sq = 0.0_f32;
        for d in 0..dim {
            let a = points[0][d];
            let b = points[1][d];
            center[d] = 0.5 * (a + b);
            let diff = a - b;
            dist_sq += diff * diff;
        }
        return Ok((center, 0.5 * dist_sq.sqrt()));
    }

    // General case: Bădoiu–Clarkson iteration starting from the centroid.
    let n = points.len();
    let inv_n = 1.0_f32 / (n as f32);
    let mut center = vec![0.0_f32; dim];
    for p in points {
        for d in 0..dim {
            center[d] += p[d];
        }
    }
    for c in center.iter_mut() {
        *c *= inv_n;
    }

    for i in 0..BC_ITERATIONS {
        // Locate the point farthest from the current centre.
        let mut far_idx = 0usize;
        let mut far_dist_sq = -1.0_f32;
        for (idx, p) in points.iter().enumerate() {
            let mut dist_sq = 0.0_f32;
            for d in 0..dim {
                let diff = p[d] - center[d];
                dist_sq += diff * diff;
            }
            if dist_sq > far_dist_sq {
                far_dist_sq = dist_sq;
                far_idx = idx;
            }
        }
        // Move the centre a step of 1/(i+2) towards the farthest point.
        let step = 1.0_f32 / ((i + 2) as f32);
        let far = &points[far_idx];
        for d in 0..dim {
            center[d] += step * (far[d] - center[d]);
        }
    }

    // Final radius = distance to the farthest point from the converged centre.
    let mut radius_sq = 0.0_f32;
    for p in points {
        let mut dist_sq = 0.0_f32;
        for d in 0..dim {
            let diff = p[d] - center[d];
            dist_sq += diff * diff;
        }
        if dist_sq > radius_sq {
            radius_sq = dist_sq;
        }
    }

    Ok((center, radius_sq.sqrt()))
}

/// Configuration for building a [`CechFiltration`].
#[derive(Debug, Clone)]
pub struct CechConfig {
    /// Maximum simplex dimension to enumerate (0 = vertices only).
    pub max_dim: usize,
    /// Maximum MEB radius; simplices with a larger radius are pruned.
    pub max_radius: f32,
}

/// A Čech filtration: simplices paired with their MEB-radius filtration value.
///
/// The simplices are stored sorted ascending by filtration value, with ties broken by
/// dimension (lower first) and then lexicographically on the vertex list.
#[derive(Debug, Clone)]
pub struct CechFiltration {
    simplices: Vec<(Vec<usize>, f32)>,
    cfg: CechConfig,
}

impl CechFiltration {
    /// Build the Čech filtration from a flat row-major point cloud.
    ///
    /// `points` has length `n * dim` (point `i`, coordinate `c` at index `i * dim + c`).
    /// Vertices (0-simplices) are assigned filtration value `0`.  For each dimension
    /// `k = 1..=max_dim`, every `(k + 1)`-subset of the points is enumerated, its MEB
    /// radius computed, and the simplex retained iff that radius is ≤ `max_radius`.
    ///
    /// # Errors
    /// - [`TdaError::EmptyPointCloud`] if `n == 0` or `points` is empty.
    /// - [`TdaError::DimensionMismatch`] if `dim == 0` or `points.len() != n * dim`.
    /// - [`TdaError::ParameterOutOfRange`] if `max_radius < 0`.
    pub fn build(points: &[f32], n: usize, dim: usize, cfg: &CechConfig) -> TdaResult<Self> {
        if n == 0 || points.is_empty() {
            return Err(TdaError::EmptyPointCloud);
        }
        if dim == 0 {
            return Err(TdaError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if points.len() != n * dim {
            return Err(TdaError::DimensionMismatch {
                expected: n * dim,
                got: points.len(),
            });
        }
        if cfg.max_radius < 0.0 {
            return Err(TdaError::ParameterOutOfRange(
                "max_radius must be non-negative".to_owned(),
            ));
        }

        // Materialise points as per-point coordinate vectors for the MEB routine.
        let coords: Vec<Vec<f32>> = (0..n)
            .map(|i| points[i * dim..(i + 1) * dim].to_vec())
            .collect();

        let mut simplices: Vec<(Vec<usize>, f32)> = Vec::new();

        // 0-simplices: every vertex at filtration value 0.
        for i in 0..n {
            simplices.push((vec![i], 0.0));
        }

        // Higher-dimensional simplices via lexicographic subset enumeration.
        for k in 1..=cfg.max_dim {
            let size = k + 1;
            if size > n {
                break;
            }
            let mut indices: Vec<usize> = (0..size).collect();
            loop {
                let subset: Vec<Vec<f32>> =
                    indices.iter().map(|&idx| coords[idx].clone()).collect();
                let (_, radius) = minimum_enclosing_ball(&subset)?;
                if radius <= cfg.max_radius {
                    simplices.push((indices.clone(), radius));
                }
                if !next_combination(&mut indices, n) {
                    break;
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

    /// The largest filtration value present (`0` if the filtration is empty).
    pub fn max_value(&self) -> f32 {
        self.simplices
            .iter()
            .map(|(_, v)| *v)
            .fold(0.0_f32, f32::max)
    }

    /// The configuration this filtration was built with.
    pub fn config(&self) -> &CechConfig {
        &self.cfg
    }
}

/// Advance `indices` to the next k-combination of `{0, …, n-1}` in lexicographic order.
/// Returns `false` once `indices` is the final combination.
fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        if indices[i] < n - (k - i) {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Euclidean distance between two equal-length coordinate vectors.
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn meb_single_point_radius_zero() {
        let (center, radius) = minimum_enclosing_ball(&[vec![3.0, -1.0]]).expect("meb");
        assert!(radius.abs() < 1e-6, "single-point radius should be 0");
        assert_eq!(center, vec![3.0, -1.0]);
    }

    #[test]
    fn meb_two_points_midpoint_half_distance() {
        let pts = vec![vec![0.0_f32, 0.0], vec![4.0, 0.0]];
        let (center, radius) = minimum_enclosing_ball(&pts).expect("meb");
        assert!((center[0] - 2.0).abs() < 1e-6);
        assert!(center[1].abs() < 1e-6);
        assert!(
            (radius - 2.0).abs() < 1e-6,
            "radius should be half-distance"
        );
    }

    #[test]
    fn meb_equilateral_triangle_circumradius() {
        // Equilateral triangle with side length s = 2.  Circumradius = s / sqrt(3).
        let s = 2.0_f32;
        let h = s * (3.0_f32).sqrt() / 2.0;
        let pts = vec![vec![0.0_f32, 0.0], vec![s, 0.0], vec![0.5 * s, h]];
        let (center, radius) = minimum_enclosing_ball(&pts).expect("meb");
        let expected = s / (3.0_f32).sqrt();
        assert!(
            (radius - expected).abs() < 1e-3,
            "circumradius {radius} != {expected}"
        );
        // Circumcentre is (approximately) equidistant from all three vertices.  The
        // Bădoiu–Clarkson centre converges more slowly than the radius, so allow a
        // slightly looser tolerance on the per-vertex distance.
        for p in &pts {
            assert!((dist(&center, p) - radius).abs() < 5e-3);
        }
    }

    #[test]
    fn meb_radius_ge_every_pairwise_half_distance() {
        let pts = vec![
            vec![0.0_f32, 0.0],
            vec![3.0, 0.0],
            vec![0.0, 4.0],
            vec![3.0, 4.0],
        ];
        let (_, radius) = minimum_enclosing_ball(&pts).expect("meb");
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let half = 0.5 * dist(&pts[i], &pts[j]);
                assert!(
                    radius >= half - 1e-4,
                    "radius {radius} < pairwise half-distance {half}"
                );
            }
        }
    }

    #[test]
    fn meb_collinear_points_half_span() {
        // Collinear points along the x-axis: MEB radius is half the total span.
        let pts = vec![
            vec![1.0_f32, 0.0],
            vec![5.0, 0.0],
            vec![3.0, 0.0],
            vec![2.0, 0.0],
        ];
        let (center, radius) = minimum_enclosing_ball(&pts).expect("meb");
        assert!((radius - 2.0).abs() < 1e-3, "half-span should be 2");
        assert!((center[0] - 3.0).abs() < 1e-3, "centre at midpoint of span");
    }

    #[test]
    fn meb_three_dimensional_points() {
        // Two points in 3D -> exact midpoint / half-distance.
        let pts = vec![vec![0.0_f32, 0.0, 0.0], vec![0.0, 0.0, 6.0]];
        let (center, radius) = minimum_enclosing_ball(&pts).expect("meb");
        assert!((radius - 3.0).abs() < 1e-6);
        assert!((center[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn meb_empty_list_errors() {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert!(minimum_enclosing_ball(&empty).is_err());
    }

    #[test]
    fn meb_dimension_mismatch_errors() {
        let pts = vec![vec![0.0_f32, 0.0], vec![1.0]];
        assert!(minimum_enclosing_ball(&pts).is_err());
    }

    #[test]
    fn meb_zero_dimension_errors() {
        let pts = vec![vec![], vec![]];
        assert!(minimum_enclosing_ball(&pts).is_err());
    }

    #[test]
    fn build_all_vertices_at_zero() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let cfg = CechConfig {
            max_dim: 2,
            max_radius: 10.0,
        };
        let filt = CechFiltration::build(&pts, 3, 2, &cfg).expect("build");
        let n_vertices = filt
            .simplices()
            .iter()
            .filter(|(v, val)| v.len() == 1 && val.abs() < 1e-6)
            .count();
        assert_eq!(n_vertices, 3, "all 3 vertices must be present at value 0");
    }

    #[test]
    fn build_edge_value_is_half_length() {
        // Two points 6 apart: the single edge has filtration value 3.
        let pts = vec![0.0_f32, 0.0, 6.0, 0.0];
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: 10.0,
        };
        let filt = CechFiltration::build(&pts, 2, 2, &cfg).expect("build");
        let edge = filt
            .simplices()
            .iter()
            .find(|(v, _)| v.len() == 2)
            .expect("edge present");
        assert!((edge.1 - 3.0).abs() < 1e-4, "edge value should be 3");
    }

    #[test]
    fn build_sorted_ascending_by_value() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let cfg = CechConfig {
            max_dim: 3,
            max_radius: 10.0,
        };
        let filt = CechFiltration::build(&pts, 4, 2, &cfg).expect("build");
        for w in filt.simplices().windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6, "filtration not sorted ascending");
        }
    }

    #[test]
    fn build_max_radius_prunes_high_radius_simplices() {
        // Two points 6 apart: edge radius is 3 > max_radius 1, so only vertices remain.
        let pts = vec![0.0_f32, 0.0, 6.0, 0.0];
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: 1.0,
        };
        let filt = CechFiltration::build(&pts, 2, 2, &cfg).expect("build");
        assert!(
            filt.simplices().iter().all(|(v, _)| v.len() == 1),
            "high-radius edge should be pruned"
        );
        assert_eq!(filt.n_simplices(), 2);
    }

    #[test]
    fn build_max_dim_limits_simplex_dimension() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: 100.0,
        };
        let filt = CechFiltration::build(&pts, 4, 2, &cfg).expect("build");
        assert!(
            filt.simplices().iter().all(|(v, _)| v.len() <= 2),
            "max_dim=1 must exclude triangles"
        );
    }

    #[test]
    fn build_n_simplices_and_max_value_sane() {
        let pts = vec![0.0_f32, 0.0, 2.0, 0.0, 0.0, 2.0];
        let cfg = CechConfig {
            max_dim: 2,
            max_radius: 100.0,
        };
        let filt = CechFiltration::build(&pts, 3, 2, &cfg).expect("build");
        // 3 vertices + 3 edges + 1 triangle.
        assert_eq!(filt.n_simplices(), 7);
        assert!(filt.max_value() > 0.0);
        // Max value is the triangle's circumradius (right isoceles legs of 2): hypotenuse
        // 2*sqrt(2), MEB radius = half the hypotenuse = sqrt(2).
        assert!((filt.max_value() - (2.0_f32).sqrt()).abs() < 1e-3);
    }

    #[test]
    fn build_equilateral_triangle_2simplex_circumradius() {
        let s = 2.0_f32;
        let h = s * (3.0_f32).sqrt() / 2.0;
        let pts = vec![0.0_f32, 0.0, s, 0.0, 0.5 * s, h];
        let cfg = CechConfig {
            max_dim: 2,
            max_radius: 100.0,
        };
        let filt = CechFiltration::build(&pts, 3, 2, &cfg).expect("build");
        let tri = filt
            .simplices()
            .iter()
            .find(|(v, _)| v.len() == 3)
            .expect("triangle present");
        let expected = s / (3.0_f32).sqrt();
        assert!(
            (tri.1 - expected).abs() < 1e-3,
            "2-simplex value should equal circumradius"
        );
    }

    #[test]
    fn build_deterministic() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.3, 0.7, 1.0, 1.0, 0.9, 0.2, 0.5];
        let cfg = CechConfig {
            max_dim: 2,
            max_radius: 100.0,
        };
        let a = CechFiltration::build(&pts, 5, 2, &cfg).expect("build a");
        let b = CechFiltration::build(&pts, 5, 2, &cfg).expect("build b");
        assert_eq!(a.simplices(), b.simplices(), "build must be deterministic");
    }

    #[test]
    fn build_points_length_mismatch_errors() {
        let pts = vec![0.0_f32, 0.0, 1.0];
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: 10.0,
        };
        assert!(CechFiltration::build(&pts, 2, 2, &cfg).is_err());
    }

    #[test]
    fn build_dim_zero_errors() {
        let pts = vec![0.0_f32, 1.0];
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: 10.0,
        };
        assert!(CechFiltration::build(&pts, 2, 0, &cfg).is_err());
    }

    #[test]
    fn build_n_zero_errors() {
        let pts: Vec<f32> = Vec::new();
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: 10.0,
        };
        assert!(CechFiltration::build(&pts, 0, 2, &cfg).is_err());
    }

    #[test]
    fn build_negative_max_radius_errors() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0];
        let cfg = CechConfig {
            max_dim: 1,
            max_radius: -1.0,
        };
        assert!(CechFiltration::build(&pts, 2, 2, &cfg).is_err());
    }

    #[test]
    fn build_max_dim_zero_vertices_only() {
        let pts = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let cfg = CechConfig {
            max_dim: 0,
            max_radius: 100.0,
        };
        let filt = CechFiltration::build(&pts, 3, 2, &cfg).expect("build");
        assert_eq!(filt.n_simplices(), 3);
        assert!(filt.simplices().iter().all(|(v, _)| v.len() == 1));
    }
}
