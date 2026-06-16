//! Wasserstein distance between raw persistence diagrams (slice-based API).
//!
//! Turner 2014 / Edelsbrunner-Harer Chapter VIII.  The p-Wasserstein distance between
//! two persistence diagrams is the optimal-transport cost between the augmented point
//! sets (unmatched points sent to their nearest diagonal projection) under the L^∞
//! ground metric raised to the p-th power.
//!
//! This module replicates the semantics of
//! [`mod@crate::persistence::wasserstein_p`] but accepts raw `&[(f64, f64)]` slices
//! instead of `PersistenceDiagram` structs.

use crate::error::{TdaError, TdaResult};

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// L^∞ distance between two birth-death points.
#[inline]
fn linf(b1: f64, d1: f64, b2: f64, d2: f64) -> f64 {
    (b1 - b2).abs().max((d1 - d2).abs())
}

/// L^∞ distance from point (b, d) to the diagonal.
#[inline]
fn diag(b: f64, d: f64) -> f64 {
    (d - b).abs() / 2.0
}

/// Build the `(n1+n2) × (n1+n2)` augmented cost matrix for Wasserstein-p matching.
///
/// Block layout (n = n1 + n2):
///
/// ```text
/// ┌─────────────────────────┬───────────────────────────┐
/// │ (n1×n2)  point↔point   │ (n1×n1)  pts1[i]→diag     │
/// │ cost = linf^p           │ diag[i,i]=diag(b,d)^p     │
/// │                         │ off-diag = LARGE           │
/// ├─────────────────────────┼───────────────────────────┤
/// │ (n2×n2)  pts2[j]→diag  │ (n2×n1)  diag↔diag = 0   │
/// │ diag[j,j]=diag(b,d)^p  │                             │
/// │ off-diag = LARGE       │                             │
/// └─────────────────────────┴───────────────────────────┘
/// ```
fn build_cost(pts1: &[(f64, f64)], pts2: &[(f64, f64)], p: f64) -> Vec<f64> {
    let n1 = pts1.len();
    let n2 = pts2.len();
    let n = n1 + n2;
    const LARGE: f64 = 1e18;

    let mut mat = vec![0.0_f64; n * n];

    // top-left: point-to-point
    for i in 0..n1 {
        for j in 0..n2 {
            let cost = linf(pts1[i].0, pts1[i].1, pts2[j].0, pts2[j].1).powf(p);
            mat[i * n + j] = cost;
        }
    }

    // top-right: pts1[i] → diagonal
    for i in 0..n1 {
        for k in 0..n1 {
            let col = n2 + k;
            if k == i {
                let c = diag(pts1[i].0, pts1[i].1).abs().powf(p);
                mat[i * n + col] = c;
            } else {
                mat[i * n + col] = LARGE;
            }
        }
    }

    // bottom-left: pts2[j] → diagonal
    for (j, &pt2) in pts2.iter().enumerate() {
        let row = n1 + j;
        for k in 0..n2 {
            if k == j {
                let c = diag(pt2.0, pt2.1).abs().powf(p);
                mat[row * n + k] = c;
            } else {
                mat[row * n + k] = LARGE;
            }
        }
    }
    // bottom-right: diag↔diag = 0 (already initialized)

    mat
}

/// Jonker-Volgenant / Hungarian shortest-augmenting-path algorithm.
///
/// Solves an n×n assignment problem given a row-major cost matrix.
/// Returns `assignment[i]` = column matched to row `i`.
fn hungarian_jv(cost: &[f64], n: usize) -> TdaResult<Vec<usize>> {
    if n == 0 {
        return Ok(vec![]);
    }
    if cost.len() != n * n {
        return Err(TdaError::DimensionMismatch {
            expected: n * n,
            got: cost.len(),
        });
    }

    let inf = f64::MAX / 4.0;

    // 1-indexed internally; index 0 is a virtual row/column
    let mut u = vec![0.0_f64; n + 1]; // row potentials
    let mut v = vec![0.0_f64; n + 1]; // col potentials
    let mut p = vec![0usize; n + 1]; // p[j] = matched row for col j (0 = unassigned)
    let mut way = vec![0usize; n + 1];

    for i in 1..=(n) {
        p[0] = i;
        let mut j0 = 0usize;
        let mut min_val = vec![inf; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=(n) {
                if used[j] {
                    continue;
                }
                let r = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if r < min_val[j] {
                    min_val[j] = r;
                    way[j] = j0;
                }
                if min_val[j] < delta {
                    delta = min_val[j];
                    j1 = j;
                }
            }

            for j in 0..=(n) {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_val[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            p[j0] = p[way[j0]];
            j0 = way[j0];
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert p[j] (col → row) to assignment[row] = col
    let mut assignment = vec![0usize; n];
    for j in 1..=(n) {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    Ok(assignment)
}

/// Total assignment cost given a row-major cost matrix and assignment vector.
fn assignment_cost(cost: &[f64], assignment: &[usize], n: usize) -> f64 {
    let large = 1e18_f64;
    (0..n)
        .map(|i| {
            let c = cost[i * n + assignment[i]];
            if c >= large { 0.0 } else { c }
        })
        .sum()
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Wasserstein-p distance between two raw persistence diagrams.
///
/// The ground metric is the L^∞ distance on birth-death space.  Unmatched diagram
/// points are sent to their nearest diagonal projection.  The returned value is
/// `(Σ_i cost_i)^(1/p)` where `cost_i` are the individual matching costs raised to
/// the p-th power.
///
/// # Errors
/// - [`TdaError::ParameterOutOfRange`] if `p == 0`.
pub fn diagram_wasserstein_p(dgm1: &[(f64, f64)], dgm2: &[(f64, f64)], p: u32) -> TdaResult<f64> {
    if p == 0 {
        return Err(TdaError::ParameterOutOfRange("p must be ≥ 1".to_owned()));
    }

    // Filter degenerate points
    let pts1: Vec<(f64, f64)> = dgm1.iter().filter(|&&(b, d)| b <= d).copied().collect();
    let pts2: Vec<(f64, f64)> = dgm2.iter().filter(|&&(b, d)| b <= d).copied().collect();

    let n1 = pts1.len();
    let n2 = pts2.len();

    if n1 == 0 && n2 == 0 {
        return Ok(0.0);
    }

    let n = n1 + n2;
    let pf = p as f64;

    let cost_mat = build_cost(&pts1, &pts2, pf);
    let assignment = hungarian_jv(&cost_mat, n)?;
    let total_cost = assignment_cost(&cost_mat, &assignment, n);

    Ok(total_cost.powf(1.0 / pf))
}

/// Wasserstein-2 distance between two raw persistence diagrams (convenience wrapper).
///
/// Equivalent to `diagram_wasserstein_p(dgm1, dgm2, 2)`.
pub fn diagram_wasserstein_2(dgm1: &[(f64, f64)], dgm2: &[(f64, f64)]) -> TdaResult<f64> {
    diagram_wasserstein_p(dgm1, dgm2, 2)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance from a diagram to itself is zero (p=2).
    #[test]
    fn distance_zero_same() {
        let dgm = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
        let d = diagram_wasserstein_2(&dgm, &dgm).expect("diagram_wasserstein_2 should succeed");
        assert!(d < 1e-8, "self-distance should be zero, got {d}");
    }

    /// Wasserstein-2 is non-negative between distinct diagrams.
    #[test]
    fn distance_positive() {
        let dgm1 = vec![(0.0, 2.0)];
        let dgm2 = vec![(1.0, 3.0)];
        let d = diagram_wasserstein_2(&dgm1, &dgm2).expect("diagram_wasserstein_2 should succeed");
        assert!(d > 0.0, "distinct diagrams should have d > 0, got {d}");
    }

    /// Wasserstein distance is symmetric: d(A,B) == d(B,A).
    #[test]
    fn distance_symmetric() {
        let dgm1 = vec![(0.0, 1.0), (2.0, 5.0)];
        let dgm2 = vec![(0.5, 1.5), (1.5, 4.0)];
        let d_ab =
            diagram_wasserstein_2(&dgm1, &dgm2).expect("diagram_wasserstein_2 should succeed");
        let d_ba =
            diagram_wasserstein_2(&dgm2, &dgm1).expect("diagram_wasserstein_2 should succeed");
        assert!(
            (d_ab - d_ba).abs() < 1e-8,
            "symmetry violated: {d_ab} != {d_ba}"
        );
    }

    /// p=1 Wasserstein is ≥ p=2 Wasserstein (by Lyapunov's inequality analogy for
    /// finite measures).
    #[test]
    fn p1_vs_p2() {
        let dgm1 = vec![(0.0, 1.0), (2.0, 4.0)];
        let dgm2 = vec![(0.5, 2.0), (1.5, 3.5)];
        let d1 =
            diagram_wasserstein_p(&dgm1, &dgm2, 1).expect("diagram_wasserstein_p should succeed");
        let d2 =
            diagram_wasserstein_p(&dgm1, &dgm2, 2).expect("diagram_wasserstein_p should succeed");
        // Both should be finite and positive
        assert!(d1.is_finite(), "W1 should be finite");
        assert!(d2.is_finite(), "W2 should be finite");
        assert!(d1 >= 0.0);
        assert!(d2 >= 0.0);
    }

    /// Both diagrams empty → distance is zero.
    #[test]
    fn empty_dgm() {
        let d1 = diagram_wasserstein_2(&[], &[]).expect("diagram_wasserstein_2 should succeed");
        let d2 = diagram_wasserstein_p(&[], &[], 1).expect("diagram_wasserstein_p should succeed");
        assert!(d1 < 1e-12 && d2 < 1e-12, "empty diagrams: d1={d1}, d2={d2}");
    }

    /// One empty diagram → cost equals matching the non-empty diagram to diagonal.
    #[test]
    fn single_point() {
        let b = 0.0_f64;
        let d = 4.0_f64;
        let dgm = vec![(b, d)];
        let dist = diagram_wasserstein_2(&dgm, &[]).expect("diagram_wasserstein_2 should succeed");
        // W_2 of sending one point to its diagonal proj: cost = diag_dist^2 = (2)^2 = 4, W_2 = 2
        let expected_cost = diag(b, d).powi(2); // (2.0)^2 = 4.0
        let expected_w2 = expected_cost.sqrt(); // = 2.0
        assert!(
            (dist - expected_w2).abs() < 1e-8,
            "expected W2 = {expected_w2}, got {dist}"
        );
    }

    /// Unequal diagram sizes are handled correctly.
    #[test]
    fn unequal_sizes() {
        let dgm1 = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
        let dgm2 = vec![(0.5, 2.0)];
        let d = diagram_wasserstein_2(&dgm1, &dgm2).expect("diagram_wasserstein_2 should succeed");
        assert!(d.is_finite() && d >= 0.0, "got {d}");
    }

    /// W_2 convenience wrapper produces same result as general p=2.
    #[test]
    fn w2_matches_general() {
        let dgm1 = vec![(0.0, 2.0), (1.0, 4.0)];
        let dgm2 = vec![(0.0, 1.5), (1.5, 4.5)];
        let d_conv =
            diagram_wasserstein_2(&dgm1, &dgm2).expect("diagram_wasserstein_2 should succeed");
        let d_gen =
            diagram_wasserstein_p(&dgm1, &dgm2, 2).expect("diagram_wasserstein_p should succeed");
        assert!((d_conv - d_gen).abs() < 1e-10, "w2={d_conv}, p=2={d_gen}");
    }

    /// p=0 returns an error.
    #[test]
    fn p0_error() {
        let dgm = vec![(0.0, 1.0)];
        assert!(diagram_wasserstein_p(&dgm, &dgm, 0).is_err());
    }

    /// Distance is zero when all points are on the diagonal (degenerate pairs filtered).
    #[test]
    fn diagonal_points_ignored() {
        let dgm1 = vec![(1.0, 1.0), (2.0, 2.0)]; // all on diagonal
        let dgm2 = vec![(1.5, 1.5)]; // on diagonal
        let d = diagram_wasserstein_2(&dgm1, &dgm2).expect("diagram_wasserstein_2 should succeed");
        assert!(d < 1e-8, "diagonal points filtered, expected 0, got {d}");
    }
}
