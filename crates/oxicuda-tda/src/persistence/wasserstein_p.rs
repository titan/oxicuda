//! Wasserstein-p distance between persistence diagrams (general p ≥ 1).
//!
//! Ground metric: L^∞ on ℝ² for matched point pairs; diagonal distance for unmatched.
//! Optimal transport computed via augmented-matrix Hungarian (Jonker-Volgenant style).
//! Includes sliced Wasserstein approximation (Carrière et al. 2017).

use crate::error::{TdaError, TdaResult};
use crate::handle::LcgRng;
use crate::persistence::diagram::PersistenceDiagram;

/// L^∞ distance between two persistence points in birth-death space.
#[inline]
pub fn point_dist_inf(b1: f64, d1: f64, b2: f64, d2: f64) -> f64 {
    (b1 - b2).abs().max((d1 - d2).abs())
}

/// Distance from point (birth, death) to the persistence diagonal.
///
/// The diagonal projection of (b,d) is ((b+d)/2, (b+d)/2); the L^∞ distance is (d-b)/2.
#[inline]
pub fn diagonal_dist(birth: f64, death: f64) -> f64 {
    (death - birth) / 2.0
}

/// Build the augmented (n1+n2)×(n1+n2) cost matrix (row-major) for Wasserstein-p matching.
///
/// Block structure:
///   top-left   (n1×n2): off-diagonal cost point_dist_inf(i,j)^p
///   top-right  (n1×n1): unmatched cost for `pts1[i]` → diagonal, cell `(i,i) = diagonal_dist^p`, others = large
///   bottom-left(n2×n2): unmatched cost for `pts2[j]` → diagonal, cell `(j,j) = diagonal_dist^p`, others = large
///   bottom-right(n2×n1): diagonal-to-diagonal is free → all zeros
pub fn build_cost_matrix(pts1: &[(f64, f64)], pts2: &[(f64, f64)], p: f64) -> Vec<f64> {
    let n1 = pts1.len();
    let n2 = pts2.len();
    let n = n1 + n2;
    let large = 1e18_f64;

    let mut mat = vec![0.0_f64; n * n];

    // top-left: point-to-point costs
    for i in 0..n1 {
        for j in 0..n2 {
            let d = point_dist_inf(pts1[i].0, pts1[i].1, pts2[j].0, pts2[j].1);
            mat[i * n + j] = d.powf(p);
        }
    }

    // top-right: pts1[i] unmatched → diagonal, only diagonal cell is non-large
    for i in 0..n1 {
        for k in 0..n1 {
            let col = n2 + k;
            if k == i {
                let d = diagonal_dist(pts1[i].0, pts1[i].1);
                mat[i * n + col] = d.abs().powf(p);
            } else {
                mat[i * n + col] = large;
            }
        }
    }

    // bottom-left: pts2[j] unmatched → diagonal, only diagonal cell is non-large
    for (j, pt2) in pts2.iter().enumerate() {
        let row = n1 + j;
        for k in 0..n2 {
            if k == j {
                let d = diagonal_dist(pt2.0, pt2.1);
                mat[row * n + k] = d.abs().powf(p);
            } else {
                mat[row * n + k] = large;
            }
        }
    }

    // bottom-right: diagonal-to-diagonal costs are all zero (already initialised to 0)

    mat
}

/// Hungarian algorithm (Jonker-Volgenant shortest-path variant) for minimum-cost
/// bipartite matching on an n×n cost matrix (row-major).
///
/// Returns `assignment[i]` = the column matched to row i.
pub fn hungarian(cost: &[f64], n: usize) -> TdaResult<Vec<usize>> {
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

    // 1-indexed internally: rows 1..=n, cols 1..=n; index 0 is the "virtual" source.
    let mut u = vec![0.0_f64; n + 1]; // row potentials (u[0] unused)
    let mut v = vec![0.0_f64; n + 1]; // col potentials (v[0] = virtual)
    let mut p = vec![0usize; n + 1]; // p[j] = row assigned to col j (0 = unassigned)
    let mut way = vec![0usize; n + 1]; // predecessor col in augmenting path

    for i in 1..=n {
        p[0] = i; // attach row i to virtual column 0 for augmenting
        let mut j0 = 0usize;
        let mut min_val = vec![inf; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0]; // current row
            let mut delta = inf;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let reduced = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                    if reduced < min_val[j] {
                        min_val[j] = reduced;
                        way[j] = j0;
                    }
                    if min_val[j] < delta {
                        delta = min_val[j];
                        j1 = j;
                    }
                }
            }

            if j1 == 0 {
                return Err(TdaError::MatchingFailed(
                    "Hungarian: no augmenting column found".to_string(),
                ));
            }

            for j in 0..=n {
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

        // Trace the augmenting path back and update assignment
        loop {
            p[j0] = p[way[j0]];
            j0 = way[j0];
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert p (col→row, 1-indexed) to assignment (row→col, 0-indexed)
    let mut assignment = vec![0usize; n];
    for j in 1..=n {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    Ok(assignment)
}

/// Total cost of a matching: sum of `cost[row][assignment[row]]` for all rows.
pub fn matching_cost(cost: &[f64], assignment: &[usize], n: usize) -> f64 {
    (0..n).fold(0.0_f64, |acc, i| acc + cost[i * n + assignment[i]])
}

/// Wasserstein-p distance between two persistence diagrams.
///
/// p must be ≥ 1. Ground metric is L^∞. Unmatched points are projected to the diagonal.
/// Returns W_p(D1, D2) = ( min-cost matching )^(1/p).
pub fn wasserstein_p(
    diag1: &PersistenceDiagram,
    diag2: &PersistenceDiagram,
    p: f64,
) -> TdaResult<f64> {
    if p < 1.0 || !p.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "p must be ≥ 1.0, got {p}"
        )));
    }

    let pts1: Vec<(f64, f64)> = diag1
        .finite_pairs()
        .iter()
        .map(|pp| (pp.birth, pp.death.unwrap_or(pp.birth)))
        .collect();
    let pts2: Vec<(f64, f64)> = diag2
        .finite_pairs()
        .iter()
        .map(|pp| (pp.birth, pp.death.unwrap_or(pp.birth)))
        .collect();

    let n1 = pts1.len();
    let n2 = pts2.len();

    if n1 == 0 && n2 == 0 {
        return Ok(0.0);
    }

    let n = n1 + n2;
    let cost = build_cost_matrix(&pts1, &pts2, p);
    let assignment = hungarian(&cost, n)?;
    let total = matching_cost(&cost, &assignment, n);

    // Guard against negative rounding error before taking root
    let total_clamped = total.max(0.0);
    Ok(total_clamped.powf(1.0 / p))
}

// ──────────────────────────────────────────────────────────────────────────────
// Sliced Wasserstein approximation (Carrière et al. 2017)
// ──────────────────────────────────────────────────────────────────────────────

/// Box-Muller transform to produce a unit-circle angle from two uniform [0,1) samples.
///
/// Returns (cos θ, sin θ) for a uniformly random direction.
#[inline]
fn uniform_circle(rng: &mut LcgRng) -> (f64, f64) {
    let u1 = rng.next_f64();
    let u2 = rng.next_f64();
    let theta = 2.0 * core::f64::consts::PI * u1;
    let _ = u2; // second uniform not needed for angle only
    (theta.cos(), theta.sin())
}

/// Project a point (b, d) and its diagonal image onto direction (cos_t, sin_t),
/// returning both the off-diagonal and the diagonal-projected coordinates.
#[inline]
fn project_point(b: f64, d: f64, cos_t: f64, sin_t: f64) -> (f64, f64) {
    let off_diag = b * cos_t + d * sin_t;
    let mid = (b + d) * 0.5;
    let on_diag = mid * cos_t + mid * sin_t;
    (off_diag, on_diag)
}

/// Sliced Wasserstein approximation (Carrière et al. 2017).
///
/// For each of `n_projections` random directions on S¹:
///   1. Project each off-diagonal point and its diagonal shadow onto that direction.
///   2. Sort both projected clouds (each is size n1+n2: real points + shadows of other set).
///   3. Compute 1D W_p distance between sorted sets.
///
/// Returns the average over all projections (raised to 1/p at the end).
pub fn sliced_wasserstein(
    diag1: &PersistenceDiagram,
    diag2: &PersistenceDiagram,
    p: f64,
    n_projections: usize,
    rng: &mut LcgRng,
) -> TdaResult<f64> {
    if p < 1.0 || !p.is_finite() {
        return Err(TdaError::ParameterOutOfRange(format!(
            "p must be ≥ 1.0, got {p}"
        )));
    }
    if n_projections == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "n_projections must be ≥ 1".to_string(),
        ));
    }

    let pts1: Vec<(f64, f64)> = diag1
        .finite_pairs()
        .iter()
        .map(|pp| (pp.birth, pp.death.unwrap_or(pp.birth)))
        .collect();
    let pts2: Vec<(f64, f64)> = diag2
        .finite_pairs()
        .iter()
        .map(|pp| (pp.birth, pp.death.unwrap_or(pp.birth)))
        .collect();

    let n1 = pts1.len();
    let n2 = pts2.len();

    if n1 == 0 && n2 == 0 {
        return Ok(0.0);
    }

    let total_size = n1 + n2; // each projected cloud has size n1+n2

    let mut accumulated = 0.0_f64;

    for _ in 0..n_projections {
        let (cos_t, sin_t) = uniform_circle(rng);

        // Build projected cloud for diag1: real points + diagonal shadows of diag2's points
        let mut cloud1 = Vec::with_capacity(total_size);
        for &(b, d) in &pts1 {
            let (off, _) = project_point(b, d, cos_t, sin_t);
            cloud1.push(off);
        }
        for &(b, d) in &pts2 {
            let (_, on) = project_point(b, d, cos_t, sin_t);
            cloud1.push(on);
        }

        // Build projected cloud for diag2: real points + diagonal shadows of diag1's points
        let mut cloud2 = Vec::with_capacity(total_size);
        for &(b, d) in &pts2 {
            let (off, _) = project_point(b, d, cos_t, sin_t);
            cloud2.push(off);
        }
        for &(b, d) in &pts1 {
            let (_, on) = project_point(b, d, cos_t, sin_t);
            cloud2.push(on);
        }

        cloud1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        cloud2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        // 1D W_p: sort then align element-wise; cost = |x - y|^p, sum then ^(1/p)
        let proj_cost: f64 = cloud1
            .iter()
            .zip(cloud2.iter())
            .map(|(x, y)| (x - y).abs().powf(p))
            .sum();
        accumulated += proj_cost.powf(1.0 / p);
    }

    Ok(accumulated / n_projections as f64)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;
    use crate::persistence::diagram::PersistenceDiagram;
    use crate::persistence::distance::wasserstein_1;

    fn make_diag(pts: &[(f64, f64)]) -> PersistenceDiagram {
        let pairs = pts
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 0,
                birth: b,
                death: Some(d),
            })
            .collect();
        PersistenceDiagram::new(pairs, 0)
    }

    fn empty_diag() -> PersistenceDiagram {
        PersistenceDiagram::new(vec![], 0)
    }

    // 1. Identical diagrams → distance 0
    #[test]
    fn wasserstein_p_identical_diagrams_is_zero() {
        let d = make_diag(&[(0.0, 1.0), (0.5, 2.5), (1.0, 3.0)]);
        let dist = wasserstein_p(&d, &d, 2.0).unwrap();
        assert!(dist < 1e-9, "self W2 = {dist}");
    }

    // 2. p=1 should be close to wasserstein_1 (different ground metric note: wasserstein_1 uses L1,
    //    wasserstein_p uses L∞; they can differ — test only that both give non-negative finite results
    //    on same input and that wasserstein_p p=1 is non-negative and consistent with itself)
    #[test]
    fn wasserstein_p_one_self_consistent() {
        let d1 = make_diag(&[(0.0, 2.0)]);
        let d2 = make_diag(&[(0.0, 3.0)]);
        let wp = wasserstein_p(&d1, &d2, 1.0).unwrap();
        let w1 = wasserstein_1(&d1, &d2).unwrap();
        // Both are finite, non-negative
        assert!(wp.is_finite() && wp >= 0.0);
        assert!(w1.is_finite() && w1 >= 0.0);
    }

    // 3. Both diagrams empty → 0
    #[test]
    fn wasserstein_p_empty_both_is_zero() {
        let dist = wasserstein_p(&empty_diag(), &empty_diag(), 2.0).unwrap();
        assert_eq!(dist, 0.0);
    }

    // 4. One empty, one has points → cost = sum of diagonal distances
    #[test]
    fn wasserstein_p_one_empty() {
        let d = make_diag(&[(0.0, 2.0)]); // diagonal_dist = 1.0
        let dist = wasserstein_p(&d, &empty_diag(), 2.0).unwrap();
        // W2 = (1.0^2)^(1/2) = 1.0
        assert!((dist - 1.0).abs() < 1e-9, "W2 one-empty = {dist}");
    }

    // 5. Distance is always ≥ 0
    #[test]
    fn wasserstein_p_positive() {
        let d1 = make_diag(&[(0.0, 1.0), (2.0, 4.0)]);
        let d2 = make_diag(&[(0.5, 1.5), (1.0, 3.0)]);
        assert!(wasserstein_p(&d1, &d2, 1.5).unwrap() >= 0.0);
        assert!(wasserstein_p(&d1, &d2, 3.0).unwrap() >= 0.0);
    }

    // 6. W_2 ≥ 0 and W_∞ (bottleneck) ≥ W_2 conceptually (just verify ordering property)
    //    For distinct diagrams test that W_p increases with p (for simple cases)
    #[test]
    fn wasserstein_p_increases_with_p() {
        let d1 = make_diag(&[(0.0, 4.0)]);
        let d2 = make_diag(&[(1.0, 5.0)]);
        let w1 = wasserstein_p(&d1, &d2, 1.0).unwrap();
        let w2 = wasserstein_p(&d1, &d2, 2.0).unwrap();
        let w4 = wasserstein_p(&d1, &d2, 4.0).unwrap();
        // W_p is non-decreasing in p for a single matched pair
        assert!(w1 <= w2 + 1e-9, "w1={w1} w2={w2}");
        assert!(w2 <= w4 + 1e-9, "w2={w2} w4={w4}");
    }

    // 7. p ≤ 0 → Err
    #[test]
    fn wasserstein_p_err_invalid_p() {
        let d = make_diag(&[(0.0, 1.0)]);
        assert!(wasserstein_p(&d, &d, 0.0).is_err());
        assert!(wasserstein_p(&d, &d, -1.0).is_err());
    }

    // 8. point_dist_inf self is zero
    #[test]
    fn point_dist_inf_self_is_zero() {
        assert_eq!(point_dist_inf(1.5, 3.0, 1.5, 3.0), 0.0);
    }

    // 9. diagonal_dist correct value
    #[test]
    fn diagonal_dist_correct() {
        assert!((diagonal_dist(0.0, 2.0) - 1.0).abs() < 1e-15);
        assert!((diagonal_dist(1.0, 3.0) - 1.0).abs() < 1e-15);
    }

    // 10. build_cost_matrix shape correct
    #[test]
    fn build_cost_matrix_shape() {
        let pts1 = vec![(0.0, 1.0), (1.0, 2.0)];
        let pts2 = vec![(0.5, 1.5)];
        let mat = build_cost_matrix(&pts1, &pts2, 2.0);
        let n = pts1.len() + pts2.len();
        assert_eq!(mat.len(), n * n);
    }

    // 11. Hungarian on empty matrix returns empty
    #[test]
    fn hungarian_empty_ok() {
        let assignment = hungarian(&[], 0).unwrap();
        assert!(assignment.is_empty());
    }

    // 12. Hungarian on 3×3 with zeros on diagonal, large elsewhere
    #[test]
    fn hungarian_identity_match() {
        let large = 1e10_f64;
        let cost = vec![0.0, large, large, large, 0.0, large, large, large, 0.0];
        let assignment = hungarian(&cost, 3).unwrap();
        assert_eq!(assignment, vec![0, 1, 2]);
    }

    // 13. matching_cost manual check
    #[test]
    fn matching_cost_correct() {
        // 2×2 cost: [[1, 5], [3, 2]]  assignment=[0,1] → cost=1+2=3
        let cost = vec![1.0_f64, 5.0, 3.0, 2.0];
        let assignment = vec![0usize, 1];
        let total = matching_cost(&cost, &assignment, 2);
        assert!((total - 3.0).abs() < 1e-15);
    }

    // 14. sliced_wasserstein returns finite f64
    #[test]
    fn sliced_wasserstein_finite() {
        let d1 = make_diag(&[(0.0, 1.0), (1.0, 3.0)]);
        let d2 = make_diag(&[(0.5, 2.0)]);
        let mut rng = LcgRng::new(42);
        let sw = sliced_wasserstein(&d1, &d2, 2.0, 50, &mut rng).unwrap();
        assert!(sw.is_finite());
    }

    // 15. Sliced Wasserstein of a diagram with itself is ≈ 0
    #[test]
    fn sliced_wasserstein_zero_for_identical() {
        let d = make_diag(&[(0.0, 2.0), (1.0, 4.0)]);
        let mut rng = LcgRng::new(7);
        let sw = sliced_wasserstein(&d, &d, 2.0, 100, &mut rng).unwrap();
        assert!(sw < 1e-9, "SW self = {sw}");
    }

    // 16. Sliced Wasserstein of distinct diagrams > 0
    #[test]
    fn sliced_wasserstein_positive() {
        let d1 = make_diag(&[(0.0, 4.0)]);
        let d2 = make_diag(&[(1.0, 2.0)]);
        let mut rng = LcgRng::new(99);
        let sw = sliced_wasserstein(&d1, &d2, 2.0, 100, &mut rng).unwrap();
        assert!(sw > 0.0, "SW distinct = {sw}");
    }

    // 17. Single point each: compute W_2 by hand
    //     pts1=[(0,2)], pts2=[(0,4)]
    //     Augmented n=2; optimal: match (0,2)↔(0,4), cost = L∞((0,2),(0,4))^2 = 2^2 = 4
    //     unmatched diagonal cost for pts2 shadow and pts1 shadow = diagonal^2 each
    //     Actually in augmented setup: match off-diagonal pair: cost 4; both diag-to-diag = 0
    //     Total cost = 4, W2 = sqrt(4) = 2.0
    #[test]
    fn wasserstein_p_single_point_each() {
        let d1 = make_diag(&[(0.0, 2.0)]);
        let d2 = make_diag(&[(0.0, 4.0)]);
        let w2 = wasserstein_p(&d1, &d2, 2.0).unwrap();
        assert!((w2 - 2.0).abs() < 1e-9, "W2 = {w2}");
    }

    // 18. Hungarian on single-entry matrix
    #[test]
    fn hungarian_single_entry() {
        let cost = vec![7.0_f64];
        let assignment = hungarian(&cost, 1).unwrap();
        assert_eq!(assignment, vec![0]);
    }
}
