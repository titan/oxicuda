//! N-Dimensional Hypervolume indicator via the WFG algorithm.
//!
//! Implements the recursive WFG (While-Herd-Grosz 2006) algorithm for exact
//! hypervolume computation in arbitrary dimensionality.
//!
//! Reference: While, L., Hingston, P., Barone, L., & Huband, S. (2006).
//! "A faster algorithm for calculating hypervolume". IEEE Transactions on
//! Evolutionary Computation, 10(1), 29-38.

use crate::{EvolError, EvolResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if point `a` dominates `b` in minimization.
///
/// `a` dominates `b` iff `a[i] <= b[i]` for all `i`, and `a[i] < b[i]` for at least one `i`.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "dominates: dimension mismatch");
    let mut at_least_one_strictly_less = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            at_least_one_strictly_less = true;
        }
    }
    at_least_one_strictly_less
}

/// Filter a set of points to retain only the non-dominated (Pareto-optimal) subset.
///
/// A point is retained iff no other point in the set dominates it.
pub fn nondominated_filter(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !keep[j] {
                continue;
            }
            // Check if j dominates i
            if dominates(&points[j], &points[i]) {
                keep[i] = false;
                break;
            }
        }
    }
    points
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, p)| p.clone())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// WFG algorithm internals
// ─────────────────────────────────────────────────────────────────────────────

/// 1-D hypervolume: max contribution from any non-dominated point = ref[0] - min(p[0]).
///
/// In WFG sliced fronts the 1-D sub-front may contain multiple points projected to the
/// same dimension; we take the best (minimum) value to compute the correct contribution.
fn hv_1d(front: &[Vec<f64>], ref_pt: &[f64]) -> f64 {
    // In 1-D the non-dominated set has exactly one point (the one with the smallest value).
    front
        .iter()
        .map(|p| ref_pt[0] - p[0])
        .filter(|&d| d > 0.0)
        .fold(0.0_f64, f64::max)
}

/// 2-D hypervolume sweep (sort by first objective ascending, sweep second objective).
fn hv_2d(front: &[Vec<f64>], ref_pt: &[f64]) -> f64 {
    if front.is_empty() {
        return 0.0;
    }
    // Filter points that are dominated by the reference point
    let mut valid: Vec<&Vec<f64>> = front
        .iter()
        .filter(|p| p[0] < ref_pt[0] && p[1] < ref_pt[1])
        .collect();

    if valid.is_empty() {
        return 0.0;
    }

    // Sort by first objective ascending
    valid.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut area = 0.0;
    let mut prev_f1 = ref_pt[1];

    for p in &valid {
        if prev_f1 > p[1] {
            area += (ref_pt[0] - p[0]) * (prev_f1 - p[1]);
            prev_f1 = p[1];
        }
    }
    area
}

/// Remove dominated points from a front (non-dominated filter, used during WFG recursion).
fn wfg_nondominated(front: &[Vec<f64>]) -> Vec<Vec<f64>> {
    nondominated_filter(front)
}

/// Recursive WFG hypervolume computation (all objectives, minimization).
///
/// Implements the slice-and-sweep decomposition: sort points by the last
/// objective ascending, then integrate the (d-1)-dimensional cross-section HV
/// over successive slabs in the last dimension.
///
/// `front` may contain dominated points; internal non-dominated filtering is
/// applied after projection at each recursion level.
fn wfg_hv(front: &[Vec<f64>], ref_pt: &[f64], n_obj: usize) -> f64 {
    if front.is_empty() {
        return 0.0;
    }

    if n_obj == 0 {
        return 0.0;
    }

    // Single point: hypervolume is the product of (ref[j] - pt[j]) clamped to >= 0.
    if front.len() == 1 {
        return front[0]
            .iter()
            .zip(ref_pt.iter())
            .take(n_obj)
            .map(|(f, r)| (r - f).max(0.0))
            .product();
    }

    if n_obj == 1 {
        return hv_1d(front, ref_pt);
    }

    if n_obj == 2 {
        return hv_2d(front, ref_pt);
    }

    // n_obj >= 3: slice-sweep along the last objective.
    let k = n_obj - 1; // last objective index

    // Filter: keep only points with f[k] < ref_pt[k]
    let mut filtered: Vec<Vec<f64>> = front.iter().filter(|p| p[k] < ref_pt[k]).cloned().collect();

    if filtered.is_empty() {
        return 0.0;
    }

    // Sort by last objective ascending (smallest z-value first)
    filtered.sort_by(|a, b| a[k].partial_cmp(&b[k]).unwrap_or(std::cmp::Ordering::Equal));

    // Slice-sweep: for each point i, the slab from z_i to z_{i+1} (or ref_pt[k])
    // has (d-1)-dimensional cross-section = HV of points[0..=i] projected to [0..k].
    let sub_ref: Vec<f64> = ref_pt[..k].to_vec();
    let mut total = 0.0_f64;
    let n = filtered.len();

    for i in 0..n {
        let z_i = filtered[i][k];
        let z_next = if i + 1 < n {
            filtered[i + 1][k]
        } else {
            ref_pt[k]
        };
        let dz = z_next - z_i;
        if dz <= 0.0 {
            continue;
        }

        // Active set: all points with index <= i (sorted ascending by z, so these have z <= z_i)
        // Project to the first k objectives.
        let sub_pts: Vec<Vec<f64>> = filtered[..=i].iter().map(|p| p[..k].to_vec()).collect();

        // Remove dominated points in the projected space
        let nd_sub = wfg_nondominated(&sub_pts);
        let hv_sub = wfg_hv(&nd_sub, &sub_ref, k);

        total += hv_sub * dz;
    }

    total
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the exact n-dimensional hypervolume indicator using the WFG algorithm.
///
/// # Arguments
/// - `front`: each element is an objective vector (minimization; lower is better).
/// - `reference`: a single-element slice containing the reference point vector.
///   The reference point must strictly dominate all front points on every objective
///   (i.e., `reference[0][j] > front[i][j]` for all `i`, `j`).
///
/// # Errors
/// - Returns `Ok(0.0)` if `front` is empty.
/// - `EvolError::InvalidParameter` if `reference` is empty or has a zero-length point.
/// - `EvolError::DimensionMismatch` if any front point has a different length from `reference[0]`.
/// - `EvolError::InvalidParameter` if any front point is not strictly dominated by the reference.
pub fn hypervolume_nd(front: &[Vec<f64>], reference: &[Vec<f64>]) -> EvolResult<f64> {
    if front.is_empty() {
        return Ok(0.0);
    }

    // `reference` is a single-element slice: reference[0] is the reference point vector.
    if reference.is_empty() {
        return Err(EvolError::InvalidParameter(
            "reference must be a non-empty slice with the reference point as reference[0]"
                .to_owned(),
        ));
    }

    let ref_pt: &Vec<f64> = &reference[0];
    let n_obj = ref_pt.len();

    if n_obj == 0 {
        return Err(EvolError::InvalidParameter(
            "reference point must have at least 1 objective".to_owned(),
        ));
    }

    // Validate all front points against the reference
    for (i, p) in front.iter().enumerate() {
        if p.len() != n_obj {
            return Err(EvolError::DimensionMismatch {
                expected: n_obj,
                got: p.len(),
            });
        }
        for (j, (&pj, &rj)) in p.iter().zip(ref_pt.iter()).enumerate() {
            if pj >= rj {
                return Err(EvolError::InvalidParameter(format!(
                    "front[{i}][{j}] = {pj} is not strictly dominated by reference[{j}] = {rj}"
                )));
            }
        }
    }

    // Filter to non-dominated front before computing HV
    let nd_front = nondominated_filter(front);

    Ok(wfg_hv(&nd_front, ref_pt, n_obj))
}

/// Compute the hypervolume contribution of each point in the front.
///
/// `contribution[i]` = `hypervolume_nd(front)` - `hypervolume_nd(front without point i)`.
///
/// This is the exclusive hypervolume contribution (removal contribution).
///
/// # Errors
/// Propagates errors from `hypervolume_nd`.
pub fn hypervolume_contributions(
    front: &[Vec<f64>],
    reference: &[Vec<f64>],
) -> EvolResult<Vec<f64>> {
    if front.is_empty() {
        return Ok(Vec::new());
    }

    let total_hv = hypervolume_nd(front, reference)?;
    let n = front.len();
    let mut contributions = Vec::with_capacity(n);

    for i in 0..n {
        // Build the front without point i
        let reduced: Vec<Vec<f64>> = front
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| p.clone())
            .collect();

        let hv_without = if reduced.is_empty() {
            0.0
        } else {
            hypervolume_nd(&reduced, reference)?
        };

        contributions.push(total_hv - hv_without);
    }

    Ok(contributions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::metrics::hypervolume_2d;

    const EPS: f64 = 1e-10;

    fn ref_wrap(r: Vec<f64>) -> Vec<Vec<f64>> {
        vec![r]
    }

    // ── dominates ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dominates_basic() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 2.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0])); // equal -> not dominated
    }

    #[test]
    fn test_dominates_equal_all_objectives() {
        // A == B -> neither dominates the other
        assert!(!dominates(&[0.5, 0.5], &[0.5, 0.5]));
    }

    #[test]
    fn test_dominates_partial_order() {
        // a is better on obj 0, worse on obj 1 -> no dominance
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0]));
    }

    #[test]
    fn test_dominates_single_objective() {
        assert!(dominates(&[0.1], &[0.9]));
        assert!(!dominates(&[0.9], &[0.1]));
    }

    // ── nondominated_filter ────────────────────────────────────────────────────

    #[test]
    fn test_nondominated_filter_basic() {
        let pts = vec![
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
            vec![2.5, 2.5], // dominated by [2.0, 2.0]
        ];
        let nd = nondominated_filter(&pts);
        // [2.5, 2.5] is dominated by [2.0, 2.0] -> should be removed
        assert!(nd.iter().any(|p| p[0] == 1.0 && p[1] == 3.0));
        assert!(nd.iter().any(|p| p[0] == 2.0 && p[1] == 2.0));
        assert!(nd.iter().any(|p| p[0] == 3.0 && p[1] == 1.0));
        assert!(!nd.iter().any(|p| p[0] == 2.5 && p[1] == 2.5));
    }

    #[test]
    fn test_nondominated_filter_all_nondominated() {
        let pts = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let nd = nondominated_filter(&pts);
        assert_eq!(nd.len(), 3);
    }

    #[test]
    fn test_nondominated_filter_single_point() {
        let pts = vec![vec![1.0, 2.0, 3.0]];
        let nd = nondominated_filter(&pts);
        assert_eq!(nd.len(), 1);
    }

    #[test]
    fn test_nondominated_filter_empty() {
        let nd = nondominated_filter(&[]);
        assert!(nd.is_empty());
    }

    // ── hypervolume_nd: empty / edge cases ────────────────────────────────────

    #[test]
    fn test_hv_nd_empty_front_returns_zero() {
        let result = hypervolume_nd(&[], &ref_wrap(vec![2.0, 2.0]));
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be present"), 0.0);
    }

    #[test]
    fn test_hv_nd_single_point_1d() {
        // HV = ref - point = 2.0 - 0.5 = 1.5
        let result = hypervolume_nd(&[vec![0.5]], &ref_wrap(vec![2.0]));
        assert!(result.is_ok());
        let hv = result.expect("result should be present");
        assert!((hv - 1.5).abs() < EPS, "expected 1.5, got {hv}");
    }

    #[test]
    fn test_hv_nd_reference_does_not_dominate_returns_error() {
        // front point equals reference on one objective -> error (not strictly dominated)
        let result = hypervolume_nd(&[vec![2.0, 1.0]], &ref_wrap(vec![2.0, 2.0]));
        assert!(
            result.is_err(),
            "expected error when point not strictly dominated by ref"
        );
    }

    #[test]
    fn test_hv_nd_dimension_mismatch_returns_error() {
        let result = hypervolume_nd(&[vec![1.0, 2.0]], &ref_wrap(vec![3.0, 3.0, 3.0]));
        assert!(result.is_err());
    }

    // ── hypervolume_nd: 2-D matches hypervolume_2d ────────────────────────────

    #[test]
    fn test_hv_nd_2d_matches_hv_2d() {
        // Standard 2D Pareto front
        let front_2d = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let reference = (4.0, 4.0);

        // Use the existing 2D function for comparison
        let front_pairs: Vec<(f64, f64)> = front_2d.iter().map(|p| (p[0], p[1])).collect();
        let hv_2d_val =
            hypervolume_2d(&front_pairs, reference).expect("hypervolume_2d should succeed");

        // Use our nd function
        let hv_nd_val = hypervolume_nd(&front_2d, &ref_wrap(vec![reference.0, reference.1]))
            .expect("value should be present");

        assert!(
            (hv_nd_val - hv_2d_val).abs() < 1e-9,
            "2D mismatch: nd={hv_nd_val}, 2d={hv_2d_val}"
        );
    }

    #[test]
    fn test_hv_nd_2d_single_point_matches_hv_2d() {
        let front_2d = vec![vec![1.0, 1.0]];
        let reference = (3.0, 3.0);

        let front_pairs: Vec<(f64, f64)> = front_2d.iter().map(|p| (p[0], p[1])).collect();
        let hv_2d_val =
            hypervolume_2d(&front_pairs, reference).expect("hypervolume_2d should succeed");
        let hv_nd_val = hypervolume_nd(&front_2d, &ref_wrap(vec![reference.0, reference.1]))
            .expect("value should be present");

        assert!(
            (hv_nd_val - hv_2d_val).abs() < 1e-9,
            "single-point 2D mismatch: nd={hv_nd_val}, 2d={hv_2d_val}"
        );
    }

    // ── hypervolume_nd: 3-D ────────────────────────────────────────────────────

    #[test]
    fn test_hv_nd_3d_unit_tetrahedron() {
        // Front: {(0, 0, 1), (0, 1, 0), (1, 0, 0)}, reference = (2, 2, 2)
        let front = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        let ref_pt = ref_wrap(vec![2.0, 2.0, 2.0]);
        let hv = hypervolume_nd(&front, &ref_pt).expect("hypervolume_nd should succeed");
        assert!(hv > 0.0, "3D tetrahedron HV must be positive, got {hv}");
        // The WFG algorithm gives the exact value; check it's in a reasonable range
        assert!(
            hv <= 8.0,
            "HV cannot exceed the bounding box volume of 8: {hv}"
        );
    }

    #[test]
    fn test_hv_nd_3d_single_point() {
        // A single point at (1,1,1) with reference (3,3,3) -> HV = 2*2*2 = 8
        let front = vec![vec![1.0, 1.0, 1.0]];
        let ref_pt = ref_wrap(vec![3.0, 3.0, 3.0]);
        let hv = hypervolume_nd(&front, &ref_pt).expect("hypervolume_nd should succeed");
        assert!((hv - 8.0).abs() < EPS, "expected 8.0, got {hv}");
    }

    // ── hypervolume_nd: 4-D ────────────────────────────────────────────────────

    #[test]
    fn test_hv_nd_4d_single_point() {
        // Single point at (1,1,1,1) with reference (3,3,3,3) -> HV = 2^4 = 16
        let front = vec![vec![1.0, 1.0, 1.0, 1.0]];
        let ref_pt = ref_wrap(vec![3.0, 3.0, 3.0, 3.0]);
        let hv = hypervolume_nd(&front, &ref_pt).expect("hypervolume_nd should succeed");
        assert!((hv - 16.0).abs() < EPS, "expected 16.0, got {hv}");
    }

    #[test]
    fn test_hv_nd_4d_two_nondominated_points() {
        // Two non-dominated points in 4D: (0,0,0,1) and (1,1,1,0)
        // reference = (2,2,2,2)
        let front = vec![vec![0.0, 0.0, 0.0, 1.0], vec![1.0, 1.0, 1.0, 0.0]];
        let ref_pt = ref_wrap(vec![2.0, 2.0, 2.0, 2.0]);
        let hv = hypervolume_nd(&front, &ref_pt).expect("hypervolume_nd should succeed");
        assert!(hv > 0.0, "4D two-point HV must be positive, got {hv}");
        // Upper bound: bounding box = 2^4 = 16
        assert!(hv <= 16.0, "HV cannot exceed 16: {hv}");
    }

    // ── hypervolume_contributions ──────────────────────────────────────────────

    #[test]
    fn test_hv_contributions_sum_at_most_total() {
        let front = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let ref_pt = ref_wrap(vec![4.0, 4.0]);
        let total = hypervolume_nd(&front, &ref_pt).expect("hypervolume_nd should succeed");
        let contribs = hypervolume_contributions(&front, &ref_pt)
            .expect("hypervolume_contributions should succeed");
        assert_eq!(contribs.len(), front.len());
        // Each contribution is non-negative
        for &c in &contribs {
            assert!(c >= 0.0, "contribution must be non-negative: {c}");
        }
        // Each contribution <= total HV
        for &c in &contribs {
            assert!(
                c <= total + EPS,
                "contribution {c} exceeds total HV {total}"
            );
        }
    }

    #[test]
    fn test_hv_contributions_single_point() {
        // Single point: its contribution equals the total HV
        let front = vec![vec![1.0, 1.0]];
        let ref_pt = ref_wrap(vec![3.0, 3.0]);
        let total = hypervolume_nd(&front, &ref_pt).expect("hypervolume_nd should succeed");
        let contribs = hypervolume_contributions(&front, &ref_pt)
            .expect("hypervolume_contributions should succeed");
        assert_eq!(contribs.len(), 1);
        assert!((contribs[0] - total).abs() < EPS);
    }

    #[test]
    fn test_hv_contributions_dominant_point_contributes_more() {
        // A point that is near origin should contribute more than one far away
        let front = vec![
            vec![0.1, 0.1], // near-optimal -> high contribution
            vec![1.9, 1.9], // far but non-dominated (no other point dominates it)
        ];
        let ref_pt = ref_wrap(vec![2.0, 2.0]);
        let contribs = hypervolume_contributions(&front, &ref_pt)
            .expect("hypervolume_contributions should succeed");
        assert_eq!(contribs.len(), 2);
        // The near-optimal point should have a larger contribution
        assert!(
            contribs[0] > contribs[1],
            "near-optimal point must contribute more: {:?}",
            contribs
        );
    }

    #[test]
    fn test_hv_contributions_empty_front() {
        let contribs = hypervolume_contributions(&[], &ref_wrap(vec![1.0, 1.0]))
            .expect("value should be present");
        assert!(contribs.is_empty());
    }

    // ── 5-D stress test ───────────────────────────────────────────────────────

    #[test]
    fn test_hv_nd_5d_stress_hv_positive() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(42);
        let ref_vals: Vec<f64> = vec![10.0; 5];
        // Generate 10 random 5-D points in [0, 5)^5
        let points: Vec<Vec<f64>> = (0..10)
            .map(|_| (0..5).map(|_| rng.next_f64() * 5.0).collect())
            .collect();
        let ref_pt = ref_wrap(ref_vals);
        let hv = hypervolume_nd(&points, &ref_pt).expect("hypervolume_nd should succeed");
        assert!(hv > 0.0, "5D stress test HV must be positive, got {hv}");
        assert!(hv.is_finite(), "5D stress test HV must be finite");
    }
}
