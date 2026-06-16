//! Bottleneck distance between raw persistence diagrams (slice-based API).
//!
//! This module provides functions that operate directly on `&[(f64, f64)]` birth-death
//! slices rather than on [`crate::persistence::PersistenceDiagram`] structs, making them usable in
//! standalone pipelines.  The underlying algorithm is identical to
//! `persistence::distance::bottleneck_distance`: binary search over candidate
//! thresholds + DFS augmenting-path bipartite matching.
//!
//! Cohen-Steiner, Edelsbrunner & Harer (2010) "Lipschitz Functions have L_p-Stable
//! Persistence" — bottleneck is the L^∞ optimal-transport distance between diagrams
//! augmented with their diagonals.

use crate::error::TdaResult;

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// L^∞ distance between two birth-death points.
#[inline]
fn linf_dist(b1: f64, d1: f64, b2: f64, d2: f64) -> f64 {
    (b1 - b2).abs().max((d1 - d2).abs())
}

/// L^∞ distance from a point (b, d) to the diagonal ((b+d)/2, (b+d)/2).
#[inline]
fn diag_dist(b: f64, d: f64) -> f64 {
    (d - b).abs() / 2.0
}

/// DFS-based augmenting path search for bipartite matching at threshold `t`.
///
/// `adj[i]` lists the columns that row `i` is allowed to match (cost ≤ t).
/// `match_col[j]` = currently matched row for column `j` (usize::MAX if free).
fn augment(
    u: usize,
    adj: &[Vec<usize>],
    match_col: &mut [usize],
    visited_col: &mut [bool],
) -> bool {
    for &v in &adj[u] {
        if visited_col[v] {
            continue;
        }
        visited_col[v] = true;
        let prev = match_col[v];
        if prev == usize::MAX || augment(prev, adj, match_col, visited_col) {
            match_col[v] = u;
            return true;
        }
    }
    false
}

/// Check whether a perfect matching exists when every edge costs ≤ `threshold`.
fn perfect_matching_exists(
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    n: usize,
    threshold: f64,
) -> bool {
    debug_assert_eq!(pts1.len(), n);
    debug_assert_eq!(pts2.len(), n);

    // Build adjacency
    let adj: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| linf_dist(pts1[i].0, pts1[i].1, pts2[j].0, pts2[j].1) <= threshold)
                .collect()
        })
        .collect();

    // Hopcroft-Karp-style greedy init followed by augmenting paths
    let mut match_col = vec![usize::MAX; n];
    // Greedy pass first
    let mut matched_row = vec![false; n];
    for i in 0..n {
        for &j in &adj[i] {
            if match_col[j] == usize::MAX {
                match_col[j] = i;
                matched_row[i] = true;
                break;
            }
        }
    }
    // Augmenting paths for remaining
    let unmatched: Vec<usize> = (0..n).filter(|&i| !matched_row[i]).collect();
    for i in unmatched {
        let mut visited_col = vec![false; n];
        // Need to rebuild adj lazily for augment — reuse the same adj
        if augment(i, &adj, &mut match_col, &mut visited_col) {
            matched_row[i] = true;
        }
    }
    // Verify all rows are matched
    match_col.iter().all(|&r| r != usize::MAX) && {
        let mut covered = vec![false; n];
        for &r in &match_col {
            if r == usize::MAX {
                return false;
            }
            covered[r] = true;
        }
        covered.iter().all(|&c| c)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Bottleneck distance W_∞ between two persistence diagrams given as raw slices.
///
/// Each diagram is a slice of `(birth, death)` pairs; pairs with `birth > death` are
/// silently skipped.  The diagonal augmentation (unmatched points are matched to their
/// nearest diagonal projection) is handled automatically.
///
/// Returns `0.0` if both diagrams are empty.
///
/// # Algorithm
/// 1. Augment each set to equal size `n1 + n2` by adding diagonal projections.
/// 2. Collect all candidate threshold values (pairwise L^∞ costs + diagonal distances).
/// 3. Binary search: smallest threshold for which a perfect matching exists.
///
/// Complexity: O((n1 + n2)² log(n1 + n2)) candidate thresholds, each matched in
/// O((n1 + n2)²) DFS.  Total O((n1 + n2)⁴ log(n1 + n2)) worst case, but practical
/// sizes are small (diagrams up to ~1 000 pairs are routine in TDA).
pub fn bottleneck_distance(dgm1: &[(f64, f64)], dgm2: &[(f64, f64)]) -> TdaResult<f64> {
    // Filter degenerate points
    let pts1: Vec<(f64, f64)> = dgm1.iter().filter(|&&(b, d)| b <= d).copied().collect();
    let pts2: Vec<(f64, f64)> = dgm2.iter().filter(|&&(b, d)| b <= d).copied().collect();

    let n1 = pts1.len();
    let n2 = pts2.len();

    if n1 == 0 && n2 == 0 {
        return Ok(0.0);
    }

    // Augmented size: each set gains |other set| diagonal projections
    let n = n1 + n2;

    // pts1_aug: real pts1 + diagonal projections of pts2
    let mut pts1_aug: Vec<(f64, f64)> = pts1.clone();
    for &(b, d) in &pts2 {
        let m = (b + d) / 2.0;
        pts1_aug.push((m, m));
    }

    // pts2_aug: real pts2 + diagonal projections of pts1
    let mut pts2_aug: Vec<(f64, f64)> = pts2.clone();
    for &(b, d) in &pts1 {
        let m = (b + d) / 2.0;
        pts2_aug.push((m, m));
    }

    debug_assert_eq!(pts1_aug.len(), n);
    debug_assert_eq!(pts2_aug.len(), n);

    // Collect candidate thresholds
    let mut candidates: Vec<f64> = Vec::with_capacity(n * n + 2 * n);
    for &(b1, d1) in &pts1_aug {
        for &(b2, d2) in &pts2_aug {
            candidates.push(linf_dist(b1, d1, b2, d2));
        }
    }
    for &(b, d) in pts1_aug.iter().chain(pts2_aug.iter()) {
        candidates.push(diag_dist(b, d));
    }
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    if candidates.is_empty() {
        return Ok(0.0);
    }

    // Binary search for the smallest threshold that admits a perfect matching
    let mut lo = 0usize;
    let mut hi = candidates.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let t = candidates[mid];
        if perfect_matching_exists(&pts1_aug, &pts2_aug, n, t) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    Ok(if lo < candidates.len() {
        candidates[lo]
    } else {
        *candidates.last().unwrap_or(&0.0)
    })
}

/// Persistence landscape L_k(t) for a raw diagram slice.
///
/// For each time `t` in `t_grid`, computes the k-th largest tent value
/// `tent(b, d, t) = max(0, min(t - b, d - t))` over all diagram pairs `(b, d)`.
///
/// Indexing is **1-based** (`k = 1` is the largest tent value).
///
/// Returns a zero vector if the diagram is empty or `k` exceeds the number of pairs.
///
/// # Errors
/// Returns `TdaError::ParameterOutOfRange` if `k == 0`.
pub fn persistence_landscape(dgm: &[(f64, f64)], k: usize, t_grid: &[f64]) -> TdaResult<Vec<f64>> {
    use crate::error::TdaError;

    if k == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "k must be ≥ 1 (1-indexed)".to_owned(),
        ));
    }

    // Filter degenerate pairs
    let pairs: Vec<(f64, f64)> = dgm.iter().filter(|&&(b, d)| b < d).copied().collect();

    if pairs.is_empty() || k > pairs.len() {
        return Ok(vec![0.0; t_grid.len()]);
    }

    let result = t_grid
        .iter()
        .map(|&t| {
            let mut vals: Vec<f64> = pairs
                .iter()
                .map(|&(b, d)| ((t - b).min(d - t)).max(0.0))
                .collect();
            // Sort descending and take k-th largest (1-indexed → index k-1)
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            if k <= vals.len() { vals[k - 1] } else { 0.0 }
        })
        .collect();

    Ok(result)
}

/// Mean persistence landscape (average tent value over all pairs) at each grid point.
///
/// Equivalent to calling `persistence_landscape` with `k = 1` and summing all tent
/// values divided by the number of pairs at each `t`.
///
/// Returns a zero vector if the diagram is empty.
pub fn landscape_mean(dgm: &[(f64, f64)], t_grid: &[f64]) -> Vec<f64> {
    let pairs: Vec<(f64, f64)> = dgm.iter().filter(|&&(b, d)| b < d).copied().collect();

    if pairs.is_empty() {
        return vec![0.0; t_grid.len()];
    }

    let n = pairs.len() as f64;
    t_grid
        .iter()
        .map(|&t| {
            let sum: f64 = pairs
                .iter()
                .map(|&(b, d)| ((t - b).min(d - t)).max(0.0))
                .sum();
            sum / n
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── bottleneck_distance tests ────────────────────────────────────────────

    /// Bottleneck distance from a diagram to itself is zero.
    #[test]
    fn bottleneck_zero_same() {
        let dgm = vec![(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)];
        let d = bottleneck_distance(&dgm, &dgm).expect("bottleneck_distance should succeed");
        assert!(d < 1e-12, "distance to self should be zero, got {d}");
    }

    /// Bottleneck distance is symmetric: d(A, B) == d(B, A).
    #[test]
    fn bottleneck_symmetric() {
        let dgm1 = vec![(0.0, 1.0), (1.0, 4.0)];
        let dgm2 = vec![(0.0, 2.0), (0.5, 1.5)];
        let d_ab = bottleneck_distance(&dgm1, &dgm2).expect("bottleneck_distance should succeed");
        let d_ba = bottleneck_distance(&dgm2, &dgm1).expect("bottleneck_distance should succeed");
        assert!(
            (d_ab - d_ba).abs() < 1e-12,
            "symmetry violated: d(A,B)={d_ab}, d(B,A)={d_ba}"
        );
    }

    /// Bottleneck distance between non-identical diagrams is positive.
    #[test]
    fn bottleneck_positive() {
        let dgm1 = vec![(0.0, 2.0)];
        let dgm2 = vec![(1.0, 3.0)];
        let d = bottleneck_distance(&dgm1, &dgm2).expect("bottleneck_distance should succeed");
        assert!(
            d > 0.0,
            "distance between distinct diagrams should be > 0, got {d}"
        );
    }

    /// Both diagrams empty → distance is zero.
    #[test]
    fn empty_dgm_distance() {
        let d = bottleneck_distance(&[], &[]).expect("bottleneck_distance should succeed");
        assert!(d < 1e-12, "empty diagrams should have distance 0, got {d}");
    }

    /// One diagram empty, one with a single point: distance = diag_dist of that point.
    #[test]
    fn single_point_dgm() {
        let b = 1.0_f64;
        let d_val = 3.0_f64;
        let dgm = vec![(b, d_val)];
        let dist = bottleneck_distance(&dgm, &[]).expect("bottleneck_distance should succeed");
        let expected = (d_val - b) / 2.0;
        assert!(
            (dist - expected).abs() < 1e-10,
            "expected {expected}, got {dist}"
        );
    }

    /// Known example: two diagrams each with one off-diagonal point, 1-to-1 match.
    #[test]
    fn bottleneck_known_one_to_one() {
        // dgm1 = {(0, 4)}, dgm2 = {(1, 3)}
        // Cost to match directly: L∞((0,4),(1,3)) = max(1, 1) = 1
        // Cost to unmatch dgm1 and use diagonal: diag_dist(0,4)=2, diag_dist(1,3)=1
        // Direct match costs 1 which is better than both unmatched (max(2,1)=2)
        let dgm1 = vec![(0.0, 4.0)];
        let dgm2 = vec![(1.0, 3.0)];
        let dist = bottleneck_distance(&dgm1, &dgm2).expect("bottleneck_distance should succeed");
        assert!((dist - 1.0).abs() < 1e-10, "expected 1.0, got {dist}");
    }

    /// Bottleneck satisfies triangle inequality: d(A,C) ≤ d(A,B) + d(B,C).
    #[test]
    fn bottleneck_triangle_inequality() {
        let a = vec![(0.0, 1.0), (2.0, 5.0)];
        let b = vec![(0.5, 1.5), (2.0, 4.0)];
        let c = vec![(1.0, 2.0), (3.0, 6.0)];
        let dab = bottleneck_distance(&a, &b).expect("bottleneck_distance should succeed");
        let dbc = bottleneck_distance(&b, &c).expect("bottleneck_distance should succeed");
        let dac = bottleneck_distance(&a, &c).expect("bottleneck_distance should succeed");
        assert!(
            dac <= dab + dbc + 1e-10,
            "triangle inequality violated: {dac} > {dab} + {dbc}"
        );
    }

    /// Diagrams with degenerate (birth == death) points: those are skipped.
    #[test]
    fn bottleneck_degenerate_points_skipped() {
        let dgm1 = vec![(1.0, 1.0), (0.0, 2.0)];
        let dgm2 = vec![(0.0, 2.0)];
        // (1, 1) is on the diagonal → ignored; should give distance ~0
        let dist = bottleneck_distance(&dgm1, &dgm2).expect("bottleneck_distance should succeed");
        assert!(dist < 1e-10, "degenerate point skipped, got {dist}");
    }

    /// Multiple points, equal diagrams → zero.
    #[test]
    fn bottleneck_multi_point_same() {
        let dgm = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (0.0, 0.5)];
        let d = bottleneck_distance(&dgm, &dgm).expect("bottleneck_distance should succeed");
        assert!(d < 1e-12, "self-distance should be zero, got {d}");
    }

    // ── persistence_landscape tests ──────────────────────────────────────────

    /// Landscape output length matches t_grid length.
    #[test]
    fn landscape_shape() {
        let dgm = vec![(0.0, 2.0), (1.0, 3.0)];
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.2).collect();
        let lnd = persistence_landscape(&dgm, 1, &t).expect("persistence_landscape should succeed");
        assert_eq!(lnd.len(), t.len());
    }

    /// Landscape values are non-negative.
    #[test]
    fn landscape_nonneg() {
        let dgm = vec![(0.0, 4.0), (1.0, 3.0), (2.0, 5.0)];
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let lnd = persistence_landscape(&dgm, 1, &t).expect("persistence_landscape should succeed");
        for &v in &lnd {
            assert!(v >= 0.0, "landscape value is negative: {v}");
        }
    }

    /// Landscape values are finite (no NaN or Inf).
    #[test]
    fn landscape_finite() {
        let dgm = vec![(0.0, 10.0), (5.0, 7.0)];
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let lnd = persistence_landscape(&dgm, 1, &t).expect("persistence_landscape should succeed");
        for &v in &lnd {
            assert!(v.is_finite(), "landscape value is not finite: {v}");
        }
    }

    /// For a single pair (b, d), the landscape k=1 at t=(b+d)/2 equals (d-b)/2.
    #[test]
    fn landscape_tent_peak() {
        let b = 0.0_f64;
        let d = 4.0_f64;
        let t_mid = (b + d) / 2.0; // 2.0
        let lnd = persistence_landscape(&[(b, d)], 1, &[t_mid]).expect("value should be present");
        let expected = (d - b) / 2.0; // 2.0
        assert!(
            (lnd[0] - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            lnd[0]
        );
    }

    /// k=2 returns zero for a single-pair diagram (only one tent).
    #[test]
    fn landscape_k2_single_pair() {
        let dgm = vec![(0.0, 4.0)];
        let t = vec![2.0];
        let lnd = persistence_landscape(&dgm, 2, &t).expect("persistence_landscape should succeed");
        assert!((lnd[0]).abs() < 1e-12, "k=2 with 1 pair should be 0");
    }

    /// k=0 returns an error.
    #[test]
    fn landscape_k0_error() {
        let dgm = vec![(0.0, 1.0)];
        let t = vec![0.5];
        assert!(
            persistence_landscape(&dgm, 0, &t).is_err(),
            "k=0 should be an error"
        );
    }

    // ── landscape_mean tests ─────────────────────────────────────────────────

    /// Mean landscape output length matches t_grid length.
    #[test]
    fn landscape_mean_shape() {
        let dgm = vec![(0.0, 2.0), (1.0, 5.0)];
        let t: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
        let mean = landscape_mean(&dgm, &t);
        assert_eq!(mean.len(), t.len());
    }

    /// Mean landscape is non-negative.
    #[test]
    fn landscape_mean_nonneg() {
        let dgm = vec![(0.0, 3.0), (1.0, 4.0), (2.0, 6.0)];
        let t: Vec<f64> = (0..40).map(|i| i as f64 * 0.2).collect();
        let mean = landscape_mean(&dgm, &t);
        for &v in &mean {
            assert!(v >= 0.0, "mean landscape value is negative: {v}");
        }
    }

    /// Mean landscape for empty diagram is all zeros.
    #[test]
    fn landscape_mean_empty() {
        let mean = landscape_mean(&[], &[0.0, 1.0, 2.0]);
        assert_eq!(mean, vec![0.0, 0.0, 0.0]);
    }

    /// Mean landscape for a single pair equals the tent function itself.
    #[test]
    fn landscape_mean_single_pair() {
        let b = 0.0_f64;
        let d = 4.0_f64;
        let t_mid = 2.0_f64;
        let mean = landscape_mean(&[(b, d)], &[t_mid]);
        let expected = 2.0_f64; // tent peak = (d-b)/2 = 2, mean of 1 pair = itself
        assert!(
            (mean[0] - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            mean[0]
        );
    }
}
