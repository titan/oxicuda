//! Bottleneck and Wasserstein distances between persistence diagrams.
//!
//! Both metrics compare diagrams in the same homological dimension.
//! Unmatched points are matched to their diagonal projections.

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

/// L∞ distance between two persistence points (b1, d1) and (b2, d2).
#[inline]
fn point_distance(b1: f64, d1: f64, b2: f64, d2: f64) -> f64 {
    (b1 - b2).abs().max((d1 - d2).abs())
}

/// Distance from a persistence point (birth, death) to the diagonal.
///
/// The diagonal projection of (b, d) is ((b+d)/2, (b+d)/2).
/// The L∞ distance from (b, d) to ((b+d)/2, (b+d)/2) is (d - b) / 2.
#[inline]
fn diagonal_distance(birth: f64, death: f64) -> f64 {
    (death - birth) / 2.0
}

/// Bottleneck distance W_∞ between two persistence diagrams.
///
/// Exact computation via binary search + bipartite matching.
///
/// Algorithm:
/// 1. Collect all finite points from both diagrams.
/// 2. Augment each set with diagonal projections for every point in the *other* set.
/// 3. Collect all candidate threshold values (pairwise distances + diagonal distances).
/// 4. Binary search on threshold T: check if perfect matching with all costs ≤ T exists.
/// 5. Use DFS-based Hopcroft-Karp bipartite matching for each candidate threshold.
pub fn bottleneck_distance(
    diag1: &PersistenceDiagram,
    diag2: &PersistenceDiagram,
) -> TdaResult<f64> {
    // Collect finite points
    let pts1: Vec<(f64, f64)> = diag1
        .finite_pairs()
        .iter()
        .map(|p| (p.birth, p.death.unwrap_or(0.0)))
        .collect();
    let pts2: Vec<(f64, f64)> = diag2
        .finite_pairs()
        .iter()
        .map(|p| (p.birth, p.death.unwrap_or(0.0)))
        .collect();

    let n1 = pts1.len();
    let n2 = pts2.len();

    if n1 == 0 && n2 == 0 {
        return Ok(0.0);
    }

    // Augment: each point from set A can be matched to its diagonal projection
    // and vice-versa.  The effective augmented sets have the same size n1 + n2.
    // pts1_aug: pts1 + diagonal projections of pts2
    // pts2_aug: pts2 + diagonal projections of pts1
    let n = n1 + n2;
    let mut pts1_aug: Vec<(f64, f64)> = pts1.clone();
    for &(b, d) in &pts2 {
        let m = (b + d) / 2.0;
        pts1_aug.push((m, m));
    }
    let mut pts2_aug: Vec<(f64, f64)> = pts2.clone();
    for &(b, d) in &pts1 {
        let m = (b + d) / 2.0;
        pts2_aug.push((m, m));
    }

    // Build all candidate threshold values
    let mut candidates: Vec<f64> = Vec::new();
    for p1 in pts1_aug.iter() {
        for p2 in pts2_aug.iter() {
            let cost = point_distance(p1.0, p1.1, p2.0, p2.1);
            candidates.push(cost);
        }
    }
    // Include diagonal distances
    for &(b, d) in pts1_aug.iter().chain(pts2_aug.iter()) {
        candidates.push(diagonal_distance(b, d));
    }
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    if candidates.is_empty() {
        return Ok(0.0);
    }

    // Binary search on the sorted candidate list
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

    if lo >= candidates.len() {
        // All candidates exhausted — return max
        return Ok(*candidates.last().unwrap_or(&0.0));
    }
    Ok(candidates[lo])
}

/// Check whether a perfect bipartite matching exists with all edge costs ≤ threshold `t`.
///
/// Uses augmenting-path (DFS) approach.
fn perfect_matching_exists(left: &[(f64, f64)], right: &[(f64, f64)], n: usize, t: f64) -> bool {
    // Build adjacency for left side: left[i] can be matched to right[j] if cost ≤ t
    let mut match_r = vec![usize::MAX; n]; // match_r[j] = i if right j is matched to left i
    let mut matched = 0usize;

    for i in 0..n {
        let mut visited = vec![false; n];
        if augment(i, left, right, n, t, &mut match_r, &mut visited) {
            matched += 1;
        }
    }
    matched == n
}

fn augment(
    i: usize,
    left: &[(f64, f64)],
    right: &[(f64, f64)],
    n: usize,
    t: f64,
    match_r: &mut Vec<usize>,
    visited: &mut Vec<bool>,
) -> bool {
    for j in 0..n {
        if visited[j] {
            continue;
        }
        let cost = point_distance(left[i].0, left[i].1, right[j].0, right[j].1);
        if cost <= t + 1e-12 {
            visited[j] = true;
            let prev = match_r[j];
            if prev == usize::MAX || augment(prev, left, right, n, t, match_r, visited) {
                match_r[j] = i;
                return true;
            }
        }
    }
    false
}

/// 1-Wasserstein distance between two persistence diagrams.
///
/// Uses a greedy matching on finite points, augmented with diagonal contributions.
pub fn wasserstein_1(diag1: &PersistenceDiagram, diag2: &PersistenceDiagram) -> TdaResult<f64> {
    let pts1: Vec<(f64, f64)> = diag1
        .finite_pairs()
        .iter()
        .map(|p| (p.birth, p.death.unwrap_or(0.0)))
        .collect();
    let pts2: Vec<(f64, f64)> = diag2
        .finite_pairs()
        .iter()
        .map(|p| (p.birth, p.death.unwrap_or(0.0)))
        .collect();

    let n1 = pts1.len();
    let n2 = pts2.len();

    if n1 == 0 && n2 == 0 {
        return Ok(0.0);
    }

    // Augment both sets with diagonal projections to make them the same size
    let n = n1 + n2;
    let mut pts1_aug: Vec<(f64, f64)> = pts1.clone();
    for &(b, d) in &pts2 {
        let m = (b + d) / 2.0;
        pts1_aug.push((m, m));
    }
    let mut pts2_aug: Vec<(f64, f64)> = pts2.clone();
    for &(b, d) in &pts1 {
        let m = (b + d) / 2.0;
        pts2_aug.push((m, m));
    }

    // Hungarian algorithm O(n³) for minimum-weight perfect matching (L∞ cost)
    // We use the classic augmenting-path method with a cost matrix.
    let total = hungarian_min_cost(&pts1_aug, &pts2_aug, n).map_err(TdaError::MatchingFailed)?;
    Ok(total)
}

/// Hungarian algorithm for minimum-cost perfect bipartite matching.
///
/// Returns the sum of matched edge costs (L1 on the (birth, death) plane for each pair).
fn hungarian_min_cost(left: &[(f64, f64)], right: &[(f64, f64)], n: usize) -> Result<f64, String> {
    if n == 0 {
        return Ok(0.0);
    }

    // Build cost matrix: L1 norm (|b1-b2| + |d1-d2|)
    let mut cost: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            cost[i][j] = (left[i].0 - right[j].0).abs() + (left[i].1 - right[j].1).abs();
        }
    }

    // Jonker-Volgenant / Kuhn-Munkres (simple O(n³) Kuhn-Munkres)
    let big = f64::MAX / 2.0;
    let mut u = vec![0.0_f64; n + 1]; // row potentials
    let mut v = vec![0.0_f64; n + 1]; // col potentials
    let mut p = vec![0usize; n + 1]; // p[j] = row matched to col j (1-indexed)
    let mut way = vec![0usize; n + 1]; // way[j] = previous col in augmenting path

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minval = vec![big; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = big;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minval[j] {
                        minval[j] = cur;
                        way[j] = j0;
                    }
                    if minval[j] < delta {
                        delta = minval[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minval[j] -= delta;
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

    // Compute total cost
    let mut total = 0.0_f64;
    for j in 1..=n {
        if p[j] != 0 {
            total += cost[p[j] - 1][j - 1];
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;
    use crate::persistence::diagram::PersistenceDiagram;

    fn diag_from_points(pts: &[(f64, f64)]) -> PersistenceDiagram {
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

    #[test]
    fn bottleneck_self_zero() {
        let d = diag_from_points(&[(0.0, 1.0), (0.5, 2.0)]);
        let dist = bottleneck_distance(&d, &d).expect("ok");
        assert!(dist < 1e-10, "self bottleneck = {dist}");
    }

    #[test]
    fn wasserstein_empty_diagrams() {
        let d1 = PersistenceDiagram::new(vec![], 0);
        let d2 = PersistenceDiagram::new(vec![], 0);
        assert_eq!(wasserstein_1(&d1, &d2).expect("ok"), 0.0);
    }
}
