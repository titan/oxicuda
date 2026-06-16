//! Strong Witness Complex (de Silva & Carlsson 2004).
//!
//! A simplex σ = [l₀, …, l_k] is in the strong witness complex at scale R if there
//! exists a witness w such that:
//!   1. D[l_i, w] ≤ R for all i in σ  (all vertex–witness distances ≤ R), and
//!   2. The landmarks l₀, …, l_k are the (k+1) nearest landmarks to w
//!      (i.e. no landmark outside σ is closer to w than the farthest landmark in σ).
//!
//! Equivalently, a simplex's filtration value is the minimum over all valid witnesses of
//! `max_{i} D[l_i, w]`, where "valid" means the l_i form a nearest-neighbour prefix for w.
//!
//! # Algorithm
//! 1. For each witness w, sort the landmarks by ascending distance to w.
//! 2. Consider every prefix of length 1..=(max_dim+1) of that sorted list whose max
//!    distance is ≤ max_radius.  Each such prefix is a candidate simplex (as a set).
//! 3. For each unique simplex, record the minimum filtration value (max-distance) seen
//!    across all witnesses.
//! 4. Sort ascending by filtration value; ties: lower dimension first, then lexicographic.

use crate::error::{TdaError, TdaResult};

// ── Configuration & result ────────────────────────────────────────────────────

/// Configuration for the strong witness complex builder.
#[derive(Debug, Clone)]
pub struct StrongWitnessConfig {
    /// Maximum distance threshold; simplices with filtration value > max_radius are pruned.
    pub max_radius: f64,
    /// Maximum simplex dimension to include (0 = vertices only, 1 = edges, …).
    pub max_dim: usize,
}

/// The result of building a strong witness complex.
#[derive(Debug, Clone)]
pub struct StrongWitnessResult {
    /// Simplices as sorted vertex index lists (landmark indices), paired with their
    /// filtration values.  Sorted ascending by value; ties: dim asc, then lex.
    pub simplices: Vec<(Vec<usize>, f64)>,
    /// Number of landmarks used.
    pub n_landmarks: usize,
    /// Number of witnesses used.
    pub n_witnesses: usize,
}

// ── Main function ─────────────────────────────────────────────────────────────

/// Build the strong witness complex.
///
/// `dist_lw`: flat row-major `n_landmarks × n_witnesses` distance matrix.
/// `dist_lw[l * n_witnesses + w]` = distance from landmark `l` to witness `w`.
///
/// # Errors
/// - [`TdaError::EmptyPointCloud`] if `n_landmarks == 0` or `n_witnesses == 0`.
/// - [`TdaError::DimensionMismatch`] if `dist_lw.len() != n_landmarks * n_witnesses`.
/// - [`TdaError::ParameterOutOfRange`] if `max_radius < 0`.
/// - [`TdaError::NanFiltrationValue`] if any distance entry is NaN.
pub fn strong_witness_complex(
    dist_lw: &[f64],
    n_landmarks: usize,
    n_witnesses: usize,
    cfg: &StrongWitnessConfig,
) -> TdaResult<StrongWitnessResult> {
    if n_landmarks == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if n_witnesses == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if dist_lw.len() != n_landmarks * n_witnesses {
        return Err(TdaError::DimensionMismatch {
            expected: n_landmarks * n_witnesses,
            got: dist_lw.len(),
        });
    }
    if cfg.max_radius < 0.0 {
        return Err(TdaError::ParameterOutOfRange(
            "max_radius must be non-negative".to_owned(),
        ));
    }
    for &d in dist_lw {
        if d.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }

    // Maximum prefix length we need (capped by the number of landmarks).
    let prefix_len = (cfg.max_dim + 1).min(n_landmarks);

    // simplex_best_value: maps sorted simplex (Vec<usize>) -> minimum filtration value.
    let mut simplex_map: std::collections::HashMap<Vec<usize>, f64> =
        std::collections::HashMap::new();

    // Add all vertices (0-simplices) unconditionally at value 0.
    for l in 0..n_landmarks {
        simplex_map.insert(vec![l], 0.0);
    }

    for w in 0..n_witnesses {
        // Collect (distance, landmark_index) for this witness and sort by distance.
        let mut dists: Vec<(f64, usize)> = (0..n_landmarks)
            .map(|l| (dist_lw[l * n_witnesses + w], l))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Consider prefix of length k+1 for k = 0..=max_dim.
        // The filtration value is the max distance in the prefix = dists[k].0.
        for k in 0..prefix_len {
            let max_dist = dists[k].0;
            if max_dist > cfg.max_radius {
                // All longer prefixes will also exceed max_radius.
                break;
            }
            if k == 0 {
                // Vertex — already inserted above at value 0.
                continue;
            }
            // Build the sorted vertex set for this (k+1)-subset (indices 0..=k).
            let mut verts: Vec<usize> = (0..=k).map(|i| dists[i].1).collect();
            verts.sort_unstable();

            // Update the minimum filtration value for this simplex.
            let entry = simplex_map.entry(verts).or_insert(f64::INFINITY);
            if max_dist < *entry {
                *entry = max_dist;
            }
        }
    }

    // Convert map to sorted Vec.
    let mut simplices: Vec<(Vec<usize>, f64)> = simplex_map.into_iter().collect();
    simplices.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.len().cmp(&b.0.len()))
            .then_with(|| a.0.cmp(&b.0))
    });

    Ok(StrongWitnessResult {
        simplices,
        n_landmarks,
        n_witnesses,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_radius: f64, max_dim: usize) -> StrongWitnessConfig {
        StrongWitnessConfig {
            max_radius,
            max_dim,
        }
    }

    // 1. Error: empty landmarks.
    #[test]
    fn error_empty_landmarks() {
        let d: Vec<f64> = vec![];
        assert!(strong_witness_complex(&d, 0, 3, &cfg(1.0, 2)).is_err());
    }

    // 2. Error: empty witnesses.
    #[test]
    fn error_empty_witnesses() {
        let d: Vec<f64> = vec![];
        assert!(strong_witness_complex(&d, 3, 0, &cfg(1.0, 2)).is_err());
    }

    // 3. Error: dimension mismatch.
    #[test]
    fn error_dim_mismatch() {
        // n_landmarks=2, n_witnesses=3 → expects 6 entries; give 5.
        let d = vec![0.0_f64; 5];
        assert!(strong_witness_complex(&d, 2, 3, &cfg(1.0, 2)).is_err());
    }

    // 4. Error: negative max_radius.
    #[test]
    fn error_negative_max_radius() {
        let d = vec![0.0_f64; 4];
        assert!(strong_witness_complex(&d, 2, 2, &cfg(-1.0, 1)).is_err());
    }

    // 5. Error: NaN distance.
    #[test]
    fn error_nan_distance() {
        let d = vec![f64::NAN, 1.0, 0.5, 2.0];
        assert!(strong_witness_complex(&d, 2, 2, &cfg(10.0, 1)).is_err());
    }

    // 6. Single landmark: only one vertex at value 0.
    #[test]
    fn single_landmark() {
        // 1 landmark, 3 witnesses.
        let d = vec![0.1_f64, 0.5, 0.3]; // dist[0, 0], dist[0, 1], dist[0, 2]
        let result = strong_witness_complex(&d, 1, 3, &cfg(1.0, 2)).expect("ok");
        assert_eq!(result.simplices.len(), 1);
        assert_eq!(result.simplices[0].0, vec![0]);
        assert!(
            (result.simplices[0].1).abs() < 1e-12,
            "vertex must be at value 0"
        );
    }

    // 7. max_dim=0 yields only vertices.
    #[test]
    fn max_dim_zero_vertices_only() {
        // 3 landmarks, 3 witnesses.
        let d: Vec<f64> = vec![
            // Rows = landmarks, cols = witnesses.
            0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.2, 0.3, 0.1,
        ];
        let result = strong_witness_complex(&d, 3, 3, &cfg(1.0, 0)).expect("ok");
        assert!(
            result.simplices.iter().all(|(v, _)| v.len() == 1),
            "max_dim=0 should yield only vertices"
        );
    }

    // 8. 3-landmark 3-witness case: vertices always present.
    #[test]
    fn three_landmark_three_witness_has_all_vertices() {
        let d: Vec<f64> = vec![0.1, 2.0, 3.0, 2.0, 0.1, 3.0, 3.0, 3.0, 0.1];
        let result = strong_witness_complex(&d, 3, 3, &cfg(5.0, 2)).expect("ok");
        // All 3 vertices should be present at value 0.
        let vert_count = result
            .simplices
            .iter()
            .filter(|(v, val)| v.len() == 1 && val.abs() < 1e-12)
            .count();
        assert_eq!(vert_count, 3);
    }

    // 9. Sorted ascending by filtration value.
    #[test]
    fn sorted_ascending() {
        let d: Vec<f64> = vec![0.1, 0.5, 1.0, 0.9, 0.2, 0.8, 0.7, 0.6, 0.3];
        let result = strong_witness_complex(&d, 3, 3, &cfg(5.0, 2)).expect("ok");
        for w in result.simplices.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-12,
                "filtration not sorted at {:?} -> {:?}",
                w[0],
                w[1]
            );
        }
    }

    // 10. max_radius pruning: with tiny max_radius only vertices remain.
    #[test]
    fn max_radius_prunes_edges() {
        let d: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
        // max_radius = 0: no edges possible (all distances > 0).
        let result = strong_witness_complex(&d, 2, 2, &cfg(0.0, 1)).expect("ok");
        assert!(
            result.simplices.iter().all(|(v, _)| v.len() == 1),
            "all edges should be pruned"
        );
    }

    // 11. Self-distance case: landmark is also the witness (distance 0).
    #[test]
    fn self_distance_zero() {
        // 2 landmarks, each is its own nearest witness (distances 0).
        let d: Vec<f64> = vec![0.0, 3.0, 3.0, 0.0];
        let result = strong_witness_complex(&d, 2, 2, &cfg(5.0, 1)).expect("ok");
        // Vertex 0 filtration = 0, vertex 1 filtration = 0.
        for (v, val) in result.simplices.iter().filter(|(v, _)| v.len() == 1) {
            assert!(
                val.abs() < 1e-12,
                "vertex {:?} should have value 0, got {}",
                v,
                val
            );
        }
    }

    // 12. n_landmarks and n_witnesses fields are correctly set.
    #[test]
    fn result_fields_correct() {
        let d: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let result = strong_witness_complex(&d, 2, 4, &cfg(1.0, 1)).expect("ok");
        assert_eq!(result.n_landmarks, 2);
        assert_eq!(result.n_witnesses, 4);
    }

    // 13. Larger case: 4 landmarks, 5 witnesses.
    #[test]
    fn larger_case() {
        // 4 landmarks × 5 witnesses distance matrix (all in [0, 1]).
        let d: Vec<f64> = vec![
            0.1, 0.9, 0.8, 0.7, 0.6, 0.9, 0.1, 0.7, 0.6, 0.5, 0.8, 0.7, 0.1, 0.5, 0.4, 0.7, 0.6,
            0.5, 0.1, 0.3,
        ];
        let result = strong_witness_complex(&d, 4, 5, &cfg(1.0, 2)).expect("ok");
        // Should have all 4 vertices at value 0.
        let verts = result
            .simplices
            .iter()
            .filter(|(v, _)| v.len() == 1)
            .count();
        assert_eq!(verts, 4);
        // Simplices should be sorted.
        for w in result.simplices.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-12);
        }
    }
}
