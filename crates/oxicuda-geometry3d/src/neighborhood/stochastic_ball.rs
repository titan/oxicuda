//! Stochastic ball query (random in-radius subsampling).
//!
//! The plain [`crate::neighborhood::ball_query::ball_query`] returns the first
//! `k_max` points found inside the radius, which biases the neighbourhood
//! toward low point indices. PointNet++ instead draws a *random* fixed-size
//! subset of the in-radius candidates so that, over training, every neighbour
//! is sampled and the grouping is order-independent.
//!
//! This implementation follows the standard PointNet++ grouping convention:
//!
//! * Gather **all** points with `‖p − q‖² < r²`.
//! * If the candidate count exceeds `nsample`, draw `nsample` of them without
//!   replacement via a partial Fisher–Yates shuffle on the crate's
//!   deterministic [`LcgRng`].
//! * If `0 < count < nsample`, pad the result by repeating the found neighbours
//!   (cycling), so every output row has exactly `nsample` valid entries.
//! * If `count == 0`, the row is filled with the sentinel `usize::MAX`.
//!
//! Determinism: given the same seed the sampled indices are reproducible, so
//! unit tests do not depend on external randomness.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

/// Stochastic ball query.
///
/// Returns `(indices, counts)` where `indices` is row-major `[nq × nsample]`
/// and `counts[q]` is the number of *distinct* in-radius neighbours found for
/// query `q` (before padding). Empty slots use `usize::MAX`.
///
/// # Errors
///
/// * [`Geom3dError::EmptyPointCloud`] if `nq == 0` or `np == 0`.
/// * [`Geom3dError::DimensionMismatch`] if the coordinate buffers are mis-sized.
/// * [`Geom3dError::InvalidRadius`] if `radius <= 0` or non-finite.
/// * [`Geom3dError::InvalidK`] if `nsample == 0`.
pub fn stochastic_ball_query(
    queries: &[f32],
    nq: usize,
    points: &[f32],
    np: usize,
    nsample: usize,
    radius: f32,
    seed: u64,
) -> Geom3dResult<(Vec<usize>, Vec<usize>)> {
    if nq == 0 || np == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if queries.len() != nq * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: nq * 3,
            got: queries.len(),
        });
    }
    if points.len() != np * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: np * 3,
            got: points.len(),
        });
    }
    if radius <= 0.0 || !radius.is_finite() {
        return Err(Geom3dError::InvalidRadius { radius });
    }
    if nsample == 0 {
        return Err(Geom3dError::InvalidK { k: 0, n: np });
    }

    let r_sq = radius * radius;
    let mut indices = vec![usize::MAX; nq * nsample];
    let mut counts = vec![0usize; nq];
    let mut rng = LcgRng::new(seed);
    let mut candidates: Vec<usize> = Vec::new();

    for qi in 0..nq {
        let qx = queries[qi * 3];
        let qy = queries[qi * 3 + 1];
        let qz = queries[qi * 3 + 2];

        candidates.clear();
        for pi in 0..np {
            let dx = points[pi * 3] - qx;
            let dy = points[pi * 3 + 1] - qy;
            let dz = points[pi * 3 + 2] - qz;
            if dx * dx + dy * dy + dz * dz < r_sq {
                candidates.push(pi);
            }
        }
        counts[qi] = candidates.len();

        let row = &mut indices[qi * nsample..(qi + 1) * nsample];
        match candidates.len() {
            0 => {}
            c if c <= nsample => {
                // Use all candidates, then pad by cycling through them.
                for (slot, dst) in row.iter_mut().enumerate() {
                    *dst = candidates[slot % c];
                }
            }
            c => {
                // Partial Fisher–Yates: pick `nsample` of `c` without
                // replacement. Shuffle the front `nsample` positions.
                for slot in 0..nsample {
                    let j = slot + rng.next_usize(c - slot);
                    candidates.swap(slot, j);
                    row[slot] = candidates[slot];
                }
            }
        }
    }

    Ok((indices, counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn line_points(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect()
    }

    #[test]
    fn subsamples_without_replacement() {
        // 20 coincident points at the origin; nsample=5, large radius.
        let pts: Vec<f32> = (0..20).flat_map(|_| vec![0.0_f32, 0.0, 0.0]).collect();
        let q = vec![0.0_f32, 0.0, 0.0];
        let (idx, cnt) =
            stochastic_ball_query(&q, 1, &pts, 20, 5, 1.0, 42).expect("query should succeed");
        assert_eq!(cnt[0], 20);
        let set: HashSet<usize> = idx[..5].iter().copied().collect();
        assert_eq!(set.len(), 5, "subsample must be without replacement");
        assert!(idx[..5].iter().all(|&i| i < 20));
    }

    #[test]
    fn pads_when_fewer_than_nsample() {
        // Only points 3..=7 are within radius 2 of query 5.0.
        let pts = line_points(10);
        let q = vec![5.0_f32, 0.0, 0.0];
        let (idx, cnt) =
            stochastic_ball_query(&q, 1, &pts, 10, 8, 2.0, 1).expect("query should succeed");
        // In-radius (d < 2.0) of query 5.0 are points 4, 5, 6 (d = 1, 0, 1);
        // points 3 and 7 sit at d = 2.0 and are excluded.
        assert_eq!(cnt[0], 3);
        // Row must have exactly nsample valid entries, all in-radius.
        let valid: Vec<usize> = idx.iter().copied().filter(|&v| v != usize::MAX).collect();
        assert_eq!(valid.len(), 8, "row must be fully padded to nsample");
        for &v in &valid {
            let d = (v as f32 - 5.0).abs();
            assert!(d < 2.0, "padded index {v} must be in radius");
        }
    }

    #[test]
    fn empty_neighbourhood_sentinels() {
        let pts: Vec<f32> = vec![100.0, 100.0, 100.0];
        let q = vec![0.0_f32, 0.0, 0.0];
        let (idx, cnt) =
            stochastic_ball_query(&q, 1, &pts, 1, 4, 1.0, 0).expect("query should succeed");
        assert_eq!(cnt[0], 0);
        assert!(idx.iter().all(|&i| i == usize::MAX));
    }

    #[test]
    fn deterministic_same_seed() {
        let pts: Vec<f32> = (0..30).flat_map(|_| vec![0.0_f32, 0.0, 0.0]).collect();
        let q = vec![0.0_f32, 0.0, 0.0];
        let a = stochastic_ball_query(&q, 1, &pts, 30, 8, 1.0, 99).expect("query should succeed");
        let b = stochastic_ball_query(&q, 1, &pts, 30, 8, 1.0, 99).expect("query should succeed");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn different_seed_different_subset() {
        let pts: Vec<f32> = (0..40).flat_map(|_| vec![0.0_f32, 0.0, 0.0]).collect();
        let q = vec![0.0_f32, 0.0, 0.0];
        let a = stochastic_ball_query(&q, 1, &pts, 40, 6, 1.0, 1).expect("query should succeed");
        let b = stochastic_ball_query(&q, 1, &pts, 40, 6, 1.0, 2).expect("query should succeed");
        // Very unlikely the two random subsets coincide exactly.
        assert_ne!(a.0, b.0, "different seeds should give different subsets");
    }

    #[test]
    fn all_sampled_indices_are_in_radius() {
        let pts = line_points(50);
        let q = vec![25.0_f32, 0.0, 0.0];
        let (idx, _) =
            stochastic_ball_query(&q, 1, &pts, 50, 7, 5.0, 7).expect("query should succeed");
        for &v in idx.iter().filter(|&&v| v != usize::MAX) {
            assert!((v as f32 - 25.0).abs() < 5.0);
        }
    }

    #[test]
    fn invalid_args_error() {
        let pts = line_points(5);
        let q = vec![0.0_f32, 0.0, 0.0];
        assert!(stochastic_ball_query(&q, 1, &pts, 5, 3, 0.0, 0).is_err()); // radius
        assert!(stochastic_ball_query(&q, 1, &pts, 5, 0, 1.0, 0).is_err()); // nsample
        assert!(stochastic_ball_query(&[], 0, &pts, 5, 3, 1.0, 0).is_err()); // nq
    }
}
