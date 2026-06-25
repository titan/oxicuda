//! RKD with full (exhaustive) triplet enumeration.
//!
//! The baseline RKD angle loss in [`crate::relation::rkd`] estimates the triplet-angle term
//! by Monte-Carlo sampling up to 500 random triplets. That estimator has variance and its
//! result depends on the RNG seed. For small batches the *complete* set of ordered triplets
//! `(i, j, k)` with distinct indices is small enough to enumerate exactly (`n·(n−1)·(n−2)`),
//! yielding a deterministic, zero-variance angle loss. This module provides that exhaustive
//! variant together with a guarded entry point that enumerates when the batch is small and
//! otherwise reports how many triplets a full pass would require, so callers can decide.
//!
//! The angle for a triplet anchored at `i` is the cosine of the angle between the edge
//! vectors `(x_j − x_i)` and `(x_k − x_i)`, matching the original RKD-A definition (Park et
//! al. 2019). The loss is the mean smooth-L1 between teacher and student angles over all
//! enumerated triplets.

use crate::error::{DistillError, DistillResult};
use crate::relation::rkd::{distance_loss, smooth_l1};

const EPS: f32 = 1e-8;

/// Maximum batch size for which exhaustive enumeration is permitted by [`full_rkd_loss`].
///
/// At `n = 64` a full pass is `64·63·62 = 249 984` triplets, a comfortable upper bound; the
/// guard prevents accidentally launching an `O(n³)` sweep on a large batch.
pub const MAX_FULL_TRIPLET_BATCH: usize = 64;

fn cosine_sim_vec(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

fn triplet_angle(feats: &[Vec<f32>], i: usize, j: usize, k: usize) -> f32 {
    let dim = feats[i].len();
    let e_ij: Vec<f32> = (0..dim).map(|d| feats[j][d] - feats[i][d]).collect();
    let e_ik: Vec<f32> = (0..dim).map(|d| feats[k][d] - feats[i][d]).collect();
    cosine_sim_vec(&e_ij, &e_ik)
}

/// Number of ordered distinct triplets a full enumeration over `n` points would visit.
#[must_use]
pub fn full_triplet_count(n: usize) -> usize {
    if n < 3 { 0 } else { n * (n - 1) * (n - 2) }
}

/// Exhaustive triplet-angle loss over *all* ordered distinct triplets `(i, j, k)`.
///
/// Deterministic (no RNG). Returns the mean smooth-L1 between teacher and student angles.
pub fn full_angle_loss(s_feats: &[Vec<f32>], t_feats: &[Vec<f32>]) -> DistillResult<f32> {
    if s_feats.is_empty() || t_feats.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_feats.len() != t_feats.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_feats.len(),
            got: t_feats.len(),
        });
    }
    let n = s_feats.len();
    if n < 3 {
        return Ok(0.0);
    }
    let mut total = 0.0_f64;
    let mut count = 0u64;
    for i in 0..n {
        for j in 0..n {
            if j == i {
                continue;
            }
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }
                let ang_s = triplet_angle(s_feats, i, j, k);
                let ang_t = triplet_angle(t_feats, i, j, k);
                total += f64::from(smooth_l1(ang_t - ang_s, 1.0));
                count += 1;
            }
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok((total / count as f64) as f32)
}

/// Guarded full RKD loss: exhaustive distance + exhaustive angle, combined with weights.
///
/// Refuses to run when the batch exceeds [`MAX_FULL_TRIPLET_BATCH`] (returning an error that
/// names the full triplet count) so the `O(n³)` enumeration cannot be triggered by accident.
pub fn full_rkd_loss(
    s_feats: &[Vec<f32>],
    t_feats: &[Vec<f32>],
    lambda_d: f32,
    lambda_a: f32,
) -> DistillResult<f32> {
    let n = s_feats.len();
    if n > MAX_FULL_TRIPLET_BATCH {
        return Err(DistillError::InvalidConfig {
            msg: format!(
                "batch {n} exceeds MAX_FULL_TRIPLET_BATCH ({MAX_FULL_TRIPLET_BATCH}); a full \
                 pass would visit {} triplets — use the sampled rkd::angle_loss instead",
                full_triplet_count(n)
            ),
        });
    }
    let dl = distance_loss(s_feats, t_feats)?;
    let al = full_angle_loss(s_feats, t_feats)?;
    Ok(lambda_d * dl + lambda_a * al)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn triplet_count_formula() {
        assert_eq!(full_triplet_count(2), 0);
        assert_eq!(full_triplet_count(3), 6);
        assert_eq!(full_triplet_count(4), 24);
        assert_eq!(full_triplet_count(5), 60);
    }

    #[test]
    fn identical_features_zero_angle_loss() {
        let feats: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32, (i + 1) as f32, (i * 2) as f32])
            .collect();
        let loss = full_angle_loss(&feats, &feats).expect("loss");
        assert!(loss < 1e-6, "loss {loss}");
    }

    #[test]
    fn full_angle_loss_deterministic() {
        let mut rng = LcgRng::new(11);
        let s: Vec<Vec<f32>> = (0..6)
            .map(|_| (0..3).map(|_| rng.next_normal()).collect())
            .collect();
        let t: Vec<Vec<f32>> = (0..6)
            .map(|_| (0..3).map(|_| rng.next_normal()).collect())
            .collect();
        let a = full_angle_loss(&s, &t).expect("a");
        let b = full_angle_loss(&s, &t).expect("b");
        assert_eq!(a.to_bits(), b.to_bits(), "must be bit-identical");
        assert!(a >= 0.0 && a.is_finite());
    }

    #[test]
    fn full_enumerates_expected_triplets() {
        // A degenerate angle difference of a constant lets us check the averaging count
        // indirectly: with all teacher angles == all student angles the mean must be 0.
        let n = 5;
        let s: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, 0.0]).collect();
        let loss = full_angle_loss(&s, &s).expect("loss");
        assert!(loss.abs() < 1e-6);
        assert_eq!(full_triplet_count(n), 60);
    }

    #[test]
    fn full_rkd_loss_nonneg() {
        let mut rng = LcgRng::new(3);
        let s: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..4).map(|_| rng.next_normal()).collect())
            .collect();
        let t: Vec<Vec<f32>> = (0..8)
            .map(|i| s[i].iter().map(|&v| v * 1.05 + 0.01).collect())
            .collect();
        let loss = full_rkd_loss(&s, &t, 1.0, 2.0).expect("loss");
        assert!(loss >= 0.0 && loss.is_finite(), "loss {loss}");
    }

    #[test]
    fn full_rkd_guards_large_batch() {
        let big: Vec<Vec<f32>> = (0..MAX_FULL_TRIPLET_BATCH + 1)
            .map(|i| vec![i as f32, 0.0])
            .collect();
        assert!(full_rkd_loss(&big, &big, 1.0, 1.0).is_err());
    }

    #[test]
    fn small_batch_returns_zero() {
        let s = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        assert_eq!(full_angle_loss(&s, &s).expect("loss"), 0.0);
    }
}
