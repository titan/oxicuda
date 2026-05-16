//! RKD — Relational Knowledge Distillation (Park et al. 2019).
//!
//! Transfers pairwise distance and triplet angle relationships rather than individual embeddings.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-8;

/// Compute an n×n pairwise Euclidean distance matrix (flat row-major).
#[must_use]
pub fn pairwise_distances(feats: &[Vec<f32>]) -> Vec<f32> {
    let n = feats.len();
    let mut dists = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let d: f32 = feats[i]
                .iter()
                .zip(feats[j].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            dists[i * n + j] = d;
        }
    }
    dists
}

/// Normalise each row of the distance matrix by the mean pairwise distance of that row.
///
/// `d_normalised[i,j] = d[i,j] / (mean_j d[i,j] + ε)`.
#[must_use]
pub fn normalize_distances(dists: &[f32], n: usize) -> Vec<f32> {
    let mut out = dists.to_vec();
    for i in 0..n {
        let row = &dists[i * n..(i + 1) * n];
        let mean = if n > 1 {
            row.iter().sum::<f32>() / (n - 1) as f32
        } else {
            row.iter().sum::<f32>()
        };
        let mean_safe = mean + EPS;
        for j in 0..n {
            out[i * n + j] = dists[i * n + j] / mean_safe;
        }
    }
    out
}

/// Smooth-L1 (Huber) loss with threshold `delta`.
#[inline]
#[must_use]
pub fn smooth_l1(x: f32, delta: f32) -> f32 {
    let ax = x.abs();
    if ax < delta {
        0.5 * ax * ax / delta
    } else {
        ax - 0.5 * delta
    }
}

/// Pairwise distance loss over the upper triangle.
pub fn distance_loss(s_feats: &[Vec<f32>], t_feats: &[Vec<f32>]) -> DistillResult<f32> {
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
    let s_d = normalize_distances(&pairwise_distances(s_feats), n);
    let t_d = normalize_distances(&pairwise_distances(t_feats), n);
    let mut total = 0.0_f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = t_d[i * n + j] - s_d[i * n + j];
            total += smooth_l1(diff, 1.0);
            count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok(total / count as f32)
}

fn cosine_sim_vec(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

fn triplet_angle(feats: &[Vec<f32>], i: usize, j: usize, k: usize) -> f32 {
    let dim = feats[i].len();
    let e_s: Vec<f32> = (0..dim).map(|d| feats[j][d] - feats[i][d]).collect();
    let e_t: Vec<f32> = (0..dim).map(|d| feats[k][d] - feats[i][d]).collect();
    cosine_sim_vec(&e_s, &e_t)
}

/// Triplet angle loss sampled over up to 500 random triplets.
pub fn angle_loss(
    s_feats: &[Vec<f32>],
    t_feats: &[Vec<f32>],
    rng: &mut LcgRng,
) -> DistillResult<f32> {
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
    let max_triplets = 500usize;
    let mut total = 0.0_f32;
    let mut count = 0usize;
    for _ in 0..max_triplets {
        let i = rng.next_usize(n);
        let j = rng.next_usize(n - 1);
        let j = if j >= i { j + 1 } else { j };
        let k_raw = rng.next_usize(n - 2);
        let k = {
            let mut cand = k_raw;
            if cand >= i.min(j) {
                cand += 1;
            }
            if cand >= i.max(j) {
                cand += 1;
            }
            cand
        };
        if k >= n {
            continue;
        }
        let ang_s = triplet_angle(s_feats, i, j, k);
        let ang_t = triplet_angle(t_feats, i, j, k);
        total += smooth_l1(ang_t - ang_s, 1.0);
        count += 1;
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok(total / count as f32)
}

/// Combined RKD loss: `lambda_d · distance_loss + lambda_a · angle_loss`.
pub fn rkd_loss(
    s_feats: &[Vec<f32>],
    t_feats: &[Vec<f32>],
    lambda_d: f32,
    lambda_a: f32,
    rng: &mut LcgRng,
) -> DistillResult<f32> {
    let dl = distance_loss(s_feats, t_feats)?;
    let al = angle_loss(s_feats, t_feats, rng)?;
    Ok(lambda_d * dl + lambda_a * al)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_loss_nonneg() {
        let s: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let t: Vec<Vec<f32>> = (0..4)
            .map(|i| vec![i as f32 * 0.9, (i + 1) as f32])
            .collect();
        let loss = distance_loss(&s, &t).unwrap();
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn smooth_l1_at_zero() {
        assert!((smooth_l1(0.0, 1.0)).abs() < 1e-10);
    }
}
