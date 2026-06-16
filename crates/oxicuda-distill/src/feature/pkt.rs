//! PKT — Probabilistic Knowledge Transfer (Passalis & Tefas 2018).
//!
//! Transfers knowledge by matching kernel-induced affinity matrices between student and teacher.

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-8;

/// Cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

fn softmax_row(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let sum_safe = sum.max(1e-30);
    for v in row.iter_mut() {
        *v /= sum_safe;
    }
}

/// Build an n×n affinity matrix using cosine similarity, then apply row-wise softmax.
///
/// Returns a flat `n*n` row-major matrix.
#[must_use]
pub fn build_affinity_matrix(feats: &[Vec<f32>]) -> Vec<f32> {
    let n = feats.len();
    let mut mat = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            mat[i * n + j] = cosine_similarity(&feats[i], &feats[j]);
        }
        softmax_row(&mut mat[i * n..(i + 1) * n]);
    }
    mat
}

/// PKT loss: KL divergence KL(K_t ‖ K_s) between teacher and student affinity matrices.
pub fn pkt_loss(s_feats: &[Vec<f32>], t_feats: &[Vec<f32>]) -> DistillResult<f32> {
    if s_feats.is_empty() || t_feats.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if s_feats.len() != t_feats.len() {
        return Err(DistillError::DimensionMismatch {
            expected: s_feats.len(),
            got: t_feats.len(),
        });
    }
    let k_s = build_affinity_matrix(s_feats);
    let k_t = build_affinity_matrix(t_feats);
    let loss: f32 = k_t
        .iter()
        .zip(k_s.iter())
        .map(|(&kt, &ks)| {
            if kt <= 0.0 {
                0.0
            } else {
                kt * (kt / (ks + EPS)).ln()
            }
        })
        .sum();
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_self_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn affinity_matrix_shape() {
        let feats: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let m = build_affinity_matrix(&feats);
        assert_eq!(m.len(), 16);
    }

    #[test]
    fn pkt_loss_finite() {
        let s: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let t: Vec<Vec<f32>> = (0..3)
            .map(|i| vec![i as f32 * 0.9, (i + 1) as f32 * 1.1])
            .collect();
        let loss = pkt_loss(&s, &t).expect("pkt_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }
}
