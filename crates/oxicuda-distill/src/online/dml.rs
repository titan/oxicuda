//! DML — Deep Mutual Learning (Zhang et al. 2018) — N peer networks teaching each other.

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-10;

/// Numerically-stable softmax.
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-30);
    exps.iter().map(|&e| e / sum).collect()
}

/// KL divergence KL(p ‖ q).
#[must_use]
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi <= 0.0 {
                0.0
            } else {
                pi * (pi / (qi + EPS)).ln()
            }
        })
        .sum()
}

/// Cross-entropy −log(softmax(logits)`[label]` + ε).
#[must_use]
pub fn cross_entropy_from_probs(logits: &[f32], label: usize) -> f32 {
    let p = softmax(logits);
    let p_label = if label < p.len() { p[label] } else { EPS };
    -(p_label + EPS).ln()
}

/// DML loss for a single peer.
///
/// `ce_loss + mean_peers KL(p_self ‖ softmax(peer))`.
pub fn dml_peer_loss(
    self_logits: &[f32],
    peer_logits_list: &[&[f32]],
    label: usize,
) -> DistillResult<f32> {
    if self_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if peer_logits_list.is_empty() {
        return Err(DistillError::InvalidConfig {
            msg: "peer_logits_list must be non-empty".into(),
        });
    }
    let ce_loss = cross_entropy_from_probs(self_logits, label);
    let p_self = softmax(self_logits);
    let peer_kl: f32 = peer_logits_list
        .iter()
        .map(|&peer| kl_divergence(&p_self, &softmax(peer)))
        .sum::<f32>()
        / peer_logits_list.len() as f32;
    Ok(ce_loss + peer_kl)
}

/// Compute per-peer DML losses where each peer treats all others as its peer group.
pub fn dml_all_losses(peer_logits: &[Vec<f32>], labels: &[usize]) -> DistillResult<Vec<f32>> {
    if peer_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if peer_logits.len() != labels.len() {
        return Err(DistillError::DimensionMismatch {
            expected: peer_logits.len(),
            got: labels.len(),
        });
    }
    let n = peer_logits.len();
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let peers: Vec<&[f32]> = (0..n)
            .filter(|&j| j != i)
            .map(|j| peer_logits[j].as_slice())
            .collect();
        losses.push(dml_peer_loss(&peer_logits[i], &peers, labels[i])?);
    }
    Ok(losses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0_f32, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dml_all_losses_correct_count() {
        let logits: Vec<Vec<f32>> = (0..3)
            .map(|i| vec![i as f32, 2.0 - i as f32, 1.0])
            .collect();
        let labels = vec![0_usize, 1, 0];
        let losses = dml_all_losses(&logits, &labels).expect("dml_all_losses should succeed");
        assert_eq!(losses.len(), 3);
        for l in &losses {
            assert!(l.is_finite());
        }
    }
}
