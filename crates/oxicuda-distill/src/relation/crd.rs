//! CRD — Contrastive Representation Distillation (Tian et al. 2020).

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-8;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

/// EMA-updated teacher feature memory bank.
#[derive(Debug, Clone)]
pub struct CrdMemoryBank {
    /// Stored teacher feature vectors, shape `[n_samples × feat_dim]`.
    pub feats: Vec<Vec<f32>>,
    /// EMA momentum ∈ [0, 1].
    pub momentum: f32,
}

impl CrdMemoryBank {
    /// Initialise with unit random vectors (each entry sampled from an LCG and normalised).
    #[must_use]
    pub fn new(n_samples: usize, feat_dim: usize, momentum: f32, rng: &mut LcgRng) -> Self {
        let feats: Vec<Vec<f32>> = (0..n_samples)
            .map(|_| {
                let mut v: Vec<f32> = (0..feat_dim).map(|_| rng.next_normal()).collect();
                let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(EPS);
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect();
        Self { feats, momentum }
    }

    /// EMA update: `bank[idx] ← m · bank[idx] + (1 − m) · new_feat`.
    pub fn update(&mut self, idx: usize, new_feat: &[f32]) -> DistillResult<()> {
        if idx >= self.feats.len() {
            return Err(DistillError::DimensionMismatch {
                expected: self.feats.len(),
                got: idx + 1,
            });
        }
        let m = self.momentum;
        let stored = &mut self.feats[idx];
        if stored.len() != new_feat.len() {
            return Err(DistillError::DimensionMismatch {
                expected: stored.len(),
                got: new_feat.len(),
            });
        }
        for (s, &n) in stored.iter_mut().zip(new_feat.iter()) {
            *s = m * *s + (1.0 - m) * n;
        }
        Ok(())
    }

    /// Retrieve a feature vector by index.
    pub fn get(&self, idx: usize) -> DistillResult<&[f32]> {
        if idx >= self.feats.len() {
            return Err(DistillError::DimensionMismatch {
                expected: self.feats.len(),
                got: idx + 1,
            });
        }
        Ok(&self.feats[idx])
    }
}

/// InfoNCE-based CRD loss.
///
/// `anchor_s` — student anchor; `pos_idx` — positive sample index in the bank;
/// `neg_idxs` — negative sample indices; `tau` — temperature.
pub fn crd_loss(
    anchor_s: &[f32],
    bank: &CrdMemoryBank,
    pos_idx: usize,
    neg_idxs: &[usize],
    tau: f32,
) -> DistillResult<f32> {
    if anchor_s.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if neg_idxs.is_empty() {
        return Err(DistillError::InvalidConfig {
            msg: "neg_idxs must be non-empty".into(),
        });
    }
    let k_pos = bank.get(pos_idx)?;
    let tau_safe = tau.max(EPS);
    let pos_sim = cosine_sim(anchor_s, k_pos) / tau_safe;
    let pos_exp = pos_sim.exp();
    let neg_sum: f32 = neg_idxs
        .iter()
        .map(|&ni| {
            let k_neg = bank.feats.get(ni).map(|v| v.as_slice()).unwrap_or(&[]);
            let s = cosine_sim(anchor_s, k_neg) / tau_safe;
            s.exp()
        })
        .sum();
    let denominator = pos_exp + neg_sum;
    Ok(-(pos_exp / denominator.max(EPS)).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_update_changes_value() {
        let mut rng = LcgRng::new(5);
        let mut bank = CrdMemoryBank::new(10, 4, 0.9, &mut rng);
        let orig = bank.feats[0].clone();
        let new_feat = vec![1.0_f32, 0.0, 0.0, 0.0];
        bank.update(0, &new_feat).unwrap();
        assert_ne!(bank.feats[0], orig);
    }

    #[test]
    fn crd_loss_nonneg() {
        let mut rng = LcgRng::new(77);
        let bank = CrdMemoryBank::new(5, 4, 0.9, &mut rng);
        let anchor = vec![1.0_f32, 0.0, 0.0, 0.0];
        let loss = crd_loss(&anchor, &bank, 0, &[1, 2, 3], 0.07).unwrap();
        assert!(loss >= 0.0 && loss.is_finite());
    }
}
