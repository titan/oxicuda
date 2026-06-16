//! BYOT — Be-Your-Own-Teacher (Zhang et al. 2019) — multi-branch self-distillation.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;
use crate::online::dml::{cross_entropy_from_probs, kl_divergence, softmax};

/// A linear classification head attached to an intermediate layer.
#[derive(Debug, Clone)]
pub struct BranchClassifier {
    /// Depth index (0 = shallowest branch).
    pub depth_idx: usize,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Weight matrix `[num_classes × in_dim]`, row-major.
    pub w: Vec<f32>,
    /// Bias vector `[num_classes]`.
    pub b: Vec<f32>,
}

impl BranchClassifier {
    /// Construct a new branch classifier with Xavier-style initialisation.
    #[must_use]
    pub fn new(depth_idx: usize, in_dim: usize, num_classes: usize, rng: &mut LcgRng) -> Self {
        let scale = if in_dim == 0 {
            1.0
        } else {
            1.0 / (in_dim as f32).sqrt()
        };
        let mut w = vec![0.0_f32; num_classes * in_dim];
        for wi in w.iter_mut() {
            *wi = rng.next_normal() * scale;
        }
        let b = vec![0.0_f32; num_classes];
        Self {
            depth_idx,
            in_dim,
            num_classes,
            w,
            b,
        }
    }

    /// Forward pass: `feat @ Wᵀ + b`.
    pub fn forward(&self, feat: &[f32]) -> DistillResult<Vec<f32>> {
        if feat.len() != self.in_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.in_dim,
                got: feat.len(),
            });
        }
        let out: Vec<f32> = (0..self.num_classes)
            .map(|c| {
                let row = &self.w[c * self.in_dim..(c + 1) * self.in_dim];
                feat.iter()
                    .zip(row.iter())
                    .map(|(&fi, &wi)| fi * wi)
                    .sum::<f32>()
                    + self.b[c]
            })
            .collect();
        Ok(out)
    }
}

/// BYOT loss: non-teacher branches distil from the deepest (last) branch.
///
/// For each non-teacher branch i:
/// `loss_i = CE(branch[i], label) + temp² · KL(softmax(teacher/T) ‖ softmax(branch[i]/T))`
///
/// Returns the mean over all non-teacher branches.
pub fn byot_loss(branch_logits: &[Vec<f32>], label: usize, temp: f32) -> DistillResult<f32> {
    if branch_logits.len() < 2 {
        return Err(DistillError::InvalidConfig {
            msg: "byot_loss requires at least 2 branches (branches + teacher)".into(),
        });
    }
    let teacher = branch_logits.last().expect("branch_logits is non-empty");
    let t_safe = temp.max(1e-12);
    let p_teacher = softmax(&teacher.iter().map(|&x| x / t_safe).collect::<Vec<_>>());
    let non_teacher = &branch_logits[..branch_logits.len() - 1];
    let mut total = 0.0_f32;
    for branch in non_teacher {
        let hard = cross_entropy_from_probs(branch, label);
        let p_branch = softmax(&branch.iter().map(|&x| x / t_safe).collect::<Vec<_>>());
        let soft = temp * temp * kl_divergence(&p_teacher, &p_branch);
        total += hard + soft;
    }
    Ok(total / non_teacher.len() as f32)
}

/// Ensemble logits by averaging across all branches.
#[must_use]
pub fn byot_ensemble(branch_logits: &[Vec<f32>]) -> Vec<f32> {
    if branch_logits.is_empty() {
        return vec![];
    }
    let n = branch_logits.len() as f32;
    let d = branch_logits[0].len();
    let mut out = vec![0.0_f32; d];
    for branch in branch_logits {
        for (o, &v) in out.iter_mut().zip(branch.iter()) {
            *o += v;
        }
    }
    for o in out.iter_mut() {
        *o /= n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byot_loss_finite() {
        let branches = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![1.5_f32, 1.5, 3.0],
            vec![0.5_f32, 2.5, 3.5], // teacher (deepest)
        ];
        let loss = byot_loss(&branches, 2, 4.0).expect("byot_loss should succeed");
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn byot_ensemble_shape() {
        let branches = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let ens = byot_ensemble(&branches);
        assert_eq!(ens.len(), 2);
        assert!((ens[0] - 2.0).abs() < 1e-5);
    }
}
