//! DAFL — Data-Free Learning (Chen et al. 2019) — generator-based data synthesis.

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-10;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Two-layer MLP generator: latent → hidden (ReLU) → output.
#[derive(Debug, Clone)]
pub struct DaflGenerator {
    /// Latent noise dimension.
    pub latent_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Output (image/feature) dimension.
    pub out_dim: usize,
    /// First-layer weights `[hidden_dim × latent_dim]`.
    pub w1: Vec<f32>,
    /// First-layer biases `[hidden_dim]`.
    pub b1: Vec<f32>,
    /// Second-layer weights `[out_dim × hidden_dim]`.
    pub w2: Vec<f32>,
    /// Second-layer biases `[out_dim]`.
    pub b2: Vec<f32>,
}

impl DaflGenerator {
    /// Create a new generator with He initialisation.
    #[must_use]
    pub fn new(latent_dim: usize, hidden_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let scale1 = if latent_dim == 0 {
            1.0
        } else {
            (2.0_f32 / latent_dim as f32).sqrt()
        };
        let scale2 = if hidden_dim == 0 {
            1.0
        } else {
            (2.0_f32 / hidden_dim as f32).sqrt()
        };
        let mut w1 = vec![0.0_f32; hidden_dim * latent_dim];
        for w in w1.iter_mut() {
            *w = rng.next_normal() * scale1;
        }
        let b1 = vec![0.0_f32; hidden_dim];
        let mut w2 = vec![0.0_f32; out_dim * hidden_dim];
        for w in w2.iter_mut() {
            *w = rng.next_normal() * scale2;
        }
        let b2 = vec![0.0_f32; out_dim];
        Self {
            latent_dim,
            hidden_dim,
            out_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Generate a synthetic sample: z → (w1,b1) → ReLU → (w2,b2).
    pub fn generate(&self, z: &[f32]) -> DistillResult<Vec<f32>> {
        if z.len() != self.latent_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.latent_dim,
                got: z.len(),
            });
        }
        // Hidden layer.
        let h: Vec<f32> = (0..self.hidden_dim)
            .map(|j| {
                let row = &self.w1[j * self.latent_dim..(j + 1) * self.latent_dim];
                let dot: f32 = z.iter().zip(row.iter()).map(|(&zi, &wi)| zi * wi).sum();
                relu(dot + self.b1[j])
            })
            .collect();
        // Output layer.
        let out: Vec<f32> = (0..self.out_dim)
            .map(|k| {
                let row = &self.w2[k * self.hidden_dim..(k + 1) * self.hidden_dim];
                let dot: f32 = h.iter().zip(row.iter()).map(|(&hi, &wi)| hi * wi).sum();
                dot + self.b2[k]
            })
            .collect();
        Ok(out)
    }
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-30);
    exps.iter().map(|&e| e / sum).collect()
}

/// One-activation loss: encourage teacher to be confident on `target_class`.
///
/// `= −log(softmax(teacher)[target_class] + ε)`
#[must_use]
pub fn dafl_teacher_loss(teacher_logits: &[f32], target_class: usize) -> f32 {
    let p = stable_softmax(teacher_logits);
    let p_c = if target_class < p.len() {
        p[target_class]
    } else {
        EPS
    };
    -(p_c + EPS).ln()
}

/// Information entropy loss: maximise entropy for sample diversity.
///
/// Returns `−entropy` (minimise to maximise entropy).
#[must_use]
pub fn dafl_info_entropy_loss(teacher_logits: &[f32]) -> f32 {
    let p = stable_softmax(teacher_logits);
    let entropy: f32 = p.iter().map(|&pi| -pi * (pi + EPS).ln()).sum();
    -entropy
}

/// Activation loss: L1 to encourage non-trivial intermediate activations.
#[must_use]
pub fn dafl_activation_loss(intermediate_feat: &[f32]) -> f32 {
    if intermediate_feat.is_empty() {
        return 0.0;
    }
    intermediate_feat.iter().map(|&v| v.abs()).sum::<f32>() / intermediate_feat.len() as f32
}

/// Total generator loss combining teacher, entropy, and activation terms.
pub fn dafl_total_generator_loss(
    t_logits: &[f32],
    target_class: usize,
    intermediate: &[f32],
    lambda_ce: f32,
    lambda_ie: f32,
    lambda_act: f32,
) -> DistillResult<f32> {
    if t_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let ce = dafl_teacher_loss(t_logits, target_class);
    let ie = dafl_info_entropy_loss(t_logits);
    let act = dafl_activation_loss(intermediate);
    Ok(lambda_ce * ce + lambda_ie * ie + lambda_act * act)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_output_shape() {
        let mut rng = LcgRng::new(3);
        let generator = DaflGenerator::new(10, 32, 64, &mut rng);
        let z: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let out = generator.generate(&z).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn dafl_teacher_loss_nonneg() {
        let logits = vec![1.0_f32, 3.0, 2.0];
        let l = dafl_teacher_loss(&logits, 1);
        assert!(l >= 0.0 && l.is_finite());
    }

    #[test]
    fn dafl_total_finite() {
        let t = vec![1.0_f32, 2.0, 3.0];
        let feat = vec![0.5_f32, 0.2, 0.8];
        let l = dafl_total_generator_loss(&t, 2, &feat, 1.0, 1.0, 1.0).unwrap();
        assert!(l.is_finite());
    }
}
