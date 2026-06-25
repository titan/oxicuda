//! Deep generator for data-free distillation with label-balanced sampling.
//!
//! The baseline [`crate::data_free::dafl`] generator is a 2-layer MLP and draws synthetic
//! samples without controlling their class distribution, which in practice produces
//! mode-collapsed batches dominated by a few easy classes. This module addresses both
//! limitations identified in the literature:
//!
//! * **Deeper generator** — a 3-layer MLP (`latent → h1 → ReLU → h2 → ReLU → output`) with
//!   He initialisation, giving the synthesiser more capacity to match the teacher's input
//!   manifold than the shallow baseline.
//! * **Label-balanced sampling** — the conditioning class for each generated sample is chosen
//!   so that every class appears an equal number of times across a batch (round-robin with a
//!   deterministically-shuffled remainder), rather than sampled freely. The class index is
//!   injected into the latent code via a small one-hot conditioning block, so the generator
//!   can learn class-conditional outputs.
//!
//! Together these reproduce the "label-balanced, deeper-generator" recipe used by stronger
//! data-free distillation methods. All randomness flows through the crate [`LcgRng`].

use crate::error::{DistillError, DistillResult};
use crate::handle::LcgRng;

const EPS: f32 = 1e-10;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-30);
    exps.iter().map(|&e| e / sum).collect()
}

/// 3-layer conditional MLP generator: `[latent ⊕ one-hot(class)] → h1 → h2 → output`.
#[derive(Debug, Clone)]
pub struct DeepGenerator {
    /// Latent noise dimension (before class conditioning is appended).
    pub latent_dim: usize,
    /// Number of conditioning classes (one-hot width appended to the latent).
    pub num_classes: usize,
    /// First hidden dimension.
    pub h1_dim: usize,
    /// Second hidden dimension.
    pub h2_dim: usize,
    /// Output (synthetic sample) dimension.
    pub out_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
}

impl DeepGenerator {
    /// Construct a deep generator with He initialisation. The effective first-layer input
    /// width is `latent_dim + num_classes` (latent code concatenated with the one-hot label).
    pub fn new(
        latent_dim: usize,
        num_classes: usize,
        h1_dim: usize,
        h2_dim: usize,
        out_dim: usize,
        rng: &mut LcgRng,
    ) -> DistillResult<Self> {
        if latent_dim == 0 || num_classes == 0 || h1_dim == 0 || h2_dim == 0 || out_dim == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "all generator dimensions must be non-zero".into(),
            });
        }
        let in_dim = latent_dim + num_classes;
        let s1 = (2.0_f32 / in_dim as f32).sqrt();
        let s2 = (2.0_f32 / h1_dim as f32).sqrt();
        let s3 = (2.0_f32 / h2_dim as f32).sqrt();
        let mut w1 = vec![0.0_f32; h1_dim * in_dim];
        for w in w1.iter_mut() {
            *w = rng.next_normal() * s1;
        }
        let mut w2 = vec![0.0_f32; h2_dim * h1_dim];
        for w in w2.iter_mut() {
            *w = rng.next_normal() * s2;
        }
        let mut w3 = vec![0.0_f32; out_dim * h2_dim];
        for w in w3.iter_mut() {
            *w = rng.next_normal() * s3;
        }
        Ok(Self {
            latent_dim,
            num_classes,
            h1_dim,
            h2_dim,
            out_dim,
            w1,
            b1: vec![0.0_f32; h1_dim],
            w2,
            b2: vec![0.0_f32; h2_dim],
            w3,
            b3: vec![0.0_f32; out_dim],
        })
    }

    /// Generate one class-conditional sample from latent `z` and conditioning `class`.
    pub fn generate(&self, z: &[f32], class: usize) -> DistillResult<Vec<f32>> {
        if z.len() != self.latent_dim {
            return Err(DistillError::DimensionMismatch {
                expected: self.latent_dim,
                got: z.len(),
            });
        }
        if class >= self.num_classes {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "class {class} out of range for {} classes",
                    self.num_classes
                ),
            });
        }
        // Build conditioned input = latent ⊕ one-hot(class).
        let in_dim = self.latent_dim + self.num_classes;
        let mut x = vec![0.0_f32; in_dim];
        x[..self.latent_dim].copy_from_slice(z);
        x[self.latent_dim + class] = 1.0;
        // Layer 1.
        let h1: Vec<f32> = (0..self.h1_dim)
            .map(|j| {
                let row = &self.w1[j * in_dim..(j + 1) * in_dim];
                let dot: f32 = x.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
                relu(dot + self.b1[j])
            })
            .collect();
        // Layer 2.
        let h2: Vec<f32> = (0..self.h2_dim)
            .map(|j| {
                let row = &self.w2[j * self.h1_dim..(j + 1) * self.h1_dim];
                let dot: f32 = h1.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
                relu(dot + self.b2[j])
            })
            .collect();
        // Output layer (linear).
        let out: Vec<f32> = (0..self.out_dim)
            .map(|k| {
                let row = &self.w3[k * self.h2_dim..(k + 1) * self.h2_dim];
                let dot: f32 = h2.iter().zip(row.iter()).map(|(&a, &b)| a * b).sum();
                dot + self.b3[k]
            })
            .collect();
        Ok(out)
    }
}

/// Produce a label-balanced batch of conditioning class indices.
///
/// Each class appears `batch / num_classes` times; the `batch % num_classes` remainder is
/// distributed to a deterministically-shuffled subset of classes, and finally the whole
/// sequence is shuffled (Fisher-Yates with [`LcgRng`]) so positions are not class-ordered.
#[must_use]
pub fn label_balanced_classes(batch: usize, num_classes: usize, rng: &mut LcgRng) -> Vec<usize> {
    if batch == 0 || num_classes == 0 {
        return Vec::new();
    }
    let base = batch / num_classes;
    let remainder = batch % num_classes;
    let mut labels = Vec::with_capacity(batch);
    for c in 0..num_classes {
        for _ in 0..base {
            labels.push(c);
        }
    }
    // Distribute the remainder across a shuffled class order so it is not biased to class 0.
    if remainder > 0 {
        let mut order: Vec<usize> = (0..num_classes).collect();
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }
        for &c in order.iter().take(remainder) {
            labels.push(c);
        }
    }
    // Fisher-Yates shuffle of the full label sequence.
    for i in (1..labels.len()).rev() {
        let j = rng.next_usize(i + 1);
        labels.swap(i, j);
    }
    labels
}

/// Generate a full label-balanced synthetic batch.
///
/// Returns `(samples, labels)` where `samples[b]` is conditioned on `labels[b]`. Each latent
/// code is drawn from the standard normal via the crate RNG.
pub fn generate_balanced_batch(
    generator: &DeepGenerator,
    batch: usize,
    rng: &mut LcgRng,
) -> DistillResult<(Vec<Vec<f32>>, Vec<usize>)> {
    if batch == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "batch must be non-zero".into(),
        });
    }
    let labels = label_balanced_classes(batch, generator.num_classes, rng);
    let mut samples = Vec::with_capacity(batch);
    for &c in &labels {
        let mut z = vec![0.0_f32; generator.latent_dim];
        rng.fill_normal(&mut z);
        samples.push(generator.generate(&z, c)?);
    }
    Ok((samples, labels))
}

/// Class-balance entropy of a label batch, normalised to `[0, 1]`.
///
/// `1.0` means perfectly uniform class usage; lower values indicate imbalance. Useful as a
/// diagnostic confirming [`label_balanced_classes`] is doing its job.
#[must_use]
pub fn class_balance_entropy(labels: &[usize], num_classes: usize) -> f32 {
    if labels.is_empty() || num_classes <= 1 {
        return if num_classes <= 1 { 1.0 } else { 0.0 };
    }
    let mut counts = vec![0usize; num_classes];
    for &l in labels {
        if l < num_classes {
            counts[l] += 1;
        }
    }
    let total = labels.len() as f32;
    let entropy: f32 = counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0.0
            } else {
                let p = c as f32 / total;
                -p * p.ln()
            }
        })
        .sum();
    entropy / (num_classes as f32).ln()
}

/// Teacher one-hot confidence loss for a generated batch (encourages confident, on-class
/// teacher responses) — the deep-generator analogue of the DAFL one-hot loss, now using the
/// *known* conditioning label rather than the arg-max.
///
/// `teacher_logits` is `[batch × num_classes]` flat row-major; `labels` is the conditioning
/// class of each sample. Returns the mean `−log p_teacher[label]`.
pub fn conditional_one_hot_loss(
    teacher_logits: &[f32],
    labels: &[usize],
    num_classes: usize,
) -> DistillResult<f32> {
    if teacher_logits.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    let batch = labels.len();
    if teacher_logits.len() != batch * num_classes {
        return Err(DistillError::DimensionMismatch {
            expected: batch * num_classes,
            got: teacher_logits.len(),
        });
    }
    let mut total = 0.0_f32;
    for (b, &label) in labels.iter().enumerate() {
        if label >= num_classes {
            return Err(DistillError::InvalidConfig {
                msg: format!("label {label} out of range for {num_classes} classes"),
            });
        }
        let p = stable_softmax(&teacher_logits[b * num_classes..(b + 1) * num_classes]);
        total += -(p[label] + EPS).ln();
    }
    Ok(total / batch as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_output_shape() {
        let mut rng = LcgRng::new(1);
        let g = DeepGenerator::new(8, 5, 32, 16, 12, &mut rng).expect("gen");
        let z = vec![0.1_f32; 8];
        let out = g.generate(&z, 2).expect("generate");
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn generator_rejects_bad_inputs() {
        let mut rng = LcgRng::new(2);
        assert!(DeepGenerator::new(0, 5, 32, 16, 12, &mut rng).is_err());
        let g = DeepGenerator::new(8, 5, 32, 16, 12, &mut rng).expect("gen");
        assert!(g.generate(&[0.0_f32; 3], 0).is_err());
        assert!(g.generate(&[0.0_f32; 8], 9).is_err());
    }

    #[test]
    fn different_classes_give_different_outputs() {
        // The one-hot conditioning must actually influence the output.
        let mut rng = LcgRng::new(33);
        let g = DeepGenerator::new(8, 4, 32, 16, 10, &mut rng).expect("gen");
        let z = vec![0.25_f32; 8];
        let a = g.generate(&z, 0).expect("a");
        let b = g.generate(&z, 1).expect("b");
        let diff: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum();
        assert!(diff > 1e-3, "class conditioning had no effect: {diff}");
    }

    #[test]
    fn balanced_classes_exact_division() {
        let mut rng = LcgRng::new(7);
        let labels = label_balanced_classes(12, 4, &mut rng);
        assert_eq!(labels.len(), 12);
        let mut counts = [0usize; 4];
        for &l in &labels {
            counts[l] += 1;
        }
        assert_eq!(counts, [3, 3, 3, 3]);
    }

    #[test]
    fn balanced_classes_remainder_spread() {
        let mut rng = LcgRng::new(13);
        let labels = label_balanced_classes(10, 4, &mut rng); // 2 each + 2 remainder
        assert_eq!(labels.len(), 10);
        let mut counts = [0usize; 4];
        for &l in &labels {
            counts[l] += 1;
        }
        // Each class has at least floor(10/4)=2 and the two extras land on distinct classes.
        for &c in &counts {
            assert!((2..=3).contains(&c), "count {c}");
        }
        assert_eq!(counts.iter().sum::<usize>(), 10);
        assert_eq!(counts.iter().filter(|&&c| c == 3).count(), 2);
    }

    #[test]
    fn balanced_entropy_near_one_for_balanced() {
        let mut rng = LcgRng::new(21);
        let labels = label_balanced_classes(40, 8, &mut rng);
        let e = class_balance_entropy(&labels, 8);
        assert!(e > 0.99, "balanced entropy {e}");
    }

    #[test]
    fn balanced_entropy_low_for_single_class() {
        let labels = vec![0usize; 16];
        let e = class_balance_entropy(&labels, 4);
        assert!(e < 1e-5, "single-class entropy {e}");
    }

    #[test]
    fn balanced_batch_generates_consistently() {
        let mut rng = LcgRng::new(99);
        let g = DeepGenerator::new(6, 4, 24, 12, 9, &mut rng).expect("gen");
        let (samples, labels) = generate_balanced_batch(&g, 8, &mut rng).expect("batch");
        assert_eq!(samples.len(), 8);
        assert_eq!(labels.len(), 8);
        for s in &samples {
            assert_eq!(s.len(), 9);
            assert!(s.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn conditional_one_hot_loss_lower_when_confident() {
        // A teacher confident on the conditioning class should yield a lower loss than a
        // teacher that is confident on the wrong class.
        let labels = vec![0usize, 1];
        let confident = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0]; // ex0->0, ex1->1
        let wrong = vec![0.0, 0.0, 5.0, 5.0, 0.0, 0.0]; // mismatched
        let good = conditional_one_hot_loss(&confident, &labels, 3).expect("good");
        let bad = conditional_one_hot_loss(&wrong, &labels, 3).expect("bad");
        assert!(good < bad, "good {good} should be < bad {bad}");
        assert!(good >= 0.0);
    }

    #[test]
    fn conditional_loss_dim_mismatch_errors() {
        let labels = vec![0usize, 1];
        let logits = vec![1.0_f32; 5]; // not 2*3
        assert!(conditional_one_hot_loss(&logits, &labels, 3).is_err());
    }
}
