//! ZSKD — Zero-Shot Knowledge Distillation (Nayak et al. 2019) — class impression synthesis.

use crate::error::DistillResult;
use crate::handle::LcgRng;
use crate::online::dml::kl_divergence;

const EPS: f32 = 1e-10;

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-30);
    exps.iter().map(|&e| e / sum).collect()
}

/// Approximate Dirichlet sample using exponential approximation.
///
/// For each dimension i: sample x_i ~ Exp(1) scaled by alpha_i (via −ln(U) * alpha_i),
/// then normalise to obtain a probability vector p_i = x_i / Σ x_j.
#[must_use]
pub fn dirichlet_sample(alpha: &[f32], rng: &mut LcgRng) -> Vec<f32> {
    // Draw exponential samples scaled by concentration alpha_i.
    let samples: Vec<f32> = alpha
        .iter()
        .map(|&ai| {
            let u = rng.next_f32().max(1e-30);
            (-u.ln()) * ai.max(1e-8)
        })
        .collect();
    let sum: f32 = samples.iter().sum::<f32>().max(EPS);
    samples.iter().map(|&s| s / sum).collect()
}

/// Cross-entropy loss using soft target distribution: −Σ target`[i]` · log(softmax`[i]` + ε).
///
/// Used to measure how closely the teacher's softmax matches a target distribution.
#[must_use]
pub fn class_impression_loss(teacher_logits: &[f32], target_probs: &[f32]) -> f32 {
    let p_t = stable_softmax(teacher_logits);
    target_probs
        .iter()
        .zip(p_t.iter())
        .map(|(&qi, &pi)| -qi * (pi + EPS).ln())
        .sum()
}

/// Synthesise a class impression distribution for `target_class`.
///
/// Builds Dirichlet parameters: α`[i]` = 0.1 for i ≠ target, α`[target]` = 10.0.
/// Returns the sampled soft target probability vector.
#[must_use]
pub fn synthesize_impression(
    num_classes: usize,
    target_class: usize,
    rng: &mut LcgRng,
) -> Vec<f32> {
    let alpha: Vec<f32> = (0..num_classes)
        .map(|i| if i == target_class { 10.0 } else { 0.1 })
        .collect();
    dirichlet_sample(&alpha, rng)
}

/// Student distillation loss against synthesised teacher soft labels.
///
/// `KL(teacher_soft_T ‖ student_soft_T) · temp²`
pub fn zskd_student_loss(
    student_logits: &[f32],
    synthesized_teacher_soft: &[f32],
    temp: f32,
) -> DistillResult<f32> {
    let t_safe = temp.max(1e-12);
    let p_s = stable_softmax(
        &student_logits
            .iter()
            .map(|&x| x / t_safe)
            .collect::<Vec<_>>(),
    );
    let kl = kl_divergence(synthesized_teacher_soft, &p_s);
    Ok(temp * temp * kl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_sample_sums_to_one() {
        let mut rng = LcgRng::new(42);
        let alpha = vec![1.0_f32, 2.0, 3.0];
        let p = dirichlet_sample(&alpha, &mut rng);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn synthesize_impression_target_dominant() {
        let mut rng = LcgRng::new(7);
        let imp = synthesize_impression(5, 2, &mut rng);
        assert_eq!(imp.len(), 5);
        // With α[2]=10 vs α[others]=0.1, target class should dominate.
        let max_idx = imp
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn zskd_student_loss_nonneg() {
        let mut rng = LcgRng::new(99);
        let teacher_soft = synthesize_impression(4, 1, &mut rng);
        let s_logits = vec![1.0_f32, 3.0, 1.0, 1.0];
        let loss = zskd_student_loss(&s_logits, &teacher_soft, 4.0).unwrap();
        assert!(loss >= 0.0 && loss.is_finite());
    }
}
