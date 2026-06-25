//! Multi-generation Born-Again Networks with an empirical convergence study.
//!
//! The original BAN paper (Furlanello et al. 2018) trains a *sequence* of identically-sized
//! students, each distilled from the previous generation rather than from a fixed teacher,
//! and reports results for generations 1 through ~6. This module orchestrates that
//! iterative loop and instruments it so the convergence behaviour across generations can be
//! studied quantitatively.
//!
//! Two pieces are provided:
//!
//! * [`BanMultiGen`] — a scheduler that accumulates per-generation logit snapshots (produced
//!   by the caller's trainer), exposes the BAN distillation target for the next generation,
//!   builds the Born-Again ENSEMBLE (mean of all generations, which the paper shows beats any
//!   single generation), and computes a [`GenerationMetric`] per step measuring how much the
//!   model changed from the previous generation.
//! * [`simulate_ban_trajectory`] — a fully deterministic reference trajectory used to study
//!   convergence without a real trainer: starting from teacher logits, each generation is a
//!   temperature-softened contraction towards the previous generation's distribution. The
//!   inter-generation KL provably decreases monotonically, modelling the empirical
//!   observation that successive born-again generations change less and less.
//!
//! All distillation losses reuse the canonical Hinton objective from [`crate::logit`].

use crate::error::{DistillError, DistillResult};
use crate::logit::hinton_kd::{kl_divergence, softmax_with_temp};

const EPS: f32 = 1e-12;

/// Per-generation convergence diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationMetric {
    /// Generation index (1-based; generation 1 is the first born-again student).
    pub generation: usize,
    /// Symmetric KL between this generation's and the previous generation's soft predictions.
    pub inter_gen_kl: f32,
    /// Fraction of examples whose arg-max prediction is unchanged from the previous generation.
    pub argmax_agreement: f32,
}

/// Multi-generation born-again scheduler operating on logit snapshots.
///
/// Each entry of `generations` is a flat `[n_examples × n_classes]` logit batch for one
/// generation. Generation 0 is the original teacher.
#[derive(Debug, Clone)]
pub struct BanMultiGen {
    /// Logit snapshots, one per generation; `generations[0]` is the teacher.
    pub generations: Vec<Vec<f32>>,
    /// Number of reference examples per snapshot.
    pub n_examples: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// Distillation temperature `T > 0`.
    pub temperature: f32,
}

impl BanMultiGen {
    /// Initialise with the teacher's logit batch.
    pub fn new(
        teacher_logits: Vec<f32>,
        n_examples: usize,
        n_classes: usize,
        temperature: f32,
    ) -> DistillResult<Self> {
        if n_examples == 0 || n_classes == 0 {
            return Err(DistillError::InvalidConfig {
                msg: "n_examples and n_classes must be non-zero".into(),
            });
        }
        if teacher_logits.len() != n_examples * n_classes {
            return Err(DistillError::DimensionMismatch {
                expected: n_examples * n_classes,
                got: teacher_logits.len(),
            });
        }
        if temperature <= 0.0 || !temperature.is_finite() {
            return Err(DistillError::InvalidConfig {
                msg: format!("temperature must be finite and > 0, got {temperature}"),
            });
        }
        Ok(Self {
            generations: vec![teacher_logits],
            n_examples,
            n_classes,
            temperature,
        })
    }

    /// Highest generation index currently stored (0 = only the teacher).
    #[must_use]
    pub fn current_generation(&self) -> usize {
        self.generations.len() - 1
    }

    /// Logits used as the distillation target when training the next generation
    /// (the most recently appended generation).
    #[must_use]
    pub fn current_target(&self) -> &[f32] {
        // `generations` is always non-empty by construction.
        match self.generations.last() {
            Some(g) => g,
            None => &[],
        }
    }

    /// Append a freshly-trained generation's logits and return its convergence metric.
    pub fn push_generation(&mut self, logits: Vec<f32>) -> DistillResult<GenerationMetric> {
        let expected = self.n_examples * self.n_classes;
        if logits.len() != expected {
            return Err(DistillError::DimensionMismatch {
                expected,
                got: logits.len(),
            });
        }
        let prev = self.current_target().to_vec();
        let gen_idx = self.generations.len(); // 1-based for the new generation
        let metric = self.compute_metric(&prev, &logits, gen_idx);
        self.generations.push(logits);
        Ok(metric)
    }

    fn compute_metric(&self, prev: &[f32], cur: &[f32], gen_idx: usize) -> GenerationMetric {
        let c = self.n_classes;
        let mut kl_sum = 0.0_f32;
        let mut agree = 0usize;
        for e in 0..self.n_examples {
            let p_prev = softmax_with_temp(&prev[e * c..(e + 1) * c], self.temperature);
            let p_cur = softmax_with_temp(&cur[e * c..(e + 1) * c], self.temperature);
            // Symmetric KL is a stable, sign-free divergence between consecutive generations.
            kl_sum += 0.5 * (kl_divergence(&p_prev, &p_cur) + kl_divergence(&p_cur, &p_prev));
            if argmax(&prev[e * c..(e + 1) * c]) == argmax(&cur[e * c..(e + 1) * c]) {
                agree += 1;
            }
        }
        GenerationMetric {
            generation: gen_idx,
            inter_gen_kl: kl_sum / self.n_examples as f32,
            argmax_agreement: agree as f32 / self.n_examples as f32,
        }
    }

    /// Born-Again ensemble: per-example mean logits across all stored generations.
    ///
    /// Returns a flat `[n_examples × n_classes]` batch.
    #[must_use]
    pub fn ensemble_logits(&self) -> Vec<f32> {
        let len = self.n_examples * self.n_classes;
        let mut out = vec![0.0_f32; len];
        for g in &self.generations {
            for (o, &v) in out.iter_mut().zip(g.iter()) {
                *o += v;
            }
        }
        let n = self.generations.len() as f32;
        for o in out.iter_mut() {
            *o /= n;
        }
        out
    }

    /// Ensemble over only the last `k` generations (the paper's "BAN-k" variant), which often
    /// excludes the under-trained teacher. Clamped to the number available.
    #[must_use]
    pub fn ensemble_last_k(&self, k: usize) -> Vec<f32> {
        let len = self.n_examples * self.n_classes;
        let total = self.generations.len();
        let take = k.clamp(1, total);
        let start = total - take;
        let mut out = vec![0.0_f32; len];
        for g in &self.generations[start..] {
            for (o, &v) in out.iter_mut().zip(g.iter()) {
                *o += v;
            }
        }
        for o in out.iter_mut() {
            *o /= take as f32;
        }
        out
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Deterministic reference trajectory for an empirical convergence study.
///
/// Generates `n_generations` born-again generations from a teacher logit batch. Each
/// generation `g` is produced from generation `g − 1` by a temperature-softened contraction:
/// the new logits are `contraction · log(p_{g-1})` (re-centred), where `p_{g-1}` is the
/// softened distribution of the previous generation. With `0 < contraction < 1` the
/// distributions move monotonically towards uniform, so the inter-generation KL strictly
/// decreases — a controlled stand-in for the empirically observed diminishing returns of
/// successive born-again generations.
///
/// Returns the per-generation [`GenerationMetric`] sequence (length `n_generations`).
pub fn simulate_ban_trajectory(
    teacher_logits: &[f32],
    n_examples: usize,
    n_classes: usize,
    n_generations: usize,
    temperature: f32,
    contraction: f32,
) -> DistillResult<Vec<GenerationMetric>> {
    if n_generations == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "n_generations must be >= 1".into(),
        });
    }
    if !(0.0..1.0).contains(&contraction) {
        return Err(DistillError::InvalidConfig {
            msg: format!("contraction must be in [0, 1), got {contraction}"),
        });
    }
    let mut mgr = BanMultiGen::new(teacher_logits.to_vec(), n_examples, n_classes, temperature)?;
    let c = n_classes;
    let mut metrics = Vec::with_capacity(n_generations);
    for _ in 0..n_generations {
        let prev = mgr.current_target().to_vec();
        let mut next = vec![0.0_f32; n_examples * c];
        for e in 0..n_examples {
            let p = softmax_with_temp(&prev[e * c..(e + 1) * c], temperature);
            // New logits = contraction · log(p); subtract the row mean to keep them centred.
            let logs: Vec<f32> = p.iter().map(|&pi| contraction * (pi + EPS).ln()).collect();
            let mean = logs.iter().sum::<f32>() / c as f32;
            for k in 0..c {
                next[e * c + k] = logs[k] - mean;
            }
        }
        metrics.push(mgr.push_generation(next)?);
    }
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn teacher_batch() -> (Vec<f32>, usize, usize) {
        // 3 examples, 4 classes, each with a clear arg-max.
        let logits = vec![
            3.0, 1.0, 0.5, 0.2, // ex0 -> class 0
            0.1, 2.5, 1.0, 0.3, // ex1 -> class 1
            0.0, 0.4, 0.6, 2.8, // ex2 -> class 3
        ];
        (logits, 3, 4)
    }

    #[test]
    fn new_rejects_bad_dims() {
        assert!(BanMultiGen::new(vec![0.0; 5], 2, 3, 4.0).is_err());
        assert!(BanMultiGen::new(vec![0.0; 6], 0, 3, 4.0).is_err());
        assert!(BanMultiGen::new(vec![0.0; 6], 2, 3, 0.0).is_err());
    }

    #[test]
    fn push_increases_generation() {
        let (t, ne, nc) = teacher_batch();
        let mut mgr = BanMultiGen::new(t.clone(), ne, nc, 4.0).expect("mgr");
        assert_eq!(mgr.current_generation(), 0);
        let m = mgr.push_generation(t).expect("push");
        assert_eq!(mgr.current_generation(), 1);
        assert_eq!(m.generation, 1);
        // Pushing an identical generation: zero KL, perfect agreement.
        assert!(m.inter_gen_kl < 1e-5, "kl {}", m.inter_gen_kl);
        assert!((m.argmax_agreement - 1.0).abs() < 1e-6);
    }

    #[test]
    fn push_wrong_dim_errors() {
        let (t, ne, nc) = teacher_batch();
        let mut mgr = BanMultiGen::new(t, ne, nc, 4.0).expect("mgr");
        assert!(mgr.push_generation(vec![0.0; 5]).is_err());
    }

    #[test]
    fn ensemble_mean_of_two() {
        let (t, ne, nc) = teacher_batch();
        let mut mgr = BanMultiGen::new(t.clone(), ne, nc, 4.0).expect("mgr");
        let scaled: Vec<f32> = t.iter().map(|&v| v + 2.0).collect();
        mgr.push_generation(scaled.clone()).expect("push");
        let ens = mgr.ensemble_logits();
        for i in 0..ens.len() {
            let expected = 0.5 * (t[i] + scaled[i]);
            assert!((ens[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn ensemble_last_k_excludes_teacher() {
        let (t, ne, nc) = teacher_batch();
        let mut mgr = BanMultiGen::new(t, ne, nc, 4.0).expect("mgr");
        let g1: Vec<f32> = vec![1.0; ne * nc];
        let g2: Vec<f32> = vec![3.0; ne * nc];
        mgr.push_generation(g1).expect("g1");
        mgr.push_generation(g2).expect("g2");
        // Last 2 of {teacher, g1=1, g2=3} → mean of (1, 3) = 2.
        let ens = mgr.ensemble_last_k(2);
        for &v in &ens {
            assert!((v - 2.0).abs() < 1e-5, "v {v}");
        }
    }

    #[test]
    fn ensemble_last_k_clamps() {
        let (t, ne, nc) = teacher_batch();
        let mut mgr = BanMultiGen::new(t, ne, nc, 4.0).expect("mgr");
        mgr.push_generation(vec![1.0; ne * nc]).expect("g1");
        // Requesting more than available falls back to all.
        let a = mgr.ensemble_last_k(10);
        let b = mgr.ensemble_logits();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn trajectory_kl_decreases_monotonically() {
        // The core convergence claim: successive generations change less and less.
        let (t, ne, nc) = teacher_batch();
        let metrics = simulate_ban_trajectory(&t, ne, nc, 6, 4.0, 0.6).expect("traj");
        assert_eq!(metrics.len(), 6);
        for w in metrics.windows(2) {
            assert!(
                w[1].inter_gen_kl <= w[0].inter_gen_kl + 1e-6,
                "KL increased: {} -> {}",
                w[0].inter_gen_kl,
                w[1].inter_gen_kl
            );
        }
        // Convergence: the final inter-generation KL is tiny.
        assert!(
            metrics[5].inter_gen_kl < metrics[0].inter_gen_kl,
            "no overall convergence"
        );
    }

    #[test]
    fn trajectory_is_deterministic() {
        let (t, ne, nc) = teacher_batch();
        let a = simulate_ban_trajectory(&t, ne, nc, 5, 3.0, 0.5).expect("a");
        let b = simulate_ban_trajectory(&t, ne, nc, 5, 3.0, 0.5).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn trajectory_rejects_bad_contraction() {
        let (t, ne, nc) = teacher_batch();
        assert!(simulate_ban_trajectory(&t, ne, nc, 5, 4.0, 1.0).is_err());
        assert!(simulate_ban_trajectory(&t, ne, nc, 0, 4.0, 0.5).is_err());
    }
}
