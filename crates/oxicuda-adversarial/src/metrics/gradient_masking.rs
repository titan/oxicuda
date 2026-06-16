//! Gradient Masking Diagnostics (Athalye et al., 2018).
//!
//! Implements the diagnostic checklist from:
//! * Athalye, Carlini & Wagner (2018 ICML): *"Obfuscated Gradients Give a
//!   False Sense of Security: Circumventing Defenses to Adversarial Examples"*
//!
//! The key insight is that many gradient-based defenses inadvertently or
//! deliberately cause gradient masking (a.k.a. gradient obfuscation), which
//! makes white-box gradient attacks fail while the model is *not* truly robust.
//!
//! # Diagnostic Suite
//!
//! | Symptom                              | Masking type             |
//! |--------------------------------------|--------------------------|
//! | Black-box ≫ white-box ASR            | Obfuscated / shattered   |
//! | 1-step FGSM ≫ multi-step PGD ASR    | Vanishing / exploding    |
//! | Unbounded perturbation still fails   | Shattered gradients      |
//! | Random noise ≫ adversarial           | Masking                  |
//!
//! # References
//!
//! * Madry, Makelov, Schmidt, Tsipras & Vladu (2018 ICLR): *"Towards Deep
//!   Learning Models Resistant to Adversarial Attacks"*
//! * Buckman, Roy, Raffel & Goodfellow (2018): *"Thermometer Encoding: One Hot
//!   Way To Resist Adversarial Examples"*

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── GradMaskingConfig ────────────────────────────────────────────────────────

/// Configuration for gradient-masking diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradMaskingConfig {
    /// Perturbation budget ε; must be `> 0` and finite.
    pub eps: f32,
    /// Number of random restarts for shattered-gradients test (reserved).
    pub n_restarts: usize,
    /// EOT sample count for stochastic-defense detection (reserved).
    pub n_eot_samples: usize,
}

impl Default for GradMaskingConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            n_restarts: 10,
            n_eot_samples: 32,
        }
    }
}

// ─── GradMaskingConclusion ────────────────────────────────────────────────────

/// Conclusion drawn from gradient-masking diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum GradMaskingConclusion {
    /// Gradients are shattered: even unbounded perturbation cannot reliably fool
    /// the model, suggesting that gradients are so distorted they provide no
    /// useful signal.  Typically caused by non-differentiable preprocessing
    /// (e.g. thermometer encoding, bit-depth reduction) or exploding gradients.
    ShatteredGradients,
    /// Gradient randomness from a stochastic component (e.g. random padding,
    /// dropout, injected noise) masks the gradient signal.  Symptoms: black-box
    /// attack succeeds where white-box fails; random noise works as well as
    /// adversarial.
    StochasticGradients,
    /// Vanishing or exploding gradients prevent iterative attacks from
    /// converging.  Symptom: single-step FGSM outperforms multi-step PGD.
    VanishingExplodingGradients,
    /// No masking symptoms detected.  The model may be genuinely robust.
    LikelyClean,
}

// ─── GradientMaskingReport ────────────────────────────────────────────────────

/// Full diagnostic report from [`diagnose_gradient_masking`].
#[derive(Debug, Clone, PartialEq)]
pub struct GradientMaskingReport {
    /// `true` if black-box ASR > white-box ASR + 0.1 (sign of gradient obfuscation).
    pub black_box_better: bool,
    /// `true` if single-step FGSM ASR > multi-step PGD ASR + 0.1
    /// (sign of vanishing/exploding gradients).
    pub single_step_worse: bool,
    /// `true` if unbounded attack (ε × 1000) ASR < 0.95
    /// (sign of shattered gradients).
    pub unbounded_success: bool,
    /// `true` if random L∞-ball perturbation ASR > multi-step ASR + 0.05
    /// (sign of gradient masking).
    pub random_better: bool,
    /// High-level conclusion.
    pub conclusion: GradMaskingConclusion,
}

// ─── diagnose_gradient_masking ────────────────────────────────────────────────

/// Run gradient-masking diagnostics on pre-computed attack success rates.
///
/// All ASR values must lie in `[0, 1]`.
///
/// # Arguments
///
/// * `white_box_asr`   — ASR of white-box gradient attack (e.g. PGD with true gradients).
/// * `black_box_asr`   — ASR of black-box attack (e.g. Square Attack, transfer).
/// * `single_step_asr` — ASR of 1-step FGSM.
/// * `multi_step_asr`  — ASR of multi-step PGD.
/// * `unbounded_asr`   — ASR with ε × 1000 (effectively unbounded).
/// * `random_asr`      — ASR of uniform random perturbation in the ε-ball.
/// * `cfg`             — Diagnostic configuration.
///
/// # Errors
/// * [`AdvError::InvalidEpsilon`]  — `cfg.eps ≤ 0` or non-finite.
/// * [`AdvError::NanEncountered`]  — any ASR value is non-finite.
/// * [`AdvError::Internal`]        — any ASR value is outside `[0, 1]`.
pub fn diagnose_gradient_masking(
    white_box_asr: f32,
    black_box_asr: f32,
    single_step_asr: f32,
    multi_step_asr: f32,
    unbounded_asr: f32,
    random_asr: f32,
    cfg: &GradMaskingConfig,
) -> AdvResult<GradientMaskingReport> {
    if !(cfg.eps > 0.0 && cfg.eps.is_finite()) {
        return Err(AdvError::InvalidEpsilon { eps: cfg.eps });
    }

    let all_asrs = [
        white_box_asr,
        black_box_asr,
        single_step_asr,
        multi_step_asr,
        unbounded_asr,
        random_asr,
    ];
    for &asr in &all_asrs {
        if !asr.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "diagnose_gradient_masking",
            });
        }
        if !(0.0..=1.0).contains(&asr) {
            return Err(AdvError::Internal(format!(
                "ASR value {asr} is outside [0, 1]"
            )));
        }
    }

    // Apply diagnostic rules from Athalye et al. (2018).
    let black_box_better = black_box_asr > white_box_asr + 0.1;
    let single_step_worse = single_step_asr > multi_step_asr + 0.1;
    // unbounded_success = True when unbounded attack *still* mostly fails.
    let unbounded_success = unbounded_asr < 0.95;
    let random_better = random_asr > multi_step_asr + 0.05;

    let conclusion = if unbounded_success {
        GradMaskingConclusion::ShatteredGradients
    } else if black_box_better || random_better {
        GradMaskingConclusion::StochasticGradients
    } else if single_step_worse {
        GradMaskingConclusion::VanishingExplodingGradients
    } else {
        GradMaskingConclusion::LikelyClean
    };

    Ok(GradientMaskingReport {
        black_box_better,
        single_step_worse,
        unbounded_success,
        random_better,
        conclusion,
    })
}

// ─── random_perturbation_asr ──────────────────────────────────────────────────

/// Compute the attack success rate of random L∞-ε perturbation as a baseline.
///
/// For each of `n_trials` trials, adds uniform noise in `[-eps, eps]` to each
/// dimension of `x` and checks whether the prediction differs from `true_label`.
/// Returns the fraction of trials where the prediction changed.
///
/// This is a key diagnostic baseline: if random noise is as effective as a
/// carefully crafted adversarial example, the gradient signal is likely masked.
///
/// # Arguments
///
/// * `x`           — Original input vector.
/// * `true_label`  — Ground-truth class label.
/// * `eps`         — L∞ perturbation budget.
/// * `n_trials`    — Number of random trials.
/// * `rng`         — Deterministic LCG RNG.
/// * `predict_fn`  — Closure mapping a perturbed input to a class label.
///
/// # Errors
/// * [`AdvError::EmptyInput`]      — `x` is empty.
/// * [`AdvError::InvalidEpsilon`]  — `eps ≤ 0` or non-finite.
/// * [`AdvError::InvalidNumSteps`] — `n_trials == 0`.
/// * Propagates errors from `predict_fn`.
pub fn random_perturbation_asr(
    x: &[f32],
    true_label: usize,
    eps: f32,
    n_trials: usize,
    rng: &mut LcgRng,
    mut predict_fn: impl FnMut(&[f32]) -> AdvResult<usize>,
) -> AdvResult<f32> {
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(eps > 0.0 && eps.is_finite()) {
        return Err(AdvError::InvalidEpsilon { eps });
    }
    if n_trials == 0 {
        return Err(AdvError::InvalidNumSteps);
    }

    let dim = x.len();
    let mut perturbed = vec![0.0_f32; dim];
    let mut n_success: usize = 0;

    for _ in 0..n_trials {
        // Generate uniform noise in [-eps, eps] for each dimension.
        for i in 0..dim {
            // next_f32() ∈ [0, 1) → scale to [-eps, eps]
            let noise = (rng.next_f32() * 2.0 - 1.0) * eps;
            perturbed[i] = x[i] + noise;
        }
        let pred = predict_fn(&perturbed)?;
        if pred != true_label {
            n_success += 1;
        }
    }

    Ok(n_success as f32 / n_trials as f32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> GradMaskingConfig {
        GradMaskingConfig {
            eps: 0.1,
            n_restarts: 5,
            n_eot_samples: 16,
        }
    }

    // ── diagnose_gradient_masking ─────────────────────────────────────────────

    #[test]
    fn diagnose_likely_clean() {
        let report = diagnose_gradient_masking(0.8, 0.7, 0.5, 0.8, 0.99, 0.1, &default_cfg())
            .expect("value should be present");
        assert_eq!(report.conclusion, GradMaskingConclusion::LikelyClean);
        assert!(!report.black_box_better);
        assert!(!report.single_step_worse);
        assert!(!report.unbounded_success);
        assert!(!report.random_better);
    }

    #[test]
    fn diagnose_shattered_gradients() {
        // unbounded_asr = 0.3 < 0.95 → shattered
        let report = diagnose_gradient_masking(0.1, 0.2, 0.2, 0.1, 0.3, 0.05, &default_cfg())
            .expect("value should be present");
        assert_eq!(report.conclusion, GradMaskingConclusion::ShatteredGradients);
        assert!(report.unbounded_success);
    }

    #[test]
    fn diagnose_stochastic_gradients_black_box() {
        // black_box_asr = 0.9, white_box_asr = 0.1 → black_box_better
        // unbounded = 0.98 → not shattered
        let report = diagnose_gradient_masking(0.1, 0.9, 0.3, 0.3, 0.98, 0.15, &default_cfg())
            .expect("value should be present");
        assert_eq!(
            report.conclusion,
            GradMaskingConclusion::StochasticGradients
        );
        assert!(report.black_box_better);
    }

    #[test]
    fn diagnose_stochastic_gradients_random_better() {
        // random_asr = 0.8, multi_step_asr = 0.1 → random_better
        // unbounded = 0.98 → not shattered; black_box not better
        let report = diagnose_gradient_masking(0.3, 0.35, 0.2, 0.1, 0.98, 0.8, &default_cfg())
            .expect("value should be present");
        assert_eq!(
            report.conclusion,
            GradMaskingConclusion::StochasticGradients
        );
        assert!(report.random_better);
    }

    #[test]
    fn diagnose_vanishing_exploding_gradients() {
        // single_step_asr = 0.9, multi_step_asr = 0.1 → single_step_worse
        // unbounded = 0.98 → not shattered; black_box and random not better
        let report = diagnose_gradient_masking(0.3, 0.35, 0.9, 0.1, 0.98, 0.12, &default_cfg())
            .expect("value should be present");
        assert_eq!(
            report.conclusion,
            GradMaskingConclusion::VanishingExplodingGradients
        );
        assert!(report.single_step_worse);
    }

    #[test]
    fn diagnose_shattered_takes_priority_over_other_symptoms() {
        // Even when other symptoms are present, shattered takes priority.
        let report = diagnose_gradient_masking(0.05, 0.9, 0.9, 0.1, 0.3, 0.9, &default_cfg())
            .expect("value should be present");
        assert_eq!(report.conclusion, GradMaskingConclusion::ShatteredGradients);
    }

    #[test]
    fn diagnose_invalid_eps_errors() {
        let bad_cfg = GradMaskingConfig {
            eps: 0.0,
            ..default_cfg()
        };
        let r = diagnose_gradient_masking(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, &bad_cfg);
        assert!(matches!(r, Err(AdvError::InvalidEpsilon { .. })));
    }

    #[test]
    fn diagnose_nan_asr_errors() {
        let r = diagnose_gradient_masking(f32::NAN, 0.5, 0.5, 0.5, 0.5, 0.5, &default_cfg());
        assert!(matches!(r, Err(AdvError::NanEncountered { .. })));
    }

    #[test]
    fn diagnose_out_of_range_asr_errors() {
        let r = diagnose_gradient_masking(1.5, 0.5, 0.5, 0.5, 0.5, 0.5, &default_cfg());
        assert!(matches!(r, Err(AdvError::Internal(_))));
    }

    // ── random_perturbation_asr ───────────────────────────────────────────────

    #[test]
    fn random_perturbation_constant_classifier_zero_asr() {
        // Classifier always returns true_label=0 → ASR = 0.
        let mut rng = LcgRng::new(42);
        let x = vec![0.5_f32; 8];
        let asr = random_perturbation_asr(&x, 0, 0.1, 100, &mut rng, |_| Ok(0))
            .expect("value should be present");
        assert!((asr - 0.0).abs() < 1e-6);
    }

    #[test]
    fn random_perturbation_always_wrong_classifier_full_asr() {
        // Classifier always returns 99 ≠ true_label=0 → ASR = 1.
        let mut rng = LcgRng::new(7);
        let x = vec![0.5_f32; 4];
        let asr = random_perturbation_asr(&x, 0, 0.1, 50, &mut rng, |_| Ok(99))
            .expect("value should be present");
        assert!((asr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn random_perturbation_empty_input_errors() {
        let mut rng = LcgRng::new(0);
        let r = random_perturbation_asr(&[], 0, 0.1, 10, &mut rng, |_| Ok(0));
        assert!(matches!(r, Err(AdvError::EmptyInput)));
    }

    #[test]
    fn random_perturbation_invalid_eps_errors() {
        let mut rng = LcgRng::new(0);
        let x = vec![0.5_f32; 4];
        let r = random_perturbation_asr(&x, 0, -0.1, 10, &mut rng, |_| Ok(0));
        assert!(matches!(r, Err(AdvError::InvalidEpsilon { .. })));
    }

    #[test]
    fn random_perturbation_zero_trials_errors() {
        let mut rng = LcgRng::new(0);
        let x = vec![0.5_f32; 4];
        let r = random_perturbation_asr(&x, 0, 0.1, 0, &mut rng, |_| Ok(0));
        assert!(matches!(r, Err(AdvError::InvalidNumSteps)));
    }

    #[test]
    fn random_perturbation_asr_in_unit_range() {
        // ASR must be in [0, 1] for any classifier.
        let mut rng = LcgRng::new(123);
        let x = vec![0.5_f32; 16];
        // Half-and-half classifier.
        let mut counter = 0_usize;
        let asr = random_perturbation_asr(&x, 0, 0.1, 200, &mut rng, |_| {
            counter += 1;
            Ok(counter % 2) // alternates between 0 and 1
        })
        .expect("value should be present");
        assert!((0.0..=1.0).contains(&asr));
    }

    #[test]
    fn random_perturbation_deterministic_with_same_seed() {
        let x = vec![0.3_f32; 6];
        let mut rng1 = LcgRng::new(999);
        let mut rng2 = LcgRng::new(999);
        let predict = |v: &[f32]| -> AdvResult<usize> { Ok(if v[0] > 0.3 { 1 } else { 0 }) };
        let asr1 = random_perturbation_asr(&x, 0, 0.1, 50, &mut rng1, predict)
            .expect("random_perturbation_asr should succeed");
        let asr2 = random_perturbation_asr(&x, 0, 0.1, 50, &mut rng2, predict)
            .expect("random_perturbation_asr should succeed");
        assert!((asr1 - asr2).abs() < 1e-6);
    }
}
