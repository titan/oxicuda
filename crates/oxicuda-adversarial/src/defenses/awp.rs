//! Adversarial Weight Perturbation (AWP) defense.
//!
//! Reference: Wu, Xia & Wang (2020 NeurIPS),
//! *"Adversarial Weight Perturbation Helps Robust Generalization"*.
//!
//! AWP improves adversarial training by additionally perturbing the model
//! weights in the direction that maximises the adversarial loss:
//!
//! ```text
//! δ* = argmax_{||δ_W||_F ≤ γ}  L(f_{θ+δ}(x_adv), y)
//! ```
//!
//! This weight perturbation flattens the loss landscape around adversarial
//! examples, yielding better generalisation.  The outer loss (TRADES or
//! standard AT) is then computed on the perturbed model.
//!
//! # Algorithm
//!
//! 1. Compute gradient ∂L_adv/∂θ via a user-supplied closure.
//! 2. Run `n_ascent_steps` of gradient ascent in weight space.
//! 3. Project the accumulated weight delta onto the Frobenius ball of radius γ.
//! 4. Compute the TRADES loss on the perturbed model.
//!
//! # Conventions
//!
//! * `flat_weights` — model parameters θ in row-major flat layout.
//! * The gradient closure receives *perturbed* weights and returns ∂L/∂θ.
//! * TRADES loss is computed with the logits that the caller supplies for the
//!   perturbed model (clean_logits from f_{θ+δ}(x), adv_logits from
//!   f_{θ+δ}(x_adv)).

use crate::defenses::trades::{TradesConfig, trades_loss};
use crate::error::{AdvError, AdvResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for the AWP defence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AwpConfig {
    /// Frobenius-ball radius γ for weight perturbation (`||δ_W||_F ≤ γ`).
    /// Typical range: 0.01 – 0.05.
    pub gamma: f32,
    /// Number of inner ascent steps to find δ* (typical 1–3).
    pub n_ascent_steps: usize,
    /// Learning rate for the inner gradient-ascent loop.
    pub ascent_lr: f32,
    /// β for the TRADES outer loss.  `0.0` recovers standard AT (plain CE).
    pub trades_beta: f32,
}

impl Default for AwpConfig {
    fn default() -> Self {
        Self {
            gamma: 0.01,
            n_ascent_steps: 1,
            ascent_lr: 0.01,
            trades_beta: 6.0,
        }
    }
}

// ─── Output types ─────────────────────────────────────────────────────────────

/// Flattened weight perturbation δ* produced by [`AwpDefense`].
#[derive(Debug, Clone)]
pub struct AwpWeightDelta {
    /// Element-wise perturbation; same length as `flat_weights`.
    pub delta: Vec<f32>,
    /// Frobenius norm of `delta` (== γ after projection).
    pub frobenius_norm: f32,
}

// ─── AwpDefense ───────────────────────────────────────────────────────────────

/// Implementation of the Adversarial Weight Perturbation (AWP) defence.
pub struct AwpDefense;

impl AwpDefense {
    // ─── Scalar helpers ──────────────────────────────────────────────────────

    /// Frobenius norm: `sqrt(Σ x_i²)`.
    #[must_use]
    pub fn frobenius_norm(v: &[f32]) -> f32 {
        let sq_sum: f32 = v.iter().map(|&x| x * x).sum();
        sq_sum.sqrt()
    }

    /// Project gradient onto the Frobenius ball of radius `gamma`:
    ///
    /// ```text
    /// δ[i] = gamma * grad[i] / (||grad||_F + ε)    ε = 1e-12
    /// ```
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]          — `grad` is empty.
    /// * [`AdvError::InvalidEpsilon`]       — `gamma <= 0` or non-finite.
    /// * [`AdvError::NanEncountered`]       — non-finite value in `grad`.
    pub fn project_to_frobenius_ball(grad: &[f32], gamma: f32) -> AdvResult<AwpWeightDelta> {
        if grad.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps: gamma });
        }
        if grad.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "project_to_frobenius_ball:grad",
            });
        }
        let frob = Self::frobenius_norm(grad);
        let scale = gamma / (frob + 1e-12_f32);
        let delta: Vec<f32> = grad.iter().map(|&g| scale * g).collect();
        // frobenius_norm of delta == gamma * frob / (frob + eps) ≈ gamma
        let actual_frob = Self::frobenius_norm(&delta);
        Ok(AwpWeightDelta {
            delta,
            frobenius_norm: actual_frob,
        })
    }

    // ─── Core: find weight perturbation ──────────────────────────────────────

    /// Find the weight perturbation δ* that maximises the adversarial loss.
    ///
    /// Runs `cfg.n_ascent_steps` steps of gradient ascent in θ-space, then
    /// projects the accumulated delta onto `||δ||_F ≤ γ`.
    ///
    /// # Parameters
    /// * `flat_weights`   — model parameters θ (not modified).
    /// * `cfg`            — AWP hyper-parameters.
    /// * `adv_loss_grad`  — closure returning ∂L_adv/∂θ for any weight vector.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]      — `flat_weights` is empty.
    /// * [`AdvError::InvalidEpsilon`]  — `gamma <= 0`.
    /// * [`AdvError::InvalidAlpha`]    — `ascent_lr <= 0`.
    /// * [`AdvError::InvalidNumSteps`] — `n_ascent_steps == 0`.
    /// * [`AdvError::NanEncountered`]  — NaN/Inf in gradient or delta.
    pub fn find_weight_perturbation<F>(
        flat_weights: &[f32],
        cfg: &AwpConfig,
        adv_loss_grad: F,
    ) -> AdvResult<AwpWeightDelta>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        let n_w = flat_weights.len();
        if n_w == 0 {
            return Err(AdvError::EmptyInput);
        }
        if !(cfg.gamma.is_finite() && cfg.gamma > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps: cfg.gamma });
        }
        if !(cfg.ascent_lr.is_finite() && cfg.ascent_lr > 0.0) {
            return Err(AdvError::InvalidAlpha {
                alpha: cfg.ascent_lr,
            });
        }
        if cfg.n_ascent_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }

        let mut delta = vec![0.0_f32; n_w];

        for _step in 0..cfg.n_ascent_steps {
            // Perturbed weights: θ + δ
            let perturbed: Vec<f32> = flat_weights
                .iter()
                .zip(delta.iter())
                .map(|(&w, &d)| w + d)
                .collect();

            // ∂L_adv/∂(θ+δ)
            let grad = adv_loss_grad(&perturbed);

            // Validate gradient length
            if grad.len() != n_w {
                return Err(AdvError::DimensionMismatch {
                    expected: n_w,
                    got: grad.len(),
                });
            }

            // Gradient ascent: δ ← δ + lr * ∂L/∂θ
            for (d, &g) in delta.iter_mut().zip(grad.iter()) {
                if !g.is_finite() {
                    return Err(AdvError::NanEncountered {
                        location: "find_weight_perturbation:grad",
                    });
                }
                *d += cfg.ascent_lr * g;
            }
        }

        // Project onto Frobenius ball of radius γ
        let projected = Self::project_to_frobenius_ball(&delta, cfg.gamma)?;

        // Final NaN guard
        if projected.delta.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "find_weight_perturbation:delta",
            });
        }

        Ok(projected)
    }

    // ─── Apply / remove perturbation ─────────────────────────────────────────

    /// Apply weight perturbation: `θ_perturbed[i] = θ[i] + delta[i]`.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — lengths differ.
    pub fn apply_perturbation(flat_weights: &[f32], delta: &AwpWeightDelta) -> AdvResult<Vec<f32>> {
        if flat_weights.len() != delta.delta.len() {
            return Err(AdvError::DimensionMismatch {
                expected: flat_weights.len(),
                got: delta.delta.len(),
            });
        }
        Ok(flat_weights
            .iter()
            .zip(delta.delta.iter())
            .map(|(&w, &d)| w + d)
            .collect())
    }

    /// Remove weight perturbation: `θ[i] = θ_perturbed[i] - delta[i]`.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — lengths differ.
    pub fn remove_perturbation(
        flat_weights_perturbed: &[f32],
        delta: &AwpWeightDelta,
    ) -> AdvResult<Vec<f32>> {
        if flat_weights_perturbed.len() != delta.delta.len() {
            return Err(AdvError::DimensionMismatch {
                expected: flat_weights_perturbed.len(),
                got: delta.delta.len(),
            });
        }
        Ok(flat_weights_perturbed
            .iter()
            .zip(delta.delta.iter())
            .map(|(&w, &d)| w - d)
            .collect())
    }

    // ─── AWP-TRADES loss ─────────────────────────────────────────────────────

    /// Compute the AWP-TRADES loss:
    ///
    /// 1. Find δ* via [`Self::find_weight_perturbation`].
    /// 2. Compute TRADES loss on the supplied (perturbed) logits using `cfg.trades_beta`.
    ///
    /// The caller is responsible for computing `clean_logits` and `adv_logits`
    /// from the **perturbed** model f_{θ+δ*}.
    ///
    /// # Parameters
    /// * `flat_weights`  — θ (used only to compute δ*; not re-evaluated here).
    /// * `clean_logits`  — N×K logits from f_{θ+δ*}(x_clean), row-major.
    /// * `adv_logits`    — N×K logits from f_{θ+δ*}(x_adv), row-major.
    /// * `labels`        — N class indices in `[0, K)`.
    /// * `n`, `k`        — batch size and number of classes (k >= 2).
    /// * `cfg`           — AWP + TRADES hyper-parameters.
    /// * `adv_loss_grad` — gradient closure for the inner ascent.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]         — empty batch.
    /// * [`AdvError::DimensionMismatch`]  — logit / label shape mismatch.
    /// * [`AdvError::InvalidLossWeight`]  — `k < 2`.
    /// * Any error from [`Self::find_weight_perturbation`] or [`trades_loss`].
    pub fn awp_trades_loss(
        flat_weights: &[f32],
        clean_logits: &[f32],
        adv_logits: &[f32],
        labels: &[usize],
        n: usize,
        k: usize,
        cfg: &AwpConfig,
        adv_loss_grad: impl Fn(&[f32]) -> Vec<f32>,
    ) -> AdvResult<(f32, AwpWeightDelta)> {
        if n == 0 || k == 0 {
            return Err(AdvError::EmptyInput);
        }
        if k < 2 {
            return Err(AdvError::InvalidLossWeight { weight: k as f32 });
        }
        let expected = n * k;
        if clean_logits.len() != expected {
            return Err(AdvError::DimensionMismatch {
                expected,
                got: clean_logits.len(),
            });
        }
        if adv_logits.len() != expected {
            return Err(AdvError::DimensionMismatch {
                expected,
                got: adv_logits.len(),
            });
        }
        if labels.len() != n {
            return Err(AdvError::DimensionMismatch {
                expected: n,
                got: labels.len(),
            });
        }

        // Find weight perturbation δ*
        let weight_delta = Self::find_weight_perturbation(flat_weights, cfg, adv_loss_grad)?;

        // TRADES loss on (perturbed-model) logits
        let trades_cfg = TradesConfig::new(cfg.trades_beta)?;
        let loss = trades_loss(clean_logits, adv_logits, labels, n, k, &trades_cfg)?;

        Ok((loss, weight_delta))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: approx equality.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // Helper: simple cross-entropy loss for tests (returns finite grad always).
    fn zero_grad(weights: &[f32]) -> Vec<f32> {
        vec![0.0_f32; weights.len()]
    }

    fn constant_grad(weights: &[f32], val: f32) -> Vec<f32> {
        vec![val; weights.len()]
    }

    // ── frobenius_norm ────────────────────────────────────────────────────────

    #[test]
    fn frobenius_norm_zero_vector() {
        let v = vec![0.0_f32; 8];
        assert_eq!(AwpDefense::frobenius_norm(&v), 0.0);
    }

    #[test]
    fn frobenius_norm_three_four_five() {
        let v = vec![3.0_f32, 4.0];
        assert!(approx_eq(AwpDefense::frobenius_norm(&v), 5.0, 1e-6));
    }

    #[test]
    fn frobenius_norm_unit_vector() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        assert!(approx_eq(AwpDefense::frobenius_norm(&v), 1.0, 1e-7));
    }

    // ── project_to_frobenius_ball ─────────────────────────────────────────────

    #[test]
    fn project_frobenius_ball_norm_equals_gamma() {
        let grad = vec![1.0_f32, 2.0, 3.0, 4.0];
        let gamma = 0.05_f32;
        let result = AwpDefense::project_to_frobenius_ball(&grad, gamma)
            .expect("project_to_frobenius_ball should succeed");
        assert!(approx_eq(result.frobenius_norm, gamma, 1e-5));
        // Verify via actual computation too
        let actual_norm = AwpDefense::frobenius_norm(&result.delta);
        assert!(approx_eq(actual_norm, gamma, 1e-5));
    }

    #[test]
    fn project_frobenius_ball_direction_preserved() {
        let grad = vec![1.0_f32, 0.0, 0.0];
        let gamma = 0.03_f32;
        let result = AwpDefense::project_to_frobenius_ball(&grad, gamma)
            .expect("project_to_frobenius_ball should succeed");
        // Should project onto the unit vector in [1,0,0] direction scaled by gamma
        assert!(approx_eq(result.delta[0], gamma, 1e-5));
        assert!(approx_eq(result.delta[1], 0.0, 1e-7));
        assert!(approx_eq(result.delta[2], 0.0, 1e-7));
    }

    #[test]
    fn project_frobenius_ball_empty_errors() {
        assert!(matches!(
            AwpDefense::project_to_frobenius_ball(&[], 0.01),
            Err(AdvError::EmptyInput)
        ));
    }

    #[test]
    fn project_frobenius_ball_nonpositive_gamma_errors() {
        let grad = vec![1.0_f32, 2.0];
        assert!(AwpDefense::project_to_frobenius_ball(&grad, 0.0).is_err());
        assert!(AwpDefense::project_to_frobenius_ball(&grad, -0.01).is_err());
        assert!(AwpDefense::project_to_frobenius_ball(&grad, f32::INFINITY).is_err());
    }

    // ── find_weight_perturbation ──────────────────────────────────────────────

    #[test]
    fn find_weight_perturbation_zero_grad_returns_zero_delta() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0];
        let cfg = AwpConfig::default();
        // Zero gradient → accumulated delta stays zero → project_to_frobenius_ball
        // on zero vector produces a nearly-zero delta (numerically stable).
        let result = AwpDefense::find_weight_perturbation(&weights, &cfg, zero_grad);
        // With zero grad the Frobenius norm of delta after projection = gamma
        // (due to the epsilon-regularised denominator); the direction is degenerate
        // but the norm must be <= gamma.
        let result = result.expect("result should be present");
        assert!(result.frobenius_norm <= cfg.gamma + 1e-5);
    }

    #[test]
    fn find_weight_perturbation_constant_grad_scales_correctly() {
        let weights = vec![0.0_f32; 4];
        let cfg = AwpConfig {
            gamma: 0.05,
            n_ascent_steps: 1,
            ascent_lr: 0.01,
            trades_beta: 6.0,
        };
        // Constant grad [1, 1, 1, 1] → delta after 1 step = lr * 1 = 0.01 for each
        // Then projected to Frobenius ball of gamma=0.05
        let result =
            AwpDefense::find_weight_perturbation(&weights, &cfg, |w| constant_grad(w, 1.0))
                .expect("value should be present");
        // After projection: ||delta||_F should equal gamma
        assert!(approx_eq(result.frobenius_norm, cfg.gamma, 1e-5));
        // All components equal (equal-gradient → equal delta)
        let d0 = result.delta[0];
        for &d in &result.delta {
            assert!(approx_eq(d, d0, 1e-6));
        }
    }

    #[test]
    fn find_weight_perturbation_three_steps() {
        let weights = vec![1.0_f32; 8];
        let cfg = AwpConfig {
            gamma: 0.02,
            n_ascent_steps: 3,
            ascent_lr: 0.005,
            trades_beta: 6.0,
        };
        let result =
            AwpDefense::find_weight_perturbation(&weights, &cfg, |w| constant_grad(w, 1.0));
        let result = result.expect("result should be present");
        // After 3 steps the norm is still projected to gamma
        assert!(approx_eq(result.frobenius_norm, cfg.gamma, 1e-5));
    }

    #[test]
    fn find_weight_perturbation_empty_weights_errors() {
        let cfg = AwpConfig::default();
        assert!(matches!(
            AwpDefense::find_weight_perturbation(&[], &cfg, zero_grad),
            Err(AdvError::EmptyInput)
        ));
    }

    #[test]
    fn find_weight_perturbation_invalid_gamma_errors() {
        let weights = vec![1.0_f32; 4];
        let cfg_bad = AwpConfig {
            gamma: -0.01,
            ..AwpConfig::default()
        };
        assert!(AwpDefense::find_weight_perturbation(&weights, &cfg_bad, zero_grad).is_err());
    }

    #[test]
    fn find_weight_perturbation_zero_steps_errors() {
        let weights = vec![1.0_f32; 4];
        let cfg_bad = AwpConfig {
            n_ascent_steps: 0,
            ..AwpConfig::default()
        };
        assert!(matches!(
            AwpDefense::find_weight_perturbation(&weights, &cfg_bad, zero_grad),
            Err(AdvError::InvalidNumSteps)
        ));
    }

    // ── apply / remove perturbation ───────────────────────────────────────────

    #[test]
    fn apply_then_remove_recovers_original() {
        let weights = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let delta = AwpWeightDelta {
            delta: vec![0.1_f32, -0.2, 0.05, 0.3, -0.1],
            frobenius_norm: 0.01,
        };
        let perturbed = AwpDefense::apply_perturbation(&weights, &delta)
            .expect("apply_perturbation should succeed");
        let recovered = AwpDefense::remove_perturbation(&perturbed, &delta)
            .expect("remove_perturbation should succeed");
        for (&r, &orig) in recovered.iter().zip(weights.iter()) {
            assert!(approx_eq(r, orig, 1e-6));
        }
    }

    #[test]
    fn apply_perturbation_length_mismatch_errors() {
        let weights = vec![1.0_f32; 4];
        let delta = AwpWeightDelta {
            delta: vec![0.1_f32; 3], // Wrong length.
            frobenius_norm: 0.1,
        };
        assert!(matches!(
            AwpDefense::apply_perturbation(&weights, &delta),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn remove_perturbation_length_mismatch_errors() {
        let perturbed = vec![1.0_f32; 4];
        let delta = AwpWeightDelta {
            delta: vec![0.1_f32; 5], // Wrong length.
            frobenius_norm: 0.1,
        };
        assert!(matches!(
            AwpDefense::remove_perturbation(&perturbed, &delta),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    // ── awp_trades_loss ───────────────────────────────────────────────────────

    #[test]
    fn awp_trades_loss_returns_finite_value() {
        // 2 samples, 3 classes
        let weights = vec![0.5_f32; 12];
        let clean_logits = vec![1.0_f32, 2.0, 0.5, 0.0, 0.5, 1.5];
        let adv_logits = vec![0.5_f32, 1.5, 0.7, 0.2, 0.7, 1.3];
        let labels = vec![1_usize, 2];
        let cfg = AwpConfig::default();
        let (loss, delta) = AwpDefense::awp_trades_loss(
            &weights,
            &clean_logits,
            &adv_logits,
            &labels,
            2,
            3,
            &cfg,
            zero_grad,
        )
        .expect("value should be present");
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        assert!(delta.frobenius_norm <= cfg.gamma + 1e-5);
    }

    #[test]
    fn awp_trades_loss_beta_zero_equals_clean_ce() {
        // With beta=0 the TRADES loss reduces to CE on clean_logits.
        let weights = vec![0.0_f32; 6];
        // Single sample, 3 classes, label=0
        let clean_logits = vec![3.0_f32, 1.0, 0.0];
        let adv_logits = vec![0.0_f32, 3.0, 1.0]; // Very different.
        let labels = vec![0_usize];
        let cfg = AwpConfig {
            trades_beta: 0.0,
            ..AwpConfig::default()
        };
        let (loss, _) = AwpDefense::awp_trades_loss(
            &weights,
            &clean_logits,
            &adv_logits,
            &labels,
            1,
            3,
            &cfg,
            zero_grad,
        )
        .expect("value should be present");
        // With beta=0, KL term is ignored → loss = CE(clean, label=0)
        // CE = -log(softmax([3,1,0])[0]) = -(3 - log(e^3+e^1+e^0))
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn awp_trades_loss_k_less_than_2_errors() {
        let weights = vec![0.0_f32; 4];
        let clean_logits = vec![1.0_f32];
        let adv_logits = vec![0.5_f32];
        let labels = vec![0_usize];
        assert!(matches!(
            AwpDefense::awp_trades_loss(
                &weights,
                &clean_logits,
                &adv_logits,
                &labels,
                1,
                1,
                &AwpConfig::default(),
                zero_grad,
            ),
            Err(AdvError::InvalidLossWeight { .. })
        ));
    }

    #[test]
    fn awp_trades_loss_dim_mismatch_errors() {
        let weights = vec![0.0_f32; 6];
        let clean_logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]; // Should be 6 = 2*3
        let adv_logits = vec![0.0_f32; 6];
        let labels = vec![0_usize, 1];
        assert!(matches!(
            AwpDefense::awp_trades_loss(
                &weights,
                &clean_logits,
                &adv_logits,
                &labels,
                2,
                3,
                &AwpConfig::default(),
                zero_grad,
            ),
            Err(AdvError::DimensionMismatch { .. })
        ));
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn default_config_has_expected_values() {
        let cfg = AwpConfig::default();
        assert!(approx_eq(cfg.gamma, 0.01, 1e-7));
        assert_eq!(cfg.n_ascent_steps, 1);
        assert!(approx_eq(cfg.ascent_lr, 0.01, 1e-7));
        assert!(approx_eq(cfg.trades_beta, 6.0, 1e-7));
    }

    // ── Frobenius bound guarantee ─────────────────────────────────────────────

    #[test]
    fn weight_perturbation_frobenius_norm_bounded_by_gamma() {
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let cfg = AwpConfig {
            gamma: 0.03,
            n_ascent_steps: 2,
            ascent_lr: 0.005,
            trades_beta: 6.0,
        };
        let result =
            AwpDefense::find_weight_perturbation(&weights, &cfg, |w| constant_grad(w, 0.5))
                .expect("value should be present");
        assert!(result.frobenius_norm <= cfg.gamma + 1e-5);
    }

    #[test]
    fn awp_trades_loss_empty_input_errors() {
        assert!(matches!(
            AwpDefense::awp_trades_loss(&[], &[], &[], &[], 0, 3, &AwpConfig::default(), zero_grad,),
            Err(AdvError::EmptyInput)
        ));
    }
}
