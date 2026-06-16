//! Targeted FGSM and PGD adversarial attacks.
//!
//! Unlike standard (untargeted) attacks that maximise the loss for the true
//! label, **targeted** attacks minimise the cross-entropy loss for a chosen
//! *target* class, thereby steering the model toward predicting that class.
//!
//! Both FGSM-targeted (single-step) and PGD-targeted (multi-step with random
//! start) are implemented here using coordinate-wise finite-difference
//! gradient estimation of the target logit with respect to the input.
//!
//! ## Gradient estimation
//!
//! Since the classifier is treated as a pure black-box function
//! `logit_fn: &[f32] → Vec<f32>`, backpropagation is unavailable.  The
//! gradient of the target logit with respect to each input coordinate is
//! estimated via symmetric finite differences:
//!
//! ```text
//! ∂ logit[t] / ∂ x[i]  ≈  (logit_fn(x + h·eᵢ)[t] − logit_fn(x − h·eᵢ)[t]) / (2h)
//! ```
//!
//! This requires **2·d** calls to `logit_fn` per step, but is correct for
//! any smooth, locally-linear classifier.  Test inputs use d ≤ 8, so this
//! is not a bottleneck in practice.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for targeted attacks.
#[derive(Debug, Clone)]
pub struct TargetedConfig {
    /// L∞ perturbation budget ε ≥ 0.
    pub epsilon: f32,
    /// Number of PGD steps.  `1` yields single-step FGSM-T.
    pub n_steps: usize,
    /// Per-step size α ≥ 0.
    pub step_size: f32,
    /// Index of the class we want the model to predict.
    pub target_class: usize,
}

// ─── Attack ──────────────────────────────────────────────────────────────────

/// Targeted FGSM / PGD attack wrapper.
#[derive(Debug)]
pub struct TargetedAttack {
    config: TargetedConfig,
}

impl TargetedAttack {
    /// Construct and validate a [`TargetedAttack`].
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]  — `epsilon` is negative or non-finite.
    /// * [`AdvError::InvalidAlpha`]    — `step_size` is negative or non-finite.
    /// * [`AdvError::InvalidNumSteps`] — `n_steps` is zero.
    pub fn new(config: TargetedConfig) -> AdvResult<Self> {
        if !(config.epsilon.is_finite() && config.epsilon >= 0.0) {
            return Err(AdvError::InvalidEpsilon {
                eps: config.epsilon,
            });
        }
        if !(config.step_size.is_finite() && config.step_size >= 0.0) {
            return Err(AdvError::InvalidAlpha {
                alpha: config.step_size,
            });
        }
        if config.n_steps < 1 {
            return Err(AdvError::InvalidNumSteps);
        }
        Ok(Self { config })
    }

    /// Expose the validated configuration.
    #[inline]
    pub fn config(&self) -> &TargetedConfig {
        &self.config
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Validate common inputs shared by all targeted attack entry points.
fn validate_inputs(x: &[f32], x_min: f32, x_max: f32) -> AdvResult<()> {
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(x_min.is_finite() && x_max.is_finite()) || x_min >= x_max {
        return Err(AdvError::InvalidLossWeight {
            weight: x_max - x_min,
        });
    }
    Ok(())
}

/// Validate that `target_class` is within range for a logit vector.
fn validate_target_class(logits: &[f32], target_class: usize) -> AdvResult<()> {
    if logits.is_empty() {
        return Err(AdvError::Internal("empty logits".into()));
    }
    if target_class >= logits.len() {
        return Err(AdvError::DimensionMismatch {
            expected: logits.len() - 1,
            got: target_class,
        });
    }
    Ok(())
}

/// Coordinate-wise finite-difference gradient of `logit_fn(·)[target]`.
///
/// Returns a `d`-element vector where element `i` approximates
/// `∂ logit[target] / ∂ x[i]` using a symmetric two-point stencil with
/// step `h`.
///
/// To avoid allocations inside the hot loop the function reuses two scratch
/// buffers that are reset to the original coordinate after each probe.
fn finite_diff_grad_target<F>(x: &[f32], logit_fn: &F, target: usize, h: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let d = x.len();
    let mut grad = vec![0.0_f32; d];
    // Scratch buffers — reused for every coordinate.
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    for i in 0..d {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        let l_plus = logit_fn(&x_plus);
        let l_minus = logit_fn(&x_minus);
        // Clamp target index: if logit_fn returns fewer dims on a weird input
        // we fall back to 0 to avoid panicking (validated at call site).
        let t = target.min(l_plus.len().saturating_sub(1));
        grad[i] = (l_plus[t] - l_minus[t]) / (2.0 * h);
        // Restore the scratch buffers for the next iteration.
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    grad
}

/// Element-wise sign: `+1` / `-1` / `0`.
#[inline]
fn sign_f32(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Project coordinate `x_adv_i` onto the L∞ ε-ball around `x_i` and then
/// clamp to `[x_min, x_max]`.
#[inline]
fn project_and_clamp(x_adv_i: f32, x_i: f32, eps: f32, x_min: f32, x_max: f32) -> f32 {
    x_adv_i.clamp(x_i - eps, x_i + eps).clamp(x_min, x_max)
}

// ─── Public API ──────────────────────────────────────────────────────────────

impl TargetedAttack {
    /// Single-step targeted FGSM.
    ///
    /// Computes the finite-difference gradient of the target logit with respect
    /// to `x`, takes a step of size `step_size` in the gradient's direction
    /// (increasing the target logit), and projects back onto the L∞ ε-ball
    /// ∩ `[x_min, x_max]`.
    ///
    /// # Errors
    /// * [`AdvError::EmptyInput`]         — `x` is empty.
    /// * [`AdvError::InvalidLossWeight`]  — `x_min >= x_max` or non-finite bounds.
    /// * [`AdvError::Internal`]           — `logit_fn` returns an empty vector.
    /// * [`AdvError::DimensionMismatch`]  — `target_class >= n_classes`.
    pub fn fgsm_targeted<F>(
        &self,
        x: &[f32],
        logit_fn: F,
        x_min: f32,
        x_max: f32,
    ) -> AdvResult<Vec<f32>>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        validate_inputs(x, x_min, x_max)?;
        // Probe logit_fn once to determine n_classes and validate target_class.
        let probe = logit_fn(x);
        validate_target_class(&probe, self.config.target_class)?;

        let d = x.len();
        let eps = self.config.epsilon;
        let target = self.config.target_class;
        let step = self.config.step_size;

        // Estimate gradient of logit[target] w.r.t. x via finite differences.
        let grad = finite_diff_grad_target(x, &logit_fn, target, 1e-4_f32);

        // FGSM-T step: move in the direction that *increases* the target logit.
        let mut x_adv = vec![0.0_f32; d];
        for i in 0..d {
            let stepped = x[i] + step * sign_f32(grad[i]);
            x_adv[i] = project_and_clamp(stepped, x[i], eps, x_min, x_max);
        }
        Ok(x_adv)
    }

    /// Multi-step targeted PGD with random start.
    ///
    /// Initialises `x_adv` with a uniform random perturbation drawn from the
    /// L∞ ε-ball, then iterates `n_steps` targeted FGSM gradient ascent steps
    /// (each projected back onto the ε-ball ∩ `[x_min, x_max]`).
    ///
    /// # Errors
    /// Same as [`TargetedAttack::fgsm_targeted`].
    pub fn pgd_targeted<F>(
        &self,
        x: &[f32],
        logit_fn: F,
        x_min: f32,
        x_max: f32,
        rng: &mut LcgRng,
    ) -> AdvResult<Vec<f32>>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        validate_inputs(x, x_min, x_max)?;
        // Probe logit_fn once to determine n_classes and validate target_class.
        let probe = logit_fn(x);
        validate_target_class(&probe, self.config.target_class)?;

        let d = x.len();
        let eps = self.config.epsilon;
        let target = self.config.target_class;
        let step = self.config.step_size;

        // Random start: uniform draw from [-eps, +eps], projected to box.
        let mut x_adv: Vec<f32> = x
            .iter()
            .map(|&xi| {
                let delta = (rng.next_f32() * 2.0 - 1.0) * eps;
                project_and_clamp(xi + delta, xi, eps, x_min, x_max)
            })
            .collect();

        // PGD iterations.
        for _ in 0..self.config.n_steps {
            let grad = finite_diff_grad_target(&x_adv, &logit_fn, target, 1e-4_f32);
            let mut next = vec![0.0_f32; d];
            for i in 0..d {
                let stepped = x_adv[i] + step * sign_f32(grad[i]);
                next[i] = project_and_clamp(stepped, x[i], eps, x_min, x_max);
            }
            x_adv = next;
        }
        Ok(x_adv)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 3-class logit function: logits = [x[0], -x[0], x[0]*0.5].
    /// At x = [0.5, ...] the argmax is class 0.
    fn simple_logit_fn(x: &[f32]) -> Vec<f32> {
        vec![x[0], -x[0], x[0] * 0.5]
    }

    fn default_cfg(target: usize) -> TargetedConfig {
        TargetedConfig {
            epsilon: 0.3,
            n_steps: 3,
            step_size: 0.1,
            target_class: target,
        }
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── 1. fgsm_output_shape ─────────────────────────────────────────────────
    #[test]
    fn fgsm_output_shape() {
        let x = vec![0.5_f32; 4];
        let attack = TargetedAttack::new(default_cfg(1)).expect("value should be present");
        let adv = attack
            .fgsm_targeted(&x, simple_logit_fn, -1.0, 1.0)
            .expect("value should be present");
        assert_eq!(adv.len(), x.len());
    }

    // ── 2. fgsm_within_epsilon ───────────────────────────────────────────────
    #[test]
    fn fgsm_within_epsilon() {
        let x = vec![0.5_f32, -0.3, 0.1, 0.0];
        let eps = 0.2_f32;
        let cfg = TargetedConfig {
            epsilon: eps,
            n_steps: 1,
            step_size: 0.1,
            target_class: 1,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let adv = attack
            .fgsm_targeted(&x, simple_logit_fn, -1.0, 1.0)
            .expect("value should be present");
        for (a, xi) in adv.iter().zip(x.iter()) {
            assert!(
                (a - xi).abs() <= eps + 1e-5,
                "delta too large: {} vs eps {}",
                (a - xi).abs(),
                eps
            );
        }
    }

    // ── 3. pgd_output_shape ──────────────────────────────────────────────────
    #[test]
    fn pgd_output_shape() {
        let x = vec![0.5_f32; 4];
        let attack = TargetedAttack::new(default_cfg(2)).expect("value should be present");
        let mut rng = make_rng();
        let adv = attack
            .pgd_targeted(&x, simple_logit_fn, -1.0, 1.0, &mut rng)
            .expect("value should be present");
        assert_eq!(adv.len(), x.len());
    }

    // ── 4. pgd_within_epsilon ────────────────────────────────────────────────
    #[test]
    fn pgd_within_epsilon() {
        let x = vec![0.5_f32, -0.2, 0.3, 0.1];
        let eps = 0.25_f32;
        let cfg = TargetedConfig {
            epsilon: eps,
            n_steps: 5,
            step_size: 0.05,
            target_class: 1,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let mut rng = make_rng();
        let adv = attack
            .pgd_targeted(&x, simple_logit_fn, -1.0, 1.0, &mut rng)
            .expect("value should be present");
        for (a, xi) in adv.iter().zip(x.iter()) {
            assert!(
                (a - xi).abs() <= eps + 1e-5,
                "delta={} > eps={}",
                (a - xi).abs(),
                eps
            );
        }
    }

    // ── 5. fgsm_increases_target_logit ───────────────────────────────────────
    #[test]
    fn fgsm_increases_target_logit() {
        // x = [0.5]; argmax is class 0.  We target class 1 (logit = -x[0]).
        // Gradient of logit[1] = -x[0] wrt x[0] is -1.0, so sign is -1.
        // FGSM-T should DECREASE x[0] ⇒ logit[1] = -x[0] INCREASES.
        let x = vec![0.5_f32, 0.0, 0.0, 0.0];
        let cfg = TargetedConfig {
            epsilon: 0.3,
            n_steps: 1,
            step_size: 0.3,
            target_class: 1,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let adv = attack
            .fgsm_targeted(&x, simple_logit_fn, -1.0, 1.0)
            .expect("value should be present");
        let orig_target_logit = simple_logit_fn(&x)[1];
        let adv_target_logit = simple_logit_fn(&adv)[1];
        assert!(
            adv_target_logit > orig_target_logit - 1e-5,
            "target logit should not decrease: {} vs {}",
            adv_target_logit,
            orig_target_logit
        );
    }

    // ── 6. step_size_0_no_change ─────────────────────────────────────────────
    #[test]
    fn step_size_0_no_change() {
        let x = vec![0.3_f32, -0.4, 0.2, 0.1];
        let cfg = TargetedConfig {
            epsilon: 0.2,
            n_steps: 1,
            step_size: 0.0,
            target_class: 1,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let adv = attack
            .fgsm_targeted(&x, simple_logit_fn, -1.0, 1.0)
            .expect("value should be present");
        // step_size = 0 → no movement; only box clamp (x is already in bounds).
        for (a, xi) in adv.iter().zip(x.iter()) {
            assert!((a - xi).abs() < 1e-6, "unexpected change: {} vs {}", a, xi);
        }
    }

    // ── 7. logit_fn_called ───────────────────────────────────────────────────
    #[test]
    fn logit_fn_called() {
        use std::cell::Cell;
        let call_count = Cell::new(0_usize);
        let x = vec![0.5_f32, 0.0, 0.0, 0.0];
        let cfg = TargetedConfig {
            epsilon: 0.1,
            n_steps: 1,
            step_size: 0.05,
            target_class: 0,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let _adv = attack
            .fgsm_targeted(
                &x,
                |xi| {
                    call_count.set(call_count.get() + 1);
                    simple_logit_fn(xi)
                },
                -1.0,
                1.0,
            )
            .expect("value should be present");
        // 1 probe call + 2·d calls (d=4) = 9 calls expected.
        assert!(
            call_count.get() >= 9,
            "expected at least 9 calls, got {}",
            call_count.get()
        );
    }

    // ── 8. epsilon_0_returns_x ───────────────────────────────────────────────
    #[test]
    fn epsilon_0_returns_x() {
        let x = vec![0.3_f32, -0.4, 0.2, 0.1];
        let cfg = TargetedConfig {
            epsilon: 0.0,
            n_steps: 1,
            step_size: 0.0,
            target_class: 1,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let adv = attack
            .fgsm_targeted(&x, simple_logit_fn, -1.0, 1.0)
            .expect("value should be present");
        for (a, xi) in adv.iter().zip(x.iter()) {
            assert!((a - xi).abs() < 1e-6);
        }
    }

    // ── 9. target_class_out_of_range_error ───────────────────────────────────
    #[test]
    fn target_class_out_of_range_error() {
        let x = vec![0.5_f32, 0.0, 0.0, 0.0];
        // simple_logit_fn returns 3 classes (0..2), so target=10 is out of range.
        let cfg = TargetedConfig {
            epsilon: 0.1,
            n_steps: 1,
            step_size: 0.05,
            target_class: 10,
        };
        let attack = TargetedAttack::new(cfg).expect("new should succeed");
        let result = attack.fgsm_targeted(&x, simple_logit_fn, -1.0, 1.0);
        assert!(
            result.is_err(),
            "expected an error for out-of-range target_class"
        );
        assert!(
            matches!(result.unwrap_err(), AdvError::DimensionMismatch { .. }),
            "expected DimensionMismatch"
        );
    }

    // ── 10. constructor_validates_epsilon ────────────────────────────────────
    #[test]
    fn constructor_validates_epsilon() {
        let bad = TargetedConfig {
            epsilon: -0.1,
            n_steps: 1,
            step_size: 0.05,
            target_class: 0,
        };
        assert!(matches!(
            TargetedAttack::new(bad).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
    }

    // ── 11. constructor_validates_n_steps ────────────────────────────────────
    #[test]
    fn constructor_validates_n_steps() {
        let bad = TargetedConfig {
            epsilon: 0.1,
            n_steps: 0,
            step_size: 0.05,
            target_class: 0,
        };
        assert!(matches!(
            TargetedAttack::new(bad).unwrap_err(),
            AdvError::InvalidNumSteps
        ));
    }

    // ── 12. pgd_deterministic ────────────────────────────────────────────────
    #[test]
    fn pgd_deterministic() {
        let x = vec![0.5_f32, -0.2, 0.3, 0.1];
        let attack = TargetedAttack::new(default_cfg(2)).expect("value should be present");
        let adv1 = attack
            .pgd_targeted(&x, simple_logit_fn, -1.0, 1.0, &mut LcgRng::new(7))
            .expect("value should be present");
        let adv2 = attack
            .pgd_targeted(&x, simple_logit_fn, -1.0, 1.0, &mut LcgRng::new(7))
            .expect("value should be present");
        for (a, b) in adv1.iter().zip(adv2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
