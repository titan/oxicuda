//! Carlini & Wagner (CW) L2 attack — simplified pure-CPU variant.
//!
//! Reference: Carlini & Wagner (2017), *"Towards Evaluating the Robustness of
//! Neural Networks"*, IEEE S&P.
//!
//! This module implements a deliberately *simplified* version of the original
//! CW-L2 attack, suitable for CPU evaluation harnesses and unit testing:
//!
//! * **No** trade-off binary search over `c`.
//! * **No** change-of-variable (`tanh`-reparameterisation).
//! * The attack loss minimised at every step is
//!
//!   ```text
//!   L(δ) = ‖δ‖²₂ + c · max(0, max_{j ≠ y} z_j(x + δ) − z_y(x + δ) + κ)
//!   ```
//!
//!   where `z` are the model logits and `κ` is the confidence margin.
//!
//! The user supplies a black-box closure returning **both** the logits at the
//! current iterate **and** the gradient of `L` with respect to `x`. The
//! optimiser is plain gradient descent with step size `lr`, projected back
//! onto the L2-ball of radius `eps` and box-clamped to `[lo, hi]` after every
//! step.

use crate::error::{AdvError, AdvResult};
use crate::threat_model::lp_ball::project_l2;

/// Hyperparameters for the simplified CW-L2 attack.
#[derive(Debug, Clone, Copy)]
pub struct CwConfig {
    /// Confidence weight on the misclassification term. Default `1.0`.
    pub c: f32,
    /// Confidence margin κ. Default `0.0` (decision boundary).
    pub kappa: f32,
    /// Gradient-descent step size. Default `0.01`.
    pub lr: f32,
    /// Number of gradient-descent steps (≥ 1).
    pub n_steps: usize,
    /// L2 budget used for projection after every update.
    pub eps: f32,
}

impl Default for CwConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kappa: 0.0,
            lr: 0.01,
            n_steps: 100,
            eps: 1.0,
        }
    }
}

impl CwConfig {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]    — non-finite or non-positive `eps`.
    /// * [`AdvError::InvalidAlpha`]      — non-finite or non-positive `lr`.
    /// * [`AdvError::InvalidNumSteps`]   — `n_steps == 0`.
    /// * [`AdvError::InvalidLossWeight`] — non-finite `c` or `kappa`.
    pub fn new(c: f32, kappa: f32, lr: f32, n_steps: usize, eps: f32) -> AdvResult<Self> {
        if !c.is_finite() {
            return Err(AdvError::InvalidLossWeight { weight: c });
        }
        if !kappa.is_finite() {
            return Err(AdvError::InvalidLossWeight { weight: kappa });
        }
        if !(lr.is_finite() && lr > 0.0) {
            return Err(AdvError::InvalidAlpha { alpha: lr });
        }
        if n_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps });
        }
        Ok(Self {
            c,
            kappa,
            lr,
            n_steps,
            eps,
        })
    }
}

/// Compute the CW attack loss `‖δ‖²₂ + c · max(0, f(z, y, κ))` from precomputed
/// logits `z`. Used by tests; exposed for documentation.
///
/// # Errors
/// * [`AdvError::DimensionMismatch`] — if `y_true >= z.len()`.
pub fn cw_loss_value(
    delta: &[f32],
    z: &[f32],
    y_true: usize,
    c: f32,
    kappa: f32,
) -> AdvResult<f32> {
    if y_true >= z.len() {
        return Err(AdvError::DimensionMismatch {
            expected: y_true + 1,
            got: z.len(),
        });
    }
    let zy = z[y_true];
    let mut max_other = f32::NEG_INFINITY;
    for (j, &v) in z.iter().enumerate() {
        if j != y_true && v > max_other {
            max_other = v;
        }
    }
    if !max_other.is_finite() {
        max_other = zy; // only one class — margin term is 0.
    }
    let margin = (max_other - zy + kappa).max(0.0);
    let l2_sq: f32 = delta.iter().map(|&d| d * d).sum();
    Ok(l2_sq + c * margin)
}

/// Run the simplified CW-L2 attack.
///
/// The closure `logits_grad(x_adv)` must return `(logits, ∇_x L_attack(x_adv))`
/// where `L_attack` is the CW objective described in the module docs.
/// Returning the logits is required so that we can both (a) detect a
/// misclassification (early-stop signal — currently unused, but reserved) and
/// (b) propagate downstream metrics that need the model output.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — empty `x`.
/// * [`AdvError::InvalidLossWeight`] — degenerate box.
/// * [`AdvError::DimensionMismatch`] — bad gradient or `y_true` out of range.
/// * [`AdvError::NanEncountered`]    — non-finite logits or gradients.
pub fn cw_attack<F>(
    x: &[f32],
    y_true: usize,
    lo: f32,
    hi: f32,
    cfg: &CwConfig,
    logits_grad: F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<(Vec<f32>, Vec<f32>)>,
{
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }

    let n = x.len();
    let mut adv = x.to_vec();

    for _ in 0..cfg.n_steps {
        let (logits, grad) = logits_grad(&adv)?;
        if y_true >= logits.len() {
            return Err(AdvError::DimensionMismatch {
                expected: y_true + 1,
                got: logits.len(),
            });
        }
        if grad.len() != n {
            return Err(AdvError::DimensionMismatch {
                expected: n,
                got: grad.len(),
            });
        }
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "cw_attack:logits",
            });
        }
        if grad.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "cw_attack:grad",
            });
        }

        // Plain gradient descent: x ← x − lr · ∇L (we minimise the CW loss).
        let stepped: Vec<f32> = adv
            .iter()
            .zip(grad.iter())
            .map(|(&xi, &gi)| xi - cfg.lr * gi)
            .collect();

        // Project back onto the L2 ε-ball around the original input and clamp.
        adv = project_l2(&stepped, x, cfg.eps, lo, hi)?;
    }
    Ok(adv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threat_model::lp_ball::l2_norm;

    /// Two-class linear classifier `z = (w·x, −w·x)` whose CW gradient is
    /// `2·δ + c · sign(margin) · (−2 w)` for `y_true = 0`.
    fn linear_two_class(
        w: Vec<f32>,
        x_orig: Vec<f32>,
        c: f32,
        kappa: f32,
    ) -> impl Fn(&[f32]) -> AdvResult<(Vec<f32>, Vec<f32>)> {
        move |x: &[f32]| {
            let dot: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let logits = vec![dot, -dot];
            // delta = x - x_orig
            let delta: Vec<f32> = x.iter().zip(x_orig.iter()).map(|(a, b)| a - b).collect();
            // margin = z_other - z_y + kappa = (-dot) - dot + kappa = -2*dot + kappa
            let margin = -2.0 * dot + kappa;
            let active = margin > 0.0;
            // d/dx (margin)_+ = -2 * w if active else 0
            // d/dx ‖δ‖² = 2 δ
            let g: Vec<f32> = delta
                .iter()
                .zip(w.iter())
                .map(|(&d, &wi)| 2.0 * d + if active { c * (-2.0 * wi) } else { 0.0 })
                .collect();
            Ok((logits, g))
        }
    }

    #[test]
    fn config_defaults_valid() {
        let c = CwConfig::default();
        assert!(c.lr > 0.0 && c.eps > 0.0 && c.n_steps > 0);
    }

    #[test]
    fn config_validation() {
        assert!(CwConfig::new(1.0, 0.0, 0.01, 10, 0.5).is_ok());
        assert!(CwConfig::new(f32::NAN, 0.0, 0.01, 10, 0.5).is_err());
        assert!(CwConfig::new(1.0, f32::NAN, 0.01, 10, 0.5).is_err());
        assert!(CwConfig::new(1.0, 0.0, 0.0, 10, 0.5).is_err());
        assert!(CwConfig::new(1.0, 0.0, 0.01, 0, 0.5).is_err());
        assert!(CwConfig::new(1.0, 0.0, 0.01, 10, -0.5).is_err());
    }

    #[test]
    fn cw_loss_basic() {
        let z = vec![1.0_f32, 0.0, 0.0];
        // y=0 is the largest ⇒ margin term = max(0, 0 − 1 + 0) = 0.
        let l = cw_loss_value(&[0.5_f32, -0.5], &z, 0, 1.0, 0.0).unwrap();
        assert!((l - 0.5).abs() < 1e-6); // only the ‖δ‖² term contributes.
    }

    #[test]
    fn cw_loss_misclassified_adds_margin() {
        let z = vec![0.0_f32, 1.0];
        let l = cw_loss_value(&[0.0_f32], &z, 0, 1.0, 0.0).unwrap();
        // δ=0, margin = max(0, 1 − 0 + 0) = 1, c = 1 ⇒ loss = 1.
        assert!((l - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smoke_drives_loss_down() {
        // Set up an *initially misclassified* input so the attack can reduce
        // the (positive) margin term:  y_true = 0 but x · w = −0.5  ⇒
        //   z = (−0.5, +0.5) ⇒ margin = z_1 − z_0 + κ = 1.0 > 0
        //   initial loss = ‖0‖² + c · 1.0 = c
        // The optimiser can drive δ in direction +w to flip the sign of
        // x·w and zero out the margin, paying a small ‖δ‖² penalty.
        let w = vec![1.0_f32, 0.0];
        let x = vec![-0.5_f32, 0.0];
        let cfg = CwConfig::new(10.0, 0.0, 0.05, 100, 5.0).unwrap();
        let cls = linear_two_class(w.clone(), x.clone(), cfg.c, cfg.kappa);
        let initial_loss = {
            let (z0, _) = cls(&x).unwrap();
            cw_loss_value(&[0.0; 2], &z0, 0, cfg.c, cfg.kappa).unwrap()
        };
        let y = cw_attack(&x, 0, -10.0, 10.0, &cfg, &cls).unwrap();
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let final_loss = {
            let (zf, _) = cls(&y).unwrap();
            cw_loss_value(&delta, &zf, 0, cfg.c, cfg.kappa).unwrap()
        };
        assert!(
            final_loss < initial_loss,
            "expected loss to decrease: initial={initial_loss}, final={final_loss}"
        );
    }

    #[test]
    fn projection_enforced() {
        let w = vec![1.0_f32; 8];
        let x = vec![0.5_f32; 8];
        let cfg = CwConfig::new(100.0, 0.0, 0.5, 100, 0.2).unwrap();
        let cls = linear_two_class(w, x.clone(), cfg.c, cfg.kappa);
        let y = cw_attack(&x, 0, 0.0, 1.0, &cfg, &cls).unwrap();
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l2_norm(&delta) <= 0.2 + 1e-4);
        for v in &y {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn invalid_y_true_errors() {
        let w = vec![1.0_f32];
        let x = vec![0.5_f32];
        let cfg = CwConfig::new(1.0, 0.0, 0.01, 1, 1.0).unwrap();
        let cls = linear_two_class(w, x.clone(), cfg.c, cfg.kappa);
        // logits has length 2 ⇒ y_true = 5 must error.
        assert!(matches!(
            cw_attack(&x, 5, -1.0, 1.0, &cfg, &cls).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn empty_input_rejected() {
        let x: Vec<f32> = vec![];
        let cfg = CwConfig::new(1.0, 0.0, 0.01, 1, 1.0).unwrap();
        let cls = |_x: &[f32]| Ok((vec![0.0_f32, 0.0], vec![]));
        assert_eq!(
            cw_attack(&x, 0, -1.0, 1.0, &cfg, cls).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn nan_in_logits_caught() {
        let x = vec![0.0_f32; 3];
        let cfg = CwConfig::new(1.0, 0.0, 0.01, 1, 1.0).unwrap();
        let cls = |_x: &[f32]| Ok((vec![f32::NAN, 0.0], vec![0.0_f32; 3]));
        assert!(matches!(
            cw_attack(&x, 0, -1.0, 1.0, &cfg, cls).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn nan_in_grad_caught() {
        let x = vec![0.0_f32; 3];
        let cfg = CwConfig::new(1.0, 0.0, 0.01, 1, 1.0).unwrap();
        let cls = |_x: &[f32]| Ok((vec![0.0_f32, 0.0], vec![1.0, f32::NAN, 1.0]));
        assert!(matches!(
            cw_attack(&x, 0, -1.0, 1.0, &cfg, cls).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn dim_mismatch_in_grad() {
        let x = vec![0.0_f32; 4];
        let cfg = CwConfig::new(1.0, 0.0, 0.01, 1, 1.0).unwrap();
        let cls = |_x: &[f32]| Ok((vec![0.0_f32, 0.0], vec![1.0_f32; 3]));
        assert!(matches!(
            cw_attack(&x, 0, -1.0, 1.0, &cfg, cls).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }
}
