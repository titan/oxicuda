//! Fast Gradient Sign Method (FGSM).
//!
//! Single-step L∞ adversarial attack from
//! Goodfellow, Shlens & Szegedy (2015), *"Explaining and Harnessing
//! Adversarial Examples"*, ICLR.
//!
//! Given an input `x`, a per-coordinate budget `eps`, a box constraint
//! `[lo, hi]`, and a black-box loss-gradient closure, FGSM computes
//!
//! ```text
//! x_adv = clamp(x + eps * sign(∇_x L(x)), lo, hi)
//! ```
//!
//! It is the cheapest baseline attack and the building block for PGD,
//! MIM and AutoPGD. For reproducible evaluation it is usually combined
//! with random initialisation (see [`crate::attacks::pgd`]).

use crate::error::{AdvError, AdvResult};

/// Run a single FGSM step.
///
/// # Parameters
/// * `x`         — original input tensor (flattened).
/// * `eps`       — per-coordinate L∞ budget (must be `>= 0` and finite).
/// * `lo`, `hi`  — box constraint applied after the gradient step
///   (must satisfy `lo < hi` and both finite).
/// * `loss_grad` — closure returning `∇_x L(x)` of identical shape to `x`.
///
/// # Returns
/// A fresh `Vec<f32>` containing `clamp(x + eps * sign(∇L(x)), lo, hi)`.
///
/// # Errors
/// * [`AdvError::EmptyInput`]        — if `x.is_empty()`.
/// * [`AdvError::InvalidEpsilon`]    — if `eps` is non-finite or negative.
/// * [`AdvError::InvalidLossWeight`] — if the box `[lo, hi]` is degenerate.
/// * [`AdvError::DimensionMismatch`] — if `loss_grad` returns a vector of the
///   wrong length.
/// * [`AdvError::NanEncountered`]    — if any returned gradient entry is
///   non-finite.
pub fn fgsm_attack<F>(x: &[f32], eps: f32, lo: f32, hi: f32, loss_grad: F) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(AdvError::InvalidEpsilon { eps });
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }

    let grad = loss_grad(x)?;
    if grad.len() != x.len() {
        return Err(AdvError::DimensionMismatch {
            expected: x.len(),
            got: grad.len(),
        });
    }
    if grad.iter().any(|g| !g.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "fgsm_attack:loss_grad",
        });
    }

    Ok(x.iter()
        .zip(grad.iter())
        .map(|(&xi, &gi)| {
            let s = if gi > 0.0 {
                1.0_f32
            } else if gi < 0.0 {
                -1.0_f32
            } else {
                0.0_f32
            };
            (xi + eps * s).clamp(lo, hi)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic loss with gradient equal to a fixed reference.
    fn const_grad(g: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |_x: &[f32]| Ok(g.clone())
    }

    /// Quadratic loss `½‖x − target‖²` with gradient `x − target`.
    fn quad_grad(target: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| {
            Ok(x.iter()
                .zip(target.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>())
        }
    }

    #[test]
    fn smoke_basic_sign_step() {
        let x = vec![0.5_f32, 0.5, 0.5];
        let g = vec![1.0_f32, -1.0, 0.0];
        let y = fgsm_attack(&x, 0.1, 0.0, 1.0, const_grad(g)).expect("value should be present");
        assert!((y[0] - 0.6).abs() < 1e-6);
        assert!((y[1] - 0.4).abs() < 1e-6);
        assert!((y[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zero_eps_returns_clamped_input() {
        let x = vec![0.2_f32, 1.5, -0.3];
        let g = vec![1.0_f32, -1.0, 0.0];
        let y = fgsm_attack(&x, 0.0, 0.0, 1.0, const_grad(g)).expect("value should be present");
        assert!((y[0] - 0.2).abs() < 1e-6);
        assert!((y[1] - 1.0).abs() < 1e-6);
        assert!((y[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sign_correctness_increases_quadratic_loss() {
        // Loss = ½‖x − target‖² → grad = x − target. FGSM moves x AWAY
        // from `target`, so the loss must strictly increase.
        let target = vec![0.5_f32; 4];
        let x = vec![0.6_f32, 0.4, 0.7, 0.3];
        let baseline_loss: f32 = x
            .iter()
            .zip(target.iter())
            .map(|(a, b)| 0.5 * (a - b).powi(2))
            .sum();
        let y = fgsm_attack(&x, 0.05, -10.0, 10.0, quad_grad(target.clone()))
            .expect("value should be present");
        let new_loss: f32 = y
            .iter()
            .zip(target.iter())
            .map(|(a, b)| 0.5 * (a - b).powi(2))
            .sum();
        assert!(new_loss > baseline_loss);
    }

    #[test]
    fn budget_respected_per_coordinate() {
        let x = vec![0.5_f32; 8];
        let g = vec![2.5_f32; 8]; // magnitude irrelevant — sign only
        let y = fgsm_attack(&x, 0.07, -10.0, 10.0, const_grad(g)).expect("value should be present");
        for v in &y {
            assert!((*v - 0.57).abs() < 1e-5);
        }
    }

    #[test]
    fn dim_mismatch_in_grad_propagates() {
        let x = vec![0.0_f32; 4];
        let bad = const_grad(vec![1.0_f32; 3]);
        let err = fgsm_attack(&x, 0.1, 0.0, 1.0, bad).unwrap_err();
        assert_eq!(
            err,
            AdvError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        );
    }

    #[test]
    fn rejects_invalid_eps() {
        let x = vec![0.0_f32; 3];
        assert!(matches!(
            fgsm_attack(&x, -0.1, 0.0, 1.0, const_grad(vec![1.0; 3])).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
        assert!(matches!(
            fgsm_attack(&x, f32::NAN, 0.0, 1.0, const_grad(vec![1.0; 3])).unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
    }

    #[test]
    fn rejects_degenerate_box() {
        let x = vec![0.0_f32; 3];
        assert!(fgsm_attack(&x, 0.1, 1.0, 1.0, const_grad(vec![1.0; 3])).is_err());
        assert!(fgsm_attack(&x, 0.1, 1.0, 0.5, const_grad(vec![1.0; 3])).is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let x: Vec<f32> = vec![];
        assert_eq!(
            fgsm_attack(&x, 0.1, 0.0, 1.0, const_grad(vec![])).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn nan_in_gradient_is_caught() {
        let x = vec![0.0_f32; 3];
        let bad = const_grad(vec![1.0, f32::NAN, 1.0]);
        assert!(matches!(
            fgsm_attack(&x, 0.1, -1.0, 1.0, bad).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }
}
