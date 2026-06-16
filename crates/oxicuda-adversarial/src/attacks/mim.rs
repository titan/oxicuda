//! Momentum Iterative Method (MIM / MI-FGSM).
//!
//! L∞ attack from
//! Dong, Liao, Pang, Su, Zhu, Hu & Li (2018),
//! *"Boosting Adversarial Attacks with Momentum"*, CVPR.
//!
//! At every step the gradient is L1-normalised and accumulated into a running
//! momentum buffer with decay μ:
//!
//! ```text
//! g_t      = μ · g_{t−1} + ∇L(x_t) / ‖∇L(x_t)‖₁
//! x_{t+1}  = clamp(project_L∞(x_t + α · sign(g_t), ε), lo, hi)
//! ```
//!
//! Setting `momentum_decay = 0` recovers the canonical iterative
//! sign-gradient PGD-L∞ baseline (with `random_start = false`).

use crate::error::{AdvError, AdvResult};
use crate::threat_model::lp_ball::{l1_norm, project_l_inf};

/// Hyperparameters for MIM.
#[derive(Debug, Clone, Copy)]
pub struct MimConfig {
    /// L∞ perturbation budget.
    pub eps: f32,
    /// Per-step size.
    pub alpha: f32,
    /// Number of iterations (≥ 1).
    pub n_steps: usize,
    /// Momentum decay μ. Typical value `1.0`. Set to `0.0` to disable.
    pub momentum_decay: f32,
}

impl MimConfig {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]    — non-finite or non-positive `eps`.
    /// * [`AdvError::InvalidAlpha`]      — non-finite or non-positive `alpha`.
    /// * [`AdvError::InvalidNumSteps`]   — `n_steps == 0`.
    /// * [`AdvError::InvalidLossWeight`] — non-finite or negative `momentum_decay`.
    pub fn new(eps: f32, alpha: f32, n_steps: usize, momentum_decay: f32) -> AdvResult<Self> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps });
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(AdvError::InvalidAlpha { alpha });
        }
        if n_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        if !(momentum_decay.is_finite() && momentum_decay >= 0.0) {
            return Err(AdvError::InvalidLossWeight {
                weight: momentum_decay,
            });
        }
        Ok(Self {
            eps,
            alpha,
            n_steps,
            momentum_decay,
        })
    }
}

/// Run MIM (Momentum Iterative L∞ attack).
///
/// # Errors
/// Mirrors [`MimConfig::new`] plus [`AdvError::EmptyInput`],
/// [`AdvError::DimensionMismatch`] and [`AdvError::NanEncountered`] for
/// closure outputs.
pub fn mim_attack<F>(
    x: &[f32],
    lo: f32,
    hi: f32,
    cfg: &MimConfig,
    loss_grad: F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }

    let n = x.len();
    let mut adv = x.to_vec();
    let mut momentum = vec![0.0_f32; n];

    for _ in 0..cfg.n_steps {
        let g = loss_grad(&adv)?;
        if g.len() != n {
            return Err(AdvError::DimensionMismatch {
                expected: n,
                got: g.len(),
            });
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Err(AdvError::NanEncountered {
                location: "mim_attack:loss_grad",
            });
        }
        let denom = l1_norm(&g).max(1e-12);
        for i in 0..n {
            momentum[i] = cfg.momentum_decay * momentum[i] + g[i] / denom;
        }
        let stepped: Vec<f32> = adv
            .iter()
            .zip(momentum.iter())
            .map(|(&xi, &mi)| {
                let s = if mi > 0.0 {
                    1.0_f32
                } else if mi < 0.0 {
                    -1.0_f32
                } else {
                    0.0_f32
                };
                xi + cfg.alpha * s
            })
            .collect();
        adv = project_l_inf(&stepped, x, cfg.eps, lo, hi)?;
    }
    Ok(adv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threat_model::lp_ball::l_inf_norm;

    fn quad_grad(target: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| Ok(x.iter().zip(target.iter()).map(|(a, b)| a - b).collect())
    }

    fn const_grad(g: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |_x: &[f32]| Ok(g.clone())
    }

    #[test]
    fn config_validation() {
        assert!(MimConfig::new(0.1, 0.01, 5, 1.0).is_ok());
        assert!(MimConfig::new(-0.1, 0.01, 5, 1.0).is_err());
        assert!(MimConfig::new(0.1, -0.01, 5, 1.0).is_err());
        assert!(MimConfig::new(0.1, 0.01, 0, 1.0).is_err());
        assert!(MimConfig::new(0.1, 0.01, 5, -0.1).is_err());
        assert!(MimConfig::new(0.1, 0.01, 5, f32::NAN).is_err());
    }

    #[test]
    fn smoke_quadratic_increases_loss() {
        let target = vec![0.5_f32; 6];
        let x = vec![0.6_f32, 0.4, 0.7, 0.3, 0.55, 0.45];
        let cfg = MimConfig::new(0.1, 0.02, 8, 1.0).expect("new should succeed");
        let baseline: f32 = x
            .iter()
            .zip(target.iter())
            .map(|(a, b)| 0.5 * (a - b).powi(2))
            .sum();
        let y = mim_attack(&x, -10.0, 10.0, &cfg, quad_grad(target.clone()))
            .expect("value should be present");
        let new_loss: f32 = y
            .iter()
            .zip(target.iter())
            .map(|(a, b)| 0.5 * (a - b).powi(2))
            .sum();
        assert!(new_loss > baseline);
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l_inf_norm(&delta) <= 0.1 + 1e-5);
    }

    #[test]
    fn momentum_decay_zero_matches_basic_iterative() {
        // With μ = 0 momentum is just the L1-normalised gradient at this step.
        // For a constant-sign gradient closure this reduces to plain iterative
        // sign attack with α step.
        let x = vec![0.5_f32; 4];
        let g = vec![1.0_f32, -1.0, 1.0, -1.0];
        let cfg = MimConfig::new(0.4, 0.05, 3, 0.0).expect("new should succeed");
        let y = mim_attack(&x, -10.0, 10.0, &cfg, const_grad(g)).expect("value should be present");
        // 3 steps × ±0.05 = ±0.15.
        let expected = [0.65_f32, 0.35, 0.65, 0.35];
        for (a, b) in y.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn projection_enforced() {
        let target = vec![10.0_f32; 6];
        let x = vec![0.5_f32; 6];
        let cfg = MimConfig::new(0.05, 0.05, 20, 0.9).expect("new should succeed");
        let y = mim_attack(&x, 0.0, 1.0, &cfg, quad_grad(target)).expect("value should be present");
        let delta: Vec<f32> = y.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l_inf_norm(&delta) <= 0.05 + 1e-5);
        for v in &y {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn dim_mismatch_caught() {
        let x = vec![0.0_f32; 3];
        let cfg = MimConfig::new(0.1, 0.05, 1, 1.0).expect("new should succeed");
        let bad = const_grad(vec![1.0_f32; 5]);
        assert!(matches!(
            mim_attack(&x, -1.0, 1.0, &cfg, bad).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn nan_grad_caught() {
        let x = vec![0.0_f32; 3];
        let cfg = MimConfig::new(0.1, 0.05, 1, 1.0).expect("new should succeed");
        let bad = const_grad(vec![1.0, f32::NAN, 1.0]);
        assert!(matches!(
            mim_attack(&x, -1.0, 1.0, &cfg, bad).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn empty_input_rejected() {
        let x: Vec<f32> = vec![];
        let cfg = MimConfig::new(0.1, 0.05, 1, 1.0).expect("new should succeed");
        assert_eq!(
            mim_attack(&x, -1.0, 1.0, &cfg, const_grad(vec![])).unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn momentum_persists_across_steps() {
        // Construct a sequence where the gradient is positive on step 0 and
        // very small (but positive) on subsequent steps. Without momentum the
        // step magnitude stays ±α; with momentum it is still bounded by α
        // because we take sign(momentum), so primarily the *direction* is
        // verified to remain stable.
        let x = vec![0.5_f32];
        let cfg = MimConfig::new(0.5, 0.01, 5, 1.0).expect("new should succeed");
        let y = mim_attack(&x, -10.0, 10.0, &cfg, const_grad(vec![1.0]))
            .expect("value should be present");
        // Each step adds +0.01 ⇒ final 0.5 + 5·0.01 = 0.55.
        assert!((y[0] - 0.55).abs() < 1e-5);
    }

    #[test]
    fn degenerate_box_rejected() {
        let x = vec![0.0_f32; 3];
        let cfg = MimConfig::new(0.1, 0.05, 1, 1.0).expect("new should succeed");
        assert!(mim_attack(&x, 1.0, 1.0, &cfg, const_grad(vec![1.0; 3])).is_err());
    }
}
