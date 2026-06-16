//! Memory-Aware Synapses (MAS) regularization.
//!
//! Implements the method from:
//! Aljundi et al. "Memory Aware Synapses: Learning what (not) to forget."
//! ECCV 2018.
//!
//! MAS estimates parameter importance as the L1-norm of the gradient of the
//! output function (not the loss), then applies a EWC-style penalty:
//! `L_MAS = λ · Σ_i Ω_i · (θ_i - θ*_i)²`
//!
//! Importance is updated online via exponential moving average:
//! `Ω = momentum · Ω + (1 - momentum) · |∇L|`

use crate::error::{ContinualError, ContinualResult};

/// Configuration for MAS regularization.
#[derive(Debug, Clone)]
pub struct MasConfig {
    /// Regularization strength (λ). Must be ≥ 0 and finite.
    pub lambda: f32,
}

impl Default for MasConfig {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

impl MasConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> ContinualResult<()> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(ContinualError::InvalidLambda {
                lambda: self.lambda,
            });
        }
        Ok(())
    }
}

/// MAS importance weights, one per parameter.
#[derive(Debug, Clone)]
pub struct MasImportance {
    /// Per-parameter importance weights (Ω_i ≥ 0).
    pub omega: Vec<f32>,
}

impl MasImportance {
    /// Create zero-initialised importance weights.
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            omega: vec![0.0_f32; dim],
        }
    }

    /// Return the dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.omega.len()
    }
}

/// Update the MAS importance estimate via exponential moving average.
///
/// `Ω_i ← momentum · Ω_i + (1 - momentum) · |grad_i|`
///
/// `momentum` must be in `[0, 1]`. A typical value is 0.9.
pub fn mas_importance_update(
    omega: &mut [f32],
    gradient: &[f32],
    momentum: f32,
) -> ContinualResult<()> {
    if !(0.0..=1.0).contains(&momentum) || !momentum.is_finite() {
        return Err(ContinualError::InvalidMomentum { momentum });
    }
    if omega.len() != gradient.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: omega.len(),
            got: gradient.len(),
        });
    }
    let one_minus_m = 1.0 - momentum;
    for (w, &g) in omega.iter_mut().zip(gradient.iter()) {
        *w = momentum * (*w) + one_minus_m * g.abs();
    }
    Ok(())
}

/// Compute the MAS regularization penalty.
///
/// `penalty = λ · Σ_i Ω_i · (θ_i - θ*_i)²`
pub fn mas_penalty(
    params: &[f32],
    anchor: &[f32],
    importance: &MasImportance,
    cfg: &MasConfig,
) -> ContinualResult<f32> {
    cfg.validate()?;
    let d = params.len();
    if anchor.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: anchor.len(),
        });
    }
    if importance.omega.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: importance.omega.len(),
        });
    }
    let mut total = 0.0_f32;
    for i in 0..d {
        let delta = params[i] - anchor[i];
        total += importance.omega[i] * delta * delta;
    }
    let result = cfg.lambda * total;
    if !result.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "mas_penalty",
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mas_importance_tracks_gradient_magnitude() {
        let mut omega = vec![0.0_f32; 4];
        let grad = vec![2.0_f32, -3.0, 0.5, -1.5];
        // With momentum=0: omega = |grad|
        mas_importance_update(&mut omega, &grad, 0.0)
            .expect("MAS importance update should succeed with valid gradients");
        assert!((omega[0] - 2.0).abs() < 1e-6);
        assert!((omega[1] - 3.0).abs() < 1e-6);
        assert!((omega[2] - 0.5).abs() < 1e-6);
        assert!((omega[3] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn mas_importance_all_non_negative() {
        let mut omega = vec![0.5_f32; 8];
        let grad: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        mas_importance_update(&mut omega, &grad, 0.9)
            .expect("MAS importance update should succeed with valid gradients");
        for &w in &omega {
            assert!(w >= 0.0, "MAS omega must be non-negative, got {w}");
        }
    }

    #[test]
    fn mas_penalty_zero_at_anchor() {
        let params = vec![1.0_f32, 2.0, 3.0];
        let importance = MasImportance {
            omega: vec![1.0, 2.0, 3.0],
        };
        let cfg = MasConfig::default();
        let pen = mas_penalty(&params, &params, &importance, &cfg)
            .expect("MAS penalty should compute with matching dimensions");
        assert!(pen.abs() < 1e-6, "MAS penalty at anchor should be 0");
    }

    #[test]
    fn mas_penalty_grows_with_displacement() {
        let anchor = vec![0.0_f32; 4];
        let importance = MasImportance {
            omega: vec![1.0; 4],
        };
        let cfg = MasConfig::default();
        let small = vec![0.1_f32; 4];
        let large = vec![1.0_f32; 4];
        let pen_small = mas_penalty(&small, &anchor, &importance, &cfg)
            .expect("MAS penalty should compute with matching dimensions");
        let pen_large = mas_penalty(&large, &anchor, &importance, &cfg)
            .expect("MAS penalty should compute with matching dimensions");
        assert!(
            pen_large > pen_small,
            "MAS penalty should grow with displacement"
        );
    }

    #[test]
    fn mas_importance_ema_with_momentum() {
        let mut omega = vec![1.0_f32; 2];
        let grad = vec![0.0_f32; 2]; // all zero gradient
        // With momentum=0.9: omega = 0.9 * 1.0 + 0.1 * 0 = 0.9
        mas_importance_update(&mut omega, &grad, 0.9)
            .expect("MAS importance update should succeed with valid gradients");
        assert!((omega[0] - 0.9).abs() < 1e-6);
        assert!((omega[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn mas_invalid_momentum_returns_err() {
        let mut omega = vec![0.0_f32; 4];
        let grad = vec![1.0_f32; 4];
        assert!(mas_importance_update(&mut omega, &grad, 1.5).is_err());
        assert!(mas_importance_update(&mut omega, &grad, -0.1).is_err());
    }

    #[test]
    fn mas_penalty_scales_with_lambda() {
        let anchor = vec![0.0_f32; 4];
        let params = vec![1.0_f32; 4];
        let importance = MasImportance {
            omega: vec![1.0; 4],
        };
        let cfg1 = MasConfig { lambda: 1.0 };
        let cfg2 = MasConfig { lambda: 3.0 };
        let p1 = mas_penalty(&params, &anchor, &importance, &cfg1)
            .expect("MAS penalty should compute with matching dimensions");
        let p2 = mas_penalty(&params, &anchor, &importance, &cfg2)
            .expect("MAS penalty should compute with matching dimensions");
        assert!((p2 - 3.0 * p1).abs() < 1e-5);
    }

    #[test]
    fn mas_importance_zeros() {
        let imp = MasImportance::zeros(16);
        assert_eq!(imp.dim(), 16);
        assert!(imp.omega.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn mas_config_invalid_lambda() {
        let cfg = MasConfig { lambda: f32::NAN };
        assert!(cfg.validate().is_err());
    }
}
