//! Synaptic Intelligence (SI) regularization.
//!
//! Implements the method from:
//! Zenke et al. "Continual learning through synaptic intelligence."
//! ICML 2017.
//!
//! SI accumulates per-parameter importance weights online during training
//! based on the dot product of parameter change and gradient:
//! `Ω_i += |Δθ_i · ∇L_i|`
//!
//! The penalty at task switch is:
//! `λ · Σ_i Ω_i / (ΔΘ_i² + ξ) · (θ_i - θ*_i)²`

use crate::error::{ContinualError, ContinualResult};

/// Configuration for Synaptic Intelligence regularization.
#[derive(Debug, Clone)]
pub struct SiConfig {
    /// Regularization strength (λ). Must be ≥ 0 and finite.
    pub lambda: f32,
    /// Numerical stability constant (ξ) added to denominator. Typically 0.1.
    pub xi: f32,
}

impl Default for SiConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            xi: 0.1,
        }
    }
}

impl SiConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> ContinualResult<()> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(ContinualError::InvalidLambda {
                lambda: self.lambda,
            });
        }
        if !self.xi.is_finite() || self.xi <= 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: self.xi });
        }
        Ok(())
    }
}

/// SI running state accumulated during training on a single task.
#[derive(Debug, Clone)]
pub struct SiState {
    /// Accumulated importance estimate: `Ω_i = Σ |Δθ_i · grad_i|`.
    pub running_omega: Vec<f32>,
    /// Accumulated sum of gradient×delta terms (before abs).
    pub running_gradient_sum: Vec<f32>,
    /// Parameter values at the start of the current task (checkpoint).
    pub prev_params: Vec<f32>,
}

impl SiState {
    /// Initialise SI state from the current parameter snapshot.
    #[must_use]
    pub fn new(params: &[f32]) -> Self {
        let d = params.len();
        Self {
            running_omega: vec![0.0_f32; d],
            running_gradient_sum: vec![0.0_f32; d],
            prev_params: params.to_vec(),
        }
    }

    /// Reset for a new task: save current params as prev, zero accumulators.
    pub fn reset(&mut self, params: &[f32]) {
        let d = params.len();
        self.prev_params = params.to_vec();
        self.running_omega = vec![0.0_f32; d];
        self.running_gradient_sum = vec![0.0_f32; d];
    }
}

/// Update the SI importance accumulator for one gradient step.
///
/// `Ω_i += |Δθ_i · grad_i|` where `Δθ_i = params_i - prev_params_i`.
///
/// Call this after every gradient update step during training.
pub fn si_importance_update(
    state: &mut SiState,
    gradient: &[f32],
    params: &[f32],
) -> ContinualResult<()> {
    let d = state.prev_params.len();
    if gradient.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: gradient.len(),
        });
    }
    if params.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: params.len(),
        });
    }
    for i in 0..d {
        let delta = params[i] - state.prev_params[i];
        let contribution = delta * gradient[i];
        state.running_omega[i] += contribution.abs();
    }
    Ok(())
}

/// Compute the SI penalty for the current parameters against an anchor.
///
/// `penalty = λ · Σ_i [Ω_i / (ΔΘ_i² + ξ)] · (θ_i - θ*_i)²`
///
/// where `ΔΘ_i = (θ_i_at_task_end - θ_i_at_task_start)`.
///
/// `anchor`: parameter values at the task's beginning (θ*).
/// `omega`: importance weights accumulated during that task.
pub fn si_penalty(
    current_params: &[f32],
    anchor: &[f32],
    omega: &[f32],
    cfg: &SiConfig,
) -> ContinualResult<f32> {
    cfg.validate()?;
    let d = current_params.len();
    if anchor.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: anchor.len(),
        });
    }
    if omega.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: omega.len(),
        });
    }
    let mut penalty = 0.0_f32;
    for i in 0..d {
        let delta_theta_sq = (current_params[i] - anchor[i]).powi(2);
        // Normalise importance by total param movement + xi
        let denom = (current_params[i] - anchor[i]).powi(2) + cfg.xi;
        // importance_normed = Ω_i / (ΔΘ² + ξ)
        let importance = omega[i] / denom;
        penalty += importance * delta_theta_sq;
    }
    let result = cfg.lambda * penalty;
    if !result.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "si_penalty",
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_penalty_zero_at_anchor() {
        let params = vec![1.0_f32, 2.0, 3.0];
        let omega = vec![5.0_f32, 3.0, 1.0];
        let cfg = SiConfig::default();
        let pen = si_penalty(&params, &params, &omega, &cfg).unwrap();
        assert!(
            pen.abs() < 1e-6,
            "SI penalty should be 0 at anchor, got {pen}"
        );
    }

    #[test]
    fn si_penalty_positive_after_displacement() {
        let anchor = vec![0.0_f32; 4];
        let omega = vec![1.0_f32; 4];
        let current = vec![1.0_f32; 4];
        let cfg = SiConfig::default();
        let pen = si_penalty(&current, &anchor, &omega, &cfg).unwrap();
        assert!(pen > 0.0, "SI penalty should be > 0 after displacement");
    }

    #[test]
    fn si_penalty_monotone_with_lambda() {
        let anchor = vec![0.0_f32; 4];
        let omega = vec![1.0_f32; 4];
        let current = vec![1.0_f32; 4];
        let cfg1 = SiConfig {
            lambda: 1.0,
            xi: 0.1,
        };
        let cfg2 = SiConfig {
            lambda: 2.0,
            xi: 0.1,
        };
        let p1 = si_penalty(&current, &anchor, &omega, &cfg1).unwrap();
        let p2 = si_penalty(&current, &anchor, &omega, &cfg2).unwrap();
        assert!(p2 > p1, "Penalty should grow with lambda");
    }

    #[test]
    fn si_omega_non_negative_after_update() {
        let params_init = vec![0.0_f32; 4];
        let mut state = SiState::new(&params_init);
        // Positive gradient step
        let grad = vec![0.5_f32, -0.3, 0.1, -0.7];
        let params_after = vec![0.1_f32, -0.05, 0.02, -0.1];
        si_importance_update(&mut state, &grad, &params_after).unwrap();
        for &w in &state.running_omega {
            assert!(w >= 0.0, "SI omega must be non-negative, got {w}");
        }
    }

    #[test]
    fn si_state_reset_zeroes_accumulators() {
        let params = vec![1.0_f32; 4];
        let mut state = SiState::new(&params);
        state.running_omega = vec![5.0; 4];
        let new_params = vec![2.0_f32; 4];
        state.reset(&new_params);
        assert!(state.running_omega.iter().all(|&v| v == 0.0));
        assert_eq!(state.prev_params, new_params);
    }

    #[test]
    fn si_importance_update_known_value() {
        // delta = 1.0 - 0.0 = 1.0; grad = 2.0; contribution = |1.0 * 2.0| = 2.0
        let params_init = vec![0.0_f32];
        let mut state = SiState::new(&params_init);
        let grad = vec![2.0_f32];
        let params = vec![1.0_f32];
        si_importance_update(&mut state, &grad, &params).unwrap();
        assert!((state.running_omega[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn si_config_invalid_lambda() {
        let cfg = SiConfig {
            lambda: -1.0,
            xi: 0.1,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn si_penalty_dimension_mismatch_returns_err() {
        let current = vec![1.0_f32; 4];
        let anchor = vec![0.0_f32; 3]; // wrong dim
        let omega = vec![1.0_f32; 4];
        let cfg = SiConfig::default();
        assert!(si_penalty(&current, &anchor, &omega, &cfg).is_err());
    }
}
