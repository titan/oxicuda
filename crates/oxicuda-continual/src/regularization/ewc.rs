//! Elastic Weight Consolidation (EWC) regularization.
//!
//! Implements the method from:
//! Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks."
//! PNAS 2017.
//!
//! EWC penalises changes to parameters that were important for previous tasks,
//! weighted by the diagonal Fisher information matrix:
//! `L_EWC = λ/2 · Σ_t Σ_i F_i^t · (θ_i - θ_i^{*t})²`

use crate::error::{ContinualError, ContinualResult};

/// Configuration for EWC regularization.
#[derive(Debug, Clone)]
pub struct EwcConfig {
    /// Regularization strength (λ). Must be ≥ 0 and finite.
    pub lambda: f32,
    /// Maximum number of tasks to retain anchors for.
    pub n_tasks: usize,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            n_tasks: 10,
        }
    }
}

impl EwcConfig {
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

/// Diagonal Fisher information matrix estimate per parameter.
#[derive(Debug, Clone)]
pub struct FisherDiag {
    /// Per-parameter Fisher diagonal entries. Non-negative.
    pub params: Vec<f32>,
}

impl FisherDiag {
    /// Create a zero-initialised Fisher of the given dimension.
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            params: vec![0.0_f32; dim],
        }
    }

    /// Return the dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.params.len()
    }
}

/// EWC regularizer holding one anchor + Fisher per completed task.
#[derive(Debug, Clone, Default)]
pub struct EwcRegularizer {
    /// Parameter snapshots at each task's end.
    pub anchors: Vec<Vec<f32>>,
    /// Diagonal Fisher matrices, one per completed task.
    pub fishers: Vec<FisherDiag>,
}

impl EwcRegularizer {
    /// Create an empty regularizer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of tasks registered so far.
    #[must_use]
    pub fn n_tasks(&self) -> usize {
        self.anchors.len()
    }
}

/// Compute the empirical diagonal Fisher:
/// `F_i = (1/N) · Σ_{n=1}^{N} g_{n,i}²`
///
/// `gradients` contains the concatenated per-sample gradients stacked row-major:
/// length must be `n_samples * D` where `D` is the parameter count.
///
/// Returns `Err(ContinualError::EmptyInput)` if `gradients` is empty or
/// `n_samples == 0`.
pub fn compute_fisher_empirical(
    gradients: &[f32],
    n_samples: usize,
) -> ContinualResult<FisherDiag> {
    if gradients.is_empty() || n_samples == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if gradients.len() % n_samples != 0 {
        return Err(ContinualError::DimensionMismatch {
            expected: gradients.len() / n_samples * n_samples,
            got: gradients.len(),
        });
    }
    let dim = gradients.len() / n_samples;
    let inv_n = 1.0_f32 / n_samples as f32;
    let mut fisher = vec![0.0_f32; dim];
    for sample in 0..n_samples {
        let offset = sample * dim;
        for (i, f) in fisher.iter_mut().enumerate() {
            let g = gradients[offset + i];
            *f += g * g;
        }
    }
    for f in &mut fisher {
        *f *= inv_n;
    }
    Ok(FisherDiag { params: fisher })
}

/// Compute the total EWC loss for `current_params` against all registered tasks.
///
/// `L = λ/2 · Σ_t Σ_i F_i^t · (θ_i - θ_i^{*t})²`
///
/// Returns 0.0 if no tasks are registered yet.
pub fn ewc_loss(
    current_params: &[f32],
    reg: &EwcRegularizer,
    cfg: &EwcConfig,
) -> ContinualResult<f32> {
    cfg.validate()?;
    if reg.anchors.is_empty() {
        return Ok(0.0_f32);
    }
    let d = current_params.len();
    let mut total = 0.0_f32;
    for (anchor, fisher) in reg.anchors.iter().zip(reg.fishers.iter()) {
        if anchor.len() != d {
            return Err(ContinualError::DimensionMismatch {
                expected: d,
                got: anchor.len(),
            });
        }
        if fisher.params.len() != d {
            return Err(ContinualError::DimensionMismatch {
                expected: d,
                got: fisher.params.len(),
            });
        }
        for i in 0..d {
            let delta = current_params[i] - anchor[i];
            total += fisher.params[i] * delta * delta;
        }
    }
    let loss = 0.5 * cfg.lambda * total;
    if !loss.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "ewc_loss",
        });
    }
    Ok(loss)
}

/// Register a completed task into the regularizer.
///
/// Appends the parameter snapshot and Fisher diagonal to the regularizer.
pub fn add_task(reg: &mut EwcRegularizer, params: Vec<f32>, fisher: FisherDiag) {
    reg.anchors.push(params);
    reg.fishers.push(fisher);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewc_loss_zero_at_anchor() {
        let params = vec![1.0_f32, 2.0, 3.0, 4.0];
        let fisher = FisherDiag {
            params: vec![1.0, 2.0, 3.0, 4.0],
        };
        let mut reg = EwcRegularizer::new();
        add_task(&mut reg, params.clone(), fisher);
        let cfg = EwcConfig::default();
        let loss = ewc_loss(&params, &reg, &cfg)
            .expect("EWC loss should compute at anchor with valid params and reg");
        assert!(
            loss.abs() < 1e-6,
            "EWC loss should be 0 at anchor, got {loss}"
        );
    }

    #[test]
    fn ewc_loss_positive_after_perturbation() {
        let anchor = vec![1.0_f32, 2.0, 3.0, 4.0];
        let fisher = FisherDiag {
            params: vec![1.0, 1.0, 1.0, 1.0],
        };
        let mut reg = EwcRegularizer::new();
        add_task(&mut reg, anchor, fisher);
        let perturbed = vec![2.0_f32, 3.0, 4.0, 5.0];
        let cfg = EwcConfig::default();
        let loss = ewc_loss(&perturbed, &reg, &cfg)
            .expect("EWC loss should compute with valid params and reg");
        assert!(loss > 0.0, "EWC loss should be > 0 after perturbation");
    }

    #[test]
    fn ewc_loss_scales_with_lambda() {
        let anchor = vec![0.0_f32; 8];
        let fisher = FisherDiag {
            params: vec![1.0_f32; 8],
        };
        let mut reg = EwcRegularizer::new();
        add_task(&mut reg, anchor, fisher);
        let params = vec![1.0_f32; 8];
        let cfg1 = EwcConfig {
            lambda: 1.0,
            n_tasks: 5,
        };
        let cfg2 = EwcConfig {
            lambda: 2.0,
            n_tasks: 5,
        };
        let l1 = ewc_loss(&params, &reg, &cfg1).expect("EWC loss should compute with lambda=1.0");
        let l2 = ewc_loss(&params, &reg, &cfg2).expect("EWC loss should compute with lambda=2.0");
        assert!(
            (l2 - 2.0 * l1).abs() < 1e-5,
            "EWC loss should scale linearly with lambda"
        );
    }

    #[test]
    fn fisher_entries_non_negative() {
        let grads = vec![1.0_f32, -2.0, 0.5, -0.5, 3.0, -1.5, 0.0, 0.0];
        let n_samples = 2;
        let fisher = compute_fisher_empirical(&grads, n_samples)
            .expect("Fisher matrix should compute from valid gradients");
        for &f in &fisher.params {
            assert!(f >= 0.0, "Fisher entry must be non-negative, got {f}");
        }
    }

    #[test]
    fn fisher_empirical_known_values() {
        // Single sample: F_i = g_i^2
        let grads = vec![2.0_f32, 3.0];
        let fisher = compute_fisher_empirical(&grads, 1)
            .expect("Fisher matrix should compute from single-sample gradients");
        assert!((fisher.params[0] - 4.0).abs() < 1e-6);
        assert!((fisher.params[1] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn fisher_empirical_two_samples_averages() {
        // F_i = (g1^2 + g2^2) / 2
        let grads = vec![2.0_f32, 0.0, 0.0, 4.0];
        let fisher = compute_fisher_empirical(&grads, 2)
            .expect("Fisher matrix should compute from two-sample gradients and average correctly");
        // param 0: (4 + 0) / 2 = 2.0
        // param 1: (0 + 16) / 2 = 8.0
        assert!((fisher.params[0] - 2.0).abs() < 1e-6);
        assert!((fisher.params[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn ewc_loss_multi_task_accumulates() {
        let anchor1 = vec![0.0_f32; 4];
        let fisher1 = FisherDiag {
            params: vec![1.0; 4],
        };
        let anchor2 = vec![1.0_f32; 4];
        let fisher2 = FisherDiag {
            params: vec![1.0; 4],
        };
        let mut reg = EwcRegularizer::new();
        add_task(&mut reg, anchor1, fisher1);
        add_task(&mut reg, anchor2, fisher2);
        assert_eq!(reg.n_tasks(), 2);
        let params = vec![0.5_f32; 4];
        let cfg = EwcConfig {
            lambda: 1.0,
            n_tasks: 10,
        };
        let loss = ewc_loss(&params, &reg, &cfg)
            .expect("EWC loss should compute with two registered tasks");
        // Task1: 0.5*1.0*(0.5-0)^2 * 4 = 0.5
        // Task2: 0.5*1.0*(0.5-1)^2 * 4 = 0.5
        // Total ≈ 1.0
        assert!((loss - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ewc_loss_empty_regularizer_is_zero() {
        let reg = EwcRegularizer::new();
        let params = vec![1.0_f32, 2.0, 3.0];
        let cfg = EwcConfig::default();
        let loss = ewc_loss(&params, &reg, &cfg)
            .expect("EWC loss should compute with empty regularizer and return zero");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn compute_fisher_empty_returns_err() {
        let result = compute_fisher_empirical(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn ewc_config_invalid_lambda() {
        let cfg = EwcConfig {
            lambda: -1.0,
            n_tasks: 5,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn fisher_diag_zeros() {
        let f = FisherDiag::zeros(16);
        assert_eq!(f.dim(), 16);
        assert!(f.params.iter().all(|&v| v == 0.0));
    }
}
