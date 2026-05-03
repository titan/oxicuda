//! Bilevel optimisation for DARTS: inner weight SGD + outer arch-param Adam.

use crate::error::{NasError, NasResult};

// ─── BilevelConfig ───────────────────────────────────────────────────────────

/// Hyperparameters for the bilevel DARTS optimiser.
#[derive(Debug, Clone)]
pub struct BilevelConfig {
    /// Inner (weight update) learning rate.
    pub weight_lr: f32,
    /// Outer (architecture param update) learning rate.
    pub arch_lr: f32,
    /// L2 weight decay applied to architecture parameters.
    pub arch_weight_decay: f32,
    /// Number of inner SGD steps per outer iteration.
    pub n_inner_steps: usize,
}

impl Default for BilevelConfig {
    fn default() -> Self {
        Self {
            weight_lr: 0.025,
            arch_lr: 3e-4,
            arch_weight_decay: 1e-3,
            n_inner_steps: 1,
        }
    }
}

// ─── BilevelOptimizer ────────────────────────────────────────────────────────

/// Bilevel optimiser for DARTS: SGD on weights, Adam on arch params.
///
/// The inner optimiser takes one or more SGD steps using the training gradient.
/// The outer optimiser applies Adam to update the architecture parameters using
/// the validation gradient.
#[derive(Debug, Clone)]
pub struct BilevelOptimizer {
    /// Optimiser configuration.
    pub config: BilevelConfig,
    /// First moment (m) for Adam on arch params.
    pub arch_m: Vec<f32>,
    /// Second moment (v) for Adam on arch params.
    pub arch_v: Vec<f32>,
    /// Number of completed outer steps (used for Adam bias correction).
    pub step: usize,
}

impl BilevelOptimizer {
    /// Construct a new optimiser with zero Adam state.
    #[must_use]
    pub fn new(config: BilevelConfig, n_arch_params: usize) -> Self {
        Self {
            config,
            arch_m: vec![0.0_f32; n_arch_params],
            arch_v: vec![0.0_f32; n_arch_params],
            step: 0,
        }
    }

    /// Inner step: update `weights` by one SGD step with the given gradient.
    ///
    /// `weights -= weight_lr * grad` (applied `n_inner_steps` times with the same grad).
    pub fn inner_step(&self, weights: &mut [f32], grad: &[f32]) -> NasResult<()> {
        if weights.len() != grad.len() {
            return Err(NasError::DimensionMismatch {
                expected: weights.len(),
                got: grad.len(),
            });
        }
        let lr = self.config.weight_lr;
        for _ in 0..self.config.n_inner_steps {
            for (w, &g) in weights.iter_mut().zip(grad.iter()) {
                *w -= lr * g;
            }
        }
        Ok(())
    }

    /// Outer step: update `arch_params` using Adam with the validation gradient.
    ///
    /// Uses β₁ = 0.9, β₂ = 0.999, ε = 1e-8, plus L2 regularisation.
    pub fn outer_step(&mut self, arch_params: &mut [f32], arch_grad: &[f32]) -> NasResult<()> {
        let n = arch_params.len();
        if arch_grad.len() != n {
            return Err(NasError::DimensionMismatch {
                expected: n,
                got: arch_grad.len(),
            });
        }
        if self.arch_m.len() != n || self.arch_v.len() != n {
            return Err(NasError::InvalidWeightShape);
        }

        self.step += 1;
        let t = self.step as f32;
        let lr = self.config.arch_lr;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let wd = self.config.arch_weight_decay;

        // bias correction denominators
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for i in 0..n {
            // gradient with L2 regularisation
            let g = arch_grad[i] + wd * arch_params[i];

            // update moments
            self.arch_m[i] = beta1 * self.arch_m[i] + (1.0 - beta1) * g;
            self.arch_v[i] = beta2 * self.arch_v[i] + (1.0 - beta2) * g * g;

            // bias-corrected moments
            let m_hat = self.arch_m[i] / bc1;
            let v_hat = self.arch_v[i] / bc2;

            arch_params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inner_step_moves_in_gradient_direction() {
        let config = BilevelConfig {
            weight_lr: 0.1,
            n_inner_steps: 1,
            ..Default::default()
        };
        let opt = BilevelOptimizer::new(config, 4);
        let mut w = vec![1.0_f32; 4];
        let grad = vec![1.0_f32; 4];
        opt.inner_step(&mut w, &grad)
            .expect("test invariant: inner step");
        assert!((w[0] - 0.9).abs() < 1e-6, "w[0] = {}", w[0]);
    }

    #[test]
    fn outer_step_changes_arch_params() {
        let config = BilevelConfig::default();
        let mut opt = BilevelOptimizer::new(config, 4);
        let mut arch = vec![0.0_f32; 4];
        let grad = vec![1.0_f32; 4];
        opt.outer_step(&mut arch, &grad)
            .expect("test invariant: outer step");
        // arch params must have changed
        assert!(arch.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn size_mismatch_errors() {
        let opt = BilevelOptimizer::new(BilevelConfig::default(), 4);
        let mut w = vec![0.0_f32; 4];
        let grad_bad = vec![0.0_f32; 3];
        assert!(opt.inner_step(&mut w, &grad_bad).is_err());
    }
}
