//! ADOPT optimizer — Adapted Decoupled Proximal Theoretic optimizer.
//!
//! ADOPT (Taniguchi et al., 2024) is an adaptive gradient method that fixes the
//! convergence issue in Adam by normalising the gradient with the second moment
//! *before* computing the exponential moving average of the first moment.  This
//! decoupling removes the dependency of the update direction on the step size and
//! yields theoretically optimal non-convex convergence without hyperparameter
//! tuning beyond a reasonable initial learning rate.
//!
//! ## Algorithm
//!
//! ```text
//! t ← t + 1
//! v_t ← β₂·v_{t-1} + (1−β₂)·g²          // second moment (no bias-correct needed here)
//! m_t ← β₁·m_{t-1} + (1−β₁)·g            // first moment
//! m̂_t = m_t / (1 − β₁^t)                 // bias-corrected first moment
//! θ   ← θ − α · m̂_t / max(√v_t, ε)       // parameter update
//! ```

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Adopt`] optimizer.
#[derive(Debug, Clone)]
pub struct AdoptConfig {
    /// Learning rate (must be > 0).
    pub lr: f32,
    /// First-moment decay coefficient (default 0.9).
    pub beta1: f32,
    /// Second-moment decay coefficient (default 0.999).
    pub beta2: f32,
    /// Denominator epsilon for numerical stability (must be > 0).
    pub eps: f32,
}

impl Default for AdoptConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// ADOPT adaptive gradient optimizer.
///
/// Stores first-moment `m` and second-moment `v` state buffers on the host.
/// This implementation operates on flat `f32` slices and is suitable for
/// CPU-side parameter management or small auxiliary parameters.
pub struct Adopt {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    config: AdoptConfig,
}

impl Adopt {
    /// Create a new `Adopt` optimizer for parameters of size `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidLearningRate`] if `config.lr <= 0`.
    /// Returns [`TrainError::Internal`] if `config.eps <= 0`.
    pub fn new(dim: usize, config: AdoptConfig) -> TrainResult<Self> {
        if config.lr <= 0.0 {
            return Err(TrainError::InvalidLearningRate {
                lr: config.lr as f64,
            });
        }
        if config.eps <= 0.0 {
            return Err(TrainError::Internal {
                msg: format!("eps must be positive, got {}", config.eps),
            });
        }
        Ok(Self {
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            t: 0,
            config,
        })
    }

    /// Perform one optimizer step, updating `params` in-place with `grads`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::ParamCountMismatch`] if lengths differ.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> TrainResult<()> {
        if params.len() != grads.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: params.len(),
                got: grads.len(),
            });
        }
        if params.len() != self.m.len() {
            return Err(TrainError::ParamCountMismatch {
                expected: self.m.len(),
                got: params.len(),
            });
        }

        self.t += 1;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let bias_correction1 = 1.0 - b1.powi(self.t as i32);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let v_hat = self.v[i].max(eps);
            let m_hat = self.m[i] / bias_correction1;
            params[i] -= lr * m_hat / v_hat.sqrt();
        }
        Ok(())
    }

    /// Reset all optimizer state (moments and step counter).
    pub fn reset(&mut self) {
        for x in &mut self.m {
            *x = 0.0;
        }
        for x in &mut self.v {
            *x = 0.0;
        }
        self.t = 0;
    }

    /// Return current step count.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.t
    }

    /// Return a reference to the first-moment buffer.
    #[must_use]
    pub fn m(&self) -> &[f32] {
        &self.m
    }

    /// Return a reference to the second-moment buffer.
    #[must_use]
    pub fn v(&self) -> &[f32] {
        &self.v
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AdoptConfig {
        AdoptConfig {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Params must change after a gradient step with non-zero gradient.
    #[test]
    fn step_changes_params() {
        let mut opt = Adopt::new(4, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.1_f32; 4];
        opt.step(&mut params, &grads).expect("step should succeed");
        for &p in &params {
            assert!(
                p < 1.0,
                "params should decrease after positive gradient step"
            );
        }
    }

    /// Minimising f(x) = x² with gradient g = 2x over 200 steps should
    /// drive x toward 0.
    #[test]
    fn step_converges_quadratic() {
        let mut opt = Adopt::new(1, default_config()).expect("valid config");
        let mut params = vec![2.0_f32];
        for _ in 0..200 {
            let g = 2.0 * params[0];
            opt.step(&mut params, &[g]).expect("step should succeed");
        }
        assert!(
            params[0].abs() < 0.1,
            "optimizer should converge x² toward 0, got {}",
            params[0]
        );
    }

    /// After reset, internal state should be zeroed and step counter reset.
    #[test]
    fn reset_clears_state() {
        let mut opt = Adopt::new(2, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 2];
        opt.step(&mut params, &[0.5, 0.5]).expect("step ok");
        assert_eq!(opt.step_count(), 1);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
        for &v in opt.m() {
            assert_eq!(v, 0.0, "m should be zero after reset");
        }
        for &v in opt.v() {
            assert_eq!(v, 0.0, "v should be zero after reset");
        }
        // Verify that after reset the optimizer behaves like a fresh instance.
        let mut fresh_opt = Adopt::new(2, default_config()).expect("valid config");
        let mut params_fresh = vec![1.0_f32; 2];
        let mut params_reset = vec![1.0_f32; 2];
        fresh_opt
            .step(&mut params_fresh, &[0.5, 0.5])
            .expect("step ok");
        opt.step(&mut params_reset, &[0.5, 0.5])
            .expect("step ok after reset");
        for i in 0..2 {
            assert!(
                (params_fresh[i] - params_reset[i]).abs() < 1e-6,
                "reset optimizer should behave like fresh at index {i}"
            );
        }
    }

    /// Constructing with lr=0 must return an error.
    #[test]
    fn lr_zero_no_update() {
        let cfg = AdoptConfig {
            lr: 0.0,
            ..default_config()
        };
        let result = Adopt::new(4, cfg);
        assert!(
            matches!(result, Err(TrainError::InvalidLearningRate { .. })),
            "lr=0 should produce InvalidLearningRate"
        );
    }

    /// Calling step with a gradient slice of wrong length must error.
    #[test]
    fn dim_mismatch_error() {
        let mut opt = Adopt::new(4, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.1_f32; 3]; // wrong length
        let result = opt.step(&mut params, &grads);
        assert!(
            matches!(result, Err(TrainError::ParamCountMismatch { .. })),
            "length mismatch should produce ParamCountMismatch"
        );
    }

    /// When v is nearly zero, the eps floor prevents division by zero,
    /// so params remain finite.
    #[test]
    fn eps_positive_prevents_div_by_zero() {
        let cfg = AdoptConfig {
            lr: 1e-3,
            beta1: 0.0, // no momentum
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut opt = Adopt::new(1, cfg).expect("valid config");
        let mut params = vec![1.0_f32];
        // Very small gradient — v remains tiny
        for _ in 0..50 {
            opt.step(&mut params, &[1e-20_f32])
                .expect("step should succeed");
        }
        assert!(
            params[0].is_finite(),
            "params must remain finite even with tiny gradients"
        );
    }

    /// With beta1=0, the first moment equals g exactly (no exponential smoothing).
    #[test]
    fn beta1_zero_sgd_like() {
        let cfg = AdoptConfig {
            lr: 1e-2,
            beta1: 0.0,
            beta2: 0.999,
            eps: 1e-6,
        };
        let mut opt = Adopt::new(1, cfg).expect("valid config");
        let mut params_adopt = vec![1.0_f32];
        // With beta1=0: m_t = g, bias_correction = 1 (since (1 - 0^t) = 1 for t>=1)
        opt.step(&mut params_adopt, &[1.0_f32]).expect("step ok");
        // v after step 1 = (1 - beta2) * g^2 = 0.001 * 1 = 0.001
        // m_hat = 1.0 / 1.0 = 1.0
        // update = lr * 1.0 / sqrt(0.001) ≈ 0.01 / 0.03162 ≈ 0.316
        let expected_update = 1e-2_f32 / (0.001_f32).sqrt();
        let expected_param = 1.0 - expected_update;
        assert!(
            (params_adopt[0] - expected_param).abs() < 1e-5,
            "beta1=0 should make m_t=g exactly; expected {} got {}",
            expected_param,
            params_adopt[0]
        );
    }

    /// After many steps params must all be finite.
    #[test]
    fn params_finite() {
        let mut opt = Adopt::new(8, default_config()).expect("valid config");
        let mut params = vec![5.0_f32; 8];
        for _ in 0..500 {
            let grads: Vec<f32> = params.iter().map(|&p| 2.0 * p).collect();
            opt.step(&mut params, &grads).expect("step should succeed");
        }
        for &p in &params {
            assert!(p.is_finite(), "all params must remain finite");
        }
    }

    /// Second-moment entries v must always be non-negative.
    #[test]
    fn v_nonneg() {
        let mut opt = Adopt::new(4, default_config()).expect("valid config");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![-0.5_f32, 0.3, -1.0, 0.0];
        for _ in 0..20 {
            opt.step(&mut params, &grads).expect("step ok");
        }
        for (i, &v) in opt.v().iter().enumerate() {
            assert!(v >= 0.0, "v[{i}] must be non-negative, got {v}");
        }
    }

    /// Negative learning rate must produce an error.
    #[test]
    fn negative_lr_errors() {
        let cfg = AdoptConfig {
            lr: -0.001,
            ..default_config()
        };
        let result = Adopt::new(4, cfg);
        assert!(
            matches!(result, Err(TrainError::InvalidLearningRate { .. })),
            "negative lr should produce InvalidLearningRate"
        );
    }

    /// Negative eps must produce an error.
    #[test]
    fn negative_eps_errors() {
        let cfg = AdoptConfig {
            eps: -1e-8,
            ..default_config()
        };
        let result = Adopt::new(4, cfg);
        assert!(
            matches!(result, Err(TrainError::Internal { .. })),
            "negative eps should produce Internal error"
        );
    }
}
