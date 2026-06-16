//! AdaBelief optimizer — Zhuang et al., 2020.
//!
//! "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
//! Gradients", NeurIPS 2020.
//!
//! AdaBelief is a drop-in modification of Adam.  Where Adam adapts the step
//! size using the second raw moment of the gradient `v_t = EMA(g²)`, AdaBelief
//! instead tracks the EMA of the squared deviation of the gradient from its
//! own first-moment estimate — the *belief* in the gradient direction:
//!
//! ```text
//! t   ← t + 1
//! m_t ← β₁·m_{t-1} + (1−β₁)·g_t                       // first moment
//! s_t ← β₂·s_{t-1} + (1−β₂)·(g_t − m_t)² + ε          // belief / variance
//! m̂_t = m_t / (1 − β₁^t)
//! ŝ_t = s_t / (1 − β₂^t)
//! θ   ← θ − α · m̂_t / (√ŝ_t + ε)
//! ```
//!
//! When the observed gradient agrees with the trend (`g_t ≈ m_t`) `s_t` is
//! small, yielding a large step; when it deviates, `s_t` grows and the step
//! shrinks.  This gives Adam-like fast convergence with SGD-like
//! generalisation.

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`AdaBelief`] optimizer.
#[derive(Debug, Clone)]
pub struct AdaBeliefConfig {
    /// Learning rate (must be > 0).
    pub lr: f32,
    /// First-moment decay coefficient (default 0.9).
    pub beta1: f32,
    /// Second-moment (belief) decay coefficient (default 0.999).
    pub beta2: f32,
    /// Denominator epsilon for numerical stability (must be > 0).
    pub eps: f32,
    /// Decoupled weight decay coefficient (AdamW-style; default 0).
    pub weight_decay: f32,
}

impl Default for AdaBeliefConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-16,
            weight_decay: 0.0,
        }
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// AdaBelief adaptive gradient optimizer.
///
/// Stores first-moment `m` and belief `s` state buffers on the host and
/// operates on flat `f32` slices.
pub struct AdaBelief {
    m: Vec<f32>,
    s: Vec<f32>,
    t: usize,
    config: AdaBeliefConfig,
}

impl AdaBelief {
    /// Create a new `AdaBelief` optimizer for parameters of size `dim`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::InvalidLearningRate`] if `config.lr <= 0`.
    /// * [`TrainError::Internal`] if `config.eps <= 0`, if either β lies outside
    ///   `[0, 1)`, or if `weight_decay < 0`.
    pub fn new(dim: usize, config: AdaBeliefConfig) -> TrainResult<Self> {
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
        if !(0.0..1.0).contains(&config.beta1) || !(0.0..1.0).contains(&config.beta2) {
            return Err(TrainError::Internal {
                msg: format!(
                    "beta1/beta2 must be in [0, 1), got {} / {}",
                    config.beta1, config.beta2
                ),
            });
        }
        if config.weight_decay < 0.0 {
            return Err(TrainError::Internal {
                msg: format!("weight_decay must be >= 0, got {}", config.weight_decay),
            });
        }
        Ok(Self {
            m: vec![0.0; dim],
            s: vec![0.0; dim],
            t: 0,
            config,
        })
    }

    /// Perform one optimizer step, updating `params` in-place with `grads`.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] if lengths differ.
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
        let wd = self.config.weight_decay;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            let diff = g - self.m[i];
            // Belief: EMA of (g − m)², plus eps inside for stability.
            self.s[i] = b2 * self.s[i] + (1.0 - b2) * diff * diff + eps;
            let m_hat = self.m[i] / bc1;
            let s_hat = self.s[i] / bc2;
            // Decoupled weight decay (AdamW-style).
            if wd != 0.0 {
                params[i] -= lr * wd * params[i];
            }
            params[i] -= lr * m_hat / (s_hat.sqrt() + eps);
        }
        Ok(())
    }

    /// Reset all optimizer state (moments and step counter).
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.s.fill(0.0);
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

    /// Return a reference to the belief (variance) buffer.
    #[must_use]
    pub fn s(&self) -> &[f32] {
        &self.s
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AdaBeliefConfig {
        AdaBeliefConfig {
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-12,
            weight_decay: 0.0,
        }
    }

    #[test]
    fn step_changes_params() {
        let mut opt = AdaBelief::new(4, default_config()).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.1_f32; 4];
        opt.step(&mut params, &grads).expect("step ok");
        for &p in &params {
            assert!(p < 1.0, "params should decrease with positive grad");
        }
    }

    #[test]
    fn converges_quadratic() {
        let mut opt = AdaBelief::new(1, default_config()).expect("valid");
        let mut params = vec![2.0_f32];
        for _ in 0..500 {
            let g = 2.0 * params[0];
            opt.step(&mut params, &[g]).expect("step ok");
        }
        assert!(
            params[0].abs() < 0.1,
            "should converge x² toward 0, got {}",
            params[0]
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut opt = AdaBelief::new(2, default_config()).expect("valid");
        let mut params = vec![1.0_f32; 2];
        opt.step(&mut params, &[0.5, 0.5]).expect("step ok");
        assert_eq!(opt.step_count(), 1);
        opt.reset();
        assert_eq!(opt.step_count(), 0);
        assert!(opt.m().iter().all(|&v| v == 0.0));
        assert!(opt.s().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lr_zero_error() {
        let cfg = AdaBeliefConfig {
            lr: 0.0,
            ..default_config()
        };
        assert!(matches!(
            AdaBelief::new(4, cfg),
            Err(TrainError::InvalidLearningRate { .. })
        ));
    }

    #[test]
    fn negative_eps_error() {
        let cfg = AdaBeliefConfig {
            eps: -1e-8,
            ..default_config()
        };
        assert!(matches!(
            AdaBelief::new(4, cfg),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn invalid_beta_error() {
        let cfg = AdaBeliefConfig {
            beta1: 1.0,
            ..default_config()
        };
        assert!(matches!(
            AdaBelief::new(4, cfg),
            Err(TrainError::Internal { .. })
        ));
        let cfg2 = AdaBeliefConfig {
            beta2: 1.5,
            ..default_config()
        };
        assert!(matches!(
            AdaBelief::new(4, cfg2),
            Err(TrainError::Internal { .. })
        ));
    }

    #[test]
    fn dim_mismatch_error() {
        let mut opt = AdaBelief::new(4, default_config()).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![0.1_f32; 3];
        assert!(matches!(
            opt.step(&mut params, &grads),
            Err(TrainError::ParamCountMismatch { .. })
        ));
    }

    #[test]
    fn belief_nonneg() {
        let mut opt = AdaBelief::new(4, default_config()).expect("valid");
        let mut params = vec![1.0_f32; 4];
        let grads = vec![-0.5_f32, 0.3, -1.0, 0.0];
        for _ in 0..20 {
            opt.step(&mut params, &grads).expect("step ok");
        }
        for (i, &v) in opt.s().iter().enumerate() {
            assert!(v >= 0.0, "belief s[{i}] must be non-negative, got {v}");
        }
    }

    #[test]
    fn params_finite_after_many_steps() {
        let mut opt = AdaBelief::new(8, default_config()).expect("valid");
        let mut params = vec![5.0_f32; 8];
        for _ in 0..500 {
            let grads: Vec<f32> = params.iter().map(|&p| 2.0 * p).collect();
            opt.step(&mut params, &grads).expect("step ok");
        }
        assert!(params.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn weight_decay_shrinks_params() {
        // With zero gradient but positive weight decay, params should shrink.
        let cfg = AdaBeliefConfig {
            lr: 1e-1,
            weight_decay: 0.1,
            ..default_config()
        };
        let mut opt = AdaBelief::new(2, cfg).expect("valid");
        let mut params = vec![1.0_f32; 2];
        let grads = vec![0.0_f32; 2];
        opt.step(&mut params, &grads).expect("step ok");
        for &p in &params {
            assert!(p < 1.0, "weight decay should shrink params, got {p}");
        }
    }

    #[test]
    fn negative_weight_decay_error() {
        let cfg = AdaBeliefConfig {
            weight_decay: -0.1,
            ..default_config()
        };
        assert!(matches!(
            AdaBelief::new(4, cfg),
            Err(TrainError::Internal { .. })
        ));
    }
}
