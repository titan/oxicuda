//! Adaptive first / second moment optimizers for VQE parameter updates.
//!
//! Provides two standard adaptive optimizers commonly used in variational
//! quantum algorithms when the energy landscape is noisy or has heterogeneous
//! curvature across parameters.
//!
//! - **Adam** (Kingma & Ba 2015): combines exponential moving averages of the
//!   gradient (first moment `m`) and its square (second moment `v`) with bias
//!   correction. The parameter update is
//!   `θ ← θ − lr · m̂ / (√v̂ + ε)` where `m̂ = m / (1 − β₁ᵗ)` and
//!   `v̂ = v / (1 − β₂ᵗ)`.
//! - **RMSProp** (Tieleman & Hinton 2012): keeps only an exponential moving
//!   average of the squared gradient and uses
//!   `θ ← θ − lr · g / (√v + ε)`.
//!
//! Both optimizers operate purely on the provided gradient vector and do not
//! themselves perform any quantum-circuit evaluation; the caller is expected
//! to compute the gradient via parameter-shift, SPSA, or finite differences
//! and pass it to [`VqeOptimizerState::step`].

use crate::error::{QuantumError, QuantumResult};

/// Variant of the optimizer with its associated hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VqeOptKind {
    /// Adam (Kingma & Ba 2015) with bias-corrected first / second moments.
    Adam {
        /// Exponential decay rate for the first-moment estimate `m` (typical 0.9).
        beta1: f32,
        /// Exponential decay rate for the second-moment estimate `v` (typical 0.999).
        beta2: f32,
        /// Numerical-stability constant added to `√v̂` (typical 1e-8).
        eps: f32,
    },
    /// RMSProp (Tieleman & Hinton 2012) with running-average squared gradient.
    Rmsprop {
        /// Exponential decay rate for the squared-gradient running average (typical 0.9).
        decay: f32,
        /// Numerical-stability constant added to `√v` (typical 1e-8).
        eps: f32,
    },
}

/// Mutable state of a VQE optimizer.
///
/// `m` is the first-moment vector (used by Adam only; unused by RMSProp but
/// kept allocated for a uniform layout). `v` is the second-moment / running
/// average of squared gradients. `t` is the step counter (incremented by Adam
/// each call to [`Self::step`]).
#[derive(Debug, Clone)]
pub struct VqeOptimizerState {
    /// Current parameter vector being optimized.
    pub params: Vec<f32>,
    /// First-moment running average (Adam). Same length as `params`.
    pub m: Vec<f32>,
    /// Second-moment running average. Same length as `params`.
    pub v: Vec<f32>,
    /// Adam step counter (incremented at the start of each Adam step).
    pub t: usize,
    /// Learning rate `lr` applied to the (possibly bias-corrected) update direction.
    pub lr: f32,
    /// Optimizer variant + its hyperparameters.
    pub kind: VqeOptKind,
}

impl VqeOptimizerState {
    /// Construct a new optimizer state with zero-initialized moment vectors.
    ///
    /// Validates that `params` is non-empty, `lr > 0`, that `eps > 0`, and that
    /// the configuration-specific decay rates lie in `[0, 1)`.
    pub fn new(params: Vec<f32>, lr: f32, kind: VqeOptKind) -> QuantumResult<Self> {
        if params.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(QuantumError::InvalidParameter {
                name: "lr".to_string(),
            });
        }
        match kind {
            VqeOptKind::Adam { beta1, beta2, eps } => {
                if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
                    return Err(QuantumError::InvalidParameter {
                        name: "beta1".to_string(),
                    });
                }
                if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
                    return Err(QuantumError::InvalidParameter {
                        name: "beta2".to_string(),
                    });
                }
                if !eps.is_finite() || eps <= 0.0 {
                    return Err(QuantumError::InvalidParameter {
                        name: "eps".to_string(),
                    });
                }
            }
            VqeOptKind::Rmsprop { decay, eps } => {
                if !decay.is_finite() || !(0.0..1.0).contains(&decay) {
                    return Err(QuantumError::InvalidParameter {
                        name: "decay".to_string(),
                    });
                }
                if !eps.is_finite() || eps <= 0.0 {
                    return Err(QuantumError::InvalidParameter {
                        name: "eps".to_string(),
                    });
                }
            }
        }
        let n = params.len();
        Ok(Self {
            params,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            kind,
        })
    }

    /// Apply one optimizer step with the supplied gradient vector.
    ///
    /// `grad` must have the same length as `params`; otherwise a dimension
    /// mismatch error is returned. The internal `m`, `v`, and `t` state is
    /// updated in-place along with `params`.
    pub fn step(&mut self, grad: &[f32]) -> QuantumResult<()> {
        if grad.len() != self.params.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.params.len(),
                got: grad.len(),
            });
        }
        match self.kind {
            VqeOptKind::Adam { beta1, beta2, eps } => {
                self.t = self.t.saturating_add(1);
                let t = self.t as i32;
                let bc1 = 1.0 - beta1.powi(t);
                let bc2 = 1.0 - beta2.powi(t);
                for (i, &g) in grad.iter().enumerate() {
                    let m_new = beta1 * self.m[i] + (1.0 - beta1) * g;
                    let v_new = beta2 * self.v[i] + (1.0 - beta2) * g * g;
                    self.m[i] = m_new;
                    self.v[i] = v_new;
                    let m_hat = m_new / bc1;
                    let v_hat = v_new / bc2;
                    self.params[i] -= self.lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
            VqeOptKind::Rmsprop { decay, eps } => {
                self.t = self.t.saturating_add(1);
                for (i, &g) in grad.iter().enumerate() {
                    let v_new = decay * self.v[i] + (1.0 - decay) * g * g;
                    self.v[i] = v_new;
                    self.params[i] -= self.lr * g / (v_new.sqrt() + eps);
                }
            }
        }
        Ok(())
    }

    /// Borrow the current parameter vector.
    #[must_use]
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Clear the moment buffers and step counter, keeping `params`, `lr`, and `kind`.
    pub fn reset_state(&mut self) {
        for slot in self.m.iter_mut() {
            *slot = 0.0;
        }
        for slot in self.v.iter_mut() {
            *slot = 0.0;
        }
        self.t = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_adam_kind() -> VqeOptKind {
        VqeOptKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn default_rmsprop_kind() -> VqeOptKind {
        VqeOptKind::Rmsprop {
            decay: 0.9,
            eps: 1e-8,
        }
    }

    #[test]
    fn new_initializes_moments_to_zero() -> QuantumResult<()> {
        let st = VqeOptimizerState::new(vec![1.0, 2.0, 3.0], 0.01, default_adam_kind())?;
        assert_eq!(st.m, vec![0.0, 0.0, 0.0]);
        assert_eq!(st.v, vec![0.0, 0.0, 0.0]);
        assert_eq!(st.t, 0);
        assert_eq!(st.params, vec![1.0, 2.0, 3.0]);
        assert_eq!(st.lr, 0.01);
        Ok(())
    }

    #[test]
    fn adam_step_updates_t() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![1.0, 2.0], 0.01, default_adam_kind())?;
        st.step(&[0.1, -0.2])?;
        assert_eq!(st.t, 1);
        st.step(&[0.1, -0.2])?;
        assert_eq!(st.t, 2);
        Ok(())
    }

    #[test]
    fn adam_moment_formulas_one_step() -> QuantumResult<()> {
        // Single Adam step: with default β1=0.9, β2=0.999, ε=1e-8, params=[0.0]
        // and gradient g=2.0, then
        //   m = 0.1·g = 0.2
        //   v = 0.001·g² = 0.004
        //   bias-correction: m̂ = m/(1-0.9) = 2.0;  v̂ = v/(1-0.999) = 4.0
        //   update = lr · m̂ / (√v̂ + ε) ≈ 0.01 · 2.0 / 2.0 = 0.01
        // Therefore the new parameter ≈ -0.01.
        let mut st = VqeOptimizerState::new(vec![0.0], 0.01, default_adam_kind())?;
        st.step(&[2.0])?;
        assert!((st.m[0] - 0.2).abs() < 1e-6, "m={}", st.m[0]);
        assert!((st.v[0] - 0.004).abs() < 1e-6, "v={}", st.v[0]);
        assert!(
            (st.params[0] + 0.01).abs() < 1e-5,
            "params={}",
            st.params[0]
        );
        Ok(())
    }

    #[test]
    fn rmsprop_running_average_update() -> QuantumResult<()> {
        // First step on RMSProp with default decay=0.9, eps=1e-8, params=[0.0], g=2.0
        //   v = 0.9·0 + 0.1·4 = 0.4
        //   step: lr · g / (√v + ε) = 0.01 · 2 / (√0.4 + 1e-8) ≈ 0.0316228
        let mut st = VqeOptimizerState::new(vec![0.0], 0.01, default_rmsprop_kind())?;
        st.step(&[2.0])?;
        assert!((st.v[0] - 0.4).abs() < 1e-6, "v={}", st.v[0]);
        let expected = -0.01 * 2.0 / (0.4_f32.sqrt() + 1e-8);
        assert!(
            (st.params[0] - expected).abs() < 1e-5,
            "params={} expected={}",
            st.params[0],
            expected
        );
        Ok(())
    }

    #[test]
    fn zero_gradient_leaves_params_unchanged() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![1.5, -2.0], 0.05, default_adam_kind())?;
        st.step(&[0.0, 0.0])?;
        assert!((st.params[0] - 1.5).abs() < 1e-7);
        assert!((st.params[1] + 2.0).abs() < 1e-7);

        let mut st2 = VqeOptimizerState::new(vec![1.5, -2.0], 0.05, default_rmsprop_kind())?;
        st2.step(&[0.0, 0.0])?;
        assert!((st2.params[0] - 1.5).abs() < 1e-7);
        assert!((st2.params[1] + 2.0).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn positive_gradient_decreases_param() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![10.0], 0.1, default_adam_kind())?;
        st.step(&[1.0])?;
        assert!(st.params[0] < 10.0, "params={}", st.params[0]);

        let mut st2 = VqeOptimizerState::new(vec![10.0], 0.1, default_rmsprop_kind())?;
        st2.step(&[1.0])?;
        assert!(st2.params[0] < 10.0, "params={}", st2.params[0]);
        Ok(())
    }

    #[test]
    fn adam_vs_rmsprop_differ_after_step() -> QuantumResult<()> {
        let mut adam = VqeOptimizerState::new(vec![0.0], 0.01, default_adam_kind())?;
        let mut rms = VqeOptimizerState::new(vec![0.0], 0.01, default_rmsprop_kind())?;
        adam.step(&[2.0])?;
        rms.step(&[2.0])?;
        assert!(
            (adam.params[0] - rms.params[0]).abs() > 1e-6,
            "Adam and RMSProp should not coincide after one step: adam={} rms={}",
            adam.params[0],
            rms.params[0]
        );
        Ok(())
    }

    #[test]
    fn reset_state_zeros_moments_and_t() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![1.0, 2.0], 0.01, default_adam_kind())?;
        st.step(&[0.5, -0.3])?;
        st.step(&[0.4, 0.2])?;
        assert!(st.t == 2);
        let params_snapshot = st.params.clone();
        st.reset_state();
        assert_eq!(st.m, vec![0.0, 0.0]);
        assert_eq!(st.v, vec![0.0, 0.0]);
        assert_eq!(st.t, 0);
        assert_eq!(st.params, params_snapshot);
        assert_eq!(st.lr, 0.01);
        Ok(())
    }

    #[test]
    fn adam_quadratic_descent_toward_zero() -> QuantumResult<()> {
        // Minimize f(x) = ½ x²   ⇒   g = x.
        let mut st = VqeOptimizerState::new(vec![1.0], 0.05, default_adam_kind())?;
        for _ in 0..400 {
            let g = st.params[0];
            st.step(&[g])?;
        }
        assert!(
            st.params[0].abs() < 1e-2,
            "Adam did not drive x→0: x={}",
            st.params[0]
        );
        Ok(())
    }

    #[test]
    fn rmsprop_quadratic_descent_toward_zero() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![1.0], 0.01, default_rmsprop_kind())?;
        for _ in 0..400 {
            let g = st.params[0];
            st.step(&[g])?;
        }
        assert!(
            st.params[0].abs() < 1e-2,
            "RMSProp did not drive x→0: x={}",
            st.params[0]
        );
        Ok(())
    }

    #[test]
    fn deterministic_given_fixed_inputs() -> QuantumResult<()> {
        let mut a = VqeOptimizerState::new(vec![0.5, -0.3], 0.02, default_adam_kind())?;
        let mut b = VqeOptimizerState::new(vec![0.5, -0.3], 0.02, default_adam_kind())?;
        let grads: [[f32; 2]; 5] = [
            [0.1, -0.1],
            [0.2, 0.0],
            [-0.3, 0.4],
            [0.05, -0.05],
            [-0.1, 0.2],
        ];
        for g in &grads {
            a.step(g)?;
            b.step(g)?;
        }
        assert_eq!(a.params, b.params);
        assert_eq!(a.m, b.m);
        assert_eq!(a.v, b.v);
        assert_eq!(a.t, b.t);
        Ok(())
    }

    #[test]
    fn err_non_positive_lr() {
        assert!(VqeOptimizerState::new(vec![1.0], 0.0, default_adam_kind()).is_err());
        assert!(VqeOptimizerState::new(vec![1.0], -0.1, default_adam_kind()).is_err());
        assert!(VqeOptimizerState::new(vec![1.0], 0.0, default_rmsprop_kind()).is_err());
    }

    #[test]
    fn err_non_positive_eps() {
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 0.0,
                },
            )
            .is_err()
        );
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: -1e-8,
                },
            )
            .is_err()
        );
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Rmsprop {
                    decay: 0.9,
                    eps: 0.0,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn err_decay_rates_out_of_range() {
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Adam {
                    beta1: 1.0,
                    beta2: 0.999,
                    eps: 1e-8,
                },
            )
            .is_err()
        );
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Adam {
                    beta1: 0.9,
                    beta2: 1.0,
                    eps: 1e-8,
                },
            )
            .is_err()
        );
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Adam {
                    beta1: -0.1,
                    beta2: 0.999,
                    eps: 1e-8,
                },
            )
            .is_err()
        );
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Rmsprop {
                    decay: 1.0,
                    eps: 1e-8,
                },
            )
            .is_err()
        );
        assert!(
            VqeOptimizerState::new(
                vec![1.0],
                0.01,
                VqeOptKind::Rmsprop {
                    decay: -0.5,
                    eps: 1e-8,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn err_grad_wrong_length() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![1.0, 2.0], 0.01, default_adam_kind())?;
        let r = st.step(&[0.1]);
        assert!(r.is_err());
        let r2 = st.step(&[0.1, 0.2, 0.3]);
        assert!(r2.is_err());

        let mut st2 = VqeOptimizerState::new(vec![1.0, 2.0], 0.01, default_rmsprop_kind())?;
        assert!(st2.step(&[]).is_err());
        Ok(())
    }

    #[test]
    fn err_params_empty() {
        assert!(VqeOptimizerState::new(vec![], 0.01, default_adam_kind()).is_err());
        assert!(VqeOptimizerState::new(vec![], 0.01, default_rmsprop_kind()).is_err());
    }

    #[test]
    fn constant_gradient_roughly_constant_step_magnitude() -> QuantumResult<()> {
        // For Adam under a constant gradient g, after sufficient steps the
        // first moment converges to g and the second moment to g², so the
        // bias-corrected ratio m̂/√v̂ → ±1 and each step displaces the
        // parameter by roughly ±lr. Verify the step magnitude stabilizes.
        let mut st = VqeOptimizerState::new(vec![0.0], 0.01, default_adam_kind())?;
        let mut prev = st.params[0];
        let mut last_delta = 0.0_f32;
        for k in 0..200 {
            st.step(&[1.0])?;
            let delta = (st.params[0] - prev).abs();
            prev = st.params[0];
            if k == 199 {
                last_delta = delta;
            }
        }
        // After 200 steps the per-step displacement magnitude should be
        // within an order of magnitude of lr=0.01.
        assert!(
            last_delta > 1e-3 && last_delta < 5e-2,
            "unexpected last_delta={last_delta}"
        );
        Ok(())
    }

    #[test]
    fn default_adam_hyperparameters_work() -> QuantumResult<()> {
        // β1 = 0.9, β2 = 0.999, eps = 1e-8 -- the canonical Adam defaults.
        let kind = VqeOptKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let mut st = VqeOptimizerState::new(vec![0.0, 0.0], 0.01, kind)?;
        for _ in 0..20 {
            st.step(&[0.5, -0.5])?;
        }
        assert!(st.params[0].is_finite() && st.params[1].is_finite());
        assert!(
            st.params[0] < 0.0 && st.params[1] > 0.0,
            "expected descent in both directions: {:?}",
            st.params
        );
        Ok(())
    }

    #[test]
    fn params_getter_borrows_state() -> QuantumResult<()> {
        let st = VqeOptimizerState::new(vec![1.1, 2.2, 3.3], 0.01, default_adam_kind())?;
        let p = st.params();
        assert_eq!(p.len(), 3);
        assert!((p[0] - 1.1).abs() < 1e-7);
        assert!((p[2] - 3.3).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn rmsprop_multi_step_v_increases_monotonically_for_constant_grad() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![0.0], 0.01, default_rmsprop_kind())?;
        let mut prev_v = 0.0_f32;
        for _ in 0..10 {
            st.step(&[1.0])?;
            assert!(
                st.v[0] > prev_v - 1e-9,
                "RMSProp v should approach 1 from below: prev={prev_v} v={}",
                st.v[0]
            );
            prev_v = st.v[0];
        }
        // After many steps with constant g=1, the moving average v → 1.
        assert!(st.v[0] > 0.5, "v did not climb high enough: {}", st.v[0]);
        assert!(st.v[0] < 1.0 + 1e-6, "v overshot: {}", st.v[0]);
        Ok(())
    }

    #[test]
    fn reset_then_step_reproduces_first_step() -> QuantumResult<()> {
        let mut st = VqeOptimizerState::new(vec![0.0], 0.01, default_adam_kind())?;
        st.step(&[2.0])?;
        let after_first = st.params[0];

        st.reset_state();
        st.params[0] = 0.0;
        st.step(&[2.0])?;
        assert!(
            (st.params[0] - after_first).abs() < 1e-7,
            "reset+step did not match first step: {after_first} vs {}",
            st.params[0]
        );
        Ok(())
    }
}
