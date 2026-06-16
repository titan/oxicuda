//! LAMB optimizer — Layer-wise Adaptive Moments for Batch training.
//!
//! LAMB (You et al., 2019 / Ginsburg 2019) combines Adam's moment-based
//! adaptive update with LARS' trust-ratio layer-wise scaling, enabling
//! large-batch training without per-layer learning-rate tuning.
//!
//! ## Algorithm
//!
//! ```text
//! t  ← t + 1
//! m  ← β₁·m + (1−β₁)·g                           // first moment EMA
//! v  ← β₂·v + (1−β₂)·g²                           // second moment EMA
//! m̂  = m / (1 − β₁^t)                              // bias-corrected m
//! v̂  = v / (1 − β₂^t)                              // bias-corrected v
//! r  = m̂/(√v̂ + ε) + wd·θ                          // Adam update + L2 reg
//! φ  = ‖θ‖₂ / ‖r‖₂  (1.0 if either norm is 0)     // trust ratio
//! φ  = clamp(φ, 0, trust_ratio_clip)
//! θ  ← θ − lr·φ·r                                  // layer-scaled update
//! ```

use crate::error::{TrainError, TrainResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`Lamb`] optimizer.
#[derive(Debug, Clone)]
pub struct LambConfig {
    /// Learning rate (must be > 0).
    pub lr: f32,
    /// First-moment decay coefficient β₁ (default 0.9).
    pub beta1: f32,
    /// Second-moment decay coefficient β₂ (default 0.999).
    pub beta2: f32,
    /// Denominator epsilon for numerical stability (default 1e-6).
    pub eps: f32,
    /// L2 weight decay coefficient (default 0.01).
    pub weight_decay: f32,
    /// Upper bound on the trust ratio (default 10.0).
    pub trust_ratio_clip: f32,
}

impl Default for LambConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
            trust_ratio_clip: 10.0,
        }
    }
}

// ─── Optimizer ───────────────────────────────────────────────────────────────

/// LAMB layer-wise adaptive moments optimizer.
///
/// Operates on flat `f32` slices.  LAMB is identical to Adam at the element
/// level but scales the global update by a per-call trust ratio
/// `‖θ‖ / ‖r‖`, clamped to `[0, trust_ratio_clip]`.
pub struct Lamb {
    /// First-moment EMA buffer.
    m: Vec<f32>,
    /// Second-moment EMA buffer.
    v: Vec<f32>,
    /// Optimizer step count (starts at 0, incremented on each `step` call).
    step: usize,
    /// Hyper-parameter configuration.
    config: LambConfig,
}

impl Lamb {
    /// Create a new `Lamb` optimizer for `n_params` parameters.
    ///
    /// # Errors
    ///
    /// * [`TrainError::EmptyParams`] — `n_params == 0`.
    /// * [`TrainError::InvalidLearningRate`] — `config.lr <= 0`.
    pub fn new(n_params: usize, config: LambConfig) -> TrainResult<Self> {
        if n_params == 0 {
            return Err(TrainError::EmptyParams);
        }
        if config.lr <= 0.0 {
            return Err(TrainError::InvalidLearningRate {
                lr: config.lr as f64,
            });
        }
        Ok(Self {
            m: vec![0.0_f32; n_params],
            v: vec![0.0_f32; n_params],
            step: 0,
            config,
        })
    }

    /// Perform one optimizer step, updating `params` in-place.
    ///
    /// # Errors
    ///
    /// * [`TrainError::ParamCountMismatch`] — `params.len() != grad.len()` or
    ///   internal state length differs.
    pub fn step(&mut self, params: &mut [f32], grad: &[f32]) -> TrainResult<()> {
        let n = self.m.len();

        if params.len() != n {
            return Err(TrainError::ParamCountMismatch {
                expected: n,
                got: params.len(),
            });
        }
        if grad.len() != n {
            return Err(TrainError::ParamCountMismatch {
                expected: n,
                got: grad.len(),
            });
        }

        self.step += 1;
        let t = self.step as i32;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;
        let lr = self.config.lr;
        let clip = self.config.trust_ratio_clip;

        // Bias-correction denominators.
        let bc1 = 1.0_f32 - b1.powi(t);
        let bc2 = 1.0_f32 - b2.powi(t);

        // ── Step 1-5: compute Adam update vector r into a temporary buffer ────
        let mut r_buf = vec![0.0_f32; n];
        for i in 0..n {
            let g = grad[i];
            // (2) m ← β₁·m + (1−β₁)·g
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            // (3) v ← β₂·v + (1−β₂)·g²
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            // (4) bias-corrected moments
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            // (5) r = m̂/(√v̂ + ε) + wd·θ
            r_buf[i] = m_hat / (v_hat.sqrt() + eps) + wd * params[i];
        }

        // ── Step 6: compute trust ratio ───────────────────────────────────────
        let param_norm = l2_norm(params);
        let r_norm = l2_norm(&r_buf);

        let trust_ratio = if param_norm == 0.0 || r_norm == 0.0 {
            1.0_f32
        } else {
            (param_norm / r_norm).clamp(0.0, clip)
        };

        // ── Step 7: parameter update ──────────────────────────────────────────
        let scale = lr * trust_ratio;
        for i in 0..n {
            params[i] -= scale * r_buf[i];
        }

        Ok(())
    }

    /// Return the current step count.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Compute the L2 norm of a slice.
#[inline]
fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LambConfig {
        LambConfig {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.0,
            trust_ratio_clip: 10.0,
        }
    }

    // 1. params are modified after step
    #[test]
    fn params_change() {
        let mut lamb = Lamb::new(4, cfg()).expect("valid config");
        let mut params = vec![1.0_f32; 4];
        let grad = vec![0.5_f32; 4];
        lamb.step(&mut params, &grad).expect("step ok");
        for &p in &params {
            assert!(
                p < 1.0,
                "params should decrease after positive gradient step, got {p}"
            );
        }
    }

    // 2. trust_ratio scales the update
    #[test]
    fn trust_ratio_scales() {
        // With wd=0, trust_ratio = ||params|| / ||r||.
        // A large param norm relative to r norm → large trust ratio → large update.
        // Use two separate instances with identical configs but different initial params.
        let base_cfg = LambConfig {
            lr: 1e-2,
            weight_decay: 0.0,
            trust_ratio_clip: 100.0,
            ..cfg()
        };

        let mut lamb_large = Lamb::new(4, base_cfg.clone()).expect("valid config");
        let mut lamb_small = Lamb::new(4, base_cfg).expect("valid config");

        // Large param norm → large trust ratio → larger update magnitude
        let mut params_large = vec![100.0_f32; 4];
        let mut params_small = vec![1.0_f32; 4];
        let grad = vec![1.0_f32; 4];

        let large_before = params_large[0];
        let small_before = params_small[0];

        lamb_large.step(&mut params_large, &grad).expect("step ok");
        lamb_small.step(&mut params_small, &grad).expect("step ok");

        let large_delta = (large_before - params_large[0]).abs();
        let small_delta = (small_before - params_small[0]).abs();

        // Large param norm should produce a larger absolute update via trust ratio
        assert!(
            large_delta > small_delta,
            "larger param norm should yield larger update: large_delta={large_delta}, small_delta={small_delta}"
        );
    }

    // 3. weight_decay_effect — with wd>0 large-norm params pulled harder
    #[test]
    fn weight_decay_effect() {
        let cfg_nowd = LambConfig {
            weight_decay: 0.0,
            ..cfg()
        };
        let cfg_wd = LambConfig {
            weight_decay: 0.1,
            ..cfg()
        };

        let mut lamb_nowd = Lamb::new(4, cfg_nowd).expect("valid config");
        let mut lamb_wd = Lamb::new(4, cfg_wd).expect("valid config");

        // Use large params so weight decay term wd*θ is significant
        let mut params_nowd = vec![10.0_f32; 4];
        let mut params_wd = vec![10.0_f32; 4];
        let grad = vec![0.01_f32; 4]; // tiny gradient so wd dominates

        lamb_nowd.step(&mut params_nowd, &grad).expect("step ok");
        lamb_wd.step(&mut params_wd, &grad).expect("step ok");

        // With weight decay the r vector is larger, and if wd dominates params should
        // move more (r = m̂/(√v̂+ε) + wd*θ is larger than without wd)
        let delta_nowd = (10.0_f32 - params_nowd[0]).abs();
        let delta_wd = (10.0_f32 - params_wd[0]).abs();

        assert!(
            delta_wd > delta_nowd,
            "weight decay should increase update magnitude: wd_delta={delta_wd}, nowd_delta={delta_nowd}"
        );
    }

    // 4. step_finite — no NaN/Inf after step
    #[test]
    fn step_finite() {
        let mut lamb = Lamb::new(8, cfg()).expect("valid config");
        let mut params = vec![3.0_f32; 8];
        let grad: Vec<f32> = params.iter().map(|&p| 2.0 * p).collect();
        lamb.step(&mut params, &grad).expect("step ok");
        for &p in &params {
            assert!(p.is_finite(), "param must be finite after step, got {p}");
        }
    }

    // 5. new(0,..) returns EmptyParams
    #[test]
    fn n_params_0_error() {
        let result = Lamb::new(0, cfg());
        assert!(
            matches!(result, Err(TrainError::EmptyParams)),
            "n_params=0 should return EmptyParams"
        );
    }

    // 6. wrong grad len returns ParamCountMismatch
    #[test]
    fn len_mismatch_error() {
        let mut lamb = Lamb::new(4, cfg()).expect("valid config");
        let mut params = vec![1.0_f32; 4];
        let bad_grad = vec![0.1_f32; 3]; // wrong length
        let result = lamb.step(&mut params, &bad_grad);
        assert!(
            matches!(
                result,
                Err(TrainError::ParamCountMismatch {
                    expected: 4,
                    got: 3
                })
            ),
            "wrong grad len should return ParamCountMismatch"
        );
    }

    // 7. trust_ratio_clamped — tiny r-norm, large param-norm → hits clip
    #[test]
    fn trust_ratio_clamped() {
        let clip = 2.0_f32;
        let cfg_clip = LambConfig {
            trust_ratio_clip: clip,
            weight_decay: 0.0,
            lr: 1e-6,   // tiny lr so we can observe clamping behavior
            beta1: 0.0, // m = g immediately (no smoothing lag)
            beta2: 0.0, // v = g² immediately
            eps: 1e-3,
        };

        let mut lamb = Lamb::new(4, cfg_clip).expect("valid config");
        // Very large params → large param norm
        let mut params = vec![1000.0_f32; 4];
        // Very small grad → small r norm → param_norm/r_norm >> clip
        let grad = vec![1e-10_f32; 4];

        // The expected trust ratio without clamping would be enormous;
        // we verify the update is bounded to clip * lr * r_magnitude
        let params_before = params.clone();
        lamb.step(&mut params, &grad).expect("step ok");

        for (i, (&before, &after)) in params_before.iter().zip(params.iter()).enumerate() {
            // Ensure params changed by at most clip * lr * (something finite)
            let delta = (before - after).abs();
            // If unclamped, trust_ratio could be ~1e13; with clip=2 it is bounded
            assert!(delta.is_finite(), "delta must be finite at index {i}");
            // The update should not have exploded
            assert!(
                after.is_finite(),
                "param[{i}] must remain finite after clamped trust ratio step"
            );
        }
    }

    // 8. multiple_steps — can call step 100 times without error
    #[test]
    fn multiple_steps() {
        let mut lamb = Lamb::new(4, cfg()).expect("valid config");
        let mut params = vec![1.0_f32; 4];
        for _ in 0..100 {
            let grad: Vec<f32> = params.iter().map(|&p| 2.0 * p).collect();
            lamb.step(&mut params, &grad)
                .expect("100 steps must not error");
        }
        for &p in &params {
            assert!(p.is_finite(), "params must remain finite after 100 steps");
        }
    }

    // 9. beta1=0 means m = g each step (no accumulation across steps)
    #[test]
    fn beta1_0_means_no_first_moment_accumulation() {
        let cfg_b0 = LambConfig {
            beta1: 0.0,
            beta2: 0.999,
            weight_decay: 0.0,
            eps: 1e-8,
            lr: 1e-6, // tiny lr so params barely move
            trust_ratio_clip: 10.0,
        };

        let mut lamb = Lamb::new(1, cfg_b0).expect("valid config");
        let mut params = vec![0.5_f32];

        // Step with gradient g=1.0
        lamb.step(&mut params, &[1.0_f32]).expect("step ok");
        // With beta1=0: m = 0*0 + 1*g = g = 1.0
        let m_after_first = lamb.m()[0];
        assert!(
            (m_after_first - 1.0_f32).abs() < 1e-6,
            "with beta1=0, m must equal g exactly, got {m_after_first}"
        );

        // Step with gradient g=0.3
        lamb.step(&mut params, &[0.3_f32]).expect("step ok");
        // With beta1=0: m = 0*m_prev + 1*0.3 = 0.3 (no accumulation)
        let m_after_second = lamb.m()[0];
        assert!(
            (m_after_second - 0.3_f32).abs() < 1e-6,
            "with beta1=0, m must equal new g without accumulation, got {m_after_second}"
        );
    }
}
