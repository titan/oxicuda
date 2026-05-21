//! DP-LAMB optimizer: LAMB + per-sample gradient clipping + Gaussian noise injection.
//!
//! References:
//! - You Y, Li J, Reddi S, Hseu J, Kumar S, Bhojanapalli S, Song X, Demmel J, Keutzer K,
//!   Hsieh CJ (2020) "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes",
//!   ICLR 2020.
//! - Abadi M et al. (2016) "Deep Learning with Differential Privacy", CCS 2016.
//!
//! # Algorithm Overview (one step)
//! Given per-sample gradients g₁, …, g_B (each ∈ ℝ^p):
//!
//! 1. **Clip**: g̃ᵢ = gᵢ · min(1, C / ‖gᵢ‖₂)  (per-sample L2 clip to bound C).
//! 2. **Aggregate**: ḡ = (Σ g̃ᵢ) / B.
//! 3. **Gaussian noise**: ḡ += N(0, σ²C²/B·I).
//! 4. **Adam moments** (bias-corrected):
//!    - m_t = β₁·m_{t-1} + (1−β₁)·ḡ;  m̂ = m_t / (1−β₁ᵗ)
//!    - v_t = β₂·v_{t-1} + (1−β₂)·ḡ²; v̂ = v_t / (1−β₂ᵗ)
//! 5. **Adam update direction**: r_j = m̂_j / (√v̂_j + ε_adam)
//! 6. **Weight decay** (decoupled): r_j += λ·θ_j
//! 7. **LAMB trust ratio**: φ = ‖θ‖₂, ψ = ‖r‖₂, τ = clamp(φ/ψ, τ_min, τ_max)
//! 8. **Update**: θ_j -= η · τ · r_j

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

// ─── DpLambConfig ────────────────────────────────────────────────────────────

/// Configuration for the DP-LAMB optimizer.
#[derive(Clone, Debug)]
pub struct DpLambConfig {
    /// Gaussian noise multiplier σ > 0. Effective noise std = σ·C/√B.
    pub sigma: f64,
    /// Per-sample L2 gradient clipping bound C > 0.
    pub grad_clip: f64,
    /// Global learning rate η > 0.
    pub learning_rate: f64,
    /// Adam first-moment decay β₁ ∈ (0, 1).
    pub beta1: f64,
    /// Adam second-moment decay β₂ ∈ (0, 1).
    pub beta2: f64,
    /// Adam numerical stability ε > 0 (denominator offset).
    pub epsilon_adam: f64,
    /// Decoupled L2 weight-decay regularisation coefficient λ ≥ 0.
    pub weight_decay: f64,
    /// Lower bound on LAMB trust ratio (≥ 0).
    pub min_trust_ratio: f64,
    /// Upper bound on LAMB trust ratio (> min_trust_ratio).
    pub max_trust_ratio: f64,
}

impl Default for DpLambConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            grad_clip: 1.0,
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon_adam: 1e-6,
            weight_decay: 0.0,
            min_trust_ratio: 0.0,
            max_trust_ratio: 10.0,
        }
    }
}

// ─── DpLambState ─────────────────────────────────────────────────────────────

/// Mutable state for DP-LAMB.
#[derive(Clone, Debug)]
pub struct DpLambState {
    /// Current parameter vector θ ∈ ℝ^dim.
    pub theta: Vec<f64>,
    /// First moment estimate m (Adam).
    pub m: Vec<f64>,
    /// Second moment estimate v (element-wise squared, Adam).
    pub v: Vec<f64>,
    /// Number of completed steps (0 before the first call to `step`).
    pub step: usize,
    /// Dimensionality of the parameter vector.
    pub dim: usize,
}

// ─── DpLamb ──────────────────────────────────────────────────────────────────

/// DP-LAMB optimizer.
///
/// Wraps a validated `DpLambConfig`. All update logic lives in `DpLamb::step`.
pub struct DpLamb {
    /// Validated configuration for this optimizer instance.
    pub cfg: DpLambConfig,
}

impl DpLamb {
    /// Construct and validate a `DpLamb` optimizer.
    ///
    /// # Errors
    /// Returns `InvalidParameter` for any out-of-range configuration value.
    pub fn new(cfg: DpLambConfig) -> PrivacyResult<Self> {
        if cfg.sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive, got {}",
                cfg.sigma
            )));
        }
        if cfg.grad_clip <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "grad_clip must be positive, got {}",
                cfg.grad_clip
            )));
        }
        if cfg.learning_rate <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be positive, got {}",
                cfg.learning_rate
            )));
        }
        if !(cfg.beta1 > 0.0 && cfg.beta1 < 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "beta1 must be in (0,1), got {}",
                cfg.beta1
            )));
        }
        if !(cfg.beta2 > 0.0 && cfg.beta2 < 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "beta2 must be in (0,1), got {}",
                cfg.beta2
            )));
        }
        if cfg.epsilon_adam <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_adam must be positive, got {}",
                cfg.epsilon_adam
            )));
        }
        if cfg.weight_decay < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "weight_decay must be ≥ 0, got {}",
                cfg.weight_decay
            )));
        }
        if cfg.min_trust_ratio < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "min_trust_ratio must be ≥ 0, got {}",
                cfg.min_trust_ratio
            )));
        }
        if cfg.max_trust_ratio <= cfg.min_trust_ratio {
            return Err(PrivacyError::InvalidParameter(format!(
                "max_trust_ratio ({}) must be > min_trust_ratio ({})",
                cfg.max_trust_ratio, cfg.min_trust_ratio
            )));
        }
        Ok(Self { cfg })
    }

    /// Initialise optimizer state from a given parameter vector.
    ///
    /// # Errors
    /// - `InvalidParameter` if `dim == 0`.
    /// - `DimensionMismatch` if `init_theta.len() != dim`.
    pub fn init_state(dim: usize, init_theta: &[f64]) -> PrivacyResult<DpLambState> {
        if dim == 0 {
            return Err(PrivacyError::InvalidParameter("dim must be > 0".into()));
        }
        if init_theta.len() != dim {
            return Err(PrivacyError::DimensionMismatch {
                expected: dim,
                got: init_theta.len(),
            });
        }
        Ok(DpLambState {
            theta: init_theta.to_vec(),
            m: vec![0.0; dim],
            v: vec![0.0; dim],
            step: 0,
            dim,
        })
    }

    /// Clip per-sample gradients to L2 norm C and return the averaged aggregate.
    ///
    /// For each sample i:
    ///   1. Compute L2 norm ‖gᵢ‖₂.
    ///   2. Scale = min(1, C / ‖gᵢ‖₂).
    ///   3. Accumulate scaled gradient.
    ///
    /// Divide accumulated sum by `batch_size` to produce the mean clipped gradient.
    ///
    /// Returns a `(dim,)` vector of the averaged clipped gradients.
    /// If `batch_size == 0`, returns a zero vector of length `dim`.
    pub fn clip_and_aggregate(
        per_sample_grads: &[f64],
        batch_size: usize,
        dim: usize,
        clip_norm: f64,
    ) -> Vec<f64> {
        let mut aggregated = vec![0.0_f64; dim];
        if batch_size == 0 {
            return aggregated;
        }
        for i in 0..batch_size {
            let start = i * dim;
            let g_i = &per_sample_grads[start..start + dim];
            let norm_sq: f64 = g_i.iter().map(|&x| x * x).sum();
            let norm = norm_sq.sqrt();
            let scale = if norm > clip_norm {
                clip_norm / norm
            } else {
                1.0
            };
            for j in 0..dim {
                aggregated[j] += g_i[j] * scale;
            }
        }
        let inv_batch = 1.0 / batch_size as f64;
        for a in aggregated.iter_mut() {
            *a *= inv_batch;
        }
        aggregated
    }

    /// Compute the LAMB trust ratio φ/ψ clamped to [min_tr, max_tr].
    ///
    /// φ = ‖θ‖₂ (parameter norm), ψ = ‖update‖₂ (Adam update direction norm).
    /// If either norm is negligibly small (< 1e-12), returns `min_tr` to avoid
    /// amplifying degenerate updates.
    pub fn compute_trust_ratio(theta: &[f64], update: &[f64], min_tr: f64, max_tr: f64) -> f64 {
        let phi_sq: f64 = theta.iter().map(|&x| x * x).sum();
        let psi_sq: f64 = update.iter().map(|&x| x * x).sum();
        let phi = phi_sq.sqrt();
        let psi = psi_sq.sqrt();
        if phi < 1e-12 || psi < 1e-12 {
            return min_tr;
        }
        (phi / psi).clamp(min_tr, max_tr)
    }

    /// Execute one DP-LAMB step.
    ///
    /// # Arguments
    /// - `state`: mutable optimizer state (θ, m, v, step counter).
    /// - `per_sample_grads`: flat `[batch_size × dim]` row-major gradient matrix.
    /// - `batch_size`: number of samples in this mini-batch.
    /// - `handle`: `PrivacyHandle` providing the RNG for Gaussian noise.
    ///
    /// # Errors
    /// - `InvalidParameter` if `batch_size == 0`.
    /// - `DimensionMismatch` if `per_sample_grads.len() != batch_size * state.dim`.
    pub fn step(
        &self,
        state: &mut DpLambState,
        per_sample_grads: &[f64],
        batch_size: usize,
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<()> {
        let dim = state.dim;

        if batch_size == 0 {
            return Err(PrivacyError::InvalidParameter(
                "batch_size must be ≥ 1".into(),
            ));
        }
        if per_sample_grads.len() != batch_size * dim {
            return Err(PrivacyError::DimensionMismatch {
                expected: batch_size * dim,
                got: per_sample_grads.len(),
            });
        }

        // ── Step 1 & 2: clip per-sample gradients and average. ───────────────
        let mut g_agg =
            Self::clip_and_aggregate(per_sample_grads, batch_size, dim, self.cfg.grad_clip);

        // ── Step 3: add calibrated Gaussian noise N(0, σ²C²/B · I). ─────────
        // noise_scale = σ·C / √B.
        let noise_scale = self.cfg.sigma * self.cfg.grad_clip / (batch_size as f64).sqrt();
        let noise = handle.generate_gaussian_noise(noise_scale, dim)?;
        for (g, n) in g_agg.iter_mut().zip(noise.iter()) {
            *g += n;
        }

        // ── Step 4: Adam moment updates. ─────────────────────────────────────
        state.step += 1;
        let b1 = self.cfg.beta1;
        let b2 = self.cfg.beta2;
        for ((m_j, v_j), &g) in state.m.iter_mut().zip(state.v.iter_mut()).zip(g_agg.iter()) {
            *m_j = b1 * *m_j + (1.0 - b1) * g;
            *v_j = b2 * *v_j + (1.0 - b2) * g * g;
        }

        // ── Step 5: bias correction. ─────────────────────────────────────────
        let t = state.step as i32;
        let beta1_t = b1.powi(t);
        let beta2_t = b2.powi(t);
        let bias1 = 1.0 - beta1_t;
        let bias2 = 1.0 - beta2_t;

        // ── Step 6: Adam update direction r. ─────────────────────────────────
        let mut r: Vec<f64> = state
            .m
            .iter()
            .zip(state.v.iter())
            .map(|(&m_j, &v_j)| {
                let m_hat = m_j / bias1;
                let v_hat = v_j / bias2;
                m_hat / (v_hat.sqrt() + self.cfg.epsilon_adam)
            })
            .collect();

        // ── Step 7: decoupled weight decay. ──────────────────────────────────
        if self.cfg.weight_decay > 0.0 {
            for (r_j, &th_j) in r.iter_mut().zip(state.theta.iter()) {
                *r_j += self.cfg.weight_decay * th_j;
            }
        }

        // ── Step 8: LAMB trust ratio. ─────────────────────────────────────────
        let trust = Self::compute_trust_ratio(
            &state.theta,
            &r,
            self.cfg.min_trust_ratio,
            self.cfg.max_trust_ratio,
        );

        // ── Step 9: parameter update. ─────────────────────────────────────────
        let lr_trust = self.cfg.learning_rate * trust;
        for (th_j, r_j) in state.theta.iter_mut().zip(r.iter()) {
            *th_j -= lr_trust * r_j;
        }

        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_handle() -> PrivacyHandle {
        PrivacyHandle::new(80, 42)
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_new_valid_default() {
        assert!(DpLamb::new(DpLambConfig::default()).is_ok());
    }

    #[test]
    fn test_new_invalid_sigma() {
        assert!(
            DpLamb::new(DpLambConfig {
                sigma: 0.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
        assert!(
            DpLamb::new(DpLambConfig {
                sigma: -1.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_invalid_grad_clip() {
        assert!(
            DpLamb::new(DpLambConfig {
                grad_clip: 0.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
        assert!(
            DpLamb::new(DpLambConfig {
                grad_clip: -0.1,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_invalid_lr() {
        assert!(
            DpLamb::new(DpLambConfig {
                learning_rate: 0.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
        assert!(
            DpLamb::new(DpLambConfig {
                learning_rate: -1e-4,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_invalid_beta1() {
        assert!(
            DpLamb::new(DpLambConfig {
                beta1: 0.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
        assert!(
            DpLamb::new(DpLambConfig {
                beta1: 1.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_invalid_weight_decay() {
        assert!(
            DpLamb::new(DpLambConfig {
                weight_decay: -0.1,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_invalid_trust_ratio_bounds() {
        // min == max → invalid
        assert!(
            DpLamb::new(DpLambConfig {
                min_trust_ratio: 5.0,
                max_trust_ratio: 5.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
        // min > max → invalid
        assert!(
            DpLamb::new(DpLambConfig {
                min_trust_ratio: 6.0,
                max_trust_ratio: 5.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_new_invalid_min_trust_ratio_negative() {
        assert!(
            DpLamb::new(DpLambConfig {
                min_trust_ratio: -1.0,
                ..DpLambConfig::default()
            })
            .is_err()
        );
    }

    // ── init_state ──────────────────────────────────────────────────────────

    #[test]
    fn test_init_state_zeroed_moments_and_step() {
        let theta = vec![1.0, 2.0, 3.0];
        let state = DpLamb::init_state(3, &theta).expect("ok");
        assert_eq!(state.step, 0);
        assert_eq!(state.dim, 3);
        assert!(state.m.iter().all(|&x| x == 0.0));
        assert!(state.v.iter().all(|&x| x == 0.0));
        assert_eq!(state.theta, theta);
    }

    #[test]
    fn test_init_state_dim_mismatch() {
        assert!(DpLamb::init_state(3, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_init_state_zero_dim() {
        assert!(DpLamb::init_state(0, &[]).is_err());
    }

    // ── clip_and_aggregate ──────────────────────────────────────────────────

    #[test]
    fn test_clip_and_aggregate_clips_large_gradient() {
        // Single sample: g = [3.0, 4.0], norm=5. C=1 → scale=0.2 → clipped=[0.6, 0.8].
        // Average over batch=1: [0.6, 0.8].
        let grads = vec![3.0_f64, 4.0_f64];
        let agg = DpLamb::clip_and_aggregate(&grads, 1, 2, 1.0);
        assert!((agg[0] - 0.6).abs() < 1e-12);
        assert!((agg[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_clip_and_aggregate_does_not_clip_small_gradient() {
        // Single sample: g = [0.3, 0.4], norm=0.5 < C=1 → scale=1, unchanged.
        let grads = vec![0.3_f64, 0.4_f64];
        let agg = DpLamb::clip_and_aggregate(&grads, 1, 2, 1.0);
        assert!((agg[0] - 0.3).abs() < 1e-12);
        assert!((agg[1] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_clip_and_aggregate_batch_average() {
        // Two identical samples [1.0, 0.0], norm=1=C → no clipping. Average = [1.0, 0.0].
        let grads = vec![1.0_f64, 0.0, 1.0, 0.0];
        let agg = DpLamb::clip_and_aggregate(&grads, 2, 2, 1.0);
        assert!((agg[0] - 1.0).abs() < 1e-12);
        assert!((agg[1]).abs() < 1e-12);
    }

    // ── compute_trust_ratio ─────────────────────────────────────────────────

    #[test]
    fn test_trust_ratio_basic() {
        // phi=3 (theta=[0,3,4] → 5), psi (update=[1,0,0] → 1) → ratio=5, clamped to max=10.
        let theta = vec![0.0, 3.0, 4.0]; // L2 norm = 5
        let update = vec![1.0, 0.0, 0.0]; // L2 norm = 1
        let ratio = DpLamb::compute_trust_ratio(&theta, &update, 0.0, 10.0);
        assert!((ratio - 5.0).abs() < 1e-12, "ratio={ratio}");
    }

    #[test]
    fn test_trust_ratio_clamp_max() {
        let theta = vec![100.0_f64]; // norm=100
        let update = vec![1.0_f64]; // norm=1, ratio=100
        let ratio = DpLamb::compute_trust_ratio(&theta, &update, 0.0, 10.0);
        assert!(
            (ratio - 10.0).abs() < 1e-12,
            "should clamp to max=10: {ratio}"
        );
    }

    #[test]
    fn test_trust_ratio_clamp_min() {
        let theta = vec![0.001_f64]; // norm=0.001
        let update = vec![100.0_f64]; // norm=100, ratio=0.00001
        let ratio = DpLamb::compute_trust_ratio(&theta, &update, 0.5, 10.0);
        assert!(
            (ratio - 0.5).abs() < 1e-12,
            "should clamp to min=0.5: {ratio}"
        );
    }

    #[test]
    fn test_trust_ratio_zero_theta_returns_min() {
        let theta = vec![0.0_f64, 0.0, 0.0];
        let update = vec![1.0_f64, 0.0, 0.0];
        let ratio = DpLamb::compute_trust_ratio(&theta, &update, 0.2, 5.0);
        assert!((ratio - 0.2).abs() < 1e-12, "zero theta → min_tr: {ratio}");
    }

    // ── step ────────────────────────────────────────────────────────────────

    #[test]
    fn test_step_counter_increments() {
        let opt = DpLamb::new(DpLambConfig::default()).expect("ok");
        let mut state = DpLamb::init_state(2, &[0.0, 0.0]).expect("ok");
        let mut handle = make_handle();
        let grads = vec![0.1_f64, 0.2]; // batch=1, dim=2
        opt.step(&mut state, &grads, 1, &mut handle).expect("ok");
        assert_eq!(state.step, 1);
        opt.step(&mut state, &grads, 1, &mut handle).expect("ok");
        assert_eq!(state.step, 2);
    }

    #[test]
    fn test_step_batch_size_zero_fails() {
        let opt = DpLamb::new(DpLambConfig::default()).expect("ok");
        let mut state = DpLamb::init_state(2, &[0.0, 0.0]).expect("ok");
        let mut handle = make_handle();
        assert!(opt.step(&mut state, &[], 0, &mut handle).is_err());
    }

    #[test]
    fn test_step_dimension_mismatch_fails() {
        let opt = DpLamb::new(DpLambConfig::default()).expect("ok");
        let mut state = DpLamb::init_state(3, &[0.0, 0.0, 0.0]).expect("ok");
        let mut handle = make_handle();
        // Providing 4 values for batch=1, dim=3 (should be 3).
        let bad_grads = vec![1.0_f64; 4];
        assert!(opt.step(&mut state, &bad_grads, 1, &mut handle).is_err());
    }

    #[test]
    fn test_step_params_change() {
        let opt = DpLamb::new(DpLambConfig::default()).expect("ok");
        let mut state = DpLamb::init_state(4, &[1.0, 1.0, 1.0, 1.0]).expect("ok");
        let theta_before = state.theta.clone();
        let mut handle = make_handle();
        let grads = vec![0.5_f64; 4]; // batch=1, dim=4
        opt.step(&mut state, &grads, 1, &mut handle).expect("ok");
        assert_ne!(state.theta, theta_before, "theta must change after step");
    }

    #[test]
    fn test_step_moments_initialized_to_zero_then_nonzero() {
        let opt = DpLamb::new(DpLambConfig::default()).expect("ok");
        let mut state = DpLamb::init_state(2, &[0.0, 0.0]).expect("ok");
        assert!(state.m.iter().all(|&x| x == 0.0));
        assert!(state.v.iter().all(|&x| x == 0.0));
        let mut handle = make_handle();
        let grads = vec![1.0_f64, -1.0];
        opt.step(&mut state, &grads, 1, &mut handle).expect("ok");
        // After step, moments should be non-zero.
        assert!(state.m.iter().any(|&x| x != 0.0));
        assert!(state.v.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_convergence_quadratic() {
        // Objective: f(θ) = (θ₀ − 2)² + (θ₁ + 1)².
        // True gradient: [2(θ₀−2), 2(θ₁+1)].
        //
        // LAMB trust ratio: φ/ψ = ‖θ‖/‖update‖. To avoid the zero-theta degenerate case
        // (where LAMB returns min_trust_ratio=0 and no update occurs), we:
        //   - start from a non-zero initial point [10.0, -10.0] (far from optimum)
        //   - set min_trust_ratio=0.1 so there is always a nonzero update
        //   - use near-zero sigma for near-deterministic convergence
        let cfg = DpLambConfig {
            sigma: 1e-6, // near-zero noise for near-deterministic convergence
            learning_rate: 1e-2,
            grad_clip: 10.0, // large clip so gradients are not over-compressed
            weight_decay: 0.0,
            min_trust_ratio: 0.1, // ensure nonzero update at all points
            max_trust_ratio: 10.0,
            ..DpLambConfig::default()
        };
        let opt = DpLamb::new(cfg).expect("ok");
        // Start at [10, -10] (far from optimum [2, -1]).
        let mut state = DpLamb::init_state(2, &[10.0_f64, -10.0]).expect("ok");
        let mut handle = make_handle();

        for _ in 0..200 {
            let th0 = state.theta[0];
            let th1 = state.theta[1];
            // Gradient of the quadratic (batch=1, single sample).
            let grad = vec![2.0 * (th0 - 2.0), 2.0 * (th1 + 1.0)];
            opt.step(&mut state, &grad, 1, &mut handle).expect("ok");
        }

        // After 200 steps, theta should be moving toward [2, -1].
        assert!(
            (state.theta[0] - 2.0).abs() < 1.0,
            "theta[0]={:.4} should approach 2.0",
            state.theta[0]
        );
        assert!(
            (state.theta[1] + 1.0).abs() < 1.0,
            "theta[1]={:.4} should approach -1.0",
            state.theta[1]
        );
    }
}
