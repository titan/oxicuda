//! DP-Adam optimizer.
//!
//! Combines per-sample gradient clipping with Gaussian noise injection and
//! Adam moment updates.  This is the standard approach used in differentially
//! private deep learning (Abadi et al. 2016, extended with Adam).
//!
//! # Algorithm (one step)
//! Given per-sample gradients g₁, …, g_B (each ∈ ℝ^p):
//! 1. Clip: g̃ᵢ = gᵢ · min(1, C / ‖gᵢ‖₂)  (L2 clip to bound C).
//! 2. Sum: G = Σ g̃ᵢ.
//! 3. Noise: G̃ = G + N(0, σ²C²·I).
//! 4. Average: ḡ = G̃ / B.
//! 5. Adam moments:
//!    - m_t = β₁ · m_{t-1} + (1 − β₁) · ḡ
//!    - v_t = β₂ · v_{t-1} + (1 − β₂) · ḡ²
//! 6. Bias-correct: m̂ = m_t / (1 − β₁ᵗ), v̂ = v_t / (1 − β₂ᵗ).
//! 7. Update: θ_{t+1} = θ_t − lr · m̂ / (√v̂ + ε_adam).

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for DP-Adam.
#[derive(Debug, Clone)]
pub struct DpAdamConfig {
    /// Gaussian noise multiplier σ (noise std = σ · grad_clip).
    pub sigma: f64,
    /// Per-sample gradient L2 clipping bound C > 0.
    pub grad_clip: f64,
    /// Learning rate η > 0.
    pub learning_rate: f64,
    /// Adam first-moment decay β₁ ∈ (0, 1).
    pub beta1: f64,
    /// Adam second-moment decay β₂ ∈ (0, 1).
    pub beta2: f64,
    /// Adam numerical stability ε > 0.
    pub epsilon_adam: f64,
}

impl Default for DpAdamConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            grad_clip: 1.0,
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon_adam: 1e-8,
        }
    }
}

impl DpAdamConfig {
    /// Construct and validate a `DpAdamConfig`.
    ///
    /// # Errors
    /// Returns `InvalidParameter` for any out-of-range value.
    pub fn new(
        sigma: f64,
        grad_clip: f64,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon_adam: f64,
    ) -> PrivacyResult<Self> {
        if sigma <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        if grad_clip <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "grad_clip must be positive, got {grad_clip}"
            )));
        }
        if learning_rate <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "learning_rate must be positive, got {learning_rate}"
            )));
        }
        if !(beta1 > 0.0 && beta1 < 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "beta1 must be in (0,1), got {beta1}"
            )));
        }
        if !(beta2 > 0.0 && beta2 < 1.0) {
            return Err(PrivacyError::InvalidParameter(format!(
                "beta2 must be in (0,1), got {beta2}"
            )));
        }
        if epsilon_adam <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon_adam must be positive, got {epsilon_adam}"
            )));
        }
        Ok(Self {
            sigma,
            grad_clip,
            learning_rate,
            beta1,
            beta2,
            epsilon_adam,
        })
    }
}

/// Mutable state for DP-Adam.
#[derive(Debug)]
pub struct DpAdamState {
    /// Current parameter vector θ.
    pub params: Vec<f64>,
    /// First moment estimate m.
    m: Vec<f64>,
    /// Second moment estimate v.
    v: Vec<f64>,
    /// Current step count (1-based after first update).
    pub t: usize,
}

impl DpAdamState {
    /// Initialise DP-Adam state with zero params and moments.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            params: vec![0.0; n_params],
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            t: 0,
        }
    }

    /// Return the number of model parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Execute one DP-Adam step given a batch of per-sample gradients.
    ///
    /// # Arguments
    /// - `per_sample_grads`: flat array of shape `[batch_size × n_params]`.
    ///   Row `i` contains gradient for sample `i`.
    /// - `batch_size`: number of samples in the batch.
    /// - `cfg`: DP-Adam configuration.
    /// - `rng`: LCG for Gaussian noise.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `per_sample_grads.len() != batch_size * n_params`.
    /// - `InvalidParameter` if `batch_size == 0`.
    pub fn step(
        &mut self,
        per_sample_grads: &[f64],
        batch_size: usize,
        cfg: &DpAdamConfig,
        rng: &mut LcgRng,
    ) -> PrivacyResult<()> {
        let n = self.params.len();
        if batch_size == 0 {
            return Err(PrivacyError::InvalidParameter(
                "batch_size must be ≥ 1".into(),
            ));
        }
        if per_sample_grads.len() != batch_size * n {
            return Err(PrivacyError::DimensionMismatch {
                expected: batch_size * n,
                got: per_sample_grads.len(),
            });
        }

        // Step 1 & 2: per-sample clip + sum.
        let mut sum_clipped = vec![0.0f64; n];
        for b in 0..batch_size {
            let start = b * n;
            let sample_grad = &per_sample_grads[start..start + n];
            let norm_sq: f64 = sample_grad.iter().map(|&g| g * g).sum();
            let norm = norm_sq.sqrt().max(f64::EPSILON);
            let scale = (cfg.grad_clip / norm).min(1.0);
            for j in 0..n {
                sum_clipped[j] += sample_grad[j] * scale;
            }
        }

        // Step 3: add Gaussian noise N(0, σ²C²·I).
        let noise_std = cfg.sigma * cfg.grad_clip;
        let mut i = 0;
        while i < n {
            let (z1, z2) = rng.normal_pair();
            sum_clipped[i] += z1 * noise_std;
            if i + 1 < n {
                sum_clipped[i + 1] += z2 * noise_std;
            }
            i += 2;
        }

        // Step 4: average over batch.
        let batch_f = batch_size as f64;
        for v in sum_clipped.iter_mut() {
            *v /= batch_f;
        }
        let avg_grad = sum_clipped;

        // Step 5: Adam moment updates.
        self.t += 1;
        let b1 = cfg.beta1;
        let b2 = cfg.beta2;
        for (j, &g) in avg_grad.iter().enumerate().take(n) {
            self.m[j] = b1 * self.m[j] + (1.0 - b1) * g;
            self.v[j] = b2 * self.v[j] + (1.0 - b2) * g * g;
        }

        // Step 6 & 7: bias-corrected Adam update.
        let t_f = self.t as f64;
        let bias1 = 1.0 - b1.powf(t_f);
        let bias2 = 1.0 - b2.powf(t_f);
        for j in 0..n {
            let m_hat = self.m[j] / bias1;
            let v_hat = self.v[j] / bias2;
            self.params[j] -= cfg.learning_rate * m_hat / (v_hat.sqrt() + cfg.epsilon_adam);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_adam_step_increments_t() {
        let cfg = DpAdamConfig::default();
        let mut rng = LcgRng::new(42);
        let mut state = DpAdamState::new(4);
        let grads = vec![0.1f64; 4]; // batch_size=1
        state.step(&grads, 1, &cfg, &mut rng).expect("ok");
        assert_eq!(state.t, 1);
    }

    #[test]
    fn test_dp_adam_params_change_after_step() {
        let cfg = DpAdamConfig::default();
        let mut rng = LcgRng::new(7);
        let mut state = DpAdamState::new(3);
        let params_before = state.params.clone();
        let grads = vec![1.0f64; 3];
        state.step(&grads, 1, &cfg, &mut rng).expect("ok");
        assert_ne!(state.params, params_before);
    }

    #[test]
    fn test_dp_adam_dimension_mismatch() {
        let cfg = DpAdamConfig::default();
        let mut rng = LcgRng::new(0);
        let mut state = DpAdamState::new(4);
        // batch_size=1, n_params=4, but providing only 3 values → error.
        let bad_grads = vec![1.0f64; 3];
        assert!(state.step(&bad_grads, 1, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_dp_adam_multiple_steps() {
        let cfg = DpAdamConfig::default();
        let mut rng = LcgRng::new(123);
        let mut state = DpAdamState::new(2);
        let grads = vec![0.5f64; 4]; // batch_size=2, n_params=2
        for _ in 0..5 {
            state.step(&grads, 2, &cfg, &mut rng).expect("ok");
        }
        assert_eq!(state.t, 5);
        for &p in &state.params {
            assert!(p.is_finite(), "param must be finite");
        }
    }
}
