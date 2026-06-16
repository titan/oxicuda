//! Consistency Models scheduler.
//!
//! Implements Consistency Models (Song et al. 2023, ICML) for one-step and
//! multi-step generation via consistency functions that map any point on a
//! diffusion trajectory to its starting point.
//!
//! The consistency function uses EDM-style preconditioning:
//! `f(x, t) = c_skip(t) · x + c_out(t) · F_θ(x, t)`.
//!
//! # Reference
//! Song et al., "Consistency Models", ICML 2023.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── ConsistencyConfig ────────────────────────────────────────────────────────

/// Configuration for the Consistency Models sigma schedule and preconditioning.
///
/// Defines the noise boundary parameters (σ_min, σ_max), data distribution
/// scale (σ_data), discretisation step count, and schedule curvature (ρ).
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    /// Lower noise boundary ε (default 0.002). Must be > 0.
    pub sigma_min: f32,
    /// Upper noise boundary T (default 80.0). Must be > sigma_min.
    pub sigma_max: f32,
    /// Data distribution standard deviation (default 0.5). Must be > 0.
    pub sigma_data: f32,
    /// Number of discretisation steps N (default 40). Must be ≥ 1.
    pub n_steps: usize,
    /// Schedule curvature exponent ρ (default 7.0). Must be > 0.
    pub rho: f32,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            sigma_data: 0.5,
            n_steps: 40,
            rho: 7.0,
        }
    }
}

// ─── ConsistencyScheduler ─────────────────────────────────────────────────────

/// Scheduler implementing the Consistency Models framework.
///
/// Manages the σ schedule, EDM-style preconditioning scalars (c_skip, c_out),
/// the consistency function output, single-step and multi-step sampling, and
/// the consistency distillation loss.
///
/// # Reference
/// Song et al., "Consistency Models", ICML 2023, Sections 2 and 3.1.
#[derive(Debug, Clone)]
pub struct ConsistencyScheduler {
    /// Configuration that governs all schedule parameters.
    pub cfg: ConsistencyConfig,
    /// Precomputed sigma schedule with `n_steps` values.
    ///
    /// `sigmas[0]` = σ_max (largest noise level), `sigmas[n-1]` ≈ σ_min.
    pub sigmas: Vec<f32>,
}

impl ConsistencyScheduler {
    /// Build a new [`ConsistencyScheduler`] from the given configuration.
    ///
    /// Validates all configuration fields and constructs the σ schedule
    /// using the EDM power-law formula (Karras et al. 2022 / Song et al. 2023):
    ///
    /// ```text
    /// σᵢ = (σ_max^{1/ρ} + i/(N-1) * (σ_min^{1/ρ} - σ_max^{1/ρ}))^ρ
    /// ```
    ///
    /// # Errors
    /// - [`GenError::InvalidGuidanceScale`] if any scalar parameter is invalid.
    /// - [`GenError::EmptyInput`] if `n_steps == 0`.
    pub fn new(cfg: ConsistencyConfig) -> GenResult<Self> {
        if cfg.sigma_min <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(cfg.sigma_min));
        }
        if cfg.sigma_max <= cfg.sigma_min {
            return Err(GenError::InvalidGuidanceScale(cfg.sigma_max));
        }
        if cfg.sigma_data <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(cfg.sigma_data));
        }
        if cfg.n_steps == 0 {
            return Err(GenError::EmptyInput("n_steps must be >= 1"));
        }
        if cfg.rho <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(cfg.rho));
        }

        let sigmas = if cfg.n_steps == 1 {
            vec![cfg.sigma_max]
        } else {
            let inv_rho = 1.0 / cfg.rho;
            let max_pow = cfg.sigma_max.powf(inv_rho);
            let min_pow = cfg.sigma_min.powf(inv_rho);
            let n = cfg.n_steps;
            (0..n)
                .map(|i| {
                    let frac = i as f32 / (n - 1) as f32;
                    let inner = max_pow + frac * (min_pow - max_pow);
                    inner.powf(cfg.rho)
                })
                .collect()
        };

        Ok(Self { cfg, sigmas })
    }

    /// Skip-connection coefficient c_skip(σ).
    ///
    /// `c_skip(σ) = σ_data² / (σ² + σ_data²)`
    ///
    /// At σ = σ_data this equals 0.5; as σ → 0 it approaches 1; as σ → ∞
    /// it approaches 0 (the network must do all the work at high noise).
    #[inline]
    #[must_use]
    pub fn c_skip(&self, sigma: f32) -> f32 {
        let sd2 = self.cfg.sigma_data * self.cfg.sigma_data;
        sd2 / (sigma * sigma + sd2)
    }

    /// Output-scaling coefficient c_out(σ).
    ///
    /// `c_out(σ) = σ · σ_data / √(σ² + σ_data²)`
    ///
    /// At σ = 0 this is 0 (no correction needed when there is no noise).
    #[inline]
    #[must_use]
    pub fn c_out(&self, sigma: f32) -> f32 {
        let sd = self.cfg.sigma_data;
        let sd2 = sd * sd;
        sigma * sd / (sigma * sigma + sd2).sqrt()
    }

    /// Apply the consistency function: `output = c_skip(σ)·x + c_out(σ)·f_out`.
    ///
    /// Implements Eq. (2) in Song et al. 2023: the consistency function maps
    /// any (x, σ) on the diffusion trajectory to the estimated clean sample.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x` is empty.
    /// - [`GenError::DimensionMismatch`] if `x.len() != f_out.len()`.
    pub fn consistency_output(&self, x: &[f32], sigma: f32, f_out: &[f32]) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if x.len() != f_out.len() {
            return Err(GenError::DimensionMismatch {
                expected: x.len(),
                got: f_out.len(),
            });
        }
        let cs = self.c_skip(sigma);
        let co = self.c_out(sigma);
        let out = x
            .iter()
            .zip(f_out)
            .map(|(&xi, &fi)| cs * xi + co * fi)
            .collect();
        Ok(out)
    }

    /// Single-step sample: one forward pass of the consistency function.
    ///
    /// Equivalent to [`Self::consistency_output`] at `sigma_t`. Produces the
    /// denoised estimate in a single model evaluation.
    ///
    /// # Errors
    /// Same as [`Self::consistency_output`].
    pub fn single_step_sample(
        &self,
        x_t: &[f32],
        sigma_t: f32,
        f_out: &[f32],
    ) -> GenResult<Vec<f32>> {
        self.consistency_output(x_t, sigma_t, f_out)
    }

    /// Multi-step sample following the iterative refinement procedure.
    ///
    /// Starting from x_t at σ_max, alternates between:
    /// 1. Applying the consistency function to obtain a clean estimate x_hat.
    /// 2. Adding scaled noise at the *next* noise level σ_{i+1} to re-inject
    ///    stochasticity (except at the final step).
    ///
    /// Noise scale: `√max(σ_{i+1}² - σ_min², 0)` to keep noise within bounds.
    ///
    /// # Arguments
    /// - `x_t` — starting noisy input (shape: any flat `f32` buffer).
    /// - `f_theta` — closure `|x: &[f32], sigma: f32| -> Vec<f32>` representing
    ///   the consistency model network.
    /// - `rng` — deterministic LCG RNG for noise injection.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_t` is empty.
    /// - Propagates errors from [`Self::consistency_output`].
    pub fn multi_step_sample<F>(
        &self,
        x_t: &[f32],
        mut f_theta: F,
        rng: &mut LcgRng,
    ) -> GenResult<Vec<f32>>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t is empty"));
        }

        let mut x = x_t.to_vec();
        let sigma_min = self.cfg.sigma_min;
        let n = self.cfg.n_steps;

        for i in 0..n {
            let sigma_i = self.sigmas[i];
            let f_out = f_theta(&x, sigma_i);
            let x_hat = self.consistency_output(&x, sigma_i, &f_out)?;

            if i < n - 1 {
                let sigma_next = self.sigmas[i + 1];
                let noise_var = (sigma_next * sigma_next - sigma_min * sigma_min).max(0.0);
                let noise_scale = noise_var.sqrt();
                x = x_hat
                    .iter()
                    .map(|&xh| {
                        let (z, _) = rng.next_normal_pair();
                        xh + noise_scale * z
                    })
                    .collect();
            } else {
                x = x_hat;
            }
        }

        Ok(x)
    }

    /// Consistency distillation loss between student and EMA-teacher outputs.
    ///
    /// Computes the mean squared error between `f_theta` (student) and
    /// `f_theta_ema` (exponential moving average teacher) outputs, which is
    /// the objective used for training consistency models (Song et al. 2023,
    /// Section 3.1).
    ///
    /// `loss = (1/n) Σᵢ (f_theta[i] - f_theta_ema[i])²`
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if either slice is empty.
    /// - [`GenError::DimensionMismatch`] if lengths differ.
    pub fn consistency_loss(&self, f_theta: &[f32], f_theta_ema: &[f32]) -> GenResult<f32> {
        if f_theta.is_empty() {
            return Err(GenError::EmptyInput("f_theta is empty"));
        }
        if f_theta.len() != f_theta_ema.len() {
            return Err(GenError::DimensionMismatch {
                expected: f_theta.len(),
                got: f_theta_ema.len(),
            });
        }
        let n = f_theta.len() as f32;
        let mse = f_theta
            .iter()
            .zip(f_theta_ema)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / n;
        Ok(mse)
    }

    /// Return σ at a given discretisation step index.
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `step >= n_steps`.
    pub fn sigma_at(&self, step: usize) -> GenResult<f32> {
        if step >= self.cfg.n_steps {
            return Err(GenError::InvalidTimestep {
                t: step,
                max_t: self.cfg.n_steps,
            });
        }
        Ok(self.sigmas[step])
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 0.01;
    const TINY: f32 = 1e-6;

    fn default_sched() -> ConsistencyScheduler {
        ConsistencyScheduler::new(ConsistencyConfig::default()).expect("value should be present")
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Schedule shape and boundary tests ────────────────────────────────────

    #[test]
    fn sigma_schedule_first_is_max() {
        let s = default_sched();
        let diff = (s.sigmas[0] - s.cfg.sigma_max).abs();
        assert!(
            diff < EPS,
            "sigmas[0]={} vs sigma_max={}",
            s.sigmas[0],
            s.cfg.sigma_max
        );
    }

    #[test]
    fn sigma_schedule_last_is_min() {
        let s = default_sched();
        let n = s.cfg.n_steps;
        let diff = (s.sigmas[n - 1] - s.cfg.sigma_min).abs();
        assert!(
            diff < EPS,
            "sigmas[n-1]={} vs sigma_min={}",
            s.sigmas[n - 1],
            s.cfg.sigma_min
        );
    }

    #[test]
    fn sigma_schedule_monotone() {
        let s = default_sched();
        for w in s.sigmas.windows(2) {
            assert!(
                w[1] <= w[0] + TINY,
                "sigma schedule not non-increasing: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    // ── Preconditioning scalar tests ──────────────────────────────────────────

    #[test]
    fn c_skip_at_sigma_data_is_half() {
        let s = default_sched();
        let sigma_data = s.cfg.sigma_data;
        let cs = s.c_skip(sigma_data);
        assert!(
            (cs - 0.5).abs() < TINY,
            "c_skip(sigma_data) = {cs}, expected 0.5"
        );
    }

    #[test]
    fn c_out_at_zero_is_zero() {
        let s = default_sched();
        let co = s.c_out(0.0);
        assert!(co.abs() < TINY, "c_out(0) = {co}, expected 0.0");
    }

    #[test]
    fn c_skip_monotone_decreasing_in_sigma() {
        // c_skip = σ_data² / (σ² + σ_data²) is strictly decreasing in σ
        let s = default_sched();
        let small_sigma = 0.1_f32;
        let large_sigma = 10.0_f32;
        let cs_small = s.c_skip(small_sigma);
        let cs_large = s.c_skip(large_sigma);
        assert!(
            cs_large < cs_small,
            "c_skip should decrease with sigma: c_skip({large_sigma})={cs_large} >= c_skip({small_sigma})={cs_small}"
        );
    }

    // ── consistency_output tests ──────────────────────────────────────────────

    #[test]
    fn consistency_output_shape() {
        let s = default_sched();
        let x = vec![1.0_f32; 32];
        let f_out = vec![0.5_f32; 32];
        let out = s
            .consistency_output(&x, 1.0, &f_out)
            .expect("consistency_output should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn consistency_output_f_out_zero() {
        // When f_out = 0, output = c_skip(σ) * x
        let s = default_sched();
        let sigma = 2.0_f32;
        let x = vec![3.0_f32, -1.0, 0.5];
        let f_out = vec![0.0_f32; 3];
        let out = s
            .consistency_output(&x, sigma, &f_out)
            .expect("consistency_output should succeed");
        let cs = s.c_skip(sigma);
        for (&o, &xi) in out.iter().zip(&x) {
            let expected = cs * xi;
            assert!(
                (o - expected).abs() < TINY,
                "output[i]={o} expected {expected}"
            );
        }
    }

    #[test]
    fn consistency_output_dim_mismatch() {
        let s = default_sched();
        let x = vec![1.0_f32; 4];
        let f_out = vec![0.5_f32; 8];
        assert!(matches!(
            s.consistency_output(&x, 1.0, &f_out),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn consistency_output_empty_x() {
        let s = default_sched();
        assert!(matches!(
            s.consistency_output(&[], 1.0, &[]),
            Err(GenError::EmptyInput(_))
        ));
    }

    // ── single_step_sample tests ──────────────────────────────────────────────

    #[test]
    fn single_step_sample_shape() {
        let s = default_sched();
        let x = vec![1.0_f32; 16];
        let f_out = vec![0.0_f32; 16];
        let out = s
            .single_step_sample(&x, 5.0, &f_out)
            .expect("single_step_sample should succeed");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn single_step_equals_consistency_output() {
        let s = default_sched();
        let sigma = 3.0_f32;
        let x = vec![1.0_f32, 2.0, 3.0];
        let f_out = vec![0.1_f32, 0.2, 0.3];
        let a = s
            .single_step_sample(&x, sigma, &f_out)
            .expect("single_step_sample should succeed");
        let b = s
            .consistency_output(&x, sigma, &f_out)
            .expect("consistency_output should succeed");
        for (&ai, &bi) in a.iter().zip(&b) {
            assert!((ai - bi).abs() < TINY, "single_step != consistency_output");
        }
    }

    // ── multi_step_sample tests ───────────────────────────────────────────────

    #[test]
    fn multi_step_shape() {
        let s = default_sched();
        let x = vec![0.5_f32; 8];
        let mut rng = make_rng();
        // f_theta returns zeros (perfect denoiser)
        let out = s
            .multi_step_sample(&x, |_, _| vec![0.0_f32; 8], &mut rng)
            .expect("value should be present");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn multi_step_n_steps_1_no_noise() {
        // With n_steps=1, there is no noise injection (no i < n-1 branch reached)
        let cfg = ConsistencyConfig {
            n_steps: 1,
            ..ConsistencyConfig::default()
        };
        let s = ConsistencyScheduler::new(cfg).expect("new should succeed");
        let x = vec![1.0_f32; 4];
        let mut rng = make_rng();
        // f_theta = identity: f_out = x
        let out = s
            .multi_step_sample(&x, |xi, _| xi.to_vec(), &mut rng)
            .expect("value should be present");
        assert_eq!(out.len(), 4);
        // Result is consistency_output with f_out = x_t at sigma_max
        let sigma = s.sigmas[0];
        let cs = s.c_skip(sigma);
        let co = s.c_out(sigma);
        for (&o, &xi) in out.iter().zip(&x) {
            let expected = cs * xi + co * xi;
            assert!(
                (o - expected).abs() < TINY,
                "n_steps=1: output={o} expected {expected}"
            );
        }
    }

    #[test]
    fn multi_step_empty_x() {
        let s = default_sched();
        let mut rng = make_rng();
        assert!(matches!(
            s.multi_step_sample(&[], |_, _| vec![], &mut rng),
            Err(GenError::EmptyInput(_))
        ));
    }

    // ── consistency_loss tests ────────────────────────────────────────────────

    #[test]
    fn consistency_loss_zero() {
        let s = default_sched();
        let v = vec![1.0_f32, 2.0, 3.0];
        let loss = s
            .consistency_loss(&v, &v)
            .expect("consistency_loss should succeed");
        assert!(
            loss.abs() < TINY,
            "loss with identical inputs should be 0, got {loss}"
        );
    }

    #[test]
    fn consistency_loss_mse() {
        let s = default_sched();
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        // MSE = (1 + 4 + 9) / 3 = 14/3
        let expected = 14.0_f32 / 3.0;
        let loss = s
            .consistency_loss(&a, &b)
            .expect("consistency_loss should succeed");
        assert!(
            (loss - expected).abs() < 1e-5,
            "MSE mismatch: {loss} vs {expected}"
        );
    }

    #[test]
    fn consistency_loss_empty() {
        let s = default_sched();
        assert!(matches!(
            s.consistency_loss(&[], &[]),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn consistency_loss_dim_mismatch() {
        let s = default_sched();
        let a = vec![1.0_f32; 4];
        let b = vec![1.0_f32; 6];
        assert!(matches!(
            s.consistency_loss(&a, &b),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    // ── sigma_at tests ────────────────────────────────────────────────────────

    #[test]
    fn sigma_at_valid() {
        let s = default_sched();
        let v = s.sigma_at(0).expect("sigma_at should succeed");
        assert!((v - s.sigmas[0]).abs() < TINY);
    }

    #[test]
    fn sigma_at_out_of_range() {
        let s = default_sched();
        let n = s.cfg.n_steps;
        assert!(matches!(
            s.sigma_at(n),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    // ── Error-path construction tests ─────────────────────────────────────────

    #[test]
    fn err_sigma_min_zero() {
        let cfg = ConsistencyConfig {
            sigma_min: 0.0,
            ..ConsistencyConfig::default()
        };
        assert!(matches!(
            ConsistencyScheduler::new(cfg),
            Err(GenError::InvalidGuidanceScale(_))
        ));
    }

    #[test]
    fn err_sigma_max_le_min() {
        let cfg = ConsistencyConfig {
            sigma_min: 1.0,
            sigma_max: 0.5,
            ..ConsistencyConfig::default()
        };
        assert!(matches!(
            ConsistencyScheduler::new(cfg),
            Err(GenError::InvalidGuidanceScale(_))
        ));
    }

    #[test]
    fn err_n_steps_zero() {
        let cfg = ConsistencyConfig {
            n_steps: 0,
            ..ConsistencyConfig::default()
        };
        assert!(matches!(
            ConsistencyScheduler::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn default_config_values() {
        let cfg = ConsistencyConfig::default();
        assert!((cfg.sigma_min - 0.002).abs() < TINY, "sigma_min default");
        assert!((cfg.sigma_max - 80.0).abs() < TINY, "sigma_max default");
        assert!((cfg.sigma_data - 0.5).abs() < TINY, "sigma_data default");
        assert_eq!(cfg.n_steps, 40, "n_steps default");
        assert!((cfg.rho - 7.0).abs() < TINY, "rho default");
    }
}
