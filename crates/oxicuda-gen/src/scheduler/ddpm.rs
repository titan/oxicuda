//! DDPM (Denoising Diffusion Probabilistic Models) scheduler.
//!
//! Implements forward diffusion (adding noise) and reverse diffusion
//! (denoising step) as described in Ho et al. 2020.

use crate::error::{GenError, GenResult};
use crate::scheduler::beta_schedule::BetaSchedule;

// ─── DdpmScheduler ────────────────────────────────────────────────────────────

/// Scheduler for Denoising Diffusion Probabilistic Models (DDPM).
///
/// Provides forward (`add_noise`) and reverse (`step`) diffusion operations.
///
/// # Reference
/// Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
#[derive(Debug, Clone)]
pub struct DdpmScheduler {
    schedule: BetaSchedule,
    num_steps: usize,
    clip_sample: bool,
    clip_range: f32,
}

impl DdpmScheduler {
    /// Create a new DDPM scheduler with 1000 steps and default linear schedule.
    pub fn new(num_steps: usize) -> GenResult<Self> {
        if num_steps == 0 {
            return Err(GenError::EmptyInput("num_steps must be > 0"));
        }
        let schedule = BetaSchedule::linear(num_steps, 0.0001, 0.02)?;
        Ok(Self {
            schedule,
            num_steps,
            clip_sample: true,
            clip_range: 1.0,
        })
    }

    /// Create a scheduler from a precomputed beta schedule.
    pub fn with_schedule(schedule: BetaSchedule) -> Self {
        let n = schedule.num_steps();
        Self {
            schedule,
            num_steps: n,
            clip_sample: true,
            clip_range: 1.0,
        }
    }

    /// Set sample clipping (defaults to `true`, range `[-1, 1]`).
    pub fn with_clip_sample(mut self, clip: bool, range: f32) -> Self {
        self.clip_sample = clip;
        self.clip_range = range;
        self
    }

    /// Forward diffusion: `q(x_t | x_0) = N(√ᾱ_t * x_0, (1-ᾱ_t)*I)`.
    ///
    /// `x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * noise`
    ///
    /// # Errors
    /// - `InvalidTimestep` if `t >= T`
    /// - `DimensionMismatch` if `x0.len() != noise.len()`
    /// - `EmptyInput` if `x0` is empty
    pub fn add_noise(&self, x0: &[f32], noise: &[f32], t: usize) -> GenResult<Vec<f32>> {
        if x0.is_empty() {
            return Err(GenError::EmptyInput("x0 is empty"));
        }
        if t >= self.num_steps {
            return Err(GenError::InvalidTimestep {
                t,
                max_t: self.num_steps,
            });
        }
        if x0.len() != noise.len() {
            return Err(GenError::DimensionMismatch {
                expected: x0.len(),
                got: noise.len(),
            });
        }
        let sqrt_ab = self.schedule.sqrt_alphas_bar()[t];
        let sqrt_one_minus_ab = self.schedule.sqrt_one_minus_alphas_bar()[t];
        let x_t = x0
            .iter()
            .zip(noise)
            .map(|(&x, &n)| sqrt_ab * x + sqrt_one_minus_ab * n)
            .collect();
        Ok(x_t)
    }

    /// Reverse step: `p(x_{t-1} | x_t)` given predicted noise `ε̂`.
    ///
    /// Computes:
    /// 1. `x_0_pred = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t`
    /// 2. `mean = √ᾱ_{t-1} * x_0_pred * β_t/(1-ᾱ_t) + √α_t * (1-ᾱ_{t-1})/(1-ᾱ_t) * x_t`
    /// 3. `σ_t = √(β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t))`
    /// 4. `x_{t-1} = mean + σ_t * z`
    ///
    /// # Errors
    /// - `InvalidTimestep` if `t >= T`
    /// - `DimensionMismatch` on shape mismatch
    /// - `EmptyInput` if inputs are empty
    pub fn step(
        &self,
        eps_hat: &[f32],
        x_t: &[f32],
        t: usize,
        noise: &[f32],
    ) -> GenResult<Vec<f32>> {
        if eps_hat.is_empty() {
            return Err(GenError::EmptyInput("eps_hat is empty"));
        }
        if t >= self.num_steps {
            return Err(GenError::InvalidTimestep {
                t,
                max_t: self.num_steps,
            });
        }
        if eps_hat.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: eps_hat.len(),
                got: x_t.len(),
            });
        }
        if noise.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: noise.len(),
            });
        }

        let beta_t = self.schedule.betas()[t];
        let alpha_t = self.schedule.alphas()[t];
        let alpha_bar_t = self.schedule.alphas_bar()[t];
        let sqrt_one_minus_ab = self.schedule.sqrt_one_minus_alphas_bar()[t];

        // ᾱ_{t-1} — for t=0, use 1.0 (before any noise)
        let alpha_bar_prev = if t == 0 {
            1.0_f32
        } else {
            self.schedule.alphas_bar()[t - 1]
        };

        // Posterior variance: σ² = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        let one_minus_ab = (1.0 - alpha_bar_t).max(1e-10);
        let sigma = if t == 0 {
            0.0
        } else {
            (beta_t * (1.0 - alpha_bar_prev) / one_minus_ab)
                .max(0.0)
                .sqrt()
        };

        // Coefficients for the posterior mean
        let sqrt_ab_prev = alpha_bar_prev.sqrt();
        let coeff_x0 = sqrt_ab_prev * beta_t / one_minus_ab;
        let coeff_xt = alpha_t.sqrt() * (1.0 - alpha_bar_prev) / one_minus_ab;

        let result: Vec<f32> = eps_hat
            .iter()
            .zip(x_t)
            .zip(noise)
            .map(|((&eps, &xt), &z)| {
                // Predict x_0
                let sqrt_ab = alpha_bar_t.sqrt();
                let x0_pred = (xt - sqrt_one_minus_ab * eps) / sqrt_ab.max(1e-10);
                let x0_pred = if self.clip_sample {
                    x0_pred.clamp(-self.clip_range, self.clip_range)
                } else {
                    x0_pred
                };
                // Posterior mean
                let mean = coeff_x0 * x0_pred + coeff_xt * xt;
                mean + sigma * z
            })
            .collect();
        Ok(result)
    }

    /// Predict `x_0` from noisy sample `x_t` and predicted noise `ε̂`.
    ///
    /// `x_0_pred = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t`
    ///
    /// # Errors
    /// - `InvalidTimestep` if `t >= T`
    /// - `DimensionMismatch` on shape mismatch
    pub fn predict_x0(&self, x_t: &[f32], eps_hat: &[f32], t: usize) -> GenResult<Vec<f32>> {
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t is empty"));
        }
        if t >= self.num_steps {
            return Err(GenError::InvalidTimestep {
                t,
                max_t: self.num_steps,
            });
        }
        if x_t.len() != eps_hat.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: eps_hat.len(),
            });
        }
        let sqrt_ab = self.schedule.sqrt_alphas_bar()[t];
        let sqrt_one_minus_ab = self.schedule.sqrt_one_minus_alphas_bar()[t];
        let result = x_t
            .iter()
            .zip(eps_hat)
            .map(|(&xt, &eps)| (xt - sqrt_one_minus_ab * eps) / sqrt_ab.max(1e-10))
            .collect();
        Ok(result)
    }

    /// Return a reference to the underlying beta schedule.
    pub fn schedule(&self) -> &BetaSchedule {
        &self.schedule
    }

    /// Return the number of diffusion steps.
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn new_scheduler_1000_steps() {
        let sched = DdpmScheduler::new(1000).unwrap();
        assert_eq!(sched.num_steps(), 1000);
    }

    #[test]
    fn add_noise_output_shape() {
        let sched = DdpmScheduler::new(100).unwrap();
        let mut rng = make_rng();
        let x0 = randn(&mut rng, 64);
        let noise = randn(&mut rng, 64);
        let x_t = sched.add_noise(&x0, &noise, 50).unwrap();
        assert_eq!(x_t.len(), 64);
    }

    #[test]
    fn add_noise_at_t0_close_to_x0() {
        // At t=0, beta is very small (~0.0001), so x_t ≈ x_0
        let sched = DdpmScheduler::new(1000).unwrap();
        let mut rng = make_rng();
        let x0 = randn(&mut rng, 32);
        let noise = randn(&mut rng, 32);
        let x_t = sched.add_noise(&x0, &noise, 0).unwrap();
        // sqrt(alphas_bar[0]) ≈ sqrt(1 - 0.0001) ≈ 0.99995
        let max_diff: f32 = x0
            .iter()
            .zip(&x_t)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // should not be identical due to small noise contribution
        assert!(max_diff < 0.1, "x_t too far from x0 at t=0: {max_diff}");
    }

    #[test]
    fn add_noise_dimension_mismatch() {
        let sched = DdpmScheduler::new(100).unwrap();
        let x0 = vec![1.0_f32; 10];
        let noise = vec![0.0_f32; 5];
        assert!(matches!(
            sched.add_noise(&x0, &noise, 0),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn add_noise_invalid_timestep() {
        let sched = DdpmScheduler::new(100).unwrap();
        let x0 = vec![1.0_f32; 8];
        let noise = vec![0.0_f32; 8];
        assert!(matches!(
            sched.add_noise(&x0, &noise, 100),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn step_output_shape() {
        let sched = DdpmScheduler::new(100).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let noise = randn(&mut rng, 32);
        let x_prev = sched.step(&eps, &x_t, 50, &noise).unwrap();
        assert_eq!(x_prev.len(), 32);
    }

    #[test]
    fn step_at_t0_no_stochastic_noise() {
        // At t=0, sigma should be 0, so the step is deterministic regardless of z
        let sched = DdpmScheduler::new(100).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 16);
        let x_t = randn(&mut rng, 16);
        let noise1 = randn(&mut rng, 16);
        let noise2 = randn(&mut rng, 16);
        let x1 = sched.step(&eps, &x_t, 0, &noise1).unwrap();
        let x2 = sched.step(&eps, &x_t, 0, &noise2).unwrap();
        let max_diff: f32 = x1
            .iter()
            .zip(&x2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff < 1e-5,
            "t=0 step should be deterministic: {max_diff}"
        );
    }

    #[test]
    fn predict_x0_output_shape() {
        let sched = DdpmScheduler::new(100).unwrap();
        let mut rng = make_rng();
        let x_t = randn(&mut rng, 64);
        let eps = randn(&mut rng, 64);
        let x0 = sched.predict_x0(&x_t, &eps, 50).unwrap();
        assert_eq!(x0.len(), 64);
    }

    #[test]
    fn predict_x0_roundtrip() {
        // If we add noise with true eps, predict_x0 should recover x_0 approximately
        let sched = DdpmScheduler::new(1000).unwrap();
        let mut rng = make_rng();
        let x0: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0 - 0.5).collect();
        let noise = randn(&mut rng, 32);
        let t = 10;
        let x_t = sched.add_noise(&x0, &noise, t).unwrap();
        // Predict x0 using the true noise (oracle)
        let x0_pred = sched.predict_x0(&x_t, &noise, t).unwrap();
        let max_diff: f32 = x0
            .iter()
            .zip(&x0_pred)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 0.01, "predict_x0 roundtrip error: {max_diff}");
    }

    #[test]
    fn step_dimension_mismatch() {
        let sched = DdpmScheduler::new(100).unwrap();
        let eps = vec![0.0_f32; 10];
        let x_t = vec![0.0_f32; 5];
        let noise = vec![0.0_f32; 10];
        assert!(matches!(
            sched.step(&eps, &x_t, 50, &noise),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn step_all_outputs_finite() {
        let sched = DdpmScheduler::new(100).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let noise = randn(&mut rng, 32);
        for t in 0..10 {
            let x_prev = sched.step(&eps, &x_t, t, &noise).unwrap();
            assert!(x_prev.iter().all(|v| v.is_finite()), "non-finite at t={t}");
        }
    }

    #[test]
    fn with_schedule_custom() {
        let bs = BetaSchedule::linear(50, 0.001, 0.01).unwrap();
        let sched = DdpmScheduler::with_schedule(bs);
        assert_eq!(sched.num_steps(), 50);
    }
}
