//! DDIM (Denoising Diffusion Implicit Models) scheduler.
//!
//! Implements the DDIM reverse diffusion step from Song et al. 2021.
//! Supports both deterministic (η=0) and stochastic (η=1, ~DDPM) modes.

use crate::error::{GenError, GenResult};
use crate::scheduler::beta_schedule::BetaSchedule;

// ─── DdimScheduler ────────────────────────────────────────────────────────────

/// Scheduler for Denoising Diffusion Implicit Models (DDIM).
///
/// Supports subsampling of training timesteps for faster inference,
/// and a controllable stochasticity parameter η.
///
/// # Reference
/// Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
#[derive(Debug, Clone)]
pub struct DdimScheduler {
    schedule: BetaSchedule,
    eta: f32,
    num_train_steps: usize,
    num_inference_steps: usize,
    timesteps: Vec<usize>,
}

impl DdimScheduler {
    /// Create a new DDIM scheduler.
    ///
    /// # Arguments
    /// - `num_train_steps`: Total training steps (e.g. 1000).
    /// - `num_inference_steps`: Number of inference steps (e.g. 50).
    /// - `eta`: Stochasticity parameter. 0 = deterministic, 1 = DDPM-like.
    ///
    /// # Errors
    /// - `EmptyInput` if any count is 0
    /// - `InvalidGuidanceScale` if `eta < 0`
    pub fn new(num_train_steps: usize, num_inference_steps: usize, eta: f32) -> GenResult<Self> {
        if num_train_steps == 0 {
            return Err(GenError::EmptyInput("num_train_steps must be > 0"));
        }
        if num_inference_steps == 0 {
            return Err(GenError::EmptyInput("num_inference_steps must be > 0"));
        }
        if eta < 0.0 {
            return Err(GenError::InvalidGuidanceScale(eta));
        }
        let schedule = BetaSchedule::linear(num_train_steps, 0.0001, 0.02)?;
        let timesteps = Self::compute_timesteps(num_train_steps, num_inference_steps);
        Ok(Self {
            schedule,
            eta,
            num_train_steps,
            num_inference_steps,
            timesteps,
        })
    }

    /// Create a scheduler with a custom beta schedule.
    pub fn with_custom_schedule(
        schedule: BetaSchedule,
        num_inference_steps: usize,
        eta: f32,
    ) -> GenResult<Self> {
        if num_inference_steps == 0 {
            return Err(GenError::EmptyInput("num_inference_steps must be > 0"));
        }
        if eta < 0.0 {
            return Err(GenError::InvalidGuidanceScale(eta));
        }
        let n = schedule.num_steps();
        let timesteps = Self::compute_timesteps(n, num_inference_steps);
        Ok(Self {
            num_train_steps: n,
            num_inference_steps,
            eta,
            timesteps,
            schedule,
        })
    }

    /// Compute subsampled timestep indices (uniform spacing in training steps).
    ///
    /// Produces `num_inference_steps` evenly spaced indices in `[0, T)`.
    fn compute_timesteps(num_train: usize, num_infer: usize) -> Vec<usize> {
        let step_ratio = num_train / num_infer.max(1);
        let steps: Vec<usize> = (0..num_infer)
            .map(|i| ((num_infer - 1 - i) * step_ratio).min(num_train - 1))
            .collect();
        steps
    }

    /// Returns the number of training timesteps this scheduler was built with.
    pub fn num_train_steps(&self) -> usize {
        self.num_train_steps
    }

    /// DDIM reverse step.
    ///
    /// Given predicted noise `eps` and current noisy sample `x_t` at
    /// `timesteps[step_idx]`, produces `x_{t-1}`.
    ///
    /// Algorithm:
    /// 1. `x_0_pred = (x_t - √(1-ᾱ_t) * eps) / √ᾱ_t`
    /// 2. `σ_t = η * √((1-ᾱ_{t-1})/(1-ᾱ_t)) * √(1 - ᾱ_t/ᾱ_{t-1})`
    /// 3. `dir = √(1-ᾱ_{t-1} - σ²) * eps`
    /// 4. `x_{t-1} = √ᾱ_{t-1} * x_0_pred + dir + σ * z`
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step_idx >= num_inference_steps`
    /// - `DimensionMismatch` on shape mismatch
    pub fn step(
        &self,
        eps: &[f32],
        x_t: &[f32],
        step_idx: usize,
        noise: &[f32],
    ) -> GenResult<Vec<f32>> {
        if eps.is_empty() {
            return Err(GenError::EmptyInput("eps is empty"));
        }
        if step_idx >= self.num_inference_steps {
            return Err(GenError::InvalidTimestep {
                t: step_idx,
                max_t: self.num_inference_steps,
            });
        }
        if eps.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: eps.len(),
                got: x_t.len(),
            });
        }
        if noise.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: noise.len(),
            });
        }

        let t = self.timesteps[step_idx];
        let alpha_bar_t = self.schedule.alphas_bar()[t];
        let sqrt_ab_t = alpha_bar_t.sqrt();
        let sqrt_one_minus_ab_t = self.schedule.sqrt_one_minus_alphas_bar()[t];

        // ᾱ_{t-1}: use schedule at previous timestep or 1.0 if last step
        let alpha_bar_prev = if step_idx + 1 < self.num_inference_steps {
            let t_prev = self.timesteps[step_idx + 1];
            self.schedule.alphas_bar()[t_prev]
        } else {
            1.0_f32
        };

        let sqrt_ab_prev = alpha_bar_prev.sqrt();

        // Sigma_t = eta * sqrt((1-ab_prev)/(1-ab_t)) * sqrt(1 - ab_t/ab_prev)
        let one_minus_ab_t = (1.0 - alpha_bar_t).max(1e-10);
        let one_minus_ab_prev = (1.0 - alpha_bar_prev).max(0.0);
        let ratio = alpha_bar_t / alpha_bar_prev.max(1e-10);
        let sigma = if alpha_bar_prev >= 1.0 - 1e-7 {
            0.0
        } else {
            self.eta * (one_minus_ab_prev / one_minus_ab_t).sqrt() * (1.0 - ratio).max(0.0).sqrt()
        };

        // Direction coefficient: sqrt(1 - ab_prev - sigma^2)
        let dir_coeff = (one_minus_ab_prev - sigma * sigma).max(0.0).sqrt();

        let result: Vec<f32> = eps
            .iter()
            .zip(x_t)
            .zip(noise)
            .map(|((&e, &xt), &z)| {
                let x0_pred = (xt - sqrt_one_minus_ab_t * e) / sqrt_ab_t.max(1e-10);
                sqrt_ab_prev * x0_pred + dir_coeff * e + sigma * z
            })
            .collect();
        Ok(result)
    }

    /// Return the subsampled timestep indices.
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Return the eta (stochasticity) parameter.
    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// Return the number of inference steps.
    pub fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Return a reference to the underlying beta schedule.
    pub fn schedule(&self) -> &BetaSchedule {
        &self.schedule
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(123)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn new_scheduler_valid() {
        let sched = DdimScheduler::new(1000, 50, 0.0).unwrap();
        assert_eq!(sched.num_inference_steps(), 50);
        assert_eq!(sched.eta(), 0.0);
    }

    #[test]
    fn timesteps_count() {
        let sched = DdimScheduler::new(1000, 50, 0.0).unwrap();
        assert_eq!(sched.timesteps().len(), 50);
    }

    #[test]
    fn timesteps_are_in_valid_range() {
        let sched = DdimScheduler::new(1000, 50, 0.0).unwrap();
        for &t in sched.timesteps() {
            assert!(t < 1000, "timestep {t} out of range");
        }
    }

    #[test]
    fn deterministic_at_eta_zero() {
        // With eta=0 and same inputs, two calls with different noise should give same result
        let sched = DdimScheduler::new(1000, 10, 0.0).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let noise1 = randn(&mut rng, 32);
        let noise2 = randn(&mut rng, 32);
        let out1 = sched.step(&eps, &x_t, 0, &noise1).unwrap();
        let out2 = sched.step(&eps, &x_t, 0, &noise2).unwrap();
        let max_diff: f32 = out1
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff < 1e-5,
            "eta=0 should be deterministic, diff={max_diff}"
        );
    }

    #[test]
    fn step_output_shape() {
        let sched = DdimScheduler::new(1000, 10, 0.5).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 64);
        let x_t = randn(&mut rng, 64);
        let noise = randn(&mut rng, 64);
        let out = sched.step(&eps, &x_t, 0, &noise).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn step_invalid_step_idx() {
        let sched = DdimScheduler::new(1000, 10, 0.0).unwrap();
        let eps = vec![0.0_f32; 8];
        let x_t = vec![0.0_f32; 8];
        let noise = vec![0.0_f32; 8];
        assert!(matches!(
            sched.step(&eps, &x_t, 10, &noise),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn step_dimension_mismatch() {
        let sched = DdimScheduler::new(1000, 10, 0.0).unwrap();
        let eps = vec![0.0_f32; 8];
        let x_t = vec![0.0_f32; 4];
        let noise = vec![0.0_f32; 8];
        assert!(matches!(
            sched.step(&eps, &x_t, 0, &noise),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn step_output_finite() {
        let sched = DdimScheduler::new(1000, 10, 1.0).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let noise = randn(&mut rng, 32);
        for step_idx in 0..10 {
            let out = sched.step(&eps, &x_t, step_idx, &noise).unwrap();
            assert!(
                out.iter().all(|v| v.is_finite()),
                "non-finite at step {step_idx}"
            );
        }
    }

    #[test]
    fn timesteps_uniform_subsampling() {
        let sched = DdimScheduler::new(1000, 10, 0.0).unwrap();
        let ts = sched.timesteps();
        // Should be uniformly spaced (approximately)
        assert_eq!(ts.len(), 10);
        // The first timestep (highest noise) should be large
        assert!(ts[0] >= 800, "first timestep too small: {}", ts[0]);
    }

    #[test]
    fn eta_one_adds_stochasticity() {
        // With eta=1, the result should depend on noise
        let sched = DdimScheduler::new(1000, 10, 1.0).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let noise1 = randn(&mut rng, 32);
        let noise2: Vec<f32> = noise1.iter().map(|v| -v).collect(); // flipped noise
        let out1 = sched.step(&eps, &x_t, 3, &noise1).unwrap();
        let out2 = sched.step(&eps, &x_t, 3, &noise2).unwrap();
        let max_diff: f32 = out1
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // With eta=1, noise should matter
        assert!(
            max_diff > 1e-4,
            "eta=1 should be stochastic, diff={max_diff}"
        );
    }

    #[test]
    fn invalid_eta_rejected() {
        assert!(DdimScheduler::new(1000, 10, -0.1).is_err());
    }
}
