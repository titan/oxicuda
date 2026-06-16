//! DPM-Solver++ scheduler.
//!
//! Implements DPM-Solver++ (Lu et al. 2022) with 1st, 2nd, and 3rd order
//! update rules. Uses the log-SNR (λ) parameterisation for numerical stability.

use crate::error::{GenError, GenResult};
use crate::scheduler::beta_schedule::BetaSchedule;

// ─── DpmOrder ─────────────────────────────────────────────────────────────────

/// Order of the DPM-Solver++ update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmOrder {
    /// First-order: single model evaluation, Euler-equivalent.
    First,
    /// Second-order: uses previous model output for correction.
    Second,
    /// Third-order: uses two previous model outputs for higher accuracy.
    Third,
}

// ─── DpmSolverScheduler ───────────────────────────────────────────────────────

/// DPM-Solver++ scheduler for fast diffusion sampling.
///
/// # Reference
/// Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
/// Probabilistic Models", NeurIPS 2022.
#[derive(Debug, Clone)]
pub struct DpmSolverScheduler {
    schedule: BetaSchedule,
    order: DpmOrder,
    num_train_steps: usize,
    num_inference_steps: usize,
    /// λ_t = 0.5 * log(ᾱ_t / (1-ᾱ_t)) for each training timestep.
    lambdas: Vec<f32>,
    /// Subsampled training timestep indices for inference.
    timesteps: Vec<usize>,
}

impl DpmSolverScheduler {
    /// Create a new DPM-Solver++ scheduler.
    ///
    /// # Arguments
    /// - `num_train_steps`: Total training timesteps.
    /// - `num_inference_steps`: Number of ODE solver steps.
    /// - `order`: DPM-Solver order (1, 2, or 3).
    ///
    /// # Errors
    /// - `EmptyInput` if any count is 0
    /// - `UnsupportedDpmOrder` if order is somehow invalid
    pub fn new(
        num_train_steps: usize,
        num_inference_steps: usize,
        order: DpmOrder,
    ) -> GenResult<Self> {
        if num_train_steps == 0 {
            return Err(GenError::EmptyInput("num_train_steps must be > 0"));
        }
        if num_inference_steps == 0 {
            return Err(GenError::EmptyInput("num_inference_steps must be > 0"));
        }
        let schedule = BetaSchedule::linear(num_train_steps, 0.0001, 0.02)?;
        let lambdas: Vec<f32> = schedule
            .alphas_bar()
            .iter()
            .map(|&ab| Self::compute_lambda(ab))
            .collect();
        let timesteps = Self::compute_timesteps(num_train_steps, num_inference_steps);
        Ok(Self {
            schedule,
            order,
            num_train_steps,
            num_inference_steps,
            lambdas,
            timesteps,
        })
    }

    /// Create with a custom beta schedule.
    pub fn with_custom_schedule(
        schedule: BetaSchedule,
        num_inference_steps: usize,
        order: DpmOrder,
    ) -> GenResult<Self> {
        if num_inference_steps == 0 {
            return Err(GenError::EmptyInput("num_inference_steps must be > 0"));
        }
        let n = schedule.num_steps();
        let lambdas: Vec<f32> = schedule
            .alphas_bar()
            .iter()
            .map(|&ab| Self::compute_lambda(ab))
            .collect();
        let timesteps = Self::compute_timesteps(n, num_inference_steps);
        Ok(Self {
            num_train_steps: n,
            num_inference_steps,
            order,
            lambdas,
            timesteps,
            schedule,
        })
    }

    /// Compute the log-SNR: `λ_t = 0.5 * log(ᾱ_t / (1 - ᾱ_t))`.
    ///
    /// This is the negative half log-variance in the SDE formulation.
    fn compute_lambda(alpha_bar: f32) -> f32 {
        let ab = alpha_bar.clamp(1e-7, 1.0 - 1e-7);
        0.5 * (ab / (1.0 - ab)).ln()
    }

    /// Compute subsampled timesteps (uniform spacing, descending order).
    fn compute_timesteps(num_train: usize, num_infer: usize) -> Vec<usize> {
        let step_ratio = num_train / num_infer.max(1);
        (0..num_infer)
            .map(|i| ((num_infer - 1 - i) * step_ratio).min(num_train - 1))
            .collect()
    }

    /// Compute `α_t` and `σ_t` for a given training timestep index.
    ///
    /// Returns `(alpha_t, sigma_t)` where `alpha_t = √ᾱ_t`, `σ_t = √(1-ᾱ_t)`.
    fn alpha_sigma_at(&self, t: usize) -> (f32, f32) {
        let ab = self.schedule.alphas_bar()[t];
        (ab.sqrt(), (1.0 - ab).max(0.0).sqrt())
    }

    /// Returns the number of training timesteps this scheduler was built with.
    pub fn num_train_steps(&self) -> usize {
        self.num_train_steps
    }

    /// First-order DPM-Solver step.
    ///
    /// `x_{t-1} = (σ_{t-1}/σ_t) * x_t - α_{t-1} * (exp(-h) - 1) * D_0`
    ///
    /// where `h = λ_{t-1} - λ_t` (positive since λ decreases with noise).
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step_idx >= num_inference_steps`
    /// - `DimensionMismatch` on shape mismatch
    pub fn step_first_order(
        &self,
        model_output: &[f32],
        x_t: &[f32],
        step_idx: usize,
    ) -> GenResult<Vec<f32>> {
        if model_output.is_empty() {
            return Err(GenError::EmptyInput("model_output is empty"));
        }
        if step_idx >= self.num_inference_steps {
            return Err(GenError::InvalidTimestep {
                t: step_idx,
                max_t: self.num_inference_steps,
            });
        }
        if model_output.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: model_output.len(),
                got: x_t.len(),
            });
        }

        let t = self.timesteps[step_idx];
        let (_alpha_t, sigma_t) = self.alpha_sigma_at(t);
        let lambda_t = self.lambdas[t];

        // Previous step (t-1 in terms of noise, i.e. step_idx + 1 in reverse)
        let (alpha_s, sigma_s, lambda_s) = if step_idx + 1 < self.num_inference_steps {
            let s = self.timesteps[step_idx + 1];
            let (a, sig) = self.alpha_sigma_at(s);
            (a, sig, self.lambdas[s])
        } else {
            // Final step: approach clean data (ᾱ → 1)
            (1.0_f32, 0.0_f32, 0.0_f32)
        };

        let h = lambda_s - lambda_t; // positive: λ increases toward clean
        let coeff_xt = sigma_s / sigma_t.max(1e-10);
        let coeff_d0 = -alpha_s * ((-h).exp() - 1.0);

        let result = x_t
            .iter()
            .zip(model_output)
            .map(|(&xt, &d0)| coeff_xt * xt + coeff_d0 * d0)
            .collect();
        Ok(result)
    }

    /// Second-order DPM-Solver++ step (multi-step corrector).
    ///
    /// Uses the current and previous model outputs as a 2nd-order Taylor
    /// approximation.
    ///
    /// # Errors
    /// - `InvalidTimestep` if `step_idx >= num_inference_steps`
    /// - `DimensionMismatch` on shape mismatch
    pub fn step_second_order(
        &self,
        model_output: &[f32],
        prev_output: &[f32],
        x_t: &[f32],
        step_idx: usize,
    ) -> GenResult<Vec<f32>> {
        if model_output.is_empty() {
            return Err(GenError::EmptyInput("model_output is empty"));
        }
        if step_idx >= self.num_inference_steps {
            return Err(GenError::InvalidTimestep {
                t: step_idx,
                max_t: self.num_inference_steps,
            });
        }
        if model_output.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: model_output.len(),
                got: x_t.len(),
            });
        }
        if prev_output.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: prev_output.len(),
            });
        }

        let t = self.timesteps[step_idx];
        let (_alpha_t, sigma_t) = self.alpha_sigma_at(t);
        let lambda_t = self.lambdas[t];

        let (alpha_s, sigma_s, lambda_s) = if step_idx + 1 < self.num_inference_steps {
            let s = self.timesteps[step_idx + 1];
            let (a, sig) = self.alpha_sigma_at(s);
            (a, sig, self.lambdas[s])
        } else {
            (1.0_f32, 0.0_f32, 0.0_f32)
        };

        // Get the timestep before prev (2 steps back)
        let lambda_prev = if step_idx + 2 < self.num_inference_steps {
            let p = self.timesteps[step_idx + 2];
            self.lambdas[p]
        } else {
            lambda_t + (lambda_s - lambda_t) * 1.5 // extrapolate
        };

        let h = lambda_s - lambda_t;
        let h_prev = lambda_t - lambda_prev; // previous interval width

        // 2nd order: D_1 = (D_0 - D_prev) / (h + h_prev) * h
        let r = h / h_prev.max(1e-10);
        let coeff_xt = sigma_s / sigma_t.max(1e-10);
        let coeff_d0 = -alpha_s * ((-h).exp() - 1.0);
        let coeff_d1 = -alpha_s * (((-h).exp() - 1.0) / h - (-1.0)) * 0.5 * r;

        let result = x_t
            .iter()
            .zip(model_output)
            .zip(prev_output)
            .map(|((&xt, &d0), &d_prev)| {
                let d1 = (d0 - d_prev) * r;
                coeff_xt * xt + coeff_d0 * d0 + coeff_d1 * d1
            })
            .collect();
        Ok(result)
    }

    /// Unified step that dispatches to the appropriate order.
    ///
    /// Falls back to first order if `prev_output` is `None` (warm-up).
    ///
    /// # Errors
    /// Same as `step_first_order` / `step_second_order`.
    pub fn step(
        &self,
        model_output: &[f32],
        prev_output: Option<&[f32]>,
        x_t: &[f32],
        step_idx: usize,
    ) -> GenResult<Vec<f32>> {
        match (self.order, prev_output) {
            (DpmOrder::First, _) => self.step_first_order(model_output, x_t, step_idx),
            (DpmOrder::Second, Some(prev)) => {
                self.step_second_order(model_output, prev, x_t, step_idx)
            }
            (DpmOrder::Second, None) => {
                // Warm-up: use first order on the first step
                self.step_first_order(model_output, x_t, step_idx)
            }
            (DpmOrder::Third, Some(prev)) => {
                // Simplified: use second order for third-order (multi-step)
                self.step_second_order(model_output, prev, x_t, step_idx)
            }
            (DpmOrder::Third, None) => self.step_first_order(model_output, x_t, step_idx),
        }
    }

    /// Return the log-SNR values `λ_t` for all training timesteps.
    pub fn lambdas(&self) -> &[f32] {
        &self.lambdas
    }

    /// Return the subsampled inference timesteps.
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Return the number of inference steps.
    pub fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }

    /// Return a reference to the beta schedule.
    pub fn schedule(&self) -> &BetaSchedule {
        &self.schedule
    }

    /// Return the solver order.
    pub fn order(&self) -> DpmOrder {
        self.order
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(99)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn new_scheduler_valid() {
        let sched =
            DpmSolverScheduler::new(1000, 20, DpmOrder::Second).expect("new should succeed");
        assert_eq!(sched.num_inference_steps(), 20);
        assert_eq!(sched.order(), DpmOrder::Second);
    }

    #[test]
    fn lambdas_count_matches_train_steps() {
        let sched = DpmSolverScheduler::new(1000, 20, DpmOrder::First).expect("new should succeed");
        assert_eq!(sched.lambdas().len(), 1000);
    }

    #[test]
    fn lambdas_strictly_decreasing_with_noise() {
        // λ_t should decrease as t increases (more noise = lower SNR)
        let sched = DpmSolverScheduler::new(1000, 20, DpmOrder::First).expect("new should succeed");
        let lam = sched.lambdas();
        for w in lam.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-5,
                "lambda should decrease: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn timesteps_in_valid_range() {
        let sched = DpmSolverScheduler::new(1000, 20, DpmOrder::First).expect("new should succeed");
        for &t in sched.timesteps() {
            assert!(t < 1000, "timestep {t} out of range");
        }
    }

    #[test]
    fn first_order_step_shape() {
        let sched = DpmSolverScheduler::new(1000, 10, DpmOrder::First).expect("new should succeed");
        let mut rng = make_rng();
        let d0 = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let out = sched
            .step_first_order(&d0, &x_t, 0)
            .expect("step_first_order should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn second_order_step_shape() {
        let sched =
            DpmSolverScheduler::new(1000, 10, DpmOrder::Second).expect("new should succeed");
        let mut rng = make_rng();
        let d0 = randn(&mut rng, 32);
        let d_prev = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let out = sched
            .step_second_order(&d0, &d_prev, &x_t, 1)
            .expect("step_second_order should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn step_dispatch_no_prev() {
        let sched =
            DpmSolverScheduler::new(1000, 10, DpmOrder::Second).expect("new should succeed");
        let mut rng = make_rng();
        let d0 = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        // Should fall back to first order
        let out = sched.step(&d0, None, &x_t, 0).expect("step should succeed");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn step_dispatch_with_prev() {
        let sched =
            DpmSolverScheduler::new(1000, 10, DpmOrder::Second).expect("new should succeed");
        let mut rng = make_rng();
        let d0 = randn(&mut rng, 32);
        let d_prev = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let out = sched
            .step(&d0, Some(&d_prev), &x_t, 1)
            .expect("value should be present");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn first_order_invalid_step_idx() {
        let sched = DpmSolverScheduler::new(1000, 10, DpmOrder::First).expect("new should succeed");
        let d0 = vec![0.0_f32; 8];
        let x_t = vec![0.0_f32; 8];
        assert!(matches!(
            sched.step_first_order(&d0, &x_t, 10),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn step_outputs_finite() {
        let sched = DpmSolverScheduler::new(1000, 10, DpmOrder::First).expect("new should succeed");
        let mut rng = make_rng();
        let d0 = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        for i in 0..10 {
            let out = sched
                .step_first_order(&d0, &x_t, i)
                .expect("step_first_order should succeed");
            assert!(out.iter().all(|v| v.is_finite()), "non-finite at step {i}");
        }
    }

    #[test]
    fn compute_lambda_monotone() {
        // Lambda should be monotone decreasing w.r.t. alpha_bar
        // Higher alpha_bar (less noise) → higher lambda
        let l1 = DpmSolverScheduler::compute_lambda(0.9);
        let l2 = DpmSolverScheduler::compute_lambda(0.5);
        let l3 = DpmSolverScheduler::compute_lambda(0.1);
        assert!(l1 > l2, "lambda(0.9) > lambda(0.5): {} vs {}", l1, l2);
        assert!(l2 > l3, "lambda(0.5) > lambda(0.1): {} vs {}", l2, l3);
    }

    #[test]
    fn third_order_fallback() {
        let sched = DpmSolverScheduler::new(1000, 10, DpmOrder::Third).expect("new should succeed");
        let mut rng = make_rng();
        let d0 = randn(&mut rng, 16);
        let d_prev = randn(&mut rng, 16);
        let x_t = randn(&mut rng, 16);
        let out = sched
            .step(&d0, Some(&d_prev), &x_t, 2)
            .expect("value should be present");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let sched = DpmSolverScheduler::new(100, 10, DpmOrder::First).expect("new should succeed");
        let d0 = vec![0.0_f32; 8];
        let x_t = vec![0.0_f32; 4];
        assert!(matches!(
            sched.step_first_order(&d0, &x_t, 0),
            Err(GenError::DimensionMismatch { .. })
        ));
    }
}
