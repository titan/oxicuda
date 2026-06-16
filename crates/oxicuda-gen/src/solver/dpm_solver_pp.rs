//! DPM-Solver++ 2M standalone implementation.
//!
//! Implements the DPM-Solver++ multi-step (2M) algorithm from Lu et al. 2022,
//! operating directly over a linear-beta noise schedule. This is a self-contained
//! solver that owns its own schedule rather than delegating to `BetaSchedule`.
//!
//! # Reference
//! Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
//! Probabilistic Models", NeurIPS 2022. <https://arxiv.org/abs/2211.01095>

use crate::error::{GenError, GenResult};

// ─── DpmAlgorithm ─────────────────────────────────────────────────────────────

/// Selects whether to use the first-order or multi-step second-order update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpmAlgorithm {
    /// First-order (Euler-equivalent) DPM-Solver++ step.
    DpmSolverPp1M,
    /// Second-order multi-step DPM-Solver++ (uses previous model output).
    DpmSolverPp2M,
}

// ─── DpmSolverPpConfig ────────────────────────────────────────────────────────

/// Configuration for the standalone DPM-Solver++ 2M solver.
#[derive(Debug, Clone)]
pub struct DpmSolverPpConfig {
    /// Number of ODE solver steps (inference timesteps), e.g. 20.
    pub n_timesteps: usize,
    /// Starting beta value for the linear schedule (e.g. 0.0001).
    pub beta_start: f64,
    /// Ending beta value for the linear schedule (e.g. 0.02).
    pub beta_end: f64,
    /// Which multi-step algorithm to use.
    pub algorithm_type: DpmAlgorithm,
}

// ─── DpmSolverPp ──────────────────────────────────────────────────────────────

/// DPM-Solver++ 2M sampler with a linear beta schedule.
///
/// Pre-computes `betas`, `alphas_cumprod`, `sigmas`, and `timesteps` at
/// construction time for efficient repeated sampling.
#[derive(Debug, Clone)]
pub struct DpmSolverPp {
    /// Solver configuration.
    pub config: DpmSolverPpConfig,
    /// `β_t = beta_start + (beta_end - beta_start) * t/(n-1)`, length n.
    pub betas: Vec<f64>,
    /// `ᾱ_t = ∏_{i=0}^{t} (1 - β_i)`, length n.
    pub alphas_cumprod: Vec<f64>,
    /// `σ_t = √(1 - ᾱ_t)`, length n.
    pub sigmas: Vec<f64>,
    /// Descending indices `[n-1, n-2, ..., 0]` for reverse diffusion ordering.
    pub timesteps: Vec<usize>,
}

impl DpmSolverPp {
    /// Build a new DPM-Solver++ solver from the given configuration.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `n_timesteps == 0`
    /// - [`GenError::InvalidBetaSchedule`] if `beta_start` or `beta_end` are outside `(0, 1)`
    ///   or if `beta_start >= beta_end`
    pub fn new(config: DpmSolverPpConfig) -> GenResult<Self> {
        let n = config.n_timesteps;
        if n == 0 {
            return Err(GenError::EmptyInput("n_timesteps must be > 0"));
        }
        if config.beta_start <= 0.0
            || config.beta_start >= 1.0
            || config.beta_end <= 0.0
            || config.beta_end >= 1.0
            || config.beta_start >= config.beta_end
        {
            return Err(GenError::InvalidBetaSchedule);
        }

        // Build linear beta schedule: β_t = beta_start + (beta_end - beta_start) * t/(n-1)
        let betas: Vec<f64> = if n == 1 {
            vec![config.beta_start]
        } else {
            (0..n)
                .map(|t| {
                    config.beta_start
                        + (config.beta_end - config.beta_start) * t as f64 / (n - 1) as f64
                })
                .collect()
        };

        // Compute cumulative product of (1 - β_t)
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut running_prod = 1.0_f64;
        for &b in &betas {
            running_prod *= 1.0 - b;
            alphas_cumprod.push(running_prod);
        }

        // σ_t = √(1 - ᾱ_t)
        let sigmas: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&ab| (1.0 - ab).max(0.0).sqrt())
            .collect();

        // Reverse timestep ordering: [n-1, n-2, ..., 0]
        let timesteps: Vec<usize> = (0..n).rev().collect();

        Ok(Self {
            config,
            betas,
            alphas_cumprod,
            sigmas,
            timesteps,
        })
    }

    /// Compute the log-SNR `λ_t = 0.5 * ln(ᾱ_t / (1 - ᾱ_t))` at index `t_idx`.
    ///
    /// The index is a direct index into `alphas_cumprod` (0..n_timesteps).
    pub fn lambda(&self, t_idx: usize) -> f64 {
        let ab = self.alphas_cumprod[t_idx].clamp(1e-10, 1.0 - 1e-10);
        0.5 * (ab / (1.0 - ab)).ln()
    }

    /// First-order DPM-Solver++ step (Euler in log-SNR space).
    ///
    /// Moves from noisy state `x_t` at index `from_idx` to cleaner state at
    /// index `to_idx`, using the model noise prediction `eps_hat`.
    ///
    /// Formula:
    /// ```text
    /// x_s = (α_s / α_t) * x_t + σ_s * (exp(-h) - 1) * D_0
    /// ```
    /// where `h = λ_s - λ_t`, `α_t = √ᾱ_t`, `α_s = √ᾱ_s`.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_t` or `eps_hat` is empty
    /// - [`GenError::DimensionMismatch`] if lengths differ
    /// - [`GenError::InvalidTimestep`] if either index is out of range
    pub fn step_1m(
        &self,
        x_t: &[f64],
        eps_hat: &[f64],
        from_idx: usize,
        to_idx: usize,
    ) -> GenResult<Vec<f64>> {
        let n = self.config.n_timesteps;
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t is empty"));
        }
        if x_t.len() != eps_hat.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: eps_hat.len(),
            });
        }
        if from_idx >= n {
            return Err(GenError::InvalidTimestep {
                t: from_idx,
                max_t: n,
            });
        }
        if to_idx >= n {
            return Err(GenError::InvalidTimestep {
                t: to_idx,
                max_t: n,
            });
        }

        // Degenerate: source and target are the same timestep — no step needed.
        if from_idx == to_idx {
            return Ok(x_t.to_vec());
        }

        let alpha_t = self.alphas_cumprod[from_idx].sqrt();
        let alpha_s = self.alphas_cumprod[to_idx].sqrt();
        let sigma_s = self.sigmas[to_idx];
        let sigma_t = self.sigmas[from_idx];

        let lambda_t = self.lambda(from_idx);
        let lambda_s = self.lambda(to_idx);
        let h = lambda_s - lambda_t;

        // Clamp to avoid catastrophic cancellation if sigmas are near-zero
        let ratio = if sigma_t.abs() > 1e-12 {
            alpha_s / alpha_t
        } else {
            // Degenerate: t is already clean
            1.0
        };
        let exp_minus_h = (-h).exp();

        let result = x_t
            .iter()
            .zip(eps_hat.iter())
            .map(|(&xt, &eps)| ratio * xt + sigma_s * (exp_minus_h - 1.0) * eps)
            .collect();

        Ok(result)
    }

    /// Second-order multi-step DPM-Solver++ step.
    ///
    /// When `prev_eps` is `None` (first inference step), falls back to the
    /// first-order [`Self::step_1m`].
    ///
    /// Formula (2M corrector):
    /// ```text
    /// D_0 = eps_hat
    /// D_1 = (eps_hat - prev_eps) / (1 + 1/r)      [element-wise]
    /// x_s = (α_s/α_t)*x_t
    ///       - σ_s*(exp(-h) - 1) * D_0
    ///       - σ_s*((exp(-h) - 1)/h + 1) * (1/r) * D_1
    /// ```
    /// where `r = h_prev / h`, `h = λ_s - λ_t`, `h_prev = λ_t - λ_{t_prev}`.
    ///
    /// # Errors
    /// - Same as [`Self::step_1m`]; additionally:
    /// - [`GenError::DimensionMismatch`] if `prev_eps.len() != x_t.len()`
    pub fn step_2m(
        &self,
        x_t: &[f64],
        eps_hat: &[f64],
        prev_eps: Option<&[f64]>,
        from_idx: usize,
        to_idx: usize,
    ) -> GenResult<Vec<f64>> {
        let n = self.config.n_timesteps;
        if x_t.is_empty() {
            return Err(GenError::EmptyInput("x_t is empty"));
        }
        if x_t.len() != eps_hat.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: eps_hat.len(),
            });
        }
        if from_idx >= n {
            return Err(GenError::InvalidTimestep {
                t: from_idx,
                max_t: n,
            });
        }
        if to_idx >= n {
            return Err(GenError::InvalidTimestep {
                t: to_idx,
                max_t: n,
            });
        }

        // Degenerate: source and target are the same timestep — no step needed.
        if from_idx == to_idx {
            return Ok(x_t.to_vec());
        }

        let prev_eps = match prev_eps {
            None => return self.step_1m(x_t, eps_hat, from_idx, to_idx),
            Some(p) => {
                if p.len() != x_t.len() {
                    return Err(GenError::DimensionMismatch {
                        expected: x_t.len(),
                        got: p.len(),
                    });
                }
                p
            }
        };

        let alpha_t = self.alphas_cumprod[from_idx].sqrt();
        let alpha_s = self.alphas_cumprod[to_idx].sqrt();
        let sigma_s = self.sigmas[to_idx];

        let lambda_t = self.lambda(from_idx);
        let lambda_s = self.lambda(to_idx);
        let h = lambda_s - lambda_t;

        // Previous timestep index (one step further into the noisy direction)
        let prev_t_idx = from_idx.saturating_add(1).min(n - 1);
        let lambda_prev_t = self.lambda(prev_t_idx);
        let h_prev = lambda_t - lambda_prev_t;

        let r = if h_prev.abs() > 1e-12 {
            h_prev / h
        } else {
            // Degenerate schedule region; fall back gracefully to 1M
            return self.step_1m(x_t, eps_hat, from_idx, to_idx);
        };

        let exp_minus_h = (-h).exp();
        let ratio = alpha_s / alpha_t;

        let result = x_t
            .iter()
            .zip(eps_hat.iter())
            .zip(prev_eps.iter())
            .map(|((&xt, &eps), &peps)| {
                let d_0 = eps;
                let d_1 = (eps - peps) / (1.0 + 1.0 / r);

                ratio * xt
                    - sigma_s * (exp_minus_h - 1.0) * d_0
                    - sigma_s * ((exp_minus_h - 1.0) / h + 1.0) * (1.0 / r) * d_1
            })
            .collect();

        Ok(result)
    }

    /// Run the full reverse-diffusion sampling loop.
    ///
    /// Starts from pure noise `x_noisy` and iterates over all `n_timesteps`
    /// steps in descending order, calling `model_fn` at each step to obtain
    /// the denoised noise prediction.
    ///
    /// # Arguments
    /// - `x_noisy`: Initial noise tensor (shape `[D]`).
    /// - `model_fn`: Closure `|x: &[f64], t_idx: usize| -> GenResult<Vec<f64>>`
    ///   representing the score/noise model evaluated at timestep index `t_idx`.
    ///
    /// # Errors
    /// - Any error propagated from the model closure.
    /// - [`GenError::EmptyInput`] if `x_noisy` is empty.
    pub fn sample<F>(&self, x_noisy: &[f64], model_fn: &mut F) -> GenResult<Vec<f64>>
    where
        F: FnMut(&[f64], usize) -> GenResult<Vec<f64>>,
    {
        if x_noisy.is_empty() {
            return Err(GenError::EmptyInput("x_noisy must not be empty"));
        }

        let n = self.config.n_timesteps;
        let mut x_t = x_noisy.to_vec();
        let mut prev_eps: Option<Vec<f64>> = None;

        for step_i in 0..n {
            let from_idx = self.timesteps[step_i];
            // Target index: next in the descending sequence, or index 0 at the end
            let to_idx = if step_i + 1 < n {
                self.timesteps[step_i + 1]
            } else {
                0
            };

            let eps_hat = model_fn(&x_t, from_idx)?;

            let x_next = match self.config.algorithm_type {
                DpmAlgorithm::DpmSolverPp1M => self.step_1m(&x_t, &eps_hat, from_idx, to_idx)?,
                DpmAlgorithm::DpmSolverPp2M => {
                    self.step_2m(&x_t, &eps_hat, prev_eps.as_deref(), from_idx, to_idx)?
                }
            };

            prev_eps = Some(eps_hat);
            x_t = x_next;
        }

        Ok(x_t)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solver(n: usize) -> DpmSolverPp {
        DpmSolverPp::new(DpmSolverPpConfig {
            n_timesteps: n,
            beta_start: 0.0001,
            beta_end: 0.02,
            algorithm_type: DpmAlgorithm::DpmSolverPp2M,
        })
        .expect("value should be present")
    }

    #[test]
    fn timesteps_len() {
        let solver = make_solver(20);
        assert_eq!(solver.timesteps.len(), 20);
    }

    #[test]
    fn timesteps_descending() {
        let solver = make_solver(20);
        // timesteps should be [19, 18, ..., 0]
        for w in solver.timesteps.windows(2) {
            assert!(
                w[0] > w[1],
                "timesteps not descending: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn lambda_order() {
        let solver = make_solver(100);
        let l0 = solver.lambda(0);
        let l_last = solver.lambda(99);
        // lambda decreases with t (more noise = lower SNR)
        assert!(
            l0 > l_last,
            "lambda(0)={l0} should be > lambda(99)={l_last}"
        );
        assert!(
            (l0 - l_last).abs() > 1e-6,
            "lambda values must differ: {l0} vs {l_last}"
        );
    }

    #[test]
    fn lambda_monotone_decreasing() {
        let solver = make_solver(50);
        let lambdas: Vec<f64> = (0..50).map(|i| solver.lambda(i)).collect();
        for w in lambdas.windows(2) {
            assert!(
                w[1] < w[0] + 1e-10,
                "lambda not monotone decreasing: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn sigmas_in_range() {
        let solver = make_solver(100);
        for (i, &s) in solver.sigmas.iter().enumerate() {
            assert!((0.0..=1.0).contains(&s), "sigma[{i}]={s} not in [0, 1]");
        }
    }

    #[test]
    fn alpha_bar_decreasing() {
        let solver = make_solver(100);
        for w in solver.alphas_cumprod.windows(2) {
            assert!(
                w[1] < w[0],
                "alphas_cumprod not strictly decreasing: {} vs {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn step_1m_output_shape() {
        let solver = make_solver(20);
        let dim = 32;
        let x_t: Vec<f64> = (0..dim).map(|i| i as f64 * 0.01).collect();
        let eps: Vec<f64> = (0..dim).map(|i| -(i as f64) * 0.005).collect();
        let out = solver
            .step_1m(&x_t, &eps, 19, 18)
            .expect("step_1m should succeed");
        assert_eq!(out.len(), dim);
    }

    #[test]
    fn step_1m_output_finite() {
        let solver = make_solver(20);
        let dim = 64;
        let x_t: Vec<f64> = (0..dim).map(|i| (i as f64 - 32.0) * 0.1).collect();
        let eps: Vec<f64> = (0..dim).map(|i| (i as f64 - 32.0) * 0.05).collect();
        let out = solver
            .step_1m(&x_t, &eps, 19, 18)
            .expect("step_1m should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "step_1m output[{i}]={v} not finite");
        }
    }

    #[test]
    fn step_2m_first_step_fallback() {
        // With prev_eps=None, step_2m should fall back to step_1m
        let solver = make_solver(20);
        let dim = 16;
        let x_t: Vec<f64> = vec![0.5; dim];
        let eps: Vec<f64> = vec![0.1; dim];
        // Both should produce the same result
        let out_1m = solver
            .step_1m(&x_t, &eps, 19, 18)
            .expect("step_1m should succeed");
        let out_2m = solver
            .step_2m(&x_t, &eps, None, 19, 18)
            .expect("step_2m should succeed");
        for (i, (&a, &b)) in out_1m.iter().zip(&out_2m).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "1m/2m(no prev) mismatch at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn step_2m_output_finite() {
        let solver = make_solver(20);
        let dim = 32;
        let x_t: Vec<f64> = (0..dim).map(|i| i as f64 * 0.01).collect();
        let eps: Vec<f64> = vec![0.2; dim];
        let prev_eps: Vec<f64> = vec![0.15; dim];
        let out = solver
            .step_2m(&x_t, &eps, Some(&prev_eps), 18, 17)
            .expect("value should be present");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "step_2m output[{i}]={v} not finite");
        }
    }

    #[test]
    fn sample_output_shape() {
        let solver = make_solver(10);
        let dim = 16;
        let x_noisy: Vec<f64> = vec![1.0; dim];
        // Simple identity model: returns the input scaled by a small constant
        let mut model = |x: &[f64], _t: usize| -> GenResult<Vec<f64>> {
            Ok(x.iter().map(|&v| v * 0.1).collect())
        };
        let out = solver
            .sample(&x_noisy, &mut model)
            .expect("sample should succeed");
        assert_eq!(out.len(), dim);
    }

    #[test]
    fn sample_output_finite() {
        let solver = make_solver(10);
        let dim = 32;
        let x_noisy: Vec<f64> = (0..dim).map(|i| (i as f64 - 16.0) * 0.1).collect();
        let mut model = |_x: &[f64], _t: usize| -> GenResult<Vec<f64>> { Ok(vec![0.0; dim]) };
        let out = solver
            .sample(&x_noisy, &mut model)
            .expect("sample should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "sample output[{i}]={v} not finite");
        }
    }

    #[test]
    fn n_timesteps_zero_error() {
        let result = DpmSolverPp::new(DpmSolverPpConfig {
            n_timesteps: 0,
            beta_start: 0.0001,
            beta_end: 0.02,
            algorithm_type: DpmAlgorithm::DpmSolverPp2M,
        });
        assert!(
            matches!(result, Err(GenError::EmptyInput(_))),
            "expected EmptyInput error for n_timesteps=0"
        );
    }

    #[test]
    fn invalid_beta_schedule_error() {
        // beta_start >= beta_end should fail
        let result = DpmSolverPp::new(DpmSolverPpConfig {
            n_timesteps: 20,
            beta_start: 0.02,
            beta_end: 0.0001,
            algorithm_type: DpmAlgorithm::DpmSolverPp1M,
        });
        assert!(
            matches!(result, Err(GenError::InvalidBetaSchedule)),
            "expected InvalidBetaSchedule error"
        );
    }

    #[test]
    fn dim_mismatch_rejected() {
        let solver = make_solver(20);
        let x_t = vec![0.0_f64; 8];
        let eps = vec![0.0_f64; 16];
        let result = solver.step_1m(&x_t, &eps, 19, 18);
        assert!(
            matches!(result, Err(GenError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    #[test]
    fn algorithm_type_1m_sample() {
        let solver = DpmSolverPp::new(DpmSolverPpConfig {
            n_timesteps: 5,
            beta_start: 0.0001,
            beta_end: 0.02,
            algorithm_type: DpmAlgorithm::DpmSolverPp1M,
        })
        .expect("value should be present");
        let x_noisy = vec![1.0_f64; 8];
        let mut model = |_x: &[f64], _t: usize| -> GenResult<Vec<f64>> { Ok(vec![0.0; 8]) };
        let out = solver
            .sample(&x_noisy, &mut model)
            .expect("sample should succeed");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
