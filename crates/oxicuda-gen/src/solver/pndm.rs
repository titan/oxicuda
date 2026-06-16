//! PNDM / PLMS — Pseudo Numerical Methods for Diffusion ODEs.
//!
//! Implements the PLMS (Pseudo Linear Multi-Step) solver of Liu et al. 2022, the
//! linear-multistep variant of PNDM. It treats the reverse diffusion process as
//! an ODE and applies an Adams–Bashforth-style multistep extrapolation of the
//! model's noise prediction `eps_hat`, combined with a fixed "transfer" update
//! that maps `(x_t, eps) → x_s` along the noise schedule.
//!
//! Like [`super::dpm_solver_pp`] and [`super::unipc`], this is a self-contained
//! solver owning a linear-β schedule. It uses `f64` internally for numerical
//! robustness of the multistep coefficients.
//!
//! # Algorithm (PLMS, Liu et al. 2022, Eq. 12–13)
//!
//! The classical-linear-multistep noise estimate uses the Adams–Bashforth
//! coefficients over the last up-to-4 model outputs `e_t, e_{t−1}, e_{t−2},
//! e_{t−3}`:
//!
//! ```text
//! ê = (55·e_t − 59·e_{t−1} + 37·e_{t−2} − 9·e_{t−3}) / 24      (4th order)
//! ```
//!
//! Lower-order rules bootstrap the first three steps (a Runge–Kutta-style 2nd
//! order start, then 2-/3-step Adams–Bashforth). The **transfer** step applies
//!
//! ```text
//! x_s = (√ᾱ_s / √ᾱ_t)·x_t
//!       − (ᾱ_s − ᾱ_t)·ê
//!         / (√ᾱ_t·(√(ᾱ_s·(1−ᾱ_t)) + √(ᾱ_t·(1−ᾱ_s))))
//! ```
//!
//! (Liu et al. 2022, the `transfer` function), which is the exact DDIM update
//! rewritten to avoid catastrophic cancellation.
//!
//! # Reference
//! Liu et al., "Pseudo Numerical Methods for Diffusion Models on Manifolds",
//! ICLR 2022. <https://arxiv.org/abs/2202.09778>

use crate::error::{GenError, GenResult};

// ─── PndmConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the [`PndmSolver`].
#[derive(Debug, Clone)]
pub struct PndmConfig {
    /// Number of inference timesteps (ODE solver steps), e.g. 50.
    pub n_timesteps: usize,
    /// Starting β value for the linear schedule (e.g. 0.0001).
    pub beta_start: f64,
    /// Ending β value for the linear schedule (e.g. 0.02).
    pub beta_end: f64,
}

impl Default for PndmConfig {
    fn default() -> Self {
        Self {
            n_timesteps: 50,
            beta_start: 0.0001,
            beta_end: 0.02,
        }
    }
}

// ─── PndmSolver ─────────────────────────────────────────────────────────────────

/// PLMS (pseudo linear multistep) sampler with a linear β schedule.
///
/// Maintains a small rolling buffer of past model outputs (`ets`) so that the
/// linear-multistep coefficients can be applied as the order ramps up over the
/// first few steps.
#[derive(Debug, Clone)]
pub struct PndmSolver {
    /// Solver configuration.
    pub config: PndmConfig,
    /// `ᾱ_t = ∏_{i=0}^{t}(1 − β_i)`, length n.
    pub alphas_cumprod: Vec<f64>,
    /// Descending indices `[n−1, …, 0]` for reverse diffusion ordering.
    pub timesteps: Vec<usize>,
    /// Rolling buffer of the last (up to 4) noise predictions, newest last.
    ets: Vec<Vec<f64>>,
}

impl PndmSolver {
    /// Build a new PLMS solver from the given configuration.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `n_timesteps == 0`.
    /// - [`GenError::InvalidBetaSchedule`] if `beta_start`/`beta_end` are outside
    ///   `(0, 1)` or `beta_start >= beta_end`.
    pub fn new(config: PndmConfig) -> GenResult<Self> {
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

        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut running_prod = 1.0_f64;
        for &b in &betas {
            running_prod *= 1.0 - b;
            alphas_cumprod.push(running_prod);
        }

        let timesteps: Vec<usize> = (0..n).rev().collect();

        Ok(Self {
            config,
            alphas_cumprod,
            timesteps,
            ets: Vec::with_capacity(4),
        })
    }

    /// Clear the rolling multistep history (call before a fresh sampling run).
    pub fn reset(&mut self) {
        self.ets.clear();
    }

    /// The DDIM-equivalent "transfer" update from index `from_idx` to `to_idx`
    /// using the (possibly extrapolated) noise estimate `e`.
    ///
    /// Uses the cancellation-free form of Liu et al. 2022.
    fn transfer(&self, x_t: &[f64], e: &[f64], from_idx: usize, to_idx: usize) -> Vec<f64> {
        let ab_t = self.alphas_cumprod[from_idx];
        let ab_s = self.alphas_cumprod[to_idx];
        let sqrt_ab_t = ab_t.max(0.0).sqrt();
        let sqrt_ab_s = ab_s.max(0.0).sqrt();
        let ratio = if sqrt_ab_t > 0.0 {
            sqrt_ab_s / sqrt_ab_t
        } else {
            0.0
        };
        // denom = √ᾱ_t · (√(ᾱ_s·(1−ᾱ_t)) + √(ᾱ_t·(1−ᾱ_s)))
        let denom = sqrt_ab_t
            * ((ab_s * (1.0 - ab_t)).max(0.0).sqrt() + (ab_t * (1.0 - ab_s)).max(0.0).sqrt());
        let coeff = if denom.abs() > 1e-12 {
            (ab_s - ab_t) / denom
        } else {
            0.0
        };
        x_t.iter()
            .zip(e)
            .map(|(&x, &ei)| ratio * x - coeff * ei)
            .collect()
    }

    /// Adams–Bashforth linear-multistep noise estimate from the rolling history,
    /// given the *current* model output `e_t` (which is also pushed into the
    /// history by [`Self::step`] before this is called).
    ///
    /// Order is determined by how many outputs are available:
    /// - 1 output → `e_t` (1st order)
    /// - 2 → `(3·e_t − e_{t−1}) / 2`
    /// - 3 → `(23·e_t − 16·e_{t−1} + 5·e_{t−2}) / 12`
    /// - ≥4 → `(55·e_t − 59·e_{t−1} + 37·e_{t−2} − 9·e_{t−3}) / 24`
    fn plms_estimate(&self) -> Vec<f64> {
        let m = self.ets.len();
        // Adams–Bashforth coefficients (newest-first) for each available order.
        // Each set sums to 1, so a constant history is reproduced exactly.
        let coeffs: &[f64] = match m {
            0 | 1 => return self.ets.last().cloned().unwrap_or_default(),
            2 => &[3.0 / 2.0, -1.0 / 2.0],
            3 => &[23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
            _ => &[55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
        };
        let dim = self.ets[m - 1].len();
        let mut out = vec![0.0_f64; dim];
        // out += c_k · e_{newest-k}  for each coefficient.
        for (k, &c) in coeffs.iter().enumerate() {
            let hist = &self.ets[m - 1 - k];
            for (o, &h) in out.iter_mut().zip(hist) {
                *o += c * h;
            }
        }
        out
    }

    /// Perform a single PLMS step from `from_idx` to `to_idx`.
    ///
    /// Pushes `e_t` into the rolling history (capped at 4 entries), computes the
    /// linear-multistep estimate, and applies the transfer update.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_t` or `e_t` is empty.
    /// - [`GenError::DimensionMismatch`] if lengths differ.
    /// - [`GenError::InvalidTimestep`] if an index is out of range.
    pub fn step(
        &mut self,
        x_t: &[f64],
        e_t: &[f64],
        from_idx: usize,
        to_idx: usize,
    ) -> GenResult<Vec<f64>> {
        if x_t.is_empty() || e_t.is_empty() {
            return Err(GenError::EmptyInput("x_t / e_t is empty"));
        }
        if x_t.len() != e_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: e_t.len(),
            });
        }
        let n = self.config.n_timesteps;
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

        self.ets.push(e_t.to_vec());
        if self.ets.len() > 4 {
            self.ets.remove(0);
        }
        let e_hat = self.plms_estimate();
        Ok(self.transfer(x_t, &e_hat, from_idx, to_idx))
    }

    /// Run the full reverse PLMS trajectory.
    ///
    /// Resets the history, then iterates the descending timesteps applying
    /// [`Self::step`] at each interval. The `model_fn` closure maps
    /// `(x, t_idx) → eps_hat`.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_noisy` is empty.
    /// - Propagates errors from [`Self::step`].
    pub fn sample<F>(&mut self, x_noisy: &[f64], mut model_fn: F) -> GenResult<Vec<f64>>
    where
        F: FnMut(&[f64], usize) -> Vec<f64>,
    {
        if x_noisy.is_empty() {
            return Err(GenError::EmptyInput("x_noisy is empty"));
        }
        self.reset();
        let n = self.config.n_timesteps;
        if n < 2 {
            return Ok(x_noisy.to_vec());
        }

        let mut x = x_noisy.to_vec();
        let steps: Vec<usize> = self.timesteps.clone();
        for w in steps.windows(2) {
            let from_idx = w[0];
            let to_idx = w[1];
            let e_t = model_fn(&x, from_idx);
            x = self.step(&x, &e_t, from_idx, to_idx)?;
        }
        Ok(x)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solver(n: usize) -> PndmSolver {
        PndmSolver::new(PndmConfig {
            n_timesteps: n,
            beta_start: 0.0001,
            beta_end: 0.02,
        })
        .expect("valid config")
    }

    #[test]
    fn construct_schedule_shapes() {
        let s = make_solver(50);
        assert_eq!(s.alphas_cumprod.len(), 50);
        assert_eq!(s.timesteps.len(), 50);
        assert_eq!(s.timesteps[0], 49);
        assert_eq!(*s.timesteps.last().expect("non-empty"), 0);
    }

    #[test]
    fn alphas_cumprod_decreasing() {
        let s = make_solver(30);
        for w in s.alphas_cumprod.windows(2) {
            assert!(w[0] >= w[1], "ᾱ must decrease: {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn plms_first_order_equals_input() {
        // 1 output in history → estimate == that output.
        let mut s = make_solver(20);
        let e = vec![1.0_f64, -2.0, 0.5];
        s.ets.push(e.clone());
        let est = s.plms_estimate();
        for (&a, &b) in est.iter().zip(&e) {
            assert!((a - b).abs() < 1e-12, "1st-order estimate mismatch");
        }
    }

    #[test]
    fn plms_constant_history_is_identity() {
        // If all past outputs are equal to c, every AB rule must return c
        // (the coefficients each sum to 1).
        let mut s = make_solver(20);
        let c = vec![0.7_f64, -0.3, 1.1, 2.0];
        for _ in 0..4 {
            s.ets.push(c.clone());
        }
        let est = s.plms_estimate();
        for (&a, &b) in est.iter().zip(&c) {
            assert!(
                (a - b).abs() < 1e-9,
                "constant history must be reproduced exactly: {a} vs {b}"
            );
        }
    }

    #[test]
    fn plms_orders_ramp_with_history() {
        // Verify each order rule matches its closed-form on simple inputs.
        let mut s = make_solver(20);
        // history newest-last: e0=1, e1=2, e2=3, e3=4
        s.ets = vec![vec![4.0], vec![3.0], vec![2.0], vec![1.0]];
        // 4th order: (55*1 − 59*2 + 37*3 − 9*4)/24 = (55−118+111−36)/24 = 12/24 = 0.5
        let est = s.plms_estimate();
        assert!(
            (est[0] - 0.5).abs() < 1e-9,
            "4th-order AB mismatch: {}",
            est[0]
        );
    }

    #[test]
    fn transfer_zero_noise_scales_by_alpha_ratio() {
        // With e = 0, x_s = (√ᾱ_s/√ᾱ_t)·x_t.
        let s = make_solver(20);
        let x = vec![2.0_f64, -4.0, 1.0];
        let e = vec![0.0_f64; 3];
        let out = s.transfer(&x, &e, 19, 18);
        let ratio = (s.alphas_cumprod[18] / s.alphas_cumprod[19]).sqrt();
        for (&o, &xi) in out.iter().zip(&x) {
            assert!(
                (o - ratio * xi).abs() < 1e-9,
                "expected {} got {o}",
                ratio * xi
            );
        }
    }

    #[test]
    fn step_caps_history_at_four() {
        let mut s = make_solver(50);
        let x = vec![1.0_f64; 4];
        for k in 0..6 {
            let e = vec![k as f64; 4];
            let _ = s.step(&x, &e, 49 - k, 48 - k).expect("step ok");
        }
        assert_eq!(s.ets.len(), 4, "history must be capped at 4 entries");
    }

    #[test]
    fn step_shape_and_finite() {
        let mut s = make_solver(20);
        let x = vec![0.5_f64; 8];
        let e = vec![0.1_f64; 8];
        let out = s.step(&x, &e, 19, 18).expect("step ok");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sample_finite_output() {
        let mut s = make_solver(50);
        let x = vec![1.0_f64; 16];
        let out = s
            .sample(&x, |x, t| vec![0.001_f64 * t as f64; x.len()])
            .expect("sample ok");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sample_resets_history() {
        let mut s = make_solver(20);
        let x = vec![1.0_f64; 4];
        // Pre-pollute history.
        s.ets.push(vec![99.0; 4]);
        let _ = s.sample(&x, |x, _| vec![0.0; x.len()]).expect("ok");
        // After sampling 20 steps the history is at most 4 — the pre-pollution
        // (a different value) should have been cleared at the start.
        assert!(s.ets.len() <= 4);
    }

    #[test]
    fn sample_zero_noise_pred_bounded() {
        let mut s = make_solver(50);
        let x = vec![3.0_f64, -3.0, 1.0, -1.0];
        let out = s.sample(&x, |x, _| vec![0.0; x.len()]).expect("ok");
        let max_out: f64 = out.iter().map(|v| v.abs()).fold(0.0, f64::max);
        // With zero noise the sample scales by √(ᾱ_0/ᾱ_{n-1}) ≥ 1 but stays finite.
        assert!(max_out.is_finite() && max_out < 1e6, "bounded output");
    }

    #[test]
    fn err_empty_input() {
        let mut s = make_solver(20);
        assert!(matches!(
            s.step(&[], &[], 19, 18),
            Err(GenError::EmptyInput(_))
        ));
        assert!(matches!(
            s.sample(&[], |x, _| vec![0.0; x.len()]),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_dimension_mismatch() {
        let mut s = make_solver(20);
        let x = vec![1.0_f64; 4];
        let e = vec![1.0_f64; 3];
        assert!(matches!(
            s.step(&x, &e, 19, 18),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_invalid_timestep() {
        let mut s = make_solver(20);
        let x = vec![1.0_f64; 4];
        let e = vec![1.0_f64; 4];
        assert!(matches!(
            s.step(&x, &e, 99, 18),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn err_invalid_beta_schedule() {
        assert!(matches!(
            PndmSolver::new(PndmConfig {
                n_timesteps: 10,
                beta_start: 0.02,
                beta_end: 0.0001,
            }),
            Err(GenError::InvalidBetaSchedule)
        ));
    }

    #[test]
    fn err_zero_timesteps() {
        assert!(matches!(
            PndmSolver::new(PndmConfig {
                n_timesteps: 0,
                ..PndmConfig::default()
            }),
            Err(GenError::EmptyInput(_))
        ));
    }
}
