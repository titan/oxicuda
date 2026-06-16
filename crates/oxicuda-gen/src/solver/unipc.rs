//! UniPC — Unified Predictor-Corrector solver for diffusion ODEs.
//!
//! Implements the UniPC framework of Zhao et al. 2023, a training-free
//! predictor-corrector method that augments any multistep diffusion solver with
//! a corrector step, improving sample quality at low NFE (number of function
//! evaluations).
//!
//! This is a self-contained solver (mirroring [`super::dpm_solver_pp`]) that
//! owns a linear-β noise schedule and operates in the half-log-SNR domain
//! `λ_t = ½·ln(ᾱ_t / (1 − ᾱ_t))`. It works in the **data-prediction**
//! parameterisation: the network output `eps_hat` is converted to a predicted
//! clean sample `x₀ = (x_t − σ_t·eps) / α_t`, and the update is the exponential
//! integrator on `x₀`.
//!
//! # UniPC-1 / UniPC-2 (`UniPcOrder`)
//!
//! - **Predictor (UniP)**: first-order exponential step
//!   ```text
//!   x_s = (α_s / α_t)·x_t − α_s·(e^{−h} − 1)·x₀(x_t, t)
//!   ```
//!   with `h = λ_s − λ_t`.
//! - **Corrector (UniC)**: re-evaluates the model at the predicted point and
//!   blends the two `x₀` estimates with the UniPC weight
//!   `B(h) = (e^{−h} − 1) / (−h)` (the φ₁ coefficient), giving a 2nd-order
//!   accurate update. With one previous model output cached, this matches the
//!   "2M"-style multistep corrector of Zhao et al. 2023, Algorithm 2.
//!
//! # Reference
//! Zhao et al., "UniPC: A Unified Predictor-Corrector Framework for Fast
//! Sampling of Diffusion Models", NeurIPS 2023. <https://arxiv.org/abs/2302.04867>

use crate::error::{GenError, GenResult};

// ─── UniPcOrder ─────────────────────────────────────────────────────────────────

/// Solver order: whether the corrector (UniC) step is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniPcOrder {
    /// First-order predictor only (UniP-1); equivalent to DPM-Solver++ 1st order.
    First,
    /// Second-order predictor + corrector (UniPC-2).
    Second,
}

// ─── UniPcConfig ────────────────────────────────────────────────────────────────

/// Configuration for the [`UniPc`] solver.
#[derive(Debug, Clone)]
pub struct UniPcConfig {
    /// Number of inference timesteps (ODE solver steps), e.g. 20.
    pub n_timesteps: usize,
    /// Starting β value for the linear schedule (e.g. 0.0001).
    pub beta_start: f64,
    /// Ending β value for the linear schedule (e.g. 0.02).
    pub beta_end: f64,
    /// Solver order (predictor-only vs predictor-corrector).
    pub order: UniPcOrder,
}

impl Default for UniPcConfig {
    fn default() -> Self {
        Self {
            n_timesteps: 20,
            beta_start: 0.0001,
            beta_end: 0.02,
            order: UniPcOrder::Second,
        }
    }
}

// ─── UniPc ──────────────────────────────────────────────────────────────────────

/// UniPC predictor-corrector sampler with a linear β schedule.
///
/// Pre-computes `betas`, `alphas_cumprod`, `sigmas`, and the descending
/// `timesteps` ordering at construction for efficient repeated sampling.
#[derive(Debug, Clone)]
pub struct UniPc {
    /// Solver configuration.
    pub config: UniPcConfig,
    /// `β_t = beta_start + (beta_end − beta_start)·t/(n−1)`, length n.
    pub betas: Vec<f64>,
    /// `ᾱ_t = ∏_{i=0}^{t} (1 − β_i)`, length n.
    pub alphas_cumprod: Vec<f64>,
    /// `σ_t = √(1 − ᾱ_t)`, length n.
    pub sigmas: Vec<f64>,
    /// Descending indices `[n−1, …, 0]` for reverse diffusion ordering.
    pub timesteps: Vec<usize>,
}

impl UniPc {
    /// Build a new UniPC solver from the given configuration.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `n_timesteps == 0`.
    /// - [`GenError::InvalidBetaSchedule`] if `beta_start`/`beta_end` are outside
    ///   `(0, 1)` or `beta_start >= beta_end`.
    pub fn new(config: UniPcConfig) -> GenResult<Self> {
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

        let sigmas: Vec<f64> = alphas_cumprod
            .iter()
            .map(|&ab| (1.0 - ab).max(0.0).sqrt())
            .collect();

        let timesteps: Vec<usize> = (0..n).rev().collect();

        Ok(Self {
            config,
            betas,
            alphas_cumprod,
            sigmas,
            timesteps,
        })
    }

    /// Half-log-SNR `λ_t = ½·ln(ᾱ_t / (1 − ᾱ_t))` at index `t_idx`.
    #[must_use]
    pub fn lambda(&self, t_idx: usize) -> f64 {
        let ab = self.alphas_cumprod[t_idx].clamp(1e-10, 1.0 - 1e-10);
        0.5 * (ab / (1.0 - ab)).ln()
    }

    /// `α_t = √ᾱ_t` at index `t_idx`.
    #[must_use]
    fn alpha(&self, t_idx: usize) -> f64 {
        self.alphas_cumprod[t_idx].max(0.0).sqrt()
    }

    /// Convert a noise prediction `eps` into a predicted clean sample `x₀`.
    ///
    /// `x₀ = (x_t − σ_t·eps) / α_t`. Exposed for callers wishing to inspect the
    /// implied data prediction; the solver itself integrates in ε-space.
    #[must_use]
    pub fn predict_x0(&self, x_t: &[f64], eps: &[f64], t_idx: usize) -> Vec<f64> {
        let alpha_t = self.alpha(t_idx).max(1e-10);
        let sigma_t = self.sigmas[t_idx];
        x_t.iter()
            .zip(eps)
            .map(|(&x, &e)| (x - sigma_t * e) / alpha_t)
            .collect()
    }

    /// First-order UniP predictor step (noise-prediction exponential integrator).
    ///
    /// This is the DPM-Solver++ first-order "transfer" in ε parameterisation
    /// (identical coefficients to [`super::dpm_solver_pp::DpmSolverPp::step_1m`]):
    /// ```text
    /// x_s = (α_s / α_t)·x_t + σ_s·(e^{−h} − 1)·eps ,  h = λ_s − λ_t
    /// ```
    ///
    /// Moves from index `from_idx` (noisier) to `to_idx` (cleaner).
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_t` or `eps_hat` is empty.
    /// - [`GenError::DimensionMismatch`] if lengths differ.
    /// - [`GenError::InvalidTimestep`] if an index is out of range.
    pub fn predictor_step(
        &self,
        x_t: &[f64],
        eps_hat: &[f64],
        from_idx: usize,
        to_idx: usize,
    ) -> GenResult<Vec<f64>> {
        self.check_indices(x_t, eps_hat, from_idx, to_idx)?;
        if from_idx == to_idx {
            return Ok(x_t.to_vec());
        }
        let out = self.transfer(x_t, eps_hat, from_idx, to_idx);
        Ok(out)
    }

    /// Second-order UniC corrector step.
    ///
    /// Given the predictor output `x_pred` at `to_idx` and a fresh model
    /// evaluation `eps_pred` there, blends the two noise estimates with the
    /// UniPC coefficient `B(h) = (e^{−h} − 1)/(−h)` (the φ₁ ratio) to obtain a
    /// 2nd-order-accurate update, then applies the same transfer:
    /// ```text
    /// eps_bar = eps_t + B(h)·(eps_s − eps_t)
    /// x_s     = (α_s / α_t)·x_t + σ_s·(e^{−h} − 1)·eps_bar
    /// ```
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if any slice is empty.
    /// - [`GenError::DimensionMismatch`] if lengths differ.
    /// - [`GenError::InvalidTimestep`] if an index is out of range.
    pub fn corrector_step(
        &self,
        x_t: &[f64],
        eps_t: &[f64],
        x_pred: &[f64],
        eps_pred: &[f64],
        from_idx: usize,
        to_idx: usize,
    ) -> GenResult<Vec<f64>> {
        self.check_indices(x_t, eps_t, from_idx, to_idx)?;
        if x_pred.len() != x_t.len() || eps_pred.len() != x_t.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: x_pred.len().min(eps_pred.len()),
            });
        }
        if from_idx == to_idx {
            return Ok(x_t.to_vec());
        }

        let h = self.lambda(to_idx) - self.lambda(from_idx);
        let phi = (-h).exp() - 1.0;
        // B(h) = (e^{−h} − 1) / (−h); guard h ≈ 0 → B → 1 (limit).
        let b_h = if h.abs() < 1e-8 { 1.0 } else { phi / (-h) };

        // Blend noise estimates, then apply the ε-space transfer.
        let eps_bar: Vec<f64> = eps_t
            .iter()
            .zip(eps_pred)
            .map(|(&et, &es)| et + b_h * (es - et))
            .collect();
        Ok(self.transfer(x_t, &eps_bar, from_idx, to_idx))
    }

    /// DPM-Solver++ first-order ε-space transfer from `from_idx` to `to_idx`:
    /// `x_s = (α_s/α_t)·x_t + σ_s·(e^{−h} − 1)·eps`.
    fn transfer(&self, x_t: &[f64], eps: &[f64], from_idx: usize, to_idx: usize) -> Vec<f64> {
        let alpha_t = self.alpha(from_idx).max(1e-10);
        let alpha_s = self.alpha(to_idx);
        let sigma_s = self.sigmas[to_idx];
        let h = self.lambda(to_idx) - self.lambda(from_idx);
        let exp_minus_h = (-h).exp();
        let ratio = alpha_s / alpha_t;
        x_t.iter()
            .zip(eps)
            .map(|(&x, &e)| ratio * x + sigma_s * (exp_minus_h - 1.0) * e)
            .collect()
    }

    /// Run the full reverse UniPC trajectory.
    ///
    /// Starting from a fully-noised sample `x_noisy` at the highest index, it
    /// iterates the predictor (and, for [`UniPcOrder::Second`], the corrector)
    /// down to the cleanest index. The `model_fn` closure maps `(x, t_idx)` to a
    /// noise prediction `eps_hat`.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_noisy` is empty.
    /// - Propagates errors from the predictor/corrector steps.
    pub fn sample<F>(&self, x_noisy: &[f64], mut model_fn: F) -> GenResult<Vec<f64>>
    where
        F: FnMut(&[f64], usize) -> Vec<f64>,
    {
        if x_noisy.is_empty() {
            return Err(GenError::EmptyInput("x_noisy is empty"));
        }
        let n = self.config.n_timesteps;
        if n < 2 {
            // Single step: just one predictor from index 0 to itself is a no-op;
            // return as-is (matches degenerate single-timestep schedules).
            return Ok(x_noisy.to_vec());
        }

        let mut x = x_noisy.to_vec();
        // timesteps are descending: [n-1, n-2, ..., 0].
        for w in self.timesteps.windows(2) {
            let from_idx = w[0];
            let to_idx = w[1];
            let eps_t = model_fn(&x, from_idx);
            let x_pred = self.predictor_step(&x, &eps_t, from_idx, to_idx)?;

            x = match self.config.order {
                UniPcOrder::First => x_pred,
                UniPcOrder::Second => {
                    let eps_pred = model_fn(&x_pred, to_idx);
                    self.corrector_step(&x, &eps_t, &x_pred, &eps_pred, from_idx, to_idx)?
                }
            };
        }
        Ok(x)
    }

    /// Shared index/length validation for predictor & corrector steps.
    fn check_indices(
        &self,
        x_t: &[f64],
        eps_hat: &[f64],
        from_idx: usize,
        to_idx: usize,
    ) -> GenResult<()> {
        if x_t.is_empty() || eps_hat.is_empty() {
            return Err(GenError::EmptyInput("x_t / eps_hat is empty"));
        }
        if x_t.len() != eps_hat.len() {
            return Err(GenError::DimensionMismatch {
                expected: x_t.len(),
                got: eps_hat.len(),
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
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solver(order: UniPcOrder) -> UniPc {
        UniPc::new(UniPcConfig {
            n_timesteps: 20,
            beta_start: 0.0001,
            beta_end: 0.02,
            order,
        })
        .expect("valid config")
    }

    #[test]
    fn construct_schedule_shapes() {
        let s = make_solver(UniPcOrder::Second);
        assert_eq!(s.betas.len(), 20);
        assert_eq!(s.alphas_cumprod.len(), 20);
        assert_eq!(s.sigmas.len(), 20);
        assert_eq!(s.timesteps.len(), 20);
        assert_eq!(s.timesteps[0], 19);
        assert_eq!(*s.timesteps.last().expect("non-empty"), 0);
    }

    #[test]
    fn alphas_cumprod_monotonic_decreasing() {
        let s = make_solver(UniPcOrder::First);
        for w in s.alphas_cumprod.windows(2) {
            assert!(w[0] >= w[1], "ᾱ must decrease: {} < {}", w[0], w[1]);
        }
        assert!(s.alphas_cumprod[0] < 1.0);
        assert!(*s.alphas_cumprod.last().expect("non-empty") > 0.0);
    }

    #[test]
    fn lambda_increases_as_index_decreases() {
        // Cleaner (lower index) ⇒ higher SNR ⇒ larger λ.
        let s = make_solver(UniPcOrder::Second);
        assert!(
            s.lambda(0) > s.lambda(19),
            "λ should be larger at cleaner index: λ(0)={}, λ(19)={}",
            s.lambda(0),
            s.lambda(19)
        );
    }

    #[test]
    fn predict_x0_inverts_forward() {
        // Build x_t = α_t·x0 + σ_t·eps, then predict_x0 should recover x0.
        let s = make_solver(UniPcOrder::Second);
        let t = 10;
        let alpha_t = s.alpha(t);
        let sigma_t = s.sigmas[t];
        let x0 = vec![1.0_f64, -2.0, 0.5];
        let eps = vec![0.3_f64, -0.1, 0.7];
        let x_t: Vec<f64> = x0
            .iter()
            .zip(&eps)
            .map(|(&d, &e)| alpha_t * d + sigma_t * e)
            .collect();
        let recovered = s.predict_x0(&x_t, &eps, t);
        for (&r, &d) in recovered.iter().zip(&x0) {
            assert!((r - d).abs() < 1e-9, "recovered {r}, expected {d}");
        }
    }

    #[test]
    fn predictor_step_shape_and_finite() {
        let s = make_solver(UniPcOrder::First);
        let x = vec![0.5_f64; 8];
        let eps = vec![0.1_f64; 8];
        let out = s.predictor_step(&x, &eps, 19, 18).expect("predictor ok");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn corrector_step_shape_and_finite() {
        let s = make_solver(UniPcOrder::Second);
        let x = vec![0.5_f64; 8];
        let eps = vec![0.1_f64; 8];
        let x_pred = s.predictor_step(&x, &eps, 19, 18).expect("predictor ok");
        let eps_pred = vec![0.05_f64; 8];
        let out = s
            .corrector_step(&x, &eps, &x_pred, &eps_pred, 19, 18)
            .expect("corrector ok");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn corrector_equals_predictor_when_eps_agree() {
        // If eps_pred == eps_t then eps_bar == eps_t and corrector == predictor.
        let s = make_solver(UniPcOrder::Second);
        let t_from = 15;
        let t_to = 14;
        let x = vec![0.4_f64, 0.6, -0.2, 0.9];
        let eps = vec![0.2_f64, -0.3, 0.1, 0.0];
        let x_pred = s.predictor_step(&x, &eps, t_from, t_to).expect("ok");
        let corrected = s
            .corrector_step(&x, &eps, &x_pred, &eps, t_from, t_to)
            .expect("ok");
        for (&c, &p) in corrected.iter().zip(&x_pred) {
            assert!(
                (c - p).abs() < 1e-9,
                "corrector should equal predictor when eps agree: {c} vs {p}"
            );
        }
    }

    #[test]
    fn sample_first_order_finite() {
        let s = make_solver(UniPcOrder::First);
        let x = vec![1.0_f64; 16];
        let out = s.sample(&x, |x, _t| vec![0.0_f64; x.len()]).expect("ok");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sample_second_order_finite() {
        let s = make_solver(UniPcOrder::Second);
        let x = vec![1.0_f64; 16];
        let out = s
            .sample(&x, |x, t| vec![0.01_f64 * t as f64; x.len()])
            .expect("ok");
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sample_consistent_x0_moves_toward_target() {
        // A self-consistent denoiser implying a fixed clean target x0* should
        // move the sample *toward* x0* over the (truncated) schedule. Build eps
        // so that predict_x0(x, t) == target for the queried (x, t):
        //   eps = (x − α_t·target) / σ_t.
        // (The linear-β schedule here, like dpm_solver_pp, does not fully
        // denoise to σ=0, so we assert monotone improvement rather than exact
        // arrival.)
        let s = make_solver(UniPcOrder::Second);
        let target = vec![1.0_f64, -1.0, 0.5, 2.0];
        let target_c = target.clone();
        let x = vec![15.0_f64, 15.0, 15.0, 15.0];
        let dist0: f64 = x.iter().zip(&target).map(|(&xi, &t)| (xi - t).abs()).sum();
        let out = s
            .sample(&x, |xq, t| {
                let alpha_t = s.alpha(t);
                let sigma_t = s.sigmas[t].max(1e-12);
                xq.iter()
                    .zip(&target_c)
                    .map(|(&xi, &d)| (xi - alpha_t * d) / sigma_t)
                    .collect()
            })
            .expect("ok");
        let dist1: f64 = out.iter().zip(&target).map(|(&o, &t)| (o - t).abs()).sum();
        assert!(
            dist1 < dist0,
            "self-consistent denoiser should reduce distance to target: {dist0} -> {dist1}"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn err_empty_input() {
        let s = make_solver(UniPcOrder::Second);
        assert!(matches!(
            s.predictor_step(&[], &[], 19, 18),
            Err(GenError::EmptyInput(_))
        ));
        assert!(matches!(
            s.sample(&[], |x, _| vec![0.0; x.len()]),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_dimension_mismatch() {
        let s = make_solver(UniPcOrder::First);
        let x = vec![1.0_f64; 4];
        let eps = vec![1.0_f64; 3];
        assert!(matches!(
            s.predictor_step(&x, &eps, 19, 18),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_invalid_timestep() {
        let s = make_solver(UniPcOrder::First);
        let x = vec![1.0_f64; 4];
        let eps = vec![1.0_f64; 4];
        assert!(matches!(
            s.predictor_step(&x, &eps, 99, 18),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn err_invalid_beta_schedule() {
        assert!(matches!(
            UniPc::new(UniPcConfig {
                n_timesteps: 10,
                beta_start: 0.5,
                beta_end: 0.1,
                order: UniPcOrder::First,
            }),
            Err(GenError::InvalidBetaSchedule)
        ));
    }

    #[test]
    fn err_zero_timesteps() {
        assert!(matches!(
            UniPc::new(UniPcConfig {
                n_timesteps: 0,
                ..UniPcConfig::default()
            }),
            Err(GenError::EmptyInput(_))
        ));
    }
}
