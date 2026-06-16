//! EDM (Elucidated Diffusion Model) scheduler.
//!
//! Implements the EDM framework from Karras et al. 2022 (NeurIPS), providing:
//! - Power-law sigma schedule (Eq. 5).
//! - Network preconditioning scalars: c_skip, c_out, c_in, c_noise (Section 5).
//! - Heun's 2nd-order ODE solver step (Algorithm 1).
//! - Log-normal training noise sampling (Section 5 / Appendix B).
//! - Full trajectory sampler running all ODE steps.
//!
//! # Reference
//! Karras et al., "Elucidating the Design Space of Diffusion-Based Generative
//! Models", NeurIPS 2022.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── EdmConfig ────────────────────────────────────────────────────────────────

/// Configuration for the EDM scheduler.
///
/// All defaults match those from Table 1 / Section 5 of Karras et al. 2022.
#[derive(Debug, Clone)]
pub struct EdmConfig {
    /// Lower bound on the noise schedule σ_min (default 0.002). Must be > 0.
    pub sigma_min: f32,
    /// Upper bound on the noise schedule σ_max (default 80.0). Must be > sigma_min.
    pub sigma_max: f32,
    /// Standard deviation of the data distribution σ_data (default 0.5). Must be > 0.
    pub sigma_data: f32,
    /// Schedule curvature exponent ρ (default 7.0). Must be > 0.
    pub rho: f32,
    /// Mean of the log-normal training noise distribution p_mean (default -1.2).
    pub p_mean: f32,
    /// Std of the log-normal training noise distribution p_std (default 1.2).
    pub p_std: f32,
    /// Number of ODE solver steps (default 18). Must be ≥ 1.
    pub n_steps: usize,
    /// Enable Heun's 2nd-order correction (default true).
    ///
    /// When `false`, the solver degrades to plain Euler integration.
    pub heun_correction: bool,
}

impl Default for EdmConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            sigma_data: 0.5,
            rho: 7.0,
            p_mean: -1.2,
            p_std: 1.2,
            n_steps: 18,
            heun_correction: true,
        }
    }
}

// ─── EdmScheduler ─────────────────────────────────────────────────────────────

/// EDM sampler implementing the Karras et al. 2022 framework.
///
/// Manages the σ schedule, EDM preconditioning scalars, a Heun ODE step,
/// log-normal noise sampling for training, and a full trajectory sampler.
///
/// The sigma schedule contains `n_steps + 1` values: `sigmas[0]` = σ_max,
/// `sigmas[n_steps]` = σ_min.
///
/// # Reference
/// Karras et al., "Elucidating the Design Space of Diffusion-Based Generative
/// Models", NeurIPS 2022, Algorithm 1 and Section 5.
#[derive(Debug, Clone)]
pub struct EdmScheduler {
    /// Configuration controlling the schedule and solver behaviour.
    pub cfg: EdmConfig,
    /// Precomputed sigma schedule with `n_steps + 1` values.
    ///
    /// `sigmas[i]` decreases monotonically from σ_max to σ_min.
    pub sigmas: Vec<f32>,
}

impl EdmScheduler {
    /// Build a new [`EdmScheduler`] from the given configuration.
    ///
    /// Constructs the σ schedule using the EDM power-law formula (Karras et al.
    /// 2022, Eq. 5):
    ///
    /// ```text
    /// σᵢ = (σ_max^{1/ρ} + i/N * (σ_min^{1/ρ} - σ_max^{1/ρ}))^ρ
    /// ```
    ///
    /// giving `n_steps + 1` values with `σ_0 = σ_max` and `σ_N = σ_min`.
    ///
    /// # Errors
    /// - [`GenError::InvalidGuidanceScale`] if any scalar parameter is invalid.
    /// - [`GenError::EmptyInput`] if `n_steps == 0`.
    pub fn new(cfg: EdmConfig) -> GenResult<Self> {
        if cfg.sigma_min <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(cfg.sigma_min));
        }
        if cfg.sigma_max <= cfg.sigma_min {
            return Err(GenError::InvalidGuidanceScale(cfg.sigma_max));
        }
        if cfg.sigma_data <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(cfg.sigma_data));
        }
        if cfg.rho <= 0.0 {
            return Err(GenError::InvalidGuidanceScale(cfg.rho));
        }
        if cfg.n_steps == 0 {
            return Err(GenError::EmptyInput("n_steps must be >= 1"));
        }

        let inv_rho = 1.0 / cfg.rho;
        let max_pow = cfg.sigma_max.powf(inv_rho);
        let min_pow = cfg.sigma_min.powf(inv_rho);
        let n = cfg.n_steps;

        let sigmas: Vec<f32> = (0..=n)
            .map(|i| {
                let frac = i as f32 / n as f32;
                let inner = max_pow + frac * (min_pow - max_pow);
                inner.powf(cfg.rho)
            })
            .collect();

        Ok(Self { cfg, sigmas })
    }

    // ── Preconditioning scalars (Karras et al. 2022, Section 5) ──────────────

    /// Skip-connection coefficient c_skip(σ).
    ///
    /// `c_skip(σ) = σ_data² / (σ² + σ_data²)`
    #[inline]
    #[must_use]
    pub fn c_skip(&self, sigma: f32) -> f32 {
        let sd2 = self.cfg.sigma_data * self.cfg.sigma_data;
        sd2 / (sigma * sigma + sd2)
    }

    /// Output-scaling coefficient c_out(σ).
    ///
    /// `c_out(σ) = σ · σ_data / √(σ² + σ_data²)`
    #[inline]
    #[must_use]
    pub fn c_out(&self, sigma: f32) -> f32 {
        let sd = self.cfg.sigma_data;
        let sd2 = sd * sd;
        sigma * sd / (sigma * sigma + sd2).sqrt()
    }

    /// Input-scaling coefficient c_in(σ).
    ///
    /// `c_in(σ) = 1 / √(σ² + σ_data²)`
    ///
    /// Normalises the network input to unit variance.
    #[inline]
    #[must_use]
    pub fn c_in(&self, sigma: f32) -> f32 {
        let sd2 = self.cfg.sigma_data * self.cfg.sigma_data;
        1.0 / (sigma * sigma + sd2).sqrt()
    }

    /// Noise-level encoding c_noise(σ).
    ///
    /// `c_noise(σ) = 0.25 · ln(max(σ, ε))` with ε = 1e-8.
    ///
    /// Maps σ to a conditioning scalar suitable for timestep embeddings.
    #[inline]
    #[must_use]
    pub fn c_noise(&self, sigma: f32) -> f32 {
        0.25 * sigma.max(1e-8_f32).ln()
    }

    // ── Preconditioning operations ────────────────────────────────────────────

    /// Scale the input: `c_in(σ) · x` (element-wise).
    ///
    /// Produces the network input by normalising `x` to approximately unit
    /// variance.
    #[must_use]
    pub fn preconditioning_scale(&self, x: &[f32], sigma: f32) -> Vec<f32> {
        let scale = self.c_in(sigma);
        x.iter().map(|&xi| scale * xi).collect()
    }

    /// Apply preconditioning to combine skip connection and network output.
    ///
    /// `output[i] = c_skip(σ) · x[i] + c_out(σ) · f_out[i]`
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `x.len() != f_out.len()`.
    pub fn preconditioning_output(
        &self,
        x: &[f32],
        sigma: f32,
        f_out: &[f32],
    ) -> GenResult<Vec<f32>> {
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

    // ── ODE solver ────────────────────────────────────────────────────────────

    /// Perform one step of Heun's 2nd-order ODE solver (Karras et al. 2022,
    /// Algorithm 1).
    ///
    /// **Euler step** (always performed):
    /// ```text
    /// d_cur = (x - D(x, σ_cur)) / σ_cur
    /// x_hat = x + (σ_next - σ_cur) · d_cur
    /// ```
    ///
    /// **Heun correction** (when `heun_correction && σ_next > 0`):
    /// ```text
    /// d_hat = (x_hat - D(x_hat, σ_next)) / σ_next
    /// x_next = x + (σ_next - σ_cur) · (d_cur/2 + d_hat/2)
    /// ```
    ///
    /// # Arguments
    /// - `x` — current noisy sample.
    /// - `sigma_cur` — current noise level σᵢ.
    /// - `sigma_next` — next (lower) noise level σᵢ₊₁.
    /// - `denoiser` — closure `|x: &[f32], sigma: f32| -> Vec<f32>` representing
    ///   the EDM-preconditioned network D(x, σ).
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x` is empty.
    pub fn ode_step<F>(
        &self,
        x: &[f32],
        sigma_cur: f32,
        sigma_next: f32,
        mut denoiser: F,
    ) -> GenResult<Vec<f32>>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }

        // First denoiser call and Euler step
        let d_cur_raw = denoiser(x, sigma_cur);
        let dt = sigma_next - sigma_cur; // negative (σ decreasing)

        let d_cur: Vec<f32> = x
            .iter()
            .zip(&d_cur_raw)
            .map(|(&xi, &di)| (xi - di) / sigma_cur)
            .collect();

        let x_hat: Vec<f32> = x
            .iter()
            .zip(&d_cur)
            .map(|(&xi, &dci)| xi + dt * dci)
            .collect();

        // Heun correction (2nd-order) only when σ_next > 0
        if self.cfg.heun_correction && sigma_next > 0.0 {
            let d_hat_raw = denoiser(&x_hat, sigma_next);
            let d_hat: Vec<f32> = x_hat
                .iter()
                .zip(&d_hat_raw)
                .map(|(&xh, &dh)| (xh - dh) / sigma_next)
                .collect();

            let x_next = x
                .iter()
                .zip(&d_cur)
                .zip(&d_hat)
                .map(|((&xi, &dci), &dhi)| xi + dt * (0.5 * dci + 0.5 * dhi))
                .collect();
            Ok(x_next)
        } else {
            Ok(x_hat)
        }
    }

    // ── Training noise sampling ───────────────────────────────────────────────

    /// Draw a training noise level σ from the log-normal distribution.
    ///
    /// `σ = exp(p_mean + p_std · z)` where `z ~ N(0, 1)`.
    ///
    /// The result is clamped to `[0.1 · σ_min, 10 · σ_max]` for numerical
    /// safety in training loops.
    pub fn sample_sigma(&self, rng: &mut LcgRng) -> f32 {
        let (z, _) = rng.next_normal_pair();
        let sigma = (self.cfg.p_mean + self.cfg.p_std * z).exp();
        sigma.clamp(self.cfg.sigma_min * 0.1, self.cfg.sigma_max * 10.0)
    }

    // ── Full trajectory sampler ───────────────────────────────────────────────

    /// Run the full ODE sampler trajectory over all `n_steps` steps.
    ///
    /// Iterates through `sigmas[0..n_steps]` pairs and calls
    /// [`Self::ode_step`] at each level, producing a clean sample from `x_init`.
    ///
    /// # Arguments
    /// - `x_init` — initial noisy sample at σ_max.
    /// - `denoiser` — EDM-preconditioned network D(x, σ).
    /// - `rng` — LCG RNG (used for sampling; passed for API consistency even
    ///   though the deterministic ODE solver itself does not require it).
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `x_init` is empty.
    /// - Propagates errors from [`Self::ode_step`].
    pub fn sample<F>(
        &self,
        x_init: &[f32],
        mut denoiser: F,
        _rng: &mut LcgRng,
    ) -> GenResult<Vec<f32>>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if x_init.is_empty() {
            return Err(GenError::EmptyInput("x_init is empty"));
        }

        let mut x = x_init.to_vec();
        let n = self.cfg.n_steps;

        for i in 0..n {
            let sigma_cur = self.sigmas[i];
            let sigma_next = self.sigmas[i + 1];
            x = self.ode_step(&x, sigma_cur, sigma_next, &mut denoiser)?;
        }

        Ok(x)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;
    const TINY: f32 = 1e-6;

    fn default_sched() -> EdmScheduler {
        EdmScheduler::new(EdmConfig::default()).expect("value should be present")
    }

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Schedule shape and boundary tests ────────────────────────────────────

    #[test]
    fn sigma_schedule_first() {
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
    fn sigma_schedule_last() {
        let s = default_sched();
        let n = s.cfg.n_steps;
        let diff = (s.sigmas[n] - s.cfg.sigma_min).abs();
        assert!(
            diff < EPS,
            "sigmas[n]={} vs sigma_min={}",
            s.sigmas[n],
            s.cfg.sigma_min
        );
    }

    #[test]
    fn sigma_schedule_len() {
        let s = default_sched();
        assert_eq!(
            s.sigmas.len(),
            s.cfg.n_steps + 1,
            "schedule length should be n_steps+1"
        );
    }

    #[test]
    fn sigma_schedule_monotone() {
        let s = default_sched();
        for w in s.sigmas.windows(2) {
            assert!(
                w[1] <= w[0] + TINY,
                "schedule not non-increasing: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    // ── Preconditioning scalar tests ──────────────────────────────────────────

    #[test]
    fn c_skip_at_sigma_data() {
        let s = default_sched();
        let sigma_data = s.cfg.sigma_data;
        let cs = s.c_skip(sigma_data);
        assert!(
            (cs - 0.5).abs() < TINY,
            "c_skip(sigma_data)={cs}, expected 0.5"
        );
    }

    #[test]
    fn c_out_formula() {
        // c_out(1.0) = 1.0 * sigma_data / sqrt(1.0 + sigma_data^2)
        let sigma_data = 0.5_f32;
        let cfg = EdmConfig {
            sigma_data,
            ..EdmConfig::default()
        };
        let s = EdmScheduler::new(cfg).expect("new should succeed");
        let expected = 1.0 * sigma_data / (1.0 + sigma_data * sigma_data).sqrt();
        let got = s.c_out(1.0);
        assert!(
            (got - expected).abs() < TINY,
            "c_out(1.0)={got} expected {expected}"
        );
    }

    #[test]
    fn c_in_formula() {
        // c_in(0) = 1 / sigma_data  (as σ → 0)
        let sigma_data = 0.5_f32;
        let cfg = EdmConfig {
            sigma_data,
            ..EdmConfig::default()
        };
        let s = EdmScheduler::new(cfg).expect("new should succeed");
        let expected = 1.0 / sigma_data;
        let got = s.c_in(0.0);
        assert!(
            (got - expected).abs() < TINY,
            "c_in(0)={got} expected {expected}"
        );
    }

    #[test]
    fn c_noise_at_one() {
        // c_noise(1.0) = 0.25 * ln(1.0) = 0.0
        let s = default_sched();
        let cn = s.c_noise(1.0);
        assert!(cn.abs() < TINY, "c_noise(1.0)={cn}, expected 0.0");
    }

    // ── Preconditioning operations tests ──────────────────────────────────────

    #[test]
    fn preconditioning_scale_shape() {
        let s = default_sched();
        let x = vec![1.0_f32; 16];
        let out = s.preconditioning_scale(&x, 1.0);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn preconditioning_scale_values() {
        let s = default_sched();
        let sigma = 2.0_f32;
        let x = vec![1.0_f32, 2.0, 3.0];
        let scale = s.c_in(sigma);
        let out = s.preconditioning_scale(&x, sigma);
        for (&o, &xi) in out.iter().zip(&x) {
            assert!((o - scale * xi).abs() < TINY, "scale mismatch");
        }
    }

    #[test]
    fn preconditioning_output_shape() {
        let s = default_sched();
        let x = vec![1.0_f32; 8];
        let f_out = vec![0.5_f32; 8];
        let out = s
            .preconditioning_output(&x, 1.0, &f_out)
            .expect("preconditioning_output should succeed");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn err_preconditioning_dim_mismatch() {
        let s = default_sched();
        let x = vec![1.0_f32; 4];
        let f_out = vec![0.5_f32; 8];
        assert!(matches!(
            s.preconditioning_output(&x, 1.0, &f_out),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    // ── ODE step tests ────────────────────────────────────────────────────────

    #[test]
    fn ode_step_shape() {
        let s = default_sched();
        let x = vec![1.0_f32; 8];
        let out = s
            .ode_step(&x, 1.0, 0.5, |xi, _| xi.to_vec())
            .expect("value should be present");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn ode_step_euler_no_heun() {
        // With heun_correction=false, only the Euler step runs (no second denoiser call).
        // Verify this by using an identity denoiser and checking that the output is
        // still a valid finite vector of the correct shape.
        let cfg = EdmConfig {
            heun_correction: false,
            ..EdmConfig::default()
        };
        let s = EdmScheduler::new(cfg).expect("new should succeed");
        let x = vec![2.0_f32; 4];
        let out = s
            .ode_step(&x, 1.0, 0.5, |xi, _| xi.to_vec())
            .expect("value should be present");
        assert_eq!(out.len(), 4, "output shape preserved");
        // The identity denoiser gives d_cur = (x - x)/sigma = 0, so x_hat = x + dt*0 = x
        // (no change expected with identity denoiser in Euler mode)
        for (&o, &xi) in out.iter().zip(&x) {
            assert!(
                (o - xi).abs() < 1e-5,
                "identity-denoiser Euler step should leave x unchanged: got {o}, expected {xi}"
            );
        }
    }

    #[test]
    fn ode_step_empty_x() {
        let s = default_sched();
        assert!(matches!(
            s.ode_step(&[], 1.0, 0.5, |xi, _| xi.to_vec()),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn ode_step_finite_output() {
        let s = default_sched();
        let x = vec![1.0_f32, -1.0, 0.5];
        let out = s
            .ode_step(&x, 2.0, 1.0, |xi, _| xi.to_vec())
            .expect("value should be present");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite ODE output");
    }

    // ── sample_sigma tests ────────────────────────────────────────────────────

    #[test]
    fn sample_sigma_positive() {
        let s = default_sched();
        let mut rng = make_rng();
        for _ in 0..200 {
            let sigma = s.sample_sigma(&mut rng);
            assert!(sigma > 0.0, "sample_sigma must be positive, got {sigma}");
        }
    }

    #[test]
    fn sample_sigma_within_clamped_range() {
        let s = default_sched();
        let mut rng = make_rng();
        let lo = s.cfg.sigma_min * 0.1;
        let hi = s.cfg.sigma_max * 10.0;
        for _ in 0..500 {
            let sigma = s.sample_sigma(&mut rng);
            assert!(
                sigma >= lo && sigma <= hi,
                "sample_sigma={sigma} out of clamped range [{lo}, {hi}]"
            );
        }
    }

    // ── sample (trajectory) tests ─────────────────────────────────────────────

    #[test]
    fn sample_shape() {
        let s = default_sched();
        let x = vec![1.0_f32; 16];
        let mut rng = make_rng();
        let out = s
            .sample(&x, |xi, _| xi.to_vec(), &mut rng)
            .expect("value should be present");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn sample_n_steps_1() {
        let cfg = EdmConfig {
            n_steps: 1,
            ..EdmConfig::default()
        };
        let s = EdmScheduler::new(cfg).expect("new should succeed");
        let x = vec![2.0_f32; 4];
        let mut rng = make_rng();
        let out = s
            .sample(&x, |xi, _| xi.to_vec(), &mut rng)
            .expect("value should be present");
        assert_eq!(out.len(), 4, "n_steps=1 sample shape");
    }

    #[test]
    fn sample_finite_output() {
        let s = default_sched();
        let x = vec![1.0_f32; 8];
        let mut rng = make_rng();
        // Identity denoiser
        let out = s
            .sample(&x, |xi, _| xi.to_vec(), &mut rng)
            .expect("value should be present");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite sample output"
        );
    }

    // ── Error-path construction tests ─────────────────────────────────────────

    #[test]
    fn err_sigma_min_zero() {
        let cfg = EdmConfig {
            sigma_min: 0.0,
            ..EdmConfig::default()
        };
        assert!(matches!(
            EdmScheduler::new(cfg),
            Err(GenError::InvalidGuidanceScale(_))
        ));
    }

    #[test]
    fn err_sigma_max_le_min() {
        let cfg = EdmConfig {
            sigma_min: 1.0,
            sigma_max: 0.5,
            ..EdmConfig::default()
        };
        assert!(matches!(
            EdmScheduler::new(cfg),
            Err(GenError::InvalidGuidanceScale(_))
        ));
    }

    #[test]
    fn err_n_steps_zero() {
        let cfg = EdmConfig {
            n_steps: 0,
            ..EdmConfig::default()
        };
        assert!(matches!(
            EdmScheduler::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn default_config_sigma_min() {
        let cfg = EdmConfig::default();
        assert!((cfg.sigma_min - 0.002).abs() < TINY, "sigma_min default");
        assert!((cfg.sigma_max - 80.0).abs() < TINY, "sigma_max default");
        assert!((cfg.sigma_data - 0.5).abs() < TINY, "sigma_data default");
        assert_eq!(cfg.n_steps, 18, "n_steps default");
        assert!(cfg.heun_correction, "heun_correction default");
    }
}
