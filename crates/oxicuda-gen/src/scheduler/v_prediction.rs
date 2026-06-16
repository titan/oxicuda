//! V-prediction parameterization for variance-preserving diffusion.
//!
//! Implements the v-parameterization of Salimans & Ho 2022, "Progressive
//! Distillation for Fast Sampling of Diffusion Models" (Appendix D).
//!
//! For a variance-preserving (VP) forward process the signal rate is
//! `α_t = √ᾱ_t` and the noise rate is `σ_t = √(1 − ᾱ_t)`, so that
//! `α_t² + σ_t² = 1`. The forward corruption is
//!
//! ```text
//! z_t = α_t · x + σ_t · ε .
//! ```
//!
//! The *velocity* target is defined as
//!
//! ```text
//! v ≡ α_t · ε − σ_t · x .
//! ```
//!
//! The map `(x, ε) → (z_t, v)` is an orthonormal rotation in the
//! `(x, ε)` plane (rotation matrix `[[α, σ], [−σ, α]]` with `α² + σ² = 1`),
//! so it is exactly invertible:
//!
//! ```text
//! x = α_t · z_t − σ_t · v ,
//! ε = σ_t · z_t + α_t · v .
//! ```
//!
//! Predicting `v` rather than `ε` yields an objective whose effective loss
//! weighting is roughly constant across timesteps, which stabilises training
//! and distillation.
//!
//! # Reference
//! Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion
//! Models", ICLR 2022, Appendix D.

use crate::error::{GenError, GenResult};

// ─── VPredictionConfig ──────────────────────────────────────────────────────────

/// Configuration for the [`VPrediction`] parameterization.
#[derive(Debug, Clone, PartialEq)]
pub struct VPredictionConfig {
    /// Number of diffusion timesteps. Must be ≥ 1.
    pub n_timesteps: usize,
    /// First (smallest) beta in the linear schedule. Must be > 0.
    pub beta_start: f32,
    /// Last (largest) beta in the linear schedule. Must satisfy
    /// `beta_start ≤ beta_end < 1`.
    pub beta_end: f32,
}

// ─── VPrediction ─────────────────────────────────────────────────────────────────

/// V-prediction parameterization with precomputed signal and noise rates.
///
/// Stores the cumulative product schedule `ᾱ_t`, the signal rate
/// `α_t = √ᾱ_t`, and the noise rate `σ_t = √(1 − ᾱ_t)`, and provides the
/// velocity target together with exact inverses derived from the orthonormal
/// `(x, ε) → (z, v)` rotation.
///
/// # Reference
/// Salimans & Ho, "Progressive Distillation for Fast Sampling of Diffusion
/// Models", ICLR 2022, Appendix D.
#[derive(Debug, Clone)]
pub struct VPrediction {
    /// Configuration controlling the linear beta schedule.
    pub cfg: VPredictionConfig,
    /// Cumulative product `ᾱ_t = Π_{s≤t} (1 − β_s)`.
    pub alphas_cumprod: Vec<f32>,
    /// Signal rate `α_t = √ᾱ_t`.
    pub alpha_t: Vec<f32>,
    /// Noise rate `σ_t = √(1 − ᾱ_t)`.
    pub sigma_t: Vec<f32>,
}

impl VPrediction {
    /// Build a new [`VPrediction`] from the given configuration.
    ///
    /// Uses a linear beta schedule
    /// `β_t = beta_start + (beta_end − beta_start) · t / (n_timesteps − 1)`
    /// for `t ∈ 0..n_timesteps` (with `β_0 = beta_start` when
    /// `n_timesteps == 1`), forms `ᾱ_t` as the cumulative product of
    /// `1 − β_t`, then `α_t = √ᾱ_t` and `σ_t = √(1 − ᾱ_t)`. Square-root
    /// arguments are clamped to be non-negative for numerical safety.
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if `n_timesteps == 0`.
    /// - [`GenError::InvalidBetaSchedule`] if `beta_start ≤ 0`,
    ///   `beta_end < beta_start`, or `beta_end ≥ 1`.
    pub fn new(cfg: VPredictionConfig) -> GenResult<Self> {
        if cfg.n_timesteps == 0 {
            return Err(GenError::EmptyInput("n_timesteps must be >= 1"));
        }
        if cfg.beta_start <= 0.0 {
            return Err(GenError::InvalidBetaSchedule);
        }
        if cfg.beta_end < cfg.beta_start {
            return Err(GenError::InvalidBetaSchedule);
        }
        if cfg.beta_end >= 1.0 {
            return Err(GenError::InvalidBetaSchedule);
        }

        let n = cfg.n_timesteps;
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut alpha_t = Vec::with_capacity(n);
        let mut sigma_t = Vec::with_capacity(n);

        let mut cumprod = 1.0_f32;
        for t in 0..n {
            let beta = if n == 1 {
                cfg.beta_start
            } else {
                cfg.beta_start + (cfg.beta_end - cfg.beta_start) * t as f32 / (n as f32 - 1.0)
            };
            let alpha_step = 1.0 - beta;
            cumprod *= alpha_step;
            let abar = cumprod.max(0.0);
            alphas_cumprod.push(abar);
            alpha_t.push(abar.max(0.0).sqrt());
            sigma_t.push((1.0 - abar).max(0.0).sqrt());
        }

        Ok(Self {
            cfg,
            alphas_cumprod,
            alpha_t,
            sigma_t,
        })
    }

    /// Validate a timestep index against `n_timesteps`.
    #[inline]
    fn check_t(&self, t: usize) -> GenResult<()> {
        if t >= self.cfg.n_timesteps {
            return Err(GenError::InvalidTimestep {
                t,
                max_t: self.cfg.n_timesteps,
            });
        }
        Ok(())
    }

    /// Validate that two slices have equal length.
    #[inline]
    fn check_pair(a: &[f32], b: &[f32]) -> GenResult<()> {
        if a.len() != b.len() {
            return Err(GenError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        Ok(())
    }

    /// Velocity target `v = α_t · ε − σ_t · x0` (element-wise).
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `t ≥ n_timesteps`.
    /// - [`GenError::DimensionMismatch`] if `x0.len() != eps.len()`.
    pub fn compute_v(&self, x0: &[f32], eps: &[f32], t: usize) -> GenResult<Vec<f32>> {
        self.check_t(t)?;
        Self::check_pair(x0, eps)?;
        let a = self.alpha_t[t];
        let s = self.sigma_t[t];
        let out = x0.iter().zip(eps).map(|(&x, &e)| a * e - s * x).collect();
        Ok(out)
    }

    /// Forward corruption `z_t = α_t · x0 + σ_t · ε` (element-wise).
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `t ≥ n_timesteps`.
    /// - [`GenError::DimensionMismatch`] if `x0.len() != eps.len()`.
    pub fn add_noise(&self, x0: &[f32], eps: &[f32], t: usize) -> GenResult<Vec<f32>> {
        self.check_t(t)?;
        Self::check_pair(x0, eps)?;
        let a = self.alpha_t[t];
        let s = self.sigma_t[t];
        let out = x0.iter().zip(eps).map(|(&x, &e)| a * x + s * e).collect();
        Ok(out)
    }

    /// Recover the clean signal `x0 = α_t · z_t − σ_t · v` (element-wise).
    ///
    /// This is the exact inverse implied by the orthonormal `(x, ε) → (z, v)`
    /// rotation.
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `t ≥ n_timesteps`.
    /// - [`GenError::DimensionMismatch`] if `z_t.len() != v_pred.len()`.
    pub fn predict_x0(&self, z_t: &[f32], v_pred: &[f32], t: usize) -> GenResult<Vec<f32>> {
        self.check_t(t)?;
        Self::check_pair(z_t, v_pred)?;
        let a = self.alpha_t[t];
        let s = self.sigma_t[t];
        let out = z_t
            .iter()
            .zip(v_pred)
            .map(|(&z, &v)| a * z - s * v)
            .collect();
        Ok(out)
    }

    /// Recover the noise `ε = σ_t · z_t + α_t · v` (element-wise).
    ///
    /// This is the exact inverse implied by the orthonormal `(x, ε) → (z, v)`
    /// rotation.
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `t ≥ n_timesteps`.
    /// - [`GenError::DimensionMismatch`] if `z_t.len() != v_pred.len()`.
    pub fn predict_eps(&self, z_t: &[f32], v_pred: &[f32], t: usize) -> GenResult<Vec<f32>> {
        self.check_t(t)?;
        Self::check_pair(z_t, v_pred)?;
        let a = self.alpha_t[t];
        let s = self.sigma_t[t];
        let out = z_t
            .iter()
            .zip(v_pred)
            .map(|(&z, &v)| s * z + a * v)
            .collect();
        Ok(out)
    }

    /// Signal-to-noise ratio `SNR(t) = ᾱ_t / (1 − ᾱ_t)`.
    ///
    /// The denominator is clamped to `1e-12` so that a near-clean step
    /// (`ᾱ_t ≈ 1`) yields a large but finite value rather than overflowing.
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `t ≥ n_timesteps`.
    pub fn snr(&self, t: usize) -> GenResult<f32> {
        self.check_t(t)?;
        let abar = self.alphas_cumprod[t];
        let denom = (1.0 - abar).max(1e-12);
        Ok(abar / denom)
    }

    /// Loss weighting for the v-prediction objective.
    ///
    /// V-prediction's hallmark is a roughly *constant* loss weighting across
    /// timesteps. Concretely, the eps-objective weight `SNR(t)` and the
    /// x0-objective weight `1` are reconciled by the "truncated SNR + 1"
    /// view: the effective v-loss weight is
    /// `SNR(t) / (SNR(t) + 1) · (SNR(t) + 1) / SNR(t) = 1`. We therefore
    /// return the constant unit weight.
    ///
    /// # Errors
    /// - [`GenError::InvalidTimestep`] if `t ≥ n_timesteps`.
    pub fn loss_weight(&self, t: usize) -> GenResult<f32> {
        self.check_t(t)?;
        Ok(1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;
    const TINY: f32 = 1e-6;

    fn default_cfg() -> VPredictionConfig {
        VPredictionConfig {
            n_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 0.02,
        }
    }

    fn make_vp() -> VPrediction {
        VPrediction::new(default_cfg()).expect("value should be present")
    }

    #[test]
    fn alphas_cumprod_monotone_non_increasing() {
        let vp = make_vp();
        for w in vp.alphas_cumprod.windows(2) {
            assert!(
                w[1] <= w[0] + TINY,
                "alphas_cumprod not non-increasing: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn alpha_sq_plus_sigma_sq_is_one() {
        let vp = make_vp();
        for t in 0..vp.cfg.n_timesteps {
            let a = vp.alpha_t[t];
            let s = vp.sigma_t[t];
            let sum = a * a + s * s;
            assert!((sum - 1.0).abs() < EPS, "α²+σ²={sum} at t={t}");
        }
    }

    #[test]
    fn compute_v_hand_example() {
        // At t=0 with the default schedule, beta_0 = 1e-4 so abar_0 = 1 - 1e-4.
        // alpha ≈ 0.99995, sigma ≈ 0.01.
        let vp = make_vp();
        let x0 = vec![1.0_f32, -2.0];
        let eps = vec![0.5_f32, 0.25];
        let v = vp
            .compute_v(&x0, &eps, 0)
            .expect("compute_v should succeed");
        let a = vp.alpha_t[0];
        let s = vp.sigma_t[0];
        for i in 0..2 {
            let expected = a * eps[i] - s * x0[i];
            assert!(
                (v[i] - expected).abs() < EPS,
                "compute_v[{i}]={} expected {expected}",
                v[i]
            );
        }
    }

    #[test]
    fn add_noise_formula() {
        let vp = make_vp();
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let eps = vec![0.1_f32, -0.2, 0.3];
        let t = 50;
        let z = vp
            .add_noise(&x0, &eps, t)
            .expect("add_noise should succeed");
        let a = vp.alpha_t[t];
        let s = vp.sigma_t[t];
        for i in 0..3 {
            let expected = a * x0[i] + s * eps[i];
            assert!(
                (z[i] - expected).abs() < EPS,
                "add_noise[{i}]={} expected {expected}",
                z[i]
            );
        }
    }

    #[test]
    fn predict_x0_roundtrip() {
        let vp = make_vp();
        let x0 = vec![0.7_f32, -1.3, 2.1, 0.0];
        let eps = vec![0.2_f32, 0.9, -0.4, 1.1];
        for &t in &[0_usize, 10, 50, 99] {
            let z = vp
                .add_noise(&x0, &eps, t)
                .expect("add_noise should succeed");
            let v = vp
                .compute_v(&x0, &eps, t)
                .expect("compute_v should succeed");
            let x0_hat = vp.predict_x0(&z, &v, t).expect("predict_x0 should succeed");
            for i in 0..x0.len() {
                assert!(
                    (x0_hat[i] - x0[i]).abs() < EPS,
                    "predict_x0 roundtrip[{i}] at t={t}: {} != {}",
                    x0_hat[i],
                    x0[i]
                );
            }
        }
    }

    #[test]
    fn predict_eps_roundtrip() {
        let vp = make_vp();
        let x0 = vec![0.7_f32, -1.3, 2.1, 0.0];
        let eps = vec![0.2_f32, 0.9, -0.4, 1.1];
        for &t in &[0_usize, 10, 50, 99] {
            let z = vp
                .add_noise(&x0, &eps, t)
                .expect("add_noise should succeed");
            let v = vp
                .compute_v(&x0, &eps, t)
                .expect("compute_v should succeed");
            let eps_hat = vp
                .predict_eps(&z, &v, t)
                .expect("predict_eps should succeed");
            for i in 0..eps.len() {
                assert!(
                    (eps_hat[i] - eps[i]).abs() < EPS,
                    "predict_eps roundtrip[{i}] at t={t}: {} != {}",
                    eps_hat[i],
                    eps[i]
                );
            }
        }
    }

    #[test]
    fn snr_decreasing_in_t() {
        let vp = make_vp();
        let mut prev = f32::INFINITY;
        for t in 0..vp.cfg.n_timesteps {
            let snr = vp.snr(t).expect("snr should succeed");
            assert!(
                snr <= prev + TINY,
                "SNR not decreasing at t={t}: {snr} > {prev}"
            );
            prev = snr;
        }
    }

    #[test]
    fn snr_positive() {
        let vp = make_vp();
        for t in 0..vp.cfg.n_timesteps {
            let snr = vp.snr(t).expect("snr should succeed");
            assert!(snr > 0.0, "SNR must be positive at t={t}: {snr}");
        }
    }

    #[test]
    fn snr_at_t0_large() {
        // Near-clean step => SNR should be large.
        let vp = make_vp();
        let snr0 = vp.snr(0).expect("snr should succeed");
        assert!(snr0 > 100.0, "SNR at t=0 should be large, got {snr0}");
    }

    #[test]
    fn loss_weight_is_one() {
        let vp = make_vp();
        for t in 0..vp.cfg.n_timesteps {
            let w = vp.loss_weight(t).expect("loss_weight should succeed");
            assert!((w - 1.0).abs() < TINY, "loss_weight should be 1.0, got {w}");
        }
    }

    #[test]
    fn early_signal_rate_near_one() {
        let vp = make_vp();
        assert!(
            vp.alpha_t[0] > 0.99,
            "early signal rate should be near 1, got {}",
            vp.alpha_t[0]
        );
    }

    #[test]
    fn late_noise_rate_grows() {
        // With this schedule the noise rate increases towards the final step.
        let vp = make_vp();
        let n = vp.cfg.n_timesteps;
        assert!(
            vp.sigma_t[n - 1] > vp.sigma_t[0],
            "late noise rate {} should exceed early {}",
            vp.sigma_t[n - 1],
            vp.sigma_t[0]
        );
    }

    #[test]
    fn late_noise_rate_near_one_long_schedule() {
        // A long, aggressive schedule drives ᾱ_t → 0 so σ_t → 1.
        let vp = VPrediction::new(VPredictionConfig {
            n_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
        })
        .expect("value should be present");
        let n = vp.cfg.n_timesteps;
        assert!(
            vp.sigma_t[n - 1] > 0.99,
            "late noise rate should approach 1, got {}",
            vp.sigma_t[n - 1]
        );
    }

    #[test]
    fn err_t_out_of_range() {
        let vp = make_vp();
        let x0 = vec![1.0_f32; 3];
        let eps = vec![0.0_f32; 3];
        assert!(matches!(
            vp.compute_v(&x0, &eps, 100),
            Err(GenError::InvalidTimestep { .. })
        ));
        assert!(matches!(vp.snr(100), Err(GenError::InvalidTimestep { .. })));
        assert!(matches!(
            vp.loss_weight(100),
            Err(GenError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn err_x0_eps_length_mismatch() {
        let vp = make_vp();
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let eps = vec![0.0_f32, 0.0];
        assert!(matches!(
            vp.compute_v(&x0, &eps, 0),
            Err(GenError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            vp.add_noise(&x0, &eps, 0),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_z_v_length_mismatch() {
        let vp = make_vp();
        let z = vec![1.0_f32, 2.0, 3.0];
        let v = vec![0.0_f32, 0.0];
        assert!(matches!(
            vp.predict_x0(&z, &v, 0),
            Err(GenError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            vp.predict_eps(&z, &v, 0),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_n_timesteps_zero() {
        let cfg = VPredictionConfig {
            n_timesteps: 0,
            beta_start: 1e-4,
            beta_end: 0.02,
        };
        assert!(matches!(
            VPrediction::new(cfg),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn err_beta_start_non_positive() {
        let cfg = VPredictionConfig {
            n_timesteps: 100,
            beta_start: 0.0,
            beta_end: 0.02,
        };
        assert!(matches!(
            VPrediction::new(cfg),
            Err(GenError::InvalidBetaSchedule)
        ));
    }

    #[test]
    fn err_beta_end_less_than_start() {
        let cfg = VPredictionConfig {
            n_timesteps: 100,
            beta_start: 0.02,
            beta_end: 0.01,
        };
        assert!(matches!(
            VPrediction::new(cfg),
            Err(GenError::InvalidBetaSchedule)
        ));
    }

    #[test]
    fn err_beta_end_ge_one() {
        let cfg = VPredictionConfig {
            n_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 1.0,
        };
        assert!(matches!(
            VPrediction::new(cfg),
            Err(GenError::InvalidBetaSchedule)
        ));
    }

    #[test]
    fn dim_one_works() {
        let vp = make_vp();
        let x0 = vec![1.5_f32];
        let eps = vec![-0.5_f32];
        let t = 25;
        let z = vp
            .add_noise(&x0, &eps, t)
            .expect("add_noise should succeed");
        let v = vp
            .compute_v(&x0, &eps, t)
            .expect("compute_v should succeed");
        let x0_hat = vp.predict_x0(&z, &v, t).expect("predict_x0 should succeed");
        let eps_hat = vp
            .predict_eps(&z, &v, t)
            .expect("predict_eps should succeed");
        assert_eq!(x0_hat.len(), 1);
        assert!((x0_hat[0] - x0[0]).abs() < EPS, "dim=1 x0: {}", x0_hat[0]);
        assert!(
            (eps_hat[0] - eps[0]).abs() < EPS,
            "dim=1 eps: {}",
            eps_hat[0]
        );
    }

    #[test]
    fn deterministic_construction() {
        let a = make_vp();
        let b = make_vp();
        for t in 0..a.cfg.n_timesteps {
            assert!((a.alpha_t[t] - b.alpha_t[t]).abs() < TINY);
            assert!((a.sigma_t[t] - b.sigma_t[t]).abs() < TINY);
            assert!((a.alphas_cumprod[t] - b.alphas_cumprod[t]).abs() < TINY);
        }
    }

    #[test]
    fn n_timesteps_one_uses_beta_start() {
        // With n_timesteps == 1, beta_0 = beta_start and abar_0 = 1 - beta_start.
        let vp = VPrediction::new(VPredictionConfig {
            n_timesteps: 1,
            beta_start: 0.01,
            beta_end: 0.02,
        })
        .expect("value should be present");
        assert_eq!(vp.alphas_cumprod.len(), 1);
        let expected_abar = 1.0 - 0.01_f32;
        assert!(
            (vp.alphas_cumprod[0] - expected_abar).abs() < EPS,
            "abar_0={} expected {expected_abar}",
            vp.alphas_cumprod[0]
        );
    }
}
