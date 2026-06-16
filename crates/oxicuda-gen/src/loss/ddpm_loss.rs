//! DDPM denoising diffusion loss.
//!
//! Implements the simple noise-prediction loss from Ho et al. (2020)
//! "Denoising Diffusion Probabilistic Models" (NeurIPS 2020).
//!
//! The forward process corrupts a clean sample `x_0` into a noisy sample
//! `x_t` via:
//!
//! ```text
//! x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε,   ε ~ N(0, I)
//! ```
//!
//! where `ᾱ_t = ∏_{s=0}^{t} (1 - β_s)`.  The model is trained to predict
//! the added noise `ε` using one of several pixel-space loss types.

use crate::error::{GenError, GenResult};

/// Type alias for the crate-level LCG random number generator.
pub type GenRng = crate::handle::LcgRng;

// ─── DdpmLossType ────────────────────────────────────────────────────────────

/// The pixel-space loss used to compare predicted noise to the true noise.
///
/// All loss types produce a non-negative scalar averaged over the full
/// `batch_size × d` prediction tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DdpmLossType {
    /// Mean-squared error: `mean((ε_pred - ε)²)`.
    L2,
    /// Mean absolute error: `mean(|ε_pred - ε|)`.
    L1,
    /// Huber loss with `δ = 1.0`.
    ///
    /// `L(a) = 0.5·a²` if `|a| ≤ 1`, else `|a| - 0.5`.
    Huber,
}

// ─── DdpmLossConfig ──────────────────────────────────────────────────────────

/// Configuration for the DDPM loss module.
#[derive(Debug, Clone)]
pub struct DdpmLossConfig {
    /// Number of diffusion timesteps `T` in the schedule.
    pub n_timesteps: usize,
    /// Starting value of the linear beta schedule (`β_0`).
    pub beta_start: f64,
    /// Ending value of the linear beta schedule (`β_{T-1}`).
    pub beta_end: f64,
    /// Which pixel-space loss to use when comparing noise predictions.
    pub loss_type: DdpmLossType,
}

impl DdpmLossConfig {
    /// Create a configuration with the canonical DDPM settings and L2 loss.
    ///
    /// Uses `T = 1000`, `β_start = 1e-4`, `β_end = 0.02`.
    #[must_use]
    pub fn default_ddpm() -> Self {
        Self {
            n_timesteps: 1000,
            beta_start: 1e-4,
            beta_end: 0.02,
            loss_type: DdpmLossType::L2,
        }
    }
}

// ─── DdpmLoss ────────────────────────────────────────────────────────────────

/// DDPM denoising loss with a linear beta schedule.
///
/// Pre-computes and stores the cumulative product of alphas (`ᾱ_t`) and the
/// individual betas (`β_t`) for all `T` timesteps.
///
/// # Example
///
/// ```rust
/// use oxicuda_gen::loss::DdpmLoss;
/// use oxicuda_gen::loss::DdpmLossConfig;
///
/// let config = DdpmLossConfig::default_ddpm();
/// let loss_fn = DdpmLoss::new(config).expect("new should succeed");
/// assert_eq!(loss_fn.n_timesteps(), 1000);
/// ```
#[derive(Debug, Clone)]
pub struct DdpmLoss {
    /// `ᾱ_t = ∏_{s=0}^{t} (1 - β_s)` for each timestep `t`.
    alphas_cumprod: Vec<f64>,
    /// Individual `β_t` for each timestep `t`.
    betas: Vec<f64>,
    /// Configuration this instance was built from.
    config: DdpmLossConfig,
}

impl DdpmLoss {
    /// Construct a `DdpmLoss` from the given configuration.
    ///
    /// Computes the linear beta schedule and its cumulative-alpha products.
    ///
    /// # Errors
    ///
    /// Returns [`GenError::EmptyInput`] if `n_timesteps == 0`.
    /// Returns [`GenError::InvalidBetaSchedule`] if any beta is not in `(0, 1)`
    /// or if `beta_start >= beta_end`.
    pub fn new(config: DdpmLossConfig) -> GenResult<Self> {
        if config.n_timesteps == 0 {
            return Err(GenError::EmptyInput("n_timesteps must be > 0"));
        }
        if !(0.0 < config.beta_start && config.beta_start < 1.0) {
            return Err(GenError::InvalidBetaSchedule);
        }
        if !(0.0 < config.beta_end && config.beta_end < 1.0) {
            return Err(GenError::InvalidBetaSchedule);
        }
        if config.beta_start >= config.beta_end {
            return Err(GenError::InvalidBetaSchedule);
        }

        let n = config.n_timesteps;
        let mut betas = Vec::with_capacity(n);
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut cumprod = 1.0_f64;

        for t in 0..n {
            let beta = if n == 1 {
                config.beta_start
            } else {
                config.beta_start
                    + (config.beta_end - config.beta_start) * (t as f64) / ((n - 1) as f64)
            };
            betas.push(beta);
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        Ok(Self {
            alphas_cumprod,
            betas,
            config,
        })
    }

    /// Apply the forward diffusion process: corrupt `x_0` to `x_t`.
    ///
    /// `x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · noise`
    ///
    /// # Errors
    ///
    /// - [`GenError::InvalidTimestep`] if `t >= n_timesteps`.
    /// - [`GenError::DimensionMismatch`] if `x0` and `noise` have different lengths.
    pub fn q_sample(&self, x0: &[f64], t: usize, noise: &[f64]) -> GenResult<Vec<f64>> {
        if t >= self.config.n_timesteps {
            return Err(GenError::InvalidTimestep {
                t,
                max_t: self.config.n_timesteps - 1,
            });
        }
        if x0.len() != noise.len() {
            return Err(GenError::DimensionMismatch {
                expected: x0.len(),
                got: noise.len(),
            });
        }
        let alpha_bar = self.alphas_cumprod[t];
        let sqrt_alpha_bar = alpha_bar.sqrt();
        let sqrt_one_minus = (1.0 - alpha_bar).sqrt();

        let x_t = x0
            .iter()
            .zip(noise.iter())
            .map(|(&x, &n)| sqrt_alpha_bar * x + sqrt_one_minus * n)
            .collect();
        Ok(x_t)
    }

    /// Compute the denoising loss between predicted and target noise.
    ///
    /// Both `noise_pred` and `noise_target` are flat tensors of length
    /// `batch_size × d`.  The loss is averaged element-wise over the full tensor.
    ///
    /// # Errors
    ///
    /// - [`GenError::DimensionMismatch`] if `noise_pred` or `noise_target` length
    ///   does not equal `batch_size * d`.
    pub fn compute(
        &self,
        noise_pred: &[f64],
        noise_target: &[f64],
        batch_size: usize,
        d: usize,
    ) -> GenResult<f64> {
        let expected = batch_size * d;
        if noise_pred.len() != expected {
            return Err(GenError::DimensionMismatch {
                expected,
                got: noise_pred.len(),
            });
        }
        if noise_target.len() != expected {
            return Err(GenError::DimensionMismatch {
                expected,
                got: noise_target.len(),
            });
        }
        if expected == 0 {
            return Ok(0.0);
        }
        let loss = noise_pred
            .iter()
            .zip(noise_target.iter())
            .map(|(&p, &t)| {
                let diff = p - t;
                match self.config.loss_type {
                    DdpmLossType::L2 => diff * diff,
                    DdpmLossType::L1 => diff.abs(),
                    DdpmLossType::Huber => {
                        let a = diff.abs();
                        if a <= 1.0 { 0.5 * diff * diff } else { a - 0.5 }
                    }
                }
            })
            .sum::<f64>()
            / (expected as f64);
        Ok(loss)
    }

    /// Compute the min-SNR weighting for timestep `t`.
    ///
    /// `SNR_t = ᾱ_t / (1 - ᾱ_t)`
    ///
    /// The Min-SNR weight (Hang et al., 2023) is `min(SNR_t, 5) / SNR_t`,
    /// which prevents high-SNR timesteps from dominating the loss.
    ///
    /// # Errors
    ///
    /// - [`GenError::InvalidTimestep`] if `t >= n_timesteps`.
    pub fn snr_weight(&self, t: usize) -> GenResult<f64> {
        if t >= self.config.n_timesteps {
            return Err(GenError::InvalidTimestep {
                t,
                max_t: self.config.n_timesteps - 1,
            });
        }
        let alpha_bar = self.alphas_cumprod[t];
        // Guard against alpha_bar reaching exactly 1.0 (degenerate schedule)
        let one_minus = (1.0 - alpha_bar).max(1e-12);
        let snr = alpha_bar / one_minus;
        Ok(snr.min(5.0) / snr.max(1e-12))
    }

    /// Return the number of diffusion timesteps `T`.
    #[must_use]
    #[inline]
    pub fn n_timesteps(&self) -> usize {
        self.config.n_timesteps
    }

    /// Return the precomputed cumulative-alpha array `ᾱ_0 … ᾱ_{T-1}`.
    #[must_use]
    #[inline]
    pub fn alphas_cumprod(&self) -> &[f64] {
        &self.alphas_cumprod
    }

    /// Return the precomputed beta array `β_0 … β_{T-1}`.
    #[must_use]
    #[inline]
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_loss() -> DdpmLoss {
        DdpmLoss::new(DdpmLossConfig::default_ddpm()).expect("DDPM loss with default canonical config (T=1000, beta_start=1e-4, beta_end=0.02, L2) should construct")
    }

    fn make_rng() -> GenRng {
        GenRng::new(42)
    }

    fn randn_f64(rng: &mut GenRng, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let (a, _) = rng.next_normal_pair();
                a as f64
            })
            .collect()
    }

    #[test]
    fn q_sample_shape() {
        let loss = make_loss();
        let mut rng = make_rng();
        let x0 = randn_f64(&mut rng, 64);
        let noise = randn_f64(&mut rng, 64);
        let x_t = loss.q_sample(&x0, 100, &noise).expect(
            "q_sample at valid timestep t=100 with matching x0 and noise lengths should succeed",
        );
        assert_eq!(x_t.len(), 64);
        assert!(x_t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn q_sample_t0_approx_x0() {
        // At t=0 the schedule barely moves: ᾱ_0 ≈ 1 - β_0 ≈ 0.9999
        // so x_t should be very close to x0
        let loss = make_loss();
        let mut rng = make_rng();
        let x0 = randn_f64(&mut rng, 32);
        let noise = randn_f64(&mut rng, 32);
        let x_t = loss.q_sample(&x0, 0, &noise).expect(
            "q_sample at t=0 (start of schedule) with matching x0 and noise should succeed",
        );
        let max_diff = x0
            .iter()
            .zip(&x_t)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        // At t=0, sqrt(alpha_bar_0) ≈ 0.9999 and sqrt(1-alpha_bar_0) ≈ 0.01
        // so the difference is dominated by the noise term which is O(0.01)
        assert!(
            max_diff < 0.5,
            "at t=0 x_t should be close to x0: max_diff={max_diff}"
        );
    }

    #[test]
    fn q_sample_t_max_approx_noise() {
        // At t=T-1 (t=999), ᾱ_{T-1} is very small, so x_t ≈ noise
        let loss = make_loss();
        let mut rng = make_rng();
        let x0: Vec<f64> = vec![1.0; 32]; // constant x0
        let noise = randn_f64(&mut rng, 32);
        let x_t = loss
            .q_sample(&x0, 999, &noise)
            .expect("q_sample should succeed");
        let alpha_bar = loss.alphas_cumprod()[999];
        // Verify sqrt(alpha_bar) is tiny so x_t contribution from x0 is small
        assert!(
            alpha_bar < 0.01,
            "alpha_bar at t=999 should be < 0.01, got {alpha_bar}"
        );
        // The scaled x0 contribution should be negligible
        let x0_contribution = alpha_bar.sqrt() * 1.0; // max |x0| = 1.0
        assert!(
            x0_contribution < 0.15,
            "x0 contribution at t=999 too large: {x0_contribution}"
        );
        assert!(x_t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn loss_nonneg() {
        let loss = make_loss();
        let mut rng = make_rng();
        let pred = randn_f64(&mut rng, 128);
        let target = randn_f64(&mut rng, 128);
        let l = loss
            .compute(&pred, &target, 2, 64)
            .expect("compute should succeed");
        assert!(l >= 0.0, "loss must be non-negative: {l}");
    }

    #[test]
    fn l2_vs_l1_different() {
        // Create two loss functions differing only in type
        let cfg_l2 = DdpmLossConfig {
            n_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 0.02,
            loss_type: DdpmLossType::L2,
        };
        let cfg_l1 = DdpmLossConfig {
            loss_type: DdpmLossType::L1,
            ..cfg_l2.clone()
        };
        let loss_l2 = DdpmLoss::new(cfg_l2).expect("new should succeed");
        let loss_l1 = DdpmLoss::new(cfg_l1).expect("new should succeed");
        // With large errors L2 >> L1, with small errors L2 < L1 — just verify they differ
        let pred = vec![2.0_f64; 8];
        let target = vec![0.0_f64; 8];
        let l2 = loss_l2
            .compute(&pred, &target, 1, 8)
            .expect("compute should succeed");
        let l1 = loss_l1
            .compute(&pred, &target, 1, 8)
            .expect("compute should succeed");
        assert!(
            (l2 - l1).abs() > 1e-6,
            "L2 and L1 should differ on error=2.0: l2={l2}, l1={l1}"
        );
    }

    #[test]
    fn snr_weight_in_range() {
        let loss = make_loss();
        for t in [0_usize, 100, 500, 999] {
            let w = loss.snr_weight(t).expect("snr_weight should succeed");
            assert!(
                (0.0..=1.0).contains(&w),
                "SNR weight at t={t} out of [0,1]: {w}"
            );
        }
    }

    #[test]
    fn snr_weight_finite() {
        let loss = make_loss();
        for t in 0..loss.n_timesteps() {
            let w = loss.snr_weight(t).expect("snr_weight should succeed");
            assert!(w.is_finite(), "SNR weight at t={t} is not finite: {w}");
        }
    }

    #[test]
    fn compute_same_pred_target_zero() {
        let loss = make_loss();
        let v = vec![1.5_f64; 64];
        let l = loss.compute(&v, &v, 2, 32).expect("compute should succeed");
        assert!(l < 1e-10, "loss on identical pred/target should be ~0: {l}");
    }

    #[test]
    fn batch_shape_mismatch_error() {
        let loss = make_loss();
        let pred = vec![0.0_f64; 10];
        let target = vec![0.0_f64; 8]; // wrong length
        let err = loss.compute(&pred, &target, 2, 5);
        assert!(
            matches!(err, Err(GenError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got: {err:?}"
        );
    }

    #[test]
    fn n_timesteps_0_error() {
        let cfg = DdpmLossConfig {
            n_timesteps: 0,
            beta_start: 1e-4,
            beta_end: 0.02,
            loss_type: DdpmLossType::L2,
        };
        let err = DdpmLoss::new(cfg);
        assert!(
            matches!(err, Err(GenError::EmptyInput(_))),
            "expected EmptyInput, got: {err:?}"
        );
    }

    #[test]
    fn invalid_timestep_error() {
        let loss = make_loss();
        let x0 = vec![0.0_f64; 4];
        let noise = vec![0.0_f64; 4];
        let err = loss.q_sample(&x0, 1000, &noise);
        assert!(
            matches!(
                err,
                Err(GenError::InvalidTimestep {
                    t: 1000,
                    max_t: 999
                })
            ),
            "expected InvalidTimestep{{t:1000, max_t:999}}, got: {err:?}"
        );
    }

    #[test]
    fn huber_loss_value() {
        // For diff=0.5: L = 0.5 * 0.25 = 0.125
        // For diff=2.0: L = 2.0 - 0.5 = 1.5
        let cfg = DdpmLossConfig {
            n_timesteps: 100,
            beta_start: 1e-4,
            beta_end: 0.02,
            loss_type: DdpmLossType::Huber,
        };
        let loss = DdpmLoss::new(cfg).expect("new should succeed");
        let pred_small = vec![0.5_f64];
        let target_small = vec![0.0_f64];
        let l_small = loss
            .compute(&pred_small, &target_small, 1, 1)
            .expect("compute should succeed");
        assert!(
            (l_small - 0.125).abs() < 1e-10,
            "Huber for diff=0.5 should be 0.125: {l_small}"
        );

        let pred_large = vec![2.0_f64];
        let target_large = vec![0.0_f64];
        let l_large = loss
            .compute(&pred_large, &target_large, 1, 1)
            .expect("compute should succeed");
        assert!(
            (l_large - 1.5).abs() < 1e-10,
            "Huber for diff=2.0 should be 1.5: {l_large}"
        );
    }

    #[test]
    fn betas_monotone_increasing() {
        let loss = make_loss();
        let betas = loss.betas();
        for i in 1..betas.len() {
            assert!(
                betas[i] >= betas[i - 1],
                "betas should be monotone increasing at i={i}"
            );
        }
    }

    #[test]
    fn alphas_cumprod_monotone_decreasing() {
        let loss = make_loss();
        let ac = loss.alphas_cumprod();
        for i in 1..ac.len() {
            assert!(
                ac[i] <= ac[i - 1],
                "alphas_cumprod should be monotone decreasing at i={i}"
            );
        }
        assert!(
            ac[0] > 0.99,
            "alphas_cumprod[0] should be close to 1: {}",
            ac[0]
        );
        assert!(
            ac[ac.len() - 1] > 0.0,
            "alphas_cumprod[-1] should be > 0: {}",
            ac[ac.len() - 1]
        );
    }
}
