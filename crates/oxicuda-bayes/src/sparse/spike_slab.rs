//! Spike-and-slab variable selection for sparse Bayesian linear regression.
//!
//! Implements the point-mass spike-and-slab prior (Mitchell & Beauchamp 1988;
//! George & McCulloch 1997) with a fully conjugate Gibbs sampler.  Each
//! regression coefficient `β_j` is governed by a binary inclusion indicator
//! `γ_j ∈ {0, 1}`:
//!
//! ```text
//! γ_j ~ Bernoulli(π),
//! β_j | γ_j = 1 ~ N(0, τ² σ²)   (the slab),
//! β_j | γ_j = 0  = 0            (the spike, a point mass at zero),
//! y = Xβ + ε,    ε ~ N(0, σ² I).
//! ```
//!
//! The prior inclusion probability `π` follows a Beta(`a₀`, `b₀`) hyperprior so
//! that the model adapts its own sparsity level, and the noise variance `σ²`
//! follows an Inverse-Gamma hyperprior.  Marginalising `β_j` analytically when
//! sampling each `γ_j` (a *collapsed* Gibbs move) mixes far better than the
//! naive blocked sampler.
//!
//! # Full conditionals (collapsed over the active `β`)
//!
//! For coordinate `j`, let `r₋ⱼ = y − Σ_{k≠j} xₖ βₖ` be the partial residual,
//! `c = xⱼᵀxⱼ`, and `d = xⱼᵀ r₋ⱼ`.  With slab prior variance `v = τ² σ²` the
//! inclusion log-odds is
//!
//! ```text
//! log O_j = log(π/(1−π)) + ½ log( σ² / (σ² + v·c) ) + d² v / (2 σ² (σ² + v·c)),
//! ```
//!
//! and, conditional on inclusion, `β_j ~ N(μ_j, s_j²)` with
//! `s_j² = 1 / (c/σ² + 1/v)` and `μ_j = s_j² · d / σ²`.  The hyperparameters are
//! refreshed each sweep:
//!
//! ```text
//! π  | γ ~ Beta(a₀ + |γ|, b₀ + p − |γ|),
//! σ² | · ~ InvGamma( (n + |γ|)/2 + α₀,  (‖y − Xβ‖² + βᵀβ/τ²)/2 + β₀ ).
//! ```
//!
//! All arithmetic is `f32` and pure-Rust; randomness comes from [`LcgRng`].

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the [`SpikeSlabRegression`] Gibbs sampler.
#[derive(Debug, Clone)]
pub struct SpikeSlabConfig {
    /// Number of recorded posterior draws (≥ 1).
    pub n_samples: usize,
    /// Burn-in iterations discarded before recording.
    pub n_burnin: usize,
    /// Slab variance scale `τ²` (the included coefficients have prior variance
    /// `τ² σ²`). Must be strictly positive.
    pub slab_scale: f32,
    /// Beta(`a₀`, `b₀`) hyperprior on the inclusion probability `π`: shape `a₀`.
    pub pi_a: f32,
    /// Beta hyperprior shape `b₀`.
    pub pi_b: f32,
    /// Inverse-Gamma hyperprior shape `α₀` for the noise variance `σ²`.
    pub sigma_shape: f32,
    /// Inverse-Gamma hyperprior scale `β₀` for the noise variance `σ²`.
    pub sigma_scale: f32,
}

impl Default for SpikeSlabConfig {
    fn default() -> Self {
        Self {
            n_samples: 2_000,
            n_burnin: 1_000,
            slab_scale: 10.0,
            pi_a: 1.0,
            pi_b: 1.0,
            sigma_shape: 1e-3,
            sigma_scale: 1e-3,
        }
    }
}

impl SpikeSlabConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `n_samples == 0`, or any scale /
    /// shape parameter is non-positive or non-finite.
    pub fn validate(&self) -> BayesResult<()> {
        if self.n_samples == 0 {
            return Err(BayesError::InvalidConfig(
                "spike-slab n_samples must be >= 1".into(),
            ));
        }
        for (name, v) in [
            ("slab_scale", self.slab_scale),
            ("pi_a", self.pi_a),
            ("pi_b", self.pi_b),
            ("sigma_shape", self.sigma_shape),
            ("sigma_scale", self.sigma_scale),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(BayesError::InvalidConfig(format!(
                    "spike-slab {name} must be positive and finite, got {v}"
                )));
            }
        }
        Ok(())
    }
}

// ─── Fit result ─────────────────────────────────────────────────────────────

/// Posterior summary of a spike-and-slab fit.
#[derive(Debug, Clone)]
pub struct SpikeSlabFit {
    /// Posterior mean of each coefficient `E[β_j]` (already shrunk by the
    /// inclusion probability since excluded draws contribute exactly zero).
    pub beta_mean: Vec<f32>,
    /// Posterior **inclusion probability** `P(γ_j = 1 | y)` per coefficient —
    /// the marginal posterior probability that predictor `j` is in the model.
    pub inclusion_prob: Vec<f32>,
    /// Posterior mean of the noise variance `σ²`.
    pub sigma2_mean: f32,
    /// Posterior mean of the global inclusion probability `π`.
    pub pi_mean: f32,
    /// Number of predictors `p`.
    pub n_features: usize,
}

impl SpikeSlabFit {
    /// Indices of predictors whose posterior inclusion probability exceeds
    /// `threshold` (the *median-probability model* uses `0.5`).
    #[must_use]
    pub fn selected(&self, threshold: f32) -> Vec<usize> {
        self.inclusion_prob
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

// ─── Sampler ────────────────────────────────────────────────────────────────

/// Spike-and-slab linear regression via collapsed Gibbs sampling.
#[derive(Debug, Clone)]
pub struct SpikeSlabRegression;

impl SpikeSlabRegression {
    /// Fit `y = Xβ + ε` with a point-mass spike-and-slab prior.
    ///
    /// `x` is the row-major `n × p` design matrix and `y` has length `n`.
    /// Returns posterior inclusion probabilities and shrunken coefficient
    /// means.  The chain is deterministic given the seed of `rng`.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] if `x` or `y` is empty.
    /// - [`BayesError::DimensionMismatch`] if `x.len() != n * p` (with
    ///   `n = y.len()`), inferring `p` from `x.len() / n`.
    /// - Propagates [`SpikeSlabConfig::validate`].
    pub fn fit(
        x: &[f32],
        y: &[f32],
        config: &SpikeSlabConfig,
        rng: &mut LcgRng,
    ) -> BayesResult<SpikeSlabFit> {
        config.validate()?;
        if x.is_empty() || y.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let n = y.len();
        if x.len() % n != 0 {
            return Err(BayesError::DimensionMismatch {
                expected: n * (x.len() / n.max(1)),
                got: x.len(),
            });
        }
        let p = x.len() / n;
        if p == 0 {
            return Err(BayesError::EmptyInputs);
        }

        // Pre-compute per-column squared norms c_j = xⱼᵀ xⱼ (column-major access
        // into the row-major matrix).
        let mut col_sq = vec![0.0_f32; p];
        for (i, _) in (0..n).enumerate() {
            for (j, c) in col_sq.iter_mut().enumerate() {
                let v = x[i * p + j];
                *c += v * v;
            }
        }

        // State: residual r = y − Xβ, coefficients β, indicators γ, σ², π.
        let mut beta = vec![0.0_f32; p];
        let mut gamma = vec![false; p];
        let mut residual = y.to_vec();
        // Initialise σ² from the variance of y, π from the Beta prior mean.
        let y_mean = y.iter().sum::<f32>() / n as f32;
        let y_var = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f32>() / n.max(1) as f32;
        let mut sigma2 = y_var.max(1e-6);
        let mut pi = config.pi_a / (config.pi_a + config.pi_b);

        let total = config.n_burnin + config.n_samples;
        let mut beta_sum = vec![0.0_f64; p];
        let mut incl_count = vec![0.0_f64; p];
        let mut sigma2_sum = 0.0_f64;
        let mut pi_sum = 0.0_f64;

        for iter in 0..total {
            let tau2 = config.slab_scale;
            // ── Sample each (γ_j, β_j) jointly, updating the residual. ──
            for j in 0..p {
                let c = col_sq[j];
                if c <= 0.0 {
                    // Degenerate (all-zero) column: never include.
                    if gamma[j] {
                        // Remove its (already zero) contribution; keep clean.
                        gamma[j] = false;
                        beta[j] = 0.0;
                    }
                    continue;
                }
                // Add the current β_j contribution back into the residual so it
                // becomes the partial residual r₋ⱼ = y − Σ_{k≠j} xₖ βₖ.
                if beta[j] != 0.0 {
                    for i in 0..n {
                        residual[i] += x[i * p + j] * beta[j];
                    }
                }
                // d = xⱼᵀ r₋ⱼ.
                let mut d = 0.0_f32;
                for i in 0..n {
                    d += x[i * p + j] * residual[i];
                }
                let v = tau2 * sigma2; // slab prior variance.
                // Posterior (given inclusion): s² = 1/(c/σ² + 1/v), μ = s²·d/σ².
                let post_prec = c / sigma2 + 1.0 / v;
                let s2 = 1.0 / post_prec;
                let mu = s2 * d / sigma2;
                // Inclusion log-odds (Bayes factor for the slab vs the spike).
                //   log BF = ½ log(s²/v) + μ²/(2 s²).
                let log_bf = 0.5 * (s2 / v).ln() + 0.5 * mu * mu / s2;
                let pi_clamped = pi.clamp(1e-6, 1.0 - 1e-6);
                let log_odds = (pi_clamped / (1.0 - pi_clamped)).ln() + log_bf;
                // P(γ_j = 1) = sigmoid(log_odds).
                let p_in = 1.0 / (1.0 + (-log_odds).exp());
                let include = next_unit(rng) < p_in;
                gamma[j] = include;
                if include {
                    // Draw β_j ~ N(μ, s²) and subtract its new contribution.
                    let z = standard_normal(rng);
                    beta[j] = mu + s2.sqrt() * z;
                    for i in 0..n {
                        residual[i] -= x[i * p + j] * beta[j];
                    }
                } else {
                    beta[j] = 0.0;
                }
            }

            // ── Sample π | γ ~ Beta(a₀ + |γ|, b₀ + p − |γ|). ──
            let k_active = gamma.iter().filter(|&&g| g).count() as f32;
            pi = sample_beta(
                config.pi_a + k_active,
                config.pi_b + (p as f32 - k_active),
                rng,
            );

            // ── Sample σ² | · ~ InvGamma(shape, scale). ──
            // shape = α₀ + (n + |γ|)/2 ; scale = β₀ + (‖r‖² + Σ β_j²/τ²)/2.
            let rss: f32 = residual.iter().map(|&e| e * e).sum();
            let beta_quad: f32 = beta.iter().map(|&b| b * b).sum::<f32>() / tau2;
            let ig_shape = config.sigma_shape + 0.5 * (n as f32 + k_active);
            let ig_scale = config.sigma_scale + 0.5 * (rss + beta_quad);
            sigma2 = sample_inv_gamma(ig_shape, ig_scale, rng).max(1e-10);

            // ── Accumulate post-burn-in. ──
            if iter >= config.n_burnin {
                for j in 0..p {
                    beta_sum[j] += beta[j] as f64;
                    if gamma[j] {
                        incl_count[j] += 1.0;
                    }
                }
                sigma2_sum += sigma2 as f64;
                pi_sum += pi as f64;
            }
        }

        let inv = 1.0 / config.n_samples as f64;
        let beta_mean = beta_sum.iter().map(|&s| (s * inv) as f32).collect();
        let inclusion_prob = incl_count.iter().map(|&s| (s * inv) as f32).collect();

        Ok(SpikeSlabFit {
            beta_mean,
            inclusion_prob,
            sigma2_mean: (sigma2_sum * inv) as f32,
            pi_mean: (pi_sum * inv) as f32,
            n_features: p,
        })
    }
}

// ─── Random-number helpers (full ÷2³² uniforms) ─────────────────────────────

/// Unit-uniform draw in `[0, 1)` from [`LcgRng::next_f32`] (÷2³² full range).
#[inline]
fn next_unit(rng: &mut LcgRng) -> f32 {
    rng.next_f32()
}

/// One standard-normal draw via the crate's Box-Muller pair.
#[inline]
fn standard_normal(rng: &mut LcgRng) -> f32 {
    rng.next_normal_pair().0
}

/// Sample `Gamma(shape, rate)` (shape > 0, rate > 0) via Marsaglia & Tsang
/// (2000) for `shape ≥ 1` and the boost trick for `shape < 1`.
fn sample_gamma(shape: f32, rate: f32, rng: &mut LcgRng) -> f32 {
    if shape < 1.0 {
        // Boost: G(a) = G(a+1) · U^(1/a).
        let g = sample_gamma(shape + 1.0, rate, rng);
        let u = next_unit(rng).max(1e-30);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = standard_normal(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = next_unit(rng).max(1e-30);
        // Squeeze + acceptance test.
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v / rate;
        }
    }
}

/// Sample `InvGamma(shape, scale)` as `1 / Gamma(shape, rate = scale)`.
fn sample_inv_gamma(shape: f32, scale: f32, rng: &mut LcgRng) -> f32 {
    let g = sample_gamma(shape, scale, rng);
    if g <= 0.0 { f32::INFINITY } else { 1.0 / g }
}

/// Sample `Beta(a, b)` via two independent Gamma draws: `X/(X+Y)` with
/// `X ~ Gamma(a, 1)`, `Y ~ Gamma(b, 1)`.
fn sample_beta(a: f32, b: f32, rng: &mut LcgRng) -> f32 {
    let x = sample_gamma(a, 1.0, rng);
    let y = sample_gamma(b, 1.0, rng);
    let s = x + y;
    if s <= 0.0 {
        0.5
    } else {
        (x / s).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sparse regression dataset: only a few of `p` columns are active.
    ///
    /// Columns are iid N(0,1); `y = Xβ_true + noise`. Returns (x, y, beta_true).
    fn make_sparse_data(
        n: usize,
        p: usize,
        active: &[(usize, f32)],
        noise_sd: f32,
        rng: &mut LcgRng,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut beta_true = vec![0.0_f32; p];
        for &(j, b) in active {
            beta_true[j] = b;
        }
        let mut x = vec![0.0_f32; n * p];
        for v in x.iter_mut() {
            *v = standard_normal(rng);
        }
        let mut y = vec![0.0_f32; n];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut s = 0.0;
            for j in 0..p {
                s += x[i * p + j] * beta_true[j];
            }
            *yi = s + noise_sd * standard_normal(rng);
        }
        (x, y, beta_true)
    }

    #[test]
    fn config_validation() {
        let mut c = SpikeSlabConfig::default();
        assert!(c.validate().is_ok());
        c.slab_scale = 0.0;
        assert!(c.validate().is_err());
        let c2 = SpikeSlabConfig {
            n_samples: 0,
            ..Default::default()
        };
        assert!(c2.validate().is_err());
    }

    #[test]
    fn rejects_empty_inputs() {
        let cfg = SpikeSlabConfig::default();
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            SpikeSlabRegression::fit(&[], &[1.0], &cfg, &mut rng),
            Err(BayesError::EmptyInputs)
        ));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        // x has 7 entries, y has length 2 → 7 % 2 != 0.
        let cfg = SpikeSlabConfig::default();
        let mut rng = LcgRng::new(1);
        let x = vec![0.0_f32; 7];
        let y = vec![0.0_f32; 2];
        assert!(matches!(
            SpikeSlabRegression::fit(&x, &y, &cfg, &mut rng),
            Err(BayesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn recovers_active_set() {
        // 8 predictors, only columns 1 and 5 are truly active.
        let mut data_rng = LcgRng::new(2024);
        let active = [(1usize, 3.0_f32), (5usize, -2.5_f32)];
        let (x, y, _beta_true) = make_sparse_data(160, 8, &active, 0.4, &mut data_rng);

        let cfg = SpikeSlabConfig {
            n_samples: 3_000,
            n_burnin: 1_500,
            slab_scale: 10.0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(7);
        let fit = SpikeSlabRegression::fit(&x, &y, &cfg, &mut rng).unwrap();

        // The two active predictors must have high inclusion probability...
        assert!(
            fit.inclusion_prob[1] > 0.85,
            "col1 incl {}",
            fit.inclusion_prob[1]
        );
        assert!(
            fit.inclusion_prob[5] > 0.85,
            "col5 incl {}",
            fit.inclusion_prob[5]
        );
        // ...and the inactive ones low inclusion probability.
        for j in [0usize, 2, 3, 4, 6, 7] {
            assert!(
                fit.inclusion_prob[j] < 0.5,
                "inactive col{j} incl {}",
                fit.inclusion_prob[j]
            );
        }
        // The median-probability model is exactly {1, 5}.
        let mut selected = fit.selected(0.5);
        selected.sort_unstable();
        assert_eq!(selected, vec![1, 5]);
    }

    #[test]
    fn recovers_coefficient_magnitudes() {
        let mut data_rng = LcgRng::new(909);
        let active = [(2usize, 4.0_f32)];
        let (x, y, beta_true) = make_sparse_data(200, 6, &active, 0.3, &mut data_rng);

        let cfg = SpikeSlabConfig {
            n_samples: 3_000,
            n_burnin: 1_500,
            slab_scale: 20.0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(31);
        let fit = SpikeSlabRegression::fit(&x, &y, &cfg, &mut rng).unwrap();

        // The active coefficient's posterior mean should be near the truth.
        assert!(
            (fit.beta_mean[2] - beta_true[2]).abs() < 0.5,
            "beta2 {} vs {}",
            fit.beta_mean[2],
            beta_true[2]
        );
        // The recovered noise variance should be near 0.3² = 0.09.
        assert!(
            fit.sigma2_mean > 0.02 && fit.sigma2_mean < 0.3,
            "sigma2 {}",
            fit.sigma2_mean
        );
        // Inactive coefficients shrink essentially to zero.
        for j in [0usize, 1, 3, 4, 5] {
            assert!(fit.beta_mean[j].abs() < 0.4, "beta{j} {}", fit.beta_mean[j]);
        }
    }

    #[test]
    fn null_model_keeps_inclusion_low() {
        // No predictor is active: every inclusion probability should stay low
        // and the noise variance should match the variance of pure noise.
        let mut data_rng = LcgRng::new(77);
        let (x, y, _) = make_sparse_data(150, 5, &[], 1.0, &mut data_rng);
        let cfg = SpikeSlabConfig {
            n_samples: 2_000,
            n_burnin: 1_000,
            ..Default::default()
        };
        let mut rng = LcgRng::new(88);
        let fit = SpikeSlabRegression::fit(&x, &y, &cfg, &mut rng).unwrap();
        assert!(
            fit.inclusion_prob.iter().all(|&p| p < 0.5),
            "spurious inclusion: {:?}",
            fit.inclusion_prob
        );
        assert!(fit.selected(0.5).is_empty());
        // σ² ≈ 1.0 (the noise variance).
        assert!(
            (fit.sigma2_mean - 1.0).abs() < 0.4,
            "sigma2 {}",
            fit.sigma2_mean
        );
    }

    #[test]
    fn gamma_sampler_matches_moments() {
        // Gamma(shape=4, rate=2): mean = 2, variance = 1.
        let mut rng = LcgRng::new(123);
        let n = 60_000;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for _ in 0..n {
            let g = sample_gamma(4.0, 2.0, &mut rng) as f64;
            assert!(g > 0.0 && g.is_finite());
            sum += g;
            sum_sq += g * g;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((mean - 2.0).abs() < 0.05, "gamma mean {mean}");
        assert!((var - 1.0).abs() < 0.08, "gamma var {var}");
    }

    #[test]
    fn beta_sampler_matches_mean() {
        // Beta(2, 8): mean = 0.2.
        let mut rng = LcgRng::new(55);
        let n = 60_000;
        let mut sum = 0.0_f64;
        for _ in 0..n {
            let b = sample_beta(2.0, 8.0, &mut rng) as f64;
            assert!((0.0..=1.0).contains(&b));
            sum += b;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.2).abs() < 0.01, "beta mean {mean}");
    }

    #[test]
    fn inv_gamma_sampler_matches_mean() {
        // InvGamma(shape=5, scale=4): mean = scale/(shape-1) = 1.0.
        let mut rng = LcgRng::new(909);
        let n = 60_000;
        let mut sum = 0.0_f64;
        for _ in 0..n {
            let v = sample_inv_gamma(5.0, 4.0, &mut rng) as f64;
            assert!(v > 0.0 && v.is_finite());
            sum += v;
        }
        let mean = sum / n as f64;
        assert!((mean - 1.0).abs() < 0.05, "inv-gamma mean {mean}");
    }
}
