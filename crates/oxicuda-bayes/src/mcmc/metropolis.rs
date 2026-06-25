//! Random-walk Metropolis-Hastings and univariate slice sampling.
//!
//! These are the workhorse gradient-free MCMC samplers, complementing the
//! gradient-based [`crate::mcmc::hmc`] and [`crate::mcmc::sgld`] samplers.  All
//! routines operate in full precision (`f64`) and target an *unnormalised*
//! log-density supplied as a closure, so the user never has to compute the
//! intractable normalising constant.
//!
//! | Sampler | Reference | Notes |
//! |---------|-----------|-------|
//! | Random-walk Metropolis | Metropolis et al. 1953; Hastings 1970 | isotropic Gaussian proposal |
//! | Adaptive Metropolis | Haario, Saksman & Tamminen 2001 | per-coordinate step adaptation toward a target acceptance rate |
//! | Slice sampling | Neal 2003 | univariate stepping-out + shrinkage, no tuning |
//!
//! # Conventions
//!
//! The target is a closure `log_density: Fn(&[f64]) -> f64` returning the
//! log of the unnormalised density `log p̃(θ)`.  A return value of
//! `f64::NEG_INFINITY` is interpreted as zero density (a hard constraint /
//! out-of-support point) and is always rejected — this lets callers encode box
//! constraints without a separate prior.  Any other non-finite value (`NaN` or
//! `+∞`) is reported as [`BayesError::NanEncountered`].

use crate::error::{BayesError, BayesResult};
use crate::mcmc::BayesRng;

// ─── Random-walk / Adaptive Metropolis ──────────────────────────────────────

/// Configuration for [`MetropolisSampler`].
#[derive(Debug, Clone)]
pub struct MetropolisConfig {
    /// Number of recorded post-warmup samples (≥ 1).
    pub n_samples: usize,
    /// Number of warmup (burn-in) iterations discarded before recording.
    ///
    /// When [`MetropolisConfig::adapt`] is `true` the proposal scale is tuned
    /// during this phase and then frozen for the recording phase, preserving a
    /// valid stationary distribution.
    pub n_warmup: usize,
    /// Thinning factor: keep one sample every `thin` accepted/rejected steps
    /// (≥ 1; `1` keeps every step).
    pub thin: usize,
    /// Initial isotropic proposal standard deviation `σ_prop > 0`.
    pub proposal_scale: f64,
    /// Adapt the per-coordinate proposal scale during warmup toward
    /// [`MetropolisConfig::target_accept`] (Robbins-Monro on the log-scale).
    pub adapt: bool,
    /// Target Metropolis acceptance rate used when `adapt` is `true`.
    ///
    /// The canonical optimum is ≈ 0.234 for high-dimensional targets and ≈ 0.44
    /// for one dimension (Roberts, Gelman & Gilks 1997).
    pub target_accept: f64,
}

impl Default for MetropolisConfig {
    fn default() -> Self {
        Self {
            n_samples: 1_000,
            n_warmup: 1_000,
            thin: 1,
            proposal_scale: 1.0,
            adapt: true,
            target_accept: 0.234,
        }
    }
}

impl MetropolisConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `n_samples == 0`, `thin == 0`,
    /// `proposal_scale` is not strictly positive and finite, or `target_accept`
    /// is outside `(0, 1)` while `adapt` is enabled.
    pub fn validate(&self) -> BayesResult<()> {
        if self.n_samples == 0 {
            return Err(BayesError::InvalidConfig(
                "Metropolis n_samples must be >= 1".to_string(),
            ));
        }
        if self.thin == 0 {
            return Err(BayesError::InvalidConfig(
                "Metropolis thin must be >= 1".to_string(),
            ));
        }
        if !self.proposal_scale.is_finite() || self.proposal_scale <= 0.0 {
            return Err(BayesError::InvalidConfig(format!(
                "Metropolis proposal_scale must be positive and finite, got {}",
                self.proposal_scale
            )));
        }
        if self.adapt && !(0.0 < self.target_accept && self.target_accept < 1.0) {
            return Err(BayesError::InvalidConfig(format!(
                "Metropolis target_accept must be in (0, 1), got {}",
                self.target_accept
            )));
        }
        Ok(())
    }
}

/// Result of a Metropolis run.
#[derive(Debug, Clone)]
pub struct MetropolisResult {
    /// Recorded samples, each of dimension `theta_init.len()`. Length
    /// `n_samples`.
    pub samples: Vec<Vec<f64>>,
    /// Fraction of *recording-phase* proposals that were accepted.
    pub accept_rate: f64,
    /// Final per-coordinate proposal standard deviations (post-adaptation).
    pub proposal_scale: Vec<f64>,
}

/// Random-walk Metropolis-Hastings sampler with optional adaptive scaling.
#[derive(Debug, Clone)]
pub struct MetropolisSampler {
    config: MetropolisConfig,
}

impl MetropolisSampler {
    /// Create a new sampler.
    ///
    /// # Errors
    /// Propagates [`MetropolisConfig::validate`].
    pub fn new(config: MetropolisConfig) -> BayesResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Run the chain.
    ///
    /// `log_density(θ)` returns `log p̃(θ)` (unnormalised). Returning
    /// `f64::NEG_INFINITY` marks `θ` as out of support and forces rejection.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] if `theta_init` is empty.
    /// - [`BayesError::NanEncountered`] if the initial density is not finite, or
    ///   the density returns `NaN` / `+∞` at any proposed point.
    pub fn sample(
        &self,
        theta_init: &[f64],
        log_density: impl Fn(&[f64]) -> f64,
        rng: &mut BayesRng,
    ) -> BayesResult<MetropolisResult> {
        if theta_init.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let dim = theta_init.len();

        let mut current = theta_init.to_vec();
        let mut current_lp = log_density(&current);
        if !current_lp.is_finite() {
            // The initial point must be inside the support and finite.
            return Err(BayesError::NanEncountered {
                location: "metropolis_initial_density",
            });
        }

        // Per-coordinate proposal scale on the natural (positive) axis.
        let mut scale = vec![self.config.proposal_scale; dim];
        // Robbins-Monro adaptation step size, decaying as 1/(1 + t/τ).
        let adapt_tau = (self.config.n_warmup.max(1) as f64) / 5.0;

        let total_iters = self.config.n_warmup + self.config.n_samples * self.config.thin;
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);
        let mut record_accepts = 0usize;
        let mut record_proposals = 0usize;
        let mut proposal = vec![0.0_f64; dim];

        for iter in 0..total_iters {
            let warming = iter < self.config.n_warmup;

            // Isotropic Gaussian random-walk proposal with per-coordinate scale.
            for d in 0..dim {
                proposal[d] = current[d] + scale[d] * rng.next_normal();
            }
            let prop_lp = log_density(&proposal);
            if prop_lp.is_nan() || prop_lp == f64::INFINITY {
                return Err(BayesError::NanEncountered {
                    location: "metropolis_proposal_density",
                });
            }

            // Metropolis acceptance: the symmetric proposal cancels in the ratio.
            // log α = min(0, log p̃(θ') − log p̃(θ)); −∞ density ⇒ reject.
            let log_alpha = prop_lp - current_lp;
            let accept = if prop_lp == f64::NEG_INFINITY {
                false
            } else {
                log_alpha >= 0.0 || rng.next_f64().ln() < log_alpha
            };
            if accept {
                current.copy_from_slice(&proposal);
                current_lp = prop_lp;
            }

            if warming {
                if self.config.adapt {
                    // Robbins-Monro update of the log-scale toward the target
                    // acceptance probability. acc_prob = exp(min(0, log_alpha)).
                    let acc_prob = if prop_lp == f64::NEG_INFINITY {
                        0.0
                    } else {
                        log_alpha.min(0.0).exp()
                    };
                    let gamma = 1.0 / (1.0 + iter as f64 / adapt_tau);
                    let factor = (gamma * (acc_prob - self.config.target_accept)).exp();
                    for s in scale.iter_mut() {
                        *s = (*s * factor).clamp(1e-12, 1e12);
                    }
                }
            } else {
                record_proposals += 1;
                if accept {
                    record_accepts += 1;
                }
                // Record after each `thin`-th recording-phase step.
                if record_proposals % self.config.thin == 0 {
                    samples.push(current.clone());
                }
            }
        }

        let accept_rate = if record_proposals == 0 {
            0.0
        } else {
            record_accepts as f64 / record_proposals as f64
        };

        Ok(MetropolisResult {
            samples,
            accept_rate,
            proposal_scale: scale,
        })
    }
}

// ─── Slice sampling (Neal 2003) ─────────────────────────────────────────────

/// Configuration for [`SliceSampler`].
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Number of recorded samples (≥ 1).
    pub n_samples: usize,
    /// Warmup iterations discarded before recording.
    pub n_warmup: usize,
    /// Initial estimate of the typical slice width `w > 0` (stepping-out unit).
    pub width: f64,
    /// Maximum number of stepping-out expansions `m ≥ 1` in each direction.
    pub max_steps: usize,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            n_samples: 1_000,
            n_warmup: 200,
            width: 1.0,
            max_steps: 50,
        }
    }
}

impl SliceConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `n_samples == 0`, `width` is not
    /// strictly positive and finite, or `max_steps == 0`.
    pub fn validate(&self) -> BayesResult<()> {
        if self.n_samples == 0 {
            return Err(BayesError::InvalidConfig(
                "Slice n_samples must be >= 1".to_string(),
            ));
        }
        if !self.width.is_finite() || self.width <= 0.0 {
            return Err(BayesError::InvalidConfig(format!(
                "Slice width must be positive and finite, got {}",
                self.width
            )));
        }
        if self.max_steps == 0 {
            return Err(BayesError::InvalidConfig(
                "Slice max_steps must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Coordinate-wise slice sampler (Neal 2003).
///
/// On each sweep every coordinate is updated in turn with a univariate slice
/// move: an auxiliary slice level `y = log p̃(θ) − Exp(1)` is drawn, an interval
/// is grown around the current value via stepping-out, and a new value is drawn
/// uniformly from the interval with shrinkage on rejection.  Slice sampling is
/// self-tuning — there is no proposal scale and every proposed move is accepted
/// once it lands inside the slice.
#[derive(Debug, Clone)]
pub struct SliceSampler {
    config: SliceConfig,
}

impl SliceSampler {
    /// Create a new sampler.
    ///
    /// # Errors
    /// Propagates [`SliceConfig::validate`].
    pub fn new(config: SliceConfig) -> BayesResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Update a single coordinate `d` in place via one slice move.
    fn slice_update(
        &self,
        theta: &mut [f64],
        d: usize,
        current_lp: f64,
        log_density: &impl Fn(&[f64]) -> f64,
        rng: &mut BayesRng,
    ) -> BayesResult<f64> {
        // Auxiliary slice level: y = log p̃(x₀) + log u, u ~ U(0,1).
        // Equivalently log_y = current_lp − Exp(1).
        let log_y = current_lp + rng.next_f64().max(1e-300).ln();

        let x0 = theta[d];
        let w = self.config.width;

        // Stepping out: place an interval [l, r] of width w randomly around x0,
        // then expand it (up to max_steps each side) until both ends fall below
        // the slice level.
        let u = rng.next_f64();
        let mut l = x0 - w * u;
        let mut r = l + w;
        let max_l = self.config.max_steps;
        let max_r = self.config.max_steps;

        let mut j = (rng.next_f64() * max_l as f64).floor() as usize;
        let mut k = max_r - 1 - j.min(max_r - 1);

        // Expand the left endpoint.
        loop {
            if j == 0 {
                break;
            }
            theta[d] = l;
            let lp = log_density(theta);
            if lp.is_nan() || lp == f64::INFINITY {
                theta[d] = x0;
                return Err(BayesError::NanEncountered {
                    location: "slice_step_out_left",
                });
            }
            if lp <= log_y {
                break;
            }
            l -= w;
            j -= 1;
        }
        // Expand the right endpoint.
        loop {
            if k == 0 {
                break;
            }
            theta[d] = r;
            let lp = log_density(theta);
            if lp.is_nan() || lp == f64::INFINITY {
                theta[d] = x0;
                return Err(BayesError::NanEncountered {
                    location: "slice_step_out_right",
                });
            }
            if lp <= log_y {
                break;
            }
            r += w;
            k -= 1;
        }

        // Shrinkage: sample uniformly from [l, r]; on rejection shrink the
        // interval toward x0 and retry. Guaranteed to terminate because the
        // interval contracts to x0, whose density equals current_lp ≥ log_y is
        // not guaranteed (it equals it only in expectation), so cap iterations.
        for _ in 0..1024 {
            let x1 = l + rng.next_f64() * (r - l);
            theta[d] = x1;
            let lp = log_density(theta);
            if lp.is_nan() || lp == f64::INFINITY {
                theta[d] = x0;
                return Err(BayesError::NanEncountered {
                    location: "slice_shrink",
                });
            }
            if lp > log_y {
                return Ok(lp);
            }
            // Shrink the side that x1 fell on.
            if x1 < x0 {
                l = x1;
            } else {
                r = x1;
            }
        }
        // Degenerate fallback: keep the original value.
        theta[d] = x0;
        Ok(current_lp)
    }

    /// Run the coordinate-wise slice sampler.
    ///
    /// # Errors
    /// - [`BayesError::EmptyInputs`] if `theta_init` is empty.
    /// - [`BayesError::NanEncountered`] if the density is not finite at the
    ///   initial point or returns `NaN` / `+∞` during sampling.
    pub fn sample(
        &self,
        theta_init: &[f64],
        log_density: impl Fn(&[f64]) -> f64,
        rng: &mut BayesRng,
    ) -> BayesResult<Vec<Vec<f64>>> {
        if theta_init.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let dim = theta_init.len();
        let mut theta = theta_init.to_vec();
        let mut current_lp = log_density(&theta);
        if !current_lp.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "slice_initial_density",
            });
        }

        let total = self.config.n_warmup + self.config.n_samples;
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);
        for iter in 0..total {
            for d in 0..dim {
                current_lp = self.slice_update(&mut theta, d, current_lp, &log_density, rng)?;
            }
            if iter >= self.config.n_warmup {
                samples.push(theta.clone());
            }
        }
        Ok(samples)
    }
}

// ─── Shared helpers ─────────────────────────────────────────────────────────

/// Column-wise mean of a set of equal-length samples.
///
/// # Errors
/// Returns [`BayesError::EmptyInputs`] if `samples` is empty, and
/// [`BayesError::DimensionMismatch`] if the rows have differing lengths.
pub fn sample_mean(samples: &[Vec<f64>]) -> BayesResult<Vec<f64>> {
    if samples.is_empty() {
        return Err(BayesError::EmptyInputs);
    }
    let dim = samples[0].len();
    let mut mean = vec![0.0_f64; dim];
    for row in samples {
        if row.len() != dim {
            return Err(BayesError::DimensionMismatch {
                expected: dim,
                got: row.len(),
            });
        }
        for (m, &x) in mean.iter_mut().zip(row.iter()) {
            *m += x;
        }
    }
    let inv = 1.0 / samples.len() as f64;
    for m in mean.iter_mut() {
        *m *= inv;
    }
    Ok(mean)
}

/// Column-wise (unbiased, Bessel-corrected) variance of a set of samples.
///
/// # Errors
/// Returns [`BayesError::InsufficientSamples`] if fewer than two rows are given,
/// and [`BayesError::DimensionMismatch`] on ragged input.
pub fn sample_variance(samples: &[Vec<f64>]) -> BayesResult<Vec<f64>> {
    if samples.len() < 2 {
        return Err(BayesError::InsufficientSamples {
            min: 2,
            got: samples.len(),
        });
    }
    let mean = sample_mean(samples)?;
    let dim = mean.len();
    let mut var = vec![0.0_f64; dim];
    for row in samples {
        for d in 0..dim {
            let diff = row[d] - mean[d];
            var[d] += diff * diff;
        }
    }
    let inv = 1.0 / (samples.len() as f64 - 1.0);
    for v in var.iter_mut() {
        *v *= inv;
    }
    Ok(var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::BayesRng;

    /// log density of a univariate N(mu, sigma^2), unnormalised constant kept.
    fn log_normal(x: f64, mu: f64, sigma: f64) -> f64 {
        let z = (x - mu) / sigma;
        -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    #[test]
    fn metropolis_config_validation() {
        let mut c = MetropolisConfig::default();
        assert!(c.validate().is_ok());
        c.proposal_scale = -1.0;
        assert!(c.validate().is_err());
        let c2 = MetropolisConfig {
            n_samples: 0,
            ..Default::default()
        };
        assert!(c2.validate().is_err());
        let c3 = MetropolisConfig {
            target_accept: 1.5,
            adapt: true,
            ..Default::default()
        };
        assert!(c3.validate().is_err());
    }

    #[test]
    fn metropolis_rejects_empty_init() {
        let s = MetropolisSampler::new(MetropolisConfig::default()).unwrap();
        let mut rng = BayesRng::new(1);
        let r = s.sample(&[], |_| 0.0, &mut rng);
        assert!(matches!(r, Err(BayesError::EmptyInputs)));
    }

    #[test]
    fn metropolis_rejects_nonfinite_initial_density() {
        let s = MetropolisSampler::new(MetropolisConfig::default()).unwrap();
        let mut rng = BayesRng::new(1);
        // Initial point out of support → −∞ density → error.
        let r = s.sample(&[0.0], |_| f64::NEG_INFINITY, &mut rng);
        assert!(matches!(r, Err(BayesError::NanEncountered { .. })));
    }

    #[test]
    fn metropolis_recovers_gaussian_moments() {
        // Target: N(2.0, 1.5^2). The chain mean / variance must recover it.
        let mu = 2.0;
        let sigma = 1.5;
        let cfg = MetropolisConfig {
            n_samples: 40_000,
            n_warmup: 4_000,
            thin: 1,
            proposal_scale: 2.0,
            adapt: true,
            target_accept: 0.44,
        };
        let s = MetropolisSampler::new(cfg).unwrap();
        let mut rng = BayesRng::new(20240620);
        let res = s
            .sample(&[0.0], move |t| log_normal(t[0], mu, sigma), &mut rng)
            .unwrap();
        assert_eq!(res.samples.len(), 40_000);
        let mean = sample_mean(&res.samples).unwrap()[0];
        let var = sample_variance(&res.samples).unwrap()[0];
        assert!((mean - mu).abs() < 0.1, "mean {mean} != {mu}");
        assert!(
            (var - sigma * sigma).abs() < 0.25,
            "var {var} != {}",
            sigma * sigma
        );
        // Adapted acceptance rate should be in a reasonable band.
        assert!(
            res.accept_rate > 0.2 && res.accept_rate < 0.7,
            "accept {} out of band",
            res.accept_rate
        );
    }

    #[test]
    fn metropolis_respects_hard_constraint() {
        // Truncated standard normal on [0, ∞): support encoded via −∞ density.
        let cfg = MetropolisConfig {
            n_samples: 20_000,
            n_warmup: 2_000,
            proposal_scale: 1.5,
            ..Default::default()
        };
        let s = MetropolisSampler::new(cfg).unwrap();
        let mut rng = BayesRng::new(7);
        let res = s
            .sample(
                &[0.5],
                |t| {
                    if t[0] < 0.0 {
                        f64::NEG_INFINITY
                    } else {
                        -0.5 * t[0] * t[0]
                    }
                },
                &mut rng,
            )
            .unwrap();
        // Every recorded sample must respect the constraint.
        assert!(res.samples.iter().all(|s| s[0] >= 0.0));
        // Mean of a half-normal with σ=1 is √(2/π) ≈ 0.7979.
        let mean = sample_mean(&res.samples).unwrap()[0];
        let target = (2.0 / std::f64::consts::PI).sqrt();
        assert!((mean - target).abs() < 0.1, "half-normal mean {mean}");
    }

    #[test]
    fn metropolis_bivariate_correlated_gaussian() {
        // Target: zero-mean Gaussian with correlation ρ = 0.6.
        let rho = 0.6;
        let inv_det = 1.0 / (1.0 - rho * rho);
        let cfg = MetropolisConfig {
            n_samples: 50_000,
            n_warmup: 5_000,
            proposal_scale: 1.0,
            adapt: true,
            target_accept: 0.234,
            thin: 1,
        };
        let s = MetropolisSampler::new(cfg).unwrap();
        let mut rng = BayesRng::new(99);
        let res = s
            .sample(
                &[0.0, 0.0],
                move |t| {
                    let (x, y) = (t[0], t[1]);
                    -0.5 * inv_det * (x * x - 2.0 * rho * x * y + y * y)
                },
                &mut rng,
            )
            .unwrap();
        // Recover the off-diagonal correlation.
        let n = res.samples.len() as f64;
        let mean = sample_mean(&res.samples).unwrap();
        let mut cxy = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for s in &res.samples {
            let dx = s[0] - mean[0];
            let dy = s[1] - mean[1];
            cxy += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        cxy /= n;
        vx /= n;
        vy /= n;
        let corr = cxy / (vx.sqrt() * vy.sqrt());
        assert!((corr - rho).abs() < 0.08, "corr {corr} != {rho}");
    }

    #[test]
    fn slice_config_validation() {
        let mut c = SliceConfig::default();
        assert!(c.validate().is_ok());
        c.width = 0.0;
        assert!(c.validate().is_err());
        let c2 = SliceConfig {
            max_steps: 0,
            ..Default::default()
        };
        assert!(c2.validate().is_err());
    }

    #[test]
    fn slice_recovers_gaussian_moments() {
        let mu = -1.0;
        let sigma = 0.8;
        let cfg = SliceConfig {
            n_samples: 30_000,
            n_warmup: 1_000,
            width: 1.0,
            max_steps: 50,
        };
        let s = SliceSampler::new(cfg).unwrap();
        let mut rng = BayesRng::new(2024);
        let samples = s
            .sample(&[0.0], move |t| log_normal(t[0], mu, sigma), &mut rng)
            .unwrap();
        assert_eq!(samples.len(), 30_000);
        let mean = sample_mean(&samples).unwrap()[0];
        let var = sample_variance(&samples).unwrap()[0];
        assert!((mean - mu).abs() < 0.06, "slice mean {mean} != {mu}");
        assert!(
            (var - sigma * sigma).abs() < 0.06,
            "slice var {var} != {}",
            sigma * sigma
        );
    }

    #[test]
    fn slice_bimodal_target_visits_both_modes() {
        // Mixture of N(-3, 0.5²) and N(3, 0.5²): slice sampling with wide
        // stepping-out can traverse the low-density valley.
        let cfg = SliceConfig {
            n_samples: 40_000,
            n_warmup: 2_000,
            width: 4.0,
            max_steps: 80,
        };
        let s = SliceSampler::new(cfg).unwrap();
        let mut rng = BayesRng::new(555);
        let samples = s
            .sample(
                &[-3.0],
                |t| {
                    let x = t[0];
                    let a = (-0.5 * ((x + 3.0) / 0.5).powi(2)).exp();
                    let b = (-0.5 * ((x - 3.0) / 0.5).powi(2)).exp();
                    (0.5 * a + 0.5 * b).max(1e-300).ln()
                },
                &mut rng,
            )
            .unwrap();
        let left = samples.iter().filter(|s| s[0] < 0.0).count();
        let right = samples.len() - left;
        // Both modes must be visited a non-trivial number of times.
        assert!(left > 5_000 && right > 5_000, "modes: L={left} R={right}");
        // By symmetry the overall mean is near zero.
        let mean = sample_mean(&samples).unwrap()[0];
        assert!(mean.abs() < 0.5, "bimodal mean {mean}");
    }

    #[test]
    fn slice_rejects_empty_init() {
        let s = SliceSampler::new(SliceConfig::default()).unwrap();
        let mut rng = BayesRng::new(1);
        assert!(matches!(
            s.sample(&[], |_| 0.0, &mut rng),
            Err(BayesError::EmptyInputs)
        ));
    }

    #[test]
    fn sample_mean_and_variance_helpers() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let m = sample_mean(&data).unwrap();
        assert!((m[0] - 2.0).abs() < 1e-12 && (m[1] - 20.0).abs() < 1e-12);
        let v = sample_variance(&data).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-12 && (v[1] - 100.0).abs() < 1e-9);
        assert!(matches!(sample_mean(&[]), Err(BayesError::EmptyInputs)));
        assert!(matches!(
            sample_variance(&[vec![1.0]]),
            Err(BayesError::InsufficientSamples { .. })
        ));
    }
}
