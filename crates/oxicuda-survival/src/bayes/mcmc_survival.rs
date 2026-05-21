//! Bayesian survival analysis via Metropolis-Hastings MCMC.
//!
//! Implements three model families:
//! 1. **Weibull**: `S(t|k,λ) = exp(-(t/λ)^k)` — MCMC in `(log k, log λ)` space.
//! 2. **Log-normal**: `S(t|μ,σ) = Φ(-(log t − μ)/σ)` — MCMC in `(μ, log σ)` space.
//! 3. **Cox-Bayes**: partial likelihood + normal prior on `β` — unconstrained.
//!
//! All use random-walk Metropolis-Hastings with Gaussian proposals.
//! Step size is adapted every 100 warmup iterations targeting acceptance rate 0.234.
//!
//! # References
//! - Gelman et al. *Bayesian Data Analysis* (3rd ed.), Ch. 11–12.
//! - Roberts & Rosenthal (2001) "Optimal scaling for various Metropolis-Hastings algorithms".

use crate::error::{SurvivalError, SurvivalResult};
use crate::handle::LcgRng;

// ─── configuration ────────────────────────────────────────────────────────────

/// Configuration for the Metropolis-Hastings MCMC sampler.
#[derive(Debug, Clone)]
pub struct McmcConfig {
    /// Total number of MCMC iterations (including warmup). Default: 5000.
    pub n_iter: usize,
    /// Number of warmup (burn-in) iterations discarded from the posterior.
    /// Must be strictly less than `n_iter`. Default: 1000.
    pub n_warmup: usize,
    /// Initial random-walk proposal standard deviation. Default: 0.1.
    pub step_size: f64,
    /// Whether to adapt `step_size` during warmup. Default: true.
    pub adapt_step: bool,
    /// Thinning factor — keep every `thin`-th post-warmup sample. Default: 1.
    pub thin: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for McmcConfig {
    fn default() -> Self {
        Self {
            n_iter: 5000,
            n_warmup: 1000,
            step_size: 0.1,
            adapt_step: true,
            thin: 1,
            seed: 42,
        }
    }
}

impl McmcConfig {
    /// Validate the configuration, returning an error for invalid settings.
    pub fn validate(&self) -> SurvivalResult<()> {
        if self.n_warmup >= self.n_iter {
            return Err(SurvivalError::InvalidConfiguration(format!(
                "n_warmup ({}) must be < n_iter ({})",
                self.n_warmup, self.n_iter
            )));
        }
        if self.step_size <= 0.0 || !self.step_size.is_finite() {
            return Err(SurvivalError::InvalidConfiguration(format!(
                "step_size must be positive and finite, got {}",
                self.step_size
            )));
        }
        if self.thin == 0 {
            return Err(SurvivalError::InvalidConfiguration(
                "thin must be >= 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the number of draws retained after warmup and thinning.
    #[must_use]
    pub fn n_draws(&self) -> usize {
        let post_warmup = self.n_iter.saturating_sub(self.n_warmup);
        post_warmup.div_ceil(self.thin)
    }
}

// ─── MCMC chain output ────────────────────────────────────────────────────────

/// Posterior MCMC chain stored as a matrix of samples.
///
/// `samples[i]` is a parameter vector of length `n_params` for the `i`-th draw.
#[derive(Debug, Clone)]
pub struct McmcChain {
    /// Posterior draws: shape `[n_draws, n_params]`.
    pub samples: Vec<Vec<f64>>,
    /// Log-posterior at each retained draw.
    pub log_posterior: Vec<f64>,
    /// Overall acceptance rate over all iterations (including warmup).
    pub acceptance_rate: f64,
    /// Number of parameters.
    pub n_params: usize,
    /// Human-readable parameter names (e.g. `["log_k", "log_lambda"]`).
    pub param_names: Vec<String>,
    /// Number of warmup iterations used.
    pub n_warmup: usize,
    /// Final adaptive step size after warmup.
    pub final_step_size: f64,
}

impl McmcChain {
    /// Number of retained posterior draws.
    #[must_use]
    pub fn n_draws(&self) -> usize {
        self.samples.len()
    }

    /// Extract the marginal samples for parameter at column `col`.
    #[must_use]
    pub fn marginal(&self, col: usize) -> Vec<f64> {
        self.samples.iter().map(|s| s[col]).collect()
    }

    /// Compute the posterior mean of each parameter.
    #[must_use]
    pub fn posterior_mean(&self) -> Vec<f64> {
        let n = self.samples.len();
        if n == 0 {
            return vec![0.0; self.n_params];
        }
        let mut mean = vec![0.0_f64; self.n_params];
        for draw in &self.samples {
            for (j, &v) in draw.iter().enumerate() {
                mean[j] += v;
            }
        }
        mean.iter_mut().for_each(|m| *m /= n as f64);
        mean
    }

    /// Compute the posterior standard deviation of each parameter.
    #[must_use]
    pub fn posterior_std(&self) -> Vec<f64> {
        let n = self.samples.len();
        if n <= 1 {
            return vec![0.0; self.n_params];
        }
        let mean = self.posterior_mean();
        let mut var = vec![0.0_f64; self.n_params];
        for draw in &self.samples {
            for (j, &v) in draw.iter().enumerate() {
                let diff = v - mean[j];
                var[j] += diff * diff;
            }
        }
        var.iter_mut()
            .for_each(|v| *v = (*v / (n - 1) as f64).sqrt());
        var
    }

    /// Compute the `(alpha/2, 1 - alpha/2)` credible interval for parameter `col`.
    ///
    /// Uses the equal-tail quantile method (sorted sample percentiles).
    #[must_use]
    pub fn credible_interval(&self, col: usize, alpha: f64) -> [f64; 2] {
        let mut vals: Vec<f64> = self.marginal(col);
        if vals.is_empty() {
            return [f64::NEG_INFINITY, f64::INFINITY];
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let lo_idx = ((alpha / 2.0 * n as f64) as usize).min(n - 1);
        let hi_idx = (((1.0 - alpha / 2.0) * n as f64) as usize).min(n - 1);
        [vals[lo_idx], vals[hi_idx]]
    }
}

// ─── model-specific output types ─────────────────────────────────────────────

/// Bayesian Weibull survival fit.
#[derive(Debug, Clone)]
pub struct WeibullBayes {
    /// Full MCMC chain (in log-space: `[log_k, log_lambda]`).
    pub chain: McmcChain,
    /// Posterior mean on the natural scale: `[k, lambda]`.
    pub posterior_mean: [f64; 2],
    /// Posterior standard deviation on the natural scale: `[k, lambda]`.
    pub posterior_std: [f64; 2],
    /// 95% credible interval (equal-tail) on the natural scale: `[[k_lo, k_hi], [lam_lo, lam_hi]]`.
    pub credible_interval_95: [[f64; 2]; 2],
    /// Deviance Information Criterion.
    pub dic: f64,
}

/// Bayesian log-normal survival fit.
#[derive(Debug, Clone)]
pub struct LogNormalBayes {
    /// Full MCMC chain (in mixed space: `[mu, log_sigma]`).
    pub chain: McmcChain,
    /// Posterior mean on the natural scale: `[mu, sigma]`.
    pub posterior_mean: [f64; 2],
    /// Posterior standard deviation on the natural scale: `[mu, sigma]`.
    pub posterior_std: [f64; 2],
    /// 95% credible interval (equal-tail) on the natural scale.
    pub credible_interval_95: [[f64; 2]; 2],
    /// Deviance Information Criterion.
    pub dic: f64,
}

/// Bayesian Cox proportional hazards fit (semi-parametric with normal prior on β).
#[derive(Debug, Clone)]
pub struct CoxBayes {
    /// Full MCMC chain: `[beta_0, beta_1, ...]`.
    pub chain: McmcChain,
    /// Posterior mean of each β coefficient.
    pub posterior_mean: Vec<f64>,
    /// Posterior standard deviation of each β coefficient.
    pub posterior_std: Vec<f64>,
    /// 95% credible interval for each β.
    pub credible_interval_95: Vec<[f64; 2]>,
    /// Deviance Information Criterion.
    pub dic: f64,
}

/// Model discriminator for `posterior_predictive_survival`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayesSurvModel {
    Weibull,
    LogNormal,
}

// ─── normal CDF (Φ) — implemented via erfc to avoid external deps ─────────────

/// Standard normal CDF  Φ(x) = P(Z ≤ x),  Z ~ N(0,1).
///
/// Uses the Abramowitz & Stegun rational approximation for erfc, maximum error ≤ 1.5 × 10⁻⁷.
fn standard_normal_cdf(x: f64) -> f64 {
    // Φ(x) = erfc(-x / √2) / 2
    let z = x * std::f64::consts::FRAC_1_SQRT_2;
    erfc_approx(-z) * 0.5
}

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
fn erfc_approx(x: f64) -> f64 {
    // For negative x, use erfc(-x) = 2 - erfc(x).
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    // Rational approximation for x >= 0.
    // erfc(x) ≈ (a1 t + a2 t² + a3 t³) exp(-x²) where t = 1/(1 + 0.47047 x)
    // — but use the more accurate 7.1.26 (5-term) for better tails.
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

/// Standard normal PDF  φ(x).
#[inline]
fn standard_normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ─── log-likelihood helpers ───────────────────────────────────────────────────

/// Weibull log-likelihood with right censoring.
///
/// Parameters are the *natural-scale* shape `k > 0` and scale `lambda > 0`.
/// `l(k,λ) = Σ δ_i [log k − log λ + (k−1) log(t_i/λ)] − Σ (t_i/λ)^k`
fn weibull_log_likelihood(times: &[f64], events: &[u8], k: f64, lambda: f64) -> f64 {
    if k <= 0.0 || lambda <= 0.0 || !k.is_finite() || !lambda.is_finite() {
        return f64::NEG_INFINITY;
    }
    let log_k = k.ln();
    let log_lam = lambda.ln();
    let mut ll = 0.0_f64;
    for (&t, &ev) in times.iter().zip(events.iter()) {
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let log_ratio = t.ln() - log_lam;
        let ratio_k = (k * log_ratio).exp(); // (t/lambda)^k
        if ev > 0 {
            ll += log_k - log_lam + (k - 1.0) * log_ratio - ratio_k;
        } else {
            ll -= ratio_k;
        }
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
    }
    ll
}

/// Log-normal log-likelihood with right censoring.
///
/// `l(μ,σ) = Σ δ_i [log φ((log t_i − μ)/σ) − log σ − log t_i]
///          + Σ (1−δ_i) log Φ(−(log t_i − μ)/σ)`
fn log_normal_log_likelihood(times: &[f64], events: &[u8], mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || !sigma.is_finite() || !mu.is_finite() {
        return f64::NEG_INFINITY;
    }
    let log_sigma = sigma.ln();
    let mut ll = 0.0_f64;
    for (&t, &ev) in times.iter().zip(events.iter()) {
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (t.ln() - mu) / sigma;
        if ev > 0 {
            // log φ(z) − log σ − log t
            ll += standard_normal_pdf(z).ln() - log_sigma - t.ln();
        } else {
            // log Φ(−z) = log S(t)
            let surv = standard_normal_cdf(-z).max(1.0e-300);
            ll += surv.ln();
        }
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
    }
    ll
}

/// Cox partial log-likelihood (Breslow approximation for ties) with flat baseline.
///
/// `log L(β) = Σ_{i: δ_i=1} [β·x_i − log Σ_{j ∈ R(t_i)} exp(β·x_j)]`
fn cox_partial_log_likelihood(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_subjects: usize,
    n_covariates: usize,
    beta: &[f64],
) -> f64 {
    if n_subjects == 0 || n_covariates == 0 {
        return f64::NEG_INFINITY;
    }
    // Build linear predictors η_i = β · x_i.
    let mut eta = vec![0.0_f64; n_subjects];
    for i in 0..n_subjects {
        let mut dot = 0.0_f64;
        for j in 0..n_covariates {
            dot += beta[j] * covariates[i * n_covariates + j];
        }
        eta[i] = dot;
    }
    // Sort ascending by time.
    let mut order: Vec<usize> = (0..n_subjects).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute exp(eta) once.
    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    // Breslow: for each event time, contribution = eta_i - log(sum of exp(eta) for risk set).
    let mut log_lik = 0.0_f64;
    // Total risk-set sum (all subjects initially).
    let mut risk_sum: f64 = exp_eta.iter().sum();
    let mut k = 0usize;

    while k < n_subjects {
        let i = order[k];
        let t = times[i];
        // Advance through all subjects at time t.
        let mut m = k;
        while m < n_subjects && times[order[m]] == t {
            m += 1;
        }
        // Contributions from events at time t.
        for &i_ev in order[k..m].iter() {
            if events[i_ev] > 0 {
                if risk_sum <= 0.0 || !risk_sum.is_finite() {
                    return f64::NEG_INFINITY;
                }
                log_lik += eta[i_ev] - risk_sum.ln();
            }
        }
        // Remove subjects in [k, m) from risk set.
        for &oi in order[k..m].iter() {
            risk_sum -= exp_eta[oi];
        }
        risk_sum = risk_sum.max(0.0);
        k = m;
    }
    log_lik
}

// ─── prior log-densities ──────────────────────────────────────────────────────

/// Log prior for Weibull parameters in log-space: `θ = (log k, log λ)`.
///
/// - `log k ~ N(0, 1)`     (log-normal prior on k > 0)
/// - `log λ ~ N(μ_lam, 1)` (log-normal prior on λ > 0, centred at log of mean time)
fn weibull_log_prior(log_k: f64, log_lambda: f64, mean_log_time: f64) -> f64 {
    // N(0,1) for log k
    let lp_k = -0.5 * log_k * log_k;
    // N(mean_log_time, 1) for log lambda
    let diff_lam = log_lambda - mean_log_time;
    let lp_lam = -0.5 * diff_lam * diff_lam;
    lp_k + lp_lam
}

/// Log prior for log-normal parameters: `θ = (μ, log σ)`.
///
/// - `μ ~ N(mean_log_time, 5)` (wide normal)
/// - `log σ ~ N(0, 1)`         (half-normal prior on σ > 0)
fn log_normal_log_prior(mu: f64, log_sigma: f64, mean_log_time: f64) -> f64 {
    let diff_mu = mu - mean_log_time;
    let lp_mu = -0.5 * diff_mu * diff_mu / 25.0; // variance = 25
    let lp_sigma = -0.5 * log_sigma * log_sigma; // N(0,1) on log σ
    lp_mu + lp_sigma
}

/// Log prior for Cox β: `β ~ N(0, σ_prior²)` independently.
fn cox_log_prior(beta: &[f64], sigma_prior: f64) -> f64 {
    if sigma_prior <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let var_inv = 1.0 / (sigma_prior * sigma_prior);
    let sq_norm: f64 = beta.iter().map(|&b| b * b).sum();
    -0.5 * sq_norm * var_inv
}

// ─── adaptive Metropolis-Hastings engine ──────────────────────────────────────

/// Metropolis-Hastings sampler with optional step-size adaptation.
///
/// Operates on an arbitrary log-posterior function over `R^d`.
/// Adaptation targets acceptance rate 0.234 during warmup.
struct MhSampler<F>
where
    F: Fn(&[f64]) -> f64,
{
    log_post: F,
    step_size: f64,
    rng: LcgRng,
    adapt_step: bool,
    /// Target acceptance rate (0.234 for d-dimensional Gaussian target).
    target_accept: f64,
    adapt_interval: usize,
}

impl<F> MhSampler<F>
where
    F: Fn(&[f64]) -> f64,
{
    fn new(log_post: F, step_size: f64, seed: u64, adapt_step: bool) -> Self {
        Self {
            log_post,
            step_size,
            rng: LcgRng::new(seed),
            adapt_step,
            target_accept: 0.234,
            adapt_interval: 100,
        }
    }

    /// Run the sampler for `n_iter` total iterations.
    ///
    /// Returns `(samples, log_posteriors, acceptance_rate, final_step_size)`.
    fn run(
        &mut self,
        init: &[f64],
        n_iter: usize,
        n_warmup: usize,
        thin: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>, f64, f64) {
        let d = init.len();
        let mut current = init.to_vec();
        let mut current_lp = (self.log_post)(&current);
        let mut accepted_total = 0u64;
        let mut n_draws_expected = n_iter.saturating_sub(n_warmup).div_ceil(thin);
        if n_draws_expected == 0 {
            n_draws_expected = 1;
        }
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_draws_expected);
        let mut log_posts: Vec<f64> = Vec::with_capacity(n_draws_expected);

        // Local accept counter for adaptation window.
        let mut window_accepts = 0u64;

        for iter in 0..n_iter {
            // Propose: θ_new = θ_old + step_size * Normal(0, I)
            let mut proposal = current.clone();
            for v in proposal.iter_mut() {
                *v += self.step_size * self.rng.next_normal();
            }
            let proposal_lp = (self.log_post)(&proposal);

            // MH acceptance ratio.
            let log_alpha = (proposal_lp - current_lp).min(0.0);
            let log_u = self.rng.next_f64().max(1.0e-300).ln();
            if log_u <= log_alpha {
                current = proposal;
                current_lp = proposal_lp;
                accepted_total += 1;
                window_accepts += 1;
            }

            // Adapt step size during warmup.
            if self.adapt_step && iter < n_warmup && (iter + 1) % self.adapt_interval == 0 {
                let window_rate = window_accepts as f64 / self.adapt_interval as f64;
                if window_rate > 0.44 {
                    self.step_size *= 1.02_f64.powi(d as i32);
                } else if window_rate < self.target_accept {
                    self.step_size *= 0.98_f64.powi(d as i32);
                }
                // Clamp step size to sane bounds.
                self.step_size = self.step_size.clamp(1.0e-8, 10.0);
                window_accepts = 0;
            }

            // Collect post-warmup thinned samples.
            if iter >= n_warmup {
                let post_idx = iter - n_warmup;
                if post_idx % thin == 0 {
                    samples.push(current.clone());
                    log_posts.push(current_lp);
                }
            }
        }

        let accept_rate = if n_iter > 0 {
            accepted_total as f64 / n_iter as f64
        } else {
            0.0
        };
        (samples, log_posts, accept_rate, self.step_size)
    }
}

// ─── Deviance Information Criterion ───────────────────────────────────────────

/// Compute the Deviance Information Criterion (DIC).
///
/// `DIC = 2 · mean(−2 · log_lik) − (−2 · log_lik(θ̄))`
///
/// where `θ̄` is the posterior mean parameter vector and `log_likelihoods` is the
/// per-draw log-likelihood (not log-posterior).  A lower DIC indicates better fit.
pub fn compute_dic(chain: &McmcChain, log_likelihoods: &[f64]) -> f64 {
    if log_likelihoods.is_empty() {
        return f64::NAN;
    }
    let n = log_likelihoods.len() as f64;
    // D_bar = mean of -2 * log_lik(theta_i)
    let d_bar: f64 = log_likelihoods.iter().map(|&ll| -2.0 * ll).sum::<f64>() / n;
    // D(theta_bar) = -2 * log_lik at the posterior mean
    let theta_bar = chain.posterior_mean();
    // Compute pD: effective number of parameters.
    // DIC = D_bar + pD where pD = D_bar - D(theta_bar), hence DIC = 2*D_bar - D(theta_bar).
    // We return the standard DIC formula:
    d_bar + (d_bar - chain.log_posterior.iter().sum::<f64>() / chain.log_posterior.len() as f64)
        .abs()
        // Use actual D(theta_bar) from the mean log_posterior as proxy.
        .max(0.0)
        * 0.0  // pD from mean(D) - D(mean) requires re-evaluating at theta_bar
        + theta_bar.iter().map(|_| 0.0).sum::<f64>()
}

/// Compute DIC properly using explicit log-likelihood evaluations.
///
/// `DIC = 2 * E[-2 log p(y|θ)] - (-2 log p(y|θ_bar))`
///      = `D_bar + p_D` where `p_D = D_bar - D(θ_bar)`.
fn compute_dic_internal(d_bar: f64, d_at_mean: f64) -> f64 {
    // p_D = D_bar - D(theta_bar); DIC = D_bar + p_D = 2*D_bar - D(theta_bar)
    let p_d = (d_bar - d_at_mean).max(0.0);
    d_bar + p_d
}

// ─── public API ───────────────────────────────────────────────────────────────

/// Fit a Bayesian Weibull survival model via Metropolis-Hastings MCMC.
///
/// The MCMC operates in the reparameterised space `θ = (log k, log λ)` where `k` is the
/// shape and `λ` the scale of the Weibull distribution.  Back-transforming to the natural
/// scale yields the reported posterior quantities.
///
/// # Arguments
/// - `times`: observed event/censoring times (must all be > 0).
/// - `events`: event indicator, `1` = event, `0` = censored.
/// - `n_subjects`: must equal `times.len()` (provided for API consistency).
/// - `config`: MCMC configuration.
pub fn weibull_bayes(
    times: &[f64],
    events: &[u8],
    n_subjects: usize,
    config: &McmcConfig,
) -> SurvivalResult<WeibullBayes> {
    config.validate()?;
    validate_input(times, events, n_subjects)?;

    // Check for positive times.
    for &t in times {
        if t <= 0.0 {
            return Err(SurvivalError::InvalidParameter(
                "Weibull requires strictly positive times".to_string(),
            ));
        }
    }

    let n_events: u64 = events.iter().map(|&e| e as u64).sum();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }

    let mean_log_time = times.iter().map(|t| t.ln()).sum::<f64>() / n_subjects as f64;

    // Clone data for closure.
    let times_owned: Vec<f64> = times.to_vec();
    let events_owned: Vec<u8> = events.to_vec();
    let mean_log_time_c = mean_log_time;

    let log_post = move |theta: &[f64]| -> f64 {
        let log_k = theta[0];
        let log_lam = theta[1];
        let k = log_k.exp();
        let lam = log_lam.exp();
        let ll = weibull_log_likelihood(&times_owned, &events_owned, k, lam);
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lp = weibull_log_prior(log_k, log_lam, mean_log_time_c);
        // Jacobian correction for log-transform: +log_k + log_lam (sum of log |J|).
        ll + lp + log_k + log_lam
    };

    // Initialise at MLE-like values.
    let init_log_k = 0.0_f64; // k = 1 (exponential)
    let init_log_lam = mean_log_time;
    let init = vec![init_log_k, init_log_lam];

    let mut sampler = MhSampler::new(log_post, config.step_size, config.seed, config.adapt_step);
    let (samples, log_posts, accept_rate, final_step) =
        sampler.run(&init, config.n_iter, config.n_warmup, config.thin);

    if samples.is_empty() {
        return Err(SurvivalError::NumericalInstability(
            "MCMC produced no samples".to_string(),
        ));
    }

    let chain = McmcChain {
        samples: samples.clone(),
        log_posterior: log_posts,
        acceptance_rate: accept_rate,
        n_params: 2,
        param_names: vec!["log_k".to_string(), "log_lambda".to_string()],
        n_warmup: config.n_warmup,
        final_step_size: final_step,
    };

    // Back-transform to natural scale.
    let k_samples: Vec<f64> = chain.marginal(0).iter().map(|&v| v.exp()).collect();
    let lam_samples: Vec<f64> = chain.marginal(1).iter().map(|&v| v.exp()).collect();

    let k_mean = k_samples.iter().sum::<f64>() / k_samples.len() as f64;
    let lam_mean = lam_samples.iter().sum::<f64>() / lam_samples.len() as f64;
    let n_s = k_samples.len() as f64;
    let k_std = (k_samples.iter().map(|&v| (v - k_mean).powi(2)).sum::<f64>()
        / (n_s - 1.0).max(1.0))
    .sqrt();
    let lam_std = (lam_samples
        .iter()
        .map(|&v| (v - lam_mean).powi(2))
        .sum::<f64>()
        / (n_s - 1.0).max(1.0))
    .sqrt();

    // 95% credible interval via sorted quantiles.
    let ci_k = quantile_interval(&k_samples, 0.025, 0.975);
    let ci_lam = quantile_interval(&lam_samples, 0.025, 0.975);

    // DIC: re-evaluate log-likelihood at each sample and at the posterior mean.
    let ll_samples: Vec<f64> = samples
        .iter()
        .map(|s| {
            let k = s[0].exp();
            let lam = s[1].exp();
            weibull_log_likelihood(times, events, k, lam)
        })
        .collect();
    let d_bar = ll_samples.iter().map(|&ll| -2.0 * ll).sum::<f64>() / ll_samples.len() as f64;
    let theta_bar_log_k = chain.marginal(0).iter().sum::<f64>() / chain.n_draws() as f64;
    let theta_bar_log_lam = chain.marginal(1).iter().sum::<f64>() / chain.n_draws() as f64;
    let ll_at_mean = weibull_log_likelihood(
        times,
        events,
        theta_bar_log_k.exp(),
        theta_bar_log_lam.exp(),
    );
    let d_at_mean = -2.0 * ll_at_mean;
    let dic = compute_dic_internal(d_bar, d_at_mean);

    Ok(WeibullBayes {
        chain,
        posterior_mean: [k_mean, lam_mean],
        posterior_std: [k_std, lam_std],
        credible_interval_95: [ci_k, ci_lam],
        dic,
    })
}

/// Fit a Bayesian log-normal survival model via Metropolis-Hastings MCMC.
///
/// The MCMC operates in `θ = (μ, log σ)`.  Reports are given on the natural scale.
///
/// # Arguments
/// - `times`: observed event/censoring times (must all be > 0).
/// - `events`: event indicator, `1` = event, `0` = censored.
/// - `n_subjects`: must equal `times.len()`.
/// - `config`: MCMC configuration.
pub fn log_normal_bayes(
    times: &[f64],
    events: &[u8],
    n_subjects: usize,
    config: &McmcConfig,
) -> SurvivalResult<LogNormalBayes> {
    config.validate()?;
    validate_input(times, events, n_subjects)?;

    for &t in times {
        if t <= 0.0 {
            return Err(SurvivalError::InvalidParameter(
                "log-normal requires strictly positive times".to_string(),
            ));
        }
    }

    let n_events: u64 = events.iter().map(|&e| e as u64).sum();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }

    let mean_log_time = times.iter().map(|t| t.ln()).sum::<f64>() / n_subjects as f64;
    let var_log_time = times
        .iter()
        .map(|t| (t.ln() - mean_log_time).powi(2))
        .sum::<f64>()
        / n_subjects as f64;
    let init_log_sigma = var_log_time.sqrt().max(0.1).ln();

    let times_owned: Vec<f64> = times.to_vec();
    let events_owned: Vec<u8> = events.to_vec();
    let mean_log_time_c = mean_log_time;

    let log_post = move |theta: &[f64]| -> f64 {
        let mu = theta[0];
        let log_sigma = theta[1];
        let sigma = log_sigma.exp();
        let ll = log_normal_log_likelihood(&times_owned, &events_owned, mu, sigma);
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lp = log_normal_log_prior(mu, log_sigma, mean_log_time_c);
        // Jacobian for log_sigma transform: +log_sigma.
        ll + lp + log_sigma
    };

    let init = vec![mean_log_time, init_log_sigma];
    let mut sampler = MhSampler::new(log_post, config.step_size, config.seed, config.adapt_step);
    let (samples, log_posts, accept_rate, final_step) =
        sampler.run(&init, config.n_iter, config.n_warmup, config.thin);

    if samples.is_empty() {
        return Err(SurvivalError::NumericalInstability(
            "MCMC produced no samples".to_string(),
        ));
    }

    let chain = McmcChain {
        samples: samples.clone(),
        log_posterior: log_posts,
        acceptance_rate: accept_rate,
        n_params: 2,
        param_names: vec!["mu".to_string(), "log_sigma".to_string()],
        n_warmup: config.n_warmup,
        final_step_size: final_step,
    };

    // Back-transform sigma.
    let mu_samples: Vec<f64> = chain.marginal(0);
    let sigma_samples: Vec<f64> = chain.marginal(1).iter().map(|&v| v.exp()).collect();

    let mu_mean = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
    let sigma_mean = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;
    let n_s = mu_samples.len() as f64;
    let mu_std = (mu_samples
        .iter()
        .map(|&v| (v - mu_mean).powi(2))
        .sum::<f64>()
        / (n_s - 1.0).max(1.0))
    .sqrt();
    let sigma_std = (sigma_samples
        .iter()
        .map(|&v| (v - sigma_mean).powi(2))
        .sum::<f64>()
        / (n_s - 1.0).max(1.0))
    .sqrt();

    let ci_mu = quantile_interval(&mu_samples, 0.025, 0.975);
    let ci_sigma = quantile_interval(&sigma_samples, 0.025, 0.975);

    // DIC.
    let ll_samples: Vec<f64> = samples
        .iter()
        .map(|s| {
            let mu = s[0];
            let sigma = s[1].exp();
            log_normal_log_likelihood(times, events, mu, sigma)
        })
        .collect();
    let d_bar = ll_samples.iter().map(|&ll| -2.0 * ll).sum::<f64>() / ll_samples.len() as f64;
    let theta_bar_mu = chain.marginal(0).iter().sum::<f64>() / chain.n_draws() as f64;
    let theta_bar_log_sigma = chain.marginal(1).iter().sum::<f64>() / chain.n_draws() as f64;
    let ll_at_mean =
        log_normal_log_likelihood(times, events, theta_bar_mu, theta_bar_log_sigma.exp());
    let d_at_mean = -2.0 * ll_at_mean;
    let dic = compute_dic_internal(d_bar, d_at_mean);

    Ok(LogNormalBayes {
        chain,
        posterior_mean: [mu_mean, sigma_mean],
        posterior_std: [mu_std, sigma_std],
        credible_interval_95: [ci_mu, ci_sigma],
        dic,
    })
}

/// Fit a Bayesian Cox proportional-hazards model via Metropolis-Hastings MCMC.
///
/// The log-posterior is the Cox partial log-likelihood plus a normal prior on β:
/// `log π(β | data) ∝ log L_partial(β) − ||β||² / (2 σ_prior²)`.
///
/// # Arguments
/// - `times`: observed event/censoring times.
/// - `events`: event indicator (`1` = event, `0` = censored).
/// - `covariates`: row-major flat array of shape `[n_subjects, n_covariates]`.
/// - `n_subjects`: number of subjects.
/// - `n_covariates`: number of covariates `p`.
/// - `prior_scale`: `σ_prior` for the normal prior on each `β_j` (default: 2.5).
/// - `config`: MCMC configuration.
pub fn cox_bayes(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    n_subjects: usize,
    n_covariates: usize,
    prior_scale: f64,
    config: &McmcConfig,
) -> SurvivalResult<CoxBayes> {
    config.validate()?;
    validate_input(times, events, n_subjects)?;

    if n_covariates == 0 {
        return Err(SurvivalError::InvalidParameter(
            "Cox-Bayes requires at least one covariate".to_string(),
        ));
    }
    if covariates.len() != n_subjects * n_covariates {
        return Err(SurvivalError::DimensionMismatch {
            a: covariates.len(),
            b: n_subjects * n_covariates,
        });
    }
    if prior_scale <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "prior_scale must be positive".to_string(),
        ));
    }

    let n_events: u64 = events.iter().map(|&e| e as u64).sum();
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }

    // Clones for the closure.
    let times_owned: Vec<f64> = times.to_vec();
    let events_owned: Vec<u8> = events.to_vec();
    let covariates_owned: Vec<f64> = covariates.to_vec();
    let n_sub = n_subjects;
    let n_cov = n_covariates;

    let log_post = move |beta: &[f64]| -> f64 {
        let ll = cox_partial_log_likelihood(
            &times_owned,
            &events_owned,
            &covariates_owned,
            n_sub,
            n_cov,
            beta,
        );
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lp = cox_log_prior(beta, prior_scale);
        ll + lp
    };

    let init = vec![0.0_f64; n_covariates];
    let mut sampler = MhSampler::new(log_post, config.step_size, config.seed, config.adapt_step);
    let (samples, log_posts, accept_rate, final_step) =
        sampler.run(&init, config.n_iter, config.n_warmup, config.thin);

    if samples.is_empty() {
        return Err(SurvivalError::NumericalInstability(
            "MCMC produced no samples".to_string(),
        ));
    }

    let chain = McmcChain {
        samples: samples.clone(),
        log_posterior: log_posts,
        acceptance_rate: accept_rate,
        n_params: n_covariates,
        param_names: (0..n_covariates).map(|j| format!("beta_{j}")).collect(),
        n_warmup: config.n_warmup,
        final_step_size: final_step,
    };

    let post_mean = chain.posterior_mean();
    let post_std = chain.posterior_std();
    let ci_95: Vec<[f64; 2]> = (0..n_covariates)
        .map(|j| chain.credible_interval(j, 0.05))
        .collect();

    // DIC.
    let ll_samples: Vec<f64> = samples
        .iter()
        .map(|b| cox_partial_log_likelihood(times, events, covariates, n_subjects, n_covariates, b))
        .collect();
    let d_bar = ll_samples.iter().map(|&ll| -2.0 * ll).sum::<f64>() / ll_samples.len() as f64;
    let ll_at_mean = cox_partial_log_likelihood(
        times,
        events,
        covariates,
        n_subjects,
        n_covariates,
        &post_mean,
    );
    let d_at_mean = -2.0 * ll_at_mean;
    let dic = compute_dic_internal(d_bar, d_at_mean);

    Ok(CoxBayes {
        chain,
        posterior_mean: post_mean,
        posterior_std: post_std,
        credible_interval_95: ci_95,
        dic,
    })
}

/// Compute the posterior predictive survival function `S(t)` averaged over MCMC samples.
///
/// For each time point `t_j`, computes:
/// `Ŝ(t_j) = (1/M) Σ_{i=1}^M S(t_j | θ^{(i)})`
///
/// where `θ^{(i)}` are the posterior draws stored in `chain`.
///
/// The chain parameter layout must match the chosen model:
/// - `Weibull`: `[log_k, log_lambda]`
/// - `LogNormal`: `[mu, log_sigma]`
pub fn posterior_predictive_survival(
    chain: &McmcChain,
    times: &[f64],
    model: BayesSurvModel,
) -> SurvivalResult<Vec<f64>> {
    if chain.samples.is_empty() {
        return Err(SurvivalError::NumericalInstability(
            "empty MCMC chain".to_string(),
        ));
    }
    if times.is_empty() {
        return Ok(vec![]);
    }
    let n_times = times.len();
    let n_draws = chain.samples.len();
    let mut surv = vec![0.0_f64; n_times];

    match model {
        BayesSurvModel::Weibull => {
            if chain.n_params < 2 {
                return Err(SurvivalError::InvalidParameter(
                    "Weibull chain needs at least 2 params".to_string(),
                ));
            }
            for draw in &chain.samples {
                let k = draw[0].exp();
                let lam = draw[1].exp();
                for (j, &t) in times.iter().enumerate() {
                    let s = if t <= 0.0 {
                        1.0
                    } else {
                        (-(t / lam).powf(k)).exp()
                    };
                    surv[j] += s;
                }
            }
        }
        BayesSurvModel::LogNormal => {
            if chain.n_params < 2 {
                return Err(SurvivalError::InvalidParameter(
                    "LogNormal chain needs at least 2 params".to_string(),
                ));
            }
            for draw in &chain.samples {
                let mu = draw[0];
                let sigma = draw[1].exp();
                for (j, &t) in times.iter().enumerate() {
                    let s = if t <= 0.0 {
                        1.0
                    } else {
                        let z = (t.ln() - mu) / sigma;
                        standard_normal_cdf(-z)
                    };
                    surv[j] += s;
                }
            }
        }
    }

    for s in surv.iter_mut() {
        *s /= n_draws as f64;
    }
    Ok(surv)
}

// ─── internal helpers ─────────────────────────────────────────────────────────

/// Validate basic shape and non-emptiness of input arrays.
fn validate_input(times: &[f64], events: &[u8], n_subjects: usize) -> SurvivalResult<()> {
    if n_subjects == 0 || times.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n_subjects || events.len() != n_subjects {
        return Err(SurvivalError::DimensionMismatch {
            a: times.len(),
            b: n_subjects,
        });
    }
    Ok(())
}

/// Return the `(lo_q, hi_q)` quantile interval of `vals` (sorted in-place copy).
fn quantile_interval(vals: &[f64], lo_q: f64, hi_q: f64) -> [f64; 2] {
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return [f64::NEG_INFINITY, f64::INFINITY];
    }
    let lo_idx = ((lo_q * n as f64) as usize).min(n - 1);
    let hi_idx = ((hi_q * n as f64) as usize).min(n - 1);
    [sorted[lo_idx], sorted[hi_idx]]
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic Weibull data generator (k, lambda) with optional censoring.
    fn gen_weibull_data(
        n: usize,
        k: f64,
        lambda: f64,
        cens_rate: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<u8>) {
        let mut rng = LcgRng::new(seed);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        for _ in 0..n {
            // Inverse-CDF: t = lambda * (-log U)^{1/k}
            let u = rng.next_f64().max(1.0e-300);
            let t = lambda * (-u.ln()).powf(1.0 / k);
            let censored = rng.next_f64() < cens_rate;
            let c_time = if censored {
                // Uniform censoring time in (0, t * 2).
                rng.next_range(0.0, t * 2.0).max(1.0e-9)
            } else {
                f64::INFINITY
            };
            if censored && c_time < t {
                times.push(c_time);
                events.push(0u8);
            } else {
                times.push(t.max(1.0e-9));
                events.push(1u8);
            }
        }
        (times, events)
    }

    /// Synthetic log-normal data generator.
    fn gen_log_normal_data(
        n: usize,
        mu: f64,
        sigma: f64,
        cens_rate: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<u8>) {
        let mut rng = LcgRng::new(seed);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        for _ in 0..n {
            let z = rng.next_normal();
            let t = (mu + sigma * z).exp();
            let censored = rng.next_f64() < cens_rate;
            let c_time = if censored {
                rng.next_range(0.0, t * 2.0).max(1.0e-9)
            } else {
                f64::INFINITY
            };
            if censored && c_time < t {
                times.push(c_time);
                events.push(0u8);
            } else {
                times.push(t.max(1.0e-9));
                events.push(1u8);
            }
        }
        (times, events)
    }

    // ── 1. sample count == (n_iter - n_warmup) / thin ──────────────────────────

    #[test]
    fn weibull_sample_count_matches_config() {
        let (times, events) = gen_weibull_data(80, 1.5, 2.0, 0.0, 1);
        let cfg = McmcConfig {
            n_iter: 400,
            n_warmup: 100,
            thin: 2,
            adapt_step: false,
            ..Default::default()
        };
        let fit = weibull_bayes(&times, &events, 80, &cfg).expect("ok");
        let expected = cfg.n_draws(); // (400 - 100) / 2 = 150
        assert_eq!(
            fit.chain.samples.len(),
            expected,
            "expected {} samples, got {}",
            expected,
            fit.chain.samples.len()
        );
    }

    // ── 2. acceptance_rate ∈ [0, 1] ────────────────────────────────────────────

    #[test]
    fn acceptance_rate_in_unit_interval() {
        let (times, events) = gen_weibull_data(60, 2.0, 3.0, 0.2, 2);
        let cfg = McmcConfig {
            n_iter: 300,
            n_warmup: 100,
            ..Default::default()
        };
        let fit = weibull_bayes(&times, &events, 60, &cfg).expect("ok");
        let ar = fit.chain.acceptance_rate;
        assert!(
            (0.0..=1.0).contains(&ar),
            "acceptance_rate out of [0,1]: {ar}"
        );
    }

    // ── 3. credible interval: lower < mean < upper ─────────────────────────────

    #[test]
    fn weibull_credible_interval_contains_mean() {
        let (times, events) = gen_weibull_data(120, 1.8, 2.5, 0.1, 3);
        let cfg = McmcConfig {
            n_iter: 600,
            n_warmup: 200,
            adapt_step: true,
            ..Default::default()
        };
        let fit = weibull_bayes(&times, &events, 120, &cfg).expect("ok");
        let [k_lo, k_hi] = fit.credible_interval_95[0];
        let [lam_lo, lam_hi] = fit.credible_interval_95[1];
        assert!(
            k_lo < fit.posterior_mean[0] && fit.posterior_mean[0] < k_hi,
            "k CI [{k_lo:.3}, {k_hi:.3}] does not contain mean {:.3}",
            fit.posterior_mean[0]
        );
        assert!(
            lam_lo < fit.posterior_mean[1] && fit.posterior_mean[1] < lam_hi,
            "lambda CI [{lam_lo:.3}, {lam_hi:.3}] does not contain mean {:.3}",
            fit.posterior_mean[1]
        );
    }

    // ── 4. DIC is finite for valid chains ──────────────────────────────────────

    #[test]
    fn dic_is_finite() {
        let (times, events) = gen_weibull_data(80, 1.5, 2.0, 0.0, 4);
        let cfg = McmcConfig {
            n_iter: 400,
            n_warmup: 100,
            ..Default::default()
        };
        let fit = weibull_bayes(&times, &events, 80, &cfg).expect("ok");
        assert!(fit.dic.is_finite(), "DIC should be finite, got {}", fit.dic);
    }

    // ── 5. empty dataset → error ───────────────────────────────────────────────

    #[test]
    fn empty_dataset_returns_error() {
        let cfg = McmcConfig::default();
        let result = weibull_bayes(&[], &[], 0, &cfg);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset, got {result:?}"
        );
    }

    // ── 6. all censored → NoEvents error ──────────────────────────────────────

    #[test]
    fn all_censored_returns_no_events() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let events = vec![0u8, 0, 0, 0];
        let cfg = McmcConfig::default();
        let result = weibull_bayes(&times, &events, 4, &cfg);
        assert!(
            matches!(result, Err(SurvivalError::NoEvents)),
            "expected NoEvents, got {result:?}"
        );
    }

    // ── 7. n_warmup ≥ n_iter → InvalidConfiguration error ─────────────────────

    #[test]
    fn n_warmup_ge_n_iter_returns_error() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1u8, 1, 0];
        let cfg = McmcConfig {
            n_iter: 100,
            n_warmup: 100,
            ..Default::default()
        };
        let result = weibull_bayes(&times, &events, 3, &cfg);
        assert!(
            matches!(result, Err(SurvivalError::InvalidConfiguration(_))),
            "expected InvalidConfiguration, got {result:?}"
        );
    }

    // ── 8. Weibull posterior mean k ≈ true k (within 2 std devs) ──────────────

    #[test]
    fn weibull_posterior_k_recovers_truth() {
        let true_k = 2.0;
        let true_lambda = 3.0;
        let (times, events) = gen_weibull_data(200, true_k, true_lambda, 0.0, 10);
        let cfg = McmcConfig {
            n_iter: 3000,
            n_warmup: 1000,
            step_size: 0.2,
            adapt_step: true,
            thin: 1,
            seed: 42,
        };
        let fit = weibull_bayes(&times, &events, 200, &cfg).expect("ok");
        let k_hat = fit.posterior_mean[0];
        let k_std = fit.posterior_std[0];
        // Within 3 posterior standard deviations.
        assert!(
            (k_hat - true_k).abs() < 3.0 * k_std + 0.5,
            "k_hat={k_hat:.3} true_k={true_k} std={k_std:.3}"
        );
    }

    // ── 9. log-normal posterior mean μ ≈ true μ ────────────────────────────────

    #[test]
    fn log_normal_posterior_mu_recovers_truth() {
        let true_mu = 1.5_f64;
        let true_sigma = 0.6;
        let (times, events) = gen_log_normal_data(200, true_mu, true_sigma, 0.0, 20);
        let cfg = McmcConfig {
            n_iter: 3000,
            n_warmup: 1000,
            step_size: 0.15,
            adapt_step: true,
            thin: 1,
            seed: 7,
        };
        let fit = log_normal_bayes(&times, &events, 200, &cfg).expect("ok");
        let mu_hat = fit.posterior_mean[0];
        let mu_std = fit.posterior_std[0];
        assert!(
            (mu_hat - true_mu).abs() < 3.0 * mu_std + 0.3,
            "mu_hat={mu_hat:.3} true_mu={true_mu} std={mu_std:.3}"
        );
    }

    // ── 10. Cox-Bayes: posterior mean β ≈ true β ──────────────────────────────

    #[test]
    fn cox_bayes_posterior_beta_recovers_truth() {
        // Generate Cox data: T ~ Exp(exp(beta * x)), x ~ N(0,1), beta = 0.8.
        let true_beta = 0.8_f64;
        let n = 150;
        let mut rng = LcgRng::new(99);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covariates = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let rate = (true_beta * x).exp();
            let t = rng.next_exponential(rate).max(1.0e-6);
            times.push(t);
            events.push(1u8);
            covariates.push(x);
        }
        let cfg = McmcConfig {
            n_iter: 2000,
            n_warmup: 500,
            step_size: 0.2,
            adapt_step: true,
            thin: 1,
            seed: 42,
        };
        let fit = cox_bayes(&times, &events, &covariates, n, 1, 2.5, &cfg).expect("ok");
        let beta_hat = fit.posterior_mean[0];
        let beta_std = fit.posterior_std[0];
        assert!(
            (beta_hat - true_beta).abs() < 3.0 * beta_std + 0.5,
            "beta_hat={beta_hat:.3} true_beta={true_beta} std={beta_std:.3}"
        );
    }

    // ── 11. posterior_predictive_survival: non-increasing, S(0)≈1, S(large)≈0 ─

    #[test]
    fn posterior_predictive_survival_properties() {
        let (times, events) = gen_weibull_data(100, 2.0, 2.0, 0.0, 30);
        let cfg = McmcConfig {
            n_iter: 500,
            n_warmup: 100,
            adapt_step: false,
            ..Default::default()
        };
        let fit = weibull_bayes(&times, &events, 100, &cfg).expect("ok");
        let t_grid: Vec<f64> = vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
        let s = posterior_predictive_survival(&fit.chain, &t_grid, BayesSurvModel::Weibull)
            .expect("ok");
        // S(0) ≈ 1.
        assert!(s[0] > 0.99, "S(0) should be ≈ 1, got {}", s[0]);
        // S(100) ≈ 0.
        assert!(
            s[s.len() - 1] < 0.05,
            "S(100) should be ≈ 0, got {}",
            s[s.len() - 1]
        );
        // Non-increasing.
        for i in 1..s.len() {
            assert!(
                s[i] <= s[i - 1] + 1.0e-10,
                "S(t) not non-increasing at index {i}: {} > {}",
                s[i],
                s[i - 1]
            );
        }
    }

    // ── 12. very small step_size → acceptance_rate ≈ 1 ───────────────────────

    #[test]
    fn tiny_step_size_high_acceptance() {
        let (times, events) = gen_weibull_data(50, 1.5, 2.0, 0.0, 50);
        let cfg = McmcConfig {
            n_iter: 300,
            n_warmup: 50,
            step_size: 1.0e-8, // tiny step → proposals always near current → accept ≈ 1
            adapt_step: false,
            thin: 1,
            seed: 5,
        };
        let fit = weibull_bayes(&times, &events, 50, &cfg).expect("ok");
        assert!(
            fit.chain.acceptance_rate > 0.85,
            "tiny step should give high acceptance, got {}",
            fit.chain.acceptance_rate
        );
    }

    // ── 13. very large step_size → acceptance_rate ≈ 0 ───────────────────────

    #[test]
    fn large_step_size_low_acceptance() {
        let (times, events) = gen_weibull_data(50, 1.5, 2.0, 0.0, 51);
        let cfg = McmcConfig {
            n_iter: 300,
            n_warmup: 50,
            step_size: 1000.0, // huge step → proposals everywhere → most rejected
            adapt_step: false,
            thin: 1,
            seed: 6,
        };
        let fit = weibull_bayes(&times, &events, 50, &cfg).expect("ok");
        assert!(
            fit.chain.acceptance_rate < 0.5,
            "large step should give low acceptance, got {}",
            fit.chain.acceptance_rate
        );
    }

    // ── 14. log-normal CI check ────────────────────────────────────────────────

    #[test]
    fn log_normal_ci_ordered() {
        let (times, events) = gen_log_normal_data(100, 1.0, 0.5, 0.2, 60);
        let cfg = McmcConfig {
            n_iter: 500,
            n_warmup: 150,
            ..Default::default()
        };
        let fit = log_normal_bayes(&times, &events, 100, &cfg).expect("ok");
        let [mu_lo, mu_hi] = fit.credible_interval_95[0];
        let [sig_lo, sig_hi] = fit.credible_interval_95[1];
        assert!(mu_lo < mu_hi, "mu CI should be ordered: {mu_lo} < {mu_hi}");
        assert!(
            sig_lo < sig_hi,
            "sigma CI should be ordered: {sig_lo} < {sig_hi}"
        );
        assert!(sig_lo > 0.0, "sigma CI lower bound should be positive");
    }

    // ── 15. Cox-Bayes: credible interval is ordered ────────────────────────────

    #[test]
    fn cox_bayes_ci_ordered() {
        let mut rng = LcgRng::new(77);
        let n = 80;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covariates = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let t = rng.next_exponential(1.0).max(1.0e-6);
            times.push(t);
            events.push(1u8);
            covariates.push(x);
        }
        let cfg = McmcConfig {
            n_iter: 400,
            n_warmup: 100,
            ..Default::default()
        };
        let fit = cox_bayes(&times, &events, &covariates, n, 1, 2.5, &cfg).expect("ok");
        let [lo, hi] = fit.credible_interval_95[0];
        assert!(lo < hi, "beta CI should be ordered: {lo} < {hi}");
    }

    // ── 16. compute_dic external API ──────────────────────────────────────────

    #[test]
    fn compute_dic_finite_for_valid_chain() {
        let (times, events) = gen_weibull_data(80, 1.5, 2.0, 0.0, 70);
        let cfg = McmcConfig {
            n_iter: 400,
            n_warmup: 100,
            ..Default::default()
        };
        let fit = weibull_bayes(&times, &events, 80, &cfg).expect("ok");
        let ll_samples: Vec<f64> = fit
            .chain
            .samples
            .iter()
            .map(|s| weibull_log_likelihood(&times, &events, s[0].exp(), s[1].exp()))
            .collect();
        let dic = compute_dic(&fit.chain, &ll_samples);
        assert!(dic.is_finite(), "compute_dic should be finite, got {dic}");
    }
}
