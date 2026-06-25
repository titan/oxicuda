//! Conjugate Bayesian updates with closed-form posteriors and predictives.
//!
//! For each conjugate prior/likelihood pair the posterior is available in
//! closed form, so no sampling is required.  These exact updates are the
//! building blocks of Gibbs samplers (each full conditional is a conjugate
//! update) and provide ground-truth references for the iterative samplers in
//! [`crate::mcmc`].
//!
//! | Prior | Likelihood | Posterior | Predictive |
//! |-------|-----------|-----------|------------|
//! | Beta(α, β) | Bernoulli/Binomial | Beta(α+s, β+f) | Beta-Binomial |
//! | Gamma(α, β) | Poisson | Gamma(α+Σx, β+n) | Negative-Binomial |
//! | Normal(μ₀, σ₀²) | Normal (known σ²) | Normal | Normal |
//! | Normal-Inverse-Gamma | Normal (unknown μ, σ²) | Normal-Inverse-Gamma | Student-t |
//! | Dirichlet(α) | Categorical/Multinomial | Dirichlet(α + counts) | Dirichlet-Multinomial |
//!
//! All densities use a full-precision (`f64`) Lanczos log-gamma for numerical
//! stability — the predictive PMFs accumulate gamma terms whose arguments grow
//! with the data, so the `f32` [`crate::uncertainty::evidential::lgamma`] is not
//! precise enough here.

use crate::error::{BayesError, BayesResult};

// ─── Full-precision special functions ───────────────────────────────────────

/// Log-Gamma `log Γ(x)` for `x > 0` via the Lanczos approximation (g = 5,
/// 6 terms) evaluated entirely in `f64` — accurate to ~15 significant digits.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos coefficients (g = 5, n = 6): Γ(z+1) = √(2π)·t^(z+½)·e^(−t)·ser,
    // with t = z + g + ½ and z = x − 1.
    const G: f64 = 5.0;
    const C: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        1.208_650_973_866_179e-3,
        -5.395_239_384_953e-6,
    ];
    let z = x - 1.0;
    let mut ser = 1.000_000_000_190_015_f64;
    for (k, &ck) in C.iter().enumerate() {
        ser += ck / (z + k as f64 + 1.0);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + ser.ln() + (z + 0.5) * t.ln() - t
}

// ─── Beta-Bernoulli / Beta-Binomial ─────────────────────────────────────────

/// Beta(`alpha`, `beta`) conjugate prior for a Bernoulli/Binomial success rate.
#[derive(Debug, Clone, PartialEq)]
pub struct BetaPosterior {
    /// Shape α > 0 (prior + observed successes).
    pub alpha: f64,
    /// Shape β > 0 (prior + observed failures).
    pub beta: f64,
}

impl BetaPosterior {
    /// Construct a Beta prior.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if either shape is not strictly
    /// positive and finite.
    pub fn new(alpha: f64, beta: f64) -> BayesResult<Self> {
        if !(alpha.is_finite() && beta.is_finite() && alpha > 0.0 && beta > 0.0) {
            return Err(BayesError::InvalidConfig(format!(
                "Beta shapes must be positive and finite, got ({alpha}, {beta})"
            )));
        }
        Ok(Self { alpha, beta })
    }

    /// Update with `successes` out of `trials` Bernoulli observations.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `successes > trials`.
    pub fn update(&self, successes: u64, trials: u64) -> BayesResult<Self> {
        if successes > trials {
            return Err(BayesError::InvalidConfig(format!(
                "successes ({successes}) cannot exceed trials ({trials})"
            )));
        }
        Ok(Self {
            alpha: self.alpha + successes as f64,
            beta: self.beta + (trials - successes) as f64,
        })
    }

    /// Posterior mean `α / (α + β)`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance `αβ / ((α+β)²(α+β+1))`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        self.alpha * self.beta / (s * s * (s + 1.0))
    }

    /// Posterior mode `(α−1)/(α+β−2)` when both shapes exceed 1, else the mean.
    #[must_use]
    pub fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            self.mean()
        }
    }

    /// Posterior-predictive probability of observing `k` successes in `n` new
    /// trials — the Beta-Binomial PMF.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `k > n`.
    pub fn predictive_pmf(&self, k: u64, n: u64) -> BayesResult<f64> {
        if k > n {
            return Err(BayesError::InvalidConfig(format!(
                "k ({k}) cannot exceed n ({n})"
            )));
        }
        let kf = k as f64;
        let nf = n as f64;
        // log C(n,k) + log B(k+α, n−k+β) − log B(α, β).
        let log_choose = lgamma(nf + 1.0) - lgamma(kf + 1.0) - lgamma(nf - kf + 1.0);
        let log_num = log_beta(kf + self.alpha, nf - kf + self.beta);
        let log_den = log_beta(self.alpha, self.beta);
        Ok((log_choose + log_num - log_den).exp())
    }
}

/// `log B(a, b) = lgamma(a) + lgamma(b) − lgamma(a+b)`.
fn log_beta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

// ─── Gamma-Poisson ──────────────────────────────────────────────────────────

/// Gamma(`shape`, `rate`) conjugate prior for a Poisson rate `λ`.
///
/// Uses the **rate** (inverse-scale) parameterisation: mean `= shape / rate`.
#[derive(Debug, Clone, PartialEq)]
pub struct GammaPosterior {
    /// Shape α > 0.
    pub shape: f64,
    /// Rate β > 0.
    pub rate: f64,
}

impl GammaPosterior {
    /// Construct a Gamma prior in (shape, rate) parameterisation.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if either parameter is not
    /// strictly positive and finite.
    pub fn new(shape: f64, rate: f64) -> BayesResult<Self> {
        if !(shape.is_finite() && rate.is_finite() && shape > 0.0 && rate > 0.0) {
            return Err(BayesError::InvalidConfig(format!(
                "Gamma (shape, rate) must be positive and finite, got ({shape}, {rate})"
            )));
        }
        Ok(Self { shape, rate })
    }

    /// Update with Poisson counts `data`: posterior shape `+= Σx`, rate `+= n`.
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if `data` is empty.
    pub fn update(&self, data: &[u64]) -> BayesResult<Self> {
        if data.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let sum: u64 = data.iter().sum();
        Ok(Self {
            shape: self.shape + sum as f64,
            rate: self.rate + data.len() as f64,
        })
    }

    /// Posterior mean of `λ`: `shape / rate`.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Posterior variance of `λ`: `shape / rate²`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Posterior-predictive PMF of a new count `k` — the Negative-Binomial
    /// distribution `NB(r = shape, p = rate / (rate + 1))`.
    #[must_use]
    pub fn predictive_pmf(&self, k: u64) -> f64 {
        let r = self.shape;
        let kf = k as f64;
        let p = self.rate / (self.rate + 1.0);
        // log Γ(k+r) − log Γ(r) − log k! + r·log p + k·log(1−p).
        let log_pmf =
            lgamma(kf + r) - lgamma(r) - lgamma(kf + 1.0) + r * p.ln() + kf * (1.0 - p).ln();
        log_pmf.exp()
    }
}

// ─── Normal-Normal (known variance) ─────────────────────────────────────────

/// Normal(`mean`, `variance`) conjugate prior for the mean of a Gaussian with
/// **known** observation variance.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalKnownVarPosterior {
    /// Posterior mean μ.
    pub mean: f64,
    /// Posterior variance σ₀² of the mean (not the data variance).
    pub variance: f64,
    /// Known observation variance σ².
    pub obs_variance: f64,
}

impl NormalKnownVarPosterior {
    /// Construct a Normal prior on the mean given the known observation
    /// variance.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if either variance is not strictly
    /// positive and finite, or `prior_mean` is not finite.
    pub fn new(prior_mean: f64, prior_variance: f64, obs_variance: f64) -> BayesResult<Self> {
        if !prior_mean.is_finite() {
            return Err(BayesError::InvalidConfig(
                "prior_mean must be finite".to_string(),
            ));
        }
        if !(prior_variance.is_finite() && prior_variance > 0.0) {
            return Err(BayesError::InvalidConfig(
                "prior_variance must be positive and finite".to_string(),
            ));
        }
        if !(obs_variance.is_finite() && obs_variance > 0.0) {
            return Err(BayesError::InvalidConfig(
                "obs_variance must be positive and finite".to_string(),
            ));
        }
        Ok(Self {
            mean: prior_mean,
            variance: prior_variance,
            obs_variance,
        })
    }

    /// Update with Gaussian observations `data` (precision-weighted update).
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if `data` is empty.
    pub fn update(&self, data: &[f64]) -> BayesResult<Self> {
        if data.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let n = data.len() as f64;
        let sample_mean = data.iter().sum::<f64>() / n;
        // Precisions: prior τ₀ = 1/σ₀², data τ = n/σ².
        let prior_prec = 1.0 / self.variance;
        let data_prec = n / self.obs_variance;
        let post_prec = prior_prec + data_prec;
        let post_var = 1.0 / post_prec;
        let post_mean = post_var * (prior_prec * self.mean + data_prec * sample_mean);
        Ok(Self {
            mean: post_mean,
            variance: post_var,
            obs_variance: self.obs_variance,
        })
    }

    /// Posterior-predictive distribution of a single new observation:
    /// `N(mean, variance + obs_variance)`.
    #[must_use]
    pub fn predictive(&self) -> (f64, f64) {
        (self.mean, self.variance + self.obs_variance)
    }

    /// Log-density of the posterior-predictive at `x`.
    #[must_use]
    pub fn predictive_logpdf(&self, x: f64) -> f64 {
        let (m, v) = self.predictive();
        let z = (x - m) / v.sqrt();
        -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI * v).ln()
    }
}

// ─── Normal-Inverse-Gamma (unknown mean and variance) ───────────────────────

/// Normal-Inverse-Gamma conjugate prior for the mean **and** variance of a
/// Gaussian, parameterised by `(μ₀, κ₀, α₀, β₀)` (Murphy 2007, §3).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalInverseGamma {
    /// Prior/posterior mean μ.
    pub mu: f64,
    /// Pseudo-count κ for the mean (prior strength).
    pub kappa: f64,
    /// Inverse-Gamma shape α.
    pub alpha: f64,
    /// Inverse-Gamma scale β.
    pub beta: f64,
}

impl NormalInverseGamma {
    /// Construct a Normal-Inverse-Gamma prior.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if any of κ, α, β is not strictly
    /// positive and finite, or μ is not finite.
    pub fn new(mu: f64, kappa: f64, alpha: f64, beta: f64) -> BayesResult<Self> {
        if !mu.is_finite() {
            return Err(BayesError::InvalidConfig("mu must be finite".to_string()));
        }
        for (name, v) in [("kappa", kappa), ("alpha", alpha), ("beta", beta)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(BayesError::InvalidConfig(format!(
                    "{name} must be positive and finite, got {v}"
                )));
            }
        }
        Ok(Self {
            mu,
            kappa,
            alpha,
            beta,
        })
    }

    /// Update with Gaussian observations of unknown mean and variance.
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if `data` is empty.
    pub fn update(&self, data: &[f64]) -> BayesResult<Self> {
        if data.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        let n = data.len() as f64;
        let xbar = data.iter().sum::<f64>() / n;
        let ss: f64 = data.iter().map(|&x| (x - xbar).powi(2)).sum();

        let kappa_n = self.kappa + n;
        let mu_n = (self.kappa * self.mu + n * xbar) / kappa_n;
        let alpha_n = self.alpha + n / 2.0;
        let beta_n =
            self.beta + 0.5 * ss + 0.5 * (self.kappa * n / kappa_n) * (xbar - self.mu).powi(2);
        Ok(Self {
            mu: mu_n,
            kappa: kappa_n,
            alpha: alpha_n,
            beta: beta_n,
        })
    }

    /// Posterior mean of the data mean μ (= `mu`).
    #[must_use]
    pub fn mean_of_mean(&self) -> f64 {
        self.mu
    }

    /// Posterior mean of the variance σ²: `β / (α − 1)` for `α > 1`.
    #[must_use]
    pub fn mean_of_variance(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// Posterior-predictive of a single new observation: a Student-t with
    /// `2α` degrees of freedom, location `μ`, and scale²
    /// `β(κ+1) / (ακ)`.  Returns `(location, scale², dof)`.
    #[must_use]
    pub fn predictive(&self) -> (f64, f64, f64) {
        let dof = 2.0 * self.alpha;
        let scale_sq = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa);
        (self.mu, scale_sq, dof)
    }

    /// Log-density of the posterior-predictive Student-t at `x`.
    #[must_use]
    pub fn predictive_logpdf(&self, x: f64) -> f64 {
        let (loc, scale_sq, dof) = self.predictive();
        let scale = scale_sq.sqrt();
        let z = (x - loc) / scale;
        // log t-pdf: lgamma((ν+1)/2) − lgamma(ν/2) − ½log(νπ) − log scale
        //            − (ν+1)/2 · log(1 + z²/ν).
        lgamma((dof + 1.0) / 2.0)
            - lgamma(dof / 2.0)
            - 0.5 * (dof * std::f64::consts::PI).ln()
            - scale.ln()
            - (dof + 1.0) / 2.0 * (1.0 + z * z / dof).ln()
    }
}

// ─── Dirichlet-Multinomial ──────────────────────────────────────────────────

/// Dirichlet(`alpha`) conjugate prior for categorical / multinomial counts.
#[derive(Debug, Clone, PartialEq)]
pub struct DirichletPosterior {
    /// Concentration parameters α (one per category, each > 0).
    pub alpha: Vec<f64>,
}

impl DirichletPosterior {
    /// Construct a Dirichlet prior.
    ///
    /// # Errors
    /// Returns [`BayesError::EmptyInputs`] if `alpha` is empty, and
    /// [`BayesError::InvalidConfig`] if any concentration is not strictly
    /// positive and finite.
    pub fn new(alpha: Vec<f64>) -> BayesResult<Self> {
        if alpha.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if alpha.iter().any(|&a| !(a.is_finite() && a > 0.0)) {
            return Err(BayesError::InvalidConfig(
                "all Dirichlet concentrations must be positive and finite".to_string(),
            ));
        }
        Ok(Self { alpha })
    }

    /// Update with category `counts` (same length as `alpha`).
    ///
    /// # Errors
    /// Returns [`BayesError::DimensionMismatch`] if `counts.len() != alpha.len()`.
    pub fn update(&self, counts: &[u64]) -> BayesResult<Self> {
        if counts.len() != self.alpha.len() {
            return Err(BayesError::DimensionMismatch {
                expected: self.alpha.len(),
                got: counts.len(),
            });
        }
        let alpha = self
            .alpha
            .iter()
            .zip(counts.iter())
            .map(|(&a, &c)| a + c as f64)
            .collect();
        Ok(Self { alpha })
    }

    /// Concentration sum `α₀ = Σ αₖ`.
    #[must_use]
    pub fn alpha_sum(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Posterior mean of the category probabilities `αₖ / α₀`.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        let s = self.alpha_sum();
        self.alpha.iter().map(|&a| a / s).collect()
    }

    /// Posterior-predictive probability of drawing category `k` next.
    ///
    /// Equals the posterior mean of `pₖ`.
    ///
    /// # Errors
    /// Returns [`BayesError::InvalidConfig`] if `k` is out of range.
    pub fn predictive_category(&self, k: usize) -> BayesResult<f64> {
        if k >= self.alpha.len() {
            return Err(BayesError::InvalidConfig(format!(
                "category {k} out of range {}",
                self.alpha.len()
            )));
        }
        Ok(self.alpha[k] / self.alpha_sum())
    }

    /// Posterior-predictive log-PMF of a multinomial count vector `counts`
    /// (with `n = Σ counts` total draws) — the Dirichlet-Multinomial.
    ///
    /// # Errors
    /// Returns [`BayesError::DimensionMismatch`] on length mismatch.
    pub fn predictive_logpmf(&self, counts: &[u64]) -> BayesResult<f64> {
        if counts.len() != self.alpha.len() {
            return Err(BayesError::DimensionMismatch {
                expected: self.alpha.len(),
                got: counts.len(),
            });
        }
        let n: u64 = counts.iter().sum();
        let a0 = self.alpha_sum();
        // log n! + log Γ(α₀) − log Γ(n+α₀)
        //   + Σ [ log Γ(cₖ+αₖ) − log Γ(αₖ) − log cₖ! ].
        let mut acc = lgamma(n as f64 + 1.0) + lgamma(a0) - lgamma(n as f64 + a0);
        for (&c, &a) in counts.iter().zip(self.alpha.iter()) {
            acc += lgamma(c as f64 + a) - lgamma(a) - lgamma(c as f64 + 1.0);
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_posterior_conjugate_update() {
        let prior = BetaPosterior::new(1.0, 1.0).unwrap(); // uniform
        let post = prior.update(7, 10).unwrap();
        assert_eq!(post, BetaPosterior::new(8.0, 4.0).unwrap());
        // Posterior mean = 8/12 ≈ 0.6667.
        assert!((post.mean() - 8.0 / 12.0).abs() < 1e-12);
        // Mode = (8-1)/(12-2) = 0.7.
        assert!((post.mode() - 0.7).abs() < 1e-12);
        assert!(post.variance() > 0.0);
        // Invalid: successes > trials.
        assert!(prior.update(11, 10).is_err());
    }

    #[test]
    fn beta_predictive_pmf_sums_to_one() {
        let post = BetaPosterior::new(2.0, 3.0).unwrap();
        let n = 6;
        let total: f64 = (0..=n).map(|k| post.predictive_pmf(k, n).unwrap()).sum();
        assert!((total - 1.0).abs() < 1e-9, "sum {total}");
    }

    #[test]
    fn beta_predictive_uniform_prior_is_uniform_over_successes() {
        // With Beta(1,1) the predictive over k in {0..n} is uniform (1/(n+1)).
        let post = BetaPosterior::new(1.0, 1.0).unwrap();
        let n = 5;
        for k in 0..=n {
            let p = post.predictive_pmf(k, n).unwrap();
            assert!((p - 1.0 / (n as f64 + 1.0)).abs() < 1e-9, "k={k} p={p}");
        }
    }

    #[test]
    fn beta_invalid_shapes_rejected() {
        assert!(BetaPosterior::new(0.0, 1.0).is_err());
        assert!(BetaPosterior::new(1.0, -1.0).is_err());
        assert!(BetaPosterior::new(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn gamma_poisson_conjugate_update() {
        let prior = GammaPosterior::new(2.0, 1.0).unwrap();
        let data = [3, 1, 4, 1, 5];
        let post = prior.update(&data).unwrap();
        // shape += 14, rate += 5.
        assert!((post.shape - 16.0).abs() < 1e-12);
        assert!((post.rate - 6.0).abs() < 1e-12);
        assert!((post.mean() - 16.0 / 6.0).abs() < 1e-12);
        assert!(prior.update(&[]).is_err());
    }

    #[test]
    fn gamma_poisson_predictive_sums_to_one() {
        let post = GammaPosterior::new(3.0, 2.0).unwrap();
        // Negative-binomial PMF should sum (approximately) to 1 over a large
        // truncation.
        let total: f64 = (0..2000).map(|k| post.predictive_pmf(k)).sum();
        assert!((total - 1.0).abs() < 1e-6, "sum {total}");
    }

    #[test]
    fn normal_known_var_precision_weighted_update() {
        // Prior N(0, 1), obs var 4. One observation at y=10.
        let prior = NormalKnownVarPosterior::new(0.0, 1.0, 4.0).unwrap();
        let post = prior.update(&[10.0]).unwrap();
        // post prec = 1/1 + 1/4 = 1.25 → var = 0.8.
        assert!((post.variance - 0.8).abs() < 1e-12);
        // post mean = 0.8·(1·0 + 0.25·10) = 0.8·2.5 = 2.0.
        assert!((post.mean - 2.0).abs() < 1e-12);
        // Predictive variance = 0.8 + 4 = 4.8.
        let (pm, pv) = post.predictive();
        assert!((pm - 2.0).abs() < 1e-12 && (pv - 4.8).abs() < 1e-12);
    }

    #[test]
    fn normal_known_var_concentrates_with_data() {
        // Many observations should drive the posterior mean to the sample mean
        // and shrink its variance.
        let prior = NormalKnownVarPosterior::new(0.0, 100.0, 1.0).unwrap();
        let data: Vec<f64> = (0..1000).map(|i| 5.0 + ((i % 3) as f64 - 1.0)).collect();
        let xbar = data.iter().sum::<f64>() / data.len() as f64;
        let post = prior.update(&data).unwrap();
        assert!(
            (post.mean - xbar).abs() < 0.05,
            "mean {} xbar {xbar}",
            post.mean
        );
        assert!(post.variance < 0.01, "variance {}", post.variance);
        // Predictive log-pdf is finite at the mean.
        assert!(post.predictive_logpdf(post.mean).is_finite());
    }

    #[test]
    fn nig_recovers_known_mean_and_variance() {
        // Weak prior; data from N(3, 2²). Posterior point estimates should be
        // close to the empirical mean / variance.
        let prior = NormalInverseGamma::new(0.0, 0.01, 0.01, 0.01).unwrap();
        // Deterministic symmetric data with mean 3 and known spread.
        let mut data = Vec::new();
        for i in 0..500 {
            let t = (i as f64 / 499.0) * 2.0 - 1.0; // in [-1,1]
            data.push(3.0 + 2.0 * t);
        }
        let xbar = data.iter().sum::<f64>() / data.len() as f64;
        let var_emp = data.iter().map(|&x| (x - xbar).powi(2)).sum::<f64>() / data.len() as f64;
        let post = prior.update(&data).unwrap();
        assert!(
            (post.mean_of_mean() - xbar).abs() < 0.01,
            "mean {}",
            post.mean_of_mean()
        );
        // Mean of variance ≈ empirical variance (within a few %).
        let mv = post.mean_of_variance();
        assert!(
            (mv - var_emp).abs() / var_emp < 0.05,
            "mean variance {mv} vs empirical {var_emp}"
        );
    }

    #[test]
    fn nig_predictive_is_valid_student_t() {
        let post = NormalInverseGamma::new(1.0, 4.0, 5.0, 6.0).unwrap();
        let (loc, scale_sq, dof) = post.predictive();
        assert!((loc - 1.0).abs() < 1e-12);
        assert!(scale_sq > 0.0 && dof > 0.0);
        // Symmetric: logpdf(loc+a) == logpdf(loc-a).
        let a = 0.7;
        let lp = post.predictive_logpdf(loc + a);
        let lm = post.predictive_logpdf(loc - a);
        assert!((lp - lm).abs() < 1e-12);
        // Peak at the location.
        assert!(post.predictive_logpdf(loc) > lp);
    }

    #[test]
    fn nig_invalid_params_rejected() {
        assert!(NormalInverseGamma::new(0.0, 0.0, 1.0, 1.0).is_err());
        assert!(NormalInverseGamma::new(0.0, 1.0, -1.0, 1.0).is_err());
        assert!(NormalInverseGamma::new(f64::INFINITY, 1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn dirichlet_multinomial_conjugate_update() {
        let prior = DirichletPosterior::new(vec![1.0, 1.0, 1.0]).unwrap();
        let post = prior.update(&[10, 20, 70]).unwrap();
        assert_eq!(post.alpha, vec![11.0, 21.0, 71.0]);
        let mean = post.mean();
        let s = post.alpha_sum();
        assert!((mean[2] - 71.0 / s).abs() < 1e-12);
        // Predictive of the dominant category is the largest.
        assert!(post.predictive_category(2).unwrap() > post.predictive_category(0).unwrap());
        assert!(post.update(&[1, 2]).is_err()); // wrong length
    }

    #[test]
    fn dirichlet_predictive_logpmf_normalises() {
        // For a single draw (n=1), the predictive over categories must sum to 1.
        let post = DirichletPosterior::new(vec![2.0, 3.0, 5.0]).unwrap();
        let k = post.alpha.len();
        let mut total = 0.0;
        for c in 0..k {
            let mut counts = vec![0u64; k];
            counts[c] = 1;
            total += post.predictive_logpmf(&counts).unwrap().exp();
        }
        assert!((total - 1.0).abs() < 1e-9, "sum {total}");
        // Single-draw predictive equals the category predictive.
        let mut counts = vec![0u64; k];
        counts[1] = 1;
        let pmf = post.predictive_logpmf(&counts).unwrap().exp();
        assert!((pmf - post.predictive_category(1).unwrap()).abs() < 1e-9);
    }

    #[test]
    fn dirichlet_invalid_inputs_rejected() {
        assert!(DirichletPosterior::new(vec![]).is_err());
        assert!(DirichletPosterior::new(vec![1.0, -1.0]).is_err());
        let post = DirichletPosterior::new(vec![1.0, 1.0]).unwrap();
        assert!(post.predictive_category(5).is_err());
    }

    #[test]
    fn log_beta_matches_known_value() {
        // B(1,1) = 1 → log B = 0; B(2,3) = 1/12 → log = −log 12.
        assert!(log_beta(1.0, 1.0).abs() < 1e-12);
        assert!((log_beta(2.0, 3.0) - (1.0_f64 / 12.0).ln()).abs() < 1e-10);
    }
}
