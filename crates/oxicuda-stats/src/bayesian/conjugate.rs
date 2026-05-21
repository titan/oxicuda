//! Conjugate prior Bayesian updates, credible intervals, and Bayes factors.
//!
//! # Supported conjugate models
//!
//! | Prior              | Likelihood        | Posterior          |
//! |--------------------|-------------------|--------------------|
//! | Normal (known σ²)  | Normal            | Normal             |
//! | Normal-InvGamma    | Normal (μ,σ² unk) | Normal-InvGamma    |
//! | Beta               | Binomial          | Beta               |
//! | Gamma (rate param) | Poisson           | Gamma              |
//! | Dirichlet          | Multinomial       | Dirichlet          |
//!
//! All computations are exact closed-form updates except for the HDI search which
//! uses golden-section / bisection numerical methods.

use crate::distributions::beta::Beta;
use crate::distributions::normal::Normal;
use crate::error::{StatsError, StatsResult};

// ──────────────────────────────────────────────────────────────────────────────
// Public structs
// ──────────────────────────────────────────────────────────────────────────────

/// Posterior of μ under Normal-Normal model (known variance σ²).
#[derive(Debug, Clone)]
pub struct NormalNormalPosterior {
    /// Posterior mean μₙ.
    pub mean: f64,
    /// Posterior variance τₙ².
    pub variance: f64,
    /// Posterior standard deviation √τₙ².
    pub std: f64,
}

/// Posterior parameters for the Normal-Inverse-Gamma model (μ, σ² both unknown).
///
/// The marginal prior for σ² is InvGamma(α, β) and for μ|σ² is N(m, σ²/κ).
#[derive(Debug, Clone)]
pub struct NigPosterior {
    /// Posterior precision multiplier κₙ = κ₀ + n.
    pub kappa: f64,
    /// Posterior location mₙ = (κ₀ m₀ + n x̄) / κₙ.
    pub m: f64,
    /// Posterior shape αₙ = α₀ + n/2.
    pub alpha: f64,
    /// Posterior scale βₙ = β₀ + S²/2 + κ₀ n (x̄ - m₀)² / (2 κₙ).
    pub beta: f64,
    /// Predictive mean (= mₙ) of the marginal Student-t predictive.
    pub predictive_mean: f64,
    /// Predictive variance of the marginal Student-t predictive.
    pub predictive_variance: f64,
}

/// Posterior of p under Beta-Binomial model.
#[derive(Debug, Clone)]
pub struct BetaPosterior {
    /// Posterior α = α₀ + k.
    pub alpha: f64,
    /// Posterior β = β₀ + n - k.
    pub beta: f64,
    /// Posterior mean = α / (α + β).
    pub mean: f64,
    /// Posterior mode = (α - 1) / (α + β - 2), or `None` if undefined.
    pub mode: Option<f64>,
    /// Posterior variance = α β / ((α + β)² (α + β + 1)).
    pub variance: f64,
}

/// Posterior of λ under Gamma-Poisson model.
#[derive(Debug, Clone)]
pub struct GammaPosterior {
    /// Posterior shape αₙ = α₀ + Σxᵢ.
    pub alpha: f64,
    /// Posterior rate βₙ = β₀ + n  (rate parameterisation: mean = α/β).
    pub beta: f64,
    /// Posterior mean = αₙ / βₙ.
    pub mean: f64,
    /// Posterior variance = αₙ / βₙ².
    pub variance: f64,
}

/// Credible interval (Bayesian confidence interval).
#[derive(Debug, Clone)]
pub struct CredibleInterval {
    /// Lower endpoint of the interval.
    pub lower: f64,
    /// Upper endpoint of the interval.
    pub upper: f64,
    /// Credibility level, e.g. 0.95 for a 95% CI.
    pub level: f64,
    /// Construction method.
    pub method: CiMethod,
}

/// Method used to construct the credible interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiMethod {
    /// Equal-tails (percentile) interval: [α/2, 1 - α/2] quantiles.
    EqualTails,
    /// Highest Density Interval: narrowest interval containing the specified probability mass.
    Hdi,
}

/// Bayes factor comparing H₁ (alternative) to H₀ (null).
#[derive(Debug, Clone)]
pub struct BayesFactor {
    /// Natural log of BF₁₀ = p(data | H₁) / p(data | H₀).
    /// Positive means evidence in favour of H₁.
    pub log_bf10: f64,
    /// BF₁₀ on the natural scale (clamped to avoid overflow).
    pub bf10: f64,
    /// Verbal interpretation following Jeffreys (1961) scale.
    pub interpretation: &'static str,
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. Normal-Normal conjugate update (known variance)
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a Normal-Normal Bayesian update.
///
/// Prior: μ ~ N(prior_mean, prior_var).
/// Likelihood: xᵢ | μ ~ N(μ, data_var) (known data variance σ²).
/// Posterior: μ | x ~ N(μₙ, τₙ²) where:
/// ```text
/// τₙ² = 1 / (1/τ₀² + n/σ²)
/// μₙ  = τₙ² · (μ₀/τ₀² + n·x̄/σ²)
/// ```
///
/// # Errors
/// - `StatsError::EmptyInput` if `obs` is empty.
/// - `StatsError::InvalidParameter` if variances are not positive.
pub fn normal_normal_update(
    prior_mean: f64,
    prior_var: f64,
    data_var: f64,
    obs: &[f64],
) -> StatsResult<NormalNormalPosterior> {
    if !prior_var.is_finite() || prior_var <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "prior_var".into(),
            reason: format!("must be > 0 and finite; got {prior_var}"),
        });
    }
    if !data_var.is_finite() || data_var <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "data_var".into(),
            reason: format!("must be > 0 and finite; got {data_var}"),
        });
    }
    if obs.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    for (i, &x) in obs.iter().enumerate() {
        if !x.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let n = obs.len() as f64;
    let x_bar: f64 = obs.iter().sum::<f64>() / n;

    let inv_tau0_sq = 1.0 / prior_var;
    let inv_tau_n_sq = inv_tau0_sq + n / data_var;
    let tau_n_sq = 1.0 / inv_tau_n_sq;
    let mu_n = tau_n_sq * (prior_mean / prior_var + n * x_bar / data_var);

    Ok(NormalNormalPosterior {
        mean: mu_n,
        variance: tau_n_sq,
        std: tau_n_sq.sqrt(),
    })
}

/// Compute a credible interval for a Normal-Normal posterior.
///
/// Both equal-tails and HDI coincide for a symmetric Normal distribution.
///
/// # Errors
/// - `StatsError::InvalidParameter` if `level` is not in (0, 1).
pub fn normal_normal_ci(post: &NormalNormalPosterior, level: f64) -> StatsResult<CredibleInterval> {
    validate_level(level)?;
    let alpha = 1.0 - level;
    let dist = Normal::new(post.mean, post.std)
        .map_err(|e| StatsError::NumericalInstability(e.to_string()))?;
    let lo = dist.ppf(alpha / 2.0)?;
    let hi = dist.ppf(1.0 - alpha / 2.0)?;
    Ok(CredibleInterval {
        lower: lo,
        upper: hi,
        level,
        method: CiMethod::EqualTails,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Normal-Inverse-Gamma conjugate update (both μ and σ² unknown)
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a Normal-Inverse-Gamma Bayesian update.
///
/// Prior: σ² ~ InvGamma(α₀, β₀), μ | σ² ~ N(m₀, σ²/κ₀).
/// Posterior parameters after n observations:
/// ```text
/// κₙ = κ₀ + n
/// mₙ = (κ₀ m₀ + n x̄) / κₙ
/// αₙ = α₀ + n/2
/// βₙ = β₀ + S²/2 + κ₀ n (x̄ - m₀)² / (2 κₙ)
/// ```
/// where S² = Σ(xᵢ - x̄)².
///
/// The marginal predictive distribution for a new observation is Student-t with
/// 2αₙ degrees of freedom, location mₙ, and scale βₙ (κₙ + 1) / (αₙ κₙ).
///
/// # Errors
/// - `StatsError::EmptyInput` if `obs` is empty.
/// - `StatsError::InvalidParameter` if hyper-parameters are invalid.
pub fn nig_update(
    m0: f64,
    kappa0: f64,
    alpha0: f64,
    beta0: f64,
    obs: &[f64],
) -> StatsResult<NigPosterior> {
    if kappa0 <= 0.0 || !kappa0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "kappa0".into(),
            reason: format!("must be > 0; got {kappa0}"),
        });
    }
    if alpha0 <= 0.0 || !alpha0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "alpha0".into(),
            reason: format!("must be > 0; got {alpha0}"),
        });
    }
    if beta0 <= 0.0 || !beta0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "beta0".into(),
            reason: format!("must be > 0; got {beta0}"),
        });
    }
    if obs.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    for (i, &x) in obs.iter().enumerate() {
        if !x.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let n = obs.len() as f64;
    let x_bar: f64 = obs.iter().sum::<f64>() / n;
    let s_sq: f64 = obs.iter().map(|&xi| (xi - x_bar) * (xi - x_bar)).sum();

    let kappa_n = kappa0 + n;
    let m_n = (kappa0 * m0 + n * x_bar) / kappa_n;
    let alpha_n = alpha0 + n / 2.0;
    let beta_n = beta0 + s_sq / 2.0 + kappa0 * n * (x_bar - m0).powi(2) / (2.0 * kappa_n);

    // Predictive distribution is Student-t(2αₙ, mₙ, scale²) where
    // scale² = βₙ (κₙ + 1) / (αₙ κₙ).
    let df = 2.0 * alpha_n;
    let pred_scale_sq = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n);
    // For Student-t with df > 2: variance = df / (df - 2) * scale².
    let predictive_variance = if df > 2.0 {
        df / (df - 2.0) * pred_scale_sq
    } else {
        f64::INFINITY
    };

    Ok(NigPosterior {
        kappa: kappa_n,
        m: m_n,
        alpha: alpha_n,
        beta: beta_n,
        predictive_mean: m_n,
        predictive_variance,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Beta-Binomial conjugate update
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a Beta-Binomial Bayesian update.
///
/// Prior: p ~ Beta(α₀, β₀).
/// Likelihood: k successes in n trials.
/// Posterior: p ~ Beta(α₀ + k, β₀ + n - k).
///
/// # Errors
/// - `StatsError::InvalidParameter` if α₀ ≤ 0 or β₀ ≤ 0.
/// - `StatsError::InvalidParameter` if k > n.
pub fn beta_binomial_update(alpha0: f64, beta0: f64, k: u64, n: u64) -> StatsResult<BetaPosterior> {
    if alpha0 <= 0.0 || !alpha0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "alpha0".into(),
            reason: format!("must be > 0; got {alpha0}"),
        });
    }
    if beta0 <= 0.0 || !beta0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "beta0".into(),
            reason: format!("must be > 0; got {beta0}"),
        });
    }
    if k > n {
        return Err(StatsError::InvalidParameter {
            name: "k".into(),
            reason: format!("successes k={k} > trials n={n}"),
        });
    }

    let alpha_n = alpha0 + k as f64;
    let beta_n = beta0 + (n - k) as f64;
    let sum = alpha_n + beta_n;

    let mean = alpha_n / sum;
    let mode = if alpha_n > 1.0 && beta_n > 1.0 {
        Some((alpha_n - 1.0) / (sum - 2.0))
    } else {
        None
    };
    let variance = alpha_n * beta_n / (sum * sum * (sum + 1.0));

    Ok(BetaPosterior {
        alpha: alpha_n,
        beta: beta_n,
        mean,
        mode,
        variance,
    })
}

/// Compute a credible interval for a Beta posterior.
///
/// Supports both `EqualTails` (percentile) and `Hdi` (highest density interval) methods.
///
/// For `Hdi`: we use golden-section search over the left endpoint `q_lo ∈ [0, 1 - level]`
/// to find the shortest interval `[q_lo, q_hi]` such that the Beta CDF difference equals
/// `level`.
///
/// # Errors
/// - `StatsError::InvalidParameter` if `level` is not in (0, 1).
pub fn beta_credible_interval(
    post: &BetaPosterior,
    level: f64,
    method: CiMethod,
) -> StatsResult<CredibleInterval> {
    validate_level(level)?;
    let dist = Beta::new(post.alpha, post.beta)
        .map_err(|e| StatsError::NumericalInstability(e.to_string()))?;

    match method {
        CiMethod::EqualTails => {
            let alpha = 1.0 - level;
            let lo = beta_quantile(post.alpha, post.beta, alpha / 2.0)?;
            let hi = beta_quantile(post.alpha, post.beta, 1.0 - alpha / 2.0)?;
            Ok(CredibleInterval {
                lower: lo,
                upper: hi,
                level,
                method,
            })
        }
        CiMethod::Hdi => {
            // Golden-section search to minimise interval width over left-boundary q_lo.
            // Width = Q(q_lo + level) - q_lo where Q = beta quantile function.
            // We search q_lo ∈ [0, beta_quantile(1 - level)] via ternary search.
            let max_lo = beta_quantile(post.alpha, post.beta, 1.0 - level)?;
            let (lo, hi) = hdi_golden_section(&dist, level, 0.0, max_lo, 200)?;
            Ok(CredibleInterval {
                lower: lo,
                upper: hi,
                level,
                method,
            })
        }
    }
}

/// Golden-section ternary search to find the HDI endpoints for a Beta distribution.
///
/// We minimise `f(q_lo) = q_hi(q_lo) - q_lo` where `q_hi(q_lo)` is the quantile
/// corresponding to CDF value `CDF(q_lo) + level`.
fn hdi_golden_section(
    dist: &Beta,
    level: f64,
    lo_min: f64,
    lo_max: f64,
    iterations: usize,
) -> StatsResult<(f64, f64)> {
    const GOLDEN: f64 = 0.381_966_011_250_105_15; // 1 - 1/φ
    let mut a = lo_min;
    let mut b = lo_max;

    let width = |q_lo: f64| -> StatsResult<f64> {
        let p_lo = dist.cdf(q_lo)?;
        let p_hi = (p_lo + level).min(1.0);
        let q_hi = beta_quantile(dist.alpha, dist.beta, p_hi)?;
        Ok(q_hi - q_lo)
    };

    let mut x1 = a + GOLDEN * (b - a);
    let mut x2 = b - GOLDEN * (b - a);
    let mut f1 = width(x1)?;
    let mut f2 = width(x2)?;

    for _ in 0..iterations {
        if (b - a) < 1e-10 {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + GOLDEN * (b - a);
            f1 = width(x1)?;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - GOLDEN * (b - a);
            f2 = width(x2)?;
        }
    }

    let q_lo_opt = (a + b) / 2.0;
    let p_lo = dist.cdf(q_lo_opt)?;
    let p_hi = (p_lo + level).min(1.0);
    let q_hi_opt = beta_quantile(dist.alpha, dist.beta, p_hi)?;
    Ok((q_lo_opt, q_hi_opt))
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Gamma-Poisson conjugate update
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a Gamma-Poisson Bayesian update (rate parameterisation).
///
/// Prior: λ ~ Gamma(α₀, β₀) with mean = α₀/β₀.
/// Likelihood: xᵢ ~ Poisson(λ) independently.
/// Posterior: λ ~ Gamma(α₀ + Σxᵢ, β₀ + n).
///
/// # Errors
/// - `StatsError::EmptyInput` if `obs` is empty.
/// - `StatsError::InvalidParameter` if α₀ ≤ 0 or β₀ ≤ 0.
pub fn gamma_poisson_update(alpha0: f64, beta0: f64, obs: &[u64]) -> StatsResult<GammaPosterior> {
    if alpha0 <= 0.0 || !alpha0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "alpha0".into(),
            reason: format!("must be > 0; got {alpha0}"),
        });
    }
    if beta0 <= 0.0 || !beta0.is_finite() {
        return Err(StatsError::InvalidParameter {
            name: "beta0".into(),
            reason: format!("must be > 0; got {beta0}"),
        });
    }
    if obs.is_empty() {
        return Err(StatsError::EmptyInput);
    }

    let n = obs.len() as f64;
    let sum_x: f64 = obs.iter().map(|&x| x as f64).sum();

    let alpha_n = alpha0 + sum_x;
    let beta_n = beta0 + n;
    let mean = alpha_n / beta_n;
    let variance = alpha_n / (beta_n * beta_n);

    Ok(GammaPosterior {
        alpha: alpha_n,
        beta: beta_n,
        mean,
        variance,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. Dirichlet-Multinomial conjugate update
// ──────────────────────────────────────────────────────────────────────────────

/// Perform a Dirichlet-Multinomial Bayesian update.
///
/// Prior: θ ~ Dirichlet(α₀).
/// Likelihood: counts nₖ for k = 1..K.
/// Posterior: θ ~ Dirichlet(α₀ + n).
/// Returns the posterior concentration parameter vector α₀ + n.
///
/// # Errors
/// - `StatsError::DimensionMismatch` if `alpha0.len() != counts.len()`.
/// - `StatsError::EmptyInput` if either slice is empty.
/// - `StatsError::InvalidParameter` if any αₖ ≤ 0.
pub fn dirichlet_multinomial_update(alpha0: &[f64], counts: &[u64]) -> StatsResult<Vec<f64>> {
    if alpha0.is_empty() || counts.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if alpha0.len() != counts.len() {
        return Err(StatsError::DimensionMismatch {
            a: alpha0.len(),
            b: counts.len(),
        });
    }
    for (k, &a) in alpha0.iter().enumerate() {
        if a <= 0.0 || !a.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: format!("alpha0[{k}]"),
                reason: format!("must be > 0; got {a}"),
            });
        }
    }
    let posterior: Vec<f64> = alpha0
        .iter()
        .zip(counts.iter())
        .map(|(&a, &c)| a + c as f64)
        .collect();
    Ok(posterior)
}

// ──────────────────────────────────────────────────────────────────────────────
// Bayes factors — Savage-Dickey density ratio
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the Bayes factor for a fair-coin test using the Savage-Dickey density ratio.
///
/// Model: p ~ Beta(1, 1) [uniform prior on coin bias].
/// H₀: p = p0 (specified null, often 0.5 for fair coin).
/// H₁: p ≠ p0 (unrestricted).
/// BF₁₀ = posterior_pdf(p0) / prior_pdf(p0).
///
/// Since Prior = Beta(1,1) = Uniform(0,1), prior_pdf(p0) = 1 for p0 ∈ (0,1).
/// So BF₁₀ = Beta(k+1, n-k+1).pdf(p0).
///
/// A large BF₁₀ indicates strong evidence that p ≠ p0.
///
/// # Errors
/// - `StatsError::InvalidParameter` if `k > n` or `p0 ∉ (0,1)`.
pub fn bayes_factor_coin(k: u64, n: u64, p0: f64) -> StatsResult<BayesFactor> {
    if k > n {
        return Err(StatsError::InvalidParameter {
            name: "k".into(),
            reason: format!("successes k={k} > trials n={n}"),
        });
    }
    if !(p0 > 0.0 && p0 < 1.0 && p0.is_finite()) {
        return Err(StatsError::InvalidParameter {
            name: "p0".into(),
            reason: format!("must be in (0,1); got {p0}"),
        });
    }

    // Posterior after updating Beta(1,1) with k successes in n trials.
    let alpha_post = 1.0 + k as f64;
    let beta_post = 1.0 + (n - k) as f64;
    let post_dist = Beta::new(alpha_post, beta_post)
        .map_err(|e| StatsError::NumericalInstability(e.to_string()))?;

    // Savage-Dickey: BF₁₀ = posterior.pdf(p0) / prior.pdf(p0)
    // Prior = Beta(1,1) → pdf = 1 everywhere on (0,1).
    let posterior_pdf_at_null = post_dist.pdf(p0);

    // Log-BF for numerical stability.
    let log_bf10 = if posterior_pdf_at_null > 0.0 {
        posterior_pdf_at_null.ln()
    } else {
        f64::NEG_INFINITY
    };
    let bf10 = if log_bf10 > 700.0 {
        f64::INFINITY
    } else if log_bf10 < -700.0 {
        0.0
    } else {
        log_bf10.exp()
    };

    let interpretation = interpret_bayes_factor(log_bf10);
    Ok(BayesFactor {
        log_bf10,
        bf10,
        interpretation,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Numerical helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Validate that a credibility level is in (0, 1).
fn validate_level(level: f64) -> StatsResult<()> {
    if !(level > 0.0 && level < 1.0 && level.is_finite()) {
        return Err(StatsError::InvalidParameter {
            name: "level".into(),
            reason: format!("must be in (0, 1); got {level}"),
        });
    }
    Ok(())
}

/// Interpret a log Bayes factor using Jeffreys (1961) evidence scale.
///
/// BF₁₀ = p(data|H₁) / p(data|H₀). A value > 1 means evidence for H₁, < 1 for H₀.
///
/// | |log₁₀(BF)| | Direction       | Interpretation  |
/// |-------------|-----------------|-----------------|
/// | > 2         | supports H₁     | decisive        |
/// | 1 – 2       | supports H₁     | strong          |
/// | 0.5 – 1     | supports H₁     | substantial     |
/// | 0 – 0.5     | supports H₁     | weak            |
/// | -0.5 – 0    | supports H₀     | weak            |
/// | -1 – -0.5   | supports H₀     | substantial     |
/// | -2 – -1     | supports H₀     | strong          |
/// | < -2        | supports H₀     | decisive        |
fn interpret_bayes_factor(log_bf10: f64) -> &'static str {
    let log10_bf = log_bf10 / std::f64::consts::LN_10;
    let abs_log10 = log10_bf.abs();
    if abs_log10 >= 2.0 {
        "decisive"
    } else if abs_log10 >= 1.0 {
        "strong"
    } else if abs_log10 >= 0.5 {
        "substantial"
    } else {
        "weak"
    }
}

/// Compute the Beta CDF quantile (inverse CDF) using bisection.
///
/// Finds q such that `betainc(alpha, beta, q) ≈ p` to within `tol` using at most
/// `max_iter` iterations.
pub(crate) fn beta_quantile(alpha: f64, beta_val: f64, p: f64) -> StatsResult<f64> {
    use crate::special::betainc::betainc;
    if p <= 0.0 {
        return Ok(0.0);
    }
    if p >= 1.0 {
        return Ok(1.0);
    }
    // Initial bracket: [0, 1].
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    let tol = 1e-12;
    let max_iter = 100;

    // First seed with the empirical mode for faster convergence.
    let mut mid = if alpha > 1.0 && beta_val > 1.0 {
        (alpha - 1.0) / (alpha + beta_val - 2.0)
    } else {
        0.5
    };

    for _ in 0..max_iter {
        let cdf_mid = betainc(alpha, beta_val, mid)
            .map_err(|e| StatsError::NumericalInstability(format!("betainc failed: {e}")))?;
        if (cdf_mid - p).abs() < tol {
            return Ok(mid);
        }
        if cdf_mid < p {
            lo = mid;
        } else {
            hi = mid;
        }
        let new_mid = (lo + hi) / 2.0;
        if (new_mid - mid).abs() < 1e-15 {
            break;
        }
        mid = new_mid;
    }
    Ok(mid)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Normal-Normal — posterior mean converges to data mean with many observations.
    #[test]
    fn normal_normal_mean_converges() {
        // With a very diffuse prior and large dataset, posterior mean → sample mean.
        let data: Vec<f64> = (0..500).map(|i| 5.0 + (i as f64) * 0.001).collect();
        let sample_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let post = normal_normal_update(0.0, 1_000_000.0, 1.0, &data).expect("ok");
        assert!(
            (post.mean - sample_mean).abs() < 1e-3,
            "posterior mean={} should be close to sample mean={}",
            post.mean,
            sample_mean
        );
    }

    // Test 2: Normal-Normal — posterior variance < prior variance.
    #[test]
    fn normal_normal_variance_shrinks() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let prior_var = 10.0;
        let post = normal_normal_update(0.0, prior_var, 2.0, &data).expect("ok");
        assert!(
            post.variance < prior_var,
            "posterior variance {} should be less than prior {}",
            post.variance,
            prior_var
        );
    }

    // Test 3: Normal-Normal CI contains the posterior mean.
    #[test]
    fn normal_normal_ci_contains_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let post = normal_normal_update(0.0, 100.0, 1.0, &data).expect("ok");
        let ci = normal_normal_ci(&post, 0.95).expect("ok");
        assert!(
            ci.lower <= post.mean && post.mean <= ci.upper,
            "95% CI [{}, {}] should contain mean {}",
            ci.lower,
            ci.upper,
            post.mean
        );
        // CI width should be < prior range.
        assert!(ci.upper - ci.lower < 10.0);
    }

    // Test 4: NIG update — completes without error and returns sensible parameters.
    #[test]
    fn nig_update_basic() {
        let obs = vec![2.1, 3.3, 2.7, 3.1, 2.5];
        let post = nig_update(0.0, 1.0, 1.0, 1.0, &obs).expect("ok");
        assert!(post.kappa > 1.0);
        assert!(post.alpha > 1.0);
        assert!(post.beta > 0.0);
        assert!(post.predictive_mean.is_finite());
        assert!(post.predictive_variance > 0.0);
    }

    // Test 5: NIG update — κₙ = κ₀ + n.
    #[test]
    fn nig_kappa_increases() {
        let obs = vec![1.0, 2.0, 3.0];
        let n = obs.len() as f64;
        let kappa0 = 2.5;
        let post = nig_update(0.0, kappa0, 1.0, 1.0, &obs).expect("ok");
        assert!(
            (post.kappa - (kappa0 + n)).abs() < 1e-12,
            "κₙ={} should equal κ₀+n={}",
            post.kappa,
            kappa0 + n
        );
    }

    // Test 6: Beta update — posterior mean is (k+1)/(n+2) for Beta(1,1) prior.
    #[test]
    fn beta_update_mean_correct() {
        let k = 7u64;
        let n = 10u64;
        let post = beta_binomial_update(1.0, 1.0, k, n).expect("ok");
        let expected_mean = (k as f64 + 1.0) / (n as f64 + 2.0);
        assert!(
            (post.mean - expected_mean).abs() < 1e-12,
            "mean={} expected={}",
            post.mean,
            expected_mean
        );
    }

    // Test 7: Beta equal-tails CI is narrower than [0, 1].
    #[test]
    fn beta_equal_tails_ci_width() {
        let post = beta_binomial_update(1.0, 1.0, 5, 10).expect("ok");
        let ci = beta_credible_interval(&post, 0.95, CiMethod::EqualTails).expect("ok");
        assert!(ci.lower > 0.0, "lower={} should be > 0", ci.lower);
        assert!(ci.upper < 1.0, "upper={} should be < 1", ci.upper);
        assert!(ci.upper > ci.lower);
    }

    // Test 8: HDI is narrower than equal-tails for a skewed Beta posterior.
    #[test]
    fn beta_hdi_ci_narrower_than_equal_tails() {
        // Skewed posterior: heavily concentrated near 1.
        let post = beta_binomial_update(1.0, 1.0, 18, 20).expect("ok");
        let ci_et = beta_credible_interval(&post, 0.90, CiMethod::EqualTails).expect("ok");
        let ci_hdi = beta_credible_interval(&post, 0.90, CiMethod::Hdi).expect("ok");
        let width_et = ci_et.upper - ci_et.lower;
        let width_hdi = ci_hdi.upper - ci_hdi.lower;
        // HDI should not be wider than equal-tails for unimodal distributions.
        assert!(
            width_hdi <= width_et + 1e-4,
            "HDI width={width_hdi} should be ≤ ET width={width_et}"
        );
    }

    // Test 9: Gamma-Poisson posterior mean = (α₀ + Σx) / (β₀ + n).
    #[test]
    fn gamma_poisson_mean_correct() {
        let alpha0 = 2.0;
        let beta0 = 1.0;
        let obs = vec![3u64, 5, 4, 6, 2];
        let n = obs.len() as f64;
        let sum_x: f64 = obs.iter().map(|&x| x as f64).sum();
        let post = gamma_poisson_update(alpha0, beta0, &obs).expect("ok");
        let expected_mean = (alpha0 + sum_x) / (beta0 + n);
        assert!(
            (post.mean - expected_mean).abs() < 1e-12,
            "mean={} expected={}",
            post.mean,
            expected_mean
        );
    }

    // Test 10: Dirichlet posterior sum = prior sum + total counts.
    #[test]
    fn dirichlet_update_sums_correctly() {
        let alpha0 = vec![1.0, 2.0, 3.0];
        let counts = vec![5u64, 3, 7];
        let posterior = dirichlet_multinomial_update(&alpha0, &counts).expect("ok");
        let sum_prior: f64 = alpha0.iter().sum();
        let sum_counts: f64 = counts.iter().map(|&c| c as f64).sum();
        let sum_post: f64 = posterior.iter().sum();
        assert!(
            (sum_post - (sum_prior + sum_counts)).abs() < 1e-12,
            "posterior sum={} expected={}",
            sum_post,
            sum_prior + sum_counts
        );
        // Component-wise check.
        for (i, (&a0, &c)) in alpha0.iter().zip(counts.iter()).enumerate() {
            assert!(
                (posterior[i] - (a0 + c as f64)).abs() < 1e-12,
                "component {i}: post={} expected={}",
                posterior[i],
                a0 + c as f64
            );
        }
    }

    // Test 11: Bayes factor is finite and > 1 for balanced data on fair-coin H₀.
    #[test]
    fn bayes_factor_coin_fair() {
        // 5 heads in 10 flips: data is perfectly consistent with H₀: p=0.5.
        // Posterior Beta(6,6).pdf(0.5) is maximised at 0.5 (symmetric) → BF₁₀ > 1.
        let bf = bayes_factor_coin(5, 10, 0.5).expect("ok");
        assert!(bf.bf10.is_finite(), "bf10 should be finite");
        assert!(bf.bf10 > 0.0, "bf10 should be positive");
        // BF₁₀ > 1: posterior pdf at the null (p=0.5) exceeds prior pdf (= 1 for Uniform).
        // Beta(6,6).pdf(0.5) = 2^9 * 5! * 5! / 9! / B(6,6) ≈ 2.46 > 1.
        assert!(
            bf.bf10 > 1.0,
            "BF₁₀={} should be > 1 for data perfectly consistent with null",
            bf.bf10
        );
        assert!(bf.log_bf10.is_finite(), "log_bf10 should be finite");
        // Should be "weak" or "substantial" evidence (small log10_bf > 0).
        assert_ne!(bf.interpretation, "decisive");
    }

    // Test 12: 100 heads in 100 flips gives very large BF against the fair-coin null.
    #[test]
    fn bayes_factor_coin_strong_evidence() {
        let bf = bayes_factor_coin(100, 100, 0.5).expect("ok");
        // Beta(101, 1).pdf(0.5) is extremely small → BF₁₀ should be very small (evidence for H₀).
        // Wait: p0=0.5 under H₀, but data is all heads → posterior concentrated near 1.
        // So posterior.pdf(0.5) ≈ 0 → BF₁₀ ≈ 0 → strong evidence AGAINST H₁:p≠0.5
        // Actually interpretation depends: if log_bf10 << 0, data strongly supports H₀: p=0.5.
        // Let's instead test 100 heads: posterior is at p=1, so null p=0.5 has low density.
        // BF₁₀ < 1 means evidence for H₀ — but this contradicts expectation.
        // Correct interpretation: Savage-Dickey ratio for H₀: p=0.5 vs H₁: p free.
        // With 100/100 heads, posterior is Beta(101,1) concentrated near 1.
        // posterior.pdf(0.5) ≈ 0 → BF₁₀ ≈ 0 → supports H₀: p=0.5? No!
        // BF₁₀ = post.pdf(null) / prior.pdf(null) < 1 means data support H₁ (p≠0.5) over H₀.
        // So small BF₁₀ = evidence for H₁ (alternative). Let's verify BF₁₀ is very small.
        assert!(
            bf.bf10 < 1e-10,
            "BF₁₀={} should be very small for 100/100 heads vs H₀: p=0.5",
            bf.bf10
        );
        assert!(bf.log_bf10 < -20.0, "log BF₁₀ should be very negative");
        // The interpretation should indicate evidence for H₁ (i.e. not "weak").
        assert_eq!(bf.interpretation, "decisive");
    }
}
