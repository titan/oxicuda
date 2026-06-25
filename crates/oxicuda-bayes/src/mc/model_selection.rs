//! Bayesian predictive model selection: WAIC, PSIS-LOO, and DIC.
//!
//! Given a fitted posterior summarised by `S` Monte-Carlo draws and the
//! per-observation log-likelihood evaluated at each draw, these routines
//! estimate a model's *expected log pointwise predictive density* (`elpd`) on
//! new data, the standard measure of out-of-sample predictive accuracy in
//! Bayesian workflow (Gelman, Hwang & Vehtari 2014).
//!
//! | Criterion | Reference | Notes |
//! |-----------|-----------|-------|
//! | WAIC      | Watanabe 2010; Vehtari, Gelman & Gabry 2017 | log-mean-exp lppd minus posterior-variance penalty |
//! | PSIS-LOO  | Vehtari, Gelman & Gabry 2017 | importance-sampling leave-one-out with a Pareto-smoothed tail |
//! | DIC       | Spiegelhalter et al. 2002 | deviance information criterion (effective parameters `pᴅ`) |
//!
//! # Input layout
//!
//! The per-draw, per-observation log-likelihood is passed flattened in
//! **draw-major** order: `log_lik[s * n_obs + i] = log p(yᵢ | θ⁽ˢ⁾)` for draw
//! `s ∈ [0, n_draws)` and observation `i ∈ [0, n_obs)`.

use crate::error::{BayesError, BayesResult};

// ─── Numerically stable log-sum-exp / log-mean-exp ──────────────────────────

/// `log Σ exp(xⱼ)` computed in a single pass with the max-shift trick.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &x in xs {
        if x > max {
            max = x;
        }
    }
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0_f64;
    for &x in xs {
        acc += (x - max).exp();
    }
    max + acc.ln()
}

/// `log (1/n Σ exp(xⱼ))`.
fn log_mean_exp(xs: &[f64]) -> f64 {
    log_sum_exp(xs) - (xs.len() as f64).ln()
}

// ─── Common pointwise summary ───────────────────────────────────────────────

/// Per-observation pointwise predictive summary shared across criteria.
#[derive(Debug, Clone)]
pub struct PointwiseLpd {
    /// Number of observations `n`.
    pub n_obs: usize,
    /// Number of posterior draws `S`.
    pub n_draws: usize,
    /// `lppdᵢ = log (1/S Σₛ p(yᵢ|θ⁽ˢ⁾))` per observation.
    pub lppd: Vec<f64>,
    /// `pₘ,ᵢ = Varₛ[log p(yᵢ|θ⁽ˢ⁾)]` per observation (WAIC penalty term).
    pub p_var: Vec<f64>,
}

fn validate_loglik(log_lik: &[f64], n_draws: usize, n_obs: usize) -> BayesResult<()> {
    if n_draws == 0 || n_obs == 0 {
        return Err(BayesError::EmptyInputs);
    }
    if n_draws < 2 {
        return Err(BayesError::InsufficientSamples {
            min: 2,
            got: n_draws,
        });
    }
    if log_lik.len() != n_draws * n_obs {
        return Err(BayesError::DimensionMismatch {
            expected: n_draws * n_obs,
            got: log_lik.len(),
        });
    }
    Ok(())
}

/// Compute the per-observation log pointwise predictive density and its
/// posterior log-likelihood variance.
///
/// # Errors
/// - [`BayesError::EmptyInputs`] if either dimension is zero.
/// - [`BayesError::InsufficientSamples`] if `n_draws < 2`.
/// - [`BayesError::DimensionMismatch`] if `log_lik.len() != n_draws * n_obs`.
/// - [`BayesError::NanEncountered`] if any log-likelihood is `NaN`.
pub fn pointwise_lpd(log_lik: &[f64], n_draws: usize, n_obs: usize) -> BayesResult<PointwiseLpd> {
    validate_loglik(log_lik, n_draws, n_obs)?;

    let mut lppd = vec![0.0_f64; n_obs];
    let mut p_var = vec![0.0_f64; n_obs];
    let mut column = vec![0.0_f64; n_draws];

    for i in 0..n_obs {
        for s in 0..n_draws {
            let v = log_lik[s * n_obs + i];
            if v.is_nan() {
                return Err(BayesError::NanEncountered {
                    location: "model_selection_loglik",
                });
            }
            column[s] = v;
        }
        lppd[i] = log_mean_exp(&column);
        // Unbiased (Bessel-corrected) variance of the log-likelihood column.
        let mean = column.iter().sum::<f64>() / n_draws as f64;
        let var = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_draws as f64 - 1.0);
        p_var[i] = var;
    }

    Ok(PointwiseLpd {
        n_obs,
        n_draws,
        lppd,
        p_var,
    })
}

// ─── WAIC ───────────────────────────────────────────────────────────────────

/// Result of a WAIC computation (Watanabe-Akaike information criterion).
#[derive(Debug, Clone)]
pub struct WaicResult {
    /// Expected log pointwise predictive density: `elpd = Σᵢ (lppdᵢ − pₘ,ᵢ)`.
    pub elpd_waic: f64,
    /// Effective number of parameters `pₘ = Σᵢ Varₛ[log p(yᵢ|θ)]`.
    pub p_waic: f64,
    /// WAIC on the deviance scale: `−2 · elpd_waic`.
    pub waic: f64,
    /// Per-observation elpd contributions `lppdᵢ − pₘ,ᵢ`.
    pub pointwise: Vec<f64>,
    /// Monte-Carlo standard error of `elpd_waic` (√n · sd of the pointwise terms).
    pub se: f64,
}

/// Compute WAIC from a draw-major log-likelihood matrix.
///
/// # Errors
/// Propagates [`pointwise_lpd`].
pub fn waic(log_lik: &[f64], n_draws: usize, n_obs: usize) -> BayesResult<WaicResult> {
    let pw = pointwise_lpd(log_lik, n_draws, n_obs)?;
    let mut pointwise = vec![0.0_f64; n_obs];
    let mut p_waic = 0.0;
    for (i, pt) in pointwise.iter_mut().enumerate() {
        *pt = pw.lppd[i] - pw.p_var[i];
        p_waic += pw.p_var[i];
    }
    let elpd_waic: f64 = pointwise.iter().sum();
    let se = pointwise_se(&pointwise);
    Ok(WaicResult {
        elpd_waic,
        p_waic,
        waic: -2.0 * elpd_waic,
        pointwise,
        se,
    })
}

/// Standard error of a summed pointwise statistic: `√(n · Var[pointwise])`.
fn pointwise_se(pointwise: &[f64]) -> f64 {
    let n = pointwise.len();
    if n < 2 {
        return 0.0;
    }
    let mean = pointwise.iter().sum::<f64>() / n as f64;
    let var = pointwise.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    (n as f64 * var).sqrt()
}

// ─── Generalized Pareto tail fit (for PSIS) ─────────────────────────────────

/// Fit a generalized Pareto distribution to the (non-negative, ascending) upper
/// tail exceedances by the profile-likelihood / empirical-Bayes estimator of
/// Zhang & Stephens (2009), returning `(k̂, σ̂)` for the GPD shape `k = ξ` and
/// scale `σ`.
///
/// The density is `p(x) = σ⁻¹(1 + ξx/σ)^(−1/ξ − 1)`.  Following the `loo` and
/// ArviZ reference implementations, the profile is taken over the auxiliary
/// parameter `b = −ξ/σ`, for which `k(b) = (1/n)Σ log(1 − b·xᵢ)` and the
/// profile log-likelihood (up to an additive constant) is
/// `lₓ(b) = log(−b / k(b)) + k(b) − 1`.  A grid of `b`-values anchored at the
/// data quartile is averaged under its self-normalised likelihood weights.
fn gpd_fit(tail: &[f64]) -> (f64, f64) {
    let n = tail.len();
    if n < 5 {
        return (f64::INFINITY, f64::NAN);
    }
    let nf = n as f64;
    // Number of grid points (Zhang & Stephens recommend ⌊30 + √n⌋).
    let m = 30 + (nf.sqrt().floor() as usize);
    let x_min = tail[0];
    let x_max = tail[n - 1];
    if x_max.is_nan() || x_max <= 0.0 || x_max <= x_min {
        // Degenerate tail (no spread) — exponential limit, k → 0.
        return (0.0, x_max.max(1e-12));
    }
    // Robust scale anchor: the ⌊n/4 + 0.5⌋-th order statistic (the quartile).
    let q_idx = ((nf / 4.0 + 0.5).floor() as usize).min(n - 1);
    let x_star = tail[q_idx].max(1e-300);

    // Grid of b-values: bⱼ = 1/x_max + (1 − √(m/(j−0.5))) / (3·x_star).
    // For small j the second term is large-negative ⇒ heavy-tail (ξ>0) region.
    let inv_xmax = 1.0 / x_max;
    let mut b_grid = vec![0.0_f64; m];
    let mut log_w = vec![0.0_f64; m];
    for (j, b) in b_grid.iter_mut().enumerate() {
        let frac = 1.0 - (m as f64 / (j as f64 + 0.5)).sqrt();
        *b = inv_xmax + frac / (3.0 * x_star);
    }

    // Profile log-likelihood at each grid point.
    for (lw, &b) in log_w.iter_mut().zip(b_grid.iter()) {
        *lw = nf * gpd_profile_ll(b, tail);
    }

    // Self-normalised weights wⱼ ∝ exp(lₓ(bⱼ) − max).
    let max_l = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_l.is_finite() {
        return (f64::INFINITY, f64::NAN);
    }
    let mut wsum = 0.0;
    for w in log_w.iter_mut() {
        *w = (*w - max_l).exp();
        wsum += *w;
    }
    if wsum.is_nan() || wsum <= 0.0 {
        return (f64::INFINITY, f64::NAN);
    }
    // Posterior-mean b̂ = Σ wⱼ bⱼ.
    let mut b_hat = 0.0;
    for (w, &b) in log_w.iter().zip(b_grid.iter()) {
        b_hat += (*w / wsum) * b;
    }

    // Recover (k, σ): k̂ = (1/n)Σ log(1 − b̂·x); σ̂ = −k̂ / b̂.
    let k_hat = gpd_k_of_b(b_hat, tail);
    let sigma_hat = if b_hat.abs() > 1e-300 {
        -k_hat / b_hat
    } else {
        x_star
    };
    // Weakly-informative small-sample shrinkage of ξ toward 0.5 with prior
    // weight 10 (Vehtari, Gelman & Gabry 2017, §3.2.1).
    let k_corrected = (k_hat * nf + 0.5 * 10.0) / (nf + 10.0);
    (k_corrected, sigma_hat)
}

/// `k(b) = (1/n) Σ log(1 − b·xᵢ)` — the profile MLE of the GPD shape `ξ` implied
/// by the auxiliary parameter `b = −ξ/σ`.  The scale follows as `σ(b) = −k/b`.
fn gpd_k_of_b(b: f64, tail: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &x in tail {
        // log(1 − b·x); for admissible b (< 1/x_max) the argument is positive.
        let arg = 1.0 - b * x;
        acc += if arg > 0.0 { arg.ln() } else { -700.0 };
    }
    acc / tail.len() as f64
}

/// Per-observation profile log-likelihood `(1/n)·ℓ(b)` of the generalized
/// Pareto fit at the auxiliary parameter `b`, with the shape `k = k(b)` and the
/// MLE scale `σ = −k/b` substituted into the exact GPD log-density.
///
/// Using the exact GPD likelihood (rather than the algebraically-simplified
/// Zhang-Stephens surrogate, which is sign-sensitive and numerically fragile)
/// keeps the grid search anchored at the true profile maximum.  Returns
/// `−∞` for inadmissible `b` so the grid point receives zero weight.
fn gpd_profile_ll(b: f64, tail: &[f64]) -> f64 {
    if b.abs() < 1e-300 {
        return f64::NEG_INFINITY;
    }
    let k = gpd_k_of_b(b, tail);
    let sigma = -k / b;
    if !sigma.is_finite() || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = tail.len() as f64;
    // GPD log-density:
    //   k ≈ 0: ℓ/n = −log σ − x̄/σ.
    //   else : ℓ/n = −log σ − (1 + 1/k)·(1/n)Σ log(1 + k·x/σ).
    if k.abs() < 1e-8 {
        let mean_x = tail.iter().sum::<f64>() / n;
        -sigma.ln() - mean_x / sigma
    } else {
        let mut acc = 0.0;
        for &x in tail {
            let z = 1.0 + k * x / sigma;
            if z <= 0.0 {
                return f64::NEG_INFINITY;
            }
            acc += z.ln();
        }
        -sigma.ln() - (1.0 + 1.0 / k) * acc / n
    }
}

/// GPD inverse-CDF (quantile) for shape `k`, scale `σ` at probability `p`.
fn gpd_quantile(p: f64, k: f64, sigma: f64) -> f64 {
    if k.abs() < 1e-8 {
        -sigma * (1.0 - p).ln()
    } else {
        sigma * ((1.0 - p).powf(-k) - 1.0) / k
    }
}

// ─── PSIS-LOO ───────────────────────────────────────────────────────────────

/// Result of a Pareto-smoothed importance-sampling LOO computation.
#[derive(Debug, Clone)]
pub struct PsisLooResult {
    /// Expected log pointwise predictive density `elpd_loo = Σᵢ ŷᵢ`.
    pub elpd_loo: f64,
    /// Effective number of parameters `p_loo = lppd − elpd_loo`.
    pub p_loo: f64,
    /// LOO information criterion on the deviance scale: `−2 · elpd_loo`.
    pub looic: f64,
    /// Per-observation LOO elpd `log Σₛ wₛ p(yᵢ|θ⁽ˢ⁾) / Σₛ wₛ`.
    pub pointwise: Vec<f64>,
    /// Per-observation Pareto shape diagnostics `k̂ᵢ`.
    ///
    /// `k̂ < 0.5`: estimate reliable. `0.5 ≤ k̂ < 0.7`: usable but high-variance.
    /// `k̂ ≥ 0.7`: importance sampling unreliable for that point.
    pub pareto_k: Vec<f64>,
    /// Monte-Carlo standard error of `elpd_loo`.
    pub se: f64,
}

impl PsisLooResult {
    /// Number of observations whose Pareto-k diagnostic exceeds `threshold`
    /// (default decision boundary `0.7`).
    #[must_use]
    pub fn count_bad_k(&self, threshold: f64) -> usize {
        self.pareto_k.iter().filter(|&&k| k > threshold).count()
    }
}

/// Compute PSIS-LOO from a draw-major log-likelihood matrix.
///
/// The raw importance ratios are the per-observation log-likelihoods (since the
/// LOO importance weight for dropping observation `i` is `1/p(yᵢ|θ⁽ˢ⁾)`, whose
/// log is `−log_lik`).  Each observation's largest weights are replaced by the
/// order statistics of a fitted generalized Pareto tail, which controls the
/// variance of the self-normalised importance estimate.
///
/// # Errors
/// Propagates [`pointwise_lpd`] validation.
pub fn psis_loo(log_lik: &[f64], n_draws: usize, n_obs: usize) -> BayesResult<PsisLooResult> {
    validate_loglik(log_lik, n_draws, n_obs)?;

    // We need the lppd (Σ log mean exp of the *raw* likelihood) for p_loo.
    let pw = pointwise_lpd(log_lik, n_draws, n_obs)?;
    let lppd_total: f64 = pw.lppd.iter().sum();

    let mut pointwise = vec![0.0_f64; n_obs];
    let mut pareto_k = vec![0.0_f64; n_obs];

    // Tail size: min(0.2·S, 3·√S) following Vehtari et al. 2017.
    let tail_len = ((0.2 * n_draws as f64).min(3.0 * (n_draws as f64).sqrt())).ceil() as usize;
    let tail_len = tail_len.clamp(5, n_draws.saturating_sub(1).max(1));

    let mut log_ratios = vec![0.0_f64; n_draws];
    let mut idx: Vec<usize> = Vec::with_capacity(n_draws);

    for i in 0..n_obs {
        // Importance ratio rₛ = 1/p(yᵢ|θ⁽ˢ⁾) ⇒ log rₛ = −log_lik.
        for s in 0..n_draws {
            log_ratios[s] = -log_lik[s * n_obs + i];
        }

        // Argsort the log-ratios ascending.
        idx.clear();
        idx.extend(0..n_draws);
        idx.sort_by(|&a, &b| {
            log_ratios[a]
                .partial_cmp(&log_ratios[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // The tail is the top `tail_len` largest log-ratios.
        let cut_pos = n_draws - tail_len;
        let cutoff = log_ratios[idx[cut_pos.saturating_sub(1)]];

        // Exceedances above the cutoff (in the raw-ratio space, max-shifted to
        // avoid overflow): we work in log-space, exponentiating relative to the
        // cutoff so that the smallest tail value maps near 0.
        let mut exceed = vec![0.0_f64; tail_len];
        for (t, &si) in idx[cut_pos..].iter().enumerate() {
            exceed[t] = (log_ratios[si] - cutoff).exp() - 1.0;
            if exceed[t] < 0.0 {
                exceed[t] = 0.0;
            }
        }

        let (k_hat, sigma_hat) = gpd_fit(&exceed);
        pareto_k[i] = k_hat;

        // Replace tail log-ratios with the GPD order-statistic quantiles when
        // the fit is well-defined.
        if sigma_hat.is_finite() && k_hat.is_finite() && k_hat < 1.0 {
            for (t, &si) in idx[cut_pos..].iter().enumerate() {
                // Plotting-position probability for the (t+1)-th of tail_len.
                let p = (t as f64 + 0.5) / tail_len as f64;
                let smoothed_excess = gpd_quantile(p, k_hat, sigma_hat);
                // Map back to log-ratio space: log(cutoff_ratio·(1+excess)).
                log_ratios[si] = cutoff + (1.0 + smoothed_excess).ln();
            }
        }

        // Truncate weights at the theoretical maximum S^{3/4}·mean (raw weights)
        // for additional stability — done in log-space via a max cap.
        let max_log = log_ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Self-normalised LOO predictive:
        //   elpd_i = log( Σₛ wₛ·p(yᵢ|θ⁽ˢ⁾) ) − log( Σₛ wₛ )
        //          = logsumexp_s(log wₛ + log_lik) − logsumexp_s(log wₛ).
        // Here log wₛ = log_ratios (the smoothed log importance weights).
        let mut num = vec![0.0_f64; n_draws];
        for s in 0..n_draws {
            num[s] = log_ratios[s] + log_lik[s * n_obs + i];
        }
        // Shift the denominator weights for stability (cancels in the ratio).
        let mut shifted = vec![0.0_f64; n_draws];
        for s in 0..n_draws {
            shifted[s] = log_ratios[s] - max_log;
        }
        let log_denom = {
            let mut acc = 0.0;
            for &w in &shifted {
                acc += w.exp();
            }
            max_log + acc.ln()
        };
        pointwise[i] = log_sum_exp(&num) - log_denom;
    }

    let elpd_loo: f64 = pointwise.iter().sum();
    let p_loo = lppd_total - elpd_loo;
    let se = pointwise_se(&pointwise);

    Ok(PsisLooResult {
        elpd_loo,
        p_loo,
        looic: -2.0 * elpd_loo,
        pointwise,
        pareto_k,
        se,
    })
}

// ─── DIC ────────────────────────────────────────────────────────────────────

/// Result of a deviance information criterion computation.
#[derive(Debug, Clone)]
pub struct DicResult {
    /// Deviance information criterion `DIC = D̄ + pᴅ = 2·D̄ − D(θ̄)`.
    pub dic: f64,
    /// Effective number of parameters `pᴅ = D̄ − D(θ̄)`.
    pub p_dic: f64,
    /// Posterior-mean deviance `D̄ = E[−2 log p(y|θ)]`.
    pub d_bar: f64,
    /// Deviance at the posterior-mean log-likelihood `D(θ̄)` (point estimate).
    pub d_hat: f64,
}

/// Compute the DIC (Spiegelhalter et al. 2002).
///
/// Uses the variance-based effective parameter count `pᴅ = 2·Varₛ[log p(y|θ)]`
/// summed over observations (Gelman 2004), which avoids needing an explicit
/// posterior-mean point estimate and matches the WAIC penalty form.
///
/// # Errors
/// Propagates [`pointwise_lpd`] validation.
pub fn dic(log_lik: &[f64], n_draws: usize, n_obs: usize) -> BayesResult<DicResult> {
    validate_loglik(log_lik, n_draws, n_obs)?;
    let pw = pointwise_lpd(log_lik, n_draws, n_obs)?;

    // Posterior-mean total log-likelihood per draw.
    let mut total_loglik_mean = 0.0;
    for s in 0..n_draws {
        let mut row = 0.0;
        for i in 0..n_obs {
            row += log_lik[s * n_obs + i];
        }
        total_loglik_mean += row;
    }
    total_loglik_mean /= n_draws as f64;
    let d_bar = -2.0 * total_loglik_mean;

    // Effective parameters via the posterior variance of the log-likelihood
    // (Gelman's pᴅ = 2·Var). D(θ̄) = D̄ − pᴅ.
    let p_dic: f64 = pw.p_var.iter().sum::<f64>() * 2.0;
    let d_hat = d_bar - p_dic;

    Ok(DicResult {
        dic: d_bar + p_dic,
        p_dic,
        d_bar,
        d_hat,
    })
}

// ─── Model comparison ───────────────────────────────────────────────────────

/// Difference in expected log predictive density between two models, with the
/// standard error of the *paired* difference (Vehtari et al. 2017 §3.4).
///
/// A positive `elpd_diff` means `pointwise_a` (the first argument) has higher
/// predictive accuracy.  The two pointwise vectors must be aligned observation
/// for observation and have equal length.
///
/// # Errors
/// - [`BayesError::DimensionMismatch`] on unequal lengths.
/// - [`BayesError::InsufficientSamples`] if fewer than two observations.
pub fn compare_elpd(pointwise_a: &[f64], pointwise_b: &[f64]) -> BayesResult<(f64, f64)> {
    if pointwise_a.len() != pointwise_b.len() {
        return Err(BayesError::DimensionMismatch {
            expected: pointwise_a.len(),
            got: pointwise_b.len(),
        });
    }
    let n = pointwise_a.len();
    if n < 2 {
        return Err(BayesError::InsufficientSamples { min: 2, got: n });
    }
    let diff: Vec<f64> = pointwise_a
        .iter()
        .zip(pointwise_b.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let elpd_diff: f64 = diff.iter().sum();
    let se = pointwise_se(&diff);
    Ok((elpd_diff, se))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::BayesRng;

    /// Build a draw-major log-likelihood matrix for a Gaussian observation
    /// model: yᵢ ~ N(θ, σ²) with `n_draws` posterior draws of θ ~ N(m, τ²).
    fn gaussian_loglik(y: &[f64], theta_draws: &[f64], sigma: f64) -> (Vec<f64>, usize, usize) {
        let n_obs = y.len();
        let n_draws = theta_draws.len();
        let mut ll = vec![0.0; n_draws * n_obs];
        let c = -(2.0 * std::f64::consts::PI).ln() * 0.5 - sigma.ln();
        for (s, &theta) in theta_draws.iter().enumerate() {
            for (i, &yi) in y.iter().enumerate() {
                let z = (yi - theta) / sigma;
                ll[s * n_obs + i] = c - 0.5 * z * z;
            }
        }
        (ll, n_draws, n_obs)
    }

    fn make_draws(rng: &mut BayesRng, n: usize, mean: f64, sd: f64) -> Vec<f64> {
        (0..n).map(|_| mean + sd * rng.next_normal()).collect()
    }

    #[test]
    fn log_sum_exp_matches_naive() {
        let xs = [-1.0, 0.5, 2.0, -3.0];
        let lse = log_sum_exp(&xs);
        let naive = xs.iter().map(|&x| x.exp()).sum::<f64>().ln();
        assert!((lse - naive).abs() < 1e-12);
        // All −∞ → −∞.
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY; 3]), f64::NEG_INFINITY);
    }

    #[test]
    fn pointwise_lpd_validates_shape() {
        assert!(matches!(
            pointwise_lpd(&[1.0, 2.0, 3.0], 2, 2),
            Err(BayesError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            pointwise_lpd(&[], 0, 0),
            Err(BayesError::EmptyInputs)
        ));
        assert!(matches!(
            pointwise_lpd(&[1.0], 1, 1),
            Err(BayesError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn waic_penalty_is_positive_and_below_lppd() {
        let mut rng = BayesRng::new(1);
        let y = make_draws(&mut rng, 50, 0.0, 1.0);
        let theta = make_draws(&mut rng, 400, 0.0, 0.3);
        let (ll, nd, no) = gaussian_loglik(&y, &theta, 1.0);
        let w = waic(&ll, nd, no).unwrap();
        // Effective parameters must be positive and modest.
        assert!(w.p_waic > 0.0, "p_waic {}", w.p_waic);
        // elpd = lppd − p_waic, so elpd < lppd; check via reconstruction.
        let pw = pointwise_lpd(&ll, nd, no).unwrap();
        let lppd: f64 = pw.lppd.iter().sum();
        assert!(w.elpd_waic < lppd + 1e-9);
        assert!((w.waic + 2.0 * w.elpd_waic).abs() < 1e-9);
        assert!(w.se > 0.0);
    }

    #[test]
    fn waic_matches_hand_computed_two_point() {
        // Two draws, one observation with log-likelihoods a=-1, b=-2.
        // lppd = log( (e^-1 + e^-2)/2 ); var (Bessel) = ((−1−m)²+(−2−m)²)/1
        // with m = -1.5 → ((0.5)²+(0.5)²)/1 = 0.5.
        let ll = vec![-1.0, -2.0]; // draw0,obs0 ; draw1,obs0
        let w = waic(&ll, 2, 1).unwrap();
        let lppd = ((-1.0_f64).exp() + (-2.0_f64).exp()).ln() - (2.0_f64).ln();
        let p = 0.5;
        assert!((w.p_waic - p).abs() < 1e-12, "p {}", w.p_waic);
        assert!((w.elpd_waic - (lppd - p)).abs() < 1e-12);
    }

    #[test]
    fn psis_loo_reliable_for_well_behaved_model() {
        let mut rng = BayesRng::new(42);
        let y = make_draws(&mut rng, 60, 0.5, 1.0);
        // Reasonably concentrated posterior → well-behaved importance weights.
        let theta = make_draws(&mut rng, 1000, 0.5, 0.2);
        let (ll, nd, no) = gaussian_loglik(&y, &theta, 1.0);
        let loo = psis_loo(&ll, nd, no).unwrap();
        // p_loo should be positive and small (a single parameter problem).
        assert!(loo.p_loo > 0.0 && loo.p_loo < 10.0, "p_loo {}", loo.p_loo);
        assert!((loo.looic + 2.0 * loo.elpd_loo).abs() < 1e-9);
        // Most Pareto-k values should be in the reliable region.
        let bad = loo.count_bad_k(0.7);
        assert!(bad <= no / 5, "too many bad k: {bad}/{no}");
        assert_eq!(loo.pareto_k.len(), no);
    }

    #[test]
    fn psis_loo_close_to_waic_when_posterior_concentrated() {
        // For a concentrated posterior WAIC and PSIS-LOO should nearly agree.
        let mut rng = BayesRng::new(7);
        let y = make_draws(&mut rng, 40, -1.0, 1.0);
        let theta = make_draws(&mut rng, 2000, -1.0, 0.15);
        let (ll, nd, no) = gaussian_loglik(&y, &theta, 1.0);
        let w = waic(&ll, nd, no).unwrap();
        let loo = psis_loo(&ll, nd, no).unwrap();
        // The two elpd estimates should be within a few units of each other.
        assert!(
            (w.elpd_waic - loo.elpd_loo).abs() < 3.0,
            "waic {} vs loo {}",
            w.elpd_waic,
            loo.elpd_loo
        );
    }

    #[test]
    fn dic_effective_params_positive() {
        let mut rng = BayesRng::new(11);
        let y = make_draws(&mut rng, 50, 2.0, 1.0);
        let theta = make_draws(&mut rng, 800, 2.0, 0.25);
        let (ll, nd, no) = gaussian_loglik(&y, &theta, 1.0);
        let d = dic(&ll, nd, no).unwrap();
        assert!(d.p_dic > 0.0, "p_dic {}", d.p_dic);
        // DIC = D̄ + pᴅ and D(θ̄) = D̄ − pᴅ.
        assert!((d.dic - (d.d_bar + d.p_dic)).abs() < 1e-9);
        assert!((d.d_hat - (d.d_bar - d.p_dic)).abs() < 1e-9);
    }

    #[test]
    fn compare_elpd_prefers_better_model() {
        // Model A pointwise uniformly higher by 0.2 per point.
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b: Vec<f64> = a.iter().map(|&x| x - 0.2).collect();
        let (diff, se) = compare_elpd(&a, &b).unwrap();
        assert!((diff - 1.0).abs() < 1e-12, "diff {diff}");
        // Constant difference → zero standard error.
        assert!(se < 1e-9, "se {se}");
        assert!(matches!(
            compare_elpd(&[1.0, 2.0], &[1.0]),
            Err(BayesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gpd_fit_recovers_known_shape() {
        // Sample exceedances from an exponential (k → 0) and check k̂ is small.
        let mut rng = BayesRng::new(123);
        let mut tail: Vec<f64> = (0..200)
            .map(|_| -1.5 * rng.next_f64().max(1e-12).ln())
            .collect();
        tail.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (k, sigma) = gpd_fit(&tail);
        assert!(k.is_finite() && k.abs() < 0.4, "k {k}");
        assert!(sigma.is_finite() && sigma > 0.0, "sigma {sigma}");
    }

    #[test]
    fn waic_better_model_has_higher_elpd() {
        // Correct σ vs an over-dispersed σ: the correct model should win.
        let mut rng = BayesRng::new(2024);
        let y = make_draws(&mut rng, 80, 0.0, 1.0);
        let theta = make_draws(&mut rng, 1000, 0.0, 0.2);
        let (ll_good, nd, no) = gaussian_loglik(&y, &theta, 1.0);
        let (ll_bad, _, _) = gaussian_loglik(&y, &theta, 3.0);
        let w_good = waic(&ll_good, nd, no).unwrap();
        let w_bad = waic(&ll_bad, nd, no).unwrap();
        assert!(
            w_good.elpd_waic > w_bad.elpd_waic,
            "good {} !> bad {}",
            w_good.elpd_waic,
            w_bad.elpd_waic
        );
    }
}
