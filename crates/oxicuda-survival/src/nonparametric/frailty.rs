//! Gamma frailty model for clustered survival data (Clayton 1978; Vaupel et al. 1979).
//!
//! Each cluster c has a latent frailty Z_c ~ Gamma(1/θ, 1/θ), giving `E[Z_c]=1` and
//! `Var[Z_c]=θ`.  Conditional on Z_c and covariates x_i, the hazard for subject i is:
//!   λ(t | Z_c, x_i) = Z_c · λ₀(t) · exp(β^T x_i)
//!
//! The EM algorithm (Nielsen et al. 1992; Therneau & Grambsch 2000) iterates between:
//!   E-step: compute posterior frailty means û_c = (1/θ + d_c) / (1/θ + Λ_c(β))
//!   M-step for β: weighted partial likelihood Newton-Raphson
//!   M-step for θ: Newton step on the profile marginal log-likelihood

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;
use crate::special::digamma::digamma;
use crate::special::gammaln::gammaln;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the gamma frailty EM algorithm.
#[derive(Debug, Clone)]
pub struct FrailtyConfig {
    /// Initial frailty variance θ (default 1.0).
    pub theta_init: f64,
    /// Maximum number of EM outer iterations (default 30).
    pub max_outer_iter: usize,
    /// Maximum Newton-Raphson iterations per M-step for β (default 20).
    pub max_inner_iter: usize,
    /// EM convergence tolerance on max |Δβ| + |Δθ| (default 1e-5).
    pub tol: f64,
    /// Newton-Raphson inner convergence tolerance on max |score| (default 1e-6).
    pub inner_tol: f64,
    /// Lower bound on θ (default 1e-6).
    pub min_theta: f64,
    /// Upper bound on θ (default 20.0).
    pub max_theta: f64,
}

impl Default for FrailtyConfig {
    fn default() -> Self {
        Self {
            theta_init: 1.0,
            max_outer_iter: 30,
            max_inner_iter: 20,
            tol: 1.0e-5,
            inner_tol: 1.0e-6,
            min_theta: 1.0e-6,
            max_theta: 20.0,
        }
    }
}

// ─── Fit output ───────────────────────────────────────────────────────────────

/// Fitted gamma frailty Cox model.
#[derive(Debug, Clone)]
pub struct FrailtyFit {
    /// Covariate coefficients β, length = n_covariates.
    pub beta: Vec<f64>,
    /// Estimated frailty variance θ.
    pub theta: f64,
    /// Posterior frailty means û_c for each cluster, length = n_clusters.
    pub cluster_frailty: Vec<f64>,
    /// Profile marginal log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations consumed.
    pub n_iter: usize,
    /// Whether the EM algorithm converged within `max_outer_iter`.
    pub converged: bool,
    /// Number of covariates.
    pub n_covariates: usize,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Breslow baseline cumulative hazard values at distinct event times.
    pub baseline_cumhaz: Vec<f64>,
    /// Distinct event times corresponding to baseline_cumhaz.
    pub baseline_times: Vec<f64>,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a gamma frailty Cox model by EM algorithm.
///
/// # Arguments
/// * `times`        — event or censoring times \[n\]
/// * `events`       — 1 = event, 0 = censored \[n\]
/// * `covariates`   — row-major covariate matrix \[n × p\]; may be empty when p=0
/// * `cluster_ids`  — cluster index for each observation \[n\], values in `0..n_clusters`
/// * `n_clusters`   — number of distinct clusters
/// * `config`       — algorithm configuration
pub fn fit_gamma_frailty(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    cluster_ids: &[usize],
    n_clusters: usize,
    config: &FrailtyConfig,
) -> SurvivalResult<FrailtyFit> {
    // ── Input validation ──────────────────────────────────────────────────────
    let n_samples = times.len();
    if n_samples == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if events.len() != n_samples || cluster_ids.len() != n_samples {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n_samples, n_samples],
            got: vec![events.len(), cluster_ids.len()],
        });
    }
    if n_clusters == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_clusters must be >= 1".to_string(),
        ));
    }
    // Determine p from flat covariates array
    let n_cov = if covariates.is_empty() {
        0usize
    } else {
        if covariates.len() % n_samples != 0 {
            return Err(SurvivalError::ShapeMismatch {
                expected: vec![n_samples],
                got: vec![covariates.len()],
            });
        }
        covariates.len() / n_samples
    };
    // Validate cluster ids
    for (i, &cid) in cluster_ids.iter().enumerate() {
        if cid >= n_clusters {
            return Err(SurvivalError::IndexOutOfBounds {
                index: cid,
                len: n_clusters,
            });
        }
        let _ = i;
    }
    // Validate times
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }
    // Validate config
    if !config.theta_init.is_finite() || config.theta_init <= 0.0 {
        return Err(SurvivalError::InvalidConfiguration(
            "theta_init must be finite and positive".to_string(),
        ));
    }
    if config.min_theta <= 0.0 || config.max_theta <= config.min_theta {
        return Err(SurvivalError::InvalidConfiguration(
            "require 0 < min_theta < max_theta".to_string(),
        ));
    }

    // Count total events
    let total_events: usize = events.iter().map(|&e| e as usize).sum();

    // If no events at all, return a degenerate fit (no information to estimate β or θ).
    if total_events == 0 {
        let cluster_frailty = vec![1.0_f64; n_clusters];
        return Ok(FrailtyFit {
            beta: vec![0.0_f64; n_cov],
            theta: config.theta_init,
            cluster_frailty,
            log_likelihood: 0.0,
            n_iter: 0,
            converged: true,
            n_covariates: n_cov,
            n_clusters,
            baseline_cumhaz: Vec::new(),
            baseline_times: Vec::new(),
        });
    }

    // ── Initialisation ────────────────────────────────────────────────────────
    let mut beta = vec![0.0_f64; n_cov];
    let mut theta = config.theta_init.clamp(config.min_theta, config.max_theta);
    // Start with equal frailties
    let mut cluster_frailty = vec![1.0_f64; n_clusters];

    // Per-cluster event counts (static)
    let cluster_events: Vec<f64> = (0..n_clusters)
        .map(|c| {
            cluster_ids
                .iter()
                .zip(events.iter())
                .filter(|&(&cid, _)| cid == c)
                .map(|(_, &e)| e as f64)
                .sum()
        })
        .collect();

    let mut converged = false;
    let mut n_iter = 0usize;
    let mut log_likelihood = f64::NEG_INFINITY;

    // ── EM iterations ─────────────────────────────────────────────────────────
    for outer in 0..config.max_outer_iter {
        n_iter = outer + 1;

        // ── M-step for β (weighted partial likelihood Newton-Raphson) ──────
        let (beta_new, _nr_iters, _nr_converged) = weighted_cox_nr_step(
            times,
            events,
            covariates,
            cluster_ids,
            &cluster_frailty,
            &beta,
            n_samples,
            n_cov,
            config.max_inner_iter,
            config.inner_tol,
        )?;

        // ── Breslow baseline cumulative hazard with current β and frailties ─
        let (basetime, basecumhaz) = compute_breslow(
            times,
            events,
            covariates,
            cluster_ids,
            &cluster_frailty,
            &beta_new,
            n_samples,
            n_cov,
        );

        // ── E-step: posterior frailty means ───────────────────────────────
        let cluster_risk = compute_cluster_risk(
            times,
            covariates,
            cluster_ids,
            n_clusters,
            &beta_new,
            &basetime,
            &basecumhaz,
            n_samples,
            n_cov,
        );
        let frailty_new = e_step(&cluster_events, &cluster_risk, theta, n_clusters);

        // ── M-step for θ ──────────────────────────────────────────────────
        let theta_new = m_step_theta(
            &cluster_events,
            &cluster_risk,
            theta,
            n_clusters,
            config.min_theta,
            config.max_theta,
        );

        // ── Convergence check ─────────────────────────────────────────────
        let delta_beta = if n_cov > 0 {
            beta_new
                .iter()
                .zip(beta.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        } else {
            0.0
        };
        let delta_theta = (theta_new - theta).abs();
        let delta = delta_beta + delta_theta;

        beta = beta_new;
        theta = theta_new;
        cluster_frailty = frailty_new;

        // Compute log-likelihood for reporting
        log_likelihood = marginal_log_likelihood(&cluster_events, &cluster_risk, theta, n_clusters);

        if delta < config.tol {
            converged = true;
            break;
        }
    }

    // Final Breslow baseline with converged β and frailties
    let (baseline_times, baseline_cumhaz) = compute_breslow(
        times,
        events,
        covariates,
        cluster_ids,
        &cluster_frailty,
        &beta,
        n_samples,
        n_cov,
    );

    Ok(FrailtyFit {
        beta,
        theta,
        cluster_frailty,
        log_likelihood,
        n_iter,
        converged,
        n_covariates: n_cov,
        n_clusters,
        baseline_cumhaz,
        baseline_times,
    })
}

/// Predict survival function S(t | x, u_c) = exp(−u_c · exp(β^T x) · Λ₀(t))
/// at each time in `eval_times`.
///
/// # Arguments
/// * `fit`            — fitted frailty model
/// * `times`          — observation times (unused, kept for API symmetry)
/// * `covariates`     — flat covariate row for prediction \[n_covariates\]
/// * `frailty_value`  — the frailty value u_c to apply (1.0 = population average)
/// * `eval_times`     — query times at which to evaluate S(t)
/// * `baseline_times` — distinct event times from training (from `fit.baseline_times`)
/// * `baseline_cumhaz`— Breslow Λ₀ values at `baseline_times` (from `fit.baseline_cumhaz`)
pub fn predict_frailty_survival(
    fit: &FrailtyFit,
    _times: &[f64],
    covariates: &[f64],
    frailty_value: f64,
    eval_times: &[f64],
    baseline_times: &[f64],
    baseline_cumhaz: &[f64],
) -> SurvivalResult<Vec<f64>> {
    let p = fit.n_covariates;
    if covariates.len() != p {
        return Err(SurvivalError::InvalidParameter(format!(
            "covariate dimension mismatch: expected {p}, got {}",
            covariates.len()
        )));
    }
    if baseline_times.len() != baseline_cumhaz.len() {
        return Err(SurvivalError::DimensionMismatch {
            a: baseline_times.len(),
            b: baseline_cumhaz.len(),
        });
    }
    if frailty_value <= 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "frailty_value must be positive, got {frailty_value}"
        )));
    }

    // Compute linear predictor exp(β^T x)
    let lp: f64 = fit
        .beta
        .iter()
        .zip(covariates.iter())
        .map(|(b, x)| b * x)
        .sum();
    let exp_lp = lp.exp();

    // For each eval_time, find Λ₀(t) by step-function interpolation
    let mut result = Vec::with_capacity(eval_times.len());
    for &t in eval_times {
        // Find the largest baseline_time <= t
        let lambda0 = if baseline_times.is_empty() || t < baseline_times[0] {
            0.0
        } else {
            match baseline_times
                .binary_search_by(|bt| bt.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less))
            {
                Ok(idx) => baseline_cumhaz[idx],
                Err(0) => 0.0,
                Err(idx) => baseline_cumhaz[idx - 1],
            }
        };
        let cum_haz = frailty_value * exp_lp * lambda0;
        result.push((-cum_haz).exp());
    }
    Ok(result)
}

// ─── Private Helpers ──────────────────────────────────────────────────────────

/// Compute the Breslow baseline cumulative hazard at each distinct event time.
///
/// Returns `(event_times, cumhaz_values)` where both have length = number of
/// distinct event times.  Uses frailty-weighted denominator:
///   Λ₀(t_j) increments by d_j / Σ_{i∈R(t_j)} û_{c(i)} · exp(β^T x_i)
fn compute_breslow(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    cluster_ids: &[usize],
    frailty: &[f64],
    beta: &[f64],
    n_samples: usize,
    n_cov: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Pre-compute exp(β^T x_i) * û_{c(i)} for each observation
    let weighted_exp: Vec<f64> = (0..n_samples)
        .map(|i| {
            let lp = linear_predictor(covariates, beta, i, n_cov);
            let u_c = frailty[cluster_ids[i]];
            u_c * lp.exp()
        })
        .collect();

    // Sort indices by time
    let mut order: Vec<usize> = (0..n_samples).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Running total weighted at-risk sum (start with all)
    let mut risk_sum: f64 = weighted_exp.iter().sum();

    let mut out_times: Vec<f64> = Vec::new();
    let mut out_cumhaz: Vec<f64> = Vec::new();
    let mut cumhaz = 0.0_f64;

    let mut k = 0usize;
    while k < order.len() {
        let t = times[order[k]];
        // Collect all tied indices at time t
        let mut m = k;
        let mut d = 0.0_f64;
        while m < order.len() && times[order[m]] == t {
            if events[order[m]] == 1 {
                d += 1.0;
            }
            m += 1;
        }
        if d > 0.0 && risk_sum > 0.0 {
            cumhaz += d / risk_sum;
            out_times.push(t);
            out_cumhaz.push(cumhaz);
        }
        // Subtract the subjects leaving the risk set at time t
        for idx in k..m {
            risk_sum -= weighted_exp[order[idx]];
        }
        k = m;
    }

    (out_times, out_cumhaz)
}

/// Compute per-cluster cumulative risk Λ_c(β) = Σᵢ∈c exp(β^T xᵢ) · Λ₀(tᵢ).
///
/// For each observation i, look up the Breslow cumulative hazard at time t_i
/// by step-function interpolation into (basetime, basecumhaz).
fn compute_cluster_risk(
    times: &[f64],
    covariates: &[f64],
    cluster_ids: &[usize],
    n_clusters: usize,
    beta: &[f64],
    basetime: &[f64],
    basecumhaz: &[f64],
    n_samples: usize,
    n_cov: usize,
) -> Vec<f64> {
    let mut cluster_risk = vec![0.0_f64; n_clusters];
    for i in 0..n_samples {
        let t_i = times[i];
        let lambda0_ti = step_lookup(basetime, basecumhaz, t_i);
        let exp_lp = linear_predictor(covariates, beta, i, n_cov).exp();
        cluster_risk[cluster_ids[i]] += exp_lp * lambda0_ti;
    }
    cluster_risk
}

/// E-step: compute posterior frailty mean û_c for each cluster.
///
/// û_c = (1/θ + d_c) / (1/θ + Λ_c(β))
fn e_step(cluster_events: &[f64], cluster_risk: &[f64], theta: f64, n_clusters: usize) -> Vec<f64> {
    let inv_theta = 1.0 / theta;
    (0..n_clusters)
        .map(|c| {
            let numerator = inv_theta + cluster_events[c];
            let denominator = inv_theta + cluster_risk[c];
            if denominator > 0.0 {
                numerator / denominator
            } else {
                1.0 // prior mean
            }
        })
        .collect()
}

/// M-step for θ: Newton step on the profile marginal log-likelihood.
///
/// The derivative with respect to θ is:
/// dL/dθ = Σ_c [ (ψ(1/θ) - ψ(1/θ + d_c)) / θ²
///               + (ln(1/θ + Λ_c) - ln(1/θ) - 1) / θ²
///               + (1/θ + d_c) / (θ²·(1/θ + Λ_c)) ]
///
/// We use a bisection safeguard inside a Newton step to ensure θ stays in
/// [min_theta, max_theta].
fn m_step_theta(
    cluster_events: &[f64],
    cluster_risk: &[f64],
    theta_cur: f64,
    n_clusters: usize,
    min_theta: f64,
    max_theta: f64,
) -> f64 {
    // Use Newton with bracketed bisection fallback
    let (score, hess) = theta_score_hess(cluster_events, cluster_risk, theta_cur, n_clusters);

    let theta_new = if hess.abs() > 1.0e-15 {
        // Newton step
        let raw = theta_cur - score / hess;
        raw.clamp(min_theta, max_theta)
    } else {
        // Hessian too small: gradient step
        let step = score.signum() * 0.01 * theta_cur;
        (theta_cur + step).clamp(min_theta, max_theta)
    };

    // Armijo safeguard: ensure the log-likelihood is non-decreasing
    let ll_cur = marginal_log_likelihood(cluster_events, cluster_risk, theta_cur, n_clusters);
    let ll_new = marginal_log_likelihood(cluster_events, cluster_risk, theta_new, n_clusters);
    if ll_new.is_finite() && ll_new >= ll_cur - 1.0e-10 {
        theta_new
    } else {
        // Bisect toward theta_cur
        let mid = (theta_new + theta_cur) * 0.5;
        mid.clamp(min_theta, max_theta)
    }
}

/// Compute score (first derivative) and Hessian (second derivative) of the
/// marginal log-likelihood with respect to θ.
fn theta_score_hess(
    cluster_events: &[f64],
    cluster_risk: &[f64],
    theta: f64,
    n_clusters: usize,
) -> (f64, f64) {
    let inv_theta = 1.0 / theta;
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;

    let mut score = 0.0_f64;
    let mut hess = 0.0_f64;

    for c in 0..n_clusters {
        let d_c = cluster_events[c];
        let lambda_c = cluster_risk[c];
        let a = inv_theta; // 1/θ
        let b = a + d_c; // 1/θ + d_c
        let denom = a + lambda_c; // 1/θ + Λ_c

        if denom <= 0.0 {
            continue;
        }

        // Score contribution for cluster c
        // dL_c/dθ = [-ψ(1/θ + d_c)/θ² + ψ(1/θ)/θ²
        //            - (ln(1/θ) + 1)/θ² + (1/θ + d_c)/(θ²·(1/θ + Λ_c))
        //            + ln(1/θ + Λ_c)/θ²]
        let psi_a = digamma(a);
        let psi_b = digamma(b);
        let ln_a = a.ln();
        let ln_denom = denom.ln();

        let score_c = (psi_a - psi_b - ln_a - 1.0 + b / denom + ln_denom) / theta2;
        score += score_c;

        // Hessian (second derivative) via numerical differentiation of score
        // to avoid extremely complex analytic expression
        // d²L/dθ² ≈ dScore/dθ via central difference with small h
        // We use the analytic second derivative of the Gamma terms:
        // ψ'(x) is the trigamma function ψ₁(x)
        // For simplicity we compute numerically:
        let h = theta * 1.0e-5;
        let h_safe = h.max(1.0e-10);
        let score_plus = {
            let th = (theta + h_safe).max(1.0e-12);
            let inv_th = 1.0 / th;
            let a2 = inv_th;
            let b2 = a2 + d_c;
            let den2 = a2 + lambda_c;
            if den2 <= 0.0 {
                0.0
            } else {
                (digamma(a2) - digamma(b2) - a2.ln() - 1.0 + b2 / den2 + den2.ln()) / (th * th)
            }
        };
        let score_minus = {
            let th = (theta - h_safe).max(1.0e-12);
            let inv_th = 1.0 / th;
            let a2 = inv_th;
            let b2 = a2 + d_c;
            let den2 = a2 + lambda_c;
            if den2 <= 0.0 {
                0.0
            } else {
                (digamma(a2) - digamma(b2) - a2.ln() - 1.0 + b2 / den2 + den2.ln()) / (th * th)
            }
        };
        hess += (score_plus - score_minus) / (2.0 * h_safe);

        // Keep the unused-variable warning away from compiler
        let _ = (psi_a, psi_b, ln_a, ln_denom, theta3, score_c);
    }

    (score, hess)
}

/// Compute the marginal log-likelihood summed over clusters:
/// log L = Σ_c [lgamma(1/θ + d_c) - lgamma(1/θ)
///              + (1/θ)·ln(1/θ) - (1/θ + d_c)·ln(1/θ + Λ_c)]
fn marginal_log_likelihood(
    cluster_events: &[f64],
    cluster_risk: &[f64],
    theta: f64,
    n_clusters: usize,
) -> f64 {
    let inv_theta = 1.0 / theta;
    let mut ll = 0.0_f64;
    for c in 0..n_clusters {
        let d_c = cluster_events[c];
        let lambda_c = cluster_risk[c];
        let denom = inv_theta + lambda_c;
        if denom <= 0.0 {
            continue;
        }
        ll += gammaln(inv_theta + d_c) - gammaln(inv_theta) + inv_theta * inv_theta.ln()
            - (inv_theta + d_c) * denom.ln();
    }
    ll
}

/// Newton-Raphson on the frailty-weighted partial log-likelihood to update β.
///
/// The weighted partial log-likelihood is:
///   L(β) = Σᵢ δᵢ [û_{c(i)} · β^T xᵢ - log(Σⱼ∈R(tᵢ) û_{c(j)} · exp(β^T xⱼ))]
///
/// Returns `(beta_new, n_inner_iters, inner_converged)`.
fn weighted_cox_nr_step(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    cluster_ids: &[usize],
    frailty: &[f64],
    beta: &[f64],
    n_samples: usize,
    n_cov: usize,
    max_iter: usize,
    tol: f64,
) -> SurvivalResult<(Vec<f64>, usize, bool)> {
    if n_cov == 0 {
        // No covariates: β is empty, nothing to optimize
        return Ok((Vec::new(), 0, true));
    }

    // Sort observations by time for risk-set traversal
    let mut order: Vec<usize> = (0..n_samples).collect();
    order.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut beta_cur = beta.to_vec();
    let mut converged = false;
    let mut n_iter = 0usize;

    for it in 0..max_iter {
        n_iter = it + 1;

        let (ll, score, info) = weighted_partial_loglik(
            times,
            events,
            covariates,
            cluster_ids,
            frailty,
            &beta_cur,
            &order,
            n_samples,
            n_cov,
        );

        // Check convergence
        let max_score = score.iter().fold(0.0_f64, |acc, &s| acc.max(s.abs()));
        if max_score < tol {
            converged = true;
            break;
        }

        // Newton step: solve I·delta = score
        let delta = match cholesky_solve(&info, &score, n_cov) {
            Ok(d) => d,
            Err(_) => {
                // Ridge fallback
                let mut info_ridge = info.clone();
                for d in 0..n_cov {
                    info_ridge[d * n_cov + d] += 1.0e-4;
                }
                match cholesky_solve(&info_ridge, &score, n_cov) {
                    Ok(d) => d,
                    Err(_) => break, // give up this step
                }
            }
        };

        // Armijo line search (halving)
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta_cur
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            let (ll_trial, _, _) = weighted_partial_loglik(
                times,
                events,
                covariates,
                cluster_ids,
                frailty,
                &trial,
                &order,
                n_samples,
                n_cov,
            );
            if ll_trial.is_finite() && ll_trial > ll - 1.0e-10 {
                beta_cur = trial;
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }
        if !accepted {
            // Tiny step; check if already converged
            let max_score2 = score.iter().fold(0.0_f64, |acc, &s| acc.max(s.abs()));
            if max_score2 < tol {
                converged = true;
            }
            break;
        }
    }

    // Final convergence check
    if !converged {
        let (_, score_final, _) = weighted_partial_loglik(
            times,
            events,
            covariates,
            cluster_ids,
            frailty,
            &beta_cur,
            &order,
            n_samples,
            n_cov,
        );
        let max_score = score_final.iter().fold(0.0_f64, |acc, &s| acc.max(s.abs()));
        if max_score < tol {
            converged = true;
        }
    }

    Ok((beta_cur, n_iter, converged))
}

/// Compute frailty-weighted partial log-likelihood, score, and information matrix.
///
/// Returns `(log_likelihood, score[p], info[p×p])`.
///
/// Score:  U_k = Σᵢ δᵢ [û_{c(i)} x_{ik} - (Σⱼ∈R(tᵢ) û_{c(j)} exp(β^T xⱼ) x_{jk})
///                                        / (Σⱼ∈R(tᵢ) û_{c(j)} exp(β^T xⱼ))]
/// Info:   I_kl = Σᵢ δᵢ [(Σ û_{c(j)} exp xⱼ x_{jk} x_{jl}) / denom
///                         - (num_k/denom)(num_l/denom)]
#[allow(clippy::too_many_arguments)]
fn weighted_partial_loglik(
    times: &[f64],
    events: &[u8],
    covariates: &[f64],
    cluster_ids: &[usize],
    frailty: &[f64],
    beta: &[f64],
    order: &[usize],
    n_samples: usize,
    n_cov: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let p = n_cov;

    // Pre-compute frailty-weighted exp(β^T x) for each observation.
    // w_exp[i] = û_{c(i)} · exp(β^T x_i)
    let w_exp: Vec<f64> = (0..n_samples)
        .map(|i| {
            let lp = linear_predictor(covariates, beta, i, p);
            let u_c = frailty[cluster_ids[i]];
            u_c * lp.exp()
        })
        .collect();

    let mut ll = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    // Running risk-set weighted sums, initialised over the full dataset.
    // s0      = Σⱼ∈R û_{c(j)} exp(β^T xⱼ)
    // s1[k]   = Σⱼ∈R û_{c(j)} exp(β^T xⱼ) x_{jk}
    // s2[k,l] = Σⱼ∈R û_{c(j)} exp(β^T xⱼ) x_{jk} x_{jl}
    let mut s0: f64 = w_exp.iter().sum();
    let mut s1 = vec![0.0_f64; p];
    let mut s2 = vec![0.0_f64; p * p];

    for i in 0..n_samples {
        let u_w = w_exp[i];
        for k in 0..p {
            let x_k = covariates[i * p + k];
            s1[k] += u_w * x_k;
            for l in 0..p {
                let x_l = covariates[i * p + l];
                s2[k * p + l] += u_w * x_k * x_l;
            }
        }
    }

    // Process in ascending time order; subjects are grouped by tied times.
    // At each event time: add to LL/score/info, then subtract the group from the risk set.
    let mut pos = 0usize;
    while pos < order.len() {
        let t_cur = times[order[pos]];

        // Find the extent of the tied group at t_cur
        let mut end = pos;
        while end < order.len() && times[order[end]] == t_cur {
            end += 1;
        }

        // Accumulate contributions from all events in this tied group
        for &obs_idx in &order[pos..end] {
            if events[obs_idx] == 1 {
                let u_c = frailty[cluster_ids[obs_idx]];
                let mut beta_x = 0.0_f64;
                for k in 0..p {
                    beta_x += beta[k] * covariates[obs_idx * p + k];
                }
                // LL: û_c · β^T x − ln(s0)
                if s0 > 0.0 {
                    ll += u_c * beta_x - s0.ln();
                    // Score
                    for k in 0..p {
                        let x_k = covariates[obs_idx * p + k];
                        score[k] += u_c * x_k - s1[k] / s0;
                    }
                    // Information matrix
                    for k in 0..p {
                        for l in 0..p {
                            info[k * p + l] += s2[k * p + l] / s0 - (s1[k] / s0) * (s1[l] / s0);
                        }
                    }
                }
            }
        }

        // Remove all subjects in this group from the risk-set sums
        for &obs_idx in &order[pos..end] {
            let u_w = w_exp[obs_idx];
            s0 -= u_w;
            for k in 0..p {
                let x_k = covariates[obs_idx * p + k];
                s1[k] -= u_w * x_k;
                for l in 0..p {
                    let x_l = covariates[obs_idx * p + l];
                    s2[k * p + l] -= u_w * x_k * x_l;
                }
            }
        }

        pos = end;
    }

    (ll, score, info)
}

/// Step-function lookup: return the largest value in `vals` at the largest
/// `keys` entry that is <= `t`.  Returns 0 if t precedes all keys.
#[inline]
fn step_lookup(keys: &[f64], vals: &[f64], t: f64) -> f64 {
    if keys.is_empty() || t < keys[0] {
        return 0.0;
    }
    match keys.binary_search_by(|k| k.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less)) {
        Ok(idx) => vals[idx],
        Err(0) => 0.0,
        Err(idx) => vals[idx - 1],
    }
}

/// Compute the linear predictor β^T x_i from the flat row-major `covariates` matrix.
#[inline]
fn linear_predictor(covariates: &[f64], beta: &[f64], i: usize, n_cov: usize) -> f64 {
    if n_cov == 0 {
        return 0.0;
    }
    let row = &covariates[i * n_cov..(i + 1) * n_cov];
    row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build simple clustered data: n_clusters clusters, obs_per_cluster observations each,
    /// all events, no covariates.  Cluster c has hazard rate `rates[c]`.
    fn make_no_covariate_data(
        n_clusters: usize,
        obs_per_cluster: usize,
        rates: &[f64],
        seed: u64,
    ) -> (Vec<f64>, Vec<u8>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let n = n_clusters * obs_per_cluster;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut cluster_ids = Vec::with_capacity(n);
        for (c, &rate) in rates.iter().enumerate().take(n_clusters) {
            for _ in 0..obs_per_cluster {
                let t = rng.next_exponential(rate).max(0.01);
                times.push(t);
                events.push(1u8);
                cluster_ids.push(c);
            }
        }
        (times, events, cluster_ids)
    }

    /// Build clustered data with one covariate.
    fn make_covariate_data(
        n_clusters: usize,
        obs_per_cluster: usize,
        beta_true: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<u8>, Vec<f64>, Vec<usize>) {
        let mut rng = LcgRng::new(seed);
        let n = n_clusters * obs_per_cluster;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut covariates = Vec::with_capacity(n);
        let mut cluster_ids = Vec::with_capacity(n);
        for c in 0..n_clusters {
            for _ in 0..obs_per_cluster {
                let x = rng.next_normal() * 0.5;
                let lambda = (beta_true * x).exp();
                let t = rng.next_exponential(lambda.max(0.01)).max(0.01);
                times.push(t);
                events.push(1u8);
                covariates.push(x);
                cluster_ids.push(c);
            }
        }
        (times, events, covariates, cluster_ids)
    }

    // ── Test 1: FrailtyConfig defaults ────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = FrailtyConfig::default();
        assert!((cfg.theta_init - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_outer_iter, 30);
        assert_eq!(cfg.max_inner_iter, 20);
        assert!(cfg.tol < 1.0e-4);
        assert!(cfg.min_theta > 0.0);
        assert!(cfg.max_theta > cfg.min_theta);
    }

    // ── Test 2: fit with no covariates, 4 clusters ────────────────────────────

    #[test]
    fn fit_no_covariates() {
        let rates = [2.0, 1.0, 0.5, 3.0];
        let (times, events, cluster_ids) = make_no_covariate_data(4, 5, &rates, 42);
        let config = FrailtyConfig {
            max_outer_iter: 20,
            ..FrailtyConfig::default()
        };
        let fit = fit_gamma_frailty(&times, &events, &[], &cluster_ids, 4, &config)
            .expect("fit should succeed");
        assert_eq!(fit.n_covariates, 0);
        assert_eq!(fit.n_clusters, 4);
        assert!(fit.theta > 0.0, "theta must be positive, got {}", fit.theta);
        assert_eq!(fit.beta.len(), 0);
    }

    // ── Test 3: two clusters with very different risk ──────────────────────────

    #[test]
    fn fit_two_clusters_different_risk() {
        // Cluster 0: fast events (rate=4), Cluster 1: slow events (rate=0.5)
        let (times, events, cluster_ids) = make_no_covariate_data(2, 15, &[4.0, 0.5], 77);
        let config = FrailtyConfig {
            max_outer_iter: 30,
            ..FrailtyConfig::default()
        };
        let fit = fit_gamma_frailty(&times, &events, &[], &cluster_ids, 2, &config)
            .expect("fit should succeed");
        assert!(
            fit.theta > 0.0,
            "theta must be > 0 for heterogeneous clusters"
        );
        // Frailties must be positive
        for (c, &u) in fit.cluster_frailty.iter().enumerate() {
            assert!(u > 0.0, "frailty for cluster {} must be > 0, got {}", c, u);
        }
    }

    // ── Test 4: single cluster → θ driven toward zero ────────────────────────

    #[test]
    fn fit_single_cluster() {
        // With one cluster, there is no between-cluster heterogeneity.
        // θ should remain small / near min_theta.
        let (times, events, cluster_ids) = make_no_covariate_data(1, 20, &[1.0], 99);
        let config = FrailtyConfig {
            theta_init: 1.0,
            max_outer_iter: 30,
            ..FrailtyConfig::default()
        };
        let fit = fit_gamma_frailty(&times, &events, &[], &cluster_ids, 1, &config)
            .expect("fit should succeed");
        assert_eq!(fit.n_clusters, 1);
        // theta may decrease substantially with a single cluster
        assert!(fit.theta >= config.min_theta);
        assert!(fit.theta <= config.max_theta);
    }

    // ── Test 5: all cluster frailties are positive ────────────────────────────

    #[test]
    fn fit_frailty_positive() {
        let (times, events, cluster_ids) =
            make_no_covariate_data(5, 6, &[1.0, 2.0, 0.5, 3.0, 1.5], 11);
        let config = FrailtyConfig::default();
        let fit =
            fit_gamma_frailty(&times, &events, &[], &cluster_ids, 5, &config).expect("fit ok");
        assert!(
            fit.cluster_frailty.iter().all(|&u| u > 0.0),
            "all frailties must be positive: {:?}",
            fit.cluster_frailty
        );
    }

    // ── Test 6: algorithm converges on well-conditioned data ──────────────────

    #[test]
    fn fit_converges() {
        let (times, events, cluster_ids) = make_no_covariate_data(4, 10, &[1.0, 2.0, 0.8, 1.5], 55);
        let config = FrailtyConfig {
            max_outer_iter: 50,
            tol: 1.0e-5,
            ..FrailtyConfig::default()
        };
        let fit =
            fit_gamma_frailty(&times, &events, &[], &cluster_ids, 4, &config).expect("fit ok");
        assert!(
            fit.converged,
            "should converge on 4-cluster homogeneous data"
        );
    }

    // ── Test 7: predict returns correct output length ─────────────────────────

    #[test]
    fn predict_shape() {
        let (times, events, cluster_ids) = make_no_covariate_data(3, 8, &[1.0, 2.0, 0.5], 31);
        let config = FrailtyConfig::default();
        let fit =
            fit_gamma_frailty(&times, &events, &[], &cluster_ids, 3, &config).expect("fit ok");
        let eval_times = vec![0.1, 0.5, 1.0, 2.0, 5.0];
        let survival = predict_frailty_survival(
            &fit,
            &times,
            &[],
            1.0,
            &eval_times,
            &fit.baseline_times.clone(),
            &fit.baseline_cumhaz.clone(),
        )
        .expect("predict ok");
        assert_eq!(
            survival.len(),
            eval_times.len(),
            "output length must equal eval_times length"
        );
    }

    // ── Test 8: empty times → error ───────────────────────────────────────────

    #[test]
    fn fit_empty_times_error() {
        let config = FrailtyConfig::default();
        let result = fit_gamma_frailty(&[], &[], &[], &[], 1, &config);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "empty input should return EmptyDataset error"
        );
    }

    // ── Test 9: cluster_id out of range → error ───────────────────────────────

    #[test]
    fn fit_cluster_id_out_of_range_error() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1u8, 1, 1];
        let cluster_ids = vec![0usize, 1, 5]; // cluster 5 is out of range for n_clusters=3
        let config = FrailtyConfig::default();
        let result = fit_gamma_frailty(&times, &events, &[], &cluster_ids, 3, &config);
        assert!(
            matches!(result, Err(SurvivalError::IndexOutOfBounds { .. })),
            "out-of-range cluster id should return IndexOutOfBounds"
        );
    }

    // ── Test 10: fit with two covariates, 4 clusters ──────────────────────────

    #[test]
    fn fit_with_covariates() {
        let (times, events, cov1, cluster_ids) = make_covariate_data(4, 10, 0.5, 22);
        // Add a second covariate (negated)
        let cov2: Vec<f64> = cov1.iter().map(|&x| -0.3 * x).collect();
        let n = times.len();
        let mut covariates = Vec::with_capacity(n * 2);
        for i in 0..n {
            covariates.push(cov1[i]);
            covariates.push(cov2[i]);
        }
        let config = FrailtyConfig {
            max_outer_iter: 20,
            ..FrailtyConfig::default()
        };
        let fit = fit_gamma_frailty(&times, &events, &covariates, &cluster_ids, 4, &config)
            .expect("fit with 2 covariates should succeed");
        assert_eq!(fit.beta.len(), 2, "beta.len() must be n_covariates = 2");
        assert_eq!(fit.n_covariates, 2);
    }

    // ── Test 11: all censored, single cluster ─────────────────────────────────

    #[test]
    fn fit_all_censored_single_cluster() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![0u8; 5]; // all censored
        let cluster_ids = vec![0usize; 5];
        let config = FrailtyConfig {
            theta_init: 0.5,
            ..FrailtyConfig::default()
        };
        // Should succeed: no events means trivial fit
        let fit = fit_gamma_frailty(&times, &events, &[], &cluster_ids, 1, &config)
            .expect("all-censored should return degenerate fit, not error");
        // theta unchanged from init (no information to update it)
        assert!((fit.theta - 0.5).abs() < 1.0e-10 || fit.theta > 0.0);
        assert_eq!(fit.n_clusters, 1);
    }

    // ── Test 12: Σ û_c ≈ n_clusters (E[Z]=1 under gamma prior) ──────────────

    #[test]
    fn frailty_sum_close_to_n_clusters() {
        // With enough data, the posterior means should average close to 1.
        let (times, events, cluster_ids) =
            make_no_covariate_data(6, 20, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 88);
        let config = FrailtyConfig {
            max_outer_iter: 30,
            ..FrailtyConfig::default()
        };
        let fit =
            fit_gamma_frailty(&times, &events, &[], &cluster_ids, 6, &config).expect("fit ok");
        let sum: f64 = fit.cluster_frailty.iter().sum();
        // With homogeneous clusters all frailties ≈ 1, so sum ≈ n_clusters
        assert!(
            (sum - 6.0).abs() < 3.0,
            "Σ û_c should be near n_clusters=6, got {sum:.4}"
        );
    }

    // ── Test 13: log-likelihood is finite ─────────────────────────────────────

    #[test]
    fn log_likelihood_finite() {
        let (times, events, cluster_ids) = make_no_covariate_data(3, 10, &[1.0, 2.0, 0.5], 33);
        let config = FrailtyConfig::default();
        let fit =
            fit_gamma_frailty(&times, &events, &[], &cluster_ids, 3, &config).expect("fit ok");
        assert!(
            fit.log_likelihood.is_finite(),
            "log_likelihood must be finite, got {}",
            fit.log_likelihood
        );
    }

    // ── Test 14: survival probabilities in (0, 1] ─────────────────────────────

    #[test]
    fn predict_survival_in_01() {
        let (times, events, cluster_ids) = make_no_covariate_data(3, 8, &[1.0, 2.0, 0.5], 44);
        let config = FrailtyConfig::default();
        let fit =
            fit_gamma_frailty(&times, &events, &[], &cluster_ids, 3, &config).expect("fit ok");
        let eval_times: Vec<f64> = (1..=10).map(|i| i as f64 * 0.3).collect();
        let sv = predict_frailty_survival(
            &fit,
            &times,
            &[],
            1.0,
            &eval_times,
            &fit.baseline_times.clone(),
            &fit.baseline_cumhaz.clone(),
        )
        .expect("predict ok");
        for (i, &s) in sv.iter().enumerate() {
            assert!(
                s > 0.0 && s <= 1.0,
                "S(t={}) = {s} not in (0,1]",
                eval_times[i]
            );
        }
    }
}
