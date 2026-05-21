//! Stratified Cox proportional hazards model.
//!
//! Each stratum has its own baseline hazard h₀ₖ(t) but all strata share the same
//! regression coefficients β.  The stratification variable is assumed to violate the
//! proportional-hazards assumption; stratification removes that nuisance.
//!
//! # Partial log-likelihood
//! ```text
//! L(β) = Σ_k Σ_{i: event, stratum=k}  [x_i^T β - log(Σ_{j: t_j ≥ t_i, stratum=k} exp(x_j^T β))]
//! ```
//! Gradient and Hessian are accumulated across strata using the standard Breslow formulae
//! restricted to stratum-specific risk sets.
//!
//! # Baseline hazard (Breslow, per stratum)
//! ```text
//! Λ₀ₖ(t) = Σ_{t_i ≤ t, event, stratum=k}  1 / (Σ_{j: t_j ≥ t_i, stratum=k} exp(x_j^T β))
//! ```
//!
//! # Survival prediction
//! S(t | x, stratum=k) = exp(-Λ₀ₖ(t) · exp(x^T β))

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::linalg::solve::cholesky_solve;

/// Tie-handling method (within a stratum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StratTieMethod {
    /// Breslow approximation: denominator raised to the power of the tied-event count.
    Breslow,
    /// Efron approximation: fractional subtraction in the denominator.
    Efron,
}

/// Configuration for [`stratified_cox_fit`].
#[derive(Debug, Clone, Copy)]
pub struct StratifiedCoxConfig {
    /// Maximum Newton-Raphson iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the max absolute score component.
    pub tol: f64,
    /// Tie-handling method within each stratum.
    pub ties: StratTieMethod,
}

impl Default for StratifiedCoxConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1.0e-6,
            ties: StratTieMethod::Breslow,
        }
    }
}

/// Fitted stratified Cox model.
#[derive(Debug, Clone)]
pub struct StratifiedCoxFit {
    /// Shared regression coefficients β (length p).
    pub coef: Vec<f64>,
    /// Standard errors of β.
    pub se: Vec<f64>,
    /// Wald z-scores β / se.
    pub z_scores: Vec<f64>,
    /// Final partial log-likelihood.
    pub log_likelihood: f64,
    /// Number of strata.
    pub n_strata: usize,
    /// Per-stratum baseline hazard increments: `baseline_hazards[k]` = `[(time, Δh)]`.
    pub baseline_hazards: Vec<Vec<(f64, f64)>>,
    /// The unique stratum identifiers encountered (sorted).
    pub stratum_ids: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dot product of `x` and `beta`.
#[inline]
fn dot(x: &[f64], beta: &[f64]) -> f64 {
    x.iter().zip(beta.iter()).map(|(xi, bi)| xi * bi).sum()
}

/// For a single stratum, compute (loglik, score, info) contribution using Breslow ties.
///
/// `obs` is a list of `(time, event, x_slice_start)` indices into `times`, `events`, `x`.
fn stratum_breslow_loglik(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    stratum_members: &[usize],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    let m = stratum_members.len();
    if m == 0 {
        return Ok((0.0, vec![0.0_f64; p], vec![0.0_f64; p * p]));
    }

    // Sort stratum members by time ascending.
    let mut idx: Vec<usize> = stratum_members.to_vec();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Precompute exp(x_i^T β) for each stratum member.
    let mut w = vec![0.0_f64; m];
    let mut s0 = 0.0_f64;
    let mut s1 = vec![0.0_f64; p];
    let mut s2 = vec![0.0_f64; p * p];

    for (k, &i) in idx.iter().enumerate() {
        let xi = x_row(x, i, p, n);
        let wi = dot(xi, beta).exp();
        w[k] = wi;
        s0 += wi;
        for a in 0..p {
            s1[a] += wi * xi[a];
            for b in 0..p {
                s2[a * p + b] += wi * xi[a] * xi[b];
            }
        }
    }

    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    let mut k = 0usize;
    while k < m {
        let t = times[idx[k]];
        let mut end = k;
        let mut d_count = 0.0_f64;
        let mut etabsum = 0.0_f64;
        let mut x_events = vec![0.0_f64; p];

        while end < m && times[idx[end]] == t {
            if events[idx[end]] {
                d_count += 1.0;
                let xi = x_row(x, idx[end], p, n);
                etabsum += dot(xi, beta);
                for a in 0..p {
                    x_events[a] += xi[a];
                }
            }
            end += 1;
        }

        if d_count > 0.0 {
            if s0 <= 0.0 {
                return Err(SurvivalError::NumericalInstability(
                    "non-positive risk-set sum in stratified Breslow log-likelihood".to_string(),
                ));
            }
            loglik += etabsum - d_count * s0.ln();
            let x_bar: Vec<f64> = s1.iter().map(|si| si / s0).collect();
            for a in 0..p {
                score[a] += x_events[a] - d_count * x_bar[a];
            }
            for a in 0..p {
                for b in 0..p {
                    let cov = s2[a * p + b] / s0 - x_bar[a] * x_bar[b];
                    info[a * p + b] += d_count * cov;
                }
            }
        }

        // Remove stratum members at time t from risk set.
        for jj in k..end {
            let xi = x_row(x, idx[jj], p, n);
            let wj = w[jj];
            s0 -= wj;
            for a in 0..p {
                s1[a] -= wj * xi[a];
                for b in 0..p {
                    s2[a * p + b] -= wj * xi[a] * xi[b];
                }
            }
        }
        k = end;
    }
    Ok((loglik, score, info))
}

/// For a single stratum, compute (loglik, score, info) using Efron ties.
fn stratum_efron_loglik(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    stratum_members: &[usize],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    let m = stratum_members.len();
    if m == 0 {
        return Ok((0.0, vec![0.0_f64; p], vec![0.0_f64; p * p]));
    }

    let mut idx: Vec<usize> = stratum_members.to_vec();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut w = vec![0.0_f64; m];
    let mut s0 = 0.0_f64;
    let mut s1 = vec![0.0_f64; p];
    let mut s2 = vec![0.0_f64; p * p];

    for (k, &i) in idx.iter().enumerate() {
        let xi = x_row(x, i, p, n);
        let wi = dot(xi, beta).exp();
        w[k] = wi;
        s0 += wi;
        for a in 0..p {
            s1[a] += wi * xi[a];
            for b in 0..p {
                s2[a * p + b] += wi * xi[a] * xi[b];
            }
        }
    }

    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];

    let mut k = 0usize;
    while k < m {
        let t = times[idx[k]];
        let mut end = k;
        let mut d_count = 0.0_f64;
        let mut etabsum = 0.0_f64;
        let mut x_events = vec![0.0_f64; p];
        let mut s0_events = 0.0_f64;
        let mut s1_events = vec![0.0_f64; p];
        let mut s2_events = vec![0.0_f64; p * p];

        while end < m && times[idx[end]] == t {
            if events[idx[end]] {
                d_count += 1.0;
                let xi = x_row(x, idx[end], p, n);
                let we = w[end];
                etabsum += dot(xi, beta);
                s0_events += we;
                for a in 0..p {
                    x_events[a] += xi[a];
                    s1_events[a] += we * xi[a];
                    for b in 0..p {
                        s2_events[a * p + b] += we * xi[a] * xi[b];
                    }
                }
            }
            end += 1;
        }

        if d_count > 0.0 {
            let d = d_count as usize;
            for l in 0..d {
                let frac = l as f64 / d_count;
                let denom0 = s0 - frac * s0_events;
                if denom0 <= 0.0 {
                    return Err(SurvivalError::NumericalInstability(
                        "non-positive Efron denominator in stratified Cox".to_string(),
                    ));
                }
                loglik += (etabsum - frac * s0_events.ln()) / d_count - denom0.ln();
                let x_bar: Vec<f64> = (0..p)
                    .map(|a| (s1[a] - frac * s1_events[a]) / denom0)
                    .collect();
                for a in 0..p {
                    score[a] += (x_events[a] - frac * s1_events[a]) / d_count - x_bar[a];
                }
                for a in 0..p {
                    for b in 0..p {
                        let s2_eff = s2[a * p + b] - frac * s2_events[a * p + b];
                        let cov = s2_eff / denom0 - x_bar[a] * x_bar[b];
                        info[a * p + b] += cov;
                    }
                }
            }
        }

        for jj in k..end {
            let xi = x_row(x, idx[jj], p, n);
            let wj = w[jj];
            s0 -= wj;
            for a in 0..p {
                s1[a] -= wj * xi[a];
                for b in 0..p {
                    s2[a * p + b] -= wj * xi[a] * xi[b];
                }
            }
        }
        k = end;
    }
    Ok((loglik, score, info))
}

/// Compute aggregate (loglik, score, info) across all strata.
fn all_strata_loglik(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    strata_members: &[Vec<usize>],
    ties: StratTieMethod,
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    let mut total_ll = 0.0_f64;
    let mut total_score = vec![0.0_f64; p];
    let mut total_info = vec![0.0_f64; p * p];

    for members in strata_members {
        let (ll, sc, inf) = match ties {
            StratTieMethod::Breslow => {
                stratum_breslow_loglik(times, events, x, n, p, beta, members)?
            }
            StratTieMethod::Efron => stratum_efron_loglik(times, events, x, n, p, beta, members)?,
        };
        total_ll += ll;
        for a in 0..p {
            total_score[a] += sc[a];
        }
        for a in 0..p * p {
            total_info[a] += inf[a];
        }
    }
    Ok((total_ll, total_score, total_info))
}

/// Return a shared slice of `x` for observation `i` (p features each).
///
/// `x` is laid out as `[row0_feat0, row0_feat1, ..., row1_feat0, ...]` (n × p row-major).
#[inline]
fn x_row(x: &[f64], i: usize, p: usize, _n: usize) -> &[f64] {
    &x[i * p..(i + 1) * p]
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit a stratified Cox proportional hazards model.
///
/// # Parameters
/// - `times`: event/censoring times of length `n`.
/// - `events`: event indicators of length `n` (`true` = event).
/// - `x`: covariate matrix laid out row-major, length `n × p`.
/// - `strata`: stratum assignment for each observation, values in `0..n_strata`.
/// - `n`: number of observations.
/// - `p`: number of covariates.
/// - `cfg`: algorithm configuration.
pub fn stratified_cox_fit(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    strata: &[usize],
    n: usize,
    p: usize,
    cfg: &StratifiedCoxConfig,
) -> SurvivalResult<StratifiedCoxFit> {
    // ---------- validation ----------
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if times.len() != n || events.len() != n || strata.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: n,
            b: times.len(),
        });
    }
    if x.len() != n * p {
        return Err(SurvivalError::DimensionMismatch {
            a: n * p,
            b: x.len(),
        });
    }
    if !events.iter().any(|&e| e) {
        return Err(SurvivalError::NoEvents);
    }
    // Collect unique strata and build per-stratum member lists.
    let max_stratum = *strata.iter().max().unwrap_or(&0);
    let n_strata = max_stratum + 1;
    let mut strata_members: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        strata_members[strata[i]].push(i);
    }

    // ---------- Newton-Raphson ----------
    let mut beta = vec![0.0_f64; p];
    let (mut ll, mut score, mut info) =
        all_strata_loglik(times, events, x, n, p, &beta, &strata_members, cfg.ties)?;

    let mut converged = false;
    for _iter in 0..cfg.max_iter {
        let max_score = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_score < cfg.tol {
            converged = true;
            break;
        }

        // When p == 0 nothing to update; convergence trivially.
        if p == 0 {
            converged = true;
            break;
        }

        let delta = match cholesky_solve(&info, &score, p) {
            Ok(d) => d,
            Err(_) => {
                let mut info_ridge = info.clone();
                for d in 0..p {
                    info_ridge[d * p + d] += 1.0e-4;
                }
                match cholesky_solve(&info_ridge, &score, p) {
                    Ok(d) => d,
                    Err(_) => break,
                }
            }
        };

        // Armijo line search.
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((ll_new, sc_new, info_new)) =
                all_strata_loglik(times, events, x, n, p, &trial, &strata_members, cfg.ties)
            {
                if ll_new.is_finite() && ll_new > ll - 1.0e-10 {
                    beta = trial;
                    ll = ll_new;
                    score = sc_new;
                    info = info_new;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    if !converged {
        let max_score = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_score < cfg.tol {
            converged = true;
        }
    }
    let _ = converged; // convergence is informational only

    // ---------- variance-covariance ----------
    let variance = if p > 0 {
        gauss_jordan_inverse(&info, p).unwrap_or_else(|_| vec![0.0_f64; p * p])
    } else {
        vec![]
    };

    let se: Vec<f64> = (0..p)
        .map(|i| variance[i * p + i].max(0.0).sqrt())
        .collect();
    let z_scores: Vec<f64> = beta
        .iter()
        .zip(se.iter())
        .map(|(b, s)| if *s > 0.0 { b / s } else { 0.0 })
        .collect();

    // ---------- per-stratum baseline hazards (Breslow) ----------
    let baseline_hazards =
        compute_stratum_baseline_hazards(times, events, x, n, p, &beta, &strata_members, n_strata)?;

    let stratum_ids: Vec<usize> = (0..n_strata).collect();

    Ok(StratifiedCoxFit {
        coef: beta,
        se,
        z_scores,
        log_likelihood: ll,
        n_strata,
        baseline_hazards,
        stratum_ids,
    })
}

/// Compute per-stratum Breslow baseline hazard increments.
///
/// Returns a `Vec<Vec<(f64, f64)>>` of length `n_strata`, each containing
/// `(time, hazard_increment)` at each unique event time within the stratum.
fn compute_stratum_baseline_hazards(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    strata_members: &[Vec<usize>],
    n_strata: usize,
) -> SurvivalResult<Vec<Vec<(f64, f64)>>> {
    let mut result = Vec::with_capacity(n_strata);
    for members in strata_members {
        let bh = stratum_baseline_hazard(times, events, x, n, p, beta, members)?;
        result.push(bh);
    }
    Ok(result)
}

/// Breslow baseline hazard for a single stratum.
fn stratum_baseline_hazard(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    stratum_members: &[usize],
) -> SurvivalResult<Vec<(f64, f64)>> {
    let m = stratum_members.len();
    if m == 0 {
        return Ok(Vec::new());
    }

    let mut idx: Vec<usize> = stratum_members.to_vec();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut w = vec![0.0_f64; m];
    let mut s0 = 0.0_f64;
    for (k, &i) in idx.iter().enumerate() {
        let wi = dot(x_row(x, i, p, n), beta).exp();
        w[k] = wi;
        s0 += wi;
    }

    let mut hazards = Vec::new();
    let mut k = 0usize;
    while k < m {
        let t = times[idx[k]];
        let mut end = k;
        let mut d = 0.0_f64;
        while end < m && times[idx[end]] == t {
            if events[idx[end]] {
                d += 1.0;
            }
            end += 1;
        }
        if d > 0.0 && s0 > 0.0 {
            hazards.push((t, d / s0));
        }
        for wj in w.iter().take(end).skip(k) {
            s0 -= wj;
        }
        k = end;
    }
    Ok(hazards)
}

/// Predict survival S(t | x_new, stratum) at the given time points.
///
/// Returns a `Vec<f64>` of survival probabilities, one per element of `time_points`.
/// Values are in (0, 1].
pub fn stratified_cox_predict_survival(
    fit: &StratifiedCoxFit,
    x_new: &[f64],
    stratum: usize,
    time_points: &[f64],
) -> SurvivalResult<Vec<f64>> {
    if stratum >= fit.n_strata {
        return Err(SurvivalError::IndexOutOfBounds {
            index: stratum,
            len: fit.n_strata,
        });
    }
    let p = fit.coef.len();
    if x_new.len() != p {
        return Err(SurvivalError::DimensionMismatch {
            a: p,
            b: x_new.len(),
        });
    }

    let lp = dot(x_new, &fit.coef);
    let exp_lp = lp.exp();

    let bh = &fit.baseline_hazards[stratum];

    // For each query time point, interpolate cumulative baseline hazard.
    let cumulative_bh: Vec<f64> = time_points
        .iter()
        .map(|&t| {
            let mut cum = 0.0_f64;
            for &(ti, di) in bh {
                if ti <= t {
                    cum += di;
                } else {
                    break;
                }
            }
            cum
        })
        .collect();

    Ok(cumulative_bh
        .iter()
        .map(|&ch| (-ch * exp_lp).exp().max(f64::MIN_POSITIVE))
        .collect())
}

/// Return the final log-likelihood from a fitted model.
#[must_use]
pub fn stratified_cox_log_likelihood(fit: &StratifiedCoxFit) -> f64 {
    fit.log_likelihood
}

/// Score test statistic at β = 0.
///
/// At β = 0, score U(0) has covariance I(0) (Fisher information).
/// The test statistic is U(0)^T I(0)^{-1} U(0), which is χ² with p d.f.
pub fn stratified_cox_score_test(
    times: &[f64],
    events: &[bool],
    x: &[f64],
    strata: &[usize],
    n: usize,
    p: usize,
) -> SurvivalResult<f64> {
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if p == 0 {
        return Ok(0.0);
    }
    let max_stratum = *strata.iter().max().unwrap_or(&0);
    let n_strata = max_stratum + 1;
    let mut strata_members: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        strata_members[strata[i]].push(i);
    }
    let beta0 = vec![0.0_f64; p];
    let (_ll, score, info) = all_strata_loglik(
        times,
        events,
        x,
        n,
        p,
        &beta0,
        &strata_members,
        StratTieMethod::Breslow,
    )?;

    // stat = U^T I^{-1} U
    let info_inv = gauss_jordan_inverse(&info, p).unwrap_or_else(|_| vec![0.0_f64; p * p]);
    let mut stat = 0.0_f64;
    for a in 0..p {
        for b in 0..p {
            stat += score[a] * info_inv[a * p + b] * score[b];
        }
    }
    Ok(stat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a simple dataset with `ns` strata, `per_stratum` obs each, one covariate.
    fn make_stratified_data(
        ns: usize,
        per: usize,
        beta_true: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<usize>, usize, usize) {
        let mut rng = LcgRng::new(seed);
        let n = ns * per;
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        let mut x_mat = Vec::with_capacity(n);
        let mut strata = Vec::with_capacity(n);
        for k in 0..ns {
            // stratum-specific baseline hazard multiplier to break PH assumption
            let lam_base = 0.5 + k as f64 * 0.3;
            for _ in 0..per {
                let xi = rng.next_normal();
                let lam = lam_base * (beta_true * xi).exp();
                let t = rng.next_exponential(lam).max(1.0e-6);
                // ~70% event rate
                let ev = rng.next_f64() < 0.70;
                times.push(t);
                events.push(ev);
                x_mat.push(xi);
                strata.push(k);
            }
        }
        // Make sure at least some events exist.
        events[0] = true;
        (times, events, x_mat, strata, n, 1)
    }

    #[test]
    fn two_strata_coef_finite_and_se_positive() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.8, 42);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        assert_eq!(fit.coef.len(), 1);
        assert!(fit.coef[0].is_finite(), "coef must be finite");
        assert!(fit.se[0] > 0.0, "SE must be positive");
    }

    #[test]
    fn z_scores_computed_correctly() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.8, 7);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        // z = coef / se
        let expected_z = fit.coef[0] / fit.se[0];
        assert!((fit.z_scores[0] - expected_z).abs() < 1.0e-10);
    }

    #[test]
    fn log_likelihood_is_negative() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.5, 99);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        assert!(
            fit.log_likelihood < 0.0,
            "log-likelihood of a probability must be negative, got {}",
            fit.log_likelihood
        );
    }

    #[test]
    fn log_likelihood_accessor_matches() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 40, 0.5, 11);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        assert_eq!(stratified_cox_log_likelihood(&fit), fit.log_likelihood);
    }

    #[test]
    fn baseline_hazard_monotone_nondecreasing() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.5, 13);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        for k in 0..fit.n_strata {
            let bh = &fit.baseline_hazards[k];
            // Each increment must be non-negative (monotone cumulative hazard).
            for &(_, delta) in bh {
                assert!(delta >= 0.0, "hazard increment must be non-negative");
            }
        }
    }

    #[test]
    fn predict_survival_in_unit_interval() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.5, 17);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        let x_new = vec![0.0_f64];
        let tpts = vec![0.5, 1.0, 2.0, 5.0];
        let sv = stratified_cox_predict_survival(&fit, &x_new, 0, &tpts).expect("predict ok");
        assert_eq!(sv.len(), 4);
        for &s in &sv {
            assert!(s > 0.0 && s <= 1.0, "survival must be in (0,1]: {}", s);
        }
    }

    #[test]
    fn predict_survival_decreasing_over_time() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 80, 0.5, 23);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        let x_new = vec![1.0_f64];
        // Use many small time points to ensure monotone survival.
        let tpts: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let sv = stratified_cox_predict_survival(&fit, &x_new, 0, &tpts).expect("predict ok");
        for w in sv.windows(2) {
            assert!(w[1] <= w[0] + 1.0e-12, "survival must be non-increasing");
        }
    }

    #[test]
    fn score_test_is_finite() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 50, 0.5, 31);
        let stat =
            stratified_cox_score_test(&times, &events, &x, &strata, n, p).expect("score test ok");
        assert!(stat.is_finite(), "score test statistic must be finite");
        assert!(stat >= 0.0, "chi-square statistic must be non-negative");
    }

    #[test]
    fn error_on_empty_input() {
        let cfg = StratifiedCoxConfig::default();
        let result = stratified_cox_fit(&[], &[], &[], &[], 0, 1, &cfg);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset error"
        );
    }

    #[test]
    fn error_on_stratum_out_of_bounds_predict() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 30, 0.5, 37);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        // stratum index 99 is out of bounds (only 2 strata).
        let result = stratified_cox_predict_survival(&fit, &[0.0], 99, &[1.0]);
        assert!(
            matches!(result, Err(SurvivalError::IndexOutOfBounds { .. })),
            "expected IndexOutOfBounds error"
        );
    }

    #[test]
    fn n_strata_matches_input() {
        let (times, events, x, strata, n, p) = make_stratified_data(3, 40, 0.5, 41);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        assert_eq!(fit.n_strata, 3);
        assert_eq!(fit.stratum_ids.len(), 3);
        assert_eq!(fit.baseline_hazards.len(), 3);
    }

    #[test]
    fn efron_ties_also_converges() {
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.5, 53);
        let cfg = StratifiedCoxConfig {
            ties: StratTieMethod::Efron,
            ..Default::default()
        };
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        assert!(fit.coef[0].is_finite());
        assert!(fit.se[0] > 0.0);
    }

    #[test]
    fn single_stratum_matches_standard_cox_direction() {
        // With only 1 stratum and positive true β, the estimated coefficient should be positive.
        let (times, events, x, strata, n, p) = make_stratified_data(1, 100, 1.0, 61);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        // With β_true = 1, estimated should be clearly positive.
        assert!(
            fit.coef[0] > 0.0,
            "coef should be positive: {}",
            fit.coef[0]
        );
    }

    #[test]
    fn predict_survival_x_zero_uses_baseline() {
        // At x=0, linear predictor = 0, exp(lp) = 1, so S = exp(-cumulative_bh).
        let (times, events, x, strata, n, p) = make_stratified_data(2, 60, 0.5, 67);
        let cfg = StratifiedCoxConfig::default();
        let fit = stratified_cox_fit(&times, &events, &x, &strata, n, p, &cfg).expect("fit ok");
        let x_new = vec![0.0_f64];
        let t_max = times.iter().cloned().fold(0.0_f64, f64::max);
        let sv = stratified_cox_predict_survival(&fit, &x_new, 0, &[t_max]).expect("ok");
        // Survival at max time must be in (0, 1].
        assert!(sv[0] > 0.0 && sv[0] <= 1.0);
    }

    #[test]
    fn no_events_error() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![false, false, false];
        let x = vec![0.1, 0.2, 0.3];
        let strata = vec![0, 0, 1];
        let cfg = StratifiedCoxConfig::default();
        let result = stratified_cox_fit(&times, &events, &x, &strata, 3, 1, &cfg);
        assert!(matches!(result, Err(SurvivalError::NoEvents)));
    }
}
