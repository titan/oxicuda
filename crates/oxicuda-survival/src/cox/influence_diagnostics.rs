//! Cox proportional-hazards influence diagnostics (DFBeta).
//!
//! For each subject this module approximates the effect of deleting that subject
//! on the fitted coefficient vector. The approximation is the classical
//! one-step (infinitesimal-jackknife) DFBeta:
//!
//! ```text
//!   DFBeta_i ≈ L_i · I(β̂)⁻¹
//! ```
//!
//! where `L_i` is the *efficient score residual* (a `p`-vector) for subject `i`
//! and `I(β̂)⁻¹` is the inverse Fisher information (the model covariance already
//! stored in [`CoxFit::variance`]).
//!
//! The score residual is the integral of `(x_i − x̄(t))` against the subject's
//! martingale increments,
//!
//! ```text
//!   L_i = δ_i·(x_i − x̄(t_i))
//!         − exp(β̂ᵀx_i)·Σ_{t_k ≤ t_i} (x_i − x̄(t_k))·dΛ̂₀(t_k),
//! ```
//!
//! with `x̄(t) = S₁(t)/S₀(t)` the risk-weighted covariate mean over the risk set
//! at `t`, `S₀(t) = Σ_{j∈R(t)} exp(β̂ᵀx_j)`, `S₁(t) = Σ_{j∈R(t)} exp(β̂ᵀx_j) x_j`,
//! and `dΛ̂₀(t_k) = d_k / S₀(t_k)` the Breslow baseline-hazard increment.
//!
//! Summed over all subjects, `Σ_i L_i = U(β̂)` (the Breslow partial-likelihood
//! score), which is `≈ 0` at the maximum-likelihood estimate. Consequently
//! `Σ_i DFBeta_i ≈ 0`, and each `DFBeta_i` matches, to first order, the actual
//! change `β̂_full − β̂_{−i}` obtained by refitting without subject `i`.
//!
//! # Ties handling
//! The score residual uses Breslow baseline increments, matching the only
//! baseline the crate's Cox fit produces. The leave-one-out agreement is exact
//! to first order for a Breslow-fitted model; for an Efron-fitted `β̂` it remains
//! a high-quality approximation.

use crate::cox::cox_ph::CoxFit;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Influence diagnostics for every subject in the dataset.
///
/// All matrices are `n × p` (subjects × coefficients), in row-major nesting:
/// `dfbeta[i][j]` is the approximate change in `β̂_j` from deleting subject `i`.
#[derive(Debug, Clone)]
pub struct InfluenceDiagnostics {
    /// Raw DFBeta: `dfbeta[i] = L_i · I(β̂)⁻¹`, an `n × p` matrix.
    pub dfbeta: Vec<Vec<f64>>,
    /// Standardised DFBeta: `dfbetas[i][j] = dfbeta[i][j] / SE(β̂_j)`.
    pub dfbetas: Vec<Vec<f64>>,
    /// Likelihood displacement `LD_i = DFBeta_iᵀ · I(β̂) · DFBeta_i`, length `n`.
    pub likelihood_displacement: Vec<f64>,
}

/// Per-event-time summary used to assemble the score residuals.
struct RiskSummary {
    /// Distinct event times in ascending order.
    times: Vec<f64>,
    /// Breslow baseline-hazard increment `dΛ̂₀(t_k) = d_k / S₀(t_k)` at each time.
    d_lambda: Vec<f64>,
    /// Risk-weighted covariate mean `x̄(t_k)` at each time, length `p` per entry.
    x_bar: Vec<Vec<f64>>,
}

/// Compute the per-event-time Breslow increments and weighted covariate means.
///
/// Iterates the subjects in ascending time, maintaining the risk-set sums
/// `S₀` and `S₁` and shrinking them as each time point is passed.
fn risk_summary(data: &Dataset, beta: &[f64], covariates: &[Vec<f64>]) -> RiskSummary {
    let p = beta.len();
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = idx.len();

    // Weights and running risk-set totals (start with the full sample).
    let mut weights = vec![0.0_f64; n];
    let mut s0 = 0.0_f64;
    let mut s1 = vec![0.0_f64; p];
    for (k, &i) in idx.iter().enumerate() {
        let xi = &covariates[i];
        let eta: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let w = eta.exp();
        weights[k] = w;
        s0 += w;
        for a in 0..p {
            s1[a] += w * xi[a];
        }
    }

    let mut times = Vec::new();
    let mut d_lambda = Vec::new();
    let mut x_bar = Vec::new();

    let mut k = 0usize;
    while k < n {
        let t = data.observations[idx[k]].time;
        let mut m = k;
        let mut d_count = 0.0_f64;
        while m < n && data.observations[idx[m]].time == t {
            if data.observations[idx[m]].event {
                d_count += 1.0;
            }
            m += 1;
        }
        if d_count > 0.0 && s0 > 0.0 {
            let bar: Vec<f64> = s1.iter().map(|s| s / s0).collect();
            times.push(t);
            d_lambda.push(d_count / s0);
            x_bar.push(bar);
        }
        // Remove the subjects at this time from the risk set.
        for jj in k..m {
            let xi = &covariates[idx[jj]];
            let w = weights[jj];
            s0 -= w;
            for a in 0..p {
                s1[a] -= w * xi[a];
            }
        }
        k = m;
    }

    RiskSummary {
        times,
        d_lambda,
        x_bar,
    }
}

/// Multiply a row vector `v` (length `p`) by a `p × p` row-major matrix `m`:
/// returns `vᵀ · m`, a length-`p` vector.
fn vec_times_matrix(v: &[f64], m: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p];
    for (a, &va) in v.iter().enumerate() {
        if va == 0.0 {
            continue;
        }
        for b in 0..p {
            out[b] += va * m[a * p + b];
        }
    }
    out
}

/// Quadratic form `vᵀ · m · v` with `m` a `p × p` row-major matrix.
fn quadratic_form(v: &[f64], m: &[f64], p: usize) -> f64 {
    let mv = {
        let mut out = vec![0.0_f64; p];
        for a in 0..p {
            let mut s = 0.0_f64;
            for b in 0..p {
                s += m[a * p + b] * v[b];
            }
            out[a] = s;
        }
        out
    };
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum()
}

/// Compute the efficient score residuals `L_i` for every subject.
///
/// Returns an `n × p` matrix; summed over subjects it equals the Breslow score
/// `U(β̂)`.
pub fn score_residuals(fit: &CoxFit, data: &Dataset) -> SurvivalResult<Vec<Vec<f64>>> {
    let p = fit.coefficients.len();
    let covariates = data
        .covariates
        .as_ref()
        .ok_or_else(|| SurvivalError::InvalidParameter("dataset has no covariates".to_string()))?;
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let got = covariates.first().map(|r| r.len()).unwrap_or(0);
    if got != p {
        return Err(SurvivalError::DimensionMismatch { a: got, b: p });
    }
    if covariates.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![covariates.len()],
        });
    }
    let beta = &fit.coefficients;
    let summary = risk_summary(data, beta, covariates);

    // Prefix accumulators over event times for the integral term:
    //   A(t_i) = Σ_{t_k ≤ t_i} dΛ̂₀(t_k)              (scalar)
    //   B(t_i) = Σ_{t_k ≤ t_i} x̄(t_k)·dΛ̂₀(t_k)       (p-vector)
    // Then the integral term for subject i is exp(η_i)·[ x_i·A(t_i) − B(t_i) ].
    let n_times = summary.times.len();
    let mut prefix_a = vec![0.0_f64; n_times];
    let mut prefix_b = vec![vec![0.0_f64; p]; n_times];
    let mut acc_a = 0.0_f64;
    let mut acc_b = vec![0.0_f64; p];
    for k in 0..n_times {
        let dl = summary.d_lambda[k];
        acc_a += dl;
        for (acc, bar) in acc_b.iter_mut().zip(summary.x_bar[k].iter()) {
            *acc += bar * dl;
        }
        prefix_a[k] = acc_a;
        prefix_b[k].clone_from(&acc_b);
    }

    let mut residuals = Vec::with_capacity(data.len());
    for (i, obs) in data.observations.iter().enumerate() {
        let xi = &covariates[i];
        let eta: f64 = xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        let w = eta.exp();
        let mut li = vec![0.0_f64; p];

        // Event term: δ_i·(x_i − x̄(t_i)).
        if obs.event {
            // Find x̄ at the subject's own event time (exact match expected).
            if let Ok(pos) = summary.times.binary_search_by(|tk| {
                tk.partial_cmp(&obs.time)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let bar = &summary.x_bar[pos];
                for a in 0..p {
                    li[a] += xi[a] - bar[a];
                }
            }
        }

        // Integral term: − exp(η_i)·[ x_i·A(t_i) − B(t_i) ].
        // A(t_i)/B(t_i) accumulate over event times ≤ t_i.
        let cut = upper_index(&summary.times, obs.time);
        if cut > 0 {
            let a_val = prefix_a[cut - 1];
            let b_val = &prefix_b[cut - 1];
            for a in 0..p {
                li[a] -= w * (xi[a] * a_val - b_val[a]);
            }
        }

        residuals.push(li);
    }
    Ok(residuals)
}

/// Largest count of `times` entries that are `≤ t` (i.e. the insertion point at
/// the upper bound). Equivalent to `times.partition_point(|x| *x <= t)`.
fn upper_index(times: &[f64], t: f64) -> usize {
    times.partition_point(|x| *x <= t)
}

/// Full influence diagnostics: DFBeta, standardised DFBetas and likelihood
/// displacement for every subject.
///
/// # Errors
/// * [`SurvivalError::EmptyDataset`] if `data` is empty.
/// * [`SurvivalError::DimensionMismatch`] / [`SurvivalError::ShapeMismatch`] for
///   incompatible covariate shapes.
/// * [`SurvivalError::InvalidParameter`] if the fitted model has no covariates.
pub fn influence_diagnostics(fit: &CoxFit, data: &Dataset) -> SurvivalResult<InfluenceDiagnostics> {
    let p = fit.coefficients.len();
    let residuals = score_residuals(fit, data)?;
    let cov = &fit.variance; // I(β̂)⁻¹, p×p row-major
    let info = &fit.information; // I(β̂), p×p row-major
    let se = fit.standard_errors();

    let mut dfbeta = Vec::with_capacity(residuals.len());
    let mut dfbetas = Vec::with_capacity(residuals.len());
    let mut ld = Vec::with_capacity(residuals.len());

    for li in &residuals {
        // DFBeta_i = L_i · I⁻¹.
        let db = vec_times_matrix(li, cov, p);
        // Standardised by the coefficient standard errors.
        let dbs: Vec<f64> = db
            .iter()
            .zip(se.iter())
            .map(|(d, s)| if *s > 0.0 { d / s } else { 0.0 })
            .collect();
        // Likelihood displacement: DFBeta_iᵀ · I · DFBeta_i.
        let disp = quadratic_form(&db, info, p).max(0.0);
        dfbeta.push(db);
        dfbetas.push(dbs);
        ld.push(disp);
    }

    Ok(InfluenceDiagnostics {
        dfbeta,
        dfbetas,
        likelihood_displacement: ld,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cox::cox_ph::{CoxPhConfig, TieMethod, fit_cox_ph};
    use crate::data::Observation;
    use crate::handle::LcgRng;

    fn synthetic(n: usize, beta_true: f64, seed: u64) -> Dataset {
        let mut rng = LcgRng::new(seed);
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let lambda = (beta_true * x).exp();
            let t = rng.next_exponential(lambda).max(1.0e-6);
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
        }
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    fn synthetic_censored(n: usize, beta_true: f64, seed: u64) -> Dataset {
        let mut rng = LcgRng::new(seed);
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let lambda = (beta_true * x).exp();
            let t = rng.next_exponential(lambda).max(1.0e-6);
            let c = rng.next_exponential(0.3).max(1.0e-6);
            let (time, event) = if t <= c { (t, true) } else { (c, false) };
            obs.push(Observation::new(time, event).expect("ok"));
            cov.push(vec![x]);
        }
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    /// Build a dataset omitting subject `skip`.
    fn drop_subject(data: &Dataset, skip: usize) -> Dataset {
        let cov = data.covariates.as_ref().expect("cov");
        let mut obs = Vec::with_capacity(data.len() - 1);
        let mut c = Vec::with_capacity(data.len() - 1);
        for (i, (o, x)) in data.observations.iter().zip(cov.iter()).enumerate() {
            if i == skip {
                continue;
            }
            obs.push(*o);
            c.push(x.clone());
        }
        Dataset::new(obs, Some(c), None).expect("ok")
    }

    #[test]
    fn score_residuals_sum_to_score() {
        // Σ_i L_i must equal the Breslow score U(β̂) ≈ 0 at the MLE.
        let data = synthetic(120, 0.7, 13);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let res = score_residuals(&fit, &data).expect("ok");
        let p = fit.coefficients.len();
        let mut sum = vec![0.0_f64; p];
        for li in &res {
            for a in 0..p {
                sum[a] += li[a];
            }
        }
        for s in &sum {
            assert!(s.abs() < 1.0e-6, "Σ L_i component = {s}");
        }
    }

    #[test]
    fn dfbeta_sum_to_zero_vector() {
        let data = synthetic_censored(200, 0.6, 27);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let diag = influence_diagnostics(&fit, &data).expect("ok");
        let p = fit.coefficients.len();
        let mut sum = vec![0.0_f64; p];
        for row in &diag.dfbeta {
            for a in 0..p {
                sum[a] += row[a];
            }
        }
        for s in &sum {
            assert!(s.abs() < 1.0e-6, "Σ DFBeta component = {s}");
        }
    }

    #[test]
    fn dfbeta_shapes_and_finite() {
        let data = synthetic_censored(60, 0.5, 91);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let diag = influence_diagnostics(&fit, &data).expect("ok");
        assert_eq!(diag.dfbeta.len(), data.len());
        assert_eq!(diag.dfbetas.len(), data.len());
        assert_eq!(diag.likelihood_displacement.len(), data.len());
        for row in &diag.dfbeta {
            assert_eq!(row.len(), fit.coefficients.len());
            for v in row {
                assert!(v.is_finite());
            }
        }
        for ld in &diag.likelihood_displacement {
            assert!(ld.is_finite() && *ld >= 0.0);
        }
    }

    /// Pearson correlation of two equal-length samples.
    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            num += (x - mean_a) * (y - mean_b);
            da += (x - mean_a).powi(2);
            db += (y - mean_b).powi(2);
        }
        num / (da.sqrt() * db.sqrt())
    }

    #[test]
    fn leave_one_out_agreement() {
        // The decisive ground-truth check: refit Cox without each subject and
        // compare β̂_full − β̂_{−i} against DFBeta_i = L_i·I⁻¹. Use a gentle
        // dataset (moderate signal, no single dominating point) so the one-step
        // approximation is in its valid first-order regime. Breslow ties make
        // Σ DFBeta = 0 exact.
        let data = synthetic_censored(60, 0.4, 2024);
        let cfg = CoxPhConfig {
            tie: TieMethod::Breslow,
            tol: 1.0e-10,
            max_iter: 100,
        };
        let fit = fit_cox_ph(&data, cfg).expect("ok");
        let diag = influence_diagnostics(&fit, &data).expect("ok");

        let beta_full = fit.coefficients[0];
        let mut approx = Vec::with_capacity(data.len());
        let mut actual = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            let reduced = drop_subject(&data, i);
            if reduced.n_events() == 0 {
                continue;
            }
            let fit_i = match fit_cox_ph(&reduced, cfg) {
                Ok(f) if f.converged => f,
                _ => continue,
            };
            // DFBeta convention: β̂_full − β̂_{−i} ≈ L_i · I⁻¹.
            actual.push(beta_full - fit_i.coefficients[0]);
            approx.push(diag.dfbeta[i][0]);
        }
        assert!(approx.len() >= 40, "too few comparable subjects");

        // (1) Direction/ranking: near-perfect correlation with the refit truth.
        let corr = pearson(&approx, &actual);
        assert!(corr > 0.98, "LOO correlation too low: {corr}");

        // (2) Magnitude: the approximation tracks the truth to first order. The
        // regression slope of actual on approx should be ≈ 1.
        let n = approx.len() as f64;
        let mean_a = approx.iter().sum::<f64>() / n;
        let mean_b = actual.iter().sum::<f64>() / n;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        for (x, y) in approx.iter().zip(actual.iter()) {
            sxy += (x - mean_a) * (y - mean_b);
            sxx += (x - mean_a).powi(2);
        }
        let slope = sxy / sxx;
        assert!(
            (slope - 1.0).abs() < 0.25,
            "LOO regression slope {slope} not ≈ 1"
        );

        // (3) Per-subject closeness on the bulk (within a generous first-order
        // band that scales with the size of the deletion effect).
        for (a, b) in approx.iter().zip(actual.iter()) {
            assert!(
                (a - b).abs() < 0.02 + 0.35 * b.abs(),
                "DFBeta {a} vs actual {b}"
            );
        }
    }

    #[test]
    fn high_leverage_subject_dominates() {
        // Plant a subject whose covariate is an extreme outlier relative to a
        // tight bulk, with an event at a *moderately late* time (so it does not
        // dominate its own risk-set mean — that would mask it). Its score
        // residual, and hence ‖DFBeta‖, should be the largest.
        let mut obs = Vec::new();
        let mut cov = Vec::new();
        let mut rng = LcgRng::new(55);
        let n_bulk = 60;
        for k in 0..n_bulk {
            // Tight covariate cluster near 0; spread event times across the range.
            let x = rng.next_normal() * 0.2;
            let t = 0.2 + 0.1 * k as f64;
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
        }
        // Planted outlier: large covariate, late event when the risk set is tiny
        // is bad (masking); instead place it mid-stream with an extreme covariate
        // that contradicts the (near-zero β) bulk → high influence.
        obs.push(Observation::new(0.2 + 0.1 * 5.0, true).expect("ok"));
        cov.push(vec![5.0]);
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let diag = influence_diagnostics(&fit, &data).expect("ok");
        let norms: Vec<f64> = diag
            .dfbeta
            .iter()
            .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt())
            .collect();
        let planted = norms[n_bulk]; // last entry
        let max_other = norms[..n_bulk].iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            planted > max_other,
            "planted ‖DFBeta‖ {planted} not the largest (max other {max_other})"
        );

        // Cross-check against the actual leave-one-out effect: the planted point
        // should also produce the largest |β̂_full − β̂_{−i}|.
        let cfg = CoxPhConfig::default();
        let beta_full = fit.coefficients[0];
        let loo: Vec<f64> = (0..data.len())
            .map(|i| {
                let reduced = drop_subject(&data, i);
                match fit_cox_ph(&reduced, cfg) {
                    Ok(f) => (beta_full - f.coefficients[0]).abs(),
                    Err(_) => 0.0,
                }
            })
            .collect();
        let planted_loo = loo[n_bulk];
        let max_other_loo = loo[..n_bulk].iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            planted_loo >= max_other_loo,
            "planted LOO {planted_loo} not the largest actual effect ({max_other_loo})"
        );
    }

    #[test]
    fn rejects_dim_mismatch() {
        let data = synthetic(10, 0.5, 3);
        let mut fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        fit.coefficients = vec![0.0, 0.0];
        let r = influence_diagnostics(&fit, &data);
        assert!(matches!(r, Err(SurvivalError::DimensionMismatch { .. })));
    }
}
