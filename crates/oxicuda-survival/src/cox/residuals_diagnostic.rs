//! Cox proportional-hazards residual diagnostics.
//!
//! This module implements the three classical residual processes used to assess
//! the fit of a Cox model, all built on top of the Breslow baseline cumulative
//! hazard already produced by [`crate::cox::cox_ph::fit_cox_ph`]:
//!
//! * **Martingale residuals** `M_i = δ_i − Ĥ_i`, where
//!   `Ĥ_i = Λ̂₀(t_i) · exp(β̂ᵀx_i)` is the model-based cumulative hazard at the
//!   subject's own event/censoring time. They lie in `(−∞, 1]` and, at the
//!   maximum partial-likelihood estimate with a Breslow baseline, sum to exactly
//!   zero (`Σ M_i = 0`).
//! * **Deviance residuals** `d_i = sign(M_i)·√(−2[M_i + δ_i·ln(δ_i − M_i)])`,
//!   a symmetrising transform of the martingale residuals whose squared sum is
//!   the model deviance. The `δ_i·ln(·)` term is taken as `0` when `δ_i = 0`.
//! * **Cumulative martingale process** (Lin, Wei & Ying, 1993): martingale
//!   residuals ordered by a chosen covariate, accumulated into a partial-sum
//!   process whose supremum is a graphical functional-form statistic.
//!
//! # Ties handling
//! The Breslow baseline cumulative hazard is the only baseline produced by the
//! crate's Cox fit, so the residuals here are *Breslow-consistent* regardless of
//! the tie method used during optimisation. With a Breslow baseline the identity
//! `Σ M_i = 0` is exact; with an Efron-fitted `β̂` the baseline increments still
//! use the Breslow form, so the residuals remain Breslow martingale residuals.

use crate::cox::cox_ph::CoxFit;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Output of the Lin-Wei-Ying cumulative martingale process.
///
/// The process orders subjects by the chosen covariate and accumulates their
/// martingale residuals. `sorted_covariate[k]` is the covariate value of the
/// `k`-th subject in ascending order, `cumulative[k]` the running sum of
/// martingale residuals up to and including that subject, and `sup_statistic`
/// the supremum of `|cumulative[k]|` over the whole process.
#[derive(Debug, Clone)]
pub struct CumulativeResidualProcess {
    /// Index (into the original dataset) of the covariate that drives the order.
    pub covariate_index: usize,
    /// Covariate values in ascending order.
    pub sorted_covariate: Vec<f64>,
    /// Original subject index for each ordered position.
    pub subject_order: Vec<usize>,
    /// Cumulative sum of martingale residuals along the ordering.
    pub cumulative: Vec<f64>,
    /// Supremum `max_k |cumulative[k]|` — the functional-form test statistic.
    pub sup_statistic: f64,
}

/// Evaluate the Breslow baseline cumulative hazard `Λ̂₀(t)` at an arbitrary time.
///
/// The baseline is a right-continuous step function that jumps only at event
/// times; `Λ̂₀(t)` is the value at the largest baseline knot `≤ t`, and `0` when
/// `t` precedes the first knot.
fn baseline_at(fit: &CoxFit, t: f64) -> f64 {
    let times = &fit.baseline_hazard.times;
    let cum = &fit.baseline_hazard.cumulative_hazard;
    if times.is_empty() {
        return 0.0;
    }
    // Largest index with times[idx] <= t.
    match times.binary_search_by(|tk| tk.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(idx) => cum[idx],
        Err(0) => 0.0,
        Err(ins) => cum[ins - 1],
    }
}

/// Validate that a fitted model and dataset are dimensionally compatible and
/// return the covariate matrix.
fn covariates_for<'a>(fit: &CoxFit, data: &'a Dataset) -> SurvivalResult<&'a Vec<Vec<f64>>> {
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
    Ok(covariates)
}

/// Linear predictor `β̂ᵀx_i` for subject `i`.
fn linear_predictor(beta: &[f64], xi: &[f64]) -> f64 {
    xi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum()
}

/// Martingale residuals `M_i = δ_i − Λ̂₀(t_i)·exp(β̂ᵀx_i)`.
///
/// Returns one residual per subject in the original dataset order. At the
/// maximum partial-likelihood estimate with the Breslow baseline the residuals
/// sum to zero.
///
/// # Errors
/// * [`SurvivalError::EmptyDataset`] if `data` has no observations.
/// * [`SurvivalError::DimensionMismatch`] if the covariate width does not match
///   the fitted coefficient vector.
pub fn martingale_residuals(fit: &CoxFit, data: &Dataset) -> SurvivalResult<Vec<f64>> {
    let covariates = covariates_for(fit, data)?;
    let beta = &fit.coefficients;
    let mut out = Vec::with_capacity(data.len());
    for (i, obs) in data.observations.iter().enumerate() {
        let delta = if obs.event { 1.0 } else { 0.0 };
        let eta = linear_predictor(beta, &covariates[i]);
        let cum_hazard = baseline_at(fit, obs.time) * eta.exp();
        out.push(delta - cum_hazard);
    }
    Ok(out)
}

/// Deviance residuals derived from the martingale residuals.
///
/// `d_i = sign(M_i)·√(−2[M_i + δ_i·ln(δ_i − M_i)])`, with the convention that
/// the `δ_i·ln(·)` term is `0` when `δ_i = 0`. The squared deviance residuals
/// sum to the model deviance and are more symmetric about zero than the raw
/// martingale residuals.
///
/// # Errors
/// Propagates the errors of [`martingale_residuals`].
pub fn deviance_residuals(fit: &CoxFit, data: &Dataset) -> SurvivalResult<Vec<f64>> {
    let martingale = martingale_residuals(fit, data)?;
    let mut out = Vec::with_capacity(martingale.len());
    for (m, obs) in martingale.iter().zip(data.observations.iter()) {
        let delta = if obs.event { 1.0 } else { 0.0 };
        // δ·ln(δ − M); when δ = 0 the term is defined to be 0.
        let log_term = if delta > 0.0 {
            let arg = delta - m;
            // δ − M = 1 − M > 0 always for an event (M ≤ 1, equality only when Ĥ = 0,
            // which cannot coincide with an observed event), but guard for safety.
            if arg > 0.0 { delta * arg.ln() } else { 0.0 }
        } else {
            0.0
        };
        let inner = -2.0 * (m + log_term);
        let magnitude = inner.max(0.0).sqrt();
        let sign = if *m > 0.0 {
            1.0
        } else if *m < 0.0 {
            -1.0
        } else {
            0.0
        };
        out.push(sign * magnitude);
    }
    Ok(out)
}

/// Lin-Wei-Ying cumulative martingale process for one covariate.
///
/// Orders subjects by `covariate_index` (ascending, stable on ties), accumulates
/// their martingale residuals into a partial-sum process, and reports the
/// supremum of the absolute partial sums. A large supremum relative to the
/// covariate range flags mis-specification of that covariate's functional form.
///
/// # Errors
/// * [`SurvivalError::IndexOutOfBounds`] if `covariate_index ≥ p`.
/// * Propagates the errors of [`martingale_residuals`].
pub fn cumulative_martingale_process(
    fit: &CoxFit,
    data: &Dataset,
    covariate_index: usize,
) -> SurvivalResult<CumulativeResidualProcess> {
    let p = fit.coefficients.len();
    if covariate_index >= p {
        return Err(SurvivalError::IndexOutOfBounds {
            index: covariate_index,
            len: p,
        });
    }
    let covariates = covariates_for(fit, data)?;
    let martingale = martingale_residuals(fit, data)?;

    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&a, &b| {
        covariates[a][covariate_index]
            .partial_cmp(&covariates[b][covariate_index])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_covariate = Vec::with_capacity(order.len());
    let mut cumulative = Vec::with_capacity(order.len());
    let mut running = 0.0_f64;
    let mut sup = 0.0_f64;
    for &idx in &order {
        sorted_covariate.push(covariates[idx][covariate_index]);
        running += martingale[idx];
        cumulative.push(running);
        let abs = running.abs();
        if abs > sup {
            sup = abs;
        }
    }

    Ok(CumulativeResidualProcess {
        covariate_index,
        sorted_covariate,
        subject_order: order,
        cumulative,
        sup_statistic: sup,
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
            // All events: this fixture exercises the Σ M = 0 identity directly.
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
            let c = rng.next_exponential(0.4).max(1.0e-6);
            let (time, event) = if t <= c { (t, true) } else { (c, false) };
            obs.push(Observation::new(time, event).expect("ok"));
            cov.push(vec![x]);
        }
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    #[test]
    fn martingale_sum_to_zero_all_events() {
        let data = synthetic(120, 0.8, 11);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let m = martingale_residuals(&fit, &data).expect("ok");
        let sum: f64 = m.iter().sum();
        assert!(sum.abs() < 1.0e-6, "Σ M = {sum}");
    }

    #[test]
    fn martingale_sum_to_zero_with_censoring() {
        let data = synthetic_censored(200, 0.6, 23);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let m = martingale_residuals(&fit, &data).expect("ok");
        let sum: f64 = m.iter().sum();
        assert!(sum.abs() < 1.0e-6, "Σ M = {sum}");
    }

    #[test]
    fn martingale_in_range() {
        let data = synthetic_censored(150, 0.5, 99);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let m = martingale_residuals(&fit, &data).expect("ok");
        for v in &m {
            assert!(v.is_finite());
            assert!(*v <= 1.0 + 1.0e-9, "M out of range: {v}");
        }
    }

    #[test]
    fn martingale_hand_computation_single_covariate() {
        // Tiny dataset, 5 subjects, all events, x = [0, 0, 0, 0, 0] so that the
        // model collapses to Nelson-Aalen / Breslow with β arbitrary (η = 0).
        // Times 1,2,3,4,5. With all events and identical covariates β̂ = 0.
        //   Λ̂₀(t) = Σ_{t_k ≤ t} 1/n_k.
        //   Λ̂₀(1)=1/5, Λ̂₀(2)=1/5+1/4, Λ̂₀(3)=+1/3, Λ̂₀(4)=+1/2, Λ̂₀(5)=+1.
        //   M_i = δ_i − Λ̂₀(t_i) with exp(0)=1.
        let times = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut obs = Vec::new();
        let mut cov = Vec::new();
        for &t in &times {
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(vec![0.0]);
        }
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let m = martingale_residuals(&fit, &data).expect("ok");
        let h1 = 1.0 / 5.0;
        let h2 = h1 + 1.0 / 4.0;
        let h3 = h2 + 1.0 / 3.0;
        let h4 = h3 + 1.0 / 2.0;
        let h5 = h4 + 1.0;
        let expected = [1.0 - h1, 1.0 - h2, 1.0 - h3, 1.0 - h4, 1.0 - h5];
        for (got, want) in m.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1.0e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn deviance_symmetric_and_finite() {
        let data = synthetic_censored(200, 0.7, 5);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let d = deviance_residuals(&fit, &data).expect("ok");
        for v in &d {
            assert!(v.is_finite());
        }
        // Deviance residuals are roughly centred: mean should be modest.
        let mean: f64 = d.iter().sum::<f64>() / d.len() as f64;
        assert!(mean.abs() < 0.6, "deviance mean = {mean}");
        // Both signs should be present.
        assert!(d.iter().any(|v| *v > 0.0));
        assert!(d.iter().any(|v| *v < 0.0));
    }

    #[test]
    fn deviance_sum_squares_positive() {
        let data = synthetic_censored(80, 0.5, 71);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let d = deviance_residuals(&fit, &data).expect("ok");
        let dev: f64 = d.iter().map(|v| v * v).sum();
        assert!(dev > 0.0 && dev.is_finite());
    }

    #[test]
    fn outlier_has_large_positive_martingale() {
        // Build a dataset where one subject has a very low risk score yet a very
        // early event — that subject should have the largest positive martingale.
        let mut obs = Vec::new();
        let mut cov = Vec::new();
        // 20 ordinary subjects with x≈0 spread over times.
        for k in 0..20 {
            obs.push(Observation::new(2.0 + k as f64, false).expect("ok"));
            cov.push(vec![0.0]);
        }
        // Planted outlier: large negative covariate (low risk) but earliest event.
        obs.push(Observation::new(0.5, true).expect("ok"));
        cov.push(vec![-3.0]);
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let m = martingale_residuals(&fit, &data).expect("ok");
        let outlier = *m.last().expect("non-empty");
        let max_other = m[..m.len() - 1]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            outlier >= max_other,
            "outlier martingale {outlier} not the largest (max other {max_other})"
        );
        assert!(outlier > 0.0);
    }

    #[test]
    fn cumulative_process_sup_nonneg() {
        let data = synthetic_censored(150, 0.6, 314);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let proc = cumulative_martingale_process(&fit, &data, 0).expect("ok");
        assert_eq!(proc.cumulative.len(), data.len());
        assert!(proc.sup_statistic >= 0.0);
        // Final cumulative value equals the total martingale sum ≈ 0.
        assert!(proc.cumulative.last().expect("non-empty").abs() < 1.0e-6);
        // sorted_covariate is non-decreasing.
        for w in proc.sorted_covariate.windows(2) {
            assert!(w[1] >= w[0] - 1.0e-12);
        }
    }

    #[test]
    fn cumulative_process_rejects_bad_index() {
        let data = synthetic_censored(20, 0.5, 1);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let r = cumulative_martingale_process(&fit, &data, 5);
        assert!(matches!(r, Err(SurvivalError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn residuals_reject_dim_mismatch() {
        let data = synthetic(10, 0.5, 2);
        let mut fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        // Corrupt the coefficient length to force a dimension mismatch.
        fit.coefficients = vec![0.0, 0.0];
        let r = martingale_residuals(&fit, &data);
        assert!(matches!(r, Err(SurvivalError::DimensionMismatch { .. })));
    }

    #[test]
    fn efron_fit_still_breslow_consistent_sum() {
        // Even when β̂ is obtained with Efron ties, the Breslow baseline keeps
        // Σ M ≈ 0 because the baseline increments are Breslow.
        let data = synthetic(100, 0.5, 808);
        let cfg = CoxPhConfig {
            tie: TieMethod::Efron,
            tol: 1.0e-8,
            max_iter: 80,
        };
        let fit = fit_cox_ph(&data, cfg).expect("ok");
        let m = martingale_residuals(&fit, &data).expect("ok");
        let sum: f64 = m.iter().sum();
        assert!(sum.abs() < 1.0e-6, "Σ M (Efron fit) = {sum}");
    }
}
