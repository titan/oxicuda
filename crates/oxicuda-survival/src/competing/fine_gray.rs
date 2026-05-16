//! Fine-Gray sub-distribution hazard model.
//!
//! Reweights subjects who experienced competing events by `G(t)/G(t_i)` where
//! `G` is the Kaplan-Meier of the censoring distribution.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::linalg::solve::cholesky_solve;

#[derive(Debug, Clone)]
pub struct FineGrayFit {
    pub coefficients: Vec<f64>,
    pub log_likelihood: f64,
    pub information: Vec<f64>,
    pub variance: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

fn fg_partial(
    data: &Dataset,
    causes: &[u32],
    target: u32,
    beta: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    let p = beta.len();
    let cov = data
        .covariates
        .as_ref()
        .ok_or_else(|| SurvivalError::InvalidParameter("no covariates".to_string()))?;
    let n = data.len();
    // Censoring KM: treat censorings as 'events'
    let mut idx = data.order_by_time();
    idx.sort_by(|&a, &b| {
        data.observations[a]
            .time
            .partial_cmp(&data.observations[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Build G(t) from censoring KM
    let mut g_times = Vec::new();
    let mut g_vals = Vec::new();
    let mut g_cur = 1.0_f64;
    let mut at_risk = n as f64;
    let mut k = 0usize;
    while k < idx.len() {
        let t = data.observations[idx[k]].time;
        let mut m = k;
        let mut dc = 0.0_f64; // censorings at t
        while m < idx.len() && data.observations[idx[m]].time == t {
            if !data.observations[idx[m]].event {
                dc += 1.0;
            }
            m += 1;
        }
        if dc > 0.0 && at_risk > 0.0 {
            g_cur *= 1.0 - dc / at_risk;
        }
        g_times.push(t);
        g_vals.push(g_cur);
        at_risk -= (m - k) as f64;
        k = m;
    }
    let g_at = |t: f64| -> f64 {
        // largest g_times[j] <= t
        let mut v = 1.0;
        for (i, &gt) in g_times.iter().enumerate() {
            if gt <= t {
                v = g_vals[i];
            } else {
                break;
            }
        }
        v.max(1.0e-300)
    };
    // For each subject i, weight w_i(t) at event time t:
    //   if subject i has cause == target and time = t_event => weight=1
    //   if subject i was censored at time c_i < t => not in risk set (after their time)
    //   if subject i had competing event at time tc_i < t => weight = G(t) / G(tc_i)
    //   if subject i still at risk at t => weight=1
    // Equivalent: persistent membership in risk set with diminishing weight for competing events.
    let mut event_times: Vec<f64> = data
        .observations
        .iter()
        .zip(causes.iter())
        .filter_map(|(o, c)| {
            if o.event && *c == target {
                Some(o.time)
            } else {
                None
            }
        })
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < 1.0e-12);
    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];
    for &t in &event_times {
        let g_t = g_at(t);
        let mut s0 = 0.0_f64;
        let mut s1 = vec![0.0_f64; p];
        let mut s2 = vec![0.0_f64; p * p];
        let mut x_events = vec![0.0_f64; p];
        let mut eta_events = 0.0_f64;
        let mut d_count = 0.0_f64;
        for i in 0..n {
            let ti = data.observations[i].time;
            let ci = causes[i];
            let evi = data.observations[i].event;
            // weight
            let w_i = if !evi && ti < t {
                0.0 // censored before t => out
            } else if evi && ci != target && ti < t {
                // competing event earlier => persistent with reweight
                let g_ti = g_at(ti);
                g_t / g_ti.max(1.0e-300)
            } else if ti >= t || (evi && ci == target && (ti - t).abs() < 1.0e-12) {
                1.0
            } else {
                0.0
            };
            if w_i <= 0.0 {
                continue;
            }
            let dot: f64 = cov[i].iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
            let w = w_i * dot.exp();
            s0 += w;
            for a in 0..p {
                s1[a] += w * cov[i][a];
                for b in 0..p {
                    s2[a * p + b] += w * cov[i][a] * cov[i][b];
                }
            }
            if evi && ci == target && (ti - t).abs() < 1.0e-12 {
                d_count += 1.0;
                eta_events += dot;
                for a in 0..p {
                    x_events[a] += cov[i][a];
                }
            }
        }
        if d_count == 0.0 || s0 <= 0.0 {
            continue;
        }
        loglik += eta_events - d_count * s0.ln();
        let x_bar: Vec<f64> = s1.iter().map(|x| x / s0).collect();
        for a in 0..p {
            score[a] += x_events[a] - d_count * x_bar[a];
        }
        for a in 0..p {
            for b in 0..p {
                info[a * p + b] += d_count * (s2[a * p + b] / s0 - x_bar[a] * x_bar[b]);
            }
        }
    }
    Ok((loglik, score, info))
}

/// Fit a Fine-Gray sub-distribution hazard model.
pub fn fit_fine_gray(
    data: &Dataset,
    causes: &[u32],
    target_cause: u32,
    tol: f64,
    max_iter: usize,
) -> SurvivalResult<FineGrayFit> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if causes.len() != data.len() {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![data.len()],
            got: vec![causes.len()],
        });
    }
    if target_cause == 0 {
        return Err(SurvivalError::InvalidParameter(
            "target_cause must be > 0".to_string(),
        ));
    }
    let p = data.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter("no covariates".to_string()));
    }
    let mut beta = vec![0.0_f64; p];
    let (mut ll, mut score, mut info) = fg_partial(data, causes, target_cause, &beta)?;
    let mut converged = false;
    let mut iter = 0usize;
    for it in 0..max_iter {
        iter = it + 1;
        let max_s = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_s < tol {
            converged = true;
            break;
        }
        let delta = cholesky_solve(&info, &score, p).unwrap_or_else(|_| vec![0.0; p]);
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((nl, ns, ni)) = fg_partial(data, causes, target_cause, &trial) {
                if nl.is_finite() && nl > ll - 1.0e-10 {
                    beta = trial;
                    ll = nl;
                    score = ns;
                    info = ni;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-18 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    let variance = gauss_jordan_inverse(&info, p).unwrap_or_else(|_| vec![0.0; p * p]);
    Ok(FineGrayFit {
        coefficients: beta,
        log_likelihood: ll,
        information: info,
        variance,
        iterations: iter,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    #[test]
    fn fine_gray_reduces_to_cox_no_competing_events() {
        // Larger synthetic dataset
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(202);
        let n = 100;
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let t = rng.next_exponential((0.5 * x).exp()).max(1.0e-6);
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
        }
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let causes = vec![1u32; n];
        let fg = fit_fine_gray(&data, &causes, 1, 1.0e-6, 50).expect("ok");
        let cox = crate::cox::cox_ph::fit_cox_ph(&data, crate::cox::cox_ph::CoxPhConfig::default())
            .expect("ok");
        assert!((fg.coefficients[0] - cox.coefficients[0]).abs() < 0.05);
    }

    #[test]
    fn fine_gray_handles_competing_events() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(303);
        let n = 80;
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        let mut causes = Vec::with_capacity(n);
        for i in 0..n {
            let x = rng.next_normal();
            let t = rng.next_exponential((0.3 * x).exp()).max(1.0e-6);
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
            causes.push(if i % 2 == 0 { 1u32 } else { 2u32 });
        }
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let fg = fit_fine_gray(&data, &causes, 1, 1.0e-6, 50).expect("ok");
        assert!(fg.log_likelihood.is_finite());
    }
}
