//! Cox regression with time-varying covariates (counting-process formulation).
//!
//! Risk set at time t = {intervals with start < t <= stop}.

use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;
use crate::linalg::solve::cholesky_solve;
use crate::time_varying::counting_process::CountingProcessDataset;

/// Fitted time-varying Cox model.
#[derive(Debug, Clone)]
pub struct TvCoxFit {
    pub coefficients: Vec<f64>,
    pub log_likelihood: f64,
    pub information: Vec<f64>,
    pub variance: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

fn tv_partial_loglik(
    data: &CountingProcessDataset,
    beta: &[f64],
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    let p = beta.len();
    // Collect unique event times
    let mut event_times: Vec<f64> = data
        .intervals
        .iter()
        .filter(|iv| iv.event)
        .map(|iv| iv.stop)
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < 1.0e-12);
    let mut loglik = 0.0_f64;
    let mut score = vec![0.0_f64; p];
    let mut info = vec![0.0_f64; p * p];
    for &t in &event_times {
        // risk set: start < t <= stop
        let mut s0 = 0.0_f64;
        let mut s1 = vec![0.0_f64; p];
        let mut s2 = vec![0.0_f64; p * p];
        let mut events_t: Vec<usize> = Vec::new();
        for (i, iv) in data.intervals.iter().enumerate() {
            if iv.start < t && t <= iv.stop {
                let dot: f64 = iv
                    .covariates
                    .iter()
                    .zip(beta.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let w = dot.exp();
                s0 += w;
                for a in 0..p {
                    s1[a] += w * iv.covariates[a];
                    for b in 0..p {
                        s2[a * p + b] += w * iv.covariates[a] * iv.covariates[b];
                    }
                }
                if iv.event && (iv.stop - t).abs() < 1.0e-12 {
                    events_t.push(i);
                }
            }
        }
        let d_count = events_t.len() as f64;
        if d_count == 0.0 {
            continue;
        }
        if s0 <= 0.0 {
            return Err(SurvivalError::NumericalInstability(
                "non-positive S0 in TV cox".to_string(),
            ));
        }
        let mut sum_eta = 0.0_f64;
        let mut x_events = vec![0.0_f64; p];
        for &i in &events_t {
            let iv = &data.intervals[i];
            sum_eta += iv
                .covariates
                .iter()
                .zip(beta.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            for (a, xe) in x_events.iter_mut().enumerate().take(p) {
                *xe += iv.covariates[a];
            }
        }
        loglik += sum_eta - d_count * s0.ln();
        let x_bar: Vec<f64> = s1.iter().map(|si| si / s0).collect();
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

/// Fit time-varying Cox via Newton-Raphson.
pub fn fit_time_varying_cox(
    data: &CountingProcessDataset,
    tol: f64,
    max_iter: usize,
) -> SurvivalResult<TvCoxFit> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let p = data.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter("no covariates".to_string()));
    }
    let mut beta = vec![0.0_f64; p];
    let (mut ll, mut score, mut info) = tv_partial_loglik(data, &beta)?;
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
            if let Ok((nl, ns, ni)) = tv_partial_loglik(data, &trial) {
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
    Ok(TvCoxFit {
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
    use crate::time_varying::counting_process::CountingInterval;

    #[test]
    fn tv_cox_fits_simple() {
        // Two subjects with one interval each, mimicking standard Cox
        let intervals = vec![
            CountingInterval::new(0, 0.0, 1.0, true, vec![1.0]).expect("ok"),
            CountingInterval::new(1, 0.0, 2.0, true, vec![0.0]).expect("ok"),
            CountingInterval::new(2, 0.0, 3.0, true, vec![-1.0]).expect("ok"),
        ];
        let d = CountingProcessDataset::new(intervals).expect("ok");
        let fit = fit_time_varying_cox(&d, 1.0e-6, 50).expect("ok");
        assert!(fit.converged);
        // x=1 dies earliest -> positive β
        assert!(fit.coefficients[0] > 0.0);
    }

    #[test]
    fn tv_cox_rejects_no_events() {
        let intervals = vec![CountingInterval::new(0, 0.0, 1.0, false, vec![1.0]).expect("ok")];
        let d = CountingProcessDataset::new(intervals).expect("ok");
        let r = fit_time_varying_cox(&d, 1.0e-6, 10);
        assert!(matches!(r, Err(SurvivalError::NoEvents)));
    }
}
