//! Log-logistic AFT model: `S(t) = 1 / (1 + (t/η)^k)`.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

#[derive(Debug, Clone)]
pub struct LogLogisticFit {
    pub shape: f64,
    pub scale: f64,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl LogLogisticFit {
    #[must_use]
    pub fn survival(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        1.0 / (1.0 + (t / self.scale).powf(self.shape))
    }
}

fn loglog_loglik(times: &[f64], events: &[f64], k: f64, eta: f64) -> f64 {
    if k <= 0.0 || eta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0_f64;
    for i in 0..times.len() {
        let r = times[i] / eta;
        let rk = r.powf(k);
        let denom = 1.0 + rk;
        if events[i] > 0.5 {
            // log f = log k - log η + (k-1) log r - 2 log(1+r^k)
            ll += k.ln() - eta.ln() + (k - 1.0) * r.ln() - 2.0 * denom.ln();
        } else {
            ll += -denom.ln();
        }
    }
    ll
}

/// Fit by coordinate Newton-with-halving on (log k, log η).
pub fn fit_log_logistic(data: &Dataset) -> SurvivalResult<LogLogisticFit> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let n = data.len();
    let times: Vec<f64> = data.times();
    let events: Vec<f64> = data.events_f64();
    for t in &times {
        if *t <= 0.0 {
            return Err(SurvivalError::InvalidParameter(
                "log-logistic requires positive times".to_string(),
            ));
        }
    }
    // Sensible init: η = median, k = 1
    let mut sorted_times = times.clone();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_times[n / 2];
    let mut log_k = 0.0_f64;
    let mut log_eta = median.ln();
    let mut converged = false;
    let mut iter = 0usize;
    let mut ll_prev = f64::NEG_INFINITY;
    let eps = 1.0e-5;
    for it in 0..200 {
        iter = it + 1;
        let k = log_k.exp();
        let eta = log_eta.exp();
        let ll = loglog_loglik(&times, &events, k, eta);
        if (ll - ll_prev).abs() < 1.0e-10 && it > 0 {
            converged = true;
            break;
        }
        ll_prev = ll;
        // Numerical gradient
        let g0 = (loglog_loglik(&times, &events, (log_k + eps).exp(), eta)
            - loglog_loglik(&times, &events, (log_k - eps).exp(), eta))
            / (2.0 * eps);
        let g1 = (loglog_loglik(&times, &events, k, (log_eta + eps).exp())
            - loglog_loglik(&times, &events, k, (log_eta - eps).exp()))
            / (2.0 * eps);
        // Approximate Hessian diagonal
        let h00 = (loglog_loglik(&times, &events, (log_k + eps).exp(), eta) - 2.0 * ll
            + loglog_loglik(&times, &events, (log_k - eps).exp(), eta))
            / (eps * eps);
        let h11 = (loglog_loglik(&times, &events, k, (log_eta + eps).exp()) - 2.0 * ll
            + loglog_loglik(&times, &events, k, (log_eta - eps).exp()))
            / (eps * eps);
        let dx0 = if h00.abs() < 1.0e-12 {
            1.0e-3 * g0
        } else {
            -g0 / h00
        };
        let dx1 = if h11.abs() < 1.0e-12 {
            1.0e-3 * g1
        } else {
            -g1 / h11
        };
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..30 {
            let trial_k = log_k + step * dx0;
            let trial_e = log_eta + step * dx1;
            let ll_t = loglog_loglik(&times, &events, trial_k.exp(), trial_e.exp());
            if ll_t.is_finite() && ll_t > ll - 1.0e-12 {
                log_k = trial_k;
                log_eta = trial_e;
                accepted = true;
                break;
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
    let k = log_k.exp();
    let eta = log_eta.exp();
    let ll = loglog_loglik(&times, &events, k, eta);
    Ok(LogLogisticFit {
        shape: k,
        scale: eta,
        log_likelihood: ll,
        iterations: iter,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    #[test]
    fn log_logistic_survival_at_scale_is_half() {
        let fit = LogLogisticFit {
            shape: 2.0,
            scale: 1.0,
            log_likelihood: 0.0,
            iterations: 0,
            converged: true,
        };
        assert!((fit.survival(1.0) - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn log_logistic_fit_returns_finite() {
        let data = Dataset::new(
            (1..20)
                .map(|i| Observation::new(i as f64, true).expect("ok"))
                .collect(),
            None,
            None,
        )
        .expect("ok");
        let f = fit_log_logistic(&data).expect("ok");
        assert!(f.shape > 0.0);
        assert!(f.scale > 0.0);
        assert!(f.log_likelihood.is_finite());
    }
}
