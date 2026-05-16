//! Log-normal AFT model: `log T ~ N(μ, σ²)`.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

#[derive(Debug, Clone)]
pub struct LogNormalFit {
    pub mu: f64,
    pub sigma: f64,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl LogNormalFit {
    #[must_use]
    pub fn survival(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let z = (t.ln() - self.mu) / self.sigma;
        1.0 - std_normal_cdf(z)
    }
}

fn std_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-ax * ax).exp();
    sign * y
}

fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (std::f64::consts::TAU).sqrt()
}

fn log_one_minus_phi(z: f64) -> f64 {
    // log(1 - Φ(z)) — guard against tail underflow
    let s = (1.0 - std_normal_cdf(z)).max(1.0e-300);
    s.ln()
}

/// Fit by Newton with halving line-search on (μ, log σ).
pub fn fit_log_normal(data: &Dataset) -> SurvivalResult<LogNormalFit> {
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
                "log-normal requires strictly positive times".to_string(),
            ));
        }
    }
    let mean_log_t: f64 = times.iter().map(|t| t.ln()).sum::<f64>() / n as f64;
    let var_log_t: f64 = times
        .iter()
        .map(|t| (t.ln() - mean_log_t).powi(2))
        .sum::<f64>()
        / n as f64;
    let mut mu = mean_log_t;
    let mut log_sigma = (var_log_t.max(1.0e-6)).sqrt().ln();
    let mut converged = false;
    let mut iter = 0usize;
    let mut ll_prev = f64::NEG_INFINITY;
    for it in 0..200 {
        iter = it + 1;
        let sigma = log_sigma.exp();
        let mut ll = 0.0_f64;
        let mut g = [0.0_f64; 2];
        let mut h = [[0.0_f64; 2]; 2];
        for i in 0..n {
            let z = (times[i].ln() - mu) / sigma;
            if events[i] > 0.5 {
                // log f = -log t - log sigma - 0.5 log(2π) - 0.5 z²
                ll += -times[i].ln() - log_sigma - 0.5 * std::f64::consts::TAU.ln() - 0.5 * z * z;
                g[0] += z / sigma;
                g[1] += -1.0 + z * z;
                h[0][0] += -1.0 / (sigma * sigma);
                h[0][1] += -2.0 * z / sigma;
                h[1][0] = h[0][1];
                h[1][1] += -2.0 * z * z;
            } else {
                let s = (1.0 - std_normal_cdf(z)).max(1.0e-300);
                ll += s.ln();
                // d/dμ log S = φ(z) / (σ S); d/d log σ log S = z φ(z) / S
                let phi = std_normal_pdf(z);
                let dmu = phi / (sigma * s);
                let dls = z * phi / s;
                g[0] += dmu;
                g[1] += dls;
                // crude Hessian via approximation (positive variance contribution):
                h[0][0] += -dmu * dmu;
                h[1][1] += -dls * dls;
                h[0][1] += -dmu * dls;
                h[1][0] = h[0][1];
            }
        }
        if (ll - ll_prev).abs() < 1.0e-10 && it > 0 {
            converged = true;
            break;
        }
        ll_prev = ll;
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        let (dx0, dx1) = if det.abs() < 1.0e-14 {
            (1.0e-3 * g[0], 1.0e-3 * g[1])
        } else {
            let inv00 = h[1][1] / det;
            let inv01 = -h[0][1] / det;
            let inv11 = h[0][0] / det;
            (
                -(inv00 * g[0] + inv01 * g[1]),
                -(inv01 * g[0] + inv11 * g[1]),
            )
        };
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..30 {
            let trial_mu = mu + step * dx0;
            let trial_ls = log_sigma + step * dx1;
            let ll_t = lognorm_loglik(&times, &events, trial_mu, trial_ls.exp());
            if ll_t.is_finite() && ll_t > ll - 1.0e-12 {
                mu = trial_mu;
                log_sigma = trial_ls;
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
    let sigma = log_sigma.exp();
    let ll = lognorm_loglik(&times, &events, mu, sigma);
    Ok(LogNormalFit {
        mu,
        sigma,
        log_likelihood: ll,
        iterations: iter,
        converged,
    })
}

fn lognorm_loglik(times: &[f64], events: &[f64], mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let log_sigma = sigma.ln();
    let mut ll = 0.0_f64;
    for i in 0..times.len() {
        let z = (times[i].ln() - mu) / sigma;
        if events[i] > 0.5 {
            ll += -times[i].ln() - log_sigma - 0.5 * std::f64::consts::TAU.ln() - 0.5 * z * z;
        } else {
            ll += log_one_minus_phi(z);
        }
    }
    ll
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;
    use crate::handle::LcgRng;

    #[test]
    fn lognormal_survival_in_unit() {
        let fit = LogNormalFit {
            mu: 0.0,
            sigma: 1.0,
            log_likelihood: 0.0,
            iterations: 0,
            converged: true,
        };
        let s = fit.survival(1.0);
        assert!(s > 0.49 && s < 0.51);
    }

    #[test]
    fn lognormal_fit_recovers_params() {
        let mut rng = LcgRng::new(11);
        let n = 500;
        let mut obs = Vec::with_capacity(n);
        for _ in 0..n {
            let log_t = 1.0 + 0.5 * rng.next_normal();
            obs.push(Observation::new(log_t.exp(), true).expect("ok"));
        }
        let data = Dataset::new(obs, None, None).expect("ok");
        let f = fit_log_normal(&data).expect("ok");
        assert!((f.mu - 1.0).abs() < 0.2, "mu={}", f.mu);
        assert!((f.sigma - 0.5).abs() < 0.2, "sigma={}", f.sigma);
    }
}
