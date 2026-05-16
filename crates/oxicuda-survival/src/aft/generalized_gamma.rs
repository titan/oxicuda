//! Generalised gamma AFT (Stacy): parameters `(μ, σ, q)`.
//!
//! Density (for `q != 0`):
//! ```text
//!   f(t) = |q| / (σ t Γ(1/q²)) * (q^{-2})^{1/q²} * exp(z/q - exp(qz)/q²)
//! ```
//! with `z = (log t - μ)/σ`. For `q=0` reduces to log-normal; `q=1` reduces to Weibull.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::special::gammaln;

#[derive(Debug, Clone)]
pub struct GeneralizedGammaFit {
    pub mu: f64,
    pub sigma: f64,
    pub q: f64,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn gen_gamma_log_density(t: f64, mu: f64, sigma: f64, q: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if q.abs() < 1.0e-6 {
        // log-normal limit
        let z = (t.ln() - mu) / sigma;
        return -t.ln() - sigma.ln() - 0.5 * std::f64::consts::TAU.ln() - 0.5 * z * z;
    }
    let q2 = q * q;
    let z = (t.ln() - mu) / sigma;
    let qz = q * z;
    let eqz = qz.exp();
    let alpha = 1.0 / q2;
    // f = |q|/(σ t Γ(α)) * α^α * exp(z/q − exp(qz)/q²)
    // log f = log|q| - log σ - log t - gammaln(α) + α log α + z/q - exp(qz)/q²
    q.abs().ln() - sigma.ln() - t.ln() - gammaln(alpha) + alpha * alpha.ln() + z / q - eqz / q2
}

fn gen_gamma_log_survival(t: f64, mu: f64, sigma: f64, q: f64) -> f64 {
    // For q != 0: S(t) = 1 - Γ_lower(α, eqz/q²) / Γ(α)   if q > 0
    //              S(t) = Γ_lower(α, eqz/q²) / Γ(α)       if q < 0
    if t <= 0.0 {
        return 0.0; // log(1)
    }
    if q.abs() < 1.0e-6 {
        // log-normal
        let z = (t.ln() - mu) / sigma;
        return ((1.0 - std_normal_cdf(z)).max(1.0e-300)).ln();
    }
    let q2 = q * q;
    let z = (t.ln() - mu) / sigma;
    let qz = q * z;
    let eqz = qz.exp();
    let alpha = 1.0 / q2;
    let arg = eqz / q2;
    let p = reg_lower_gamma(alpha, arg);
    let s = if q > 0.0 { 1.0 - p } else { p };
    s.max(1.0e-300).ln()
}

/// Regularised lower incomplete gamma `P(a, x) = γ(a,x) / Γ(a)` via series + continued fraction.
fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // series
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut term = sum;
        for _ in 0..200 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1.0e-12 {
                break;
            }
        }
        sum * (a * x.ln() - x - gammaln(a)).exp()
    } else {
        // continued fraction for Q(a,x) = 1 - P(a,x)
        let mut b = x + 1.0 - a;
        let mut c = 1.0e308_f64;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1.0e-300 {
                d = 1.0e-300;
            }
            c = b + an / c;
            if c.abs() < 1.0e-300 {
                c = 1.0e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1.0e-12 {
                break;
            }
        }
        let q = (a * x.ln() - x - gammaln(a)).exp() * h;
        1.0 - q
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

fn loglik(times: &[f64], events: &[f64], mu: f64, sigma: f64, q: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0_f64;
    for i in 0..times.len() {
        if events[i] > 0.5 {
            ll += gen_gamma_log_density(times[i], mu, sigma, q);
        } else {
            ll += gen_gamma_log_survival(times[i], mu, sigma, q);
        }
    }
    ll
}

/// Fit generalized gamma by coordinate gradient descent with numerical gradients.
pub fn fit_generalized_gamma(data: &Dataset) -> SurvivalResult<GeneralizedGammaFit> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let times = data.times();
    let events = data.events_f64();
    for t in &times {
        if *t <= 0.0 {
            return Err(SurvivalError::InvalidParameter(
                "generalized gamma requires positive times".to_string(),
            ));
        }
    }
    let n = times.len();
    let mean_log_t = times.iter().map(|t| t.ln()).sum::<f64>() / n as f64;
    let var_log_t = times
        .iter()
        .map(|t| (t.ln() - mean_log_t).powi(2))
        .sum::<f64>()
        / n as f64;
    let mut mu = mean_log_t;
    let mut sigma = (var_log_t.max(1.0e-6)).sqrt();
    let mut q = 1.0_f64; // start at Weibull
    let mut converged = false;
    let mut iter = 0usize;
    let mut ll_prev = f64::NEG_INFINITY;
    let eps = 1.0e-4;
    for it in 0..300 {
        iter = it + 1;
        let ll = loglik(&times, &events, mu, sigma, q);
        if (ll - ll_prev).abs() < 1.0e-9 && it > 0 {
            converged = true;
            break;
        }
        ll_prev = ll;
        // numerical gradient
        let g_mu = (loglik(&times, &events, mu + eps, sigma, q)
            - loglik(&times, &events, mu - eps, sigma, q))
            / (2.0 * eps);
        let g_sigma = (loglik(&times, &events, mu, sigma + eps, q)
            - loglik(&times, &events, mu, sigma - eps, q))
            / (2.0 * eps);
        let g_q = (loglik(&times, &events, mu, sigma, q + eps)
            - loglik(&times, &events, mu, sigma, q - eps))
            / (2.0 * eps);
        // simple steepest ascent with line search
        let mut step = 0.1_f64;
        let mut accepted = false;
        for _ in 0..30 {
            let m_t = mu + step * g_mu;
            let s_t = (sigma + step * g_sigma).max(1.0e-5);
            let q_t = q + step * g_q;
            let ll_t = loglik(&times, &events, m_t, s_t, q_t);
            if ll_t.is_finite() && ll_t > ll - 1.0e-12 {
                mu = m_t;
                sigma = s_t;
                q = q_t;
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
    let ll = loglik(&times, &events, mu, sigma, q);
    Ok(GeneralizedGammaFit {
        mu,
        sigma,
        q,
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
    fn reg_lower_gamma_known_values() {
        // P(1, x) = 1 - exp(-x)
        assert!((reg_lower_gamma(1.0, 1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1.0e-8);
        assert!((reg_lower_gamma(1.0, 0.5) - (1.0 - (-0.5_f64).exp())).abs() < 1.0e-8);
    }

    #[test]
    fn gen_gamma_density_finite() {
        let v = gen_gamma_log_density(1.0, 0.0, 1.0, 1.0);
        assert!(v.is_finite());
    }

    #[test]
    fn gen_gamma_survival_decreases() {
        let s1 = gen_gamma_log_survival(0.5, 0.0, 1.0, 1.0);
        let s2 = gen_gamma_log_survival(2.0, 0.0, 1.0, 1.0);
        assert!(s1 > s2);
    }

    #[test]
    fn gen_gamma_fit_returns_finite() {
        let data = Dataset::new(
            (1..20)
                .map(|i| Observation::new(i as f64, true).expect("ok"))
                .collect(),
            None,
            None,
        )
        .expect("ok");
        let f = fit_generalized_gamma(&data).expect("ok");
        assert!(f.log_likelihood.is_finite());
        assert!(f.sigma > 0.0);
    }
}
