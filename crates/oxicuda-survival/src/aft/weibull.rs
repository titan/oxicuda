//! Weibull AFT model: `S(t) = exp(-(t/η)^k)`, `f(t) = (k/η)(t/η)^{k-1} S(t)`.
//!
//! Parameters: `k` (shape) and `η` (scale). Equivalent parameterisation: `λ = 1/η^k`.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Fitted Weibull model.
#[derive(Debug, Clone)]
pub struct WeibullFit {
    pub shape: f64,
    pub scale: f64,
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl WeibullFit {
    /// `S(t) = exp(-(t/η)^k)`.
    #[must_use]
    pub fn survival(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        (-(t / self.scale).powf(self.shape)).exp()
    }

    /// `h(t) = (k/η)(t/η)^{k-1}`.
    #[must_use]
    pub fn hazard(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        (self.shape / self.scale) * (t / self.scale).powf(self.shape - 1.0)
    }
}

/// MLE of a Weibull survival model with right censoring, by Newton on `(log k, log η)`.
pub fn fit_weibull(data: &Dataset) -> SurvivalResult<WeibullFit> {
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
                "Weibull requires strictly positive times".to_string(),
            ));
        }
    }
    let mean_time = times.iter().sum::<f64>() / n as f64;
    let mut log_k = 0.0_f64; // k = 1
    let mut log_eta = mean_time.ln();
    let mut ll_prev = f64::NEG_INFINITY;
    let mut iter = 0usize;
    let mut converged = false;
    for it in 0..200 {
        iter = it + 1;
        let k = log_k.exp();
        let eta = log_eta.exp();
        // log-likelihood
        let mut ll = 0.0_f64;
        // gradient wrt (log k, log eta)
        let mut grad = [0.0_f64; 2];
        let mut hess = [[0.0_f64; 2]; 2];
        for i in 0..n {
            let t = times[i];
            let r = t / eta;
            let lr = r.ln();
            let rk = r.powf(k);
            if events[i] > 0.5 {
                ll += k.ln() - log_eta + (k - 1.0) * lr - rk;
            } else {
                ll -= rk;
            }
            // Differentiate wrt log_k and log_eta.
            // Let u = k*log r = k*(log t - log_eta). Then r^k = exp(u).
            // dr^k/d(log k) = u * r^k.
            // dr^k/d(log eta) = -k * r^k.
            let u = k * lr;
            let drk_dlogk = u * rk;
            let drk_dlogeta = -k * rk;
            // event contribution: log L_i = log k - log eta + (k - 1) log r - r^k
            //   = log k - log eta + k log r - log r - r^k
            //   d/d(log k) [event] = 1 + k log r - drk_dlogk = 1 + u - u r^k
            //   d/d(log eta) [event] = -1 + k * (-1) ??? need care.
            // Easier: just compute numerical-style closed form for the survival-only piece.
            // d/d(log k) of S = -r^k => drop log r dependency through k -> dS/d(log k) = -u r^k.
            // For censored: d log S/d(log k) = -drk_dlogk = -u r^k.
            // For event piece (everything except -r^k): d/d(log k) [ log k - log eta + (k-1) log r ]
            //   = 1 + k log r = 1 + u.
            //   d/d(log eta) [ log k - log eta + (k-1) log r ] = -1 + (k-1)*(-1) = -k.
            if events[i] > 0.5 {
                grad[0] += 1.0 + u - drk_dlogk;
                grad[1] += -k - drk_dlogeta;
                // hessian: d²/d(log k)² of (1 + u) = u; of (-u r^k) = -(u + u²) r^k
                let dd_event_lkk = u;
                let dd_event_lkk_rk = -(u + u * u) * rk;
                hess[0][0] += dd_event_lkk + dd_event_lkk_rk;
                // d²/d(log eta)² of (-k) = 0; of (k r^k) = -k * (-k r^k) = k² r^k... let's recompute.
                // d/d(log eta) of (-k r^k) = -k * drk_dlogeta = -k * (-k r^k) = k² r^k
                // d²/d(log eta)² = k * (k² r^k) = k³ r^k? Re-derive cleanly:
                // f = -r^k; df/d(log eta) = -drk_dlogeta = k r^k.
                // d²f/d(log eta)² = d/d(log eta) (k r^k) = k * drk_dlogeta = -k² r^k.
                // So for event: d²/d(log eta)² of (-r^k) = -k² r^k (wait — recheck sign).
                // df/d(log eta) where f = -r^k: = -drk_dlogeta = -(-k r^k) = k r^k.
                // d²f/d(log eta)²            = d/d(log eta)[k r^k] = k * drk_dlogeta = k*(-k r^k) = -k² r^k.
                hess[1][1] += -k * k * rk;
                // d²/d(log k) d(log eta) of -r^k: drk_dlogeta = -k r^k.
                // d(-r^k)/d(log eta) = -(-k r^k) = k r^k.
                // d/d(log k)[k r^k] = k r^k (from k) + k drk_dlogk = k r^k + k u r^k = k (1+u) r^k.
                // For event piece (1+u): d/d(log eta) (1+u) = 0; -k: d/d(log k) -k = -k.
                hess[0][1] += -k - k * (1.0 + u) * rk;
                hess[1][0] = hess[0][1];
            } else {
                grad[0] += -drk_dlogk;
                grad[1] += -drk_dlogeta;
                hess[0][0] += -(u + u * u) * rk;
                hess[1][1] += -k * k * rk;
                hess[0][1] += -k * (1.0 + u) * rk;
                hess[1][0] = hess[0][1];
            }
        }
        if (ll - ll_prev).abs() < 1.0e-9 && it > 0 {
            converged = true;
            break;
        }
        ll_prev = ll;
        // Newton step: solve hess * dx = grad (since maximising, dx = -H^{-1} grad if H neg-def;
        // here we solve grad/hess but Newton update needs - H^{-1} grad). We want grad -> 0.
        // Step direction: dx = - hess_inv * grad  (for Newton on -ll). But Hessian below is the
        // Hessian of ll (which is concave near max), so dx = - H^{-1} grad would move us in
        // the direction of increasing ll only if H is negative definite. Use trust-region halving.
        let det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
        if det.abs() < 1.0e-14 {
            return Err(SurvivalError::SingularMatrix);
        }
        let inv00 = hess[1][1] / det;
        let inv01 = -hess[0][1] / det;
        let inv11 = hess[0][0] / det;
        let dx0 = -(inv00 * grad[0] + inv01 * grad[1]);
        let dx1 = -(inv01 * grad[0] + inv11 * grad[1]);
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..30 {
            let trial_k = log_k + step * dx0;
            let trial_e = log_eta + step * dx1;
            let ll_t = weibull_loglik(&times, &events, trial_k.exp(), trial_e.exp());
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
    let ll = weibull_loglik(&times, &events, k, eta);
    Ok(WeibullFit {
        shape: k,
        scale: eta,
        log_likelihood: ll,
        iterations: iter,
        converged,
    })
}

fn weibull_loglik(times: &[f64], events: &[f64], k: f64, eta: f64) -> f64 {
    if k <= 0.0 || eta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut ll = 0.0_f64;
    for i in 0..times.len() {
        let t = times[i];
        let r = t / eta;
        if !r.is_finite() {
            return f64::NEG_INFINITY;
        }
        let lr = r.ln();
        let rk = r.powf(k);
        if events[i] > 0.5 {
            ll += k.ln() - eta.ln() + (k - 1.0) * lr - rk;
        } else {
            ll -= rk;
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
    fn weibull_recovers_k_near_one_on_exponential_data() {
        // generate exponential data with rate 1; Weibull should give k ~ 1
        let mut rng = LcgRng::new(7);
        let n = 500;
        let mut obs = Vec::with_capacity(n);
        for _ in 0..n {
            let t = rng.next_exponential(1.0).max(1.0e-6);
            obs.push(Observation::new(t, true).expect("ok"));
        }
        let data = Dataset::new(obs, None, None).expect("ok");
        let f = fit_weibull(&data).expect("ok");
        assert!((f.shape - 1.0).abs() < 0.3, "k={}", f.shape);
    }

    #[test]
    fn weibull_survival_monotone_decreasing() {
        let fit = WeibullFit {
            shape: 1.5,
            scale: 2.0,
            log_likelihood: 0.0,
            iterations: 0,
            converged: true,
        };
        let s1 = fit.survival(0.5);
        let s2 = fit.survival(1.0);
        let s3 = fit.survival(5.0);
        assert!(s1 > s2 && s2 > s3);
    }

    #[test]
    fn weibull_rejects_nonpositive_time() {
        let data =
            Dataset::new(vec![Observation::new(0.0, true).expect("ok")], None, None).expect("ok");
        assert!(fit_weibull(&data).is_err());
    }
}
