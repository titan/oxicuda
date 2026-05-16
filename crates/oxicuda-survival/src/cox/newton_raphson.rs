//! Newton-Raphson optimisation for Cox partial likelihood.

use crate::data::Dataset;
use crate::error::SurvivalResult;
use crate::linalg::solve::cholesky_solve;

/// Tie-handling method for partial likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieMethod {
    Breslow,
    Efron,
}

/// Outcome of a Newton-Raphson Cox fit.
#[derive(Debug, Clone)]
pub struct NewtonResult {
    pub beta: Vec<f64>,
    pub log_likelihood: f64,
    pub score: Vec<f64>,
    pub information: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Run Newton-Raphson maximisation of the Cox partial log-likelihood.
///
/// Solves `I(β) · Δβ = U(β)` and updates `β ← β + Δβ` with halving line search.
pub fn newton_raphson_cox(
    data: &Dataset,
    init: &[f64],
    tie: TieMethod,
    tol: f64,
    max_iter: usize,
) -> SurvivalResult<NewtonResult> {
    let p = init.len();
    let mut beta = init.to_vec();
    let loglik_fn = |b: &[f64]| -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
        match tie {
            TieMethod::Breslow => crate::cox::breslow_ties::breslow_log_likelihood(data, b),
            TieMethod::Efron => crate::cox::efron_ties::efron_log_likelihood(data, b),
        }
    };
    let (mut ll, mut score, mut info) = loglik_fn(&beta)?;
    let mut converged = false;
    let mut iter = 0usize;
    for it in 0..max_iter {
        iter = it + 1;
        // Check convergence on max |score|
        let max_score = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_score < tol {
            converged = true;
            break;
        }
        let delta = match cholesky_solve(&info, &score, p) {
            Ok(d) => d,
            Err(_) => {
                // try ridge boost
                let mut info_ridge = info.clone();
                for d in 0..p {
                    info_ridge[d * p + d] += 1.0e-4;
                }
                cholesky_solve(&info_ridge, &score, p)?
            }
        };
        // line search via halving
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((ll_new, sc_new, info_new)) = loglik_fn(&trial) {
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
            // tiny step taken; treat as converged if score is small
            break;
        }
    }
    if !converged {
        let max_score = score.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max_score < tol {
            converged = true;
        }
    }
    Ok(NewtonResult {
        beta,
        log_likelihood: ll,
        score,
        information: info,
        iterations: iter,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_converges_well_conditioned() {
        // Use synthetic data from an exponential Cox model — well-conditioned, converges.
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(101);
        let n = 200;
        let beta_true = 0.5_f64;
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let lambda = (beta_true * x).exp();
            let t = rng.next_exponential(lambda).max(1.0e-6);
            obs.push(crate::data::Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
        }
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let res = newton_raphson_cox(&data, &[0.0], TieMethod::Breslow, 1.0e-6, 50).expect("ok");
        assert!(res.converged);
        assert!(res.iterations < 50);
        // β should be positive
        assert!(res.beta[0] > 0.0);
    }

    #[test]
    fn newton_efron_also_converges() {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(303);
        let n = 100;
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let lambda = (0.3 * x).exp();
            let t = rng.next_exponential(lambda).max(1.0e-6);
            obs.push(crate::data::Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
        }
        let data = Dataset::new(obs, Some(cov), None).expect("ok");
        let res = newton_raphson_cox(&data, &[0.0], TieMethod::Efron, 1.0e-6, 80).expect("ok");
        assert!(res.converged);
    }
}
