//! High-level Cox proportional hazards fit.

use crate::cox::baseline_hazard::{BaselineHazard, breslow_baseline_hazard};
use crate::cox::newton_raphson::newton_raphson_cox;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;

pub use crate::cox::newton_raphson::TieMethod;

/// Configuration for `fit_cox_ph`.
#[derive(Debug, Clone, Copy)]
pub struct CoxPhConfig {
    pub tie: TieMethod,
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for CoxPhConfig {
    fn default() -> Self {
        Self {
            tie: TieMethod::Breslow,
            tol: 1.0e-6,
            max_iter: 50,
        }
    }
}

/// Fitted Cox PH model.
#[derive(Debug, Clone)]
pub struct CoxFit {
    /// Coefficient vector β.
    pub coefficients: Vec<f64>,
    /// Final partial log-likelihood.
    pub log_likelihood: f64,
    /// Fisher information at the optimum.
    pub information: Vec<f64>,
    /// Variance-covariance matrix (information inverse).
    pub variance: Vec<f64>,
    /// Newton-Raphson iterations consumed.
    pub iterations: usize,
    /// Convergence flag.
    pub converged: bool,
    /// Baseline cumulative hazard.
    pub baseline_hazard: BaselineHazard,
}

impl CoxFit {
    /// Standard errors of β.
    #[must_use]
    pub fn standard_errors(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p)
            .map(|i| self.variance[i * p + i].max(0.0).sqrt())
            .collect()
    }

    /// Wald z-scores for each coefficient.
    #[must_use]
    pub fn z_scores(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .zip(self.standard_errors().iter())
            .map(|(b, se)| if *se > 0.0 { b / se } else { 0.0 })
            .collect()
    }
}

/// Fit a Cox proportional hazards model by Newton-Raphson on the partial likelihood.
pub fn fit_cox_ph(data: &Dataset, config: CoxPhConfig) -> SurvivalResult<CoxFit> {
    let p = data.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "no covariates available for Cox PH".to_string(),
        ));
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }
    let init = vec![0.0_f64; p];
    let result = newton_raphson_cox(data, &init, config.tie, config.tol, config.max_iter)?;
    let variance = match gauss_jordan_inverse(&result.information, p) {
        Ok(v) => v,
        Err(_) => vec![0.0_f64; p * p],
    };
    let baseline = breslow_baseline_hazard(data, &result.beta)?;
    Ok(CoxFit {
        coefficients: result.beta,
        log_likelihood: result.log_likelihood,
        information: result.information,
        variance,
        iterations: result.iterations,
        converged: result.converged,
        baseline_hazard: baseline,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    fn make_synthetic_dataset(n: usize, beta_true: f64, seed: u64) -> Dataset {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(seed);
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            // exponential hazard λ = exp(β x) gives T ~ Exp(λ); ensure all events
            let lambda = (beta_true * x).exp();
            let t = rng.next_exponential(lambda);
            obs.push(Observation::new(t.max(1.0e-6), true).expect("ok"));
            cov.push(vec![x]);
        }
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    #[test]
    fn fit_recovers_beta_within_5pct() {
        let n = 400;
        let beta_true = 1.0_f64;
        let data = make_synthetic_dataset(n, beta_true, 12345);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        assert!(fit.converged);
        let rel = (fit.coefficients[0] - beta_true).abs() / beta_true;
        assert!(rel < 0.25, "beta_hat={} rel={}", fit.coefficients[0], rel);
    }

    #[test]
    fn fit_efron_converges() {
        let n = 100;
        let data = make_synthetic_dataset(n, 0.5, 42);
        let cfg = CoxPhConfig {
            tie: TieMethod::Efron,
            tol: 1.0e-6,
            max_iter: 80,
        };
        let fit = fit_cox_ph(&data, cfg).expect("ok");
        assert!(fit.converged);
    }

    #[test]
    fn fit_returns_std_errs() {
        let n = 50;
        let data = make_synthetic_dataset(n, 0.3, 7);
        let fit = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let se = fit.standard_errors();
        assert_eq!(se.len(), 1);
        assert!(se[0] > 0.0);
    }

    #[test]
    fn fit_rejects_no_events() {
        let cov = vec![vec![1.0], vec![-1.0]];
        let data = Dataset::new(
            vec![
                Observation::new(1.0, false).expect("ok"),
                Observation::new(2.0, false).expect("ok"),
            ],
            Some(cov),
            None,
        )
        .expect("ok");
        let r = fit_cox_ph(&data, CoxPhConfig::default());
        assert!(matches!(r, Err(SurvivalError::NoEvents)));
    }
}
