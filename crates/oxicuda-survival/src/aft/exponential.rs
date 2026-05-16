//! Exponential AFT: hazard `λ(t) = λ` (constant). Closed-form MLE.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Fit of an exponential model: `λ̂ = (# events) / (Σ t_i)`.
#[derive(Debug, Clone)]
pub struct ExponentialFit {
    pub rate: f64,
    pub log_likelihood: f64,
}

/// Closed-form MLE of an exponential survival model with right censoring.
pub fn fit_exponential(data: &Dataset) -> SurvivalResult<ExponentialFit> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let d: f64 = data.n_events() as f64;
    if d == 0.0 {
        return Err(SurvivalError::NoEvents);
    }
    let total_time: f64 = data.observations.iter().map(|o| o.time).sum();
    if total_time <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "total time non-positive".to_string(),
        ));
    }
    let lambda = d / total_time;
    let ll = d * lambda.ln() - lambda * total_time;
    Ok(ExponentialFit {
        rate: lambda,
        log_likelihood: ll,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    #[test]
    fn exp_fit_pure_events() {
        // 4 deaths at t=1,2,3,4; sum=10; λ̂=4/10=0.4
        let data = Dataset::new(
            vec![
                Observation::new(1.0, true).expect("ok"),
                Observation::new(2.0, true).expect("ok"),
                Observation::new(3.0, true).expect("ok"),
                Observation::new(4.0, true).expect("ok"),
            ],
            None,
            None,
        )
        .expect("ok");
        let f = fit_exponential(&data).expect("ok");
        assert!((f.rate - 0.4).abs() < 1.0e-10);
    }

    #[test]
    fn exp_fit_with_censoring() {
        // 3 events + 2 censored at t=1; events=3; total=5; λ̂=3/5
        let data = Dataset::new(
            vec![
                Observation::new(1.0, true).expect("ok"),
                Observation::new(1.0, true).expect("ok"),
                Observation::new(1.0, true).expect("ok"),
                Observation::new(1.0, false).expect("ok"),
                Observation::new(1.0, false).expect("ok"),
            ],
            None,
            None,
        )
        .expect("ok");
        let f = fit_exponential(&data).expect("ok");
        assert!((f.rate - 3.0 / 5.0).abs() < 1.0e-10);
    }

    #[test]
    fn exp_fit_rejects_no_events() {
        let data =
            Dataset::new(vec![Observation::new(1.0, false).expect("ok")], None, None).expect("ok");
        assert!(matches!(
            fit_exponential(&data),
            Err(SurvivalError::NoEvents)
        ));
    }
}
