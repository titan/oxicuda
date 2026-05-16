//! Cause-specific hazard Cox regression.
//!
//! Treats events of other causes as censored, then fits standard Cox.

use crate::cox::cox_ph::{CoxFit, CoxPhConfig, fit_cox_ph};
use crate::data::{Dataset, Observation};
use crate::error::{SurvivalError, SurvivalResult};

/// Fit a Cox model for the cause-specific hazard of `target_cause`.
///
/// Subjects whose event was a different cause are recoded as censored.
pub fn cause_specific_cox(
    data: &Dataset,
    causes: &[u32],
    target_cause: u32,
    config: CoxPhConfig,
) -> SurvivalResult<CoxFit> {
    if data.len() != causes.len() {
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
    let mut obs = Vec::with_capacity(data.len());
    for (i, o) in data.observations.iter().enumerate() {
        let new_event = o.event && causes[i] == target_cause;
        obs.push(Observation::new(o.time, new_event)?);
    }
    let new_data = Dataset::new(obs, data.covariates.clone(), data.strata.clone())?;
    fit_cox_ph(&new_data, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cause_specific_recodes_other_causes_as_censored() {
        // Larger synthetic dataset for stable convergence
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(101);
        let n = 100;
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
        let d = Dataset::new(obs, Some(cov), None).expect("ok");
        let fit = cause_specific_cox(&d, &causes, 1, CoxPhConfig::default()).expect("ok");
        assert!(fit.iterations > 0);
    }

    #[test]
    fn cause_specific_rejects_size_mismatch() {
        let d = Dataset::new(
            vec![Observation::new(1.0, true).expect("ok")],
            Some(vec![vec![1.0]]),
            None,
        )
        .expect("ok");
        let r = cause_specific_cox(&d, &[1, 1], 1, CoxPhConfig::default());
        assert!(r.is_err());
    }
}
