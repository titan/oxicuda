//! Summary statistics derived from a survival curve.

use crate::error::{SurvivalError, SurvivalResult};
use crate::nonparametric::survival_function::SurvivalFunction;

/// Median survival = smallest `t` such that `S(t) <= 0.5`. Returns `None` if no such `t`.
#[must_use]
pub fn median_survival(s: &SurvivalFunction) -> Option<f64> {
    for (t, sv) in s.times.iter().zip(s.survival.iter()) {
        if *sv <= 0.5 {
            return Some(*t);
        }
    }
    None
}

/// `S(τ)` evaluated on a survival curve.
#[must_use]
pub fn survival_at_horizon(s: &SurvivalFunction, tau: f64) -> f64 {
    s.eval(tau)
}

/// Restricted mean survival time (alias delegating to the `rmst` module).
pub fn restricted_mean_metric(s: &SurvivalFunction, tau: f64) -> SurvivalResult<f64> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "tau must be >= 0: {tau}"
        )));
    }
    crate::rmst::restricted_mean::restricted_mean_from_curve(s, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_finds_first_drop_below_half() {
        let s = SurvivalFunction::new(vec![1.0, 2.0, 3.0], vec![0.75, 0.5, 0.25]).expect("ok");
        let m = median_survival(&s).expect("found");
        assert_eq!(m, 2.0);
    }

    #[test]
    fn median_none_if_above_half() {
        let s = SurvivalFunction::new(vec![1.0, 2.0], vec![0.8, 0.7]).expect("ok");
        assert!(median_survival(&s).is_none());
    }

    #[test]
    fn survival_at_horizon_evaluates() {
        let s = SurvivalFunction::new(vec![1.0, 2.0], vec![0.5, 0.25]).expect("ok");
        assert!((survival_at_horizon(&s, 1.5) - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn restricted_mean_rejects_neg_tau() {
        let s = SurvivalFunction::new(vec![1.0], vec![0.5]).expect("ok");
        assert!(restricted_mean_metric(&s, -1.0).is_err());
    }
}
