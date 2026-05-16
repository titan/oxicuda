//! Integrate `S(t)` up to a horizon `τ` (rectangle/step-function rule).

use crate::error::{SurvivalError, SurvivalResult};
use crate::nonparametric::survival_function::SurvivalFunction;

/// RMST(τ) = ∫₀^τ S(t) dt, computed as a sum of step-function rectangles.
pub fn restricted_mean_from_curve(s: &SurvivalFunction, tau: f64) -> SurvivalResult<f64> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "tau must be >= 0, got {tau}"
        )));
    }
    if s.is_empty() {
        return Ok(tau);
    }
    let mut area = 0.0_f64;
    let mut last = 0.0_f64;
    let mut last_s = 1.0_f64;
    for i in 0..s.len() {
        let t = s.times[i];
        if t >= tau {
            area += (tau - last).max(0.0) * last_s;
            return Ok(area);
        }
        area += (t - last).max(0.0) * last_s;
        last = t;
        last_s = s.survival[i];
    }
    // tail
    area += (tau - last).max(0.0) * last_s;
    Ok(area)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmst_constant_one() {
        // S(t) = 1 for all t => RMST = tau
        let s = SurvivalFunction::new(vec![0.0], vec![1.0]).expect("ok");
        let r = restricted_mean_from_curve(&s, 5.0).expect("ok");
        assert!((r - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn rmst_one_step() {
        // S = 1 until t=1, then 0 after; RMST(tau) = min(tau, 1)
        let s = SurvivalFunction::new(vec![1.0], vec![0.0]).expect("ok");
        let r = restricted_mean_from_curve(&s, 5.0).expect("ok");
        assert!((r - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn rmst_two_steps() {
        // S = 1 for t<1, 0.5 for 1<=t<2, 0 for t>=2; RMST(3) = 1*1 + 1*0.5 + 1*0 = 1.5
        let s = SurvivalFunction::new(vec![1.0, 2.0], vec![0.5, 0.0]).expect("ok");
        let r = restricted_mean_from_curve(&s, 3.0).expect("ok");
        assert!((r - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn rmst_rejects_negative_tau() {
        let s = SurvivalFunction::new(vec![1.0], vec![0.5]).expect("ok");
        assert!(restricted_mean_from_curve(&s, -1.0).is_err());
    }
}
