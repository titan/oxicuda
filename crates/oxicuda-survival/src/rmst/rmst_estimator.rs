//! RMST estimator from a dataset with delta-method variance.

use crate::data::Dataset;
use crate::error::SurvivalResult;
use crate::nonparametric::kaplan_meier::kaplan_meier_estimate;
use crate::nonparametric::survival_function::SurvivalFunction;
use crate::rmst::restricted_mean::restricted_mean_from_curve;

/// RMST output with delta-method variance.
#[derive(Debug, Clone)]
pub struct RmstResult {
    pub tau: f64,
    pub rmst: f64,
    pub variance: f64,
}

impl RmstResult {
    /// Standard error √Var(RMST).
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.variance.max(0.0).sqrt()
    }
}

/// Estimate RMST(τ) from a dataset using the Kaplan-Meier curve.
///
/// Delta-method variance: `Var(RMST) ≈ Σ_i [∫_{t_i}^{τ} S(u) du]² · dᵢ / (nᵢ(nᵢ-dᵢ))`
pub fn rmst_from_dataset(data: &Dataset, tau: f64) -> SurvivalResult<RmstResult> {
    let km = kaplan_meier_estimate(data)?;
    let curve = SurvivalFunction::new(km.times.clone(), km.survival.clone())?;
    let area = restricted_mean_from_curve(&curve, tau)?;
    // Variance via delta method on Greenwood
    let n_steps = km.times.len();
    let mut var = 0.0_f64;
    for i in 0..n_steps {
        if km.events[i] <= 0.0 {
            continue;
        }
        let nrisk = km.at_risk[i];
        if nrisk - km.events[i] <= 0.0 {
            continue;
        }
        // contribution coefficient ∫_{t_i}^{τ} S(u) du (suffix area)
        let mut suffix = 0.0_f64;
        let mut last = km.times[i];
        let mut last_s = km.survival[i];
        for j in (i + 1)..n_steps {
            let tj = km.times[j];
            if tj >= tau {
                suffix += (tau - last).max(0.0) * last_s;
                last = tau;
                break;
            }
            suffix += (tj - last).max(0.0) * last_s;
            last = tj;
            last_s = km.survival[j];
        }
        if last < tau {
            suffix += (tau - last).max(0.0) * last_s;
        }
        let factor = km.events[i] / (nrisk * (nrisk - km.events[i]));
        var += suffix * suffix * factor;
    }
    Ok(RmstResult {
        tau,
        rmst: area,
        variance: var,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmst_unit_event_at_time_1() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        let r = rmst_from_dataset(&d, 5.0).expect("ok");
        // KM drops to 0 at t=1; RMST(5) = 1.0
        assert!((r.rmst - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn rmst_increases_with_tau() {
        let d = Dataset::from_arrays(&[1.0, 5.0, 10.0], &[true, true, true]).expect("ok");
        let r1 = rmst_from_dataset(&d, 2.0).expect("ok");
        let r2 = rmst_from_dataset(&d, 8.0).expect("ok");
        assert!(r2.rmst > r1.rmst);
    }

    #[test]
    fn rmst_variance_nonneg() {
        let d =
            Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, false, true, false]).expect("ok");
        let r = rmst_from_dataset(&d, 5.0).expect("ok");
        assert!(r.variance >= 0.0);
    }
}
