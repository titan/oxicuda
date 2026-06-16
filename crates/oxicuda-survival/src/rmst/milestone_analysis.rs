//! Milestone analysis for time-to-event trials (Royston & Parmar 2011,
//! *Stat. Med.*; *BMC Med. Res. Methodol.* 2013).
//!
//! When the proportional-hazards assumption is doubtful (e.g. delayed treatment
//! effects in immunotherapy overall-survival trials) a single hazard ratio is
//! hard to interpret. **Milestone analysis** instead summarises each arm at a
//! small set of clinically pre-specified time points (*milestones*) `τ₁, τ₂, …`
//! by two complementary quantities:
//!
//! * **Milestone survival** `Ŝ(τ)` — the Kaplan-Meier probability of being
//!   event-free at `τ`, with a Greenwood log-log confidence interval, and
//! * **Restricted mean survival time** `RMST(τ) = ∫₀^τ S(u) du`, the average
//!   event-free time over `[0, τ]`, with a delta-method variance.
//!
//! For two-arm comparisons the module also reports the **milestone survival
//! difference** `Ŝ_A(τ) − Ŝ_B(τ)` and the **RMST difference** with their
//! standard errors and Wald confidence intervals — model-free effect measures
//! that remain valid under non-proportional hazards.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::nonparametric::kaplan_meier::{KaplanMeier, kaplan_meier_estimate};
use crate::nonparametric::survival_function::SurvivalFunction;
use crate::rmst::restricted_mean::restricted_mean_from_curve;

/// Per-milestone summary for a single arm.
#[derive(Debug, Clone)]
pub struct MilestoneSummary {
    /// Milestone time `τ`.
    pub tau: f64,
    /// Kaplan-Meier survival probability `Ŝ(τ)`.
    pub survival: f64,
    /// Greenwood variance of `Ŝ(τ)`.
    pub survival_var: f64,
    /// Lower/upper log-log `(1 − α)` CI bounds for `Ŝ(τ)`.
    pub survival_ci: (f64, f64),
    /// Restricted mean survival time `RMST(τ)`.
    pub rmst: f64,
    /// Delta-method variance of `RMST(τ)`.
    pub rmst_var: f64,
}

impl MilestoneSummary {
    /// Standard error of the milestone survival.
    #[must_use]
    pub fn survival_se(&self) -> f64 {
        self.survival_var.max(0.0).sqrt()
    }

    /// Standard error of the milestone RMST.
    #[must_use]
    pub fn rmst_se(&self) -> f64 {
        self.rmst_var.max(0.0).sqrt()
    }
}

/// Two-arm milestone contrast at a single milestone time.
#[derive(Debug, Clone)]
pub struct MilestoneContrast {
    /// Milestone time `τ`.
    pub tau: f64,
    /// `Ŝ_A(τ) − Ŝ_B(τ)`.
    pub survival_diff: f64,
    /// Standard error of the survival difference.
    pub survival_diff_se: f64,
    /// Wald CI for the survival difference.
    pub survival_diff_ci: (f64, f64),
    /// `RMST_A(τ) − RMST_B(τ)`.
    pub rmst_diff: f64,
    /// Standard error of the RMST difference.
    pub rmst_diff_se: f64,
    /// Wald CI for the RMST difference.
    pub rmst_diff_ci: (f64, f64),
}

/// Evaluate the Kaplan-Meier survival and Greenwood variance exactly at a query
/// time `tau` (value carried forward from the last event time `≤ tau`).
fn km_at(km: &KaplanMeier, tau: f64) -> (f64, f64) {
    let mut s = 1.0_f64;
    let mut var = 0.0_f64;
    for i in 0..km.times.len() {
        if km.times[i] <= tau {
            s = km.survival[i];
            var = km.greenwood_var[i];
        } else {
            break;
        }
    }
    (s, var)
}

/// Log-log transformed `(1 − α)` CI for a survival probability, mirroring the
/// transform used in [`crate::nonparametric::kaplan_meier`].
fn loglog_ci(s: f64, var: f64, z: f64) -> (f64, f64) {
    if s <= 0.0 || s >= 1.0 || var <= 0.0 {
        return (s, s);
    }
    let ln_s = s.ln();
    let var_logs = var / (s * s);
    let se_loglog = (var_logs / (ln_s * ln_s)).max(0.0).sqrt();
    let theta = (-ln_s).ln();
    let lo_log = theta - z * se_loglog;
    let hi_log = theta + z * se_loglog;
    let lo = (-(hi_log.exp())).exp();
    let hi = (-(lo_log.exp())).exp();
    (lo, hi)
}

/// RMST(τ) and its delta-method variance from a Kaplan-Meier fit.
///
/// `Var(RMST) ≈ Σ_i [∫_{t_i}^{τ} S(u) du]² · dᵢ / (nᵢ(nᵢ − dᵢ))`.
fn rmst_with_var(km: &KaplanMeier, tau: f64) -> SurvivalResult<(f64, f64)> {
    let curve = SurvivalFunction::new(km.times.clone(), km.survival.clone())?;
    let area = restricted_mean_from_curve(&curve, tau)?;

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
        // Suffix area ∫_{t_i}^{τ} S(u) du.
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
    Ok((area, var))
}

/// Compute milestone summaries for one arm at each requested milestone time.
///
/// `z` is the normal quantile for the confidence level (e.g. `1.96` for 95%).
pub fn milestone_analysis(
    data: &Dataset,
    milestones: &[f64],
    z: f64,
) -> SurvivalResult<Vec<MilestoneSummary>> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if milestones.is_empty() {
        return Err(SurvivalError::InvalidParameter(
            "at least one milestone time is required".to_string(),
        ));
    }
    for &m in milestones {
        if !m.is_finite() || m < 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "milestone must be finite and >= 0, got {m}"
            )));
        }
    }
    let km = kaplan_meier_estimate(data)?;

    let mut out = Vec::with_capacity(milestones.len());
    for &tau in milestones {
        let (s, var) = km_at(&km, tau);
        let ci = loglog_ci(s, var, z);
        let (rmst, rmst_var) = rmst_with_var(&km, tau)?;
        out.push(MilestoneSummary {
            tau,
            survival: s,
            survival_var: var,
            survival_ci: ci,
            rmst,
            rmst_var,
        });
    }
    Ok(out)
}

/// Two-arm milestone comparison: arm A vs arm B at each milestone.
///
/// Returns per-milestone contrasts using independent-sample variances for both
/// the survival difference and the RMST difference.
pub fn milestone_two_arm(
    arm_a: &Dataset,
    arm_b: &Dataset,
    milestones: &[f64],
    z: f64,
) -> SurvivalResult<Vec<MilestoneContrast>> {
    let sa = milestone_analysis(arm_a, milestones, z)?;
    let sb = milestone_analysis(arm_b, milestones, z)?;

    let mut out = Vec::with_capacity(milestones.len());
    for (a, b) in sa.iter().zip(sb.iter()) {
        let surv_diff = a.survival - b.survival;
        let surv_se = (a.survival_var + b.survival_var).max(0.0).sqrt();
        let surv_ci = (surv_diff - z * surv_se, surv_diff + z * surv_se);

        let rmst_diff = a.rmst - b.rmst;
        let rmst_se = (a.rmst_var + b.rmst_var).max(0.0).sqrt();
        let rmst_ci = (rmst_diff - z * rmst_se, rmst_diff + z * rmst_se);

        out.push(MilestoneContrast {
            tau: a.tau,
            survival_diff: surv_diff,
            survival_diff_se: surv_se,
            survival_diff_ci: surv_ci,
            rmst_diff,
            rmst_diff_se: rmst_se,
            rmst_diff_ci: rmst_ci,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn survival_at_milestone_matches_km() {
        // 4 events at 1,2,3,4 → KM drops 1, .75, .5, .25, 0.
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let res = milestone_analysis(&d, &[2.0], 1.96).expect("ok");
        assert!(
            approx(res[0].survival, 0.5, 1e-9),
            "S(2) = {}",
            res[0].survival
        );
    }

    #[test]
    fn survival_before_first_event_is_one() {
        let d = Dataset::from_arrays(&[5.0, 6.0, 7.0], &[true, true, true]).expect("ok");
        let res = milestone_analysis(&d, &[1.0], 1.96).expect("ok");
        assert!(approx(res[0].survival, 1.0, 1e-12));
    }

    #[test]
    fn rmst_at_milestone_correct() {
        // All survive past τ: RMST(τ) = τ when no events before τ.
        let d = Dataset::from_arrays(&[10.0, 11.0, 12.0], &[true, true, true]).expect("ok");
        let res = milestone_analysis(&d, &[5.0], 1.96).expect("ok");
        assert!(approx(res[0].rmst, 5.0, 1e-9), "RMST(5) = {}", res[0].rmst);
    }

    #[test]
    fn rmst_increases_with_milestone() {
        let d = Dataset::from_arrays(&[1.0, 5.0, 10.0], &[true, true, true]).expect("ok");
        let res = milestone_analysis(&d, &[2.0, 8.0], 1.96).expect("ok");
        assert!(res[1].rmst > res[0].rmst);
    }

    #[test]
    fn multiple_milestones_returned_in_order() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0, 5.0], &[true, true, true, true, true])
            .expect("ok");
        let res = milestone_analysis(&d, &[1.0, 3.0, 5.0], 1.96).expect("ok");
        assert_eq!(res.len(), 3);
        assert!(res[0].survival >= res[1].survival && res[1].survival >= res[2].survival);
    }

    #[test]
    fn confidence_interval_brackets_survival() {
        let d = Dataset::from_arrays(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[true, true, true, true, true, true],
        )
        .expect("ok");
        let res = milestone_analysis(&d, &[3.0], 1.96).expect("ok");
        let (lo, hi) = res[0].survival_ci;
        assert!(lo <= res[0].survival && res[0].survival <= hi);
        assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
    }

    #[test]
    fn survival_se_nonneg() {
        let d =
            Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, false, true, false]).expect("ok");
        let res = milestone_analysis(&d, &[2.5], 1.96).expect("ok");
        assert!(res[0].survival_se() >= 0.0);
        assert!(res[0].rmst_se() >= 0.0);
    }

    #[test]
    fn two_arm_difference_sign_correct() {
        // Arm A survives longer than arm B → positive survival & RMST diff at τ.
        let arm_a =
            Dataset::from_arrays(&[8.0, 9.0, 10.0, 11.0], &[true, true, true, true]).expect("ok");
        let arm_b =
            Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let res = milestone_two_arm(&arm_a, &arm_b, &[5.0], 1.96).expect("ok");
        assert!(
            res[0].survival_diff > 0.0,
            "surv diff {}",
            res[0].survival_diff
        );
        assert!(res[0].rmst_diff > 0.0, "rmst diff {}", res[0].rmst_diff);
    }

    #[test]
    fn two_arm_identical_arms_zero_difference() {
        let arm =
            Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let res = milestone_two_arm(&arm, &arm, &[2.0], 1.96).expect("ok");
        assert!(approx(res[0].survival_diff, 0.0, 1e-12));
        assert!(approx(res[0].rmst_diff, 0.0, 1e-12));
    }

    #[test]
    fn two_arm_ci_brackets_difference() {
        let arm_a =
            Dataset::from_arrays(&[5.0, 6.0, 7.0, 8.0], &[true, true, true, true]).expect("ok");
        let arm_b =
            Dataset::from_arrays(&[2.0, 3.0, 4.0, 5.0], &[true, true, true, true]).expect("ok");
        let res = milestone_two_arm(&arm_a, &arm_b, &[4.0], 1.96).expect("ok");
        let (lo, hi) = res[0].rmst_diff_ci;
        assert!(lo <= res[0].rmst_diff && res[0].rmst_diff <= hi);
        let (slo, shi) = res[0].survival_diff_ci;
        assert!(slo <= res[0].survival_diff && res[0].survival_diff <= shi);
    }

    #[test]
    fn rejects_empty_milestones() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let res = milestone_analysis(&d, &[], 1.96);
        assert!(matches!(res, Err(SurvivalError::InvalidParameter(_))));
    }

    #[test]
    fn rejects_negative_milestone() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let res = milestone_analysis(&d, &[-1.0], 1.96);
        assert!(matches!(res, Err(SurvivalError::InvalidParameter(_))));
    }
}
