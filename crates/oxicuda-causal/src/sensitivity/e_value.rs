//! E-value sensitivity bound — VanderWeele & Ding (2017).
//!
//! VanderWeele TJ, Ding P. *Sensitivity analysis in observational research:
//! introducing the E-value.* Annals of Internal Medicine 167(4): 268–274
//! (2017). The original derivation builds on Ding & VanderWeele (2016),
//! *Sensitivity analysis without assumptions*, Epidemiology 27(3): 368–377.
//!
//! # Problem
//!
//! An estimated treatment-effect contrast (risk ratio, odds ratio, hazard
//! ratio or risk difference) may be confounded by an unobserved variable
//! `U`. The **E-value** is the minimum strength of association — on the
//! risk-ratio scale — that an unobserved `U` would need to share with
//! both the treatment and the outcome, simultaneously, to fully explain
//! away the observed effect.
//!
//! For a risk ratio `RR > 1` the bound (VW & Ding 2017 eq. 1) is
//!
//! ```text
//!   E = RR + sqrt(RR · (RR − 1)).
//! ```
//!
//! For `RR < 1` apply the formula to `1/RR` (the protective effect is
//! converted to a risk-elevating effect of the reverse contrast). The
//! E-value is always reported on the `≥ 1` scale.
//!
//! # Effect-type conversions to RR
//!
//! - **Odds ratio** (rare outcome): `OR ≈ RR`, so the E-value formula is
//!   applied directly. For non-rare outcomes we use the VW & Ding §4
//!   conversion `RR_approx = OR / (1 − p₀ + p₀·OR)` where `p₀` is the
//!   baseline probability of the outcome.
//! - **Hazard ratio**: when the outcome is rare (`p₀ ≤ 0.15`) `HR ≈ RR`;
//!   otherwise we use the VW & Ding §S.2 conversion
//!   `RR_approx = (1 − 0.5^√HR) / (1 − 0.5^√(1/HR))`.
//! - **Risk difference**: `RR = (p₀ + RD) / p₀` (only defined when
//!   `p₀ + RD > 0`).
//!
//! # Confidence-bound E-value
//!
//! The `e_value_ci` field reports the E-value for the CI bound that is
//! closer to the null (`RR = 1`). If the CI crosses the null, the
//! confidence bound is reported as `1.0`.

use crate::error::{CausalError, CausalResult};

/// Knob struct for E-value computations. Only the OR pathway currently
/// uses `rare_outcome`; the others infer the regime from `baseline_p`
/// when needed.
#[derive(Clone, Debug, Default)]
pub struct EValueConfig {
    /// When `true` we treat the odds ratio as a risk ratio directly
    /// (rare-outcome equivalence). When `false` the baseline probability
    /// is used to convert OR to RR.
    pub rare_outcome: bool,
}

/// Originating effect contrast — kept on the result for traceability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectType {
    /// Risk ratio.
    RiskRatio,
    /// Odds ratio.
    OddsRatio,
    /// Hazard ratio.
    HazardRatio,
    /// Risk difference.
    RiskDifference,
}

/// Reported E-values for the point estimate and the CI bound closer to
/// the null.
#[derive(Debug, Clone)]
pub struct EValueResult {
    /// E-value for the point estimate.
    pub e_value_point: f64,
    /// E-value for the CI bound closer to the null; `1.0` if the CI
    /// crosses the null.
    pub e_value_ci: f64,
    /// Effect scale of the input (kept for downstream reporting).
    pub effect_type: EffectType,
}

/// Stateless namespace for the E-value derivation.
pub struct EValue;

impl EValue {
    /// E-value from a risk ratio with a `(ci_lower, ci_upper)` CI.
    pub fn from_risk_ratio(
        rr: f64,
        ci_lower: f64,
        ci_upper: f64,
        _cfg: &EValueConfig,
    ) -> CausalResult<EValueResult> {
        validate_positive_ratio(rr)?;
        validate_ratio_ci(ci_lower, ci_upper)?;
        Ok(EValueResult {
            e_value_point: e_value_for_ratio(rr),
            e_value_ci: e_value_for_ci_ratio(ci_lower, ci_upper),
            effect_type: EffectType::RiskRatio,
        })
    }

    /// E-value from an odds ratio with a `(ci_lower, ci_upper)` CI. When
    /// `cfg.rare_outcome` is `true` we treat OR ≈ RR; otherwise we
    /// convert via the VW & Ding §4 formula using `baseline_p`.
    pub fn from_odds_ratio(
        or_value: f64,
        ci_lower: f64,
        ci_upper: f64,
        baseline_p: f64,
        cfg: &EValueConfig,
    ) -> CausalResult<EValueResult> {
        validate_positive_ratio(or_value)?;
        validate_ratio_ci(ci_lower, ci_upper)?;
        if !cfg.rare_outcome {
            validate_baseline_p(baseline_p)?;
        }
        let rr_point = or_to_rr(or_value, baseline_p, cfg.rare_outcome);
        let rr_lo = or_to_rr(ci_lower, baseline_p, cfg.rare_outcome);
        let rr_hi = or_to_rr(ci_upper, baseline_p, cfg.rare_outcome);
        Ok(EValueResult {
            e_value_point: e_value_for_ratio(rr_point),
            e_value_ci: e_value_for_ci_ratio(rr_lo, rr_hi),
            effect_type: EffectType::OddsRatio,
        })
    }

    /// E-value from a hazard ratio with a `(ci_lower, ci_upper)` CI. The
    /// VanderWeele & Ding (2017) §S.2 conversion is used when the
    /// outcome is not rare; the threshold is `baseline_p > 0.15`.
    pub fn from_hazard_ratio(
        hr: f64,
        ci_lower: f64,
        ci_upper: f64,
        baseline_p: f64,
    ) -> CausalResult<EValueResult> {
        validate_positive_ratio(hr)?;
        validate_ratio_ci(ci_lower, ci_upper)?;
        validate_baseline_p(baseline_p)?;
        let rare = baseline_p <= 0.15;
        let rr_point = hr_to_rr(hr, rare);
        let rr_lo = hr_to_rr(ci_lower, rare);
        let rr_hi = hr_to_rr(ci_upper, rare);
        Ok(EValueResult {
            e_value_point: e_value_for_ratio(rr_point),
            e_value_ci: e_value_for_ci_ratio(rr_lo, rr_hi),
            effect_type: EffectType::HazardRatio,
        })
    }

    /// E-value from a risk difference, with both the point estimate and
    /// CI expressed on the absolute (probability-difference) scale.
    pub fn from_risk_difference(
        rd: f64,
        ci_lower: f64,
        ci_upper: f64,
        baseline_p: f64,
    ) -> CausalResult<EValueResult> {
        if !rd.is_finite() || !ci_lower.is_finite() || !ci_upper.is_finite() {
            return Err(CausalError::IncompatibleData);
        }
        if ci_lower > ci_upper {
            return Err(CausalError::IncompatibleData);
        }
        validate_baseline_p(baseline_p)?;
        let rr_point = rd_to_rr(rd, baseline_p)?;
        let rr_lo = rd_to_rr(ci_lower, baseline_p)?;
        let rr_hi = rd_to_rr(ci_upper, baseline_p)?;
        Ok(EValueResult {
            e_value_point: e_value_for_ratio(rr_point),
            e_value_ci: e_value_for_ci_ratio(rr_lo, rr_hi),
            effect_type: EffectType::RiskDifference,
        })
    }
}

// Validation ---------------------------------------------------------------

fn validate_positive_ratio(value: f64) -> CausalResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CausalError::IncompatibleData);
    }
    Ok(())
}

fn validate_ratio_ci(lo: f64, hi: f64) -> CausalResult<()> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(CausalError::IncompatibleData);
    }
    if lo <= 0.0 || hi <= 0.0 {
        return Err(CausalError::IncompatibleData);
    }
    if lo > hi {
        return Err(CausalError::IncompatibleData);
    }
    Ok(())
}

fn validate_baseline_p(p: f64) -> CausalResult<()> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(CausalError::IncompatibleData);
    }
    Ok(())
}

// Core E-value formula -----------------------------------------------------

/// E-value for a single risk-ratio number. The function is defined on
/// `(0, ∞)`: for `rr < 1` we apply the formula to `1 / rr`.
fn e_value_for_ratio(rr: f64) -> f64 {
    if !rr.is_finite() || rr <= 0.0 {
        return 1.0;
    }
    if (rr - 1.0).abs() < f64::EPSILON {
        return 1.0;
    }
    let r = if rr >= 1.0 { rr } else { 1.0 / rr };
    r + (r * (r - 1.0)).sqrt()
}

/// E-value of the CI bound closer to the null. Returns `1.0` if the CI
/// crosses 1 (the interval includes the null hypothesis).
fn e_value_for_ci_ratio(lo: f64, hi: f64) -> f64 {
    if !lo.is_finite() || !hi.is_finite() {
        return 1.0;
    }
    if lo <= 1.0 && hi >= 1.0 {
        return 1.0;
    }
    // CI entirely on one side of 1; choose the bound closer to 1.
    let closer = if lo > 1.0 { lo } else { hi };
    e_value_for_ratio(closer)
}

// Effect-type conversions --------------------------------------------------

/// OR → RR conversion. For `rare_outcome = true` (or `baseline_p = 0`)
/// we treat OR ≈ RR. Otherwise we use the VW & Ding (2017) §4 formula
/// `RR_approx = OR / (1 − p₀ + p₀·OR)`.
fn or_to_rr(or_val: f64, baseline_p: f64, rare_outcome: bool) -> f64 {
    if rare_outcome {
        return or_val;
    }
    let denom = 1.0 - baseline_p + baseline_p * or_val;
    if denom <= 0.0 || !denom.is_finite() {
        return or_val;
    }
    or_val / denom
}

/// HR → RR conversion. For rare outcomes (`baseline_p ≤ 0.15`) HR ≈ RR.
/// Otherwise VW & Ding (2017) §S.2:
/// `RR_approx = (1 − 0.5^√HR) / (1 − 0.5^√(1/HR))`.
fn hr_to_rr(hr: f64, rare: bool) -> f64 {
    if rare {
        return hr;
    }
    if (hr - 1.0).abs() < f64::EPSILON {
        return 1.0;
    }
    let sqrt_hr = hr.sqrt();
    let sqrt_inv = (1.0 / hr).sqrt();
    let half: f64 = 0.5;
    let num = 1.0 - half.powf(sqrt_hr);
    let den = 1.0 - half.powf(sqrt_inv);
    if !den.is_finite() || den.abs() < 1e-12 {
        return hr;
    }
    num / den
}

/// RD → RR conversion: `RR = (p₀ + RD) / p₀`. Errors when the implied
/// treated probability would be non-positive.
fn rd_to_rr(rd: f64, baseline_p: f64) -> CausalResult<f64> {
    let treated = baseline_p + rd;
    if treated <= 0.0 || !treated.is_finite() {
        return Err(CausalError::IncompatibleData);
    }
    Ok(treated / baseline_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn rr_equals_one_yields_e_one() {
        let r = EValue::from_risk_ratio(1.0, 0.5, 2.0, &EValueConfig::default()).unwrap();
        assert!(approx(r.e_value_point, 1.0, 1e-12));
        // CI crosses null.
        assert!(approx(r.e_value_ci, 1.0, 1e-12));
        assert_eq!(r.effect_type, EffectType::RiskRatio);
    }

    #[test]
    fn rr_two_yields_e_two_plus_sqrt_two() {
        let expected = 2.0 + 2.0_f64.sqrt();
        let r = EValue::from_risk_ratio(2.0, 1.5, 3.0, &EValueConfig::default()).unwrap();
        assert!(approx(r.e_value_point, expected, 1e-9));
    }

    #[test]
    fn rr_half_yields_same_as_rr_two() {
        let expected = 2.0 + 2.0_f64.sqrt();
        let r = EValue::from_risk_ratio(0.5, 0.3, 0.8, &EValueConfig::default()).unwrap();
        // 1/0.5 = 2, so same numeric E-value.
        assert!(approx(r.e_value_point, expected, 1e-9));
    }

    #[test]
    fn rr_zero_or_negative_errors() {
        let cfg = EValueConfig::default();
        assert!(EValue::from_risk_ratio(0.0, 0.5, 1.5, &cfg).is_err());
        assert!(EValue::from_risk_ratio(-0.1, 0.5, 1.5, &cfg).is_err());
    }

    #[test]
    fn ci_crossing_one_yields_one_for_ci() {
        let r = EValue::from_risk_ratio(2.0, 0.5, 5.0, &EValueConfig::default()).unwrap();
        assert!(approx(r.e_value_ci, 1.0, 1e-12));
    }

    #[test]
    fn ci_fully_above_one() {
        let r = EValue::from_risk_ratio(2.0, 1.5, 3.0, &EValueConfig::default()).unwrap();
        let expected = 1.5_f64 + (1.5_f64 * 0.5).sqrt();
        assert!(approx(r.e_value_ci, expected, 1e-9));
    }

    #[test]
    fn ci_fully_below_one() {
        let r = EValue::from_risk_ratio(0.5, 0.3, 0.8, &EValueConfig::default()).unwrap();
        let flipped: f64 = 1.0 / 0.8;
        let expected = flipped + (flipped * (flipped - 1.0)).sqrt();
        assert!(approx(r.e_value_ci, expected, 1e-9));
    }

    #[test]
    fn ci_with_lower_greater_than_upper_errors() {
        let cfg = EValueConfig::default();
        assert!(EValue::from_risk_ratio(2.0, 3.0, 1.5, &cfg).is_err());
    }

    #[test]
    fn or_rare_equivalent_to_rr() {
        let cfg_rare = EValueConfig { rare_outcome: true };
        let or_r = EValue::from_odds_ratio(2.0, 1.5, 3.0, 0.01, &cfg_rare).unwrap();
        let rr_r = EValue::from_risk_ratio(2.0, 1.5, 3.0, &EValueConfig::default()).unwrap();
        assert!(approx(or_r.e_value_point, rr_r.e_value_point, 1e-9));
        assert_eq!(or_r.effect_type, EffectType::OddsRatio);
    }

    #[test]
    fn or_non_rare_differs_from_rr() {
        let cfg_dense = EValueConfig {
            rare_outcome: false,
        };
        let or_r = EValue::from_odds_ratio(3.0, 2.0, 4.0, 0.5, &cfg_dense).unwrap();
        let rr_r = EValue::from_risk_ratio(3.0, 2.0, 4.0, &EValueConfig::default()).unwrap();
        // Non-rare conversion attenuates the effective RR away from OR,
        // hence the E-value should be smaller than for the equivalent RR.
        assert!(or_r.e_value_point < rr_r.e_value_point);
    }

    #[test]
    fn or_negative_or_errors() {
        let cfg = EValueConfig::default();
        assert!(EValue::from_odds_ratio(-1.0, 0.5, 1.5, 0.3, &cfg).is_err());
        assert!(EValue::from_odds_ratio(2.0, 1.5, 3.0, -0.1, &cfg).is_err());
        assert!(EValue::from_odds_ratio(2.0, 1.5, 3.0, 1.0, &cfg).is_err());
    }

    #[test]
    fn hr_rare_outcome_close_to_rr() {
        let hr_r = EValue::from_hazard_ratio(2.0, 1.5, 3.0, 0.01).unwrap();
        let rr_r = EValue::from_risk_ratio(2.0, 1.5, 3.0, &EValueConfig::default()).unwrap();
        assert!(approx(hr_r.e_value_point, rr_r.e_value_point, 1e-9));
        assert_eq!(hr_r.effect_type, EffectType::HazardRatio);
    }

    #[test]
    fn hr_non_rare_differs() {
        let hr_r = EValue::from_hazard_ratio(2.0, 1.5, 3.0, 0.5).unwrap();
        let rr_r = EValue::from_risk_ratio(2.0, 1.5, 3.0, &EValueConfig::default()).unwrap();
        assert!((hr_r.e_value_point - rr_r.e_value_point).abs() > 1e-3);
    }

    #[test]
    fn rd_baseline_zero_errors() {
        assert!(EValue::from_risk_difference(0.1, 0.05, 0.15, 0.0).is_err());
    }

    #[test]
    fn rd_baseline_too_large_errors() {
        assert!(EValue::from_risk_difference(0.1, 0.05, 0.15, 1.0).is_err());
    }

    #[test]
    fn rd_positive_with_baseline_0_1() {
        let r = EValue::from_risk_difference(0.1, 0.05, 0.15, 0.1).unwrap();
        // RR = (0.1 + 0.1) / 0.1 = 2.0
        let expected = 2.0 + 2.0_f64.sqrt();
        assert!(approx(r.e_value_point, expected, 1e-9));
        assert_eq!(r.effect_type, EffectType::RiskDifference);
    }

    #[test]
    fn rd_destroys_treated_prob() {
        // baseline + rd < 0
        assert!(EValue::from_risk_difference(-0.2, -0.5, -0.1, 0.1).is_err());
    }

    #[test]
    fn e_value_monotone_in_rr_distance_from_one() {
        let cfg = EValueConfig::default();
        let small = EValue::from_risk_ratio(1.5, 1.2, 1.8, &cfg)
            .unwrap()
            .e_value_point;
        let mid = EValue::from_risk_ratio(2.0, 1.5, 2.5, &cfg)
            .unwrap()
            .e_value_point;
        let large = EValue::from_risk_ratio(4.0, 3.0, 5.0, &cfg)
            .unwrap()
            .e_value_point;
        assert!(small < mid && mid < large);
    }

    #[test]
    fn ci_lower_negative_errors() {
        assert!(EValue::from_risk_ratio(2.0, -0.1, 3.0, &EValueConfig::default()).is_err());
    }

    #[test]
    fn nan_inputs_error() {
        let cfg = EValueConfig::default();
        assert!(EValue::from_risk_ratio(f64::NAN, 0.5, 1.5, &cfg).is_err());
        assert!(EValue::from_risk_ratio(2.0, f64::NAN, 1.5, &cfg).is_err());
        assert!(EValue::from_risk_difference(f64::NAN, 0.0, 0.1, 0.1).is_err());
    }
}
