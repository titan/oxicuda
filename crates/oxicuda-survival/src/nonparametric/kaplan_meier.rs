//! Kaplan-Meier product-limit estimator.

use crate::data::{Dataset, RiskSet};
use crate::error::{SurvivalError, SurvivalResult};

/// Kaplan-Meier estimator output.
///
/// At each unique event time:
/// - `times[i]`           — t_i
/// - `survival[i]`        — Ŝ(t_i) = Π_{k<=i} (1 - d_k/n_k)
/// - `greenwood_var[i]`   — Greenwood's variance estimate of Ŝ(t_i)
/// - `at_risk[i]`         — n_i (number at risk just before t_i)
/// - `events[i]`          — d_i (number of events at t_i)
#[derive(Debug, Clone)]
pub struct KaplanMeier {
    pub times: Vec<f64>,
    pub survival: Vec<f64>,
    pub greenwood_var: Vec<f64>,
    pub at_risk: Vec<f64>,
    pub events: Vec<f64>,
}

impl KaplanMeier {
    /// Standard error √Var(Ŝ).
    #[must_use]
    pub fn standard_error(&self) -> Vec<f64> {
        self.greenwood_var
            .iter()
            .map(|v| v.max(0.0).sqrt())
            .collect()
    }

    /// Pointwise log-log transformed (1 − α) confidence interval.
    ///
    /// Uses the transformation `log(-log Ŝ)` which keeps CI within `[0, 1]`.
    /// Returns `(lower, upper)` pairs.
    pub fn confidence_interval(&self, alpha: f64) -> SurvivalResult<(Vec<f64>, Vec<f64>)> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(SurvivalError::InvalidParameter(format!(
                "alpha must be in (0,1): {alpha}"
            )));
        }
        let z = norm_inv_one_minus_half_alpha(alpha);
        let mut lo = Vec::with_capacity(self.times.len());
        let mut hi = Vec::with_capacity(self.times.len());
        for (s, v) in self.survival.iter().zip(self.greenwood_var.iter()) {
            if *s <= 0.0 || *s >= 1.0 || *v <= 0.0 {
                lo.push(*s);
                hi.push(*s);
                continue;
            }
            let ln_s = s.ln();
            // Var(log Ŝ) ≈ Var(Ŝ) / Ŝ²
            let var_logs = v / (s * s);
            let se_loglog = (var_logs / (ln_s * ln_s)).max(0.0).sqrt();
            let theta = (-ln_s).ln();
            let lo_log = theta - z * se_loglog;
            let hi_log = theta + z * se_loglog;
            lo.push((-(hi_log.exp())).exp());
            hi.push((-(lo_log.exp())).exp());
        }
        Ok((lo, hi))
    }
}

/// Inverse standard-normal CDF at `1 - α/2`, computed via Beasley-Springer-Moro approximation.
fn norm_inv_one_minus_half_alpha(alpha: f64) -> f64 {
    let p = 1.0 - alpha / 2.0;
    norm_inv(p)
}

/// Inverse standard normal CDF via Acklam's algorithm.
fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let q;
    let r;
    if p < p_low {
        q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        q = p - 0.5;
        r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Estimate the Kaplan-Meier curve and Greenwood variance from a dataset.
pub fn kaplan_meier_estimate(data: &Dataset) -> SurvivalResult<KaplanMeier> {
    let rs = RiskSet::from_dataset(data)?;
    let mut surv = Vec::with_capacity(rs.len());
    let mut var = Vec::with_capacity(rs.len());
    let mut s_cur = 1.0_f64;
    let mut var_log = 0.0_f64; // accumulator for Σ d_i/(n_i(n_i - d_i))
    for (_t, d, n) in rs.iter() {
        if n <= 0.0 {
            return Err(SurvivalError::NumericalInstability(
                "non-positive at-risk count".to_string(),
            ));
        }
        let factor = if d > 0.0 { 1.0 - d / n } else { 1.0 };
        s_cur *= factor.max(0.0);
        if d > 0.0 && (n - d) > 0.0 {
            var_log += d / (n * (n - d));
        }
        surv.push(s_cur);
        // Var(Ŝ) ≈ Ŝ² · var_log  (Greenwood)
        var.push(s_cur * s_cur * var_log);
    }
    Ok(KaplanMeier {
        times: rs.times,
        survival: surv,
        greenwood_var: var,
        at_risk: rs.at_risk,
        events: rs.deaths,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn km_no_censoring_steps() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let km = kaplan_meier_estimate(&d).expect("ok");
        assert!((km.survival[0] - 0.75).abs() < 1.0e-12);
        assert!((km.survival[1] - 0.5).abs() < 1.0e-12);
        assert!((km.survival[2] - 0.25).abs() < 1.0e-12);
        assert!((km.survival[3] - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn km_with_censoring() {
        // 5 subjects: t=1 event, t=2 censored, t=3 event, t=4 censored, t=5 event
        let d = Dataset::from_arrays(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[true, false, true, false, true],
        )
        .expect("ok");
        let km = kaplan_meier_estimate(&d).expect("ok");
        // S(1)=1-1/5=0.8; S(2)=0.8 (censored); S(3)=0.8*(1-1/3)=0.5333; S(5)=0.5333*(1-1/1)=0.0
        assert!((km.survival[0] - 0.8).abs() < 1.0e-12);
        assert!((km.survival[1] - 0.8).abs() < 1.0e-12);
        assert!((km.survival[2] - 0.8 * 2.0 / 3.0).abs() < 1.0e-12);
        assert!((km.survival[3] - 0.8 * 2.0 / 3.0).abs() < 1.0e-12);
        assert!(km.survival[4].abs() < 1.0e-12);
    }

    #[test]
    fn greenwood_var_known_formula() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true, true, true, true]).expect("ok");
        let km = kaplan_meier_estimate(&d).expect("ok");
        // sum d/(n(n-d)) at first step: 1/(4*3) = 1/12; var = 0.75² · 1/12
        let expected = 0.75 * 0.75 / 12.0;
        assert!((km.greenwood_var[0] - expected).abs() < 1.0e-12);
    }

    #[test]
    fn km_confidence_interval_ok() {
        let d = Dataset::from_arrays(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[true, false, true, false, true],
        )
        .expect("ok");
        let km = kaplan_meier_estimate(&d).expect("ok");
        let (lo, hi) = km.confidence_interval(0.05).expect("ok");
        assert_eq!(lo.len(), km.survival.len());
        for (l, h) in lo.iter().zip(hi.iter()) {
            assert!(l <= h);
        }
    }

    #[test]
    fn km_rejects_bad_alpha() {
        let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
        let km = kaplan_meier_estimate(&d).expect("ok");
        assert!(km.confidence_interval(1.5).is_err());
    }

    #[test]
    fn norm_inv_known() {
        // Φ^-1(0.975) ≈ 1.96
        assert!((norm_inv(0.975) - 1.96).abs() < 1.0e-3);
    }
}
