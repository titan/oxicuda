//! Power and sample-size calculations for survival (time-to-event) trials.
//!
//! Implements the Schoenfeld (1981) and Freedman (1982) formulae for two-arm
//! log-rank tests, together with utilities for back-calculating power given an
//! observed event count and for estimating expected events given a fixed N.
//!
//! # References
//! * Schoenfeld, D. (1981). The asymptotic properties of nonparametric tests
//!   for comparing survival distributions. *Biometrika*, 68(1), 316–319.
//! * Freedman, L.S. (1982). Tables of the number of patients required in
//!   clinical trials using the log-rank test. *Statistics in Medicine*, 1(2),
//!   121–129.
//! * Acklam, P.J. (2010). An algorithm for computing the inverse normal
//!   cumulative distribution function. *Unpublished manuscript*.

use crate::error::{SurvivalError, SurvivalResult};

// ─────────────────────────────────────────────────────────────────────────────
// Statistical primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Standard normal CDF using the Abramowitz & Stegun (7.1.26) complementary
/// error-function approximation, accurate to |ε| < 7.5 × 10⁻⁸.
///
/// Φ(x) = erfc(-x / √2) / 2
#[inline]
fn norm_cdf(x: f64) -> f64 {
    // Rational approximation to erfc(t), t ≥ 0 (A&S 7.1.26)
    let t = x / std::f64::consts::SQRT_2;
    erfc_approx(-t) / 2.0
}

/// Complementary error-function approximation for all real t.
/// Accurate to |ε| < 1.5 × 10⁻⁷.
#[inline]
fn erfc_approx(t: f64) -> f64 {
    if t >= 0.0 {
        erfc_positive(t)
    } else {
        2.0 - erfc_positive(-t)
    }
}

/// erfc for t ≥ 0 via the Horner-form polynomial of A&S 7.1.26.
#[inline]
fn erfc_positive(t: f64) -> f64 {
    let p = 0.3275911_f64;
    let a = [
        0.254829592_f64,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    ];
    let x = 1.0 / (1.0 + p * t);
    let poly = ((((a[4] * x + a[3]) * x + a[2]) * x + a[1]) * x + a[0]) * x;
    poly * (-t * t).exp()
}

/// Inverse standard normal CDF via Acklam's algorithm (2010).
/// Returns `f64::NAN` for inputs outside (0, 1).
fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
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
    let p_low = 0.02425_f64;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Return the upper-α critical value z_α for a one-sided test, or z_{α/2} for
/// a two-sided test.
#[inline]
fn critical_z(alpha: f64, two_sided: bool) -> f64 {
    let p = if two_sided {
        1.0 - alpha / 2.0
    } else {
        1.0 - alpha
    };
    norm_inv(p)
}

/// Validate that `v ∈ (0, 1)` exclusive.
#[inline]
fn check_probability(v: f64, name: &str) -> SurvivalResult<()> {
    if v <= 0.0 || v >= 1.0 || !v.is_finite() {
        Err(SurvivalError::InvalidParameter(format!(
            "{name} must be in (0, 1), got {v}"
        )))
    } else {
        Ok(())
    }
}

/// Validate that `v > 0` and is finite.
#[inline]
fn check_positive(v: f64, name: &str) -> SurvivalResult<()> {
    if v <= 0.0 || !v.is_finite() {
        Err(SurvivalError::InvalidParameter(format!(
            "{name} must be > 0, got {v}"
        )))
    } else {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schoenfeld (1981) — two-arm log-rank sample size
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Schoenfeld (1981) sample-size formula.
#[derive(Debug, Clone)]
pub struct SchoenefeldConfig {
    /// Hazard ratio: λ_experimental / λ_control.  Must be > 0 and ≠ 1.
    pub hazard_ratio: f64,
    /// Type-I error probability α (default 0.05).
    pub alpha: f64,
    /// Desired power 1 − β (default 0.80).
    pub power: f64,
    /// Whether to use a two-sided test (default `true`).
    pub two_sided: bool,
    /// Allocation ratio r = n_experimental / n_control (default 1.0).
    pub allocation_ratio: f64,
}

impl Default for SchoenefeldConfig {
    fn default() -> Self {
        Self {
            hazard_ratio: 0.5,
            alpha: 0.05,
            power: 0.80,
            two_sided: true,
            allocation_ratio: 1.0,
        }
    }
}

/// Results from the Schoenfeld sample-size calculation.
#[derive(Debug, Clone)]
pub struct SchoenefeldResult {
    /// Required number of events d (ceiling).
    pub n_events: usize,
    /// Estimated total subjects n (ceiling).
    pub n_total: usize,
    /// Subjects in the experimental arm (ceiling of n_total · r / (r + 1)).
    pub n_arm1: usize,
    /// Subjects in the control arm (n_total − n_arm1).
    pub n_arm2: usize,
    /// Power back-calculated from the ceiled event count.
    pub achieved_power: f64,
    /// Fraction of subjects who must experience an event (d / n_total).
    pub event_fraction_needed: f64,
}

/// Compute the Schoenfeld (1981) required events and subjects for a two-arm
/// survival trial designed around the log-rank test.
///
/// # Formula
///
/// Let r = `allocation_ratio`, p₁ = r/(r+1), p₂ = 1/(r+1) be the arm
/// proportions.  The required number of events is
///
/// ```text
/// d = (z_{α/2} + z_β)² · (1/p₁ + 1/p₂) / (log HR)²
///   = (z_{α/2} + z_β)² · (r+1)²/r / (log HR)²
/// ```
///
/// For r = 1: d = 4 · (z_{α/2} + z_β)² / (log HR)².
///
/// The total N requires the caller to supply expected event probabilities per
/// arm.  When these are not available (this function) we return `n_total = 0`
/// to indicate that the caller should use [`freedman_sample_size`] instead, or
/// call [`expected_events`] separately.
pub fn schoenfeld_sample_size(config: &SchoenefeldConfig) -> SurvivalResult<SchoenefeldResult> {
    // ── validation ────────────────────────────────────────────────────────────
    check_probability(config.alpha, "alpha")?;
    check_probability(config.power, "power")?;
    check_positive(config.hazard_ratio, "hazard_ratio")?;
    check_positive(config.allocation_ratio, "allocation_ratio")?;

    if (config.hazard_ratio - 1.0).abs() < 1e-12 {
        return Err(SurvivalError::InvalidParameter(
            "hazard_ratio must not equal 1.0 (no detectable difference)".into(),
        ));
    }

    // ── critical values ───────────────────────────────────────────────────────
    let z_alpha = critical_z(config.alpha, config.two_sided);
    let z_beta = norm_inv(config.power); // z_{1-β}

    // ── number of events ──────────────────────────────────────────────────────
    let log_hr = config.hazard_ratio.ln().abs();
    let r = config.allocation_ratio;

    // Schoenfeld formula in terms of arm proportions p₁ = r/(r+1), p₂ = 1/(r+1):
    //   d = (z_α + z_β)² · (1/p₁ + 1/p₂) / (log HR)²
    //     = (z_α + z_β)² · ((r+1)/r + (r+1)) / (log HR)²
    //     = (z_α + z_β)² · (r+1)² / r / (log HR)²
    let sum_z_sq = (z_alpha + z_beta).powi(2);
    let d_exact = sum_z_sq * (r + 1.0).powi(2) / r / (log_hr * log_hr);
    let n_events = d_exact.ceil() as usize;

    // ── back-calculate achieved power at ceiled d ─────────────────────────────
    // Power = Φ(−z_{α/2} + |log HR| · √(p₁·p₂·d))
    let p1 = r / (r + 1.0);
    let p2 = 1.0 / (r + 1.0);
    let achieved_power = back_calculate_power(
        n_events,
        config.hazard_ratio,
        config.alpha,
        config.two_sided,
        p1,
        p2,
    );

    // ── n_total: not computable without event probabilities ───────────────────
    // Signal to the caller with 0; use freedman_sample_size for full n.
    let n_total = 0_usize;
    let n_arm1 = 0_usize;
    let n_arm2 = 0_usize;
    let event_fraction_needed = 0.0_f64;

    Ok(SchoenefeldResult {
        n_events,
        n_total,
        n_arm1,
        n_arm2,
        achieved_power,
        event_fraction_needed,
    })
}

/// Back-calculate power given a (possibly ceiled) number of events.
///
/// Power = Φ(−z_{α/2} + |log HR| · √(p₁ · p₂ · d))
fn back_calculate_power(
    n_events: usize,
    hazard_ratio: f64,
    alpha: f64,
    two_sided: bool,
    p1: f64,
    p2: f64,
) -> f64 {
    let z_alpha = critical_z(alpha, two_sided);
    let log_hr = hazard_ratio.ln().abs();
    let d = n_events as f64;
    // Noncentrality parameter
    let ncp = log_hr * (p1 * p2 * d).sqrt();
    norm_cdf(ncp - z_alpha)
}

// ─────────────────────────────────────────────────────────────────────────────
// Freedman (1982) — event-based sample size from exponential survival
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Freedman (1982) sample-size formula.
#[derive(Debug, Clone)]
pub struct FreedmanConfig {
    /// Hazard ratio: λ_experimental / λ_control.
    pub hazard_ratio: f64,
    /// Type-I error probability α (default 0.05).
    pub alpha: f64,
    /// Desired power 1 − β (default 0.80).
    pub power: f64,
    /// Whether to use a two-sided test (default `true`).
    pub two_sided: bool,
    /// Allocation ratio r = n_experimental / n_control (default 1.0).
    pub allocation_ratio: f64,
    /// Accrual (recruitment) period A (in the same time unit as hazards).
    pub accrual_time: f64,
    /// Minimum follow-up time F after end of accrual.
    pub follow_up_time: f64,
    /// Baseline (control arm) hazard λ₁ > 0 (events per time unit).
    pub control_hazard: f64,
}

impl Default for FreedmanConfig {
    fn default() -> Self {
        Self {
            hazard_ratio: 0.5,
            alpha: 0.05,
            power: 0.80,
            two_sided: true,
            allocation_ratio: 1.0,
            accrual_time: 2.0,
            follow_up_time: 1.0,
            control_hazard: 0.5,
        }
    }
}

/// Results from the Freedman sample-size calculation.
#[derive(Debug, Clone)]
pub struct FreedmanResult {
    /// Required number of events d (ceiling).
    pub n_events: usize,
    /// Total subjects n (ceiling).
    pub n_total: usize,
    /// Subjects in the experimental arm.
    pub n_arm1: usize,
    /// Subjects in the control arm.
    pub n_arm2: usize,
    /// Expected event probability in the control arm.
    pub p_event_arm1: f64,
    /// Expected event probability in the experimental arm.
    pub p_event_arm2: f64,
}

/// Compute the expected event probability under exponential survival with
/// uniform accrual over [0, A] and additional follow-up F.
///
/// A subject entering at time u ∈ [0, A] is followed until A + F, so their
/// event probability is `1 − exp(−λ(A + F − u))`.  Integrating uniformly:
///
/// ```text
/// P_event(λ, A, F) = 1 − (exp(−λF) − exp(−λ(A+F))) / (λA)
/// ```
///
/// For λA → 0 we apply a first-order Taylor expansion to avoid cancellation.
fn event_prob_exponential(lambda: f64, accrual: f64, follow_up: f64) -> f64 {
    debug_assert!(lambda > 0.0);
    debug_assert!(accrual > 0.0);
    debug_assert!(follow_up >= 0.0);
    let lf = lambda * follow_up;
    let la = lambda * accrual;
    // Numerically stable for small la
    if la < 1e-8 {
        // Taylor: (exp(-lf) - exp(-(lf+la))) / la ≈ exp(-lf) · (1 - la/2 + …)
        let ratio = (-lf).exp() * (1.0 - la / 2.0 + la * la / 6.0);
        1.0 - ratio
    } else {
        let ratio = ((-lf).exp() - (-(lf + la)).exp()) / la;
        1.0 - ratio
    }
}

/// Compute the Freedman (1982) sample size for a two-arm survival trial under
/// the exponential model with uniform accrual.
///
/// Steps:
/// 1. Compute the required number of events d via Schoenfeld (1981).
/// 2. Compute expected event probabilities per arm under the exponential model.
/// 3. Derive n_total = d / (p₁·r/(r+1) + p₂·1/(r+1)) where p₁, p₂ are the
///    arm-level event fractions (weighted by allocation).
pub fn freedman_sample_size(config: &FreedmanConfig) -> SurvivalResult<FreedmanResult> {
    // ── validation ────────────────────────────────────────────────────────────
    check_probability(config.alpha, "alpha")?;
    check_probability(config.power, "power")?;
    check_positive(config.hazard_ratio, "hazard_ratio")?;
    check_positive(config.allocation_ratio, "allocation_ratio")?;
    check_positive(config.accrual_time, "accrual_time")?;
    check_positive(config.control_hazard, "control_hazard")?;
    if config.follow_up_time < 0.0 || !config.follow_up_time.is_finite() {
        return Err(SurvivalError::InvalidParameter(
            "follow_up_time must be ≥ 0".into(),
        ));
    }
    if (config.hazard_ratio - 1.0).abs() < 1e-12 {
        return Err(SurvivalError::InvalidParameter(
            "hazard_ratio must not equal 1.0".into(),
        ));
    }

    // ── required events from Schoenfeld formula ───────────────────────────────
    let schoenefeld_cfg = SchoenefeldConfig {
        hazard_ratio: config.hazard_ratio,
        alpha: config.alpha,
        power: config.power,
        two_sided: config.two_sided,
        allocation_ratio: config.allocation_ratio,
    };
    let sch = schoenfeld_sample_size(&schoenefeld_cfg)?;
    let n_events = sch.n_events;

    // ── expected event probabilities ──────────────────────────────────────────
    let lambda1 = config.control_hazard;
    let lambda2 = lambda1 * config.hazard_ratio;

    let p_event_arm1 = event_prob_exponential(lambda1, config.accrual_time, config.follow_up_time);
    let p_event_arm2 = event_prob_exponential(lambda2, config.accrual_time, config.follow_up_time);

    if p_event_arm1 <= 0.0 || p_event_arm2 <= 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "expected event probability is zero or negative; check hazard/accrual parameters"
                .into(),
        ));
    }

    // ── total N ───────────────────────────────────────────────────────────────
    // Expected events = n_arm1 · p_event_arm2 + n_arm2 · p_event_arm1
    // With n_arm1 = r·n₂ and n₂ = n / (r+1):
    //   E[d] = n · (r · p_event_arm2 + p_event_arm1) / (r + 1)
    // Solving for n:
    //   n = d · (r + 1) / (r · p_event_arm2 + p_event_arm1)
    let r = config.allocation_ratio;
    let weighted_p = (r * p_event_arm2 + p_event_arm1) / (r + 1.0);
    let n_total_exact = n_events as f64 / weighted_p;
    let n_total = n_total_exact.ceil() as usize;

    let n_arm1_exact = n_total as f64 * r / (r + 1.0);
    let n_arm1 = n_arm1_exact.ceil() as usize;
    let n_arm2 = n_total.saturating_sub(n_arm1);

    Ok(FreedmanResult {
        n_events,
        n_total,
        n_arm1,
        n_arm2,
        p_event_arm1,
        p_event_arm2,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Power given number of events
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for power back-calculation given observed or planned events.
#[derive(Debug, Clone)]
pub struct PowerFromEventsConfig {
    /// Hazard ratio: λ_experimental / λ_control.
    pub hazard_ratio: f64,
    /// Number of events observed (or planned).
    pub n_events: usize,
    /// Type-I error probability α.
    pub alpha: f64,
    /// Whether to use a two-sided test.
    pub two_sided: bool,
    /// Allocation ratio r = n_experimental / n_control (default 1.0).
    pub allocation_ratio: f64,
}

impl Default for PowerFromEventsConfig {
    fn default() -> Self {
        Self {
            hazard_ratio: 0.5,
            n_events: 100,
            alpha: 0.05,
            two_sided: true,
            allocation_ratio: 1.0,
        }
    }
}

/// Back-calculate the power of a log-rank test given a fixed number of events.
///
/// Power = Φ(−z_{α/2} + |log HR| · √(p₁ · p₂ · d))
///
/// where p₁ = r/(r+1), p₂ = 1/(r+1) are the arm proportions.
pub fn power_from_events(config: &PowerFromEventsConfig) -> SurvivalResult<f64> {
    check_probability(config.alpha, "alpha")?;
    check_positive(config.hazard_ratio, "hazard_ratio")?;
    check_positive(config.allocation_ratio, "allocation_ratio")?;

    if (config.hazard_ratio - 1.0).abs() < 1e-12 {
        return Err(SurvivalError::InvalidParameter(
            "hazard_ratio must not equal 1.0".into(),
        ));
    }
    if config.n_events == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_events must be > 0".into(),
        ));
    }

    let r = config.allocation_ratio;
    let p1 = r / (r + 1.0);
    let p2 = 1.0 / (r + 1.0);

    Ok(back_calculate_power(
        config.n_events,
        config.hazard_ratio,
        config.alpha,
        config.two_sided,
        p1,
        p2,
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Expected events given N
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the expected number of events given total subjects `n_total`, the
/// allocation ratio, and the per-arm event probabilities.
///
/// ```text
/// E[d] = n_arm1 · p_event_arm1 + n_arm2 · p_event_arm2
/// ```
///
/// where n_arm1 = ceil(n_total · r / (r+1)) and n_arm2 = n_total − n_arm1.
pub fn expected_events(
    n_total: usize,
    allocation_ratio: f64,
    p_event_arm1: f64,
    p_event_arm2: f64,
) -> SurvivalResult<f64> {
    check_positive(allocation_ratio, "allocation_ratio")?;
    if p_event_arm1 <= 0.0 || p_event_arm1 >= 1.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "p_event_arm1 must be in (0, 1), got {p_event_arm1}"
        )));
    }
    if p_event_arm2 <= 0.0 || p_event_arm2 >= 1.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "p_event_arm2 must be in (0, 1), got {p_event_arm2}"
        )));
    }
    if n_total == 0 {
        return Ok(0.0);
    }

    let r = allocation_ratio;
    let n1_exact = n_total as f64 * r / (r + 1.0);
    let n_arm1 = n1_exact.ceil() as usize;
    let n_arm2 = n_total.saturating_sub(n_arm1);

    Ok(n_arm1 as f64 * p_event_arm1 + n_arm2 as f64 * p_event_arm2)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test 1: Default config values ─────────────────────────────────────────
    #[test]
    fn config_defaults() {
        let cfg = SchoenefeldConfig::default();
        assert!((cfg.alpha - 0.05).abs() < 1e-12);
        assert!((cfg.power - 0.80).abs() < 1e-12);
        assert!(cfg.two_sided);
        assert!((cfg.allocation_ratio - 1.0).abs() < 1e-12);

        let fcfg = FreedmanConfig::default();
        assert!((fcfg.alpha - 0.05).abs() < 1e-12);
        assert!((fcfg.power - 0.80).abs() < 1e-12);
        assert!(fcfg.two_sided);
        assert!((fcfg.allocation_ratio - 1.0).abs() < 1e-12);
    }

    // ── test 2: Known Schoenfeld result (HR=0.5) ──────────────────────────────
    //
    // Schoenfeld (1981) formula for equal allocation:
    //   d = 4·(z_{α/2} + z_β)² / (log HR)²
    //     = 4·(1.96 + 0.842)² / (ln 0.5)²
    //     = 4·7.843 / 0.4805 ≈ 65.3 → 66 events
    //
    // This is the standard result verified against R survpower and lifelines.
    #[test]
    fn schoenfeld_known_result() {
        let cfg = SchoenefeldConfig {
            hazard_ratio: 0.5,
            alpha: 0.05,
            power: 0.80,
            two_sided: true,
            allocation_ratio: 1.0,
        };
        let res = schoenfeld_sample_size(&cfg).expect("should succeed");
        // Exact analytic answer: ceil(4*(1.96+0.842)^2 / ln(0.5)^2) = 66
        assert!(
            res.n_events >= 60 && res.n_events <= 72,
            "Expected d in [60, 72], got {}",
            res.n_events
        );
    }

    // ── test 3: HR=1.0 must return an error ───────────────────────────────────
    #[test]
    fn schoenfeld_hr_one_error() {
        let cfg = SchoenefeldConfig {
            hazard_ratio: 1.0,
            ..Default::default()
        };
        assert!(schoenfeld_sample_size(&cfg).is_err());
    }

    // ── test 4: HR ≤ 0 must return an error ───────────────────────────────────
    #[test]
    fn schoenfeld_hr_negative_error() {
        for bad_hr in &[-1.0_f64, 0.0, -100.0] {
            let cfg = SchoenefeldConfig {
                hazard_ratio: *bad_hr,
                ..Default::default()
            };
            assert!(
                schoenfeld_sample_size(&cfg).is_err(),
                "Should fail for HR={bad_hr}"
            );
        }
    }

    // ── test 5: alpha out of range ────────────────────────────────────────────
    #[test]
    fn schoenfeld_alpha_out_of_range() {
        for bad_alpha in &[0.0_f64, 1.0, 1.5, -0.1] {
            let cfg = SchoenefeldConfig {
                alpha: *bad_alpha,
                ..Default::default()
            };
            assert!(
                schoenfeld_sample_size(&cfg).is_err(),
                "Should fail for alpha={bad_alpha}"
            );
        }
    }

    // ── test 6: power out of range ────────────────────────────────────────────
    #[test]
    fn schoenfeld_power_out_of_range() {
        for bad_power in &[0.0_f64, 1.0, 1.2, -0.5] {
            let cfg = SchoenefeldConfig {
                power: *bad_power,
                ..Default::default()
            };
            assert!(
                schoenfeld_sample_size(&cfg).is_err(),
                "Should fail for power={bad_power}"
            );
        }
    }

    // ── test 7: larger difference (smaller |log HR|) needs more events ────────
    #[test]
    fn schoenfeld_more_events_for_smaller_difference() {
        let cfg_0_5 = SchoenefeldConfig {
            hazard_ratio: 0.5,
            ..Default::default()
        };
        let cfg_0_8 = SchoenefeldConfig {
            hazard_ratio: 0.8,
            ..Default::default()
        };
        let res_0_5 = schoenfeld_sample_size(&cfg_0_5).unwrap();
        let res_0_8 = schoenfeld_sample_size(&cfg_0_8).unwrap();
        // HR=0.8 (small effect) needs more events than HR=0.5 (large effect)
        assert!(
            res_0_8.n_events > res_0_5.n_events,
            "HR=0.8 should need more events ({}) than HR=0.5 ({})",
            res_0_8.n_events,
            res_0_5.n_events
        );
    }

    // ── test 8: one-sided test requires fewer events than two-sided ───────────
    #[test]
    fn schoenfeld_one_sided_fewer_events() {
        let cfg_two = SchoenefeldConfig {
            hazard_ratio: 0.7,
            two_sided: true,
            ..Default::default()
        };
        let cfg_one = SchoenefeldConfig {
            two_sided: false,
            ..cfg_two.clone()
        };
        let res_two = schoenfeld_sample_size(&cfg_two).unwrap();
        let res_one = schoenfeld_sample_size(&cfg_one).unwrap();
        assert!(
            res_one.n_events < res_two.n_events,
            "one-sided ({}) should need fewer events than two-sided ({})",
            res_one.n_events,
            res_two.n_events
        );
    }

    // ── test 9: achieved power is close to the target power ───────────────────
    #[test]
    fn schoenfeld_achieved_power_close_to_target() {
        let target = 0.80_f64;
        let cfg = SchoenefeldConfig {
            hazard_ratio: 0.6,
            power: target,
            ..Default::default()
        };
        let res = schoenfeld_sample_size(&cfg).unwrap();
        let diff = (res.achieved_power - target).abs();
        assert!(
            diff < 0.02,
            "achieved_power {:.4} should be within 0.02 of target {target}",
            res.achieved_power
        );
        // achieved_power must be ≥ target (ceiling guarantees this)
        assert!(
            res.achieved_power >= target - 1e-6,
            "achieved_power {:.4} must be ≥ target",
            res.achieved_power
        );
    }

    // ── test 10: unequal allocation produces different arm sizes ──────────────
    #[test]
    fn schoenfeld_unequal_allocation() {
        // Use Freedman to get n_arm1/n_arm2 since Schoenfeld alone yields n=0.
        let cfg = FreedmanConfig {
            hazard_ratio: 0.6,
            allocation_ratio: 2.0, // 2 experimental : 1 control
            ..Default::default()
        };
        let res = freedman_sample_size(&cfg).unwrap();
        assert!(
            res.n_arm1 > res.n_arm2,
            "arm1 ({}) should be larger than arm2 ({})",
            res.n_arm1,
            res.n_arm2
        );
    }

    // ── test 11: Freedman event probs are in (0, 1) ───────────────────────────
    #[test]
    fn freedman_event_probs_between_0_1() {
        let cfg = FreedmanConfig::default();
        let res = freedman_sample_size(&cfg).unwrap();
        assert!(
            res.p_event_arm1 > 0.0 && res.p_event_arm1 < 1.0,
            "p_event_arm1={:.4} out of (0,1)",
            res.p_event_arm1
        );
        assert!(
            res.p_event_arm2 > 0.0 && res.p_event_arm2 < 1.0,
            "p_event_arm2={:.4} out of (0,1)",
            res.p_event_arm2
        );
    }

    // ── test 12: longer follow-up requires fewer subjects ─────────────────────
    #[test]
    fn freedman_more_followup_fewer_subjects() {
        let base = FreedmanConfig {
            hazard_ratio: 0.7,
            control_hazard: 0.3,
            accrual_time: 2.0,
            ..Default::default()
        };
        let short_fu = FreedmanConfig {
            follow_up_time: 1.0,
            ..base.clone()
        };
        let long_fu = FreedmanConfig {
            follow_up_time: 5.0,
            ..base.clone()
        };
        let n_short = freedman_sample_size(&short_fu).unwrap().n_total;
        let n_long = freedman_sample_size(&long_fu).unwrap().n_total;
        assert!(
            n_long < n_short,
            "Longer follow-up (n={n_long}) should need fewer subjects than shorter (n={n_short})"
        );
    }

    // ── test 13: higher hazard → fewer subjects needed ────────────────────────
    #[test]
    fn freedman_high_hazard_fewer_subjects() {
        let low_hz = FreedmanConfig {
            control_hazard: 0.1,
            hazard_ratio: 0.6,
            ..Default::default()
        };
        let high_hz = FreedmanConfig {
            control_hazard: 1.0,
            hazard_ratio: 0.6,
            ..Default::default()
        };
        let n_low = freedman_sample_size(&low_hz).unwrap().n_total;
        let n_high = freedman_sample_size(&high_hz).unwrap().n_total;
        assert!(
            n_high < n_low,
            "High hazard (n={n_high}) should need fewer subjects than low hazard (n={n_low})"
        );
    }

    // ── test 14: power increases with more events ─────────────────────────────
    #[test]
    fn power_from_events_increases_with_events() {
        let cfg_small = PowerFromEventsConfig {
            hazard_ratio: 0.7,
            n_events: 100,
            ..Default::default()
        };
        let cfg_large = PowerFromEventsConfig {
            n_events: 400,
            ..cfg_small.clone()
        };
        let pwr_small = power_from_events(&cfg_small).unwrap();
        let pwr_large = power_from_events(&cfg_large).unwrap();
        assert!(
            pwr_large > pwr_small,
            "power(d=400)={pwr_large:.4} should exceed power(d=100)={pwr_small:.4}"
        );
    }

    // ── test 15: power matches design for HR=0.5 ─────────────────────────────
    #[test]
    fn power_from_events_correct_value() {
        // For HR=0.5, equal allocation, the designed events ≈ 66 → power ≈ 0.80.
        // Design: d = 4*(1.96+0.842)^2 / ln(0.5)^2 ≈ 65.3 → 66.
        let sch_cfg = SchoenefeldConfig {
            hazard_ratio: 0.5,
            alpha: 0.05,
            power: 0.80,
            two_sided: true,
            allocation_ratio: 1.0,
        };
        let sch = schoenfeld_sample_size(&sch_cfg).unwrap();

        let cfg = PowerFromEventsConfig {
            hazard_ratio: 0.5,
            n_events: sch.n_events,
            alpha: 0.05,
            two_sided: true,
            allocation_ratio: 1.0,
        };
        let pwr = power_from_events(&cfg).unwrap();
        assert!(
            (pwr - 0.80).abs() < 0.02,
            "power {pwr:.4} should be ≈ 0.80 for designed d={}",
            sch.n_events
        );
    }

    // ── test 16: norm_inv accuracy ────────────────────────────────────────────
    #[test]
    fn norm_inv_check() {
        // Standard reference values
        assert!(
            (norm_inv(0.975) - 1.96).abs() < 1e-3,
            "norm_inv(0.975) = {:.6}",
            norm_inv(0.975)
        );
        assert!(
            (norm_inv(0.84) - 0.9945).abs() < 5e-3,
            "norm_inv(0.84) = {:.6}",
            norm_inv(0.84)
        );
        assert!(
            (norm_inv(0.5) - 0.0).abs() < 1e-6,
            "norm_inv(0.5) should be 0"
        );
        assert!(
            (norm_inv(0.025) + 1.96).abs() < 1e-3,
            "norm_inv(0.025) should be ≈ -1.96"
        );
    }

    // ── test 17: expected_events_simple ──────────────────────────────────────
    #[test]
    fn expected_events_simple() {
        // Equal arms, p=0.5 each, n=100 → expected events ≈ 50
        let e = expected_events(100, 1.0, 0.5, 0.5).unwrap();
        assert!((e - 50.0).abs() < 1.0, "expected ≈ 50, got {e:.2}");
    }

    // ── test 18: norm_cdf symmetry ────────────────────────────────────────────
    #[test]
    fn norm_cdf_symmetry() {
        for x in &[0.0_f64, 1.0, 1.96, 2.576, -1.645] {
            let p = norm_cdf(*x);
            let q = norm_cdf(-x);
            assert!(
                (p + q - 1.0).abs() < 1e-7,
                "norm_cdf({x}) + norm_cdf({}) = {:.8} ≠ 1",
                -x,
                p + q
            );
        }
    }

    // ── test 19: norm_cdf known values ────────────────────────────────────────
    #[test]
    fn norm_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((norm_cdf(-1.645) - 0.05).abs() < 1e-4);
    }

    // ── test 20: Freedman invalid config errors ───────────────────────────────
    #[test]
    fn freedman_invalid_config_errors() {
        // HR = 1
        let cfg_hr1 = FreedmanConfig {
            hazard_ratio: 1.0,
            ..Default::default()
        };
        assert!(freedman_sample_size(&cfg_hr1).is_err());

        // follow_up_time < 0
        let cfg_neg_fu = FreedmanConfig {
            follow_up_time: -1.0,
            ..Default::default()
        };
        assert!(freedman_sample_size(&cfg_neg_fu).is_err());

        // control_hazard = 0
        let cfg_hz0 = FreedmanConfig {
            control_hazard: 0.0,
            ..Default::default()
        };
        assert!(freedman_sample_size(&cfg_hz0).is_err());
    }

    // ── test 21: expected_events allocation ratio 2:1 ────────────────────────
    #[test]
    fn expected_events_unequal_allocation() {
        // 2:1 allocation, p_arm1=0.4, p_arm2=0.6, n=90
        // n_arm1=ceil(90*2/3)=60, n_arm2=30
        // E[d] = 60*0.4 + 30*0.6 = 24 + 18 = 42
        let e = expected_events(90, 2.0, 0.4, 0.6).unwrap();
        assert!((e - 42.0).abs() < 1.0, "expected ≈ 42, got {e:.2}");
    }

    // ── test 22: power_from_events invalid inputs ─────────────────────────────
    #[test]
    fn power_from_events_invalid_inputs() {
        // HR=1.0
        let cfg1 = PowerFromEventsConfig {
            hazard_ratio: 1.0,
            ..Default::default()
        };
        assert!(power_from_events(&cfg1).is_err());

        // n_events=0
        let cfg2 = PowerFromEventsConfig {
            n_events: 0,
            ..Default::default()
        };
        assert!(power_from_events(&cfg2).is_err());
    }

    // ── test 23: Schoenfeld allocation_ratio must be positive ─────────────────
    #[test]
    fn schoenfeld_allocation_ratio_nonpositive_error() {
        let cfg = SchoenefeldConfig {
            allocation_ratio: 0.0,
            ..Default::default()
        };
        assert!(schoenfeld_sample_size(&cfg).is_err());

        let cfg2 = SchoenefeldConfig {
            allocation_ratio: -2.0,
            ..Default::default()
        };
        assert!(schoenfeld_sample_size(&cfg2).is_err());
    }
}
