//! Group-sequential testing with alpha-spending functions.
//!
//! Implements the Lan-DeMets (1983) / O'Brien-Fleming (1979) / Pocock (1977)
//! alpha-spending framework for interim analyses in clinical trials and sequential
//! experiments.
//!
//! # References
//! - O'Brien, P. C. & Fleming, T. R. (1979). A multiple testing procedure for
//!   clinical trials. *Biometrics*, 35, 549–556.
//! - Pocock, S. J. (1977). Group sequential methods in the design and analysis
//!   of clinical trials. *Biometrika*, 64(2), 191–199.
//! - Lan, K. K. G. & DeMets, D. L. (1983). Discrete sequential boundaries for
//!   clinical trials. *Biometrika*, 70(3), 659–663.

use crate::error::{StatsError, StatsResult};

// ─── normal distribution helpers ─────────────────────────────────────────────

/// Standard-normal CDF via the error function: Φ(x) = (1 + erf(x/√2)) / 2.
#[inline]
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Standard-normal quantile (inverse CDF) using Acklam's rational approximation
/// followed by Newton-Raphson refinement.
///
/// Peter Acklam's algorithm (max error ~1.15e-9 before refinement):
/// <https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/>
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);

    // Coefficients from Acklam
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x0 = if p < P_LOW {
        // Lower tail approximation
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail: by symmetry
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // Newton-Raphson refinement
    refine_normal_quantile(x0, p)
}

fn refine_normal_quantile(x0: f64, p: f64) -> f64 {
    const SQRT_2PI_INV: f64 = 0.398_942_280_401_432_7; // 1/√(2π)
    let mut x = x0;
    for _ in 0..5 {
        let fx = normal_cdf(x) - p;
        let fpx = SQRT_2PI_INV * (-0.5 * x * x).exp();
        if fpx.abs() < 1e-300 {
            break;
        }
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < 1e-15 {
            break;
        }
    }
    x
}

/// Abramowitz-Stegun 7.1.26 erf approximation (max error ~1.5e-7).
#[inline]
fn erf_approx(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

// ─── public types ─────────────────────────────────────────────────────────────

/// Specifies how cumulative Type-I error is allocated across interim analyses.
#[derive(Debug, Clone)]
pub enum SpendingFunction {
    /// O'Brien-Fleming (1979): conservative early, relaxed late.
    ///
    /// α*(t) = 2 (1 − Φ(z_{1−α/2} / √t))
    OBrienFleming,
    /// Pocock (1977): approximately equal boundaries at all interims.
    ///
    /// α*(t) = α · ln(1 + (e − 1) · t)
    Pocock,
    /// User-supplied vector of cumulative alpha values at equally spaced
    /// information fractions.  Must be monotone non-decreasing with `values[last] ≤ α`.
    Custom(Vec<f64>),
}

/// Configuration for group-sequential testing.
#[derive(Debug, Clone)]
pub struct AlphaSpendingConfig {
    /// Total one-sided or two-sided alpha (the code uses two-sided internally).
    pub alpha: f64,
    /// Number of interim analyses (including the final analysis).
    pub n_interim: usize,
    /// Alpha-spending function.
    pub spending_fn: SpendingFunction,
}

/// Outcome of a group-sequential test.
#[derive(Debug, Clone)]
pub struct SeqResult {
    /// Index (0-based) of the interim at which the trial was stopped, if any.
    pub stop_at: Option<usize>,
    /// Whether the null hypothesis was rejected.
    pub rejected: bool,
    /// Approximate p-value at the stopping interim (or final analysis).
    pub p_value: f64,
    /// Critical z-values at each interim analysis.
    pub critical_values: Vec<f64>,
}

// ─── alpha-spending functions ─────────────────────────────────────────────────

/// Cumulative alpha spent at information fraction `t ∈ [0, 1]`.
///
/// The returned value is in `[0, alpha]`.
pub fn alpha_spending_at(t: f64, cfg: &AlphaSpendingConfig) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let alpha = cfg.alpha;
    match &cfg.spending_fn {
        SpendingFunction::OBrienFleming => {
            if t <= 0.0 {
                return 0.0;
            }
            let z_half = normal_quantile(1.0 - alpha / 2.0);
            let arg = z_half / t.sqrt();
            let spent = 2.0 * (1.0 - normal_cdf(arg));
            spent.min(alpha)
        }
        SpendingFunction::Pocock => {
            let spent = alpha * (1.0 + (std::f64::consts::E - 1.0) * t).ln();
            spent.max(0.0).min(alpha)
        }
        SpendingFunction::Custom(values) => {
            if values.is_empty() {
                return 0.0;
            }
            let m = values.len();
            // Linear interpolation between the supplied knots
            let idx_f = t * (m - 1) as f64;
            let lo = idx_f.floor() as usize;
            let hi = (lo + 1).min(m - 1);
            let frac = idx_f - lo as f64;
            values[lo] + frac * (values[hi] - values[lo])
        }
    }
}

// ─── boundary computation ─────────────────────────────────────────────────────

/// Compute the critical z-values at each interim analysis using alpha-spending.
///
/// At analysis k (1-based), the information fraction is t_k = k / n_interim.
/// The incremental alpha is α_k = α*(t_k) − α*(t_{k−1}).
/// The two-sided critical value is z_k = Φ^{−1}(1 − α_k / 2).
///
/// Returns a `Vec<f64>` of length `n_interim`.
pub fn compute_boundaries(cfg: &AlphaSpendingConfig) -> StatsResult<Vec<f64>> {
    if cfg.n_interim == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_interim".into(),
            reason: "must be ≥ 1".into(),
        });
    }
    if cfg.alpha <= 0.0 || cfg.alpha >= 1.0 {
        return Err(StatsError::InvalidParameter {
            name: "alpha".into(),
            reason: format!("must be in (0, 1), got {}", cfg.alpha),
        });
    }
    if let SpendingFunction::Custom(vals) = &cfg.spending_fn {
        if vals.len() < cfg.n_interim {
            return Err(StatsError::InvalidParameter {
                name: "spending_fn".into(),
                reason: format!(
                    "Custom spending vector length {} < n_interim {}",
                    vals.len(),
                    cfg.n_interim
                ),
            });
        }
    }

    let mut boundaries = Vec::with_capacity(cfg.n_interim);
    let mut prev_spent = 0.0_f64;

    for k in 1..=cfg.n_interim {
        let t_k = k as f64 / cfg.n_interim as f64;
        let cum_spent = alpha_spending_at(t_k, cfg);
        let incr_alpha = (cum_spent - prev_spent).max(1e-15);
        prev_spent = cum_spent;
        // Two-sided critical value
        let z_k = normal_quantile(1.0 - incr_alpha / 2.0);
        boundaries.push(z_k);
    }

    Ok(boundaries)
}

// ─── group-sequential test ────────────────────────────────────────────────────

/// Run a group-sequential test given observed z-statistics at each interim.
///
/// Walks through the interims in order; if `|z_k| > boundary_k`, the test
/// stops and rejects H₀.  If no boundary is crossed, the test proceeds to the
/// final analysis.
///
/// The p-value is approximated as `2 * (1 − Φ(|z_stop|))` (unadjusted) at
/// the stopping interim; the caller should interpret this in the context of the
/// sequential design (i.e. boundary-crossing constitutes rejection at level α).
pub fn group_sequential_test(
    z_statistics: &[f64],
    cfg: &AlphaSpendingConfig,
) -> StatsResult<SeqResult> {
    if z_statistics.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let boundaries = compute_boundaries(cfg)?;
    let n = z_statistics.len().min(boundaries.len());

    let mut stop_at = None;
    let mut rejected = false;
    let mut stop_z = z_statistics[n - 1]; // default: final analysis

    for k in 0..n {
        let z = z_statistics[k];
        let boundary = boundaries[k];
        if z.abs() >= boundary {
            stop_at = Some(k);
            rejected = true;
            stop_z = z;
            break;
        }
    }

    // p-value: unadjusted two-sided at stopping interim
    let p_value = 2.0 * (1.0 - normal_cdf(stop_z.abs()));
    let p_value = p_value.clamp(0.0, 1.0);

    Ok(SeqResult {
        stop_at,
        rejected,
        p_value,
        critical_values: boundaries,
    })
}

// ─── sample size ─────────────────────────────────────────────────────────────

/// Compute the total sample size required for a two-sample sequential test.
///
/// Accounts for multiple-testing inflation due to interim looks.
///
/// The base fixed-sample n is:
/// n_base = (z_{1-α/2} + z_{1-β})² σ² / δ²  (per group × 2 for two-sample)
///
/// The inflation factors (approximate, from Pocock 1977 and O'Brien-Fleming
/// boundary properties) are:
/// - O'Brien-Fleming: multiply by 1 + 1.7 * ln(K) / K
/// - Pocock:         multiply by 1 + 0.83 * ln(K) / √K
/// - Custom:         multiply by Pocock's factor as a conservative default
///
/// Returns the *per-arm* sample size (total = 2 × returned value).
pub fn sample_size_sequential(
    delta: f64,
    sigma: f64,
    alpha: f64,
    power: f64,
    n_interim: usize,
    spending_fn: SpendingFunction,
) -> StatsResult<usize> {
    if delta <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "delta".into(),
            reason: format!("must be > 0, got {delta}"),
        });
    }
    if sigma <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "sigma".into(),
            reason: format!("must be > 0, got {sigma}"),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatsError::InvalidParameter {
            name: "alpha".into(),
            reason: format!("must be in (0,1), got {alpha}"),
        });
    }
    if power <= 0.0 || power >= 1.0 {
        return Err(StatsError::InvalidParameter {
            name: "power".into(),
            reason: format!("must be in (0,1), got {power}"),
        });
    }
    if n_interim == 0 {
        return Err(StatsError::InvalidParameter {
            name: "n_interim".into(),
            reason: "must be ≥ 1".into(),
        });
    }

    let z_alpha = normal_quantile(1.0 - alpha / 2.0);
    let z_beta = normal_quantile(power);
    let n_base = 2.0 * sigma * sigma * (z_alpha + z_beta).powi(2) / (delta * delta);

    let k = n_interim as f64;
    let inflation = match spending_fn {
        SpendingFunction::OBrienFleming => 1.0 + 1.7 * k.ln() / k,
        SpendingFunction::Pocock => 1.0 + 0.83 * k.ln() / k.sqrt(),
        SpendingFunction::Custom(_) => 1.0 + 0.83 * k.ln() / k.sqrt(), // conservative
    };

    let n_total = (n_base * inflation).ceil() as usize;
    // Return per-arm (each group needs n_total/2 subjects)
    let n_per_arm = n_total.div_ceil(2);
    Ok(n_per_arm)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obf_cfg(alpha: f64, n: usize) -> AlphaSpendingConfig {
        AlphaSpendingConfig {
            alpha,
            n_interim: n,
            spending_fn: SpendingFunction::OBrienFleming,
        }
    }

    fn pocock_cfg(alpha: f64, n: usize) -> AlphaSpendingConfig {
        AlphaSpendingConfig {
            alpha,
            n_interim: n,
            spending_fn: SpendingFunction::Pocock,
        }
    }

    // ── 1. OBF boundaries are monotone decreasing ────────────────────────────
    #[test]
    fn obf_boundaries_monotone_decreasing() {
        let cfg = obf_cfg(0.05, 5);
        let bnd = compute_boundaries(&cfg).expect("ok");
        for w in bnd.windows(2) {
            assert!(
                w[0] >= w[1],
                "OBF boundaries should decrease: {:.3} < {:.3}",
                w[0],
                w[1]
            );
        }
    }

    // ── 2. Pocock boundaries approximately constant ──────────────────────────
    #[test]
    fn pocock_boundaries_approximately_constant() {
        let cfg = pocock_cfg(0.05, 5);
        let bnd = compute_boundaries(&cfg).expect("ok");
        let mean_bnd = bnd.iter().sum::<f64>() / bnd.len() as f64;
        for &b in &bnd {
            assert!(
                (b - mean_bnd).abs() < 0.5,
                "Pocock boundary {b:.3} deviates from mean {mean_bnd:.3}"
            );
        }
    }

    // ── 3. alpha_spending_at(1.0) ≈ alpha ────────────────────────────────────
    #[test]
    fn alpha_spending_fully_spent_at_end_obf() {
        let cfg = obf_cfg(0.05, 4);
        let spent = alpha_spending_at(1.0, &cfg);
        assert!(
            (spent - 0.05).abs() < 1e-6,
            "OBF fully spent at t=1 should be ~0.05, got {spent}"
        );
    }

    #[test]
    fn alpha_spending_fully_spent_at_end_pocock() {
        let cfg = pocock_cfg(0.05, 4);
        let spent = alpha_spending_at(1.0, &cfg);
        assert!(
            (spent - 0.05).abs() < 1e-6,
            "Pocock fully spent at t=1 should be ~0.05, got {spent}"
        );
    }

    // ── 4. alpha_spending monotone in t ──────────────────────────────────────
    #[test]
    fn alpha_spending_monotone_in_t() {
        for cfg in [obf_cfg(0.05, 4), pocock_cfg(0.05, 4)] {
            let ts: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
            let spends: Vec<f64> = ts.iter().map(|&t| alpha_spending_at(t, &cfg)).collect();
            for w in spends.windows(2) {
                assert!(w[0] <= w[1] + 1e-12, "spending should be monotone");
            }
        }
    }

    // ── 5. Test with all |z| < boundaries → not rejected ────────────────────
    #[test]
    fn sequential_test_not_rejected_when_z_small() {
        let cfg = obf_cfg(0.05, 3);
        // Use z-statistics much smaller than the OBF boundary (~5 at interim 1)
        let zs = [0.5, 0.5, 0.5];
        let result = group_sequential_test(&zs, &cfg).expect("ok");
        assert!(!result.rejected, "should not be rejected");
        assert!(result.stop_at.is_none(), "should not stop early");
    }

    // ── 6. Test with early large z → stop early, rejected ───────────────────
    #[test]
    fn sequential_test_early_stop_when_z_large() {
        let cfg = obf_cfg(0.05, 4);
        // OBF boundary at first interim with 4 analyses is ~5.03
        // Use a z-stat of 6.0 to cross it
        let zs = [6.0, 0.0, 0.0, 0.0];
        let result = group_sequential_test(&zs, &cfg).expect("ok");
        assert!(result.rejected, "should be rejected");
        assert_eq!(result.stop_at, Some(0), "should stop at first interim");
    }

    // ── 7. sample_size_sequential > standard sample size ────────────────────
    #[test]
    fn sample_size_sequential_larger_than_fixed() {
        // Use 1 interim as the "fixed" (non-sequential) baseline: inflation factor = 1 + 1.7*0/1 = 1
        let n_fixed =
            sample_size_sequential(1.0, 1.0, 0.05, 0.8, 1, SpendingFunction::OBrienFleming)
                .expect("ok");
        // Sequential with 5 interims adds inflation
        let n_seq = sample_size_sequential(1.0, 1.0, 0.05, 0.8, 5, SpendingFunction::OBrienFleming)
            .expect("ok");
        assert!(
            n_seq >= n_fixed,
            "sequential n_per_arm={n_seq} should be ≥ fixed n={n_fixed}"
        );
    }

    // ── 8. Critical values have correct length ───────────────────────────────
    #[test]
    fn boundaries_correct_length() {
        for n in [1, 3, 5, 10] {
            let cfg = obf_cfg(0.05, n);
            let bnd = compute_boundaries(&cfg).expect("ok");
            assert_eq!(bnd.len(), n, "boundaries length mismatch for n={n}");
        }
    }

    // ── 9. Rejected at final analysis (no early stop) ────────────────────────
    #[test]
    fn sequential_test_rejected_at_final() {
        let cfg = pocock_cfg(0.05, 3);
        // Boundaries ≈ 2.39 for Pocock 3 interim. Use z=[1, 1, 3].
        let zs = [1.0, 1.0, 3.0];
        let result = group_sequential_test(&zs, &cfg).expect("ok");
        assert!(result.rejected, "should be rejected at final analysis");
    }

    // ── 10. alpha_spending_at(0) = 0 ─────────────────────────────────────────
    #[test]
    fn alpha_spending_zero_at_zero() {
        let cfg = obf_cfg(0.05, 4);
        let spent = alpha_spending_at(0.0, &cfg);
        assert!(
            spent < 1e-10,
            "OBF spending at t=0 should be 0, got {spent}"
        );
    }

    // ── 11. p_value in [0, 1] ────────────────────────────────────────────────
    #[test]
    fn sequential_test_p_value_in_range() {
        let cfg = obf_cfg(0.05, 4);
        let zs = [1.5, 2.0, 2.5, 3.0];
        let result = group_sequential_test(&zs, &cfg).expect("ok");
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "p-value {} not in [0, 1]",
            result.p_value
        );
    }

    // ── 12. OBF vs Pocock sample sizes comparison ────────────────────────────
    #[test]
    fn sample_size_pocock_vs_obf() {
        let n_obf = sample_size_sequential(1.0, 1.0, 0.05, 0.8, 4, SpendingFunction::OBrienFleming)
            .expect("ok");
        let n_pocock =
            sample_size_sequential(1.0, 1.0, 0.05, 0.8, 4, SpendingFunction::Pocock).expect("ok");
        // Pocock generally requires slightly more than OBF but both should be reasonable
        assert!(n_obf > 0 && n_pocock > 0, "sample sizes should be positive");
    }

    // ── 13. Empty z-statistics returns error ─────────────────────────────────
    #[test]
    fn sequential_test_empty_error() {
        let cfg = obf_cfg(0.05, 3);
        let result = group_sequential_test(&[], &cfg);
        assert!(result.is_err(), "empty z-statistics should return error");
    }
}
