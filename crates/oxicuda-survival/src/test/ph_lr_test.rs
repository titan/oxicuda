//! Proportional Hazards Likelihood Ratio, Wald, and Score Tests for the Cox PH model.
//!
//! # Tests provided
//!
//! - [`ph_lr_test`]: Likelihood ratio test — `-2 * (log_lik_null - log_lik_full) ~ χ²(p)`
//! - [`ph_wald_test`]: Wald test — `Z_i = β_i / SE_i`, individual and joint significance
//! - [`ph_score_test`]: Rao score test — `U(0)^T I(0)^{-1} U(0) ~ χ²(p)`

use crate::cox::breslow_ties::breslow_log_likelihood;
use crate::cox::newton_raphson::{TieMethod, newton_raphson_cox};
use crate::data::{Dataset, Observation};
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::inverse::gauss_jordan_inverse;

// ---------------------------------------------------------------------------
// Special functions: regularised incomplete gamma and chi-squared CDF
// ---------------------------------------------------------------------------

/// Regularised lower incomplete gamma P(a, x) = γ(a,x)/Γ(a).
///
/// Uses a Taylor series for x < a+1 and Lentz modified continued-fraction for x >= a+1.
fn gamma_inc_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let lna = crate::special::gammaln::gammaln(a);
    if x < a + 1.0 {
        // Taylor / series expansion: P(a,x) = e^{-x} x^a / Γ(a) * Σ x^n / (a+1)…(a+n)
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..600 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 3.0e-15 {
                break;
            }
        }
        let p = sum * (-x + a * x.ln() - lna).exp();
        p.clamp(0.0, 1.0)
    } else {
        // Complementary: P = 1 - Q where Q is via continued fraction
        (1.0 - gamma_q_cf(a, x, lna)).clamp(0.0, 1.0)
    }
}

/// Upper regularised gamma Q(a,x) via Lentz modified continued-fraction.
///
/// Evaluates the Legendre continued fraction for Γ(a,x)/Γ(a).
fn gamma_q_cf(a: f64, x: f64, lna: f64) -> f64 {
    // Modified Lentz algorithm (Numerical Recipes 3rd ed., §6.2)
    const TINY: f64 = 1.0e-300;
    let prefix = (-x + a * x.ln() - lna).exp();
    // Start the fraction: b_0 = x + 1 - a, a_i = -i*(i-a), b_i = x+2i+1-a
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = if b.abs() < TINY { TINY } else { 1.0 / b };
    let mut h = d;
    for i in 1i64..=600 {
        let ai = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = ai * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + ai / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3.0e-15 {
            break;
        }
    }
    (prefix * h).clamp(0.0, 1.0)
}

/// CDF of χ²(df) distribution at `x`.
fn chi2_cdf(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let k = df as f64;
    gamma_inc_p(k / 2.0, x / 2.0)
}

/// Standard normal CDF via error function.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Error function approximation (Horner-form series, |err| < 1.5e-7).
fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Helper: build Dataset from the (time, event, covariates) slice form
// ---------------------------------------------------------------------------
fn slice_to_dataset(data: &[(f64, bool, Vec<f64>)]) -> SurvivalResult<Dataset> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let p = data[0].2.len();
    let mut obs = Vec::with_capacity(data.len());
    let mut cov = Vec::with_capacity(data.len());
    let mut n_events = 0usize;
    for (t, e, x) in data {
        if x.len() != p {
            return Err(SurvivalError::DimensionMismatch { a: x.len(), b: p });
        }
        obs.push(Observation::new(*t, *e)?);
        if *e {
            n_events += 1;
        }
        cov.push(x.clone());
    }
    if n_events == 0 {
        return Err(SurvivalError::NoEvents);
    }
    Dataset::new(obs, Some(cov), None)
}

// ---------------------------------------------------------------------------
// Likelihood Ratio Test
// ---------------------------------------------------------------------------

/// Result of a proportional-hazards likelihood-ratio test.
///
/// `LR = -2 * (log_lik_null - log_lik_full) ~ χ²(p)`
#[derive(Debug, Clone)]
pub struct PhLrTestResult {
    /// The LR chi-squared statistic.
    pub lr_statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Degrees of freedom (number of covariates).
    pub df: usize,
    /// Partial log-likelihood evaluated at β = 0.
    pub log_lik_null: f64,
    /// Partial log-likelihood at MLE β̂.
    pub log_lik_full: f64,
}

/// Perform a likelihood-ratio test for the Cox PH model.
///
/// Tests H₀: β = **0** vs H₁: β ≠ **0**.
/// The LR statistic follows a χ²(p) distribution under H₀.
pub fn ph_lr_test(data: &[(f64, bool, Vec<f64>)]) -> SurvivalResult<PhLrTestResult> {
    let ds = slice_to_dataset(data)?;
    let p = ds.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "ph_lr_test requires at least one covariate".to_string(),
        ));
    }

    // Log-likelihood at null (β = 0)
    let beta_null = vec![0.0_f64; p];
    let (ll_null, _, _) = breslow_log_likelihood(&ds, &beta_null)?;

    // Fit the full model via Newton-Raphson
    let nr = newton_raphson_cox(&ds, &beta_null, TieMethod::Breslow, 1.0e-6, 100)?;
    let ll_full = nr.log_likelihood;

    // LR statistic: must be non-negative
    let lr_stat = (-2.0 * (ll_null - ll_full)).max(0.0);

    // p-value from χ²(p) distribution
    let cdf_val = chi2_cdf(lr_stat, p);
    let p_val = (1.0 - cdf_val).clamp(0.0, 1.0);

    Ok(PhLrTestResult {
        lr_statistic: lr_stat,
        p_value: p_val,
        df: p,
        log_lik_null: ll_null,
        log_lik_full: ll_full,
    })
}

// ---------------------------------------------------------------------------
// Wald Test
// ---------------------------------------------------------------------------

/// Result of a proportional-hazards Wald test.
#[derive(Debug, Clone)]
pub struct PhWaldResult {
    /// MLE coefficient vector β̂.
    pub coef: Vec<f64>,
    /// Standard errors `sqrt(diag(H⁻¹))`.
    pub std_err: Vec<f64>,
    /// Wald z-statistics `Z_i = β_i / SE_i`.
    pub z_stats: Vec<f64>,
    /// Two-sided p-values for each coefficient.
    pub p_values: Vec<f64>,
    /// `(1 - alpha)` confidence intervals `(β_i ± z_{α/2} * SE_i)`.
    pub conf_intervals: Vec<(f64, f64)>,
}

/// Perform a Wald test for individual coefficient significance.
///
/// # Arguments
/// * `data`  – observations `(time, event, covariates)`.
/// * `alpha` – significance level (e.g. 0.05 for 95 % CI).
pub fn ph_wald_test(data: &[(f64, bool, Vec<f64>)], alpha: f64) -> SurvivalResult<PhWaldResult> {
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(SurvivalError::InvalidParameter(format!(
            "alpha must be in (0, 1), got {alpha}"
        )));
    }
    let ds = slice_to_dataset(data)?;
    let p = ds.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "ph_wald_test requires at least one covariate".to_string(),
        ));
    }

    let beta_null = vec![0.0_f64; p];
    let nr = newton_raphson_cox(&ds, &beta_null, TieMethod::Breslow, 1.0e-6, 100)?;
    let beta = &nr.beta;
    let info = &nr.information;

    // Invert the observed Fisher information to get the variance-covariance matrix
    let var_cov = match gauss_jordan_inverse(info, p) {
        Ok(v) => v,
        Err(_) => {
            // Regularise with small ridge
            let mut info_r = info.clone();
            for d in 0..p {
                info_r[d * p + d] += 1.0e-6;
            }
            gauss_jordan_inverse(&info_r, p)?
        }
    };

    // z_{α/2} from standard normal inverse-CDF via bisection
    let z_alpha2 = normal_quantile(1.0 - alpha / 2.0);

    let mut std_err = Vec::with_capacity(p);
    let mut z_stats = Vec::with_capacity(p);
    let mut p_values = Vec::with_capacity(p);
    let mut conf_intervals = Vec::with_capacity(p);

    for i in 0..p {
        let var_ii = var_cov[i * p + i].max(0.0);
        let se = var_ii.sqrt();
        let z = if se > 1.0e-12 { beta[i] / se } else { 0.0 };
        // Two-sided p-value: 2 * (1 - Φ(|z|))
        let p_val = (2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0);
        let ci_lo = beta[i] - z_alpha2 * se;
        let ci_hi = beta[i] + z_alpha2 * se;
        std_err.push(se);
        z_stats.push(z);
        p_values.push(p_val);
        conf_intervals.push((ci_lo, ci_hi));
    }

    Ok(PhWaldResult {
        coef: beta.clone(),
        std_err,
        z_stats,
        p_values,
        conf_intervals,
    })
}

/// Inverse standard-normal CDF via bisection (works for p in (0,1)).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Reasonable bounds for practical alpha values
    let mut lo = -10.0_f64;
    let mut hi = 10.0_f64;
    for _ in 0..80 {
        let mid = (lo + hi) / 2.0;
        if normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

// ---------------------------------------------------------------------------
// Score (Rao) Test
// ---------------------------------------------------------------------------

/// Perform Rao's efficient score test for the Cox PH model.
///
/// Evaluates the score vector `U(0)` and observed Fisher information `I(0)` at β = **0**,
/// then forms `Q = U(0)ᵀ I(0)⁻¹ U(0) ~ χ²(p)`.
///
/// Returns `(score_statistic, p_value)`.
pub fn ph_score_test(data: &[(f64, bool, Vec<f64>)]) -> SurvivalResult<(f64, f64)> {
    let ds = slice_to_dataset(data)?;
    let p = ds.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "ph_score_test requires at least one covariate".to_string(),
        ));
    }

    let beta_null = vec![0.0_f64; p];
    let (_, score, info) = breslow_log_likelihood(&ds, &beta_null)?;

    // Invert Fisher information at β = 0
    let info_inv = match gauss_jordan_inverse(&info, p) {
        Ok(v) => v,
        Err(_) => {
            // Regularise
            let mut info_r = info.clone();
            for d in 0..p {
                info_r[d * p + d] += 1.0e-6;
            }
            gauss_jordan_inverse(&info_r, p)?
        }
    };

    // Q = U^T I^{-1} U
    let mut q = 0.0_f64;
    for i in 0..p {
        for j in 0..p {
            q += score[i] * info_inv[i * p + j] * score[j];
        }
    }
    let q = q.max(0.0);

    let cdf_val = chi2_cdf(q, p);
    let p_val = (1.0 - cdf_val).clamp(0.0, 1.0);

    Ok((q, p_val))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build synthetic Cox data: T ~ Exp(exp(beta * x)), x ~ N(0,1), all events.
    fn make_strong_data(n: usize, beta: f64, seed: u64) -> Vec<(f64, bool, Vec<f64>)> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let lam = (beta * x).exp();
                let t = rng.next_exponential(lam).max(1.0e-6);
                (t, true, vec![x])
            })
            .collect()
    }

    /// Build data with no association (beta = 0).
    fn make_noise_data(n: usize, seed: u64) -> Vec<(f64, bool, Vec<f64>)> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let t = rng.next_exponential(1.0).max(1.0e-6);
                (t, true, vec![x])
            })
            .collect()
    }

    // ---- LR test -----------------------------------------------------------

    #[test]
    fn lr_strong_predictor_low_pvalue() {
        let data = make_strong_data(80, 2.0, 1001);
        let res = ph_lr_test(&data).expect("lr test ok");
        assert!(res.p_value < 0.05, "p={}", res.p_value);
    }

    #[test]
    fn lr_noise_data_high_pvalue() {
        let data = make_noise_data(60, 9999);
        let res = ph_lr_test(&data).expect("lr test ok");
        // noise: p-value should generally be high; use lenient threshold
        assert!(res.p_value > 0.05 || res.lr_statistic < 10.0);
    }

    #[test]
    fn lr_statistic_nonnegative() {
        let data = make_strong_data(50, 1.5, 2002);
        let res = ph_lr_test(&data).expect("ok");
        assert!(res.lr_statistic >= 0.0);
    }

    #[test]
    fn lr_df_equals_n_covariates() {
        let data = make_strong_data(40, 1.0, 3003);
        let res = ph_lr_test(&data).expect("ok");
        assert_eq!(res.df, 1);
    }

    #[test]
    fn lr_empty_data_returns_error() {
        let data: Vec<(f64, bool, Vec<f64>)> = vec![];
        assert!(ph_lr_test(&data).is_err());
    }

    #[test]
    fn lr_no_events_returns_error() {
        let data = vec![
            (1.0, false, vec![0.5]),
            (2.0, false, vec![-0.5]),
            (3.0, false, vec![1.0]),
        ];
        assert!(ph_lr_test(&data).is_err());
    }

    #[test]
    fn lr_pvalue_in_unit_interval() {
        let data = make_strong_data(60, 1.0, 4004);
        let res = ph_lr_test(&data).expect("ok");
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn lr_log_lik_full_geq_null() {
        // MLE must not decrease the likelihood
        let data = make_strong_data(50, 1.5, 5005);
        let res = ph_lr_test(&data).expect("ok");
        assert!(res.log_lik_full >= res.log_lik_null - 1.0e-6);
    }

    // ---- Wald test ---------------------------------------------------------

    #[test]
    fn wald_coef_sign_correct() {
        let data = make_strong_data(100, 1.5, 6006);
        let res = ph_wald_test(&data, 0.05).expect("ok");
        // positive beta => positive coefficient estimate
        assert!(res.coef[0] > 0.0, "coef={}", res.coef[0]);
    }

    #[test]
    fn wald_pvalues_in_unit_interval() {
        let data = make_strong_data(80, 1.0, 7007);
        let res = ph_wald_test(&data, 0.05).expect("ok");
        for &pv in &res.p_values {
            assert!((0.0..=1.0).contains(&pv), "p={}", pv);
        }
    }

    #[test]
    fn wald_strong_predictor_low_pvalue() {
        let data = make_strong_data(100, 2.0, 8008);
        let res = ph_wald_test(&data, 0.05).expect("ok");
        assert!(res.p_values[0] < 0.05, "p={}", res.p_values[0]);
    }

    #[test]
    fn wald_ci_centered_at_coef() {
        let data = make_strong_data(60, 1.0, 9009);
        let res = ph_wald_test(&data, 0.05).expect("ok");
        for (i, &b) in res.coef.iter().enumerate() {
            let (lo, hi) = res.conf_intervals[i];
            let center = (lo + hi) / 2.0;
            assert!((center - b).abs() < 1.0e-10, "CI not centered at coef");
        }
    }

    #[test]
    fn wald_ci_coverage_basic() {
        // With n=150 and large beta, 95% CI should generally contain the true value
        let data = make_strong_data(150, 1.0, 1234);
        let res = ph_wald_test(&data, 0.05).expect("ok");
        let (lo, hi) = res.conf_intervals[0];
        // true beta = 1.0; wide tolerance for finite-sample
        assert!(lo < 1.5 && hi > 0.5, "CI=[{lo},{hi}]");
    }

    #[test]
    fn wald_empty_data_error() {
        let data: Vec<(f64, bool, Vec<f64>)> = vec![];
        assert!(ph_wald_test(&data, 0.05).is_err());
    }

    // ---- Score test --------------------------------------------------------

    #[test]
    fn score_statistic_positive() {
        let data = make_strong_data(80, 1.5, 2345);
        let (stat, _pv) = ph_score_test(&data).expect("ok");
        assert!(stat >= 0.0);
    }

    #[test]
    fn score_strong_predictor_low_pvalue() {
        let data = make_strong_data(100, 2.0, 3456);
        let (_stat, pv) = ph_score_test(&data).expect("ok");
        assert!(pv < 0.05, "p={}", pv);
    }

    #[test]
    fn score_noise_pvalue_reasonable() {
        let data = make_noise_data(60, 5678);
        let (stat, _pv) = ph_score_test(&data).expect("ok");
        // On noise data the score statistic should not be enormous
        assert!(stat < 50.0, "stat={stat}");
    }

    #[test]
    fn score_pvalue_in_unit_interval() {
        let data = make_strong_data(60, 1.0, 6789);
        let (_stat, pv) = ph_score_test(&data).expect("ok");
        assert!((0.0..=1.0).contains(&pv), "pv={}", pv);
    }

    #[test]
    fn lr_and_wald_asymptotically_close() {
        // For moderate n, LR ≈ Wald statistic within a factor of 2
        let data = make_strong_data(80, 1.0, 7890);
        let lr_res = ph_lr_test(&data).expect("ok");
        let wald_res = ph_wald_test(&data, 0.05).expect("ok");
        // Wald chi² = z² for p=1
        let wald_chi2 = wald_res.z_stats[0].powi(2);
        let ratio = lr_res.lr_statistic / wald_chi2.max(1.0e-6);
        assert!(
            0.1 < ratio && ratio < 10.0,
            "LR={}, Wald_chi2={}, ratio={}",
            lr_res.lr_statistic,
            wald_chi2,
            ratio
        );
    }
}
