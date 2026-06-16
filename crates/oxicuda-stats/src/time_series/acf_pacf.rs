//! Partial autocorrelation (PACF) via the Durbin-Levinson recursion plus
//! Bartlett large-lag standard errors and confidence bands for the ACF/PACF.
//!
//! The sample autocorrelation function itself lives in [`crate::time_series::acf`];
//! this module complements it with the *partial* autocorrelation function and the
//! standard-error machinery used to draw the familiar ACF/PACF correlogram bands.
//!
//! # References
//! - Durbin, J. (1960). "The fitting of time-series models". *Rev. Int. Stat.
//!   Inst.* 28(3):233-244.
//! - Levinson, N. (1947). "The Wiener RMS error criterion in filter design and
//!   prediction". *J. Math. Phys.* 25(1):261-278.
//! - Bartlett, M.S. (1946). "On the theoretical specification and sampling
//!   properties of autocorrelated time-series". *J. R. Stat. Soc. Suppl.*
//!   8(1):27-41.
//! - Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (2008). *Time Series Analysis:
//!   Forecasting and Control*, 4th ed., §3.2.

use crate::error::{StatsError, StatsResult};
use crate::time_series::acf;

/// Result of a partial-autocorrelation computation.
#[derive(Debug, Clone)]
pub struct PacfResult {
    /// Partial autocorrelations `φ_kk` for lags `0, 1, …, max_lag`.
    ///
    /// Element `0` is fixed at `1.0` by convention (the lag-0 "partial"
    /// autocorrelation of a series with itself).
    pub pacf: Vec<f64>,
    /// Approximate standard error of each PACF estimate.
    ///
    /// Under the null hypothesis that the true process is AR(p), the PACF
    /// estimates at lags `> p` are approximately `N(0, 1/n)`, so every entry
    /// equals `1/√n` (Quenouille 1949). Element `0` is `0.0`.
    pub se: Vec<f64>,
}

/// Compute the sample partial autocorrelation function (PACF) by the
/// Durbin-Levinson recursion applied to the sample ACF.
///
/// The PACF at lag `k`, `φ_kk`, is the last coefficient of the order-`k`
/// autoregressive fit and measures the correlation between `x_t` and
/// `x_{t-k}` after removing the linear effect of the intermediate lags.
///
/// The recursion is
///
/// ```text
/// φ_kk = (ρ_k - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_{k-j}) / (1 - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_j)
/// φ_kj = φ_{k-1,j} - φ_kk · φ_{k-1,k-j},   j = 1, …, k-1.
/// ```
///
/// The returned vector has length `max_lag + 1` with `pacf[0] == 1.0`.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if fewer than two observations.
/// - [`StatsError::InvalidParameter`] if `max_lag == 0` or `max_lag >= n`.
/// - [`StatsError::NonFiniteValue`] if the series contains a non-finite value.
pub fn pacf(x: &[f64], max_lag: usize) -> StatsResult<PacfResult> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if max_lag == 0 {
        return Err(StatsError::InvalidParameter {
            name: "max_lag".to_string(),
            reason: "number of lags must be ≥ 1".to_string(),
        });
    }
    if max_lag >= n {
        return Err(StatsError::InvalidParameter {
            name: "max_lag".to_string(),
            reason: format!("max_lag={max_lag} must be < n={n}"),
        });
    }
    if let Some(i) = x.iter().position(|v| !v.is_finite()) {
        return Err(StatsError::NonFiniteValue(i));
    }

    let rho = acf(x, max_lag);

    // Durbin-Levinson. `phi` holds the order-k coefficients φ_{k,1..k}.
    let mut pacf = vec![0.0_f64; max_lag + 1];
    pacf[0] = 1.0;
    if max_lag >= 1 {
        pacf[1] = rho[1];
    }
    let mut phi = vec![0.0_f64; max_lag + 1];
    phi[1] = rho[1];

    for k in 2..=max_lag {
        // Numerator: ρ_k - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_{k-j}
        let mut numer = rho[k];
        let mut denom = 1.0_f64;
        for j in 1..k {
            numer -= phi[j] * rho[k - j];
            denom -= phi[j] * rho[j];
        }
        // Guard against a (near-)singular denominator from a degenerate ACF.
        let phi_kk = if denom.abs() < 1e-300 {
            0.0
        } else {
            numer / denom
        };
        pacf[k] = phi_kk;

        // Update the lower-order coefficients in place using a snapshot.
        let prev = phi.clone();
        for j in 1..k {
            phi[j] = prev[j] - phi_kk * prev[k - j];
        }
        phi[k] = phi_kk;
    }

    // Quenouille standard error: 1/√n at every lag ≥ 1.
    let se_val = 1.0 / (n as f64).sqrt();
    let mut se = vec![se_val; max_lag + 1];
    se[0] = 0.0;

    Ok(PacfResult { pacf, se })
}

/// Result of an ACF computation augmented with Bartlett standard errors.
#[derive(Debug, Clone)]
pub struct AcfSeResult {
    /// Sample autocorrelations for lags `0, 1, …, max_lag` (lag 0 is `1.0`).
    pub acf: Vec<f64>,
    /// Bartlett standard error of each ACF estimate.
    pub se: Vec<f64>,
}

/// Sample ACF together with Bartlett's large-lag standard errors.
///
/// Bartlett's formula approximates the variance of `ρ̂_k` assuming the process
/// is an MA(k-1) (i.e. all autocorrelations beyond lag `k-1` vanish):
///
/// ```text
/// Var(ρ̂_k) ≈ (1/n) · (1 + 2 Σ_{i=1}^{k-1} ρ̂_i²),   k ≥ 1.
/// ```
///
/// The lag-0 standard error is `0` (the lag-0 autocorrelation is identically 1).
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if fewer than two observations.
/// - [`StatsError::InvalidParameter`] if `max_lag == 0` or `max_lag >= n`.
/// - [`StatsError::NonFiniteValue`] if the series contains a non-finite value.
pub fn acf_bartlett(x: &[f64], max_lag: usize) -> StatsResult<AcfSeResult> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if max_lag == 0 {
        return Err(StatsError::InvalidParameter {
            name: "max_lag".to_string(),
            reason: "number of lags must be ≥ 1".to_string(),
        });
    }
    if max_lag >= n {
        return Err(StatsError::InvalidParameter {
            name: "max_lag".to_string(),
            reason: format!("max_lag={max_lag} must be < n={n}"),
        });
    }
    if let Some(i) = x.iter().position(|v| !v.is_finite()) {
        return Err(StatsError::NonFiniteValue(i));
    }

    let acf_vals = acf(x, max_lag);
    let n_f = n as f64;
    let mut se = vec![0.0_f64; max_lag + 1];
    // running Σ ρ_i² for i = 1..k-1
    let mut cumulative_sq = 0.0_f64;
    for (k, se_k) in se.iter_mut().enumerate().skip(1) {
        let variance = (1.0 + 2.0 * cumulative_sq) / n_f;
        *se_k = variance.max(0.0).sqrt();
        cumulative_sq += acf_vals[k] * acf_vals[k];
    }

    Ok(AcfSeResult { acf: acf_vals, se })
}

/// Two-sided confidence bounds for an ACF/PACF correlogram.
///
/// Given a slice of standard errors `se` and a normal critical value `z`
/// (e.g. `1.959963985` for 95 %), returns the symmetric band half-widths
/// `z · se_k` for each lag. Callers shade `±` this band to flag lags whose
/// estimate is "significant".
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if `z` is negative or non-finite.
pub fn correlogram_bounds(se: &[f64], z: f64) -> StatsResult<Vec<f64>> {
    if !z.is_finite() || z < 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "z".to_string(),
            reason: format!("critical value must be finite and ≥ 0, got {z}"),
        });
    }
    Ok(se.iter().map(|&s| z * s).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic white-noise stream in roughly `[-1, 1]` from a 64-bit LCG.
    struct WhiteNoise {
        state: u64,
    }
    impl WhiteNoise {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next(&mut self) -> f64 {
            // MMIX LCG; take the high bits and centre to (-1, 1).
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((self.state >> 33) as f64) / (1u64 << 31) as f64; // [0,1)
            2.0 * u - 1.0
        }
    }

    /// AR(1): `x_t = φ x_{t-1} + ε_t` with genuine white-noise innovations so the
    /// theoretical PACF cuts off after lag 1.
    fn ar1_series(phi: f64, n: usize) -> Vec<f64> {
        let mut rng = WhiteNoise::new(0xC0FFEE ^ ((phi * 1000.0) as u64));
        let mut x = vec![0.0_f64; n];
        let mut prev = 0.0_f64;
        // Burn-in to forget the zero initial condition.
        for _ in 0..100 {
            prev = phi * prev + rng.next();
        }
        for slot in x.iter_mut() {
            prev = phi * prev + rng.next();
            *slot = prev;
        }
        x
    }

    #[test]
    fn pacf_lag0_is_one() {
        let x = ar1_series(0.6, 200);
        let res = pacf(&x, 10).expect("pacf");
        assert_eq!(res.pacf.len(), 11);
        assert!((res.pacf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pacf_lag1_equals_acf_lag1() {
        // For any series, φ_11 == ρ_1.
        let x = ar1_series(0.5, 150);
        let rho = acf(&x, 5);
        let res = pacf(&x, 5).expect("pacf");
        assert!((res.pacf[1] - rho[1]).abs() < 1e-10);
    }

    #[test]
    fn pacf_ar1_cuts_off_after_lag1() {
        // A strong AR(1) has a large φ_11 and small higher-lag PACF.
        let x = ar1_series(0.8, 400);
        let res = pacf(&x, 8).expect("pacf");
        assert!(res.pacf[1].abs() > 0.4, "phi_11={}", res.pacf[1]);
        for k in 3..=8 {
            assert!(
                res.pacf[k].abs() < 0.3,
                "PACF at lag {k} should be small, got {}",
                res.pacf[k]
            );
        }
    }

    #[test]
    fn pacf_bounded_by_one() {
        let x = ar1_series(0.7, 300);
        let res = pacf(&x, 12).expect("pacf");
        for (k, &p) in res.pacf.iter().enumerate() {
            assert!(p.abs() <= 1.0 + 1e-9, "PACF[{k}]={p} out of [-1,1]");
        }
    }

    #[test]
    fn pacf_se_is_inverse_sqrt_n() {
        let n = 256;
        let x = ar1_series(0.4, n);
        let res = pacf(&x, 5).expect("pacf");
        let expected = 1.0 / (n as f64).sqrt();
        for k in 1..=5 {
            assert!((res.se[k] - expected).abs() < 1e-12);
        }
        assert_eq!(res.se[0], 0.0);
    }

    #[test]
    fn pacf_white_noise_small() {
        // Pure white noise: every PACF (lag ≥ 1) should be statistically small,
        // i.e. comfortably within a few standard errors of zero.
        let mut rng = WhiteNoise::new(20240613);
        let x: Vec<f64> = (0..500).map(|_| rng.next()).collect();
        let res = pacf(&x, 10).expect("pacf");
        let se = 1.0 / (500.0_f64).sqrt(); // ≈ 0.0447
        for k in 1..=10 {
            assert!(
                res.pacf[k].abs() < 5.0 * se,
                "white-noise PACF at lag {k} = {} exceeds 5·SE",
                res.pacf[k]
            );
        }
    }

    #[test]
    fn pacf_errors_on_short_series() {
        assert!(matches!(
            pacf(&[1.0], 1),
            Err(StatsError::InsufficientSampleSize { got: 1, need: 2 })
        ));
    }

    #[test]
    fn pacf_errors_on_zero_lag() {
        let x = ar1_series(0.5, 50);
        assert!(matches!(
            pacf(&x, 0),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn pacf_errors_on_too_many_lags() {
        let x = ar1_series(0.5, 10);
        assert!(matches!(
            pacf(&x, 10),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn pacf_errors_on_non_finite() {
        let mut x = ar1_series(0.5, 50);
        x[7] = f64::NAN;
        assert!(matches!(pacf(&x, 5), Err(StatsError::NonFiniteValue(7))));
    }

    #[test]
    fn acf_bartlett_lag0_se_zero() {
        let x = ar1_series(0.6, 200);
        let res = acf_bartlett(&x, 10).expect("acf");
        assert_eq!(res.se[0], 0.0);
        assert!((res.acf[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn acf_bartlett_se_nondecreasing() {
        // The cumulative Σρ² only grows, so the Bartlett SE is non-decreasing.
        let x = ar1_series(0.8, 300);
        let res = acf_bartlett(&x, 15).expect("acf");
        for k in 2..=15 {
            assert!(
                res.se[k] >= res.se[k - 1] - 1e-12,
                "Bartlett SE should be non-decreasing at lag {k}: {} < {}",
                res.se[k],
                res.se[k - 1]
            );
        }
    }

    #[test]
    fn acf_bartlett_lag1_se_matches_formula() {
        let n = 200;
        let x = ar1_series(0.5, n);
        let res = acf_bartlett(&x, 5).expect("acf");
        // Var(ρ̂_1) = 1/n (sum over i=1..0 is empty).
        let expected = (1.0 / n as f64).sqrt();
        assert!((res.se[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn acf_bartlett_errors_on_non_finite() {
        let mut x = ar1_series(0.5, 50);
        x[3] = f64::INFINITY;
        assert!(matches!(
            acf_bartlett(&x, 5),
            Err(StatsError::NonFiniteValue(3))
        ));
    }

    #[test]
    fn correlogram_bounds_scales_se() {
        let se = vec![0.0, 0.1, 0.2, 0.3];
        let bounds = correlogram_bounds(&se, 2.0).expect("bounds");
        assert_eq!(bounds, vec![0.0, 0.2, 0.4, 0.6]);
    }

    #[test]
    fn correlogram_bounds_rejects_bad_z() {
        let se = vec![0.1, 0.2];
        assert!(correlogram_bounds(&se, -1.0).is_err());
        assert!(correlogram_bounds(&se, f64::NAN).is_err());
    }

    #[test]
    fn pacf_and_bartlett_agree_on_acf() {
        // Both routines call the same `acf`; the PacfResult's implied lag-1
        // equals the Bartlett result's lag-1 ACF.
        let x = ar1_series(0.65, 180);
        let p = pacf(&x, 6).expect("pacf");
        let a = acf_bartlett(&x, 6).expect("acf");
        assert!((p.pacf[1] - a.acf[1]).abs() < 1e-12);
    }
}
