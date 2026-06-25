//! Time-series statistical tests for `oxicuda-stats`.
//!
//! Implements four classical time-series diagnostics:
//!
//! 1. **Sample ACF** — sample auto-correlation function at up to `max_lag` lags.
//! 2. **Ljung-Box Q-test** — tests H₀: white noise based on the portmanteau statistic.
//! 3. **Box-Pierce Q-test** — simpler variant of the Ljung-Box test.
//! 4. **Augmented Dickey-Fuller (ADF)** — unit-root test via OLS regression.
//! 5. **KPSS test** — stationarity test (H₀: stationary) complementary to ADF.
//! 6. **Durbin-Watson** — autocorrelation in OLS residuals.
//!
//! # References
//! - Ljung & Box (1978) "On a Measure of a Lack of Fit in Time Series Models".
//!   *Biometrika* 65(2):297-303.
//! - Dickey & Fuller (1979) "Distribution of the Estimators for Autoregressive Time
//!   Series with a Unit Root". *JASA* 74(366):427-431.
//! - MacKinnon (1994) "Approximate Asymptotic Distribution Functions for Unit-Root and
//!   Cointegration Tests". *JBES* 12(2):167-176.
//! - Kwiatkowski, Phillips, Schmidt & Shin (1992) "Testing the null hypothesis of
//!   stationarity against the alternative of a unit root". *JoE* 54(1-3):159-178.
//! - Durbin & Watson (1950) "Testing for Serial Correlation in Least-Squares
//!   Regression I". *Biometrika* 37(3-4):409-428.

pub mod acf_pacf;
pub mod arima;
pub mod garch;
pub mod var_model;
pub use acf_pacf::{AcfSeResult, PacfResult, acf_bartlett, correlogram_bounds, pacf};
pub use arima::{
    ArimaConfig, ArimaFit, arima_fit, arima_forecast, arima_predict_in_sample, arima_residuals,
};
pub use garch::{
    GarchConfig, GarchModel, garch_fit, garch_forecast, garch_log_likelihood, garch_persistence,
    garch_unconditional_variance,
};
pub use var_model::{
    GrangerResult, VarFit, granger_causality, var_fit, var_forecast, var_is_stable,
    var_spectral_radius, var_unconditional_mean,
};

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::ols;
use crate::special::betainc::gammp;

// ─────────────────────────────────────────────────────────────────────────────
// Chi-squared survival function (1 - CDF)
// ─────────────────────────────────────────────────────────────────────────────

/// Survival function P(χ²(df) > x) = 1 - gammp(df/2, x/2).
fn chi2_sf(x: f64, df: f64) -> StatsResult<f64> {
    if x <= 0.0 {
        return Ok(1.0);
    }
    let p = gammp(df / 2.0, x / 2.0)?;
    Ok((1.0 - p).clamp(0.0, 1.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Sample ACF
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the sample autocorrelation function (ACF) at lags 0, 1, …, `max_lag`.
///
/// Returns a vector of length `max_lag + 1`.  The first element (lag 0) is
/// always 1.0.
///
/// ACF(k) = Cov(x_t, x_{t-k}) / Var(x_t)
///         = [Σ_{t=k}^{n-1} (x_t - x̄)(x_{t-k} - x̄)] / [Σ_{t=0}^{n-1} (x_t - x̄)²]
pub fn acf(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance: f64 = x.iter().map(|&v| (v - mean) * (v - mean)).sum();
    if variance < 1e-300 {
        // Constant series: ACF undefined; return 1 at lag 0, 0 elsewhere
        let mut out = vec![0.0; max_lag + 1];
        out[0] = 1.0;
        return out;
    }
    let lag_limit = max_lag.min(n.saturating_sub(1));
    let mut result = vec![0.0; max_lag + 1];
    for k in 0..=lag_limit {
        let cov: f64 = (k..n).map(|t| (x[t] - mean) * (x[t - k] - mean)).sum();
        result[k] = cov / variance;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Ljung-Box Q-test
// ─────────────────────────────────────────────────────────────────────────────

/// Ljung-Box portmanteau test for white noise (H₀: first `m` autocorrelations are zero).
///
/// Statistic: Q_LB = n(n+2) Σ_{k=1}^{m} ρ̂(k)² / (n - k)
/// Under H₀: Q_LB ~ χ²(m).
///
/// # Returns
/// `(Q_statistic, p_value)`.
pub fn ljung_box(x: &[f64], m: usize) -> StatsResult<(f64, f64)> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if m == 0 {
        return Err(StatsError::InvalidParameter {
            name: "m".to_string(),
            reason: "number of lags must be ≥ 1".to_string(),
        });
    }
    if m >= n {
        return Err(StatsError::InvalidParameter {
            name: "m".to_string(),
            reason: format!("m={m} must be < n={n}"),
        });
    }
    let rho = acf(x, m);
    let n_f = n as f64;
    let q: f64 = (1..=m)
        .map(|k| rho[k] * rho[k] / (n_f - k as f64))
        .sum::<f64>()
        * n_f
        * (n_f + 2.0);
    let p = chi2_sf(q, m as f64)?;
    Ok((q, p))
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Box-Pierce Q-test
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Pierce portmanteau test for white noise.
///
/// Statistic: Q_BP = n Σ_{k=1}^{m} ρ̂(k)²
/// Under H₀: Q_BP ~ χ²(m) (asymptotically).
///
/// # Returns
/// `(Q_statistic, p_value)`.
pub fn box_pierce(x: &[f64], m: usize) -> StatsResult<(f64, f64)> {
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    if m == 0 {
        return Err(StatsError::InvalidParameter {
            name: "m".to_string(),
            reason: "number of lags must be ≥ 1".to_string(),
        });
    }
    if m >= n {
        return Err(StatsError::InvalidParameter {
            name: "m".to_string(),
            reason: format!("m={m} must be < n={n}"),
        });
    }
    let rho = acf(x, m);
    let n_f = n as f64;
    let q: f64 = n_f * (1..=m).map(|k| rho[k] * rho[k]).sum::<f64>();
    let p = chi2_sf(q, m as f64)?;
    Ok((q, p))
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Augmented Dickey-Fuller (ADF) unit-root test
// ─────────────────────────────────────────────────────────────────────────────

/// Specification of deterministic components in the ADF regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfTrend {
    /// No intercept, no trend: Δy_t = γ y_{t-1} + Σ δ_j Δy_{t-j} + ε_t
    None,
    /// Intercept only: Δy_t = α + γ y_{t-1} + Σ δ_j Δy_{t-j} + ε_t
    Constant,
    /// Intercept + linear trend: Δy_t = α + βt + γ y_{t-1} + Σ δ_j Δy_{t-j} + ε_t
    ConstantTrend,
}

/// Result of the Augmented Dickey-Fuller test.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF τ-statistic = γ̂ / SE(γ̂).
    pub statistic: f64,
    /// Approximate p-value (logistic approximation to DF distribution).
    pub p_value_approx: f64,
    /// Number of augmentation lags used.
    pub lags: usize,
    /// Effective number of observations in the ADF regression.
    pub n_obs: usize,
    /// MacKinnon-style critical values at [1%, 5%, 10%].
    pub critical_values: [f64; 3],
    /// True when the statistic is more negative than the 5%-level critical value
    /// (unit root rejected at 5%).
    pub reject_unit_root: bool,
}

/// MacKinnon (1994) response-surface critical values — simplified 3-point table.
///
/// Rows: [AdfTrend::None, AdfTrend::Constant, AdfTrend::ConstantTrend]
/// Cols: [1%, 5%, 10%]
const MACKINNON_CV: [[f64; 3]; 3] = [
    [-2.5658, -1.9393, -1.6156], // No constant
    [-3.4336, -2.8621, -2.5671], // Constant
    [-3.9638, -3.4126, -3.1279], // Constant + trend
];

/// Midpoint τ for each model's 5% critical value (used in logistic p approx).
const MACKINNON_5PCT: [f64; 3] = [-1.9393, -2.8621, -3.4126];

/// Approximate p-value using a logistic curve centred at the 5% critical value.
///
/// When τ → -∞ the unit root is strongly rejected → p → 0.
/// When τ → 0 it is not rejected → p → 1.
fn adf_p_approx(tau: f64, trend: AdfTrend) -> f64 {
    let idx = match trend {
        AdfTrend::None => 0,
        AdfTrend::Constant => 1,
        AdfTrend::ConstantTrend => 2,
    };
    let tau_5pct = MACKINNON_5PCT[idx];
    // Logistic: p = 1 / (1 + exp(-2*(τ - τ_5pct))) but reflected:
    // smaller τ (more negative) → smaller p
    let p = 1.0 / (1.0 + (-2.0 * (tau - tau_5pct)).exp());
    p.clamp(0.001, 0.999)
}

/// Augmented Dickey-Fuller unit-root test.
///
/// Fits the ADF regression:
/// `Δy_t = [α] [β·t] + γ·y_{t-1} + Σ_{j=1}^{p} δ_j·Δy_{t-j} + ε_t`
///
/// and returns the τ-statistic for γ = 0 (H₀: unit root).
///
/// # Parameters
/// - `x` — the time series (length ≥ 4).
/// - `max_lags` — maximum number of augmentation lags (Δy lagged differences).
///   Lag order is selected by minimising AIC over `0..=max_lags`.
/// - `trend` — deterministic components to include.
pub fn adf_test(x: &[f64], max_lags: usize, trend: AdfTrend) -> StatsResult<AdfResult> {
    let n = x.len();
    if n < 4 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 4 });
    }

    // --- Compute first differences Δy_t = y_t - y_{t-1}  (length n-1) ---
    let dy: Vec<f64> = (1..n).map(|t| x[t] - x[t - 1]).collect();
    // Length of dy = n-1

    // Select lag order by AIC over 0..=max_lags
    let best_lags = select_adf_lag(&dy, x, max_lags, trend)?;

    // Fit ADF regression with best_lags
    adf_fit(x, &dy, best_lags, trend)
}

/// AIC-based lag selection for ADF: iterate over p=0..=max_lags, pick best.
fn select_adf_lag(dy: &[f64], x: &[f64], max_lags: usize, trend: AdfTrend) -> StatsResult<usize> {
    let n = x.len();
    // Effective limit: we need at least 3 observations after removing lags
    let lag_limit = max_lags.min(n / 4);

    let mut best_aic = f64::INFINITY;
    let mut best_p = 0usize;

    for p in 0..=lag_limit {
        if let Ok(res) = adf_fit(x, dy, p, trend) {
            let t_eff = res.n_obs;
            if t_eff < 2 {
                continue;
            }
            // Compute sigma-squared from residuals via RSS / n_eff
            let model_cols = adf_n_cols(p, trend);
            let rss = compute_adf_rss(x, dy, p, trend);
            if rss <= 0.0 {
                continue;
            }
            let sigma2 = rss / t_eff as f64;
            // AIC = n * ln(sigma2) + 2 * (p + extra_cols)
            let aic = t_eff as f64 * sigma2.ln() + 2.0 * model_cols as f64;
            let _ = res; // drop borrow
            if aic < best_aic {
                best_aic = aic;
                best_p = p;
            }
        }
    }
    Ok(best_p)
}

/// Number of columns in ADF design matrix for lag count `p` and trend specification.
fn adf_n_cols(p: usize, trend: AdfTrend) -> usize {
    // Intercept + trend indicator + y_{t-1} + p lagged differences
    let det = match trend {
        AdfTrend::None => 0,
        AdfTrend::Constant => 1,
        AdfTrend::ConstantTrend => 2,
    };
    det + 1 + p // (deterministic) + y_{t-1} + Δy_{t-1..p}
}

/// Compute ADF residual sum of squares (for AIC selection) without full OLS bookkeeping.
fn compute_adf_rss(x: &[f64], dy: &[f64], p: usize, trend: AdfTrend) -> f64 {
    match adf_fit(x, dy, p, trend) {
        Ok(res) => {
            // Re-derive RSS from the statistic and SE — or use a simpler approach:
            // We just need a proxy; use residuals if available.
            // Since AdfResult doesn't store residuals, run a thin OLS to get RSS.
            let _ = res;
            compute_adf_rss_inner(x, dy, p, trend).unwrap_or(f64::INFINITY)
        }
        Err(_) => f64::INFINITY,
    }
}

fn compute_adf_rss_inner(x: &[f64], dy: &[f64], p: usize, trend: AdfTrend) -> StatsResult<f64> {
    let (design, response) = build_adf_design(x, dy, p, trend);
    let n_eff = response.len();
    let n_cols = adf_n_cols(p, trend);
    if n_eff == 0 || n_cols == 0 || n_eff < n_cols {
        return Ok(f64::INFINITY);
    }
    let lm = ols(&design, &response, n_eff, n_cols)?;
    Ok(lm.residual_sum_squares)
}

/// Build (design_matrix, response_vector) for the ADF regression.
///
/// ADF regression: Δy_t = [α] [β·t] γ·y_{t-1} + Σ_{j=1}^{p} δ_j·Δy_{t-j}
///
/// Effective sample: t = p+1, …, n-1  (indices in `dy` from `p` onwards).
/// `dy[t] = x[t+1] - x[t]` (0-indexed).
fn build_adf_design(x: &[f64], dy: &[f64], p: usize, trend: AdfTrend) -> (Vec<f64>, Vec<f64>) {
    // dy has length n-1 (where n = x.len()).
    // Effective rows: t = p, …, len(dy)-1  (i.e., len(dy)-p rows)
    let n = x.len();
    let n_dy = dy.len(); // = n - 1
    let n_eff = n_dy.saturating_sub(p);
    let n_cols = adf_n_cols(p, trend);

    let mut design = vec![0.0; n_eff * n_cols];
    let mut response = vec![0.0; n_eff];

    // t_dy: index into dy (= t in the ADF notation, where dy[t] = Δy_{t+1})
    // We start at t_dy = p so that lagged differences dy[t_dy-1..t_dy-p] exist.
    for (row, t_dy) in (p..n_dy).enumerate() {
        // Response: Δy_t = dy[t_dy]
        response[row] = dy[t_dy];

        let mut col = 0usize;

        // Deterministic terms
        match trend {
            AdfTrend::None => {}
            AdfTrend::Constant => {
                design[row * n_cols + col] = 1.0;
                col += 1;
            }
            AdfTrend::ConstantTrend => {
                design[row * n_cols + col] = 1.0;
                col += 1;
                // trend = original time index (t_dy + 1 in 1-based notation)
                design[row * n_cols + col] = (t_dy + 1) as f64;
                col += 1;
            }
        }

        // y_{t-1}: x[t_dy] (since dy[t_dy] = x[t_dy+1]-x[t_dy], y_{t} = x[t_dy+1])
        // In Δy_t notation, y_{t-1} = x[t_dy]  (the level before the difference)
        design[row * n_cols + col] = x[t_dy];
        col += 1;

        // Lagged differences: Δy_{t-j} = dy[t_dy - j] for j=1..p
        for j in 1..=p {
            design[row * n_cols + col] = dy[t_dy - j];
            col += 1;
        }
        debug_assert_eq!(col, n_cols);
        let _ = n; // used via x.len()
    }

    (design, response)
}

/// Fit ADF regression for given lag count and return AdfResult.
fn adf_fit(x: &[f64], dy: &[f64], p: usize, trend: AdfTrend) -> StatsResult<AdfResult> {
    let (design, response) = build_adf_design(x, dy, p, trend);
    let n_eff = response.len();
    let n_cols = adf_n_cols(p, trend);

    if n_eff == 0 || n_eff < n_cols + 1 {
        return Err(StatsError::InsufficientSampleSize {
            got: n_eff,
            need: n_cols + 1,
        });
    }

    let lm = ols(&design, &response, n_eff, n_cols)?;

    // The column index for γ (coefficient on y_{t-1}) depends on trend
    let gamma_col = match trend {
        AdfTrend::None => 0,
        AdfTrend::Constant => 1,
        AdfTrend::ConstantTrend => 2,
    };

    let gamma_hat = lm.coefficients[gamma_col];

    // Variance-covariance of β̂ = σ² (X^T X)^{-1}
    // σ² = RSS / (n_eff - n_cols)
    let dof = n_eff - n_cols;
    if dof == 0 {
        return Err(StatsError::DegreesOfFreedomZero);
    }
    let sigma2 = lm.residual_sum_squares / dof as f64;

    // SE(γ̂) = sqrt(σ² * [(X^T X)^{-1}]_{gamma_col, gamma_col})
    let var_gamma = sigma2 * lm.xtx_inv[gamma_col * n_cols + gamma_col];
    if var_gamma <= 0.0 {
        return Err(StatsError::NumericalInstability(
            "non-positive variance for γ̂ in ADF".to_string(),
        ));
    }
    let se_gamma = var_gamma.sqrt();
    let tau = gamma_hat / se_gamma;

    let idx = match trend {
        AdfTrend::None => 0,
        AdfTrend::Constant => 1,
        AdfTrend::ConstantTrend => 2,
    };
    let critical_values = MACKINNON_CV[idx];
    let p_value = adf_p_approx(tau, trend);
    let reject = tau < critical_values[1]; // reject at 5%

    Ok(AdfResult {
        statistic: tau,
        p_value_approx: p_value,
        lags: p,
        n_obs: n_eff,
        critical_values,
        reject_unit_root: reject,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. KPSS test
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic component for the KPSS test regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssTrend {
    /// Null hypothesis: stationary around a constant (level stationarity).
    Level,
    /// Null hypothesis: stationary around a linear trend.
    Trend,
}

/// KPSS stationarity test.
///
/// H₀: the series is (trend-/level-)stationary.
/// H₁: the series has a unit root.
///
/// Test statistic: η = (n⁻² Σ_t S_t²) / λ̂²
/// where S_t = Σ_{s=1}^{t} ê_s and λ̂² is the Newey-West long-run variance.
///
/// Asymptotic 5% critical values:
/// - Level: η > 0.463 → reject H₀.
/// - Trend: η > 0.146 → reject H₀.
///
/// # Returns
/// `(eta_statistic, p_approx)`
///
/// The p-value approximation is based on linear interpolation of the asymptotic
/// critical-value table at 1%, 5%, 10%.
pub fn kpss_test(x: &[f64], trend: KpssTrend) -> StatsResult<(f64, f64)> {
    let n = x.len();
    if n < 4 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 4 });
    }

    // --- Step 1: Obtain residuals from deterministic regression ---
    let residuals = match trend {
        KpssTrend::Level => {
            // Demean
            let mean = x.iter().sum::<f64>() / n as f64;
            x.iter().map(|&v| v - mean).collect::<Vec<_>>()
        }
        KpssTrend::Trend => {
            // Detrend: regress x on [1, t]
            let mut design = vec![0.0; n * 2];
            for i in 0..n {
                design[i * 2] = 1.0;
                design[i * 2 + 1] = (i + 1) as f64;
            }
            let lm = ols(&design, x, n, 2)?;
            lm.residuals
        }
    };

    // --- Step 2: Partial sums S_t = Σ_{s=0}^{t} ê_s  (cumulative sum) ---
    let mut s = vec![0.0; n];
    let mut acc = 0.0;
    for (i, &e) in residuals.iter().enumerate() {
        acc += e;
        s[i] = acc;
    }

    // --- Step 3: Newey-West long-run variance estimator ---
    // Bandwidth: m = floor(4 * (n/100)^{1/4})
    let bandwidth = (4.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize;
    let bandwidth = bandwidth.max(1);

    // Sample variance (lag-0 term)
    let var0: f64 = residuals.iter().map(|&e| e * e).sum::<f64>() / n as f64;

    // Bartlett-weighted cross-correlations for lags 1..=bandwidth
    let mut long_run_var = var0;
    for lag in 1..=bandwidth {
        let w = 1.0 - lag as f64 / (bandwidth as f64 + 1.0); // Bartlett kernel
        let cov: f64 = (lag..n)
            .map(|t| residuals[t] * residuals[t - lag])
            .sum::<f64>()
            / n as f64;
        long_run_var += 2.0 * w * cov;
    }
    // Guard against non-positive long-run variance
    let long_run_var = long_run_var.max(1e-300);

    // --- Step 4: KPSS statistic ---
    let sum_s2: f64 = s.iter().map(|&si| si * si).sum();
    let eta = sum_s2 / (n as f64 * n as f64 * long_run_var);

    // --- Step 5: Approximate p-value ---
    // Critical values and approximate p via interpolation
    // Level: [10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739]
    // Trend: [10%=0.119, 5%=0.146, 2.5%=0.176, 1%=0.216]
    let cv_table: &[(f64, f64)] = match trend {
        KpssTrend::Level => &[(0.347, 0.10), (0.463, 0.05), (0.574, 0.025), (0.739, 0.01)],
        KpssTrend::Trend => &[(0.119, 0.10), (0.146, 0.05), (0.176, 0.025), (0.216, 0.01)],
    };

    let p_approx = kpss_p_interp(eta, cv_table);

    Ok((eta, p_approx))
}

/// Linearly interpolate p-value from KPSS critical value table.
///
/// `table` is sorted ascending by statistic value, with decreasing p-values.
/// Returns p ∈ (0, 1).
fn kpss_p_interp(eta: f64, table: &[(f64, f64)]) -> f64 {
    // Below first critical value → p > table[0].1
    if eta <= table[0].0 {
        return (table[0].1 + 0.10).min(1.0);
    }
    // Above last critical value → p < table[last].1
    let last = table.len() - 1;
    if eta >= table[last].0 {
        return (table[last].1 * 0.5).max(0.001);
    }
    // Interpolate in the table
    for i in 0..last {
        let (cv_lo, p_hi) = table[i];
        let (cv_hi, p_lo) = table[i + 1];
        if eta >= cv_lo && eta <= cv_hi {
            let frac = (eta - cv_lo) / (cv_hi - cv_lo);
            return p_hi + frac * (p_lo - p_hi);
        }
    }
    0.05 // fallback
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Durbin-Watson test
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Durbin-Watson statistic for a sequence of OLS residuals.
///
/// DW = Σ_{t=2}^{n} (ê_t - ê_{t-1})² / Σ_{t=1}^{n} ê_t²
///
/// Interpretation:
/// - DW ≈ 2  → no autocorrelation
/// - DW < 2  → positive autocorrelation
/// - DW > 2  → negative autocorrelation
pub fn durbin_watson(residuals: &[f64]) -> StatsResult<f64> {
    let n = residuals.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let ss_diff: f64 = (1..n)
        .map(|t| {
            let d = residuals[t] - residuals[t - 1];
            d * d
        })
        .sum();
    let ss_res: f64 = residuals.iter().map(|&e| e * e).sum();
    if ss_res < 1e-300 {
        return Err(StatsError::NumericalInstability(
            "sum of squared residuals is near zero".to_string(),
        ));
    }
    Ok(ss_diff / ss_res)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mini deterministic LCG for generating test data without the rand crate ──
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
        }

        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Standard normal via Box-Muller
        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-300);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    // ── White-noise series of length n ──
    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = TestRng::new(seed);
        (0..n).map(|_| rng.next_normal()).collect()
    }

    // ── AR(1) series with coefficient phi ──
    fn ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut rng = TestRng::new(seed);
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = phi * y[t - 1] + rng.next_normal();
        }
        y
    }

    // ── Random walk (unit root) ──
    fn random_walk(n: usize, seed: u64) -> Vec<f64> {
        let wn = white_noise(n, seed);
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = y[t - 1] + wn[t];
        }
        y
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ACF tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn acf_lag0_is_one() {
        let x = white_noise(100, 42);
        let rho = acf(&x, 10);
        assert!((rho[0] - 1.0).abs() < 1e-12, "acf[0]={}", rho[0]);
    }

    #[test]
    fn acf_white_noise_near_zero() {
        // For a large WN series, sample ACF at lags 1-10 should be near 0
        let x = white_noise(500, 7);
        let rho = acf(&x, 10);
        let max_abs: f64 = rho[1..]
            .iter()
            .map(|r| r.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        // 95% bound for WN ACF ≈ 2/sqrt(n) = 2/sqrt(500) ≈ 0.089; allow 3x for safety
        assert!(
            max_abs < 0.27,
            "max |ACF(k)| = {max_abs} for WN; expected < 0.27"
        );
    }

    #[test]
    fn acf_ar1_has_geometric_decay() {
        let phi = 0.7;
        let x = ar1(1000, phi, 13);
        let rho = acf(&x, 5);
        // ACF at lag k should be ≈ phi^k
        for (k, &actual) in rho.iter().enumerate().take(6).skip(1) {
            let expected = phi.powi(k as i32);
            assert!(
                (actual - expected).abs() < 0.08,
                "lag={k} acf={actual} expected≈{expected}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Ljung-Box tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn ljung_box_white_noise_high_p() {
        let x = white_noise(200, 99);
        let (q, p) = ljung_box(&x, 10).expect("ok");
        assert!(q.is_finite());
        // For WN we expect p > 0.05 most of the time (but it's stochastic)
        // With seed 99 and n=200, this should hold
        assert!(p > 0.01, "Ljung-Box p={p} for WN; expected > 0.01");
    }

    #[test]
    fn ljung_box_ar1_low_p() {
        // Strongly autocorrelated series → low p-value → reject WN
        let x = ar1(200, 0.9, 17);
        let (q, p) = ljung_box(&x, 10).expect("ok");
        assert!(q.is_finite());
        assert!(p < 0.001, "Ljung-Box p={p} for AR(0.9); expected << 0.001");
    }

    #[test]
    fn ljung_box_returns_finite() {
        let x = white_noise(50, 55);
        let (q, p) = ljung_box(&x, 5).expect("ok");
        assert!(q.is_finite());
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Box-Pierce tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn box_pierce_shape() {
        let x = white_noise(100, 33);
        let (q, p) = box_pierce(&x, 10).expect("ok");
        assert!(q.is_finite() && q >= 0.0);
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    #[test]
    fn box_pierce_ar1_low_p() {
        let x = ar1(300, 0.85, 21);
        let (_q, p) = box_pierce(&x, 10).expect("ok");
        assert!(
            p < 0.001,
            "Box-Pierce p={p} for AR(0.85); expected << 0.001"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ADF tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn adf_random_walk_fail_reject() {
        // A random walk should NOT reject the unit-root null at 5%
        let rw = random_walk(200, 44);
        let res = adf_test(&rw, 4, AdfTrend::Constant).expect("ok");
        assert!(
            !res.reject_unit_root,
            "ADF τ={} should not reject unit root for RW",
            res.statistic
        );
    }

    #[test]
    fn adf_stationary_series_reject() {
        // Strongly stationary series: y_t = 0.3*y_{t-1} + ε_t → should reject unit root
        let x = ar1(500, 0.3, 88);
        let res = adf_test(&x, 4, AdfTrend::Constant).expect("ok");
        assert!(
            res.reject_unit_root,
            "ADF τ={} should reject unit root for AR(0.3)",
            res.statistic
        );
    }

    #[test]
    fn adf_result_fields_valid() {
        let x = ar1(100, 0.5, 77);
        let res = adf_test(&x, 3, AdfTrend::Constant).expect("ok");
        assert!(res.statistic.is_finite());
        assert!(res.p_value_approx > 0.0 && res.p_value_approx < 1.0);
        assert!(res.n_obs > 0);
        assert_eq!(res.critical_values.len(), 3);
    }

    #[test]
    fn adf_constant_trend_model() {
        let x = ar1(150, 0.5, 66);
        let res = adf_test(&x, 2, AdfTrend::ConstantTrend).expect("ok");
        assert!(res.statistic.is_finite());
        // critical values should be the ConstantTrend row
        assert!((res.critical_values[0] - MACKINNON_CV[2][0]).abs() < 1e-6);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // KPSS tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn kpss_stationary_accepts_h0() {
        // Stationary AR(1) with small ρ → KPSS statistic should be small → do not reject H₀
        let x = ar1(300, 0.2, 55);
        let (eta, _p) = kpss_test(&x, KpssTrend::Level).expect("ok");
        // Critical value at 5% is 0.463 — stationary series should be below it
        assert!(
            eta < 0.463,
            "KPSS η={eta} for stationary AR(0.2); expected < 0.463"
        );
    }

    #[test]
    fn kpss_random_walk_rejects_h0() {
        // Random walk → non-stationary → large KPSS statistic
        let rw = random_walk(300, 66);
        let (eta, _p) = kpss_test(&rw, KpssTrend::Level).expect("ok");
        // Should be well above 0.463 for a unit root series
        assert!(
            eta > 0.463,
            "KPSS η={eta} for random walk; expected > 0.463"
        );
    }

    #[test]
    fn kpss_trend_mode() {
        let x = ar1(200, 0.3, 99);
        let (eta, p) = kpss_test(&x, KpssTrend::Trend).expect("ok");
        assert!(eta.is_finite() && eta >= 0.0);
        assert!(p > 0.0 && p <= 1.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Durbin-Watson tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn durbin_watson_no_autocorr() {
        // WN residuals → DW ≈ 2
        let r = white_noise(500, 11);
        let dw = durbin_watson(&r).expect("ok");
        assert!((dw - 2.0).abs() < 0.3, "DW={dw} for WN; expected ≈ 2.0");
    }

    #[test]
    fn durbin_watson_positive_autocorr() {
        // Positively autocorrelated residuals → DW < 2
        // Simulate: e_t = 0.8*e_{t-1} + WN
        let n = 500;
        let mut rng = TestRng::new(123);
        let mut e = vec![0.0; n];
        for t in 1..n {
            e[t] = 0.8 * e[t - 1] + rng.next_normal();
        }
        let dw = durbin_watson(&e).expect("ok");
        assert!(dw < 1.5, "DW={dw} for AR(0.8) residuals; expected < 1.5");
    }

    #[test]
    fn durbin_watson_negative_autocorr() {
        // Negatively autocorrelated residuals → DW > 2
        let n = 400;
        let mut rng = TestRng::new(321);
        let mut e = vec![0.0; n];
        for t in 1..n {
            e[t] = -0.8 * e[t - 1] + rng.next_normal();
        }
        let dw = durbin_watson(&e).expect("ok");
        assert!(dw > 2.5, "DW={dw} for MA(-0.8) residuals; expected > 2.5");
    }

    #[test]
    fn durbin_watson_too_short_error() {
        let r = vec![1.0];
        assert!(durbin_watson(&r).is_err());
    }
}
