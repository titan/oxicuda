//! Holt-Winters Exponential Smoothing (ETS).
//!
//! Implements all four variants of the Holt-Winters family:
//! - Simple (SES): single smoothing parameter α
//! - Double (Holt's linear trend): parameters α, β
//! - Triple Additive: parameters α, β, γ with additive seasonality
//! - Triple Multiplicative: parameters α, β, γ with multiplicative seasonality
//!
//! Parameter optimization uses grid search over {0.1, 0.2, ..., 0.9}^k to minimize SSE.
//!
//! # References
//! - Holt, C.E. (1957). *Forecasting seasonals and trends by exponentially weighted moving averages*.
//! - Winters, P.R. (1960). *Forecasting sales by exponentially weighted moving averages*.
//! - Gardner, E.S. (1985). *Exponential smoothing: The state of the art*. JForecast 4(1):1-28.
//! - Hyndman, R.J. et al. (2008). *Forecasting with Exponential Smoothing*. Springer.

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Variant of Holt-Winters exponential smoothing to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwVariant {
    /// Simple exponential smoothing (SES), single parameter α.
    Simple,
    /// Holt's double exponential smoothing (linear trend), parameters α and β.
    Double,
    /// Triple Holt-Winters with additive seasonality, parameters α, β, and γ.
    Additive,
    /// Triple Holt-Winters with multiplicative seasonality, parameters α, β, and γ.
    Multiplicative,
}

/// Configuration for a Holt-Winters fitting run.
#[derive(Debug, Clone)]
pub struct HwConfig {
    /// Which ETS variant to use.
    pub variant: HwVariant,
    /// Seasonal period m (required for `Additive`/`Multiplicative`; ignored otherwise).
    pub period: usize,
    /// Level smoothing parameter α ∈ (0, 1). `None` ⇒ optimize via grid search.
    pub alpha: Option<f64>,
    /// Trend smoothing parameter β ∈ (0, 1). `None` ⇒ optimize via grid search.
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter γ ∈ (0, 1). `None` ⇒ optimize via grid search.
    pub gamma: Option<f64>,
}

impl Default for HwConfig {
    fn default() -> Self {
        Self {
            variant: HwVariant::Additive,
            period: 12,
            alpha: None,
            beta: None,
            gamma: None,
        }
    }
}

/// Result of a Holt-Winters fitting run.
#[derive(Debug, Clone)]
pub struct HwResult {
    /// Configuration used for this fit.
    pub config: HwConfig,
    /// Fitted level smoothing parameter α.
    pub alpha: f64,
    /// Fitted trend smoothing parameter β (0.0 for `Simple`).
    pub beta: f64,
    /// Fitted seasonal smoothing parameter γ (0.0 for `Simple`/`Double`).
    pub gamma: f64,
    /// One-step-ahead forecasts for t = 1, …, n−1 (length n−1).
    pub fitted: Vec<f64>,
    /// Level state L_t for t = 0, …, n−1 (length n).
    pub level: Vec<f64>,
    /// Trend state T_t for t = 0, …, n−1 (length n; empty for `Simple`).
    pub trend: Vec<f64>,
    /// Seasonal state buffer S_t (empty for `Simple`/`Double`; length ≥ n for seasonal variants).
    pub seasonal: Vec<f64>,
    /// Sum of squared one-step errors over t = 1, …, n−1.
    pub sse: f64,
    /// Number of observations.
    pub n: usize,
}

// ---------------------------------------------------------------------------
// Grid of parameter values for search
// ---------------------------------------------------------------------------

const GRID: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

fn grid_values(fixed: Option<f64>) -> Vec<f64> {
    match fixed {
        Some(v) => vec![v],
        None => GRID.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_param(val: Option<f64>, name: &'static str) -> StatsResult<()> {
    if let Some(v) = val {
        if !(v > 0.0 && v < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: name.to_owned(),
                reason: format!("must be strictly in (0, 1), got {v}"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal engine — Simple ETS
// ---------------------------------------------------------------------------

/// Run Simple ETS for the given α.
///
/// Returns `(fitted[1..n], level[0..n], sse)`.
fn ses_engine(y: &[f64], alpha: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let n = y.len();
    let mut level: Vec<f64> = Vec::with_capacity(n);
    let mut fitted: Vec<f64> = Vec::with_capacity(n - 1);
    level.push(y[0]);
    let mut sse = 0.0_f64;
    for t in 1..n {
        let l_prev = level[t - 1];
        let yhat = l_prev;
        fitted.push(yhat);
        let err = y[t] - yhat;
        sse += err * err;
        level.push(alpha * y[t] + (1.0 - alpha) * l_prev);
    }
    (fitted, level, sse)
}

// ---------------------------------------------------------------------------
// Internal engine — Double ETS (Holt's linear)
// ---------------------------------------------------------------------------

/// Run Double ETS for the given (α, β).
///
/// Returns `(fitted, level[0..n], trend[0..n], sse)`.
fn double_engine(y: &[f64], alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let n = y.len();
    let mut level: Vec<f64> = Vec::with_capacity(n);
    let mut trend: Vec<f64> = Vec::with_capacity(n);
    let mut fitted: Vec<f64> = Vec::with_capacity(n - 1);
    level.push(y[0]);
    trend.push(y[1] - y[0]);
    let mut sse = 0.0_f64;
    for t in 1..n {
        let l_prev = level[t - 1];
        let tr_prev = trend[t - 1];
        let yhat = l_prev + tr_prev;
        fitted.push(yhat);
        let err = y[t] - yhat;
        sse += err * err;
        let l_new = alpha * y[t] + (1.0 - alpha) * (l_prev + tr_prev);
        let tr_new = beta * (l_new - l_prev) + (1.0 - beta) * tr_prev;
        level.push(l_new);
        trend.push(tr_new);
    }
    (fitted, level, trend, sse)
}

// ---------------------------------------------------------------------------
// Internal engine — Triple Additive ETS
// ---------------------------------------------------------------------------

/// Compute additive initialisation: (L₀, T₀, S[0..m]).
fn additive_init(y: &[f64], m: usize) -> (f64, f64, Vec<f64>) {
    let l0: f64 = y[..m].iter().sum::<f64>() / m as f64;
    let l1: f64 = y[m..2 * m].iter().sum::<f64>() / m as f64;
    let t0 = (l1 - l0) / m as f64;
    let mut s: Vec<f64> = (0..m).map(|i| y[i] - l0).collect();
    let s_mean = s.iter().sum::<f64>() / m as f64;
    for si in &mut s {
        *si -= s_mean;
    }
    (l0, t0, s)
}

/// Run Triple Additive ETS for the given (α, β, γ).
///
/// Returns `(fitted, level[0..n], trend[0..n], seasonal_buf[0..n+m], sse)`.
fn additive_engine(
    y: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    m: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let n = y.len();
    let (l0, t0, s_init) = additive_init(y, m);

    // Seasonal buffer indexed by absolute time position.
    // Positions 0..m are filled with the initialisation values.
    // Positions m..n are updated during recursion.
    let buf_len = n + m;
    let mut s_buf: Vec<f64> = vec![0.0; buf_len];
    s_buf[..m].copy_from_slice(&s_init[..m]);

    let mut level: Vec<f64> = Vec::with_capacity(n);
    let mut trend: Vec<f64> = Vec::with_capacity(n);
    let mut fitted: Vec<f64> = Vec::with_capacity(n - 1);
    level.push(l0);
    trend.push(t0);

    let mut sse = 0.0_f64;

    // t runs from 1 to n-1 (we produce n-1 one-step forecasts).
    // At step t the forecast uses L_{t-1}, T_{t-1}, and S_{t-m}
    // (which lives at s_buf[t - m] when t >= m, else wrap to s_buf[t % m]).
    for t in 1..n {
        let l_prev = level[t - 1];
        let tr_prev = trend[t - 1];

        // Index into seasonal buffer for position t-m
        let s_lag_idx = t + m - m; // = t (shifted into initialisation window)
        // The seasonal lag S_{t-m} lives at index (t - m + m) = t in s_buf when t >= m.
        // But our buffer has initialisation at [0..m] so position `t-m` in time maps to
        // s_buf index `t - m` offset by the m-wide initialisation window, i.e. just `t-m`
        // if we store initial seasons at [0..m).
        // Cleaner: the absolute index for the seasonal at time `t - m` in the buffer is:
        // t - m  (0-based) which is valid for t >= m; for t < m we fall back to init.
        let s_lag = if t >= m { s_buf[t - m] } else { s_buf[t] };

        let yhat = l_prev + tr_prev + s_lag;
        fitted.push(yhat);
        let err = y[t] - yhat;
        sse += err * err;

        // Update level: uses S_{t-m} (already in s_lag)
        let l_new = alpha * (y[t] - s_lag) + (1.0 - alpha) * (l_prev + tr_prev);
        let tr_new = beta * (l_new - l_prev) + (1.0 - beta) * tr_prev;
        // Update seasonal at position t in the buffer
        let s_new = gamma * (y[t] - l_prev - tr_prev) + (1.0 - gamma) * s_lag;
        s_buf[s_lag_idx] = s_new;

        level.push(l_new);
        trend.push(tr_new);
    }

    (fitted, level, trend, s_buf, sse)
}

// ---------------------------------------------------------------------------
// Internal engine — Triple Multiplicative ETS
// ---------------------------------------------------------------------------

/// Compute multiplicative initialisation: (L₀, T₀, S[0..m]).
fn multiplicative_init(y: &[f64], m: usize) -> (f64, f64, Vec<f64>) {
    let l0: f64 = y[..m].iter().sum::<f64>() / m as f64;
    let l1: f64 = y[m..2 * m].iter().sum::<f64>() / m as f64;
    let t0 = (l1 - l0) / m as f64;
    let safe_l0 = l0.max(1e-300);
    let s: Vec<f64> = (0..m).map(|i| y[i] / safe_l0).collect();
    (l0, t0, s)
}

/// Run Triple Multiplicative ETS for the given (α, β, γ).
///
/// Returns `(fitted, level[0..n], trend[0..n], seasonal_buf[0..n+m], sse)`.
fn multiplicative_engine(
    y: &[f64],
    alpha: f64,
    beta: f64,
    gamma: f64,
    m: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let n = y.len();
    let (l0, t0, s_init) = multiplicative_init(y, m);

    let buf_len = n + m;
    let mut s_buf: Vec<f64> = vec![1.0; buf_len];
    s_buf[..m].copy_from_slice(&s_init[..m]);

    let mut level: Vec<f64> = Vec::with_capacity(n);
    let mut trend: Vec<f64> = Vec::with_capacity(n);
    let mut fitted: Vec<f64> = Vec::with_capacity(n - 1);
    level.push(l0);
    trend.push(t0);

    let mut sse = 0.0_f64;

    for t in 1..n {
        let l_prev = level[t - 1];
        let tr_prev = trend[t - 1];
        let s_lag_idx = t;
        let s_lag = if t >= m { s_buf[t - m] } else { s_buf[t] };

        let yhat = (l_prev + tr_prev) * s_lag;
        fitted.push(yhat);
        let err = y[t] - yhat;
        sse += err * err;

        let safe_s_lag = s_lag.max(1e-300);
        let safe_l_prev = l_prev.max(1e-300);

        let l_new = alpha * (y[t] / safe_s_lag) + (1.0 - alpha) * (l_prev + tr_prev);
        let tr_new = beta * (l_new - l_prev) + (1.0 - beta) * tr_prev;
        let s_new = gamma * (y[t] / safe_l_prev) + (1.0 - gamma) * s_lag;
        s_buf[s_lag_idx] = s_new;

        level.push(l_new);
        trend.push(tr_new);
    }

    (fitted, level, trend, s_buf, sse)
}

// ---------------------------------------------------------------------------
// Grid-search optimisation wrappers
// ---------------------------------------------------------------------------

fn optimise_simple(y: &[f64], fixed_alpha: Option<f64>) -> (f64, f64) {
    let mut best_sse = f64::MAX;
    let mut best_a = 0.3_f64;
    for a in grid_values(fixed_alpha) {
        let (_, _, sse) = ses_engine(y, a);
        if sse < best_sse {
            best_sse = sse;
            best_a = a;
        }
    }
    (best_a, best_sse)
}

fn optimise_double(
    y: &[f64],
    fixed_alpha: Option<f64>,
    fixed_beta: Option<f64>,
) -> (f64, f64, f64) {
    let mut best_sse = f64::MAX;
    let mut best_a = 0.3_f64;
    let mut best_b = 0.1_f64;
    for a in grid_values(fixed_alpha) {
        for b in grid_values(fixed_beta) {
            let (_, _, _, sse) = double_engine(y, a, b);
            if sse < best_sse {
                best_sse = sse;
                best_a = a;
                best_b = b;
            }
        }
    }
    (best_a, best_b, best_sse)
}

fn optimise_additive(
    y: &[f64],
    m: usize,
    fixed_alpha: Option<f64>,
    fixed_beta: Option<f64>,
    fixed_gamma: Option<f64>,
) -> (f64, f64, f64, f64) {
    let mut best_sse = f64::MAX;
    let mut best_a = 0.3_f64;
    let mut best_b = 0.1_f64;
    let mut best_g = 0.1_f64;
    for a in grid_values(fixed_alpha) {
        for b in grid_values(fixed_beta) {
            for g in grid_values(fixed_gamma) {
                let (_, _, _, _, sse) = additive_engine(y, a, b, g, m);
                if sse < best_sse {
                    best_sse = sse;
                    best_a = a;
                    best_b = b;
                    best_g = g;
                }
            }
        }
    }
    (best_a, best_b, best_g, best_sse)
}

fn optimise_multiplicative(
    y: &[f64],
    m: usize,
    fixed_alpha: Option<f64>,
    fixed_beta: Option<f64>,
    fixed_gamma: Option<f64>,
) -> (f64, f64, f64, f64) {
    let mut best_sse = f64::MAX;
    let mut best_a = 0.3_f64;
    let mut best_b = 0.1_f64;
    let mut best_g = 0.1_f64;
    for a in grid_values(fixed_alpha) {
        for b in grid_values(fixed_beta) {
            for g in grid_values(fixed_gamma) {
                let (_, _, _, _, sse) = multiplicative_engine(y, a, b, g, m);
                if sse < best_sse {
                    best_sse = sse;
                    best_a = a;
                    best_b = b;
                    best_g = g;
                }
            }
        }
    }
    (best_a, best_b, best_g, best_sse)
}

// ---------------------------------------------------------------------------
// Count free parameters
// ---------------------------------------------------------------------------

fn num_params(variant: HwVariant) -> usize {
    match variant {
        HwVariant::Simple => 1,
        HwVariant::Double => 2,
        HwVariant::Additive | HwVariant::Multiplicative => 3,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit a Holt-Winters model to time series `y`.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] — fewer than 2 observations, or fewer than
///   `2 * period` for seasonal variants.
/// - [`StatsError::InvalidParameter`] — `period == 0`, any fixed parameter outside `(0, 1)`,
///   or negative/zero values for the multiplicative variant.
pub fn hw_fit(y: &[f64], config: &HwConfig) -> StatsResult<HwResult> {
    let n = y.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    validate_param(config.alpha, "alpha")?;
    validate_param(config.beta, "beta")?;
    validate_param(config.gamma, "gamma")?;

    match config.variant {
        HwVariant::Simple => {
            let (alpha, _) = optimise_simple(y, config.alpha);
            let (fitted, level, sse) = ses_engine(y, alpha);
            Ok(HwResult {
                config: config.clone(),
                alpha,
                beta: 0.0,
                gamma: 0.0,
                fitted,
                level,
                trend: Vec::new(),
                seasonal: Vec::new(),
                sse,
                n,
            })
        }

        HwVariant::Double => {
            let (alpha, beta, _) = optimise_double(y, config.alpha, config.beta);
            let (fitted, level, trend, sse) = double_engine(y, alpha, beta);
            Ok(HwResult {
                config: config.clone(),
                alpha,
                beta,
                gamma: 0.0,
                fitted,
                level,
                trend,
                seasonal: Vec::new(),
                sse,
                n,
            })
        }

        HwVariant::Additive => {
            let m = config.period;
            if m == 0 {
                return Err(StatsError::InvalidParameter {
                    name: "period".to_owned(),
                    reason: "must be >= 1 for additive variant".to_owned(),
                });
            }
            if n < 2 * m {
                return Err(StatsError::InsufficientSampleSize {
                    got: n,
                    need: 2 * m,
                });
            }
            let (alpha, beta, gamma, _) =
                optimise_additive(y, m, config.alpha, config.beta, config.gamma);
            let (fitted, level, trend, seasonal, sse) = additive_engine(y, alpha, beta, gamma, m);
            Ok(HwResult {
                config: config.clone(),
                alpha,
                beta,
                gamma,
                fitted,
                level,
                trend,
                seasonal,
                sse,
                n,
            })
        }

        HwVariant::Multiplicative => {
            let m = config.period;
            if m == 0 {
                return Err(StatsError::InvalidParameter {
                    name: "period".to_owned(),
                    reason: "must be >= 1 for multiplicative variant".to_owned(),
                });
            }
            if n < 2 * m {
                return Err(StatsError::InsufficientSampleSize {
                    got: n,
                    need: 2 * m,
                });
            }
            if y.iter().any(|&v| v <= 0.0) {
                return Err(StatsError::InvalidParameter {
                    name: "y".to_owned(),
                    reason: "all observations must be strictly positive for the multiplicative \
                             variant"
                        .to_owned(),
                });
            }
            let (alpha, beta, gamma, _) =
                optimise_multiplicative(y, m, config.alpha, config.beta, config.gamma);
            let (fitted, level, trend, seasonal, sse) =
                multiplicative_engine(y, alpha, beta, gamma, m);
            Ok(HwResult {
                config: config.clone(),
                alpha,
                beta,
                gamma,
                fitted,
                level,
                trend,
                seasonal,
                sse,
                n,
            })
        }
    }
}

/// Generate `h` forecasts beyond the last observation using a fitted HW model.
///
/// # Errors
/// - [`StatsError::InsufficientSampleSize`] if `h == 0`.
pub fn hw_forecast(result: &HwResult, h: usize) -> StatsResult<Vec<f64>> {
    if h == 0 {
        return Err(StatsError::InsufficientSampleSize { got: 0, need: 1 });
    }
    let n = result.n;
    let l_n = *result.level.last().unwrap_or(&0.0);

    let forecasts = match result.config.variant {
        HwVariant::Simple => {
            vec![l_n; h]
        }
        HwVariant::Double => {
            let t_n = *result.trend.last().unwrap_or(&0.0);
            (1..=h).map(|k| l_n + k as f64 * t_n).collect()
        }
        HwVariant::Additive => {
            let t_n = *result.trend.last().unwrap_or(&0.0);
            let m = result.config.period;
            let sea = &result.seasonal;
            let sea_len = sea.len();
            (1..=h)
                .map(|k| {
                    // At forecast horizon k we need S_{n-m + (k-1) mod m}
                    // The last m seasonal states live at indices n-m .. n in the buffer.
                    let s_base = n.saturating_sub(m);
                    let idx = s_base + (k - 1) % m;
                    let s = if idx < sea_len { sea[idx] } else { 0.0 };
                    l_n + k as f64 * t_n + s
                })
                .collect()
        }
        HwVariant::Multiplicative => {
            let t_n = *result.trend.last().unwrap_or(&0.0);
            let m = result.config.period;
            let sea = &result.seasonal;
            let sea_len = sea.len();
            (1..=h)
                .map(|k| {
                    let s_base = n.saturating_sub(m);
                    let idx = s_base + (k - 1) % m;
                    let s = if idx < sea_len { sea[idx] } else { 1.0 };
                    (l_n + k as f64 * t_n) * s
                })
                .collect()
        }
    };
    Ok(forecasts)
}

/// Akaike Information Criterion: AIC = n · ln(SSE/n) + 2p.
#[must_use]
pub fn hw_aic(result: &HwResult) -> f64 {
    let n = result.n as f64;
    let p = num_params(result.config.variant) as f64;
    let sse_n = (result.sse / n).max(1e-300);
    n * sse_n.ln() + 2.0 * p
}

/// Bayesian Information Criterion: BIC = n · ln(SSE/n) + p · ln(n).
#[must_use]
pub fn hw_bic(result: &HwResult) -> f64 {
    let n = result.n as f64;
    let p = num_params(result.config.variant) as f64;
    let sse_n = (result.sse / n).max(1e-300);
    n * sse_n.ln() + p * n.ln()
}

/// Compute raw residuals y_t − ŷ_t aligned to the fitted values (length n−1).
///
/// # Errors
/// - [`StatsError::DimensionMismatch`] if `y.len() != result.n`.
pub fn hw_residuals(result: &HwResult, y: &[f64]) -> StatsResult<Vec<f64>> {
    if y.len() != result.n {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: result.n,
        });
    }
    let residuals = result
        .fitted
        .iter()
        .zip(y[1..].iter())
        .map(|(&f, &yi)| yi - f)
        .collect();
    Ok(residuals)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_series(n: usize, val: f64) -> Vec<f64> {
        vec![val; n]
    }

    fn linear_series(n: usize, slope: f64) -> Vec<f64> {
        (0..n).map(|i| slope * i as f64).collect()
    }

    fn additive_seasonal(n: usize, m: usize, amplitude: f64) -> Vec<f64> {
        use std::f64::consts::PI;
        (0..n)
            .map(|i| 10.0 + amplitude * (2.0 * PI * i as f64 / m as f64).sin())
            .collect()
    }

    fn multiplicative_seasonal(n: usize, m: usize) -> Vec<f64> {
        use std::f64::consts::PI;
        (0..n)
            .map(|i| 100.0 * (1.0 + 0.3 * (2.0 * PI * i as f64 / m as f64).sin()))
            .collect()
    }

    // 1. Simple ETS fits constant signal
    #[test]
    fn simple_ets_constant_signal() {
        let y = constant_series(20, 5.0);
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            alpha: Some(0.3),
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let last = *res.fitted.last().expect("last should succeed");
        assert!(
            (last - 5.0).abs() < 0.5,
            "constant fitted should converge; got {last}"
        );
    }

    // 2. Double ETS captures linear trend
    #[test]
    fn double_ets_linear_trend() {
        let y = linear_series(30, 2.0); // 0, 2, 4, …, 58
        let cfg = HwConfig {
            variant: HwVariant::Double,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let fcast = hw_forecast(&res, 1).expect("hw_forecast should succeed");
        let expected = 2.0 * 30.0; // next value
        let rel_err = (fcast[0] - expected).abs() / expected.max(1.0);
        assert!(rel_err < 0.1, "rel_err={rel_err:.4}");
    }

    // 3. Additive ETS tracks seasonal oscillation
    #[test]
    fn additive_ets_seasonal() {
        let m = 4;
        let y = additive_seasonal(32, m, 3.0);
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let variance = y.iter().map(|&v| (v - 10.0).powi(2)).sum::<f64>() / y.len() as f64;
        let mse = res.sse / (res.n - 1) as f64;
        assert!(mse < variance, "mse={mse:.3} vs variance={variance:.3}");
    }

    // 4. Multiplicative ETS works on positive seasonal signal
    #[test]
    fn multiplicative_ets_positive_seasonal() {
        let m = 4;
        let y = multiplicative_seasonal(32, m);
        let cfg = HwConfig {
            variant: HwVariant::Multiplicative,
            period: m,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!(res.sse.is_finite());
    }

    // 5. hw_forecast h=1 returns single element
    #[test]
    fn forecast_h1() {
        let y = linear_series(20, 1.0);
        let cfg = HwConfig {
            variant: HwVariant::Double,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let f = hw_forecast(&res, 1).expect("hw_forecast should succeed");
        assert_eq!(f.len(), 1);
    }

    // 6. hw_forecast h=6 returns six elements
    #[test]
    fn forecast_h6() {
        let m = 4;
        let y = additive_seasonal(24, m, 2.0);
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let f = hw_forecast(&res, 6).expect("hw_forecast should succeed");
        assert_eq!(f.len(), 6);
    }

    // 7. AIC formula matches reference calculation
    #[test]
    fn aic_formula() {
        let y = constant_series(20, 5.0);
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            alpha: Some(0.5),
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let n = res.n as f64;
        let p = 1.0_f64;
        let expected = n * (res.sse / n).max(1e-300).ln() + 2.0 * p;
        assert!((hw_aic(&res) - expected).abs() < 1e-10);
    }

    // 8. BIC >= AIC for n > 7
    #[test]
    fn bic_ge_aic() {
        let m = 4;
        let y = additive_seasonal(48, m, 2.0);
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!(hw_bic(&res) >= hw_aic(&res));
    }

    // 9. Additive SSE <= simple SSE on strongly seasonal data
    #[test]
    fn additive_better_than_simple_on_seasonal() {
        let m = 4;
        let y = additive_seasonal(32, m, 5.0);
        let cfg_add = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        let cfg_ses = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            ..Default::default()
        };
        let sse_add = hw_fit(&y, &cfg_add).expect("hw_fit should succeed").sse;
        let sse_ses = hw_fit(&y, &cfg_ses).expect("hw_fit should succeed").sse;
        assert!(sse_add <= sse_ses, "add={sse_add:.2} ses={sse_ses:.2}");
    }

    // 10. Optimised alpha in (0, 1)
    #[test]
    fn optimised_alpha_unit_interval() {
        let y: Vec<f64> = (0..20).map(|i| i as f64 + (i as f64 * 0.3).sin()).collect();
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!(res.alpha > 0.0 && res.alpha < 1.0);
    }

    // 11. Optimised beta in (0, 1)
    #[test]
    fn optimised_beta_unit_interval() {
        let y = linear_series(20, 3.0);
        let cfg = HwConfig {
            variant: HwVariant::Double,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!(res.beta > 0.0 && res.beta < 1.0);
    }

    // 12. Optimised gamma in (0, 1)
    #[test]
    fn optimised_gamma_unit_interval() {
        let m = 4;
        let y = additive_seasonal(32, m, 3.0);
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!(res.gamma > 0.0 && res.gamma < 1.0);
    }

    // 13. Fixed alpha=0.5 used exactly (not overridden)
    #[test]
    fn fixed_alpha_exact() {
        let y = constant_series(20, 7.0);
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            alpha: Some(0.5),
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!((res.alpha - 0.5).abs() < 1e-12);
    }

    // 14. fitted.len() == n - 1
    #[test]
    fn fitted_length() {
        let y = constant_series(25, 3.0);
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert_eq!(res.fitted.len(), 24);
    }

    // 15. level.len() == n
    #[test]
    fn level_length() {
        let y = linear_series(15, 2.0);
        let cfg = HwConfig {
            variant: HwVariant::Double,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert_eq!(res.level.len(), 15);
    }

    // 16. trend.len() == n for Double/Additive/Multiplicative
    #[test]
    fn trend_length_double() {
        let y = linear_series(15, 2.0);
        let cfg = HwConfig {
            variant: HwVariant::Double,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert_eq!(res.trend.len(), 15);
    }

    // 17. seasonal.len() >= n for Additive
    #[test]
    fn seasonal_length_additive() {
        let m = 4;
        let y = additive_seasonal(32, m, 2.0);
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        assert!(res.seasonal.len() >= res.n);
    }

    // 18. hw_residuals length == fitted.len()
    #[test]
    fn residuals_length() {
        let y = constant_series(20, 5.0);
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            period: 1,
            ..Default::default()
        };
        let res = hw_fit(&y, &cfg).expect("hw_fit should succeed");
        let r = hw_residuals(&res, &y).expect("hw_residuals should succeed");
        assert_eq!(r.len(), res.fitted.len());
    }

    // 19. y.len() < 2 → InsufficientSampleSize
    #[test]
    fn insufficient_data_single() {
        let cfg = HwConfig {
            variant: HwVariant::Simple,
            ..Default::default()
        };
        assert!(matches!(
            hw_fit(&[1.0], &cfg),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }

    // 20. period=0 → InvalidParameter
    #[test]
    fn period_zero_error() {
        let y = additive_seasonal(24, 4, 2.0);
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: 0,
            ..Default::default()
        };
        assert!(matches!(
            hw_fit(&y, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    // 21. Multiplicative with negative observation → InvalidParameter
    #[test]
    fn multiplicative_negative_data() {
        let m = 4;
        let mut y = multiplicative_seasonal(32, m);
        y[5] = -1.0;
        let cfg = HwConfig {
            variant: HwVariant::Multiplicative,
            period: m,
            ..Default::default()
        };
        assert!(matches!(
            hw_fit(&y, &cfg),
            Err(StatsError::InvalidParameter { .. })
        ));
    }

    // 22. y.len() < 2*period → InsufficientSampleSize
    #[test]
    fn insufficient_data_seasonal() {
        let m = 12;
        let y = additive_seasonal(20, m, 2.0); // 20 < 24 = 2*12
        let cfg = HwConfig {
            variant: HwVariant::Additive,
            period: m,
            ..Default::default()
        };
        assert!(matches!(
            hw_fit(&y, &cfg),
            Err(StatsError::InsufficientSampleSize { .. })
        ));
    }
}
