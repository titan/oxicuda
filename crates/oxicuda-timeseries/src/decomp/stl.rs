//! STL seasonal-trend decomposition using Loess.
//!
//! Cleveland, Cleveland, McRae & Terpenning (1990)
//! "STL: A Seasonal-Trend Decomposition Procedure Based on Loess."
//! Journal of Official Statistics 6(1):3-33.

use crate::error::{TsError, TsResult};

// ── Config ───────────────────────────────────────────────────────────────────

/// Configuration for STL decomposition.
#[derive(Debug, Clone)]
pub struct StlConfig {
    /// Seasonal period (e.g., 12 for monthly data with annual seasonality). Must be ≥ 2.
    pub period: usize,
    /// Loess bandwidth for seasonal smoothing (must be odd, ≥ 3). Default: 7.
    pub s_window: usize,
    /// Loess bandwidth for trend smoothing (must be odd, ≥ 3).
    pub t_window: usize,
    /// Inner loop iterations. Default: 2.
    pub n_inner: usize,
    /// Outer robustness loop iterations. Default: 0 (no robustness).
    pub n_outer: usize,
}

impl StlConfig {
    /// Create a config with sensible defaults for the given period.
    pub fn new(period: usize) -> Self {
        let t_win = {
            let base = period + period / 2 + 1;
            if base % 2 == 0 { base + 1 } else { base }
        };
        Self {
            period,
            s_window: 7,
            t_window: t_win,
            n_inner: 2,
            n_outer: 0,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────────────────

/// Decomposition result from STL.
#[derive(Debug, Clone)]
pub struct StlResult {
    /// Trend component T_t, length N.
    pub trend: Vec<f64>,
    /// Seasonal component S_t, length N.
    pub seasonal: Vec<f64>,
    /// Remainder R_t = y - T - S, length N.
    pub remainder: Vec<f64>,
    /// Robustness weights (all 1.0 when n_outer == 0), length N.
    pub weights: Vec<f64>,
    /// Number of observations.
    pub n: usize,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Decompose a time series into trend, seasonal, and remainder components.
///
/// # Errors
///
/// - [`TsError::InvalidSequenceLength`] when `period < 2`.
/// - [`TsError::ShapeMismatch`] when `y.len() < 2 * period`.
/// - [`TsError::InvalidKernelSize`] when window sizes are invalid or `n_inner == 0`.
pub fn stl_decompose(y: &[f64], config: &StlConfig) -> TsResult<StlResult> {
    validate_config(y, config)?;

    let n = y.len();
    let mut trend = vec![0.0_f64; n];
    let mut seasonal = vec![0.0_f64; n];
    let mut weights = vec![1.0_f64; n];

    for _ in 0..config.n_outer.max(1) {
        inner_loop(y, config, &weights, &mut trend, &mut seasonal);

        if config.n_outer > 0 {
            let remainder: Vec<f64> = y
                .iter()
                .zip(trend.iter().zip(seasonal.iter()))
                .map(|(&yi, (&ti, &si))| yi - ti - si)
                .collect();
            weights = robustness_weights(&remainder);
        }
    }

    let remainder: Vec<f64> = y
        .iter()
        .zip(trend.iter().zip(seasonal.iter()))
        .map(|(&yi, (&ti, &si))| yi - ti - si)
        .collect();

    if config.n_outer == 0 {
        weights = vec![1.0_f64; n];
    }

    Ok(StlResult {
        trend,
        seasonal,
        remainder,
        weights,
        n,
    })
}

/// Compute seasonal strength: 1 - Var(R) / Var(S + R), clipped to [0, 1].
///
/// Wang, Smith & Hyndman (2006) strength measure.
#[must_use]
pub fn stl_seasonal_strength(result: &StlResult) -> f64 {
    let sr: Vec<f64> = result
        .seasonal
        .iter()
        .zip(result.remainder.iter())
        .map(|(&s, &r)| s + r)
        .collect();
    let var_r = sample_variance(&result.remainder);
    let var_sr = sample_variance(&sr);
    if var_sr < f64::EPSILON {
        return 0.0;
    }
    (1.0 - var_r / var_sr).clamp(0.0, 1.0)
}

/// Compute trend strength: 1 - Var(R) / Var(T + R), clipped to [0, 1].
#[must_use]
pub fn stl_trend_strength(result: &StlResult) -> f64 {
    let tr: Vec<f64> = result
        .trend
        .iter()
        .zip(result.remainder.iter())
        .map(|(&t, &r)| t + r)
        .collect();
    let var_r = sample_variance(&result.remainder);
    let var_tr = sample_variance(&tr);
    if var_tr < f64::EPSILON {
        return 0.0;
    }
    (1.0 - var_r / var_tr).clamp(0.0, 1.0)
}

/// Naive forecast: extend trend linearly + last seasonal cycle.
///
/// Returns `h` forecast values.
///
/// # Errors
///
/// - [`TsError::InvalidHorizon`] when `h == 0` but actually returns empty vec (no error).
/// - [`TsError::InvalidSequenceLength`] when `period == 0`.
pub fn stl_naive_forecast(result: &StlResult, period: usize, h: usize) -> TsResult<Vec<f64>> {
    if h == 0 {
        return Ok(Vec::new());
    }
    if period == 0 {
        return Err(TsError::InvalidSequenceLength(0));
    }

    let n = result.n;
    // Linear trend extrapolation from the last two trend values
    let slope = if n >= 2 {
        result.trend[n - 1] - result.trend[n - 2]
    } else {
        0.0
    };
    let last_trend = result.trend[n - 1];

    let mut forecast = Vec::with_capacity(h);
    for k in 1..=h {
        let trend_val = last_trend + slope * k as f64;
        // Wrap the seasonal index into the last full period
        let seasonal_idx = (n - period + (k - 1) % period) % result.seasonal.len();
        let seasonal_val = result.seasonal[seasonal_idx];
        forecast.push(trend_val + seasonal_val);
    }
    Ok(forecast)
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn validate_config(y: &[f64], config: &StlConfig) -> TsResult<()> {
    if config.period < 2 {
        return Err(TsError::InvalidSequenceLength(config.period));
    }
    if y.len() < 2 * config.period {
        return Err(TsError::ShapeMismatch {
            msg: format!("need ≥ 2 periods, got {} < {}", y.len(), 2 * config.period),
        });
    }
    if config.s_window < 3 || config.s_window % 2 == 0 {
        return Err(TsError::InvalidKernelSize(config.s_window));
    }
    if config.t_window < 3 || config.t_window % 2 == 0 {
        return Err(TsError::InvalidKernelSize(config.t_window));
    }
    if config.n_inner == 0 {
        return Err(TsError::InvalidKernelSize(0));
    }
    Ok(())
}

fn inner_loop(
    y: &[f64],
    config: &StlConfig,
    rob_weights: &[f64],
    trend: &mut [f64],
    seasonal: &mut [f64],
) {
    let n = y.len();
    let period = config.period;

    for _ in 0..config.n_inner {
        // Step 1: Detrend
        let detrended: Vec<f64> = y
            .iter()
            .zip(trend.iter())
            .map(|(&yi, &ti)| yi - ti)
            .collect();

        // Step 2: Cycle-subseries smoothing
        let mut s_star = vec![0.0_f64; n];
        for s in 0..period {
            let indices: Vec<usize> = (s..n).step_by(period).collect();
            let subseries: Vec<f64> = indices.iter().map(|&i| detrended[i]).collect();
            let rw: Vec<f64> = indices.iter().map(|&i| rob_weights[i]).collect();
            let smoothed = loess_smooth(&subseries, config.s_window, &rw);
            for (k, &idx) in indices.iter().enumerate() {
                s_star[idx] = smoothed[k];
            }
        }

        // Step 3: Low-pass filter on S*
        let lp = lowpass_filter(&s_star, period);

        // Step 4: Seasonal = S* - LP
        for i in 0..n {
            seasonal[i] = s_star[i] - lp[i];
        }

        // Step 5: Deseason
        let deseasoned: Vec<f64> = y
            .iter()
            .zip(seasonal.iter())
            .map(|(&yi, &si)| yi - si)
            .collect();

        // Step 6: Trend via Loess
        let t_rw = rob_weights.to_vec();
        let new_trend = loess_smooth(&deseasoned, config.t_window, &t_rw);
        trend.copy_from_slice(&new_trend);

        // Step 7: implicit — remainder computed after the outer loop
    }
}

/// Loess smoother for equispaced positions 0..n-1 with bandwidth (# neighbors).
///
/// The bandwidth is interpreted as the odd window size: each point uses up to
/// `bandwidth` nearest neighbors. Tricubic weights are applied per Cleveland (1979).
fn loess_smooth(y: &[f64], bandwidth: usize, rob_weights: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![y[0]];
    }

    // Clamp q to [1, n]
    let q = bandwidth.min(n);
    let mut out = vec![0.0_f64; n];

    for x0 in 0..n {
        // Collect distances from x0 to all points
        let mut dists: Vec<f64> = (0..n).map(|i| (i as f64 - x0 as f64).abs()).collect();

        // Find the q-th smallest distance (the bandwidth radius)
        let mut sorted_dists = dists.clone();
        sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let delta = if q <= n {
            sorted_dists[q - 1]
        } else {
            sorted_dists[n - 1]
        };
        // Avoid delta == 0 (all points at same location) → identity
        let delta = delta.max(1e-10);

        // Build tricubic weights combined with robustness weights
        let mut wsum = 0.0_f64;
        let mut wx_sum = 0.0_f64;
        let mut wy_sum = 0.0_f64;
        let mut wxx_sum = 0.0_f64;
        let mut wxy_sum = 0.0_f64;

        for i in 0..n {
            let u = dists[i] / delta;
            if u >= 1.0 {
                dists[i] = 0.0; // reuse as lambda
                continue;
            }
            let tri = {
                let t = 1.0 - u * u * u;
                t * t * t
            };
            let w = tri * rob_weights[i];
            let xi = i as f64;
            wsum += w;
            wx_sum += w * xi;
            wy_sum += w * y[i];
            wxx_sum += w * xi * xi;
            wxy_sum += w * xi * y[i];
            dists[i] = w; // reuse storage
        }

        // Solve 2×2 weighted OLS: [ wsum wx_sum; wx_sum wxx_sum ] * [b; a] = [wy_sum; wxy_sum]
        let det = wsum * wxx_sum - wx_sum * wx_sum;
        if det.abs() < f64::EPSILON || wsum < f64::EPSILON {
            // Degenerate: fall back to weighted mean
            out[x0] = if wsum > f64::EPSILON {
                wy_sum / wsum
            } else {
                y[x0]
            };
        } else {
            let b = (wxx_sum * wy_sum - wx_sum * wxy_sum) / det;
            let a = (wsum * wxy_sum - wx_sum * wy_sum) / det;
            out[x0] = a * x0 as f64 + b;
        }
    }
    out
}

/// Three-pass moving average low-pass filter (MA_period, MA_period, MA_3).
///
/// Uses endpoint-repetition padding to preserve length N throughout.
fn lowpass_filter(s: &[f64], period: usize) -> Vec<f64> {
    let pass1 = moving_average_padded(s, period);
    let pass2 = moving_average_padded(&pass1, period);
    moving_average_padded(&pass2, 3)
}

/// Centered moving average of length `k` with endpoint-repetition boundary padding.
fn moving_average_padded(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || k == 0 {
        return x.to_vec();
    }
    let half = k / 2;
    let inv_k = 1.0 / k as f64;
    let mut out = vec![0.0_f64; n];
    for (t, v) in out.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        for j in 0..k {
            // offset from t-(k-1)/2 for odd, or t-k/2+1 for even
            let offset = j as isize - half as isize;
            let src = (t as isize + offset).clamp(0, n as isize - 1) as usize;
            sum += x[src];
        }
        *v = sum * inv_k;
    }
    out
}

/// Robustness weights from remainder magnitudes (bisquare / biweight kernel).
fn robustness_weights(remainder: &[f64]) -> Vec<f64> {
    let mut abs_r: Vec<f64> = remainder.iter().map(|r| r.abs()).collect();
    let n = abs_r.len();
    if n == 0 {
        return Vec::new();
    }
    // Median of |R|
    abs_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 0 {
        (abs_r[n / 2 - 1] + abs_r[n / 2]) / 2.0
    } else {
        abs_r[n / 2]
    };
    let h = 6.0 * median;
    if h < f64::EPSILON {
        // All residuals zero: perfect fit, all weights 1
        return vec![1.0_f64; n];
    }
    remainder
        .iter()
        .map(|&r| {
            let u = r.abs() / h;
            if u >= 1.0 {
                0.0
            } else {
                let t = 1.0 - u * u;
                t * t
            }
        })
        .collect()
}

#[must_use]
fn sample_variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (v.len() - 1) as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_series(n: usize, period: usize, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|t| amp * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin())
            .collect()
    }

    fn linear_series(n: usize, slope: f64) -> Vec<f64> {
        (0..n).map(|t| t as f64 * slope).collect()
    }

    #[test]
    fn stl_reconstruction_identity() {
        let n = 48;
        let period = 12;
        let seasonal: Vec<f64> = sine_series(n, period, 2.0);
        let trend: Vec<f64> = linear_series(n, 0.1);
        let y: Vec<f64> = seasonal
            .iter()
            .zip(trend.iter())
            .map(|(s, t)| s + t)
            .collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        for (i, (&yi, (&ti, &si))) in y
            .iter()
            .zip(res.trend.iter().zip(res.seasonal.iter()))
            .enumerate()
        {
            let recon = ti + si + res.remainder[i];
            assert!((yi - recon).abs() < 1e-10, "idx={i}: y={yi} recon={recon}");
        }
    }

    #[test]
    fn stl_seasonal_period_preserved() {
        let n = 60;
        let period = 12;
        let y = sine_series(n, period, 3.0);
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        // Adjacent seasonal values spaced one period apart should be close
        for t in 0..n - period {
            let diff = (res.seasonal[t] - res.seasonal[t + period]).abs();
            assert!(diff < 1.5, "seasonal not periodic at t={t}: diff={diff}");
        }
    }

    #[test]
    fn stl_pure_linear_trend() {
        let n = 48;
        let period = 4;
        let y: Vec<f64> = (0..n).map(|t| t as f64 * 0.5).collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        // Interior trend should track the linear ramp closely
        for t in period..n - period {
            let expected = t as f64 * 0.5;
            assert!(
                (res.trend[t] - expected).abs() < 0.5,
                "t={t} trend={} expected≈{expected}",
                res.trend[t]
            );
        }
    }

    #[test]
    fn stl_trend_length() {
        let n = 36;
        let period = 6;
        let y: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        assert_eq!(res.trend.len(), n);
    }

    #[test]
    fn stl_seasonal_length() {
        let n = 36;
        let period = 6;
        let y: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        assert_eq!(res.seasonal.len(), n);
    }

    #[test]
    fn stl_remainder_length() {
        let n = 36;
        let period = 6;
        let y: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        assert_eq!(res.remainder.len(), n);
    }

    #[test]
    fn stl_weights_length() {
        let n = 36;
        let period = 6;
        let y: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        assert_eq!(res.weights.len(), n);
    }

    #[test]
    fn stl_no_outer_weights_all_one() {
        let n = 36;
        let period = 6;
        let y: Vec<f64> = (0..n).map(|t| t as f64).collect();
        let mut cfg = StlConfig::new(period);
        cfg.n_outer = 0;
        let res = stl_decompose(&y, &cfg).expect("ok");
        for &w in &res.weights {
            assert!((w - 1.0).abs() < 1e-12, "weight should be 1.0, got {w}");
        }
    }

    #[test]
    fn stl_outer_runs_without_error() {
        let n = 48;
        let period = 12;
        let y: Vec<f64> = sine_series(n, period, 2.0)
            .iter()
            .zip(linear_series(n, 0.05).iter())
            .map(|(s, t)| s + t)
            .collect();
        let mut cfg = StlConfig::new(period);
        cfg.n_outer = 1;
        stl_decompose(&y, &cfg).expect("n_outer=1 should succeed");
    }

    #[test]
    fn stl_outlier_robustness() {
        let n = 48;
        let period = 12;
        let mut y: Vec<f64> = sine_series(n, period, 1.0)
            .iter()
            .zip(linear_series(n, 0.01).iter())
            .map(|(s, t)| s + t)
            .collect();
        // Inject a large outlier
        y[20] += 100.0;
        let mut cfg = StlConfig::new(period);
        cfg.n_outer = 2;
        let res = stl_decompose(&y, &cfg).expect("ok");
        // The outlier position should receive near-zero robustness weight
        assert!(
            res.weights[20] < 0.3,
            "outlier weight should be small, got {}",
            res.weights[20]
        );
    }

    #[test]
    fn stl_seasonal_strength_near_one_for_pure_sine() {
        let n = 60;
        let period = 12;
        // Pure sine + tiny noise (deterministic)
        let y: Vec<f64> = (0..n)
            .map(|t| {
                3.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin()
                    + (t as f64 * 0.01).sin() * 0.01
            })
            .collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        let strength = stl_seasonal_strength(&res);
        assert!(
            strength > 0.7,
            "seasonal strength should be high for pure sine, got {strength}"
        );
    }

    #[test]
    fn stl_trend_strength_near_one_for_pure_linear() {
        let n = 48;
        let period = 4;
        // Nearly pure linear trend + very small seasonal
        let y: Vec<f64> = (0..n)
            .map(|t| {
                t as f64 * 1.0
                    + 0.001 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin()
            })
            .collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        let strength = stl_trend_strength(&res);
        assert!(
            strength > 0.8,
            "trend strength should be high for pure linear, got {strength}"
        );
    }

    #[test]
    fn stl_mixed_both_strengths_positive() {
        let n = 60;
        let period = 12;
        let y: Vec<f64> = (0..n)
            .map(|t| {
                2.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin() + t as f64 * 0.1
            })
            .collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        assert!(
            stl_seasonal_strength(&res) > 0.5,
            "seasonal strength too low: {}",
            stl_seasonal_strength(&res)
        );
        assert!(
            stl_trend_strength(&res) > 0.5,
            "trend strength too low: {}",
            stl_trend_strength(&res)
        );
    }

    #[test]
    fn stl_naive_forecast_length() {
        let n = 48;
        let period = 12;
        let y: Vec<f64> = sine_series(n, period, 2.0);
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        let h = 24;
        let forecast = stl_naive_forecast(&res, period, h).expect("ok");
        assert_eq!(forecast.len(), h);
    }

    #[test]
    fn stl_naive_forecast_h_zero_returns_empty() {
        let n = 48;
        let period = 12;
        let y: Vec<f64> = sine_series(n, period, 2.0);
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        let forecast = stl_naive_forecast(&res, period, 0).expect("ok");
        assert!(forecast.is_empty());
    }

    #[test]
    fn stl_period_2_simplest() {
        let n = 20;
        let period = 2;
        let y: Vec<f64> = (0..n)
            .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut cfg = StlConfig::new(period);
        cfg.s_window = 3;
        stl_decompose(&y, &cfg).expect("period=2 should succeed");
    }

    #[test]
    fn stl_monthly_48pts() {
        let n = 48;
        let period = 12;
        let y: Vec<f64> = (0..n)
            .map(|t| {
                10.0 * (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin()
                    + t as f64 * 0.2
            })
            .collect();
        let cfg = StlConfig::new(period);
        let res = stl_decompose(&y, &cfg).expect("ok");
        // Reconstruction identity
        for (i, (&yi, (&ti, &si))) in y
            .iter()
            .zip(res.trend.iter().zip(res.seasonal.iter()))
            .enumerate()
        {
            let recon = ti + si + res.remainder[i];
            assert!((yi - recon).abs() < 1e-10, "idx={i}");
        }
    }

    #[test]
    fn stl_large_n200_period7() {
        let n = 200;
        let period = 7;
        let y: Vec<f64> = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * t as f64 / period as f64).sin())
            .collect();
        let cfg = StlConfig::new(period);
        stl_decompose(&y, &cfg).expect("large N=200 period=7 should succeed");
    }

    #[test]
    fn stl_err_too_short() {
        let y = vec![1.0_f64; 5];
        let cfg = StlConfig::new(4);
        assert!(matches!(
            stl_decompose(&y, &cfg).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn stl_err_period_too_small() {
        let y: Vec<f64> = (0..10).map(|t| t as f64).collect();
        let mut cfg = StlConfig::new(2);
        cfg.period = 1;
        assert!(matches!(
            stl_decompose(&y, &cfg).unwrap_err(),
            TsError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn stl_err_s_window_even() {
        let y: Vec<f64> = (0..24).map(|t| t as f64).collect();
        let mut cfg = StlConfig::new(6);
        cfg.s_window = 6;
        assert!(matches!(
            stl_decompose(&y, &cfg).unwrap_err(),
            TsError::InvalidKernelSize(_)
        ));
    }

    #[test]
    fn stl_err_n_inner_zero() {
        let y: Vec<f64> = (0..24).map(|t| t as f64).collect();
        let mut cfg = StlConfig::new(6);
        cfg.n_inner = 0;
        assert!(matches!(
            stl_decompose(&y, &cfg).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }
}
