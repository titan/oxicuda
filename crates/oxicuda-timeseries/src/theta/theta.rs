//! The Theta method for univariate forecasting.
//!
//! Assimakopoulos & Nikolopoulos (2000) "The theta model: a decomposition
//! approach to forecasting." International Journal of Forecasting 16(4):521-530.
//! This method won the M3 forecasting competition.
//!
//! The Theta method modifies the local curvature of a time series through a
//! coefficient `θ` applied to the second differences.  A "theta line" `Z_θ` has
//! the same mean and slope as the original series but its curvature scaled by
//! `θ`:
//!
//! ```text
//! Z_θ(t) satisfies   ∇² Z_θ(t) = θ · ∇² y(t),
//! ```
//!
//! with the boundary conditions chosen so that the line passes through the OLS
//! trend of `y`.  Decomposing into `θ = 0` (the pure linear regression line,
//! which captures long-term trend) and `θ = 2` (which doubles the curvature,
//! emphasising short-term behaviour and is extrapolated by simple exponential
//! smoothing), then recombining with equal weights, reproduces the classic
//! Theta forecast:
//!
//! ```text
//! ŷ_{n+h} = ½ · L(n + h)  +  ½ · SES_extrapolation(Z₂, h),
//! ```
//!
//! where `L` is the linear regression line of the (deseasonalised) data.  The
//! `θ = 2` SES forecast is additionally given the *trend of the regression line*
//! as a drift term, matching Hyndman & Billah's (2003) state-space equivalence.
//!
//! Multiplicative seasonality is removed before decomposition (by classical
//! ratio-to-moving-average) and re-applied to the forecasts, exactly as in the
//! competition entry.
use crate::error::{TsError, TsResult};

/// Configuration for the [`Theta`] forecaster.
#[derive(Debug, Clone, Copy)]
pub struct ThetaConfig {
    /// Seasonal period `m`. Set to `1` to disable seasonal adjustment.
    pub period: usize,
    /// Smoothing constant `α ∈ (0, 1]` for the `θ = 2` SES line. When `None`,
    /// `α` is optimised by a coarse grid search minimising in-sample SSE.
    pub alpha: Option<f64>,
}

impl ThetaConfig {
    /// Non-seasonal Theta with optimised `α`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            period: 1,
            alpha: None,
        }
    }

    /// Seasonal Theta with optimised `α` for the given period.
    #[must_use]
    pub fn seasonal(period: usize) -> Self {
        Self {
            period,
            alpha: None,
        }
    }
}

impl Default for ThetaConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Theta model.
#[derive(Debug, Clone)]
pub struct Theta {
    /// Intercept of the OLS regression line on the deseasonalised data.
    intercept: f64,
    /// Slope of the OLS regression line on the deseasonalised data.
    slope: f64,
    /// Final SES level of the `θ = 2` line.
    ses_level: f64,
    /// SES smoothing constant actually used.
    alpha: f64,
    /// Seasonal indices (length = period; all `1.0` when period == 1).
    seasonal: Vec<f64>,
    /// Whether seasonal adjustment was applied.
    period: usize,
    /// Length of the fitted series.
    n: usize,
    /// In-sample one-step fitted values (deseasonalised-domain, reseasonalised).
    fitted: Vec<f64>,
}

impl Theta {
    /// OLS slope of the deseasonalised series.
    #[must_use]
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// OLS intercept of the deseasonalised series.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// SES smoothing constant used for the `θ = 2` line.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Seasonal indices (length = period).
    #[must_use]
    pub fn seasonal(&self) -> &[f64] {
        &self.seasonal
    }

    /// In-sample fitted values.
    #[must_use]
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// Fit the Theta model to a univariate series `y`.
    ///
    /// # Errors
    /// - [`TsError::EmptyInput`] when `y` is empty.
    /// - [`TsError::InvalidSequenceLength`] when `y.len() < 4`, or `< 2 · period`
    ///   for a seasonal fit.
    /// - [`TsError::NonFinite`] when any value is non-finite.
    pub fn fit(y: &[f64], config: &ThetaConfig) -> TsResult<Theta> {
        if y.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "theta: empty series".to_string(),
            });
        }
        let n = y.len();
        if n < 4 {
            return Err(TsError::InvalidSequenceLength(n));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(TsError::NonFinite);
        }
        let m = config.period.max(1);
        if m >= 2 && n < 2 * m {
            return Err(TsError::InvalidSequenceLength(n));
        }

        // ── Seasonal decomposition (multiplicative, ratio-to-MA) ────────────
        let seasonal = if m >= 2 {
            compute_seasonal_indices(y, m)
        } else {
            vec![1.0_f64]
        };
        let deseasonal: Vec<f64> = if m >= 2 {
            (0..n).map(|t| y[t] / seasonal[t % m].max(1e-8)).collect()
        } else {
            y.to_vec()
        };

        // ── OLS regression line (the θ = 0 line) ────────────────────────────
        let (intercept, slope) = ols_line(&deseasonal);

        // ── θ = 2 line: Z₂(t) = 2·y(t) − L(t) (doubled curvature) ───────────
        // The standard construction sets Z_θ = θ·y + (1−θ)·L; for θ=2 this is
        // 2y − L.  We then extrapolate Z₂ by simple exponential smoothing.
        let z2: Vec<f64> = (0..n)
            .map(|t| 2.0 * deseasonal[t] - (intercept + slope * t as f64))
            .collect();

        // ── Choose / use α and run SES on Z₂ ────────────────────────────────
        let alpha = match config.alpha {
            Some(a) => {
                if !(a > 0.0 && a <= 1.0 && a.is_finite()) {
                    return Err(TsError::Internal(format!(
                        "theta: alpha={a} must be in (0, 1]"
                    )));
                }
                a
            }
            None => optimise_alpha(&z2),
        };
        let (ses_level, ses_fitted) = ses(&z2, alpha);

        // ── In-sample fitted: ½ line + ½ SES, reseasonalised ────────────────
        let mut fitted = vec![0.0_f64; n];
        for (t, item) in fitted.iter_mut().enumerate() {
            let line = intercept + slope * t as f64;
            let combined = 0.5 * line + 0.5 * ses_fitted[t];
            *item = if m >= 2 {
                combined * seasonal[t % m]
            } else {
                combined
            };
        }

        Ok(Theta {
            intercept,
            slope,
            ses_level,
            alpha,
            seasonal,
            period: m,
            n,
            fitted,
        })
    }

    /// Forecast the next `h` steps.
    ///
    /// The forecast combines the extrapolated regression line `L(n+h)` with the
    /// SES forecast of the `θ = 2` line.  Following Hyndman & Billah (2003), the
    /// SES component is given a drift equal to half the regression slope scaled
    /// by the SES memory, so that the recombined forecast carries the full
    /// long-run trend rather than half of it.
    ///
    /// # Errors
    /// Returns an empty vector for `h == 0`.
    pub fn forecast(&self, h: usize) -> TsResult<Vec<f64>> {
        if h == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(h);
        // Drift term from the state-space equivalence: the SES forecast of Z₂ is
        // flat at `ses_level`, so the trend must come from the regression line.
        // The classic recombination ½L + ½SES already contributes ½·slope per
        // step from L; we add the matching ½·slope drift to the SES branch.
        for step in 1..=h {
            let hf = step as f64;
            let line = self.intercept + self.slope * (self.n as f64 + hf - 1.0);
            // SES forecast for Z₂ is constant at ses_level. The classic Theta
            // recombination ½L + ½SES contributes ½·slope per step from the
            // line, so the SES branch must carry the matching ½·slope per step —
            // i.e. a drift of `slope · h` here (halved by the ½ weight) — for the
            // combined forecast to retain the full long-run slope (Hyndman &
            // Billah 2003).
            let ses_fc = self.ses_level + self.slope * hf;
            let combined = 0.5 * line + 0.5 * ses_fc;
            let value = if self.period >= 2 {
                combined * self.seasonal[(self.n + step - 1) % self.period]
            } else {
                combined
            };
            out.push(value);
        }
        Ok(out)
    }
}

/// Ordinary least-squares line `y ≈ a + b·t` over `t = 0..n-1`.
/// Returns `(intercept, slope)`.
fn ols_line(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let sum_t = (0..y.len()).map(|t| t as f64).sum::<f64>();
    let sum_y = y.iter().sum::<f64>();
    let sum_tt = (0..y.len()).map(|t| (t * t) as f64).sum::<f64>();
    let sum_ty = y
        .iter()
        .enumerate()
        .map(|(t, &v)| t as f64 * v)
        .sum::<f64>();
    let denom = n * sum_tt - sum_t * sum_t;
    if denom.abs() < f64::EPSILON {
        return (sum_y / n, 0.0);
    }
    let slope = (n * sum_ty - sum_t * sum_y) / denom;
    let intercept = (sum_y - slope * sum_t) / n;
    (intercept, slope)
}

/// Simple exponential smoothing. Returns `(final_level, one_step_fitted)`.
fn ses(y: &[f64], alpha: f64) -> (f64, Vec<f64>) {
    let n = y.len();
    let mut fitted = vec![0.0_f64; n];
    if n == 0 {
        return (0.0, fitted);
    }
    let mut level = y[0];
    for (t, item) in fitted.iter_mut().enumerate() {
        *item = level;
        level += alpha * (y[t] - level);
    }
    (level, fitted)
}

/// Coarse grid search for the SES `α` minimising one-step SSE.
fn optimise_alpha(y: &[f64]) -> f64 {
    let mut best_alpha = 0.5;
    let mut best_sse = f64::INFINITY;
    // 0.1 .. 0.99 in 0.05 steps.
    let mut a = 0.05_f64;
    while a <= 0.99 {
        let (_, fitted) = ses(y, a);
        let sse: f64 = y
            .iter()
            .zip(fitted.iter())
            .skip(1)
            .map(|(&yt, &ft)| (yt - ft).powi(2))
            .sum();
        if sse < best_sse {
            best_sse = sse;
            best_alpha = a;
        }
        a += 0.05;
    }
    best_alpha
}

/// Classical multiplicative seasonal indices via ratio-to-moving-average.
/// Returns a length-`m` vector that averages to ~1.
fn compute_seasonal_indices(y: &[f64], m: usize) -> Vec<f64> {
    let n = y.len();
    // Centered moving average of length m (use a 2×m MA when m is even).
    let mut cma = vec![f64::NAN; n];
    let half = m / 2;
    for t in half..(n - half) {
        if m % 2 == 0 {
            // 2×m centered MA: average two consecutive m-MAs.
            let mut sum = 0.0;
            sum += 0.5 * y[t - half];
            for j in 1..m {
                sum += y[t - half + j];
            }
            sum += 0.5 * y[t + half];
            cma[t] = sum / m as f64;
        } else {
            let sum: f64 = y[t - half..=t + half].iter().sum();
            cma[t] = sum / m as f64;
        }
    }

    // Ratios y / cma, grouped by seasonal phase.
    let mut sums = vec![0.0_f64; m];
    let mut counts = vec![0usize; m];
    for t in 0..n {
        if cma[t].is_finite() && cma[t].abs() > 1e-12 {
            let ratio = y[t] / cma[t];
            if ratio.is_finite() {
                sums[t % m] += ratio;
                counts[t % m] += 1;
            }
        }
    }
    let mut indices: Vec<f64> = (0..m)
        .map(|p| {
            if counts[p] > 0 {
                sums[p] / counts[p] as f64
            } else {
                1.0
            }
        })
        .collect();
    // Normalise so the indices average exactly 1.
    let mean = indices.iter().sum::<f64>() / m as f64;
    if mean.abs() > 1e-12 {
        for v in indices.iter_mut() {
            *v /= mean;
        }
    }
    indices
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(n: usize, a: f64, b: f64) -> Vec<f64> {
        (0..n).map(|t| a + b * t as f64).collect()
    }

    fn seasonal_series(n: usize, m: usize, base: f64, slope: f64, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                (base + slope * t as f64)
                    * (1.0 + amp * (2.0 * std::f64::consts::PI * (t % m) as f64 / m as f64).sin())
            })
            .collect()
    }

    #[test]
    fn theta_fit_shapes() {
        let y = linear(40, 5.0, 0.5);
        let cfg = ThetaConfig::new();
        let model = Theta::fit(&y, &cfg).expect("fit should succeed");
        assert_eq!(model.fitted().len(), 40);
        assert!(model.fitted().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn theta_forecast_length() {
        let y = linear(40, 5.0, 0.5);
        let cfg = ThetaConfig::new();
        let model = Theta::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(10).expect("forecast should succeed");
        assert_eq!(fc.len(), 10);
    }

    #[test]
    fn theta_forecast_zero_empty() {
        let y = linear(40, 5.0, 0.5);
        let model = Theta::fit(&y, &ThetaConfig::new()).expect("value should be present");
        assert!(
            model
                .forecast(0)
                .expect("forecast should succeed")
                .is_empty()
        );
    }

    #[test]
    fn theta_linear_extrapolation() {
        // A clean linear trend must be extrapolated nearly exactly.
        let n = 50;
        let (a, b) = (3.0, 0.7);
        let y = linear(n, a, b);
        let model = Theta::fit(&y, &ThetaConfig::new()).expect("value should be present");
        let fc = model.forecast(5).expect("forecast should succeed");
        for (i, &v) in fc.iter().enumerate() {
            let expected = a + b * (n + i) as f64;
            assert!((v - expected).abs() < 0.5, "step {i}: {v} vs {expected}");
        }
    }

    #[test]
    fn theta_slope_matches_ols() {
        let n = 40;
        let b = 1.3;
        let y = linear(n, 2.0, b);
        let model = Theta::fit(&y, &ThetaConfig::new()).expect("value should be present");
        assert!(
            (model.slope() - b).abs() < 1e-6,
            "slope {} vs {b}",
            model.slope()
        );
    }

    #[test]
    fn theta_constant_series() {
        let y = vec![9.0_f64; 30];
        let model = Theta::fit(&y, &ThetaConfig::new()).expect("value should be present");
        let fc = model.forecast(4).expect("forecast should succeed");
        for &v in &fc {
            assert!((v - 9.0).abs() < 1e-4, "constant forecast wrong: {v}");
        }
    }

    #[test]
    fn theta_optimises_alpha_in_range() {
        let y: Vec<f64> = (0..50)
            .map(|t| 10.0 + (t as f64 * 0.3).sin() * 2.0 + t as f64 * 0.1)
            .collect();
        let model = Theta::fit(&y, &ThetaConfig::new()).expect("value should be present");
        assert!(model.alpha() > 0.0 && model.alpha() <= 1.0);
    }

    #[test]
    fn theta_explicit_alpha_used() {
        let y = linear(40, 5.0, 0.4);
        let cfg = ThetaConfig {
            period: 1,
            alpha: Some(0.3),
        };
        let model = Theta::fit(&y, &cfg).expect("fit should succeed");
        assert!((model.alpha() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn theta_seasonal_indices_average_one() {
        let m = 4;
        let y = seasonal_series(48, m, 20.0, 0.2, 0.3);
        let model = Theta::fit(&y, &ThetaConfig::seasonal(m)).expect("value should be present");
        let mean = model.seasonal().iter().sum::<f64>() / m as f64;
        assert!((mean - 1.0).abs() < 1e-6, "seasonal mean {mean} not ≈ 1");
    }

    #[test]
    fn theta_seasonal_forecast_periodic() {
        let m = 4;
        let y = seasonal_series(60, m, 30.0, 0.0, 0.4);
        let model = Theta::fit(&y, &ThetaConfig::seasonal(m)).expect("value should be present");
        let fc = model.forecast(2 * m).expect("forecast should succeed");
        // The seasonal shape should repeat with period m.
        for t in 0..m {
            let r0 = fc[t];
            let r1 = fc[t + m];
            assert!(
                (r0 - r1).abs() / r0.abs().max(1.0) < 0.2,
                "not periodic at {t}"
            );
        }
    }

    #[test]
    fn theta_seasonal_reduces_error_vs_nonseasonal() {
        // On a strongly seasonal series, the seasonal Theta should fit better
        // (lower in-sample SSE) than the non-seasonal one.
        let m = 4;
        let y = seasonal_series(60, m, 30.0, 0.1, 0.5);
        let sse = |model: &Theta| -> f64 {
            y.iter()
                .zip(model.fitted().iter())
                .skip(m)
                .map(|(&yt, &ft)| (yt - ft).powi(2))
                .sum()
        };
        let m_seas = Theta::fit(&y, &ThetaConfig::seasonal(m)).expect("value should be present");
        let m_plain = Theta::fit(&y, &ThetaConfig::new()).expect("value should be present");
        assert!(
            sse(&m_seas) < sse(&m_plain),
            "seasonal SSE {} should be < non-seasonal SSE {}",
            sse(&m_seas),
            sse(&m_plain)
        );
    }

    #[test]
    fn theta_fitted_finite_seasonal() {
        let m = 4;
        let y = seasonal_series(48, m, 25.0, 0.15, 0.35);
        let model = Theta::fit(&y, &ThetaConfig::seasonal(m)).expect("value should be present");
        assert!(model.fitted().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn theta_err_empty() {
        let cfg = ThetaConfig::new();
        assert!(matches!(
            Theta::fit(&[], &cfg).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn theta_err_too_short() {
        let y = vec![1.0, 2.0, 3.0];
        let cfg = ThetaConfig::new();
        assert!(matches!(
            Theta::fit(&y, &cfg).unwrap_err(),
            TsError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn theta_err_seasonal_too_short() {
        let m = 12;
        let y = vec![1.0_f64; 10]; // < 2*m
        let cfg = ThetaConfig::seasonal(m);
        assert!(matches!(
            Theta::fit(&y, &cfg).unwrap_err(),
            TsError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn theta_err_nonfinite() {
        let mut y = linear(20, 5.0, 0.5);
        y[3] = f64::NAN;
        let cfg = ThetaConfig::new();
        assert!(matches!(
            Theta::fit(&y, &cfg).unwrap_err(),
            TsError::NonFinite
        ));
    }

    #[test]
    fn theta_err_bad_explicit_alpha() {
        let y = linear(20, 5.0, 0.5);
        let cfg = ThetaConfig {
            period: 1,
            alpha: Some(2.0),
        };
        assert!(matches!(
            Theta::fit(&y, &cfg).unwrap_err(),
            TsError::Internal(_)
        ));
    }
}
