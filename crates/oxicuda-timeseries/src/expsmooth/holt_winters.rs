//! Holt-Winters triple exponential smoothing.
//!
//! Winters (1960) "Forecasting Sales by Exponentially Weighted Moving Averages."
//! Management Science 6(3):324-342.  Holt (1957 / 2004) "Forecasting seasonals
//! and trends by exponentially weighted moving averages." IJF 20(1):5-10.
//!
//! Triple exponential smoothing maintains three recursively-updated states —
//! **level** `ℓ`, **trend** `b`, and a vector of **seasonal** factors `s` of
//! length `m` (the period) — and combines them to forecast.  Two seasonal
//! formulations are supported:
//!
//! ## Additive seasonality
//! ```text
//! ℓ_t = α (y_t − s_{t−m})        + (1 − α)(ℓ_{t−1} + b_{t−1})
//! b_t = β (ℓ_t − ℓ_{t−1})        + (1 − β) b_{t−1}
//! s_t = γ (y_t − ℓ_t)            + (1 − γ) s_{t−m}
//! ŷ_{t+h} = ℓ_t + h·b_t + s_{t−m + ((h−1) mod m) + 1}
//! ```
//!
//! ## Multiplicative seasonality
//! ```text
//! ℓ_t = α (y_t / s_{t−m})        + (1 − α)(ℓ_{t−1} + b_{t−1})
//! b_t = β (ℓ_t − ℓ_{t−1})        + (1 − β) b_{t−1}
//! s_t = γ (y_t / ℓ_t)            + (1 − γ) s_{t−m}
//! ŷ_{t+h} = (ℓ_t + h·b_t) · s_{t−m + ((h−1) mod m) + 1}
//! ```
//!
//! Optional **damped trend** multiplies the trend contribution by a decay
//! `φ ∈ (0, 1]`: the `h`-step trend term becomes `(φ + φ² + … + φ^h)·b_t`, which
//! flattens long-horizon forecasts (Gardner & McKenzie 1985).  With `φ = 1` the
//! model reduces to the standard additive-trend form.
use crate::error::{TsError, TsResult};

/// Seasonality combination mode for Holt-Winters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seasonality {
    /// Seasonal factors are **added** to the level+trend.
    Additive,
    /// Seasonal factors **multiply** the level+trend (requires strictly positive
    /// data so divisions are well-defined).
    Multiplicative,
}

/// Configuration for [`HoltWinters`].
#[derive(Debug, Clone, Copy)]
pub struct HoltWintersConfig {
    /// Level smoothing `α ∈ [0, 1]`.
    pub alpha: f64,
    /// Trend smoothing `β ∈ [0, 1]`.
    pub beta: f64,
    /// Seasonal smoothing `γ ∈ [0, 1]`.
    pub gamma: f64,
    /// Trend damping `φ ∈ (0, 1]`. Use `1.0` for an undamped trend.
    pub phi: f64,
    /// Seasonal period `m ≥ 2`.
    pub period: usize,
    /// Seasonality combination mode.
    pub seasonality: Seasonality,
}

impl HoltWintersConfig {
    /// Convenience constructor for additive, undamped Holt-Winters.
    #[must_use]
    pub fn additive(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            phi: 1.0,
            period,
            seasonality: Seasonality::Additive,
        }
    }

    /// Convenience constructor for multiplicative, undamped Holt-Winters.
    #[must_use]
    pub fn multiplicative(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            phi: 1.0,
            period,
            seasonality: Seasonality::Multiplicative,
        }
    }

    fn validate(&self) -> TsResult<()> {
        for (name, v) in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
        ] {
            if !(0.0..=1.0).contains(&v) || !v.is_finite() {
                return Err(TsError::Internal(format!(
                    "holt-winters: {name}={v} must be in [0, 1]"
                )));
            }
        }
        if !(self.phi > 0.0 && self.phi <= 1.0 && self.phi.is_finite()) {
            return Err(TsError::Internal(format!(
                "holt-winters: phi={} must be in (0, 1]",
                self.phi
            )));
        }
        if self.period < 2 {
            return Err(TsError::InvalidSequenceLength(self.period));
        }
        Ok(())
    }
}

/// A fitted Holt-Winters model: final smoothing states plus the configuration.
#[derive(Debug, Clone)]
pub struct HoltWinters {
    config: HoltWintersConfig,
    /// Final level state `ℓ_T`.
    level: f64,
    /// Final trend state `b_T`.
    trend: f64,
    /// Final seasonal factors (length = period), aligned so `season[(t) % m]`
    /// applies to the next step in phase.
    season: Vec<f64>,
    /// One-step-ahead in-sample fitted values, length = `n`.
    fitted: Vec<f64>,
    /// In-sample residuals `y_t − ŷ_t`, length = `n`.
    residuals: Vec<f64>,
}

impl HoltWinters {
    /// Final level state.
    #[must_use]
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Final trend state.
    #[must_use]
    pub fn trend(&self) -> f64 {
        self.trend
    }

    /// Final seasonal factors (length = period).
    #[must_use]
    pub fn seasonal(&self) -> &[f64] {
        &self.season
    }

    /// In-sample one-step fitted values.
    #[must_use]
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// In-sample residuals.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// In-sample sum of squared errors (skips the first full season used for
    /// state initialisation, which has no genuine one-step forecast).
    #[must_use]
    pub fn sse(&self) -> f64 {
        let m = self.config.period;
        self.residuals.iter().skip(m).map(|r| r * r).sum()
    }

    /// Fit Holt-Winters to a univariate series `y` (length ≥ `2 · period`).
    ///
    /// States are seeded classically: the initial level is the mean of the first
    /// season, the initial trend is the average per-step change between the
    /// first two seasons, and the initial seasonal factors are the first
    /// season's deviations from (additive) / ratios to (multiplicative) that
    /// level.
    ///
    /// # Errors
    /// - [`TsError::EmptyInput`] when `y` is empty.
    /// - [`TsError::InvalidSequenceLength`] when `y.len() < 2 · period`.
    /// - [`TsError::NonFinite`] when any `y` is non-finite, or (multiplicative)
    ///   any value is `≤ 0`.
    /// - configuration errors from [`HoltWintersConfig`] validation.
    pub fn fit(y: &[f64], config: &HoltWintersConfig) -> TsResult<HoltWinters> {
        config.validate()?;
        let n = y.len();
        if n == 0 {
            return Err(TsError::EmptyInput {
                msg: "holt-winters: empty series".to_string(),
            });
        }
        let m = config.period;
        if n < 2 * m {
            return Err(TsError::InvalidSequenceLength(n));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(TsError::NonFinite);
        }
        if config.seasonality == Seasonality::Multiplicative && y.iter().any(|&v| v <= 0.0) {
            return Err(TsError::NonFinite);
        }

        // ── Initial states ──────────────────────────────────────────────────
        let level0 = y[..m].iter().sum::<f64>() / m as f64;
        // Average per-step trend between season 1 and season 2 means.
        let mean_s1 = level0;
        let mean_s2 = y[m..2 * m].iter().sum::<f64>() / m as f64;
        let trend0 = (mean_s2 - mean_s1) / m as f64;
        let mut season: Vec<f64> = match config.seasonality {
            Seasonality::Additive => y[..m].iter().map(|&v| v - level0).collect(),
            Seasonality::Multiplicative => y[..m]
                .iter()
                .map(|&v| {
                    if level0.abs() > f64::EPSILON {
                        v / level0
                    } else {
                        1.0
                    }
                })
                .collect(),
        };

        let mut level = level0;
        let mut trend = trend0;
        let mut fitted = vec![0.0_f64; n];
        let mut residuals = vec![0.0_f64; n];

        let (alpha, beta, gamma, phi) = (config.alpha, config.beta, config.gamma, config.phi);

        for t in 0..n {
            let s_prev = season[t % m];
            // One-step forecast for time t from the *previous* states.
            let forecast = match config.seasonality {
                Seasonality::Additive => level + phi * trend + s_prev,
                Seasonality::Multiplicative => (level + phi * trend) * s_prev,
            };
            fitted[t] = forecast;
            residuals[t] = y[t] - forecast;

            // ── State recursions ────────────────────────────────────────────
            let prev_level = level;
            match config.seasonality {
                Seasonality::Additive => {
                    level = alpha * (y[t] - s_prev) + (1.0 - alpha) * (prev_level + phi * trend);
                    trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend;
                    season[t % m] = gamma * (y[t] - level) + (1.0 - gamma) * s_prev;
                }
                Seasonality::Multiplicative => {
                    let denom = if s_prev.abs() > f64::EPSILON {
                        s_prev
                    } else {
                        1.0
                    };
                    level = alpha * (y[t] / denom) + (1.0 - alpha) * (prev_level + phi * trend);
                    trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend;
                    let l = if level.abs() > f64::EPSILON {
                        level
                    } else {
                        1.0
                    };
                    season[t % m] = gamma * (y[t] / l) + (1.0 - gamma) * s_prev;
                }
            }
        }

        // Rotate `season` so that index `0` corresponds to the factor that
        // applies at time `n` (i.e. the next forecast step is phase `n % m`).
        Ok(HoltWinters {
            config: *config,
            level,
            trend,
            season,
            fitted,
            residuals,
        })
    }

    /// Forecast the next `h` steps beyond the end of the fitted series.
    ///
    /// Applies damped-trend accumulation `Σ_{i=1}^{h} φ^i · b` and wraps the
    /// seasonal factors with the correct phase (the next step after a series of
    /// length `n` is seasonal phase `n % m`).
    ///
    /// # Errors
    /// Returns an empty vector for `h == 0` (no error).
    pub fn forecast(&self, h: usize) -> TsResult<Vec<f64>> {
        if h == 0 {
            return Ok(Vec::new());
        }
        let m = self.config.period;
        let phase = self.fitted.len() % m;
        let mut out = Vec::with_capacity(h);
        // Cumulative damped-trend multiplier.
        let mut phi_pow = 1.0_f64;
        let mut damp_sum = 0.0_f64;
        for step in 1..=h {
            phi_pow *= self.config.phi;
            // Σ φ^i, i = 1..=step. For φ = 1 this is exactly `step`.
            damp_sum += phi_pow;
            let trend_term = damp_sum * self.trend;
            let s = self.season[(phase + step - 1) % m];
            let value = match self.config.seasonality {
                Seasonality::Additive => self.level + trend_term + s,
                Seasonality::Multiplicative => (self.level + trend_term) * s,
            };
            out.push(value);
        }
        Ok(out)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seasonal_series(n: usize, period: usize, level: f64, slope: f64, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|t| {
                level
                    + slope * t as f64
                    + amp * (2.0 * std::f64::consts::PI * (t % period) as f64 / period as f64).sin()
            })
            .collect()
    }

    #[test]
    fn hw_fit_additive_shapes() {
        let m = 4;
        let y = seasonal_series(40, m, 10.0, 0.0, 2.0);
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        assert_eq!(model.fitted().len(), 40);
        assert_eq!(model.residuals().len(), 40);
        assert_eq!(model.seasonal().len(), m);
    }

    #[test]
    fn hw_forecast_length() {
        let m = 4;
        let y = seasonal_series(40, m, 10.0, 0.5, 2.0);
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(12).expect("forecast should succeed");
        assert_eq!(fc.len(), 12);
        assert!(fc.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hw_forecast_zero_empty() {
        let m = 4;
        let y = seasonal_series(40, m, 10.0, 0.0, 2.0);
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        assert!(
            model
                .forecast(0)
                .expect("forecast should succeed")
                .is_empty()
        );
    }

    #[test]
    fn hw_additive_tracks_linear_trend() {
        // Pure linear trend, no seasonality amplitude: forecasts should extend
        // the ramp.
        let m = 4;
        let slope = 1.0;
        let y = seasonal_series(60, m, 5.0, slope, 0.0);
        let cfg = HoltWintersConfig::additive(0.5, 0.3, 0.1, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(4).expect("forecast should succeed");
        // Next 4 values should roughly continue +slope per step.
        let last = y[y.len() - 1];
        for (i, &v) in fc.iter().enumerate() {
            let expected = last + slope * (i + 1) as f64;
            assert!(
                (v - expected).abs() < 2.0,
                "step {i}: got {v} expected≈{expected}"
            );
        }
    }

    #[test]
    fn hw_additive_recovers_seasonality() {
        // The seasonal forecast should preserve the periodic pattern.
        let m = 4;
        let amp = 5.0;
        let y = seasonal_series(48, m, 20.0, 0.0, amp);
        let cfg = HoltWintersConfig::additive(0.2, 0.05, 0.3, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(2 * m).expect("forecast should succeed");
        // Values one period apart should be close.
        for t in 0..m {
            assert!(
                (fc[t] - fc[t + m]).abs() < amp,
                "forecast not periodic at t={t}"
            );
        }
    }

    #[test]
    fn hw_multiplicative_positive_forecasts() {
        let m = 4;
        let y: Vec<f64> = (0..48)
            .map(|t| {
                (10.0 + 0.2 * t as f64)
                    * (1.0 + 0.3 * (2.0 * std::f64::consts::PI * (t % m) as f64 / m as f64).sin())
            })
            .collect();
        let cfg = HoltWintersConfig::multiplicative(0.3, 0.1, 0.2, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(8).expect("forecast should succeed");
        assert!(
            fc.iter().all(|&v| v > 0.0),
            "multiplicative forecasts must be positive"
        );
    }

    #[test]
    fn hw_damped_trend_flattens() {
        // With φ < 1 the long-horizon forecast should be lower than the undamped
        // one for a positive trend.
        let m = 4;
        let y = seasonal_series(60, m, 5.0, 1.0, 1.0);
        let undamped = HoltWintersConfig::additive(0.5, 0.3, 0.1, m);
        let mut damped = undamped;
        damped.phi = 0.7;
        let mu = HoltWinters::fit(&y, &undamped).expect("fit should succeed");
        let md = HoltWinters::fit(&y, &damped).expect("fit should succeed");
        let fu = mu.forecast(20).expect("forecast should succeed");
        let fd = md.forecast(20).expect("forecast should succeed");
        // Far-horizon damped forecast must be below undamped (trend was positive).
        assert!(
            fd[19] < fu[19],
            "damped {} not below undamped {}",
            fd[19],
            fu[19]
        );
    }

    #[test]
    fn hw_residuals_small_for_clean_signal() {
        // A clean deterministic seasonal+trend should fit with small residuals.
        let m = 6;
        let y = seasonal_series(60, m, 10.0, 0.3, 3.0);
        let cfg = HoltWintersConfig::additive(0.4, 0.1, 0.3, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        let var_y = {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64
        };
        // SSE per point (after warm-up) should be far below the series variance.
        let mse = model.sse() / (y.len() - m) as f64;
        assert!(mse < 0.5 * var_y, "mse={mse} var_y={var_y}");
    }

    #[test]
    fn hw_fitted_finite() {
        let m = 4;
        let y = seasonal_series(40, m, 10.0, 0.2, 2.0);
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        assert!(model.fitted().iter().all(|v| v.is_finite()));
        assert!(model.residuals().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn hw_alpha_one_follows_data() {
        // With α=1, β=0, γ=0 the level chases the deseasonalised observation.
        let m = 4;
        let y = seasonal_series(40, m, 10.0, 0.0, 2.0);
        let cfg = HoltWintersConfig::additive(1.0, 0.0, 0.0, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        assert!(model.level().is_finite());
    }

    #[test]
    fn hw_constant_series() {
        // A flat constant series → flat forecasts.
        let m = 3;
        let y = vec![7.0_f64; 30];
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        let model = HoltWinters::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(6).expect("forecast should succeed");
        for &v in &fc {
            assert!((v - 7.0).abs() < 1e-6, "constant forecast wrong: {v}");
        }
    }

    #[test]
    fn hw_err_too_short() {
        let m = 4;
        let y = vec![1.0_f64; 5]; // < 2*m
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        assert!(matches!(
            HoltWinters::fit(&y, &cfg).unwrap_err(),
            TsError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn hw_err_bad_alpha() {
        let m = 4;
        let y = vec![1.0_f64; 20];
        let cfg = HoltWintersConfig::additive(1.5, 0.1, 0.2, m);
        assert!(matches!(
            HoltWinters::fit(&y, &cfg).unwrap_err(),
            TsError::Internal(_)
        ));
    }

    #[test]
    fn hw_err_period_too_small() {
        let y = vec![1.0_f64; 20];
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, 1);
        assert!(matches!(
            HoltWinters::fit(&y, &cfg).unwrap_err(),
            TsError::InvalidSequenceLength(_)
        ));
    }

    #[test]
    fn hw_err_multiplicative_nonpositive() {
        let m = 4;
        let y: Vec<f64> = (0..20).map(|t| t as f64 - 10.0).collect(); // has ≤ 0
        let cfg = HoltWintersConfig::multiplicative(0.3, 0.1, 0.2, m);
        assert!(matches!(
            HoltWinters::fit(&y, &cfg).unwrap_err(),
            TsError::NonFinite
        ));
    }

    #[test]
    fn hw_err_nonfinite_input() {
        let m = 4;
        let mut y = vec![1.0_f64; 20];
        y[5] = f64::NAN;
        let cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        assert!(matches!(
            HoltWinters::fit(&y, &cfg).unwrap_err(),
            TsError::NonFinite
        ));
    }

    #[test]
    fn hw_err_bad_phi() {
        let m = 4;
        let y = vec![1.0_f64; 20];
        let mut cfg = HoltWintersConfig::additive(0.3, 0.1, 0.2, m);
        cfg.phi = 1.5;
        assert!(matches!(
            HoltWinters::fit(&y, &cfg).unwrap_err(),
            TsError::Internal(_)
        ));
    }
}
