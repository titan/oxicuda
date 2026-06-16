//! Croston's method for intermittent demand forecasting.
//!
//! Croston (1972) "Forecasting and Stock Control for Intermittent Demands."
//! Operational Research Quarterly 23(3):289-303.
//!
//! Intermittent demand series contain many zero periods interspersed with
//! occasional non-zero demands.  Applying ordinary exponential smoothing to such
//! a series is biased and reacts sluggishly.  Croston's insight is to decompose
//! the series into **two** separate exponentially-smoothed processes:
//!
//!   * `z` — the size of non-zero demands, and
//!   * `p` — the inter-arrival interval between consecutive non-zero demands.
//!
//! The forecast of demand *per period* is then `ẑ / p̂`.  Both `z` and `p` are
//! updated **only when a non-zero demand occurs**, using a common smoothing
//! constant `α`.
//!
//! ## Variants
//!
//! - [`CrostonMethod::Classic`] — the original estimator `ẑ / p̂`.  Known to be
//!   positively biased.
//! - [`CrostonMethod::Sba`] — Syntetos-Boylan Approximation (2005), which
//!   multiplies the classic forecast by `(1 − α/2)` to remove most of the bias.
//! - [`CrostonMethod::Tsb`] — Teunter-Syntetos-Babai (2011), which replaces the
//!   interval process with a **demand-probability** process updated *every*
//!   period (including zeros), making it suitable for series whose demand can go
//!   obsolete (probability decays toward zero during long zero runs).
use crate::error::{TsError, TsResult};

/// Which Croston-family estimator to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrostonMethod {
    /// Original Croston (1972): forecast = `ẑ / p̂`.
    Classic,
    /// Syntetos-Boylan Approximation (2005): `(1 − α/2) · ẑ / p̂`.
    Sba,
    /// Teunter-Syntetos-Babai (2011): probability-based, `ẑ · prob̂`.
    Tsb,
}

/// Configuration for a Croston forecaster.
#[derive(Debug, Clone, Copy)]
pub struct CrostonConfig {
    /// Smoothing constant for the demand-size process `α ∈ (0, 1]`.
    pub alpha: f64,
    /// Smoothing constant for the interval / probability process `β ∈ (0, 1]`.
    /// For the classic and SBA methods this is shared with `α` in the original
    /// formulation; we expose it separately so TSB can use a distinct value.
    pub beta: f64,
    /// Estimator variant.
    pub method: CrostonMethod,
}

impl CrostonConfig {
    /// Classic Croston with a single smoothing constant.
    #[must_use]
    pub fn classic(alpha: f64) -> Self {
        Self {
            alpha,
            beta: alpha,
            method: CrostonMethod::Classic,
        }
    }

    /// SBA (bias-corrected Croston) with a single smoothing constant.
    #[must_use]
    pub fn sba(alpha: f64) -> Self {
        Self {
            alpha,
            beta: alpha,
            method: CrostonMethod::Sba,
        }
    }

    /// TSB with separate demand-size (`alpha`) and probability (`beta`) rates.
    #[must_use]
    pub fn tsb(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            method: CrostonMethod::Tsb,
        }
    }

    fn validate(&self) -> TsResult<()> {
        for (name, v) in [("alpha", self.alpha), ("beta", self.beta)] {
            if !(v > 0.0 && v <= 1.0 && v.is_finite()) {
                return Err(TsError::Internal(format!(
                    "croston: {name}={v} must be in (0, 1]"
                )));
            }
        }
        Ok(())
    }
}

/// A fitted Croston forecaster.
#[derive(Debug, Clone)]
pub struct Croston {
    config: CrostonConfig,
    /// Smoothed demand size `ẑ`.
    demand: f64,
    /// Smoothed inter-arrival interval `p̂` (classic / SBA).
    interval: f64,
    /// Smoothed demand probability (TSB).
    probability: f64,
    /// Per-period forecast at the end of the series.
    forecast_rate: f64,
    /// In-sample per-period forecasts aligned to each input index, length `n`.
    fitted: Vec<f64>,
}

impl Croston {
    /// Smoothed non-zero demand size `ẑ`.
    #[must_use]
    pub fn demand_size(&self) -> f64 {
        self.demand
    }

    /// Smoothed inter-arrival interval `p̂` (classic / SBA; `1` for TSB).
    #[must_use]
    pub fn interval(&self) -> f64 {
        self.interval
    }

    /// Smoothed demand probability (TSB; `1/p̂` analogue otherwise).
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// In-sample one-step per-period forecast series.
    #[must_use]
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// Fit the Croston model to an intermittent-demand series `y`.
    ///
    /// All values must be `≥ 0` (demand quantities).  The demand-size and
    /// interval states are seeded from the **first** non-zero demand and the
    /// interval up to it; if the series is all zeros the forecast is `0`.
    ///
    /// # Errors
    /// - [`TsError::EmptyInput`] when `y` is empty.
    /// - [`TsError::NonFinite`] when any value is non-finite or negative.
    /// - configuration errors from [`CrostonConfig`] validation.
    pub fn fit(y: &[f64], config: &CrostonConfig) -> TsResult<Croston> {
        config.validate()?;
        if y.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "croston: empty series".to_string(),
            });
        }
        if y.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(TsError::NonFinite);
        }

        match config.method {
            CrostonMethod::Classic | CrostonMethod::Sba => Self::fit_classic_like(y, config),
            CrostonMethod::Tsb => Self::fit_tsb(y, config),
        }
    }

    fn fit_classic_like(y: &[f64], config: &CrostonConfig) -> TsResult<Croston> {
        let n = y.len();
        let alpha = config.alpha;
        let beta = config.beta;

        // Seed from the first non-zero demand.
        let first_nz = y.iter().position(|&v| v > 0.0);
        let mut fitted = vec![0.0_f64; n];

        let Some(first_idx) = first_nz else {
            // All-zero series: nothing to forecast but states are well-defined.
            return Ok(Croston {
                config: *config,
                demand: 0.0,
                interval: 1.0,
                probability: 0.0,
                forecast_rate: 0.0,
                fitted,
            });
        };

        // Initial demand = first non-zero value; initial interval = gap (≥ 1).
        let mut demand = y[first_idx];
        let mut interval = (first_idx + 1) as f64;
        let mut gap = 0.0_f64; // periods since last non-zero demand
        let bias = match config.method {
            CrostonMethod::Sba => 1.0 - alpha / 2.0,
            _ => 1.0,
        };

        let mut rate = bias * demand / interval.max(f64::EPSILON);
        for (t, item) in fitted.iter_mut().enumerate() {
            // The per-period forecast is held constant between demand epochs.
            *item = rate;
            gap += 1.0;
            if y[t] > 0.0 {
                if t == first_idx {
                    // States already seeded from this demand; just reset the gap.
                    gap = 0.0;
                } else {
                    demand += alpha * (y[t] - demand);
                    interval += beta * (gap - interval);
                    gap = 0.0;
                }
                rate = bias * demand / interval.max(f64::EPSILON);
            }
        }

        Ok(Croston {
            config: *config,
            demand,
            interval,
            probability: 1.0 / interval.max(f64::EPSILON),
            forecast_rate: rate,
            fitted,
        })
    }

    fn fit_tsb(y: &[f64], config: &CrostonConfig) -> TsResult<Croston> {
        let n = y.len();
        let alpha = config.alpha; // demand-size rate
        let beta = config.beta; // probability rate

        let mut fitted = vec![0.0_f64; n];
        // Seed: probability = fraction of non-zero periods; demand = mean of
        // non-zero demands (or 0 if none).
        let nz: Vec<f64> = y.iter().copied().filter(|&v| v > 0.0).collect();
        let mut demand = if nz.is_empty() {
            0.0
        } else {
            nz.iter().sum::<f64>() / nz.len() as f64
        };
        let mut probability = nz.len() as f64 / n as f64;

        let mut rate = demand * probability;
        for (t, item) in fitted.iter_mut().enumerate() {
            *item = rate;
            // TSB updates probability EVERY period.
            if y[t] > 0.0 {
                probability += beta * (1.0 - probability);
                demand += alpha * (y[t] - demand);
            } else {
                probability += beta * (0.0 - probability);
            }
            rate = demand * probability;
        }

        Ok(Croston {
            config: *config,
            demand,
            interval: if probability > f64::EPSILON {
                1.0 / probability
            } else {
                f64::INFINITY
            },
            probability,
            forecast_rate: rate,
            fitted,
        })
    }

    /// Forecast the next `h` periods.  Croston forecasts are **flat**: every
    /// future period takes the same per-period rate `ẑ / p̂` (or `ẑ · prob̂`).
    ///
    /// # Errors
    /// Returns an empty vector for `h == 0`.
    pub fn forecast(&self, h: usize) -> TsResult<Vec<f64>> {
        Ok(vec![self.forecast_rate; h])
    }

    /// The single constant per-period forecast rate.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.forecast_rate
    }

    /// The estimator variant this model was fitted with.
    #[must_use]
    pub fn method(&self) -> CrostonMethod {
        self.config.method
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn intermittent(n: usize, every: usize, size: f64) -> Vec<f64> {
        (0..n)
            .map(|t| if t % every == every - 1 { size } else { 0.0 })
            .collect()
    }

    #[test]
    fn croston_classic_shapes() {
        let y = intermittent(40, 4, 10.0);
        let cfg = CrostonConfig::classic(0.2);
        let model = Croston::fit(&y, &cfg).expect("fit should succeed");
        assert_eq!(model.fitted().len(), 40);
        assert!(model.fitted().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn croston_classic_rate_matches_intensity() {
        // Demand of 10 every 4 periods ⇒ per-period rate ≈ 10/4 = 2.5.
        let y = intermittent(80, 4, 10.0);
        let cfg = CrostonConfig::classic(0.1);
        let model = Croston::fit(&y, &cfg).expect("fit should succeed");
        assert!(
            (model.rate() - 2.5).abs() < 0.5,
            "rate {} not ≈ 2.5",
            model.rate()
        );
    }

    #[test]
    fn croston_forecast_is_flat() {
        let y = intermittent(40, 5, 7.0);
        let cfg = CrostonConfig::classic(0.2);
        let model = Croston::fit(&y, &cfg).expect("fit should succeed");
        let fc = model.forecast(6).expect("forecast should succeed");
        assert_eq!(fc.len(), 6);
        for w in fc.windows(2) {
            assert!((w[0] - w[1]).abs() < 1e-12, "forecast not flat");
        }
    }

    #[test]
    fn croston_forecast_zero_empty() {
        let y = intermittent(40, 5, 7.0);
        let cfg = CrostonConfig::classic(0.2);
        let model = Croston::fit(&y, &cfg).expect("fit should succeed");
        assert!(
            model
                .forecast(0)
                .expect("forecast should succeed")
                .is_empty()
        );
    }

    #[test]
    fn croston_all_zeros_forecasts_zero() {
        let y = vec![0.0_f64; 30];
        let cfg = CrostonConfig::classic(0.2);
        let model = Croston::fit(&y, &cfg).expect("fit should succeed");
        assert!((model.rate() - 0.0).abs() < 1e-12);
        let fc = model.forecast(5).expect("forecast should succeed");
        assert!(fc.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn croston_sba_below_classic() {
        // SBA forecast should be strictly below the classic forecast (bias
        // correction multiplies by 1 − α/2 < 1).
        let y = intermittent(80, 4, 10.0);
        let classic =
            Croston::fit(&y, &CrostonConfig::classic(0.2)).expect("value should be present");
        let sba = Croston::fit(&y, &CrostonConfig::sba(0.2)).expect("value should be present");
        assert!(
            sba.rate() < classic.rate(),
            "SBA {} should be < classic {}",
            sba.rate(),
            classic.rate()
        );
        // Specifically, ratio ≈ (1 − α/2) = 0.9.
        let ratio = sba.rate() / classic.rate();
        assert!((ratio - 0.9).abs() < 1e-6, "ratio {ratio} not ≈ 0.9");
    }

    #[test]
    fn croston_sba_shapes() {
        let y = intermittent(40, 3, 5.0);
        let model = Croston::fit(&y, &CrostonConfig::sba(0.15)).expect("value should be present");
        assert_eq!(model.fitted().len(), 40);
        assert!(model.rate() > 0.0);
    }

    #[test]
    fn croston_tsb_rate_positive() {
        let y = intermittent(60, 4, 8.0);
        let model =
            Croston::fit(&y, &CrostonConfig::tsb(0.2, 0.1)).expect("value should be present");
        assert!(model.rate() > 0.0);
        assert!(model.probability() > 0.0 && model.probability() <= 1.0);
    }

    #[test]
    fn croston_tsb_probability_decays_on_zeros() {
        // After a long run of trailing zeros, the TSB probability (and hence the
        // forecast) should be lower than for a denser series.
        let mut dense = intermittent(60, 3, 5.0);
        // Append many zeros to the end of a copy.
        let mut sparse = dense.clone();
        sparse.extend(std::iter::repeat_n(0.0, 30));
        // Pad dense with non-zeros pattern to equal length for fair compare.
        dense.extend(intermittent(30, 3, 5.0));

        let m_dense =
            Croston::fit(&dense, &CrostonConfig::tsb(0.2, 0.2)).expect("value should be present");
        let m_sparse =
            Croston::fit(&sparse, &CrostonConfig::tsb(0.2, 0.2)).expect("value should be present");
        assert!(
            m_sparse.probability() < m_dense.probability(),
            "trailing zeros should lower TSB probability: {} vs {}",
            m_sparse.probability(),
            m_dense.probability()
        );
    }

    #[test]
    fn croston_tsb_all_zeros_zero_rate() {
        let y = vec![0.0_f64; 40];
        let model =
            Croston::fit(&y, &CrostonConfig::tsb(0.2, 0.2)).expect("value should be present");
        assert!(model.rate().abs() < 1e-9);
    }

    #[test]
    fn croston_dense_demand_recovers_mean() {
        // If every period has the same demand, per-period rate ≈ that demand.
        let y = vec![3.0_f64; 50];
        let model =
            Croston::fit(&y, &CrostonConfig::classic(0.3)).expect("value should be present");
        assert!(
            (model.rate() - 3.0).abs() < 0.3,
            "rate {} not ≈ 3",
            model.rate()
        );
    }

    #[test]
    fn croston_interval_reflects_sparsity() {
        // Sparser demand ⇒ larger smoothed interval.
        let dense = intermittent(80, 2, 5.0);
        let sparse = intermittent(80, 8, 5.0);
        let md =
            Croston::fit(&dense, &CrostonConfig::classic(0.2)).expect("value should be present");
        let ms =
            Croston::fit(&sparse, &CrostonConfig::classic(0.2)).expect("value should be present");
        assert!(
            ms.interval() > md.interval(),
            "sparse interval {} should exceed dense {}",
            ms.interval(),
            md.interval()
        );
    }

    #[test]
    fn croston_single_nonzero() {
        let mut y = vec![0.0_f64; 20];
        y[10] = 12.0;
        let model =
            Croston::fit(&y, &CrostonConfig::classic(0.3)).expect("value should be present");
        assert!(model.rate() > 0.0 && model.rate().is_finite());
    }

    #[test]
    fn croston_method_accessor() {
        let y = intermittent(20, 4, 5.0);
        let classic =
            Croston::fit(&y, &CrostonConfig::classic(0.2)).expect("value should be present");
        let sba = Croston::fit(&y, &CrostonConfig::sba(0.2)).expect("value should be present");
        let tsb = Croston::fit(&y, &CrostonConfig::tsb(0.2, 0.1)).expect("value should be present");
        assert_eq!(classic.method(), CrostonMethod::Classic);
        assert_eq!(sba.method(), CrostonMethod::Sba);
        assert_eq!(tsb.method(), CrostonMethod::Tsb);
    }

    #[test]
    fn croston_err_empty() {
        let cfg = CrostonConfig::classic(0.2);
        assert!(matches!(
            Croston::fit(&[], &cfg).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn croston_err_negative() {
        let y = vec![1.0_f64, -2.0, 3.0];
        let cfg = CrostonConfig::classic(0.2);
        assert!(matches!(
            Croston::fit(&y, &cfg).unwrap_err(),
            TsError::NonFinite
        ));
    }

    #[test]
    fn croston_err_bad_alpha() {
        let y = intermittent(20, 4, 5.0);
        let cfg = CrostonConfig::classic(0.0); // not in (0, 1]
        assert!(matches!(
            Croston::fit(&y, &cfg).unwrap_err(),
            TsError::Internal(_)
        ));
    }

    #[test]
    fn croston_err_nonfinite() {
        let mut y = intermittent(20, 4, 5.0);
        y[3] = f64::INFINITY;
        let cfg = CrostonConfig::classic(0.2);
        assert!(matches!(
            Croston::fit(&y, &cfg).unwrap_err(),
            TsError::NonFinite
        ));
    }
}
