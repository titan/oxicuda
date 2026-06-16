//! Online sliding-window recursive AR forecaster.
//!
//! [`StreamingForecaster`] maintains a fixed-size ring buffer of the most recent
//! observations and an autoregressive model of order `p` whose coefficients are
//! updated by **Recursive Least Squares (RLS)** with an exponential forgetting
//! factor `λ`. Each new observation triggers an O(p²) update; no full re-fit is
//! ever performed.
//!
//! The model is
//!
//! ```text
//! ŷ_t = φ₁ y_{t-1} + … + φ_p y_{t-p} + c
//! ```
//!
//! (an explicit intercept `c` makes constant series reproduce exactly). RLS
//! minimises the exponentially-weighted squared error
//! `Σ λ^{t-i} (y_i − θᵀ x_i)²`, with `θ = [φ₁,…,φ_p, c]`.
//!
//! References:
//! - Ljung, L. (1999). *System Identification: Theory for the User*, 2nd ed.,
//!   Prentice Hall (RLS, §11).
//! - Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed., Prentice Hall.

use std::collections::VecDeque;

use crate::error::{TsError, TsResult};

/// Streaming recursive-least-squares AR(p) forecaster.
#[derive(Debug, Clone)]
pub struct StreamingForecaster {
    window: usize,
    ar_order: usize,
    lambda: f64,
    /// Parameter vector `θ = [φ₁,…,φ_p, intercept]`, length `p + 1`.
    theta: Vec<f64>,
    /// Inverse-correlation matrix `P` (`(p+1) × (p+1)`, row-major).
    p_mat: Vec<f64>,
    /// Ring buffer of recent observations (oldest at front).
    buffer: VecDeque<f32>,
    /// Number of observations consumed so far.
    seen: usize,
}

impl StreamingForecaster {
    /// Create a new streaming forecaster.
    ///
    /// * `window` — capacity of the recent-observation ring buffer (`> 0`).
    /// * `ar_order` — AR order `p` (`1 ≤ p < window`).
    /// * `forgetting_factor` — RLS forgetting factor `λ ∈ (0, 1]`; `1.0` is a
    ///   pure growing-window least squares, smaller values track drift faster.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `window == 0`.
    /// - [`TsError::ShapeMismatch`] when `ar_order == 0`, `ar_order >= window`,
    ///   or `forgetting_factor` is outside `(0, 1]`.
    pub fn new(window: usize, ar_order: usize, forgetting_factor: f32) -> TsResult<Self> {
        if window == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if ar_order == 0 {
            return Err(TsError::ShapeMismatch {
                msg: "ar_order must be >= 1".to_string(),
            });
        }
        if ar_order >= window {
            return Err(TsError::ShapeMismatch {
                msg: format!("ar_order ({ar_order}) must be < window ({window})"),
            });
        }
        let lambda = f64::from(forgetting_factor);
        if !(lambda > 0.0 && lambda <= 1.0) {
            return Err(TsError::ShapeMismatch {
                msg: "forgetting_factor must be in (0, 1]".to_string(),
            });
        }

        let dim = ar_order + 1;
        // Diffuse RLS prior: P₀ = δ⁻¹ I with a large δ⁻¹ for fast initial adaptation.
        const P0: f64 = 1.0e3;
        let mut p_mat = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            p_mat[i * dim + i] = P0;
        }

        Ok(Self {
            window,
            ar_order,
            lambda,
            theta: vec![0.0_f64; dim],
            p_mat,
            buffer: VecDeque::with_capacity(window),
            seen: 0,
        })
    }

    /// Ring-buffer capacity.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// AR order `p`.
    #[must_use]
    pub fn ar_order(&self) -> usize {
        self.ar_order
    }

    /// Number of observations consumed so far.
    #[must_use]
    pub fn seen(&self) -> usize {
        self.seen
    }

    /// Current AR coefficients `[φ₁, …, φ_p]` (most-recent lag first).
    #[must_use]
    pub fn coefficients(&self) -> Vec<f32> {
        self.theta[..self.ar_order]
            .iter()
            .map(|&v| v as f32)
            .collect()
    }

    /// Current intercept term `c`.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.theta[self.ar_order] as f32
    }

    /// Feed one new observation, updating the AR model recursively.
    ///
    /// When at least `p` previous observations are available, the regressor
    /// `x = [y_{t-1}, …, y_{t-p}, 1]` and target `y_t` drive one RLS step before
    /// `y` is pushed into the ring buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::NonFinite`] when `y` is not finite.
    pub fn update(&mut self, y: f32) -> TsResult<()> {
        if !y.is_finite() {
            return Err(TsError::NonFinite);
        }
        let p = self.ar_order;
        if self.buffer.len() >= p {
            let x = self.regressor_from_buffer();
            self.rls_step(&x, f64::from(y));
        }
        if self.buffer.len() == self.window {
            self.buffer.pop_front();
        }
        self.buffer.push_back(y);
        self.seen += 1;
        Ok(())
    }

    /// Produce an `h`-step-ahead forecast by recursively feeding predictions
    /// back into the AR recursion.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidHorizon`] when `h == 0`.
    pub fn forecast(&self, h: usize) -> TsResult<Vec<f32>> {
        if h == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        let p = self.ar_order;
        // Initial lag vector (most-recent first), padded with the oldest known
        // value (or 0) when fewer than p observations are available.
        let mut lags = vec![0.0_f64; p];
        let len = self.buffer.len();
        let pad = self.buffer.front().copied().map_or(0.0_f64, f64::from);
        for (i, slot) in lags.iter_mut().enumerate() {
            *slot = if i < len {
                f64::from(self.buffer[len - 1 - i])
            } else {
                pad
            };
        }

        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            let mut yhat = self.theta[p]; // intercept
            for (coef, &lag) in self.theta[..p].iter().zip(lags.iter()) {
                yhat += coef * lag;
            }
            out.push(yhat as f32);
            // Shift in the new prediction as the most-recent lag.
            lags.rotate_right(1);
            lags[0] = yhat;
        }
        Ok(out)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Build the regressor `[y_{t-1}, …, y_{t-p}, 1]` from the current buffer.
    fn regressor_from_buffer(&self) -> Vec<f64> {
        let p = self.ar_order;
        let len = self.buffer.len();
        let mut x = vec![0.0_f64; p + 1];
        for (i, slot) in x.iter_mut().take(p).enumerate() {
            *slot = f64::from(self.buffer[len - 1 - i]);
        }
        x[p] = 1.0;
        x
    }

    /// One RLS update given regressor `x` and target `y`.
    fn rls_step(&mut self, x: &[f64], y: f64) {
        let dim = self.ar_order + 1;
        let lambda = self.lambda;

        // π = P x.
        let pi: Vec<f64> = (0..dim)
            .map(|i| (0..dim).map(|j| self.p_mat[i * dim + j] * x[j]).sum())
            .collect();
        // denom = λ + xᵀ π.
        let denom = (lambda + dot(x, &pi)).max(1e-12);
        // gain = π / denom.
        let gain: Vec<f64> = pi.iter().map(|&v| v / denom).collect();
        // a-priori error e = y − θᵀ x.
        let err = y - dot(&self.theta, x);
        // θ ← θ + gain · e.
        for (t, &g) in self.theta.iter_mut().zip(gain.iter()) {
            *t += g * err;
        }
        // P ← (P − gain πᵀ) / λ, then symmetrise to curb round-off drift.
        let mut p_new = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                p_new[i * dim + j] = (self.p_mat[i * dim + j] - gain[i] * pi[j]) / lambda;
            }
        }
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (p_new[i * dim + j] + p_new[j * dim + i]);
                p_new[i * dim + j] = avg;
                p_new[j * dim + i] = avg;
            }
        }
        self.p_mat = p_new;
    }
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Generate an AR(1) series y_t = phi*y_{t-1} + noise.
    fn ar1_series(n: usize, phi: f32, noise_sd: f32, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut raw = vec![0.0_f32; n];
        rng.fill_normal(&mut raw);
        let mut y = vec![0.0_f32; n];
        let mut prev = 0.0_f32;
        for t in 0..n {
            let val = phi * prev + noise_sd * raw[t];
            y[t] = val;
            prev = val;
        }
        y
    }

    #[test]
    fn streaming_ar1_coefficient_converges() {
        let phi = 0.6_f32;
        let y = ar1_series(400, phi, 0.1, 11);
        let mut f = StreamingForecaster::new(20, 1, 1.0).expect("new");
        for &v in &y {
            f.update(v).expect("update");
        }
        let est = f.coefficients()[0];
        assert!(
            (est - phi).abs() < 0.15,
            "phi estimate {est} not near {phi}"
        );
    }

    #[test]
    fn streaming_one_step_error_decreases() {
        // Strongly autocorrelated, low-noise AR(1): the untrained model (θ = 0)
        // predicts ≈0 and is badly wrong early on, then converges to the noise
        // floor as data streams in. Squared error amplifies the early transient.
        let y = ar1_series(300, 0.9, 0.08, 23);
        let mut f = StreamingForecaster::new(10, 1, 1.0).expect("new");
        let mut sq_errs = Vec::new();
        for &v in &y {
            let pred = f.forecast(1).expect("forecast")[0];
            if f.seen() >= f.ar_order() {
                let e = v - pred;
                sq_errs.push(e * e);
            }
            f.update(v).expect("update");
        }
        // Early window includes the learning transient (from index 0); the late
        // window is the converged regime.
        let early: f32 = sq_errs[0..20].iter().sum::<f32>() / 20.0;
        let n_late = 80usize;
        let late: f32 = sq_errs[sq_errs.len() - n_late..].iter().sum::<f32>() / n_late as f32;
        assert!(
            late < early,
            "error did not decrease: early={early} late={late}"
        );
    }

    #[test]
    fn streaming_constant_series_forecasts_constant() {
        let k = 3.0_f32;
        let mut f = StreamingForecaster::new(10, 2, 1.0).expect("new");
        for _ in 0..60 {
            f.update(k).expect("update");
        }
        let fc = f.forecast(4).expect("forecast");
        for v in fc {
            assert!((v - k).abs() < 1e-2, "constant forecast {v} != {k}");
        }
    }

    #[test]
    fn streaming_forecast_length() {
        let y = ar1_series(50, 0.5, 0.1, 5);
        let mut f = StreamingForecaster::new(10, 3, 0.99).expect("new");
        for &v in &y {
            f.update(v).expect("update");
        }
        assert_eq!(f.forecast(7).expect("forecast").len(), 7);
        assert_eq!(f.coefficients().len(), 3);
    }

    #[test]
    fn streaming_forecast_before_data_returns_h() {
        // No updates yet: forecast still returns h finite values (all 0 here).
        let f = StreamingForecaster::new(8, 2, 1.0).expect("new");
        let fc = f.forecast(3).expect("forecast");
        assert_eq!(fc.len(), 3);
        assert!(fc.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn streaming_err_order_ge_window() {
        assert!(matches!(
            StreamingForecaster::new(5, 5, 1.0).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
        assert!(matches!(
            StreamingForecaster::new(5, 8, 1.0).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn streaming_err_zero_window() {
        assert!(matches!(
            StreamingForecaster::new(0, 1, 1.0).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn streaming_err_zero_order() {
        assert!(matches!(
            StreamingForecaster::new(5, 0, 1.0).unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn streaming_err_bad_forgetting() {
        assert!(StreamingForecaster::new(10, 2, 1.5).is_err());
        assert!(StreamingForecaster::new(10, 2, 0.0).is_err());
        assert!(StreamingForecaster::new(10, 2, -0.5).is_err());
    }

    #[test]
    fn streaming_err_forecast_zero_horizon() {
        let f = StreamingForecaster::new(10, 2, 1.0).expect("new");
        assert!(matches!(
            f.forecast(0).unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn streaming_err_nonfinite_update() {
        let mut f = StreamingForecaster::new(10, 2, 1.0).expect("new");
        assert!(matches!(
            f.update(f32::NAN).unwrap_err(),
            TsError::NonFinite
        ));
    }
}
