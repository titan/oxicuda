//! Box-Jenkins SARIMA(p,d,q)×(P,D,Q)_s implementation.
//!
//! Seasonal Auto-Regressive Integrated Moving Average model.
//! Supports non-seasonal AR/I/MA and seasonal AR/I/MA components
//! with configurable seasonal period `s`.
//!
//! # Model structure
//!
//! The full SARIMA polynomial is:
//! ```text
//! Φ(B^s) φ(B) (1-B)^d (1-B^s)^D y_t = Θ(B^s) θ(B) ε_t
//! ```
//! where B is the backshift operator, φ are the non-seasonal AR coefficients,
//! Φ are the seasonal AR coefficients, θ are the non-seasonal MA coefficients,
//! and Θ are the seasonal MA coefficients.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

/// Type alias for the random number generator used by SARIMA.
type TsRng = LcgRng;

// ─── SarimaConfig ─────────────────────────────────────────────────────────────

/// Configuration for a SARIMA(p,d,q)×(P,D,Q)_s model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SarimaConfig {
    /// Non-seasonal AR order.
    pub p: usize,
    /// Non-seasonal differencing order.
    pub d: usize,
    /// Non-seasonal MA order.
    pub q: usize,
    /// Seasonal AR order.
    pub cap_p: usize,
    /// Seasonal differencing order.
    pub cap_d: usize,
    /// Seasonal MA order.
    pub cap_q: usize,
    /// Seasonal period (e.g., 12 for monthly data with annual seasonality).
    pub s: usize,
}

impl SarimaConfig {
    /// Returns the minimum number of history observations required for a
    /// one-step-ahead prediction given the non-seasonal and seasonal orders.
    #[must_use]
    pub fn min_history_len(&self) -> usize {
        let ar_lag = self.p;
        let sar_lag = self
            .cap_p
            .saturating_mul(self.s)
            .max(if self.cap_p > 0 { 1 } else { 0 });
        let ma_lag = self.q;
        let sma_lag = self
            .cap_q
            .saturating_mul(self.s)
            .max(if self.cap_q > 0 { 1 } else { 0 });
        ar_lag.max(sar_lag).max(ma_lag).max(sma_lag).max(1)
    }
}

// ─── Sarima ───────────────────────────────────────────────────────────────────

/// A fitted or initialised SARIMA(p,d,q)×(P,D,Q)_s model.
///
/// Coefficients are stored in the lag-1…lag-k convention:
/// `ar_coefs[i]` is the coefficient for `y_{t-(i+1)}`.
#[derive(Debug)]
pub struct Sarima {
    /// Non-seasonal AR coefficients φ₁…φₚ (length `p`).
    pub ar_coefs: Vec<f64>,
    /// Seasonal AR coefficients Φ₁…Φ_P (length `cap_p`).
    pub sar_coefs: Vec<f64>,
    /// Non-seasonal MA coefficients θ₁…θ_q (length `q`).
    pub ma_coefs: Vec<f64>,
    /// Seasonal MA coefficients Θ₁…Θ_Q (length `cap_q`).
    pub sma_coefs: Vec<f64>,
    /// Model configuration.
    pub config: SarimaConfig,
}

impl Sarima {
    // ─── Construction ─────────────────────────────────────────────────────────

    /// Create a new `Sarima` model with small random coefficient initialisations.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidSequenceLength`] if `config.s == 0`.
    pub fn new(config: SarimaConfig, rng: &mut TsRng) -> TsResult<Self> {
        if config.s == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }

        // Scale for initial random coefficients: small values near 0 for stability.
        const INIT_SCALE: f64 = 0.1;

        let ar_coefs = (0..config.p)
            .map(|_| (rng.next_f32() as f64) * INIT_SCALE)
            .collect();
        let sar_coefs = (0..config.cap_p)
            .map(|_| (rng.next_f32() as f64) * INIT_SCALE)
            .collect();
        let ma_coefs = (0..config.q)
            .map(|_| (rng.next_f32() as f64) * INIT_SCALE)
            .collect();
        let sma_coefs = (0..config.cap_q)
            .map(|_| (rng.next_f32() as f64) * INIT_SCALE)
            .collect();

        Ok(Self {
            ar_coefs,
            sar_coefs,
            ma_coefs,
            sma_coefs,
            config,
        })
    }

    /// Create a `Sarima` model with all-zero coefficients (used internally for fitting).
    fn zeros(config: SarimaConfig) -> TsResult<Self> {
        if config.s == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        let p = config.p;
        let cap_p = config.cap_p;
        let q = config.q;
        let cap_q = config.cap_q;
        Ok(Self {
            ar_coefs: vec![0.0_f64; p],
            sar_coefs: vec![0.0_f64; cap_p],
            ma_coefs: vec![0.0_f64; q],
            sma_coefs: vec![0.0_f64; cap_q],
            config,
        })
    }

    // ─── Differencing ─────────────────────────────────────────────────────────

    /// Apply seasonal then regular differencing to `x`.
    ///
    /// Seasonal differencing `(1-B^s)^D` is applied first (for each of `cap_d` passes),
    /// followed by regular differencing `(1-B)^d` (for each of `d` passes).
    ///
    /// Returns the differenced series. If the series is too short to difference,
    /// returns a clone of the input.
    #[must_use]
    pub fn difference(&self, x: &[f64]) -> Vec<f64> {
        let s = self.config.s;
        let mut current = x.to_vec();

        // Seasonal differencing: (1 - B^s)^cap_d
        for _ in 0..self.config.cap_d {
            if current.len() <= s {
                // Not enough data to seasonal-difference; return as-is.
                break;
            }
            let new_len = current.len() - s;
            let differenced: Vec<f64> = (s..current.len())
                .map(|i| current[i] - current[i - s])
                .collect();
            debug_assert_eq!(differenced.len(), new_len);
            current = differenced;
        }

        // Regular differencing: (1 - B)^d
        for _ in 0..self.config.d {
            if current.len() <= 1 {
                break;
            }
            let differenced: Vec<f64> = (1..current.len())
                .map(|i| current[i] - current[i - 1])
                .collect();
            current = differenced;
        }

        current
    }

    /// Reverse the differencing applied by [`Self::difference`].
    ///
    /// Applies `d` regular integrations first (using the first `d` values of
    /// `original` as initial conditions), then `cap_d` seasonal integrations
    /// (using the first `cap_d * s` values of `original` as initial conditions).
    ///
    /// `original` must be the series that was passed to `difference`.
    #[must_use]
    pub fn undifference(&self, diff: &[f64], original: &[f64]) -> Vec<f64> {
        let s = self.config.s;
        let mut current = diff.to_vec();

        // Undo regular differencing: d passes of cumulative summation.
        for pass in 0..self.config.d {
            // We need the initial conditions from the `pass`-th differenced version
            // of `original`.  For simplicity we use the last value of the already
            // integrated segment as the seed (standard integration initialisation).
            // We seed with original[pass] which is the first element of the
            // `pass`-th level undifferenced original.
            let seed_idx = self.config.d.saturating_sub(1 + pass);
            let seed = original.get(seed_idx).copied().unwrap_or(0.0);

            let mut integrated = Vec::with_capacity(current.len() + 1);
            integrated.push(seed);
            for &v in &current {
                let prev = *integrated.last().unwrap_or(&seed);
                integrated.push(prev + v);
            }
            // Drop the prepended seed — callers expect same-length output matching diff.
            // Actually the convention used in `difference` removes `d` elements from the
            // front, so integration must produce `diff.len() + 1` elements with the
            // first being the seed.  We keep all of them.
            current = integrated;
        }

        // Undo seasonal differencing: cap_d passes of seasonal cumulative summation.
        for _ in 0..self.config.cap_d {
            // Season prefix: the first s values from original at the appropriate
            // differencing level.  We use the first s values of `original` as
            // the seed block (closest available approximation without storing
            // intermediate series).
            let prefix_len = s.min(original.len());
            let prefix: Vec<f64> = original[..prefix_len].to_vec();

            let mut integrated = prefix.clone();
            for &v in &current {
                let lag_idx = integrated.len().saturating_sub(s);
                let lag_val = integrated.get(lag_idx).copied().unwrap_or(0.0);
                integrated.push(lag_val + v);
            }
            // Trim the prefix so we return only the reconstructed portion.
            let trimmed_start = prefix_len;
            current = integrated[trimmed_start..].to_vec();
        }

        current
    }

    // ─── Prediction ───────────────────────────────────────────────────────────

    /// Compute a one-step-ahead forecast from `history` using stored coefficients.
    ///
    /// The AR and seasonal-AR components are used directly. The MA component is
    /// approximated as zero (no stored residual state), which is exact for a
    /// pure AR model and approximate for ARMA/SARMA.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidSequenceLength`] if `history` is shorter than
    /// the minimum number of lags required by the model orders.
    pub fn predict(&self, history: &[f64]) -> TsResult<f64> {
        let s = self.config.s;

        // Minimum length needed: at least 1, or sufficient to satisfy all lags.
        let min_ar = self.config.p;
        let min_sar = if self.config.cap_p > 0 {
            self.config.cap_p.saturating_mul(s).max(1)
        } else {
            0
        };
        let min_ma = self.config.q;
        let min_sma = if self.config.cap_q > 0 {
            self.config.cap_q.saturating_mul(s).max(1)
        } else {
            0
        };
        let min_len = min_ar.max(min_sar).max(min_ma).max(min_sma).max(1);

        if history.len() < min_len {
            return Err(TsError::InvalidSequenceLength(history.len()));
        }

        let last = history.len() - 1;
        let mut forecast = 0.0_f64;

        // Non-seasonal AR part: φ(B) y_t = Σ φᵢ · y_{t-i}
        for (i, &coef) in self.ar_coefs.iter().enumerate() {
            let lag_idx = last.saturating_sub(i);
            forecast += coef * history[lag_idx];
        }

        // Seasonal AR part: Φ(B^s) y_t = Σ Φ_k · y_{t-k·s}
        for (k, &coef) in self.sar_coefs.iter().enumerate() {
            let lag = (k + 1).saturating_mul(s);
            if lag > last {
                // Lag exceeds available history; use oldest available value.
                forecast += coef * history[0];
            } else {
                forecast += coef * history[last - lag];
            }
        }

        // MA and seasonal MA parts: residuals approximated as 0 (no state stored).
        // This is exact for pure AR models and is the zero-residual approximation
        // for the MA components, which is standard for the prediction-only use case.

        if !forecast.is_finite() {
            return Err(TsError::NonFinite);
        }

        Ok(forecast)
    }

    // ─── Fitting ──────────────────────────────────────────────────────────────

    /// Fit a SARIMA model to `data` via approximate conditional least-squares (OLS).
    ///
    /// The fitting procedure:
    /// 1. Applies differencing to make the series stationary.
    /// 2. Builds a lagged design matrix for the AR terms.
    /// 3. Uses OLS (normal equations via Gram–Schmidt orthogonalisation) to
    ///    estimate all AR and seasonal-AR coefficients simultaneously.
    /// 4. Sets MA coefficients to zero (not estimated in this simplified version).
    ///
    /// For robust production use, the Hannan-Rissanen or Gauss-Newton procedures
    /// should be applied; this implementation gives reasonable starting values
    /// suitable for subsequent iterative refinement.
    ///
    /// # Errors
    ///
    /// - [`TsError::EmptyInput`] if `data` is empty.
    /// - [`TsError::InvalidSequenceLength`] if `config.s == 0`.
    pub fn fit(data: &[f64], config: SarimaConfig) -> TsResult<Self> {
        if data.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "data is empty".into(),
            });
        }
        if config.s == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }

        let mut model = Self::zeros(config)?;

        // Difference the series to achieve stationarity.
        let stationary = model.difference(data);

        // We need at least `min_cols + 1` observations to build the regression.
        let p = model.config.p;
        let cap_p = model.config.cap_p;
        let s = model.config.s;

        // Total number of AR columns in the design matrix.
        let n_ar_cols = p + cap_p;

        if n_ar_cols == 0 || stationary.len() < 2 {
            // No AR terms to estimate; model stays at zeros.
            return Ok(model);
        }

        // Maximum lag required.
        let max_lag = p.max(cap_p.saturating_mul(s));
        if stationary.len() <= max_lag {
            // Not enough data to form even a single complete row; return zero model.
            return Ok(model);
        }

        // Build design matrix X (rows = observations, cols = AR lags) and target y.
        // Row t: y_{t} is regressed on y_{t-1}..y_{t-p}, y_{t-s}..y_{t-cap_p*s}.
        let n_obs = stationary.len() - max_lag;
        let mut x_mat = vec![0.0_f64; n_obs * n_ar_cols];
        let mut y_vec = vec![0.0_f64; n_obs];

        for row in 0..n_obs {
            let t = row + max_lag; // index into `stationary` being predicted
            y_vec[row] = stationary[t];

            // Non-seasonal AR lags: y_{t-1}, …, y_{t-p}
            for lag in 0..p {
                x_mat[row * n_ar_cols + lag] = stationary[t - lag - 1];
            }
            // Seasonal AR lags: y_{t-s}, …, y_{t-cap_p*s}
            for k in 0..cap_p {
                let lag = (k + 1) * s;
                let col = p + k;
                if lag <= t {
                    x_mat[row * n_ar_cols + col] = stationary[t - lag];
                }
                // else stays 0 (not enough seasonal history for this row)
            }
        }

        // Solve via OLS normal equations: β = (X'X)⁻¹ X'y using Cholesky / GS.
        // We use the numerically robust modified Gram-Schmidt QR decomposition.
        let coeffs = ols_qr(&x_mat, &y_vec, n_obs, n_ar_cols);

        // Distribute estimated coefficients back into the model.
        for (i, c) in model.ar_coefs.iter_mut().enumerate() {
            *c = coeffs.get(i).copied().unwrap_or(0.0);
        }
        for (k, c) in model.sar_coefs.iter_mut().enumerate() {
            *c = coeffs.get(p + k).copied().unwrap_or(0.0);
        }

        Ok(model)
    }

    // ─── Accessors ────────────────────────────────────────────────────────────

    /// Return the seasonal period `s`.
    #[must_use]
    #[inline]
    pub fn seasonal_period(&self) -> usize {
        self.config.s
    }
}

// ─── OLS via Modified Gram-Schmidt QR ────────────────────────────────────────

/// Solve the linear least-squares problem `X β ≈ y` using the modified
/// Gram-Schmidt QR decomposition.
///
/// `x` is stored row-major with shape `[n_obs, n_cols]`.
/// Returns a coefficient vector of length `n_cols`.
fn ols_qr(x: &[f64], y: &[f64], n_obs: usize, n_cols: usize) -> Vec<f64> {
    // Copy columns of X into column-major storage for in-place MGS.
    let mut q = vec![0.0_f64; n_obs * n_cols]; // column-major: q[col * n_obs + row]
    for col in 0..n_cols {
        for row in 0..n_obs {
            q[col * n_obs + row] = x[row * n_cols + col];
        }
    }

    let mut r = vec![0.0_f64; n_cols * n_cols]; // upper triangular, row-major
    let mut qty = vec![0.0_f64; n_cols]; // Q' y

    // Modified Gram-Schmidt
    for j in 0..n_cols {
        // Compute norm of column j.
        let norm_sq: f64 = (0..n_obs).map(|i| q[j * n_obs + i].powi(2)).sum();
        let norm = norm_sq.sqrt();
        r[j * n_cols + j] = norm;

        if norm < 1e-14 {
            // Column is (near) zero — set to zero and skip.
            for i in 0..n_obs {
                q[j * n_obs + i] = 0.0;
            }
            continue;
        }

        // Normalise column j.
        for i in 0..n_obs {
            q[j * n_obs + i] /= norm;
        }

        // Project q_j out of all subsequent columns k > j.
        for k in (j + 1)..n_cols {
            let dot: f64 = (0..n_obs)
                .map(|i| q[j * n_obs + i] * q[k * n_obs + i])
                .sum();
            r[j * n_cols + k] = dot;
            for i in 0..n_obs {
                q[k * n_obs + i] -= dot * q[j * n_obs + i];
            }
        }

        // Accumulate Q' y for column j.
        qty[j] = (0..n_obs).map(|i| q[j * n_obs + i] * y[i]).sum();
    }

    // Back-substitution: solve R β = Q' y.
    let mut beta = vec![0.0_f64; n_cols];
    for j in (0..n_cols).rev() {
        let r_jj = r[j * n_cols + j];
        if r_jj.abs() < 1e-14 {
            beta[j] = 0.0;
            continue;
        }
        let mut rhs = qty[j];
        for k in (j + 1)..n_cols {
            rhs -= r[j * n_cols + k] * beta[k];
        }
        beta[j] = rhs / r_jj;
    }

    beta
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    /// Build a simple AR(1) config with seasonal period s.
    fn ar1_config(s: usize) -> SarimaConfig {
        SarimaConfig {
            p: 1,
            d: 1,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s,
        }
    }

    // ── Test 1: difference_len ─────────────────────────────────────────────────

    #[test]
    fn difference_len() {
        let cfg = SarimaConfig {
            p: 0,
            d: 1,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 12,
        };
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let diff = model.difference(&x);
        // d=1 differencing of length-10 series → length 9
        assert_eq!(diff.len(), 9);
    }

    // ── Test 2: undifference_roundtrip ────────────────────────────────────────

    #[test]
    fn undifference_roundtrip() {
        let cfg = SarimaConfig {
            p: 1,
            d: 1,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 4,
        };
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");

        let original: Vec<f64> = (0..20).map(|i| (i as f64) * 1.5 + 3.0).collect();
        let diff = model.difference(&original);
        let recovered = model.undifference(&diff, &original);

        // The undifferencing recovers the portion of the series after the seed.
        // For d=1 we expect recovered to be approximately original[1..].
        // Length: undifference adds back 1 seed element, so recovered has diff.len()+1 == original.len() elements.
        assert_eq!(recovered.len(), original.len());
        for (i, (&rec, &orig)) in recovered.iter().zip(original.iter()).enumerate() {
            assert!(
                (rec - orig).abs() < 1e-8,
                "index {i}: recovered={rec} original={orig}"
            );
        }
    }

    // ── Test 3: predict_finite ─────────────────────────────────────────────────

    #[test]
    fn predict_finite() {
        let cfg = ar1_config(12);
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");
        let history: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let pred = model.predict(&history).expect("predict ok");
        assert!(pred.is_finite(), "prediction must be finite, got {pred}");
    }

    // ── Test 4: fit_finite ────────────────────────────────────────────────────

    #[test]
    fn fit_finite() {
        // Fit on a simple sine wave.
        let data: Vec<f64> = (0..120)
            .map(|i| ((i as f64) * std::f64::consts::TAU / 12.0).sin())
            .collect();
        let cfg = SarimaConfig {
            p: 2,
            d: 0,
            q: 0,
            cap_p: 1,
            cap_d: 0,
            cap_q: 0,
            s: 12,
        };
        let model = Sarima::fit(&data, cfg).expect("fit ok");

        let history: Vec<f64> = data[data.len() - 24..].to_vec();
        let pred = model.predict(&history).expect("predict ok");
        assert!(pred.is_finite(), "fit+predict must be finite, got {pred}");
    }

    // ── Test 5: s_0_error ────────────────────────────────────────────────────

    #[test]
    fn s_0_error() {
        let cfg = SarimaConfig {
            p: 1,
            d: 0,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 0, // invalid!
        };
        let mut rng = make_rng();
        let result = Sarima::new(cfg, &mut rng);
        assert!(
            matches!(result, Err(TsError::InvalidSequenceLength(0))),
            "expected InvalidSequenceLength(0), got {result:?}"
        );
    }

    // ── Test 6: predict_history_too_short_error ───────────────────────────────

    #[test]
    fn predict_history_too_short_error() {
        let cfg = SarimaConfig {
            p: 5, // requires 5 history points
            d: 0,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 12,
        };
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");
        // Provide only 3 history points, but model requires 5.
        let short_history = vec![1.0_f64, 2.0, 3.0];
        let result = model.predict(&short_history);
        assert!(
            matches!(result, Err(TsError::InvalidSequenceLength(_))),
            "expected InvalidSequenceLength, got {result:?}"
        );
    }

    // ── Test 7: constant_series_predict ──────────────────────────────────────

    #[test]
    fn constant_series_predict() {
        // Constant series: y_t = 5.0 for all t.
        // After fitting, predict should yield something near 5.0 * sum_of_coefs.
        // We just verify the result is finite and non-panic.
        let c = 5.0_f64;
        let data: Vec<f64> = vec![c; 50];
        let cfg = SarimaConfig {
            p: 1,
            d: 0,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 12,
        };
        let model = Sarima::fit(&data, cfg).expect("fit ok");
        let history: Vec<f64> = vec![c; 20];
        let pred = model.predict(&history).expect("predict ok");
        assert!(
            pred.is_finite(),
            "constant series predict must be finite: {pred}"
        );
        // For a constant series, the AR(1) coefficient should approach ~1,
        // giving a prediction near `c`.  We check it's in a broad window.
        assert!(
            (pred - c).abs() < c * 2.0 + 1.0,
            "prediction {pred} too far from constant {c}"
        );
    }

    // ── Test 8: seasonal_period ───────────────────────────────────────────────

    #[test]
    fn seasonal_period() {
        let cfg = SarimaConfig {
            p: 1,
            d: 0,
            q: 1,
            cap_p: 1,
            cap_d: 1,
            cap_q: 1,
            s: 7,
        };
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");
        assert_eq!(model.seasonal_period(), 7);
    }

    // ── Test 9: ar_coefs_len ─────────────────────────────────────────────────

    #[test]
    fn ar_coefs_len() {
        let p = 4_usize;
        let cfg = SarimaConfig {
            p,
            d: 1,
            q: 2,
            cap_p: 1,
            cap_d: 0,
            cap_q: 1,
            s: 12,
        };
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");
        assert_eq!(model.ar_coefs.len(), p, "ar_coefs length must equal p={p}");
        assert_eq!(model.ma_coefs.len(), 2, "ma_coefs length must equal q=2");
        assert_eq!(
            model.sar_coefs.len(),
            1,
            "sar_coefs length must equal cap_p=1"
        );
        assert_eq!(
            model.sma_coefs.len(),
            1,
            "sma_coefs length must equal cap_q=1"
        );
    }

    // ── Test 10: fit_with_s_0_error ───────────────────────────────────────────

    #[test]
    fn fit_with_s_0_error() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let cfg = SarimaConfig {
            p: 1,
            d: 0,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 0,
        };
        let result = Sarima::fit(&data, cfg);
        assert!(
            matches!(result, Err(TsError::InvalidSequenceLength(0))),
            "expected InvalidSequenceLength(0), got {result:?}"
        );
    }

    // ── Test 11: fit_empty_data_error ─────────────────────────────────────────

    #[test]
    fn fit_empty_data_error() {
        let cfg = SarimaConfig {
            p: 1,
            d: 0,
            q: 0,
            cap_p: 0,
            cap_d: 0,
            cap_q: 0,
            s: 12,
        };
        let result = Sarima::fit(&[], cfg);
        assert!(
            matches!(result, Err(TsError::EmptyInput { .. })),
            "expected EmptyInput, got {result:?}"
        );
    }

    // ── Test 12: seasonal_differencing_len ────────────────────────────────────

    #[test]
    fn seasonal_differencing_len() {
        let cfg = SarimaConfig {
            p: 0,
            d: 0,
            q: 0,
            cap_p: 0,
            cap_d: 1, // one seasonal diff pass
            cap_q: 0,
            s: 4,
        };
        let mut rng = make_rng();
        let model = Sarima::new(cfg, &mut rng).expect("new ok");
        let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let diff = model.difference(&x);
        // Seasonal diff with s=4, cap_d=1: output length = 12 - 4 = 8.
        assert_eq!(diff.len(), 8, "got len={}", diff.len());
    }
}
