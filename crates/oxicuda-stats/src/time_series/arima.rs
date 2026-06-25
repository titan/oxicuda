//! ARIMA (p, d, q) estimation via Hannan-Rissanen 3-stage + CSS fine-tuning.
//!
//! # Algorithm Overview
//!
//! 1. **Differencing** — apply d rounds of first-differencing to achieve stationarity.
//! 2. **Long-AR pilot** — fit a long AR(m) to the differenced series (with ridge
//!    regularisation for numerical stability) to obtain residual proxies ε̂_t
//!    (Hannan-Rissanen stage 1).
//! 3. **ARMA OLS stage** — build a regressor matrix of [φ-lags | θ-lag-residuals | const]
//!    and solve by ridged OLS to get initial (φ, θ, c) (stage 2/3).
//! 4. **CSS fine-tuning** — coordinate descent on the Conditional Sum-of-Squares
//!    objective using finite-difference gradients + step-halving line search.
//! 5. **Forecast** — recursive point forecasts for h steps ahead on the differenced
//!    series, then un-differenced back to the original scale.
//!
//! # References
//! - Hannan & Rissanen (1982) "Recursive Estimation of Mixed ARMA Processes".
//!   *Biometrika* 69(1):81-94.
//! - Box, Jenkins, Reinsel & Ljung (2015) *Time Series Analysis: Forecasting and Control*.

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration & Result structs
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for ARIMA(p, d, q) fitting.
#[derive(Debug, Clone)]
pub struct ArimaConfig {
    /// AR order.
    pub p: usize,
    /// Integration order (number of first-differences).
    pub d: usize,
    /// MA order.
    pub q: usize,
    /// Whether to include a constant / drift term.
    pub include_constant: bool,
    /// Maximum CSS fine-tuning iterations (default 200).
    pub max_iter: usize,
    /// Convergence tolerance on max parameter change (default 1e-7).
    pub tol: f64,
}

impl Default for ArimaConfig {
    fn default() -> Self {
        Self {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        }
    }
}

/// Fitted ARIMA model.
#[derive(Debug, Clone)]
pub struct ArimaFit {
    /// AR coefficients φ₁, …, φ_p.
    pub phi: Vec<f64>,
    /// MA coefficients θ₁, …, θ_q.
    pub theta: Vec<f64>,
    /// Constant / drift term.
    pub constant: f64,
    /// Estimated noise variance σ² = CSS / dof.
    pub sigma2: f64,
    /// Final conditional sum-of-squares.
    pub css: f64,
    /// Akaike information criterion.
    pub aic: f64,
    /// Bayesian information criterion.
    pub bic: f64,
    /// AR order.
    pub p: usize,
    /// Integration order.
    pub d: usize,
    /// MA order.
    pub q: usize,
    /// Effective observations used in estimation = series.len() - d.
    pub n_obs: usize,
    /// Last max(p, 1) values of the *original* series (needed for forecasting).
    pub last_x: Vec<f64>,
    /// Last q innovations from the differenced fit (needed for forecasting).
    pub last_eps: Vec<f64>,
    /// Initial values at each differencing level (for un-differencing forecasts).
    /// `diff_init[k]` = last value of the series at integration level k.
    pub diff_init: Vec<Vec<f64>>,
    /// Whether CSS fine-tuning converged within `max_iter`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Differencing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply d rounds of first-differencing. Returns the differenced series.
///
/// d = 0 returns a clone of the input.
fn apply_difference(x: &[f64], d: usize) -> Vec<f64> {
    let mut cur = x.to_vec();
    for _ in 0..d {
        cur = (1..cur.len()).map(|t| cur[t] - cur[t - 1]).collect();
    }
    cur
}

/// Reverse-integrate `h` forecast steps using the final values from the original series.
///
/// `diffs` — h forecasts on the d-differenced scale.
/// `diff_init[k]` — last known value at integration level k.
fn undifference(diffs: &[f64], diff_init: &[Vec<f64>]) -> Vec<f64> {
    let d = diff_init.len();
    if d == 0 {
        return diffs.to_vec();
    }
    let mut cur = diffs.to_vec();
    for k in (0..d).rev() {
        let last_val = *diff_init[k].last().unwrap_or(&0.0);
        let mut integrated = Vec::with_capacity(cur.len());
        let mut acc = last_val;
        for &v in &cur {
            acc += v;
            integrated.push(acc);
        }
        cur = integrated;
    }
    cur
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerically stable OLS via ridge-regularised normal equations
// ─────────────────────────────────────────────────────────────────────────────

/// Solve (XᵀX + λI) β = Xᵀy via Gauss-Jordan.
///
/// Ridge parameter `lambda` (≥ 0) ensures numerical stability when the design
/// matrix is near-singular (e.g. highly correlated AR lags).
fn ridge_solve(x: &[f64], y: &[f64], n: usize, p: usize, lambda: f64) -> StatsResult<Vec<f64>> {
    if x.len() != n * p || y.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: x.len(),
            b: n * p,
        });
    }
    if n < p {
        return Err(StatsError::InsufficientSampleSize { got: n, need: p });
    }

    // Compute XᵀX (p × p)
    let mut xtx = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += x[k * p + i] * x[k * p + j];
            }
            xtx[i * p + j] = acc;
            xtx[j * p + i] = acc;
        }
    }
    // Add ridge
    for i in 0..p {
        xtx[i * p + i] += lambda;
    }

    // Compute Xᵀy (p)
    let mut xty = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += x[k * p + i] * y[k];
        }
        xty[i] = acc;
    }

    // Invert (XᵀX + λI)
    let inv = matrix_inverse_lu(&xtx, p)?;

    // β = inv · Xᵀy
    let mut beta = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = 0.0_f64;
        for j in 0..p {
            acc += inv[i * p + j] * xty[j];
        }
        beta[i] = acc;
    }
    Ok(beta)
}

/// Compute residuals for a linear model y ≈ X β.
fn compute_linear_residuals(x: &[f64], y: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let yhat: f64 = (0..p).map(|j| x[i * p + j] * beta[j]).sum();
            y[i] - yhat
        })
        .collect()
}

/// Choose a small ridge parameter based on diagonal scale of XᵀX.
///
/// λ = ε × mean_diagonal(XᵀX), which preserves scale-invariance.
fn auto_ridge(xtx_diag: &[f64], eps: f64) -> f64 {
    let mean_diag = xtx_diag.iter().sum::<f64>() / xtx_diag.len().max(1) as f64;
    (eps * mean_diag).max(eps * 1e-6)
}

/// Compute the diagonal of XᵀX without forming the full matrix.
fn xtx_diagonal(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..p)
        .map(|j| (0..n).map(|i| x[i * p + j] * x[i * p + j]).sum::<f64>())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Innovation / CSS helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute innovations ε_t for ARMA(p, q):
///   ε_t = x[t] - c - Σ_{i=1}^p φ_i x[t-i] - Σ_{j=1}^q θ_j ε_{t-j]
///
/// Past x and ε before index 0 are treated as 0.
fn compute_innovations(x: &[f64], phi: &[f64], theta: &[f64], constant: f64) -> Vec<f64> {
    let n = x.len();
    let mut eps = vec![0.0_f64; n];
    for t in 0..n {
        let mut val = constant;
        for (i, &phi_i) in phi.iter().enumerate() {
            if t > i {
                val += phi_i * x[t - i - 1];
            }
        }
        for (j, &theta_j) in theta.iter().enumerate() {
            if t > j {
                val += theta_j * eps[t - j - 1];
            }
        }
        eps[t] = x[t] - val;
    }
    eps
}

/// Sum of squared innovations.
fn css_from_eps(eps: &[f64]) -> f64 {
    eps.iter().map(|&e| e * e).sum()
}

/// Compute CSS with one parameter perturbed by `delta`.
///
/// The flat parameter index `param_idx` addresses `[φ₁…φ_p | θ₁…θ_q | c]`, so the
/// AR/MA orders are read directly from the slice lengths.
fn css_perturbed(
    x: &[f64],
    phi: &[f64],
    theta: &[f64],
    constant: f64,
    param_idx: usize,
    delta: f64,
) -> f64 {
    let ar_order = phi.len();
    let ma_order = theta.len();
    let mut phi_p = phi.to_vec();
    let mut theta_p = theta.to_vec();
    let mut c_p = constant;

    if param_idx < ar_order {
        phi_p[param_idx] += delta;
    } else if param_idx < ar_order + ma_order {
        theta_p[param_idx - ar_order] += delta;
    } else {
        c_p += delta;
    }

    let eps = compute_innovations(x, &phi_p, &theta_p, c_p);
    css_from_eps(&eps)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hannan-Rissanen initialization
// ─────────────────────────────────────────────────────────────────────────────

/// Hannan-Rissanen 3-stage ARMA initialisation on the (already differenced) series.
///
/// Returns `(phi, theta, constant)`.
fn hr_init(
    x: &[f64],
    ar_order: usize,
    ma_order: usize,
    include_constant: bool,
) -> StatsResult<(Vec<f64>, Vec<f64>, f64)> {
    let n = x.len();

    // ── Stage 1: Long AR(m) pilot with ridge ─────────────────────────────
    // Pilot order m = ar_order + ma_order + max(ar_order, ma_order) + 1, capped at n/3
    let m_raw = ar_order + ma_order + ar_order.max(ma_order) + 1;
    let m = m_raw.min(n / 3).max(1);

    if n <= m + 1 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: m + 2,
        });
    }

    let n_pilot_eff = n - m;
    // Design: [1, x_{t-1}, …, x_{t-m}]  (constant always included in pilot)
    let n_pilot_cols = m + 1;
    let mut x_pilot = vec![0.0_f64; n_pilot_eff * n_pilot_cols];
    let mut y_pilot = vec![0.0_f64; n_pilot_eff];
    for (row, t) in (m..n).enumerate() {
        y_pilot[row] = x[t];
        x_pilot[row * n_pilot_cols] = 1.0;
        for lag in 1..=m {
            x_pilot[row * n_pilot_cols + lag] = x[t - lag];
        }
    }

    // Auto-select ridge parameter from scale of XᵀX diagonal
    let diag = xtx_diagonal(&x_pilot, n_pilot_eff, n_pilot_cols);
    let lambda_pilot = auto_ridge(&diag, 1e-8);

    let beta_pilot = ridge_solve(&x_pilot, &y_pilot, n_pilot_eff, n_pilot_cols, lambda_pilot)?;

    // Residuals eps_hat (padded with 0 for t < m)
    let pilot_res =
        compute_linear_residuals(&x_pilot, &y_pilot, &beta_pilot, n_pilot_eff, n_pilot_cols);
    let mut eps_hat = vec![0.0_f64; n];
    for (i, &r) in pilot_res.iter().enumerate() {
        eps_hat[m + i] = r;
    }

    // ── Stage 2/3: ARMA OLS ──────────────────────────────────────────────
    // Start row at skip = max(ar_order, ma_order, m) to ensure enough history
    let skip = ar_order.max(ma_order).max(m);
    if n <= skip {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: skip + 1,
        });
    }

    // Columns: [phi_lag_1..ar_order | theta_lag_1..ma_order | (1 if constant)]
    let n_arma_cols = ar_order + ma_order + include_constant as usize;

    if n_arma_cols == 0 {
        return Ok((vec![], vec![], 0.0));
    }

    let n_arma_eff = n - skip;
    let mut x_arma = vec![0.0_f64; n_arma_eff * n_arma_cols];
    let mut y_arma = vec![0.0_f64; n_arma_eff];

    for (row, t) in (skip..n).enumerate() {
        y_arma[row] = x[t];
        let mut col = 0usize;
        for lag in 1..=ar_order {
            x_arma[row * n_arma_cols + col] = x[t - lag];
            col += 1;
        }
        for lag in 1..=ma_order {
            x_arma[row * n_arma_cols + col] = eps_hat[t - lag];
            col += 1;
        }
        if include_constant {
            x_arma[row * n_arma_cols + col] = 1.0;
        }
    }

    // Auto-ridge for ARMA OLS step
    let diag_arma = xtx_diagonal(&x_arma, n_arma_eff, n_arma_cols);
    let lambda_arma = auto_ridge(&diag_arma, 1e-8);

    let coefs = ridge_solve(&x_arma, &y_arma, n_arma_eff, n_arma_cols, lambda_arma)?;

    let phi: Vec<f64> = coefs[..ar_order].to_vec();
    let theta: Vec<f64> = coefs[ar_order..ar_order + ma_order].to_vec();
    let constant = if include_constant {
        coefs[ar_order + ma_order]
    } else {
        0.0
    };

    Ok((phi, theta, constant))
}

// ─────────────────────────────────────────────────────────────────────────────
// CSS coordinate-descent fine-tuning
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinate-descent CSS minimisation with finite-difference gradients.
///
/// Returns `(phi, theta, constant, css, converged, iterations)`.
fn css_finetune(
    x: &[f64],
    phi_init: &[f64],
    theta_init: &[f64],
    constant_init: f64,
    include_constant: bool,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, f64, f64, bool, usize) {
    let ar_order = phi_init.len();
    let ma_order = theta_init.len();
    let n_params = ar_order + ma_order + include_constant as usize;

    let mut phi = phi_init.to_vec();
    let mut theta = theta_init.to_vec();
    let mut constant = constant_init;
    let mut cur_css = css_from_eps(&compute_innovations(x, &phi, &theta, constant));

    if max_iter == 0 || n_params == 0 {
        return (phi, theta, constant, cur_css, true, 0);
    }

    let h = 1e-5_f64;
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let mut max_delta = 0.0_f64;

        for k in 0..n_params {
            let css_plus = css_perturbed(x, &phi, &theta, constant, k, h);
            let css_minus = css_perturbed(x, &phi, &theta, constant, k, -h);
            let grad = (css_plus - css_minus) / (2.0 * h);

            if !grad.is_finite() || grad.abs() < 1e-300 {
                continue;
            }

            // Initial step size: auto-scaled to avoid giant jumps
            let mut lr = (cur_css.abs().sqrt() / (grad.abs() + 1e-8)).min(0.5);
            let mut step_accepted = false;

            for _ in 0..10 {
                let trial_css = css_perturbed(x, &phi, &theta, constant, k, -lr * grad);
                if trial_css < cur_css - 1e-12 * cur_css.abs().max(1e-12) {
                    step_accepted = true;
                    break;
                }
                lr *= 0.5;
                if lr < 1e-15 {
                    break;
                }
            }

            if !step_accepted {
                continue;
            }

            let actual_delta = lr * grad;
            max_delta = max_delta.max(actual_delta.abs());

            if k < ar_order {
                phi[k] -= actual_delta;
            } else if k < ar_order + ma_order {
                theta[k - ar_order] -= actual_delta;
            } else {
                constant -= actual_delta;
            }
        }

        cur_css = css_from_eps(&compute_innovations(x, &phi, &theta, constant));

        if max_delta < tol {
            converged = true;
            break;
        }
    }

    (phi, theta, constant, cur_css, converged, n_iter)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit an ARIMA(p, d, q) model to a time series.
///
/// Uses Hannan-Rissanen 3-stage OLS initialisation (with ridge regularisation
/// for numerical stability) followed by CSS fine-tuning via coordinate descent
/// with finite-difference gradients and step-halving line search.
///
/// # Errors
/// - [`StatsError::InvalidParameter`] if `p + d + q == 0`, the series contains
///   non-finite values, or `d > 2`.
/// - [`StatsError::InsufficientSampleSize`] if the series is too short
///   (`series.len() <= p + d + q + 5`).
pub fn arima_fit(series: &[f64], config: &ArimaConfig) -> StatsResult<ArimaFit> {
    let p = config.p;
    let d = config.d;
    let q = config.q;

    // ── Input validation ─────────────────────────────────────────────────────
    if p + d + q == 0 {
        return Err(StatsError::InvalidParameter {
            name: "order".to_string(),
            reason: "p+d+q must be >= 1".to_string(),
        });
    }
    if d > 2 {
        return Err(StatsError::InvalidParameter {
            name: "d".to_string(),
            reason: "d must be <= 2".to_string(),
        });
    }
    for &v in series {
        if !v.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "series".to_string(),
                reason: "contains non-finite value".to_string(),
            });
        }
    }
    let min_len = p + d + q + 5;
    if series.len() <= min_len {
        return Err(StatsError::InsufficientSampleSize {
            got: series.len(),
            need: min_len + 1,
        });
    }

    // ── Build diff_init (last value at each integration level for un-differencing) ──
    let mut diff_init: Vec<Vec<f64>> = Vec::with_capacity(d);
    {
        let mut cur = series.to_vec();
        for _ in 0..d {
            diff_init.push(vec![*cur.last().unwrap_or(&0.0)]);
            cur = (1..cur.len()).map(|t| cur[t] - cur[t - 1]).collect();
        }
    }

    // ── Differencing ─────────────────────────────────────────────────────────
    let diff_series = apply_difference(series, d);
    let n_diff = diff_series.len(); // = series.len() - d

    // ── Hannan-Rissanen initialisation ───────────────────────────────────────
    let (phi_init, theta_init, constant_init) = if p == 0 && q == 0 {
        // ARIMA(0,d,0): drift = mean of differenced series (if constant requested)
        let c = if config.include_constant {
            diff_series.iter().sum::<f64>() / n_diff as f64
        } else {
            0.0
        };
        (vec![], vec![], c)
    } else {
        hr_init(&diff_series, p, q, config.include_constant)?
    };

    // ── CSS fine-tuning ──────────────────────────────────────────────────────
    let (phi, theta, constant, final_css, converged, _n_iter) = css_finetune(
        &diff_series,
        &phi_init,
        &theta_init,
        constant_init,
        config.include_constant,
        config.max_iter,
        config.tol,
    );

    // ── Information criteria ─────────────────────────────────────────────────
    let n_f = n_diff as f64;
    let dof = (n_diff as isize - p as isize - q as isize).max(1) as f64;
    let sigma2 = (final_css / dof).max(1e-300);
    let n_free = p + q + config.include_constant as usize;
    let aic = n_f * sigma2.ln() + 2.0 * n_free as f64;
    let bic = n_f * sigma2.ln() + n_free as f64 * n_f.ln();

    // ── Last values for forecasting ──────────────────────────────────────────
    let keep_x = p.max(1).min(series.len());
    let last_x = series[series.len() - keep_x..].to_vec();

    let last_eps = if q > 0 {
        let eps_full = compute_innovations(&diff_series, &phi, &theta, constant);
        let keep_e = q.min(eps_full.len());
        eps_full[eps_full.len() - keep_e..].to_vec()
    } else {
        vec![]
    };

    Ok(ArimaFit {
        phi,
        theta,
        constant,
        sigma2,
        css: final_css,
        aic,
        bic,
        p,
        d,
        q,
        n_obs: n_diff,
        last_x,
        last_eps,
        diff_init,
        converged,
    })
}

/// Generate h-step-ahead point forecasts from a fitted ARIMA model.
///
/// Returns an empty `Vec` if `h == 0`.
///
/// Forecasts the differenced scale recursively, then un-differences d times.
pub fn arima_forecast(fit: &ArimaFit, h: usize) -> StatsResult<Vec<f64>> {
    if h == 0 {
        return Ok(vec![]);
    }

    let p = fit.p;
    let q = fit.q;

    // Reconstruct the last p values of the differenced series from fit.last_x
    let last_diff = apply_difference(&fit.last_x, fit.d);

    // x_hist: sliding window of the last p known diff values (oldest first)
    let x_hist: Vec<f64> = if p > 0 {
        let available = last_diff.len();
        if available >= p {
            last_diff[available - p..].to_vec()
        } else {
            let mut v = vec![0.0_f64; p - available];
            v.extend_from_slice(&last_diff);
            v
        }
    } else {
        vec![]
    };

    // e_hist: sliding window of last q known innovations (oldest first)
    let e_hist: Vec<f64> = if q > 0 {
        let available = fit.last_eps.len();
        if available >= q {
            fit.last_eps[available - q..].to_vec()
        } else {
            let mut v = vec![0.0_f64; q - available];
            v.extend_from_slice(&fit.last_eps);
            v
        }
    } else {
        vec![]
    };

    // Recursive forecast on the differenced scale
    let mut x_buf = x_hist;
    let mut e_buf = e_hist;
    let mut diff_forecast = Vec::with_capacity(h);

    for _ in 0..h {
        let x_len = x_buf.len();
        let e_len = e_buf.len();

        let mut val = fit.constant;
        for (i, &phi_i) in fit.phi.iter().enumerate() {
            if x_len > i {
                val += phi_i * x_buf[x_len - 1 - i];
            }
        }
        for (j, &theta_j) in fit.theta.iter().enumerate() {
            if e_len > j {
                val += theta_j * e_buf[e_len - 1 - j];
            }
        }

        diff_forecast.push(val);
        x_buf.push(val);
        e_buf.push(0.0);
    }

    if fit.d == 0 {
        Ok(diff_forecast)
    } else {
        Ok(undifference(&diff_forecast, &fit.diff_init))
    }
}

/// Compute in-sample residuals (innovations) on the differenced scale.
///
/// Returns a `Vec` of length `series.len() - d`.
pub fn arima_residuals(fit: &ArimaFit, series: &[f64]) -> StatsResult<Vec<f64>> {
    for &v in series {
        if !v.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "series".to_string(),
                reason: "contains non-finite value".to_string(),
            });
        }
    }
    let diff_series = apply_difference(series, fit.d);
    Ok(compute_innovations(
        &diff_series,
        &fit.phi,
        &fit.theta,
        fit.constant,
    ))
}

/// Compute in-sample fitted values on the differenced scale.
///
/// Returns `x_diff[t] - ε_t` for each t; length is `series.len() - d`.
pub fn arima_predict_in_sample(fit: &ArimaFit, series: &[f64]) -> StatsResult<Vec<f64>> {
    for &v in series {
        if !v.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "series".to_string(),
                reason: "contains non-finite value".to_string(),
            });
        }
    }
    let diff_series = apply_difference(series, fit.d);
    let eps = compute_innovations(&diff_series, &fit.phi, &fit.theta, fit.constant);
    let fitted: Vec<f64> = diff_series
        .iter()
        .zip(&eps)
        .map(|(&x_t, &e_t)| x_t - e_t)
        .collect();
    Ok(fitted)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Deterministic test-data generators ──────────────────────────────────

    /// Deterministic noise sequence using a minimal LCG.
    /// Values are in (-1, 1) with mean ≈ 0.
    fn lcg_noise(n: usize, seed: u64, scale: f64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = (state >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                (bits * 2.0 - 1.0) * scale // (-scale, scale)
            })
            .collect()
    }

    /// AR(1): x[t] = phi * x[t-1] + ε[t], x[0] = 0.
    fn gen_ar1(n: usize, phi: f64) -> Vec<f64> {
        let eps = lcg_noise(n, 42, 0.5);
        let mut x = vec![0.0_f64; n];
        for t in 1..n {
            x[t] = phi * x[t - 1] + eps[t];
        }
        x
    }

    /// AR(2): x[t] = phi1*x[t-1] + phi2*x[t-2] + ε[t].
    fn gen_ar2(n: usize, phi1: f64, phi2: f64) -> Vec<f64> {
        let eps = lcg_noise(n, 123, 0.5);
        let mut x = vec![0.0_f64; n];
        for t in 2..n {
            x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t];
        }
        x
    }

    /// MA(1): x[t] = theta * ε[t-1] + ε[t].
    fn gen_ma1(n: usize, theta: f64) -> Vec<f64> {
        let eps = lcg_noise(n, 77, 0.3);
        let mut x = vec![0.0_f64; n];
        for t in 1..n {
            x[t] = theta * eps[t - 1] + eps[t];
        }
        x
    }

    // ── Test 1: AR(1) φ=0.8 recovery ─────────────────────────────────────────

    #[test]
    fn ar1_phi_recovery() {
        let series = gen_ar1(500, 0.8);
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("arima_fit AR(1)");
        assert!(
            (fit.phi[0] - 0.8).abs() < 0.05,
            "AR(1) phi recovery: got {}, expected ≈ 0.8",
            fit.phi[0]
        );
    }

    // ── Test 2: MA(1) θ=0.5 recovery ─────────────────────────────────────────

    #[test]
    fn ma1_theta_recovery() {
        let series = gen_ma1(500, 0.5);
        let config = ArimaConfig {
            p: 0,
            d: 0,
            q: 1,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("arima_fit MA(1)");
        assert!(
            (fit.theta[0] - 0.5).abs() < 0.1,
            "MA(1) theta recovery: got {}, expected ≈ 0.5",
            fit.theta[0]
        );
    }

    // ── Test 3: AIC/BIC finite and negative for stationary data ──────────────

    #[test]
    fn aic_bic_finite_and_negative() {
        let series = gen_ar1(300, 0.5);
        let config = ArimaConfig::default();
        let fit = arima_fit(&series, &config).expect("fit");
        assert!(
            fit.aic.is_finite() && fit.aic < 0.0,
            "AIC should be finite and < 0, got {}",
            fit.aic
        );
        assert!(
            fit.bic.is_finite() && fit.bic < 0.0,
            "BIC should be finite and < 0, got {}",
            fit.bic
        );
    }

    // ── Test 4: residuals length = series.len() - d ───────────────────────────

    #[test]
    fn residuals_length_correct() {
        let series = gen_ar1(200, 0.6);
        let config = ArimaConfig {
            p: 1,
            d: 1,
            q: 0,
            include_constant: true,
            max_iter: 100,
            tol: 1e-6,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let res = arima_residuals(&fit, &series).expect("residuals");
        assert_eq!(
            res.len(),
            series.len() - config.d,
            "residuals length mismatch"
        );
    }

    // ── Test 5: Residual mean ≈ 0 ─────────────────────────────────────────────

    #[test]
    fn residual_mean_near_zero() {
        let series = gen_ar1(400, 0.7);
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let res = arima_residuals(&fit, &series).expect("residuals");
        let mean = res.iter().sum::<f64>() / res.len() as f64;
        assert!(mean.abs() < 0.1, "Residual mean too large: {mean}");
    }

    // ── Test 6: ARIMA(0,1,0) residuals ≈ first differences ───────────────────

    #[test]
    fn arima_010_residuals_match_first_diff() {
        let series = gen_ar1(100, 0.5);
        let config = ArimaConfig {
            p: 0,
            d: 1,
            q: 0,
            include_constant: false,
            max_iter: 0,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let res = arima_residuals(&fit, &series).expect("residuals");
        let first_diff: Vec<f64> = (1..series.len())
            .map(|t| series[t] - series[t - 1])
            .collect();
        assert_eq!(res.len(), first_diff.len());
        for (i, (&r, &d_val)) in res.iter().zip(&first_diff).enumerate() {
            assert!(
                (r - d_val).abs() < 1e-10,
                "ARIMA(0,1,0) residual[{i}] = {r}, first_diff = {d_val}"
            );
        }
    }

    // ── Test 7: forecast h=0 returns empty ───────────────────────────────────

    #[test]
    fn forecast_h0_empty() {
        let series = gen_ar1(100, 0.6);
        let config = ArimaConfig::default();
        let fit = arima_fit(&series, &config).expect("fit");
        let fc = arima_forecast(&fit, 0).expect("forecast");
        assert!(fc.is_empty(), "h=0 forecast should be empty");
    }

    // ── Test 8: forecast h=5 returns 5 finite values ─────────────────────────

    #[test]
    fn forecast_h5_finite() {
        let series = gen_ar1(200, 0.5);
        let config = ArimaConfig::default();
        let fit = arima_fit(&series, &config).expect("fit");
        let fc = arima_forecast(&fit, 5).expect("forecast h=5");
        assert_eq!(fc.len(), 5);
        for (i, &v) in fc.iter().enumerate() {
            assert!(v.is_finite(), "forecast[{i}] = {v} is not finite");
        }
    }

    // ── Test 9: CSS after fine-tuning < naive variance * n ────────────────────

    #[test]
    fn css_decreases_after_finetuning() {
        let series = gen_ar1(300, 0.7);
        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;
        let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        assert!(
            fit.css < variance * n,
            "CSS ({}) should be < initial variance * n ({})",
            fit.css,
            variance * n
        );
    }

    // ── Test 10: in-sample predictions same length as differenced series ──────

    #[test]
    fn in_sample_predictions_length() {
        let series = gen_ar1(150, 0.5);
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 50,
            tol: 1e-6,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let pred = arima_predict_in_sample(&fit, &series).expect("predict");
        // d=0: length equals series.len()
        assert_eq!(pred.len(), series.len());
    }

    // ── Test 11: constant near zero for zero-mean data ────────────────────────

    #[test]
    fn constant_near_zero_for_zero_mean() {
        let series = gen_ar1(300, 0.8); // zero-mean by construction
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        assert!(
            fit.constant.abs() < 0.5,
            "Constant should be near 0 for zero-mean data, got {}",
            fit.constant
        );
    }

    // ── Test 12: p+d+q=0 → InvalidParameter ──────────────────────────────────

    #[test]
    fn zero_order_returns_error() {
        let series = gen_ar1(50, 0.5);
        let config = ArimaConfig {
            p: 0,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 100,
            tol: 1e-7,
        };
        let result = arima_fit(&series, &config);
        assert!(
            matches!(result, Err(StatsError::InvalidParameter { .. })),
            "Expected InvalidParameter for p+d+q=0"
        );
    }

    // ── Test 13: series too short → InsufficientSampleSize ───────────────────

    #[test]
    fn series_too_short_returns_error() {
        let series = vec![1.0, 2.0, 3.0];
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 1,
            include_constant: true,
            max_iter: 10,
            tol: 1e-6,
        };
        let result = arima_fit(&series, &config);
        assert!(
            matches!(result, Err(StatsError::InsufficientSampleSize { .. })),
            "Expected InsufficientSampleSize"
        );
    }

    // ── Test 14: non-finite series → InvalidParameter ─────────────────────────

    #[test]
    fn nonfinite_series_returns_error() {
        let mut series = gen_ar1(100, 0.5);
        series[50] = f64::NAN;
        let config = ArimaConfig::default();
        let result = arima_fit(&series, &config);
        assert!(
            matches!(result, Err(StatsError::InvalidParameter { .. })),
            "Expected InvalidParameter for NaN in series"
        );
    }

    // ── Test 15: d=3 → InvalidParameter ──────────────────────────────────────

    #[test]
    fn d_greater_than_2_returns_error() {
        let series = gen_ar1(100, 0.5);
        let config = ArimaConfig {
            p: 1,
            d: 3,
            q: 0,
            include_constant: true,
            max_iter: 10,
            tol: 1e-6,
        };
        let result = arima_fit(&series, &config);
        assert!(
            matches!(result, Err(StatsError::InvalidParameter { .. })),
            "Expected InvalidParameter for d=3"
        );
    }

    // ── Test 16: AR(1) on white noise → phi ≈ 0 ──────────────────────────────

    #[test]
    fn ar1_on_white_noise_phi_near_zero() {
        // White noise from LCG: by definition uncorrelated across lags
        let series = lcg_noise(500, 999, 1.0);
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        assert!(
            fit.phi[0].abs() < 0.2,
            "AR(1) on white noise: phi[0] = {}, expected |phi| < 0.2",
            fit.phi[0]
        );
    }

    // ── Test 17: sigma2 > 0 ───────────────────────────────────────────────────

    #[test]
    fn sigma2_always_positive() {
        let series = gen_ar1(200, 0.6);
        let config = ArimaConfig::default();
        let fit = arima_fit(&series, &config).expect("fit");
        assert!(fit.sigma2 > 0.0, "sigma2 must be > 0, got {}", fit.sigma2);
    }

    // ── Test 18: converged is a bool ──────────────────────────────────────────

    #[test]
    fn converged_field_set() {
        let series = gen_ar1(100, 0.4);
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 5,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let _: bool = fit.converged; // type-check only
    }

    // ── Test 19: AR(2) parameter recovery ────────────────────────────────────

    #[test]
    fn ar2_phi_recovery() {
        let series = gen_ar2(600, 0.5, 0.2);
        let config = ArimaConfig {
            p: 2,
            d: 0,
            q: 0,
            include_constant: true,
            max_iter: 200,
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("arima_fit AR(2)");
        assert!(
            (fit.phi[0] - 0.5).abs() < 0.1,
            "AR(2) phi[0] recovery: got {}, expected ≈ 0.5",
            fit.phi[0]
        );
        assert!(
            (fit.phi[1] - 0.2).abs() < 0.1,
            "AR(2) phi[1] recovery: got {}, expected ≈ 0.2",
            fit.phi[1]
        );
    }

    // ── Test 20: constant ≈ series mean for near-constant + slow MA ───────────

    #[test]
    fn arima001_constant_near_mean() {
        let n = 200usize;
        // LCG noise around mean 2.0
        let eps = lcg_noise(n, 55, 0.05);
        let series: Vec<f64> = eps.iter().map(|&e| 2.0 + e).collect();
        let config = ArimaConfig {
            p: 0,
            d: 0,
            q: 1,
            include_constant: true,
            max_iter: 0, // HR init only
            tol: 1e-7,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let series_mean = series.iter().sum::<f64>() / n as f64;
        assert!(
            (fit.constant - series_mean).abs() < 0.1,
            "Constant ({}) should ≈ mean ({})",
            fit.constant,
            series_mean
        );
    }

    // ── Test 21: ARIMA(1,1,0) forecast is finite ─────────────────────────────

    #[test]
    fn arima_110_forecast_finite() {
        let series = gen_ar1(150, 0.5);
        let config = ArimaConfig {
            p: 1,
            d: 1,
            q: 0,
            include_constant: true,
            max_iter: 50,
            tol: 1e-6,
        };
        let fit = arima_fit(&series, &config).expect("fit");
        let fc = arima_forecast(&fit, 5).expect("forecast");
        assert_eq!(fc.len(), 5);
        for (i, &v) in fc.iter().enumerate() {
            assert!(
                v.is_finite(),
                "forecast[{i}] = {v} is not finite for ARIMA(1,1,0)"
            );
        }
    }

    // ── Test 22: Determinism ──────────────────────────────────────────────────

    #[test]
    fn deterministic_fitting() {
        let series = gen_ar1(200, 0.7);
        let config = ArimaConfig {
            p: 1,
            d: 0,
            q: 1,
            include_constant: true,
            max_iter: 50,
            tol: 1e-6,
        };
        let fit1 = arima_fit(&series, &config).expect("fit1");
        let fit2 = arima_fit(&series, &config).expect("fit2");
        assert_eq!(fit1.phi, fit2.phi, "phi differs across runs");
        assert_eq!(fit1.theta, fit2.theta, "theta differs across runs");
        assert!(
            (fit1.constant - fit2.constant).abs() < 1e-14,
            "constant differs"
        );
        assert!((fit1.css - fit2.css).abs() < 1e-12, "css differs");
    }
}
