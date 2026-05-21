//! Quantile regression via Iteratively Reweighted L1 (IRLS) / weighted least-squares.
//!
//! Implements **Koenker-Bassett (1978)** quantile regression at quantile τ ∈ (0, 1).
//! Minimises the asymmetric "check" (pinball) loss:
//!
//! ```text
//! L(β) = Σ ρ_τ(y_i − x_i^T β)
//! where ρ_τ(r) = τ·max(r, 0) + (1-τ)·max(-r, 0)
//! ```
//!
//! **Algorithm — Iteratively Reweighted L1 (Hunter & Lange 2000):**
//!
//! The idea is to linearize the L1 norm: |r_i| ≈ r_i² / |r_{i,prev}|.
//! Asymmetrising the weights by τ/|r| (r > 0) or (1-τ)/|r| (r < 0) yields
//! the quantile-WLS surrogate:
//!
//! 1. Initialise β by OLS (with intercept column if `intercept = true`).
//! 2. Compute residuals r_i = y_i - x_i^T β.
//! 3. Set `w_i = τ / max(|r_i|, ε)` if `r_i ≥ 0`,
//!    or `(1-τ) / max(|r_i|, ε)` otherwise.
//! 4. Solve WLS: β_new = (X^T W X)^{-1} X^T W y.
//! 5. Repeat until ||β_new - β||_2 < tol or max_iter reached.
//!
//! # References
//! - Koenker & Bassett (1978) "Regression Quantiles". *Econometrica* 46(1):33-50.
//! - Hunter & Lange (2000) "Quantile Regression via an MM Algorithm".
//! - Koenker (2005) *Quantile Regression*. Cambridge.

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for quantile regression.
#[derive(Debug, Clone)]
pub struct QuantileConfig {
    /// Quantile level ∈ (0, 1).  Default 0.5 (median regression).
    pub tau: f64,
    /// Maximum IRLS iterations. Default 200.
    pub max_iter: usize,
    /// Convergence tolerance on ||Δβ||₂. Default 1e-8.
    pub tol: f64,
    /// Fit an intercept term (prepends a column of ones). Default true.
    pub intercept: bool,
    /// Regularisation floor to avoid division by zero. Default 1e-6.
    pub eps: f64,
}

impl Default for QuantileConfig {
    fn default() -> Self {
        Self {
            tau: 0.5,
            max_iter: 200,
            tol: 1e-8,
            intercept: true,
            eps: 1e-6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted quantile-regression model.
#[derive(Debug, Clone)]
pub struct QuantileFit {
    /// Regression coefficients (length = n_features; excludes intercept).
    pub coefficients: Vec<f64>,
    /// Fitted intercept (0.0 when `intercept = false`).
    pub intercept_val: f64,
    /// Residuals r_i = y_i − ŷ_i.
    pub residuals: Vec<f64>,
    /// Quantile (pinball) loss Σ ρ_τ(r_i).
    pub quantile_loss: f64,
    /// Number of IRLS iterations executed.
    pub n_iter: usize,
    /// Whether the algorithm converged to within `tol`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pinball (check) loss ρ_τ(r) = τ·r if r ≥ 0, (τ-1)·r if r < 0.
#[inline]
fn check_loss(r: f64, tau: f64) -> f64 {
    if r >= 0.0 { tau * r } else { (tau - 1.0) * r }
}

/// Build the augmented design matrix: if `intercept`, prepend a column of ones.
/// Returns (X_aug, n_aug_features).
fn build_design(x: &[f64], n: usize, p: usize, intercept: bool) -> (Vec<f64>, usize) {
    if !intercept {
        return (x.to_vec(), p);
    }
    let p_aug = p + 1;
    let mut xa = vec![0.0; n * p_aug];
    for i in 0..n {
        xa[i * p_aug] = 1.0;
        for j in 0..p {
            xa[i * p_aug + 1 + j] = x[i * p + j];
        }
    }
    (xa, p_aug)
}

/// Solve the weighted least-squares problem β = (X^T W X)^{-1} X^T W y.
///
/// `w` is a length-n slice of non-negative diagonal weights.
fn solve_wls(xa: &[f64], y: &[f64], w: &[f64], n: usize, p: usize) -> StatsResult<Vec<f64>> {
    // X^T W X  (p × p)
    let mut xtwx = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let mut acc = 0.0;
            for k in 0..n {
                acc += xa[k * p + i] * w[k] * xa[k * p + j];
            }
            xtwx[i * p + j] = acc;
            xtwx[j * p + i] = acc;
        }
    }
    // X^T W y  (p)
    let mut xtwy = vec![0.0; p];
    for i in 0..p {
        let mut acc = 0.0;
        for k in 0..n {
            acc += xa[k * p + i] * w[k] * y[k];
        }
        xtwy[i] = acc;
    }
    // β = (X^T W X)^{-1} X^T W y
    let xtwx_inv = matrix_inverse_lu(&xtwx, p)?;
    let mut beta = vec![0.0; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += xtwx_inv[i * p + j] * xtwy[j];
        }
        beta[i] = acc;
    }
    Ok(beta)
}

/// Compute residuals r_i = y_i - Σ_j xa[i,j] * beta[j].
fn compute_residuals(xa: &[f64], y: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let fitted: f64 = (0..p).map(|j| xa[i * p + j] * beta[j]).sum();
            y[i] - fitted
        })
        .collect()
}

/// L2 norm of the difference between two vectors.
fn l2_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a quantile-regression model at quantile τ.
///
/// # Parameters
/// - `x` — row-major design matrix of shape `(n_samples, n_features)`.
/// - `y` — response vector of length `n_samples`.
/// - `n_samples`, `n_features` — dimensions of `x`.
/// - `cfg` — algorithm configuration including τ.
///
/// # Returns
/// A [`QuantileFit`] containing coefficients, intercept, residuals, and
/// diagnostic information.
pub fn quantile_fit(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    cfg: &QuantileConfig,
) -> StatsResult<QuantileFit> {
    // --- Input validation ---
    if n_samples == 0 {
        return Err(StatsError::EmptyInput);
    }
    if x.len() != n_samples * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_samples, n_features],
            got: vec![x.len()],
        });
    }
    if y.len() != n_samples {
        return Err(StatsError::DimensionMismatch {
            a: y.len(),
            b: n_samples,
        });
    }
    if !(cfg.tau > 0.0 && cfg.tau < 1.0) {
        return Err(StatsError::InvalidParameter {
            name: "tau".to_string(),
            reason: format!("must be in (0, 1), got {}", cfg.tau),
        });
    }
    if cfg.eps <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "eps".to_string(),
            reason: "must be positive".to_string(),
        });
    }

    // Check all values finite
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
    }

    let tau = cfg.tau;
    let eps = cfg.eps;

    // Build augmented design matrix
    let (xa, p_aug) = build_design(x, n_samples, n_features, cfg.intercept);

    // Need at least p_aug observations
    if n_samples < p_aug {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: p_aug,
        });
    }

    // --- Step 1: Initialise with unit weights (OLS) ---
    let w_init = vec![1.0; n_samples];
    let mut beta = solve_wls(&xa, y, &w_init, n_samples, p_aug).map_err(|e| match e {
        StatsError::SingularMatrix(_) => {
            // Fallback: use zero start if design is singular
            StatsError::SingularMatrix(
                "initial OLS singular; try a larger sample or remove collinear features"
                    .to_string(),
            )
        }
        other => other,
    })?;

    let mut converged = false;
    let mut n_iter = 0usize;

    // --- IRLS main loop ---
    for iter in 1..=cfg.max_iter {
        n_iter = iter;

        let r = compute_residuals(&xa, y, &beta, n_samples, p_aug);

        // Compute asymmetric IRLS weights
        let w: Vec<f64> = r
            .iter()
            .map(|&ri| {
                let abs_r = ri.abs().max(eps);
                if ri >= 0.0 {
                    tau / abs_r
                } else {
                    (1.0 - tau) / abs_r
                }
            })
            .collect();

        let beta_new = match solve_wls(&xa, y, &w, n_samples, p_aug) {
            Ok(b) => b,
            Err(StatsError::SingularMatrix(_)) => {
                // Slight perturbation: add tiny ridge and retry
                let mut xtwx_ridge = vec![0.0; p_aug * p_aug];
                for i in 0..p_aug {
                    for j in i..p_aug {
                        let mut acc = 0.0;
                        for k in 0..n_samples {
                            acc += xa[k * p_aug + i] * w[k] * xa[k * p_aug + j];
                        }
                        xtwx_ridge[i * p_aug + j] = acc;
                        xtwx_ridge[j * p_aug + i] = acc;
                    }
                    // Ridge regularisation
                    xtwx_ridge[i * p_aug + i] += 1e-10;
                }
                let xtwx_inv = matrix_inverse_lu(&xtwx_ridge, p_aug)?;
                let mut xtwy = vec![0.0; p_aug];
                for i in 0..p_aug {
                    let mut acc = 0.0;
                    for k in 0..n_samples {
                        acc += xa[k * p_aug + i] * w[k] * y[k];
                    }
                    xtwy[i] = acc;
                }
                let mut b = vec![0.0; p_aug];
                for i in 0..p_aug {
                    let mut acc = 0.0;
                    for j in 0..p_aug {
                        acc += xtwx_inv[i * p_aug + j] * xtwy[j];
                    }
                    b[i] = acc;
                }
                b
            }
            Err(e) => return Err(e),
        };

        let delta = l2_diff(&beta_new, &beta);
        beta = beta_new;

        if delta < cfg.tol {
            converged = true;
            break;
        }
    }

    // --- Final residuals and quantile loss ---
    let residuals = compute_residuals(&xa, y, &beta, n_samples, p_aug);
    let quantile_loss: f64 = residuals.iter().map(|&r| check_loss(r, tau)).sum();

    // Split intercept from slope coefficients
    let (intercept_val, coefficients) = if cfg.intercept {
        (beta[0], beta[1..].to_vec())
    } else {
        (0.0, beta)
    };

    Ok(QuantileFit {
        coefficients,
        intercept_val,
        residuals,
        quantile_loss,
        n_iter,
        converged,
    })
}

/// Predict responses for new observations using a fitted quantile model.
///
/// # Parameters
/// - `fit` — previously fitted [`QuantileFit`].
/// - `x_new` — row-major matrix of shape `(n_new, n_features)`.
/// - `n_new` — number of new observations.
pub fn quantile_predict(fit: &QuantileFit, x_new: &[f64], n_new: usize) -> StatsResult<Vec<f64>> {
    let n_features = fit.coefficients.len();
    if n_new == 0 {
        return Ok(Vec::new());
    }
    if x_new.len() != n_new * n_features {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n_new, n_features],
            got: vec![x_new.len()],
        });
    }
    let preds: Vec<f64> = (0..n_new)
        .map(|i| {
            let mut yhat = fit.intercept_val;
            for j in 0..n_features {
                yhat += x_new[i * n_features + j] * fit.coefficients[j];
            }
            yhat
        })
        .collect();
    Ok(preds)
}

/// Fit quantile regression at multiple quantile levels (a quantile band).
///
/// Returns a vector of [`QuantileFit`], one per τ in `taus`.
/// All fits share the same `x`, `y`, `n`, `p` with default config except τ.
pub fn quantile_band(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    taus: &[f64],
) -> StatsResult<Vec<QuantileFit>> {
    let mut fits = Vec::with_capacity(taus.len());
    for &tau in taus {
        let cfg = QuantileConfig {
            tau,
            ..Default::default()
        };
        fits.push(quantile_fit(x, y, n, p, &cfg)?);
    }
    Ok(fits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric dataset y = 3*x + 5 + noise (symmetric noise → median = OLS).
    fn sym_dataset(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Deterministic symmetric noise: alternating ±1
        let xs: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                3.0 * x + 5.0 + noise
            })
            .collect();
        (xs, ys)
    }

    #[test]
    fn tau_half_recovers_median() {
        let (xs, ys) = sym_dataset(40);
        let cfg = QuantileConfig {
            tau: 0.5,
            ..Default::default()
        };
        let fit = quantile_fit(&xs, &ys, 40, 1, &cfg).expect("ok");
        // Slope should be close to 3.0, intercept close to 5.0
        assert!(
            (fit.coefficients[0] - 3.0).abs() < 0.5,
            "slope={}",
            fit.coefficients[0]
        );
        assert!(
            (fit.intercept_val - 5.0).abs() < 0.5,
            "intercept={}",
            fit.intercept_val
        );
    }

    #[test]
    fn tau_invalid_error() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0];
        let cfg0 = QuantileConfig {
            tau: 0.0,
            ..Default::default()
        };
        let cfg1 = QuantileConfig {
            tau: 1.0,
            ..Default::default()
        };
        let cfg_neg = QuantileConfig {
            tau: -0.1,
            ..Default::default()
        };
        assert!(quantile_fit(&xs, &ys, 3, 1, &cfg0).is_err());
        assert!(quantile_fit(&xs, &ys, 3, 1, &cfg1).is_err());
        assert!(quantile_fit(&xs, &ys, 3, 1, &cfg_neg).is_err());
    }

    #[test]
    fn quantile_output_shape() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let cfg = QuantileConfig::default();
        let fit = quantile_fit(&xs, &ys, 5, 1, &cfg).expect("ok");
        // intercept=true: coefficients has n_features=1 element
        assert_eq!(fit.coefficients.len(), 1);
        assert_eq!(fit.residuals.len(), 5);
    }

    #[test]
    fn quantile_convergence() {
        let n = 50;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
        let cfg = QuantileConfig {
            tau: 0.5,
            max_iter: 300,
            tol: 1e-8,
            ..Default::default()
        };
        let fit = quantile_fit(&xs, &ys, n, 1, &cfg).expect("ok");
        assert!(fit.converged, "should converge on perfect line");
    }

    #[test]
    fn quantile_loss_finite() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ys: Vec<f64> = xs.iter().map(|&x| x + 0.5).collect();
        let cfg = QuantileConfig::default();
        let fit = quantile_fit(&xs, &ys, 8, 1, &cfg).expect("ok");
        assert!(fit.quantile_loss.is_finite() && fit.quantile_loss >= 0.0);
    }

    #[test]
    fn quantile_lower_tail_vs_upper_tail() {
        // y = x + asymmetric heavy upper tail → tau=0.1 < tau=0.9 predictions
        let n = 60;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                // upper-skewed: occasional large positive outliers
                let outlier = if i % 10 == 0 { 3.0 } else { 0.0 };
                x + outlier
            })
            .collect();

        let cfg_lo = QuantileConfig {
            tau: 0.1,
            ..Default::default()
        };
        let cfg_hi = QuantileConfig {
            tau: 0.9,
            ..Default::default()
        };
        let fit_lo = quantile_fit(&xs, &ys, n, 1, &cfg_lo).expect("ok");
        let fit_hi = quantile_fit(&xs, &ys, n, 1, &cfg_hi).expect("ok");

        // At the median x=0.5, tau=0.9 prediction should be higher than tau=0.1
        let x_mid = vec![0.5_f64];
        let pred_lo = quantile_predict(&fit_lo, &x_mid, 1).expect("ok");
        let pred_hi = quantile_predict(&fit_hi, &x_mid, 1).expect("ok");
        assert!(
            pred_hi[0] > pred_lo[0],
            "tau=0.9 pred {} should exceed tau=0.1 pred {}",
            pred_hi[0],
            pred_lo[0]
        );
    }

    #[test]
    fn quantile_band_shape() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ys: Vec<f64> = xs.to_vec();
        let taus = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let fits = quantile_band(&xs, &ys, 6, 1, &taus).expect("ok");
        assert_eq!(fits.len(), taus.len());
    }

    #[test]
    fn quantile_predict_shape() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x).collect();
        let cfg = QuantileConfig::default();
        let fit = quantile_fit(&xs, &ys, 5, 1, &cfg).expect("ok");
        let x_new = vec![6.0, 7.0, 8.0];
        let preds = quantile_predict(&fit, &x_new, 3).expect("ok");
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn quantile_empty_error() {
        let cfg = QuantileConfig::default();
        // n_samples = 0 should return EmptyInput
        let res = quantile_fit(&[], &[], 0, 1, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn quantile_recovers_intercept() {
        // y = c + 0*x: the intercept should recover c when x varies but y is constant.
        // Use spread x values so the design is full-rank; y is constant.
        let n = 20;
        let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let c = 7.42;
        let ys = vec![c; n];
        let cfg = QuantileConfig {
            tau: 0.5,
            max_iter: 300,
            ..Default::default()
        };
        let fit = quantile_fit(&xs, &ys, n, 1, &cfg).expect("ok");
        // Prediction at x=0: intercept_val + coeff * 0 = intercept_val ≈ c
        let x_query = vec![0.0_f64];
        let pred = quantile_predict(&fit, &x_query, 1).expect("ok");
        assert!((pred[0] - c).abs() < 1.0, "expected ≈{} got {}", c, pred[0]);
    }

    #[test]
    fn quantile_no_intercept_mode() {
        // Without intercept, coefficients.len() = n_features
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x).collect();
        let cfg = QuantileConfig {
            tau: 0.5,
            intercept: false,
            ..Default::default()
        };
        let fit = quantile_fit(&xs, &ys, 5, 1, &cfg).expect("ok");
        assert_eq!(fit.coefficients.len(), 1);
        assert_eq!(fit.intercept_val, 0.0);
        // Slope should be ≈3
        assert!((fit.coefficients[0] - 3.0).abs() < 0.5);
    }

    #[test]
    fn quantile_band_ordering() {
        // Median-crossing: at any given x, lower τ should give lower prediction than higher τ
        // Use sufficient spread in data
        let n = 80;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let mut rng_state: u64 = 12345;
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| {
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
                x * 2.0 + (u - 0.5) * 0.4 // small uniform noise
            })
            .collect();
        let taus = vec![0.1, 0.5, 0.9];
        let fits = quantile_band(&xs, &ys, n, 1, &taus).expect("ok");

        // At x=0.5: pred[τ=0.1] ≤ pred[τ=0.5] ≤ pred[τ=0.9] (approximately)
        let x_mid = vec![0.5_f64];
        let p10 = quantile_predict(&fits[0], &x_mid, 1).expect("ok")[0];
        let p50 = quantile_predict(&fits[1], &x_mid, 1).expect("ok")[0];
        let p90 = quantile_predict(&fits[2], &x_mid, 1).expect("ok")[0];

        assert!(
            p10 <= p50 + 0.1,
            "10th pct={} should be ≤ 50th pct={}",
            p10,
            p50
        );
        assert!(
            p50 <= p90 + 0.1,
            "50th pct={} should be ≤ 90th pct={}",
            p50,
            p90
        );
    }
}
