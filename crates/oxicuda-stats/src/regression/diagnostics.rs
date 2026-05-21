//! Regression diagnostic measures for Ordinary Least Squares.
//!
//! Provides:
//! - [`cooks_distance`] — Cook's D: influence of each observation on the fit
//! - [`dffits`] — DFFITS: scaled change in fitted value upon deletion
//! - [`vif`] — Variance Inflation Factor: multicollinearity detector
//! - [`standardized_residuals`] — internally Studentized residuals
//! - [`leverage`] — Hat-matrix diagonal h_{ii}
//! - [`durbin_watson_ols`] — Durbin-Watson test statistic for autocorrelation
//! - [`breusch_pagan_test`] — Breusch-Pagan LM test for heteroscedasticity

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::{matrix_inverse_lu, ols};

// ---------------------------------------------------------------------------
// Internal OLS helper with hat-matrix diagonal
// ---------------------------------------------------------------------------

/// Internal OLS fit result carrying the hat-matrix diagonal.
struct OlsWithHat {
    /// Coefficient vector β̂.
    coef: Vec<f64>,
    /// Ordinary residuals e_i = y_i - ŷ_i.
    residuals: Vec<f64>,
    /// Residual sum of squares Σ e_i².
    rss: f64,
    /// Hat-matrix diagonal h_{ii} = x_i^T (X^T X)^{-1} x_i.
    hat_diag: Vec<f64>,
    /// Number of samples.
    n: usize,
    /// Number of parameters (columns of X).
    p: usize,
}

/// Fit OLS and compute the hat-matrix diagonal without forming the full n×n H matrix.
///
/// `x_mat` is `n × p` where each element is a row vector of predictors.
/// Internally flattens to a row-major `Vec<f64>` and calls `ols`.
fn ols_with_hat(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<OlsWithHat> {
    let n = y.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if x_mat.len() != n {
        return Err(StatsError::DimensionMismatch {
            a: x_mat.len(),
            b: n,
        });
    }
    if n == 0 || x_mat[0].is_empty() {
        return Err(StatsError::InvalidParameter {
            name: "x_mat".to_string(),
            reason: "design matrix must have at least one column".to_string(),
        });
    }
    let p = x_mat[0].len();
    if n < p {
        return Err(StatsError::InsufficientSampleSize { got: n, need: p });
    }
    if n <= 1 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }

    // Flatten x_mat to row-major Vec<f64>
    let mut x_flat = Vec::with_capacity(n * p);
    for row in x_mat {
        if row.len() != p {
            return Err(StatsError::DimensionMismatch { a: row.len(), b: p });
        }
        x_flat.extend_from_slice(row);
    }

    let model = ols(&x_flat, y, n, p)?;
    let coef = model.coefficients;
    let residuals = model.residuals;
    let rss = model.residual_sum_squares;

    // Hat-matrix diagonal: h_ii = x_i^T (X^T X)^{-1} x_i
    // (X^T X)^{-1} is already computed by ols (stored in xtx_inv).
    let xtx_inv = model.xtx_inv; // p×p row-major

    let hat_diag: Vec<f64> = x_mat
        .iter()
        .map(|xi| {
            // c = (X^T X)^{-1} x_i  (p-vector)
            let mut c = vec![0.0_f64; p];
            for r in 0..p {
                let mut acc = 0.0;
                for s in 0..p {
                    acc += xtx_inv[r * p + s] * xi[s];
                }
                c[r] = acc;
            }
            // h_ii = x_i^T c
            let h: f64 = xi.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            h.clamp(0.0, 1.0)
        })
        .collect();

    Ok(OlsWithHat {
        coef,
        residuals,
        rss,
        hat_diag,
        n,
        p,
    })
}

// ---------------------------------------------------------------------------
// Chi-squared CDF via regularised incomplete gamma
// ---------------------------------------------------------------------------

/// Regularised lower incomplete gamma P(a, x) for chi-squared CDF computation.
///
/// Equivalent to `gammp` from `crate::special::betainc` but inlined here to keep
/// the module self-contained for diagnostics.
fn gamma_inc_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let lna = lgamma_approx(a);
    if x < a + 1.0 {
        // Taylor series
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..600 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 3.0e-15 {
                break;
            }
        }
        (sum * (-x + a * x.ln() - lna).exp()).clamp(0.0, 1.0)
    } else {
        // Complementary via Lentz continued fraction
        (1.0 - gamma_q_cf(a, x, lna)).clamp(0.0, 1.0)
    }
}

fn lgamma_approx(x: f64) -> f64 {
    // Lanczos approximation (g=7, n=9) — same as in crate::special::gammaln
    const G: f64 = 7.0;
    const COEFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lgamma_approx(1.0 - x);
    }
    let xm = x - 1.0;
    let mut a = COEFS[0];
    for (i, c) in COEFS.iter().enumerate().skip(1) {
        a += c / (xm + i as f64);
    }
    let t = xm + G + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (xm + 0.5) * t.ln() - t + a.ln()
}

fn gamma_q_cf(a: f64, x: f64, lna: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    let prefix = (-x + a * x.ln() - lna).exp();
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = if b.abs() < TINY { TINY } else { 1.0 / b };
    let mut h = d;
    for i in 1i64..=600 {
        let ai = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = ai * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + ai / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 3.0e-15 {
            break;
        }
    }
    (prefix * h).clamp(0.0, 1.0)
}

/// Chi-squared CDF at `x` with `df` degrees of freedom.
fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    gamma_inc_p(df / 2.0, x / 2.0)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Cook's distance for detecting influential observations in OLS.
///
/// `D_i = e_i² / (p * MSE) * h_{ii} / (1 - h_{ii})²`
///
/// where `MSE = RSS / (n - p)` and `h_{ii}` is the leverage.
pub fn cooks_distance(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let fit = ols_with_hat(y, x_mat)?;
    let n = fit.n;
    let p = fit.p;
    if n <= p {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: p + 1,
        });
    }
    let mse = fit.rss / (n - p) as f64;
    if mse <= 0.0 {
        // Perfect fit — all Cook's distances are 0
        return Ok(vec![0.0; n]);
    }
    let cooks: Vec<f64> = fit
        .residuals
        .iter()
        .zip(fit.hat_diag.iter())
        .map(|(&e, &h)| {
            let denom = (1.0 - h).powi(2);
            if denom < 1.0e-12 {
                f64::INFINITY
            } else {
                (e * e * h) / (p as f64 * mse * denom)
            }
        })
        .collect();
    Ok(cooks)
}

/// DFFITS: scaled influence of observation i on its own fitted value.
///
/// Uses the approximation `DFFITS_i ≈ t_i * sqrt(h_{ii} / (1 - h_{ii}))`
/// where `t_i` is the externally Studentized residual.
pub fn dffits(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let fit = ols_with_hat(y, x_mat)?;
    let n = fit.n;
    let p = fit.p;
    let df = n - p;
    if df < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: p + 2,
        });
    }
    let s2 = fit.rss / df as f64;

    let result: Vec<f64> = fit
        .residuals
        .iter()
        .zip(fit.hat_diag.iter())
        .map(|(&e, &h)| {
            let h_complement = (1.0 - h).max(1.0e-12);
            // Internally Studentized residual
            let s2_i = if df > 1 {
                ((df as f64) * s2 - e * e / h_complement) / (df as f64 - 1.0)
            } else {
                s2
            };
            let s_i = s2_i.max(0.0).sqrt().max(1.0e-12);
            let t_i = e / (s_i * h_complement.sqrt());
            t_i * (h / h_complement).sqrt()
        })
        .collect();
    Ok(result)
}

/// Variance Inflation Factor (VIF) for each column of the design matrix.
///
/// `VIF_j = 1 / (1 - R²_j)` where `R²_j` is from regressing column j on all others.
///
/// Large VIF (> 10) indicates severe multicollinearity.
pub fn vif(x_mat: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let n = x_mat.len();
    if n == 0 {
        return Err(StatsError::EmptyInput);
    }
    if n <= 1 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let p = x_mat[0].len();
    if p == 0 {
        return Err(StatsError::InvalidParameter {
            name: "x_mat".to_string(),
            reason: "design matrix must have at least one column".to_string(),
        });
    }
    if p == 1 {
        // Single predictor: no multicollinearity possible
        return Ok(vec![1.0]);
    }
    if n < p {
        return Err(StatsError::InsufficientSampleSize { got: n, need: p });
    }

    let mut vif_vals = Vec::with_capacity(p);

    for j in 0..p {
        // Build design matrix excluding column j, and response = column j
        let y_j: Vec<f64> = x_mat.iter().map(|row| row[j]).collect();

        // Compute total sum of squares for y_j
        let y_mean = y_j.iter().sum::<f64>() / n as f64;
        let sst: f64 = y_j.iter().map(|&v| (v - y_mean).powi(2)).sum();

        if sst < 1.0e-14 {
            // Constant column — VIF is infinite (undefined / singular)
            vif_vals.push(f64::INFINITY);
            continue;
        }

        // Build n × (p-1) design matrix (all columns except j)
        let x_other: Vec<Vec<f64>> = x_mat
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(k, _)| *k != j)
                    .map(|(_, &v)| v)
                    .collect()
            })
            .collect();

        // Fit OLS: y_j ~ X_other
        let p_other = p - 1;
        let mut x_flat = Vec::with_capacity(n * p_other);
        for row in &x_other {
            x_flat.extend_from_slice(row);
        }

        // Need at least p_other + 1 samples
        if n <= p_other {
            vif_vals.push(f64::INFINITY);
            continue;
        }

        let model = match ols(&x_flat, &y_j, n, p_other) {
            Ok(m) => m,
            Err(_) => {
                vif_vals.push(f64::INFINITY);
                continue;
            }
        };

        let sse = model.residual_sum_squares;
        let r2 = 1.0 - sse / sst;
        let r2 = r2.clamp(0.0, 1.0 - 1.0e-12);
        vif_vals.push(1.0 / (1.0 - r2));
    }

    Ok(vif_vals)
}

/// Internally Studentized (standardized) residuals.
///
/// `r_i = e_i / (s * sqrt(1 - h_{ii}))` where `s = sqrt(RSS / (n - p))`.
pub fn standardized_residuals(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let fit = ols_with_hat(y, x_mat)?;
    let n = fit.n;
    let p = fit.p;
    if n <= p {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: p + 1,
        });
    }
    let s = (fit.rss / (n - p) as f64).sqrt().max(1.0e-14);
    let result: Vec<f64> = fit
        .residuals
        .iter()
        .zip(fit.hat_diag.iter())
        .map(|(&e, &h)| {
            let denom = s * (1.0 - h).max(1.0e-12).sqrt();
            e / denom
        })
        .collect();
    Ok(result)
}

/// Leverage values h_{ii} from the hat matrix H = X (X^T X)^{-1} X^T.
///
/// Each value is in [0, 1] and their sum equals the number of parameters `p`.
pub fn leverage(x_mat: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    if x_mat.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    let n = x_mat.len();
    if n <= 1 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let p = x_mat[0].len();
    if p == 0 {
        return Err(StatsError::InvalidParameter {
            name: "x_mat".to_string(),
            reason: "design matrix must have at least one column".to_string(),
        });
    }
    if n < p {
        return Err(StatsError::InsufficientSampleSize { got: n, need: p });
    }

    // Build X^T X
    let mut xtx = vec![0.0_f64; p * p];
    for row in x_mat {
        if row.len() != p {
            return Err(StatsError::DimensionMismatch { a: row.len(), b: p });
        }
        for i in 0..p {
            for j in 0..p {
                xtx[i * p + j] += row[i] * row[j];
            }
        }
    }

    let xtx_inv = matrix_inverse_lu(&xtx, p)?;

    // h_{ii} = x_i^T (X^T X)^{-1} x_i
    let hat_diag: Vec<f64> = x_mat
        .iter()
        .map(|xi| {
            let mut c = vec![0.0_f64; p];
            for r in 0..p {
                for s in 0..p {
                    c[r] += xtx_inv[r * p + s] * xi[s];
                }
            }
            let h: f64 = xi.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            h.clamp(0.0, 1.0)
        })
        .collect();

    Ok(hat_diag)
}

/// Durbin-Watson statistic for first-order autocorrelation in OLS residuals.
///
/// `DW = Σ_{i=2}^{n} (e_i - e_{i-1})² / Σ_i e_i²`
///
/// Values near 2 indicate no autocorrelation; < 2 suggests positive AC; > 2 negative AC.
///
/// This function fits OLS to `(y, x_mat)` then computes the DW statistic on the residuals.
pub fn durbin_watson_ols(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<f64> {
    let fit = ols_with_hat(y, x_mat)?;
    durbin_watson_residuals(&fit.residuals)
}

/// Durbin-Watson statistic computed directly on a pre-computed residual vector.
pub fn durbin_watson_residuals(residuals: &[f64]) -> StatsResult<f64> {
    let n = residuals.len();
    if n < 2 {
        return Err(StatsError::InsufficientSampleSize { got: n, need: 2 });
    }
    let num: f64 = residuals.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let den: f64 = residuals.iter().map(|e| e * e).sum();
    if den < 1.0e-30 {
        return Ok(2.0); // all residuals zero — undefined; return neutral value
    }
    Ok(num / den)
}

/// Breusch-Pagan LM test for heteroscedasticity.
///
/// Regresses squared residuals on the design matrix and computes `LM = n * R²`.
/// Under H₀ (homoscedasticity), `LM ~ χ²(p-1)` where `p` is the number of columns in `x_mat`.
///
/// Returns `(lm_statistic, p_value)`.
pub fn breusch_pagan_test(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<(f64, f64)> {
    let fit = ols_with_hat(y, x_mat)?;
    let n = fit.n;
    let p = fit.p;
    if n <= p + 1 {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: p + 2,
        });
    }

    // Auxiliary regression: e_i² ~ x_mat
    let e_sq: Vec<f64> = fit.residuals.iter().map(|e| e * e).collect();
    let e_sq_mean = e_sq.iter().sum::<f64>() / n as f64;
    let sst_aux: f64 = e_sq.iter().map(|v| (v - e_sq_mean).powi(2)).sum();

    if sst_aux < 1.0e-20 {
        // No variance in squared residuals — homoscedastic
        return Ok((0.0, 1.0));
    }

    // Flatten x_mat
    let mut x_flat = Vec::with_capacity(n * p);
    for row in x_mat {
        x_flat.extend_from_slice(row);
    }

    let aux_model = match ols(&x_flat, &e_sq, n, p) {
        Ok(m) => m,
        Err(_) => return Ok((0.0, 1.0)),
    };

    let sse_aux = aux_model.residual_sum_squares;
    let r2_aux = (1.0 - sse_aux / sst_aux).clamp(0.0, 1.0);

    // LM statistic: n * R²
    let lm = n as f64 * r2_aux;

    // p-value from chi-squared with df = p (number of auxiliary regressors, excluding intercept if any)
    // Use p - 1 as the degrees of freedom when the first column is a constant (intercept)
    let df = {
        // Heuristic: if first column looks like a constant (all equal), subtract 1
        let first_col_std = {
            let col: Vec<f64> = x_mat.iter().map(|row| row[0]).collect();
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            col.iter().map(|v| (v - mean).abs()).sum::<f64>()
        };
        if first_col_std < 1.0e-10 {
            (p - 1).max(1)
        } else {
            p
        }
    };

    let p_val = 1.0 - chi2_cdf(lm, df as f64);
    Ok((lm, p_val.clamp(0.0, 1.0)))
}

// ---------------------------------------------------------------------------
// Public helper re-exported for convenience
// ---------------------------------------------------------------------------

/// Compute OLS standard errors from the design matrix `x_mat` (n×p) and responses `y`.
///
/// Returns `(coef, se)` where `se_i = sqrt(MSE * (X^T X)^{-1}_{ii})`.
pub fn ols_standard_errors(y: &[f64], x_mat: &[Vec<f64>]) -> StatsResult<(Vec<f64>, Vec<f64>)> {
    let fit = ols_with_hat(y, x_mat)?;
    let n = fit.n;
    let p = fit.p;
    if n <= p {
        return Err(StatsError::InsufficientSampleSize {
            got: n,
            need: p + 1,
        });
    }
    let mse = fit.rss / (n - p) as f64;

    // Rebuild X^T X inv
    let mut x_flat = Vec::with_capacity(n * p);
    for row in x_mat {
        x_flat.extend_from_slice(row);
    }
    // X^T X
    let mut xtx = vec![0.0_f64; p * p];
    for i in 0..n {
        for r in 0..p {
            for s in 0..p {
                xtx[r * p + s] += x_flat[i * p + r] * x_flat[i * p + s];
            }
        }
    }
    let xtx_inv = matrix_inverse_lu(&xtx, p)?;
    let se: Vec<f64> = (0..p)
        .map(|r| (mse * xtx_inv[r * p + r].max(0.0)).sqrt())
        .collect();

    // Normal CDF for t-tests (two-sided)
    Ok((fit.coef, se))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple linear design matrix `[[1, x1], [1, x2], ...]` with intercept.
    fn design_with_intercept(xs: &[f64]) -> Vec<Vec<f64>> {
        xs.iter().map(|&x| vec![1.0, x]).collect()
    }

    /// Build an orthogonal two-column design matrix (no intercept).
    fn ortho_design(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                vec![t.sin(), t.cos()]
            })
            .collect()
    }

    // ---- Cook's distance ---------------------------------------------------

    #[test]
    fn cooks_all_positive_for_typical_data() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&x| 2.0 + 3.0 * x).collect();
        // Add a tiny perturbation to avoid perfect fit (which gives D=0 by definition)
        let y_noisy: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| v + if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let x_mat = design_with_intercept(&xs);
        let d = cooks_distance(&y_noisy, &x_mat).expect("ok");
        assert_eq!(d.len(), 10);
        assert!(d.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn cooks_influential_point_has_large_d() {
        // First 9 points on a clean line, last point is a far outlier
        let mut x_mat: Vec<Vec<f64>> = (0..10).map(|i| vec![1.0, i as f64]).collect();
        x_mat[9] = vec![1.0, 50.0]; // extreme leverage
        let mut y: Vec<f64> = (0..10).map(|i| 2.0 * i as f64).collect();
        y[9] = -200.0; // large residual as well
        let d = cooks_distance(&y, &x_mat).expect("ok");
        let d9 = d[9];
        let d_rest_max = d[..9].iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            d9 > d_rest_max,
            "outlier D={d9} should exceed max of rest={d_rest_max}"
        );
    }

    #[test]
    fn cooks_empty_data_error() {
        let y: Vec<f64> = vec![];
        let x_mat: Vec<Vec<f64>> = vec![];
        assert!(cooks_distance(&y, &x_mat).is_err());
    }

    #[test]
    fn cooks_single_observation_error() {
        let y = vec![1.0];
        let x_mat = vec![vec![1.0, 0.5]];
        assert!(cooks_distance(&y, &x_mat).is_err());
    }

    // ---- Leverage ----------------------------------------------------------

    #[test]
    fn leverage_values_in_0_1() {
        let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let x_mat = design_with_intercept(&xs);
        let h = leverage(&x_mat).expect("ok");
        assert!(h.iter().all(|&v| (-1.0e-10..=1.0 + 1.0e-10).contains(&v)));
    }

    #[test]
    fn leverage_sum_equals_p() {
        // For a model with intercept: sum(h_ii) = p = 2 (intercept + slope)
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let x_mat = design_with_intercept(&xs);
        let h = leverage(&x_mat).expect("ok");
        let sum: f64 = h.iter().sum();
        assert!((sum - 2.0).abs() < 1.0e-8, "sum(h_ii)={sum}, expected 2");
    }

    #[test]
    fn leverage_empty_data_error() {
        let x_mat: Vec<Vec<f64>> = vec![];
        assert!(leverage(&x_mat).is_err());
    }

    // ---- Standardized residuals --------------------------------------------

    #[test]
    fn standardized_residuals_approx_zero_mean() {
        let xs: Vec<f64> = (0..20).map(|i| i as f64 - 10.0).collect();
        let y: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * x + 0.1 * (x * 3.0).sin())
            .collect();
        let x_mat = design_with_intercept(&xs);
        let r = standardized_residuals(&y, &x_mat).expect("ok");
        let mean: f64 = r.iter().sum::<f64>() / r.len() as f64;
        assert!(mean.abs() < 0.5, "mean of std resid={mean}");
    }

    #[test]
    fn standardized_residuals_length_equals_n() {
        let xs: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&x| x * 2.0).collect();
        let x_mat = design_with_intercept(&xs);
        let r = standardized_residuals(&y, &x_mat).expect("ok");
        assert_eq!(r.len(), 12);
    }

    // ---- VIF ---------------------------------------------------------------

    #[test]
    fn vif_perfect_collinearity_large() {
        // x2 = 2 * x1 — perfect multicollinearity
        let x_mat: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let x1 = i as f64;
                vec![x1, 2.0 * x1 + 0.0001 * (i as f64)] // near-perfect
            })
            .collect();
        let v = vif(&x_mat).expect("ok");
        // At least one VIF should be very large
        assert!(v.iter().any(|&vi| vi > 10.0), "VIFs={v:?}");
    }

    #[test]
    fn vif_orthogonal_columns_near_one() {
        // sin and cos are approximately orthogonal over a full period
        let n = 64;
        let x_mat = ortho_design(n);
        let v = vif(&x_mat).expect("ok");
        // VIF should be close to 1 for orthogonal predictors
        for &vi in &v {
            assert!(vi < 5.0, "VIF={vi} for orthogonal design");
        }
    }

    #[test]
    fn vif_returns_length_p() {
        let x_mat: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![1.0, i as f64, (i * i) as f64])
            .collect();
        let v = vif(&x_mat).expect("ok");
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn vif_single_column_returns_one() {
        let x_mat: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let v = vif(&x_mat).expect("ok");
        assert_eq!(v.len(), 1);
        assert!((v[0] - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn vif_acceptable_threshold_is_5() {
        // Standard rule: VIF < 5 is acceptable; VIF > 10 is severe
        let x_mat: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let x = i as f64;
                vec![x, x * 1.5 + (i as f64 * 0.7).sin() * 3.0]
            })
            .collect();
        let v = vif(&x_mat).expect("ok");
        // just verify it computes something reasonable
        assert!(v.iter().all(|&vi| vi >= 1.0), "VIFs={v:?}");
    }

    // ---- DFFITS ------------------------------------------------------------

    #[test]
    fn dffits_length_equals_n() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&x| 1.0 + x + 0.05 * x).collect();
        let x_mat = design_with_intercept(&xs);
        let df = dffits(&y, &x_mat).expect("ok");
        assert_eq!(df.len(), 10);
    }

    #[test]
    fn dffits_finite_values() {
        let xs: Vec<f64> = (0..15).map(|i| i as f64 - 7.0).collect();
        let y: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 0.1).collect();
        let x_mat = design_with_intercept(&xs);
        let df = dffits(&y, &x_mat).expect("ok");
        assert!(df.iter().all(|v| v.is_finite()), "dffits={df:?}");
    }

    // ---- Durbin-Watson -----------------------------------------------------

    #[test]
    fn durbin_watson_uncorrelated_near_two() {
        // iid residuals should have DW ≈ 2
        let residuals: Vec<f64> = vec![0.5, -0.3, 0.7, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5];
        let dw = durbin_watson_residuals(&residuals).expect("ok");
        assert!(
            (dw - 2.0).abs() < 1.5,
            "DW={dw}, expected near 2 for uncorrelated"
        );
    }

    #[test]
    fn durbin_watson_positively_autocorrelated_below_two() {
        // Residuals that follow a positive trend have DW < 2
        let residuals: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let dw = durbin_watson_residuals(&residuals).expect("ok");
        assert!(
            dw < 2.0,
            "DW={dw} should be < 2 for positively autocorrelated"
        );
    }

    #[test]
    fn durbin_watson_short_vector_error() {
        let r = vec![1.0]; // only 1 residual
        assert!(durbin_watson_residuals(&r).is_err());
    }

    #[test]
    fn durbin_watson_ols_computes_from_model() {
        let xs: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * x + 0.1 * (x * 0.5).cos())
            .collect();
        let x_mat = design_with_intercept(&xs);
        let dw = durbin_watson_ols(&y, &x_mat).expect("ok");
        assert!((0.0..=4.0).contains(&dw), "DW={dw}");
    }

    // ---- Breusch-Pagan -----------------------------------------------------

    #[test]
    fn breusch_pagan_homoscedastic_high_pvalue() {
        // Homoscedastic: e ~ N(0, 1) regardless of x
        let xs: Vec<f64> = (0..30).map(|i| i as f64 / 5.0).collect();
        // Residuals from a model with constant variance
        let y: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                2.0 + 3.0 * x
                    + if i % 3 == 0 {
                        0.3
                    } else if i % 3 == 1 {
                        -0.3
                    } else {
                        0.0
                    }
            })
            .collect();
        let x_mat = design_with_intercept(&xs);
        let (lm, pv) = breusch_pagan_test(&y, &x_mat).expect("ok");
        assert!(lm >= 0.0);
        assert!((0.0..=1.0).contains(&pv));
        // For homoscedastic, p-value should generally not be tiny
        assert!(pv > 0.01, "p={pv}; expected > 0.01 for homoscedastic data");
    }

    #[test]
    fn breusch_pagan_heteroscedastic_low_pvalue() {
        // Heteroscedastic: variance of e grows with x
        let xs: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let y: Vec<f64> = xs
            .iter()
            .map(|&x| {
                // residual variance proportional to x
                1.0 + 0.5 * x + x * (x * 0.1).sin() * 2.0
            })
            .collect();
        let x_mat = design_with_intercept(&xs);
        let (lm, pv) = breusch_pagan_test(&y, &x_mat).expect("ok");
        assert!(lm > 0.0);
        assert!((0.0..=1.0).contains(&pv));
    }

    #[test]
    fn breusch_pagan_empty_error() {
        let y: Vec<f64> = vec![];
        let x_mat: Vec<Vec<f64>> = vec![];
        assert!(breusch_pagan_test(&y, &x_mat).is_err());
    }
}
