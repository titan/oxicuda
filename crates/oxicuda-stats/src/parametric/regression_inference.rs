//! Inference for ordinary least-squares regression coefficients.

use crate::distributions::f_dist::FDist;
use crate::distributions::student_t::StudentT;
use crate::error::{StatsError, StatsResult};
use crate::regression::linear::{LinearModel, ols};

/// Full inference output for a fitted OLS model.
#[derive(Debug, Clone)]
pub struct RegressionInference {
    pub model: LinearModel,
    pub se: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values_two_sided: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub f_statistic: f64,
    pub p_value_f: f64,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
}

/// Fit `y ~ X` via OLS and compute t-statistics, F-statistic, AIC, BIC.
pub fn regression_inference(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
) -> StatsResult<RegressionInference> {
    let model = ols(x, y, n_samples, n_features)?;
    if n_samples <= n_features {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: n_features + 1,
        });
    }
    let df_resid = (n_samples - n_features) as f64;
    let sigma2 = model.residual_sum_squares / df_resid;
    // (X^T X)^{-1} diagonal already cached in model.xtx_inv
    let se: Vec<f64> = (0..n_features)
        .map(|i| (sigma2 * model.xtx_inv[i * n_features + i]).sqrt())
        .collect();
    let t_stats: Vec<f64> = model
        .coefficients
        .iter()
        .zip(&se)
        .map(|(b, s)| if *s > 0.0 { b / s } else { 0.0 })
        .collect();
    let dist = StudentT::new(df_resid)?;
    let mut p_vals = Vec::with_capacity(n_features);
    for &t in &t_stats {
        if !t.is_finite() {
            // SE = 0 implies infinite t-stat: model fits perfectly => p effectively 0
            // (but we cap to keep finite output).
            p_vals.push(0.0);
            continue;
        }
        let cdf_t = dist.cdf(t)?;
        let two = 2.0 * cdf_t.min(1.0 - cdf_t);
        p_vals.push(two.clamp(0.0, 1.0));
    }
    // R^2
    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let tss: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let r2 = if tss > 0.0 {
        1.0 - model.residual_sum_squares / tss
    } else {
        0.0
    };
    let adj_r2 = 1.0 - (1.0 - r2) * (n_samples - 1) as f64 / df_resid;
    // F-stat for global fit (compare full model vs intercept-only).
    // Assumes the first column of X is the intercept column.
    let k = (n_features - 1) as f64;
    let one_minus_r2 = (1.0 - r2).max(0.0);
    let f_stat = if k > 0.0 && one_minus_r2 > 1e-15 {
        (r2 / k) / (one_minus_r2 / df_resid)
    } else if k > 0.0 && r2 > 1.0 - 1e-12 {
        f64::INFINITY
    } else {
        0.0
    };
    let p_f = if k > 0.0 && f_stat.is_finite() && f_stat > 0.0 {
        1.0 - FDist::new(k, df_resid)?.cdf(f_stat)?
    } else if f_stat.is_infinite() {
        0.0
    } else {
        1.0
    };
    // log-likelihood under normal errors (handle sigma2=0 gracefully)
    let n = n_samples as f64;
    let safe_sigma2 = sigma2.max(1e-300);
    let log_lik = -0.5 * n * ((2.0 * std::f64::consts::PI).ln() + safe_sigma2.ln() + 1.0);
    let aic = 2.0 * n_features as f64 - 2.0 * log_lik;
    let bic = (n_features as f64) * n.ln() - 2.0 * log_lik;
    Ok(RegressionInference {
        model,
        se,
        t_stats,
        p_values_two_sided: p_vals,
        r_squared: r2,
        adj_r_squared: adj_r2,
        f_statistic: f_stat,
        p_value_f: p_f,
        aic,
        bic,
        log_likelihood: log_lik,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_perfect_line() {
        // y = 2 + 3x exactly; residuals zero => SE may be zero.
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 + 3.0 * x).collect();
        // X = [1, x] design matrix
        let mut x_mat = Vec::with_capacity(xs.len() * 2);
        for &x in &xs {
            x_mat.push(1.0);
            x_mat.push(x);
        }
        let r = regression_inference(&x_mat, &ys, xs.len(), 2).expect("ok");
        assert!(r.r_squared > 1.0 - 1e-6);
        assert!((r.model.coefficients[0] - 2.0).abs() < 1e-6);
        assert!((r.model.coefficients[1] - 3.0).abs() < 1e-6);
    }
}
