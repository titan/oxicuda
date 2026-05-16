//! Ridge regression (Tikhonov-regularized OLS).

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::matrix_inverse_lu;

/// Fitted ridge regression model.
#[derive(Debug, Clone)]
pub struct RidgeModel {
    pub coefficients: Vec<f64>,
    pub lambda: f64,
    pub residual_sum_squares: f64,
}

/// Fit ridge regression `min ||X beta - y||^2 + lambda * ||beta||^2`.
pub fn ridge_regression(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    lambda: f64,
) -> StatsResult<RidgeModel> {
    if lambda < 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "lambda".into(),
            reason: "must be >= 0".into(),
        });
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
    let mut xtx = vec![0.0; n_features * n_features];
    for i in 0..n_features {
        for j in i..n_features {
            let mut acc = 0.0;
            for k in 0..n_samples {
                acc += x[k * n_features + i] * x[k * n_features + j];
            }
            xtx[i * n_features + j] = acc;
            xtx[j * n_features + i] = acc;
        }
    }
    for i in 0..n_features {
        xtx[i * n_features + i] += lambda;
    }
    let mut xty = vec![0.0; n_features];
    for i in 0..n_features {
        let mut acc = 0.0;
        for k in 0..n_samples {
            acc += x[k * n_features + i] * y[k];
        }
        xty[i] = acc;
    }
    let inv = matrix_inverse_lu(&xtx, n_features)?;
    let mut beta = vec![0.0; n_features];
    for i in 0..n_features {
        let mut acc = 0.0;
        for j in 0..n_features {
            acc += inv[i * n_features + j] * xty[j];
        }
        beta[i] = acc;
    }
    let mut rss = 0.0;
    for k in 0..n_samples {
        let mut yhat = 0.0;
        for i in 0..n_features {
            yhat += x[k * n_features + i] * beta[i];
        }
        rss += (y[k] - yhat).powi(2);
    }
    Ok(RidgeModel {
        coefficients: beta,
        lambda,
        residual_sum_squares: rss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_zero_lambda_matches_ols() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x).collect();
        let mut design = Vec::with_capacity(10);
        for &x in &xs {
            design.push(1.0);
            design.push(x);
        }
        let m = ridge_regression(&design, &ys, 5, 2, 0.0).expect("ok");
        assert!((m.coefficients[0] - 1.0).abs() < 1e-6);
        assert!((m.coefficients[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ridge_positive_lambda_shrinks() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x).collect();
        let mut design = Vec::with_capacity(5);
        for &x in &xs {
            design.push(x);
        }
        let m0 = ridge_regression(&design, &ys, 5, 1, 0.0).expect("ok");
        let m1 = ridge_regression(&design, &ys, 5, 1, 10.0).expect("ok");
        assert!(m1.coefficients[0] < m0.coefficients[0]);
    }
}
