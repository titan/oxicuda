//! Logistic regression via iteratively reweighted least squares (IRLS / Newton-Raphson).

use crate::error::{StatsError, StatsResult};
use crate::regression::linear::{matrix_inverse_lu, matrix_mul};

/// Fitted logistic regression model.
#[derive(Debug, Clone)]
pub struct LogisticModel {
    pub coefficients: Vec<f64>,
    pub iterations: usize,
    pub log_likelihood: f64,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Fit binary logistic regression via Newton-Raphson IRLS.
///
/// `x` is row-major `(n_samples, n_features)`. `y` is in {0, 1}.
pub fn logistic_fit_irls(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    max_iter: usize,
    tol: f64,
) -> StatsResult<LogisticModel> {
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
    for (i, &v) in y.iter().enumerate() {
        if !(v == 0.0 || v == 1.0) {
            return Err(StatsError::InvalidParameter {
                name: format!("y[{i}]"),
                reason: format!("expected 0 or 1; got {v}"),
            });
        }
    }
    let mut beta = vec![0.0; n_features];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        // Compute predictions and weights
        let mut p = vec![0.0; n_samples];
        for k in 0..n_samples {
            let mut z = 0.0;
            for j in 0..n_features {
                z += x[k * n_features + j] * beta[j];
            }
            p[k] = sigmoid(z);
        }
        // X^T W X
        let mut xtwx = vec![0.0; n_features * n_features];
        let mut xt_y_minus_p = vec![0.0; n_features];
        for i in 0..n_features {
            for j in i..n_features {
                let mut acc = 0.0;
                for k in 0..n_samples {
                    let w = p[k] * (1.0 - p[k]);
                    acc += x[k * n_features + i] * w * x[k * n_features + j];
                }
                xtwx[i * n_features + j] = acc;
                xtwx[j * n_features + i] = acc;
            }
            let mut g = 0.0;
            for k in 0..n_samples {
                g += x[k * n_features + i] * (y[k] - p[k]);
            }
            xt_y_minus_p[i] = g;
        }
        // Regularize a tiny bit to avoid singular matrix when separability occurs
        for i in 0..n_features {
            xtwx[i * n_features + i] += 1e-10;
        }
        let inv = matrix_inverse_lu(&xtwx, n_features)?;
        // beta_new = beta + inv * X^T (y - p)
        let mut delta = vec![0.0; n_features];
        for i in 0..n_features {
            let mut acc = 0.0;
            for j in 0..n_features {
                acc += inv[i * n_features + j] * xt_y_minus_p[j];
            }
            delta[i] = acc;
        }
        for i in 0..n_features {
            beta[i] += delta[i];
        }
        // log-likelihood
        let mut ll = 0.0;
        for k in 0..n_samples {
            let mut z = 0.0;
            for j in 0..n_features {
                z += x[k * n_features + j] * beta[j];
            }
            let pk = sigmoid(z).clamp(1e-15, 1.0 - 1e-15);
            ll += y[k] * pk.ln() + (1.0 - y[k]) * (1.0 - pk).ln();
        }
        if (ll - prev_ll).abs() < tol {
            return Ok(LogisticModel {
                coefficients: beta,
                iterations: iters,
                log_likelihood: ll,
            });
        }
        prev_ll = ll;
    }
    // Compute final ll
    let mut ll_final = 0.0;
    for k in 0..n_samples {
        let mut z = 0.0;
        for j in 0..n_features {
            z += x[k * n_features + j] * beta[j];
        }
        let pk = sigmoid(z).clamp(1e-15, 1.0 - 1e-15);
        ll_final += y[k] * pk.ln() + (1.0 - y[k]) * (1.0 - pk).ln();
    }
    let _ = matrix_mul(&[1.0], &[1.0], 1, 1, 1)?; // touch helper
    Ok(LogisticModel {
        coefficients: beta,
        iterations: iters,
        log_likelihood: ll_final,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_perfect_separation_runs() {
        // Two classes well-separated
        let x = [
            1.0, -3.0, 1.0, -2.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0,
        ];
        let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let m = logistic_fit_irls(&x, &y, 6, 2, 50, 1e-10).expect("ok");
        // Slope should be positive
        assert!(m.coefficients[1] > 0.0);
        // log-likelihood should improve (be > -inf)
        assert!(m.log_likelihood.is_finite());
    }

    #[test]
    fn logistic_rejects_non_binary_y() {
        let x = [1.0, 0.5, 1.0, 1.5];
        let y = [0.0, 0.5];
        assert!(logistic_fit_irls(&x, &y, 2, 2, 10, 1e-6).is_err());
    }
}
