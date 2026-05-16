//! Ordinary least-squares (OLS) linear regression via normal equations + LU.

use crate::error::{StatsError, StatsResult};

/// A fitted linear model with coefficients and book-keeping for inference.
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub coefficients: Vec<f64>,
    pub residuals: Vec<f64>,
    pub fitted: Vec<f64>,
    pub residual_sum_squares: f64,
    pub xtx: Vec<f64>,
    pub xtx_inv: Vec<f64>,
}

/// Fit `y ~ X` by OLS. `x` is row-major shape `(n_samples, n_features)`.
pub fn ols(x: &[f64], y: &[f64], n_samples: usize, n_features: usize) -> StatsResult<LinearModel> {
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
    if n_samples < n_features {
        return Err(StatsError::InsufficientSampleSize {
            got: n_samples,
            need: n_features,
        });
    }
    // X^T X (n_features x n_features)
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
    // X^T y (n_features)
    let mut xty = vec![0.0; n_features];
    for i in 0..n_features {
        let mut acc = 0.0;
        for k in 0..n_samples {
            acc += x[k * n_features + i] * y[k];
        }
        xty[i] = acc;
    }
    // Solve (X^T X) beta = X^T y via LU
    let xtx_inv = matrix_inverse_lu(&xtx, n_features)?;
    let mut beta = vec![0.0; n_features];
    for i in 0..n_features {
        let mut acc = 0.0;
        for j in 0..n_features {
            acc += xtx_inv[i * n_features + j] * xty[j];
        }
        beta[i] = acc;
    }
    // fitted, residuals
    let mut fitted = vec![0.0; n_samples];
    for k in 0..n_samples {
        let mut acc = 0.0;
        for i in 0..n_features {
            acc += x[k * n_features + i] * beta[i];
        }
        fitted[k] = acc;
    }
    let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(a, b)| a - b).collect();
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    Ok(LinearModel {
        coefficients: beta,
        residuals,
        fitted,
        residual_sum_squares: rss,
        xtx,
        xtx_inv,
    })
}

/// Compute the inverse of a square matrix via Gauss-Jordan with partial pivoting.
pub fn matrix_inverse_lu(mat: &[f64], n: usize) -> StatsResult<Vec<f64>> {
    if mat.len() != n * n {
        return Err(StatsError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![mat.len()],
        });
    }
    let mut a = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = mat[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }
    for k in 0..n {
        // Pivot
        let mut max_val = a[k * 2 * n + k].abs();
        let mut piv = k;
        for i in (k + 1)..n {
            let v = a[i * 2 * n + k].abs();
            if v > max_val {
                max_val = v;
                piv = i;
            }
        }
        if max_val < 1e-12 {
            return Err(StatsError::SingularMatrix(format!(
                "matrix_inverse_lu: pivot {max_val} too small at column {k}"
            )));
        }
        if piv != k {
            for j in 0..2 * n {
                a.swap(k * 2 * n + j, piv * 2 * n + j);
            }
        }
        let pivot = a[k * 2 * n + k];
        for j in 0..2 * n {
            a[k * 2 * n + j] /= pivot;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = a[i * 2 * n + k];
            for j in 0..2 * n {
                a[i * 2 * n + j] -= factor * a[k * 2 * n + j];
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}

/// Matrix multiplication: `C[m x n] = A[m x k] * B[k x n]`.
pub fn matrix_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> StatsResult<Vec<f64>> {
    if a.len() != m * k || b.len() != k * n {
        return Err(StatsError::ShapeMismatch {
            expected: vec![m, k, n],
            got: vec![a.len(), b.len()],
        });
    }
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    Ok(c)
}

/// Transpose an `(m x n)` matrix to `(n x m)`.
#[must_use]
pub fn matrix_transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_fits_perfect_line() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x).collect();
        let mut design = Vec::with_capacity(10);
        for &x in &xs {
            design.push(1.0);
            design.push(x);
        }
        let m = ols(&design, &ys, 5, 2).expect("ok");
        assert!((m.coefficients[0] - 1.0).abs() < 1e-9);
        assert!((m.coefficients[1] - 2.0).abs() < 1e-9);
        assert!(m.residual_sum_squares < 1e-18);
    }

    #[test]
    fn matrix_inverse_identity() {
        let id = vec![1.0, 0.0, 0.0, 1.0];
        let inv = matrix_inverse_lu(&id, 2).expect("ok");
        for (a, b) in inv.iter().zip(&id) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn matrix_mul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let c = matrix_mul(&a, &b, 2, 2, 2).expect("ok");
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matrix_transpose_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let t = matrix_transpose(&a, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn singular_matrix_errors() {
        let s = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
        assert!(matrix_inverse_lu(&s, 2).is_err());
    }
}
