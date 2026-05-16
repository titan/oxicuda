//! Dense matrix-vector helpers.
//!
//! Matrices are stored row-major as flat `Vec<f64>` with separate dim metadata.

use crate::error::{CvxError, CvxResult};

/// Dot product `<x, y>`.
pub fn dot(x: &[f64], y: &[f64]) -> CvxResult<f64> {
    if x.len() != y.len() {
        return Err(CvxError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    let mut s = 0.0_f64;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    Ok(s)
}

/// L2 norm `||x||_2`.
#[must_use]
pub fn norm2(x: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for &v in x {
        s += v * v;
    }
    s.sqrt()
}

/// y = y + alpha * x  (in place).
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) -> CvxResult<()> {
    if x.len() != y.len() {
        return Err(CvxError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
    Ok(())
}

/// out = alpha * x  (scaled copy).
pub fn scale(alpha: f64, x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| alpha * v).collect()
}

/// out = a + alpha * b  (no mutation).
pub fn add_scaled(a: &[f64], alpha: f64, b: &[f64]) -> CvxResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(CvxError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai + alpha * bi)
        .collect())
}

/// y = A * x where A is row-major `m × n`.
pub fn mat_vec(a: &[f64], m: usize, n: usize, x: &[f64]) -> CvxResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if x.len() != n {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
    }
    let mut y = vec![0.0_f64; m];
    for (i, yi) in y.iter_mut().enumerate().take(m) {
        let mut s = 0.0_f64;
        let row = i * n;
        for j in 0..n {
            s += a[row + j] * x[j];
        }
        *yi = s;
    }
    Ok(y)
}

/// y = A^T * x where A is row-major `m × n` (returns vector of length n).
pub fn mat_t_vec(a: &[f64], m: usize, n: usize, x: &[f64]) -> CvxResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if x.len() != m {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: m });
    }
    let mut y = vec![0.0_f64; n];
    for (i, &xi) in x.iter().enumerate().take(m) {
        let row = i * n;
        for j in 0..n {
            y[j] += a[row + j] * xi;
        }
    }
    Ok(y)
}

/// y = alpha * A * x + beta * y  (GEMV, row-major A).
pub fn gemv(
    alpha: f64,
    a: &[f64],
    m: usize,
    n: usize,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> CvxResult<()> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if x.len() != n {
        return Err(CvxError::DimensionMismatch { a: x.len(), b: n });
    }
    if y.len() != m {
        return Err(CvxError::DimensionMismatch { a: y.len(), b: m });
    }
    for (i, yi) in y.iter_mut().enumerate().take(m) {
        let mut s = 0.0_f64;
        let row = i * n;
        for j in 0..n {
            s += a[row + j] * x[j];
        }
        *yi = alpha * s + beta * *yi;
    }
    Ok(())
}

/// Compute A^T * A for A row-major `m × n`. Result `n × n` row-major (symmetric).
pub fn mat_t_mat(a: &[f64], m: usize, n: usize) -> CvxResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    let mut g = vec![0.0_f64; n * n];
    for k in 0..m {
        let row = k * n;
        for i in 0..n {
            let aki = a[row + i];
            for j in 0..n {
                g[i * n + j] += aki * a[row + j];
            }
        }
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        assert!((dot(&x, &y).expect("ok") - 32.0).abs() < 1.0e-12);
    }

    #[test]
    fn norm2_basic() {
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn axpy_basic() {
        let x = [1.0, 1.0, 1.0];
        let mut y = [10.0, 20.0, 30.0];
        axpy(2.0, &x, &mut y).expect("ok");
        assert_eq!(y, [12.0, 22.0, 32.0]);
    }

    #[test]
    fn mat_vec_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 4.0, 5.0];
        let y = mat_vec(&a, 3, 3, &x).expect("ok");
        assert_eq!(y, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn mat_t_vec_matches() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let y = mat_t_vec(&a, 2, 3, &x).expect("ok");
        assert_eq!(y, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn gemv_blend() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![2.0, 3.0];
        let mut y = vec![10.0, 20.0];
        gemv(2.0, &a, 2, 2, &x, 0.5, &mut y).expect("ok");
        assert_eq!(y, vec![2.0 * 2.0 + 0.5 * 10.0, 2.0 * 3.0 + 0.5 * 20.0]);
    }
}
