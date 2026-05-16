//! Private linear-algebra helpers for compressed sensing.
//!
//! Pure-Rust implementations of:
//! - Jacobi SVD (one-sided, for thin matrices)
//! - Householder QR (least-squares)
//! - Cholesky factorisation and triangular solves
//! - LSQR iterative least-squares
//! - Normal-equations solve via Cholesky

pub mod cholesky;
pub mod jacobi_svd;
pub mod lsqr;
pub mod normal_equations;
pub mod qr_householder;

pub use cholesky::{cholesky_factor, cholesky_solve};
pub use jacobi_svd::{jacobi_svd, jacobi_svd_thin};
pub use lsqr::lsqr;
pub use normal_equations::{normal_equations_solve, solve_subset_ls};
pub use qr_householder::{householder_qr, qr_solve};

use crate::error::{CsError, CsResult};

/// Dot product `<x, y>`.
pub fn dot(x: &[f64], y: &[f64]) -> CsResult<f64> {
    if x.len() != y.len() {
        return Err(CsError::DimensionMismatch {
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

/// L-infinity norm `max_i |x_i|`.
#[must_use]
pub fn norm_inf(x: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for &v in x {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// L1 norm `sum_i |x_i|`.
#[must_use]
pub fn norm1(x: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for &v in x {
        s += v.abs();
    }
    s
}

/// y = y + alpha * x  (in place).
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) -> CsResult<()> {
    if x.len() != y.len() {
        return Err(CsError::DimensionMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
    Ok(())
}

/// y = A * x where A is row-major `m × n`.
pub fn mat_vec(a: &[f64], m: usize, n: usize, x: &[f64]) -> CsResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if x.len() != n {
        return Err(CsError::DimensionMismatch { a: x.len(), b: n });
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
pub fn mat_t_vec(a: &[f64], m: usize, n: usize, x: &[f64]) -> CsResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if x.len() != m {
        return Err(CsError::DimensionMismatch { a: x.len(), b: m });
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

/// out = a + alpha * b  (no mutation).
pub fn add_scaled(a: &[f64], alpha: f64, b: &[f64]) -> CsResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(CsError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai + alpha * bi)
        .collect())
}

/// Extract the submatrix `A[:, support]` of dimensions `m × |support|` row-major.
pub fn submat_columns(a: &[f64], m: usize, n: usize, support: &[usize]) -> CsResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    let k = support.len();
    let mut out = vec![0.0_f64; m * k];
    for i in 0..m {
        for (s_idx, &j) in support.iter().enumerate() {
            if j >= n {
                return Err(CsError::IndexOutOfBounds { index: j, len: n });
            }
            out[i * k + s_idx] = a[i * n + j];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).expect("ok") - 32.0).abs() < 1.0e-12);
    }

    #[test]
    fn norm2_basic() {
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn norm_inf_basic() {
        assert!((norm_inf(&[1.0, -3.0, 2.0]) - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn norm1_basic() {
        assert!((norm1(&[1.0, -3.0, 2.0]) - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn submat_columns_works() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let s = submat_columns(&a, 2, 3, &[0, 2]).expect("ok");
        assert_eq!(s, vec![1.0, 3.0, 4.0, 6.0]);
    }
}
