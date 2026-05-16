//! Cholesky factorisation `A = L L^T` for SPD `A`.

use crate::error::{CvxError, CvxResult};

/// Cholesky factor: returns lower triangular `L` (row-major, zeros above diag) with `A = L L^T`.
pub fn cholesky_factor(a: &[f64], n: usize) -> CvxResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let mut l = vec![0.0_f64; n * n];
    for j in 0..n {
        // Diagonal element.
        let mut sum_d = a[j * n + j];
        for k in 0..j {
            let v = l[j * n + k];
            sum_d -= v * v;
        }
        if sum_d <= 0.0 {
            return Err(CvxError::NumericalInstability(format!(
                "cholesky non-positive pivot at column {j}: {sum_d}"
            )));
        }
        let ljj = sum_d.sqrt();
        l[j * n + j] = ljj;
        // Below diagonal.
        let inv_ljj = 1.0 / ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s * inv_ljj;
        }
    }
    Ok(l)
}

/// Solve `A x = b` where `L` is the Cholesky factor of A.
/// Two triangular solves: `L y = b`, then `L^T x = y`.
pub fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> CvxResult<Vec<f64>> {
    if l.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![l.len()],
        });
    }
    if b.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    // L y = b (forward subst).
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let d = l[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CvxError::SingularMatrix(format!("zero L[{i},{i}]")));
        }
        y[i] = s / d;
    }
    // L^T x = y (back subst).
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let d = l[i * n + i];
        x[i] = s / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let l = cholesky_factor(&a, 3).expect("ok");
        for (i, &li) in l.iter().enumerate() {
            let r = i / 3;
            let c = i % 3;
            let target = if r == c { 1.0 } else { 0.0 };
            assert!((li - target).abs() < 1.0e-12);
        }
    }

    #[test]
    fn cholesky_solve_spd() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        let a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let l = cholesky_factor(&a, 3).expect("ok");
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve(&l, 3, &b).expect("ok");
        // Verify A x ≈ b.
        let mut ax = vec![0.0_f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                ax[i] += a[i * 3 + j] * x[j];
            }
        }
        for (axi, bi) in ax.iter().zip(b.iter()) {
            assert!((axi - bi).abs() < 1.0e-9);
        }
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_factor(&a, 2).is_err());
    }
}
