//! Solve linear least-squares via normal equations + Cholesky.

use crate::error::{CsError, CsResult};
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};
use crate::linalg::{mat_t_vec, submat_columns};

/// Solve `(A^T A + λ I) x = A^T b` via Cholesky.
///
/// Caller picks `lambda ≥ 0` for Tikhonov regularisation; 0 for plain normal equations.
pub fn normal_equations_solve(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    lambda: f64,
) -> CsResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m {
        return Err(CsError::DimensionMismatch { a: b.len(), b: m });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // G = A^T A row-major.
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
    if lambda > 0.0 {
        for i in 0..n {
            g[i * n + i] += lambda;
        }
    }
    let atb = mat_t_vec(a, m, n, b)?;
    let l = cholesky_factor(&g, n)?;
    cholesky_solve(&l, n, &atb)
}

/// Solve `A_S x = b` (least squares) where `A_S` is `A[:, support]` of size `m × k`.
/// Returns coefficients vector of length `support.len()`.
pub fn solve_subset_ls(
    a: &[f64],
    m: usize,
    n: usize,
    support: &[usize],
    b: &[f64],
) -> CsResult<Vec<f64>> {
    if support.is_empty() {
        return Ok(Vec::new());
    }
    let sub = submat_columns(a, m, n, support)?;
    let k = support.len();
    // Use normal-equations with small ridge for stability if rank-deficient.
    normal_equations_solve(&sub, m, k, b, 1.0e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_eq_square() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 4.0];
        let x = normal_equations_solve(&a, 2, 2, &b, 0.0).expect("ok");
        assert!((x[0] - 3.0).abs() < 1.0e-9);
        assert!((x[1] - 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn normal_eq_overdetermined() {
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 2.0];
        let x = normal_equations_solve(&a, 3, 2, &b, 0.0).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-6);
        assert!((x[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn solve_subset_ls_basic() {
        // A 3×3 identity, support=[0,2], b=[1,2,3] -> x_S = [1, 3].
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = solve_subset_ls(&a, 3, 3, &[0, 2], &b).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-6);
        assert!((x[1] - 3.0).abs() < 1.0e-6);
    }
}
