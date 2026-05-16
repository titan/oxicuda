//! Triangular and Cholesky-based solves.

use crate::error::{SurvivalError, SurvivalResult};

/// Solve `L y = b` for lower-triangular `L` (forward substitution).
pub(crate) fn forward_substitute(l: &[f64], b: &[f64], n: usize) -> SurvivalResult<Vec<f64>> {
    if l.len() != n * n || b.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: l.len(),
            b: b.len(),
        });
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let d = l[i * n + i];
        if d.abs() < f64::EPSILON {
            return Err(SurvivalError::SingularMatrix);
        }
        y[i] = s / d;
    }
    Ok(y)
}

/// Solve `L^T x = y` for lower-triangular `L` (back substitution).
pub(crate) fn back_substitute_transpose(
    l: &[f64],
    y: &[f64],
    n: usize,
) -> SurvivalResult<Vec<f64>> {
    if l.len() != n * n || y.len() != n {
        return Err(SurvivalError::DimensionMismatch {
            a: l.len(),
            b: y.len(),
        });
    }
    let mut x = vec![0.0_f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let d = l[i * n + i];
        if d.abs() < f64::EPSILON {
            return Err(SurvivalError::SingularMatrix);
        }
        x[i] = s / d;
    }
    Ok(x)
}

/// Solve `A x = b` for SPD `A` using Cholesky.
pub(crate) fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> SurvivalResult<Vec<f64>> {
    use crate::linalg::cholesky::cholesky;
    let l = cholesky(a, n, 1.0e-12)?;
    let y = forward_substitute(&l, b, n)?;
    back_substitute_transpose(&l, &y, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_spd_2x2() {
        // A = [[4, 2], [2, 5]]; b = [6, 7] => x = [1, 1]
        let a = vec![4.0, 2.0, 2.0, 5.0];
        let b = vec![6.0, 7.0];
        let x = cholesky_solve(&a, &b, 2).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-8);
        assert!((x[1] - 1.0).abs() < 1.0e-8);
    }

    #[test]
    fn forward_subst_identity() {
        let l = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 4.0];
        let y = forward_substitute(&l, &b, 2).expect("ok");
        assert_eq!(y, vec![3.0, 4.0]);
    }
}
