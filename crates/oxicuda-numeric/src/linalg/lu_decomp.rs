//! LU decomposition with partial pivoting.

use crate::error::{NumericError, NumericResult};

/// Compute the LU decomposition of an `n × n` matrix `a` with partial pivoting.
/// Returns `(lu, piv, sign)` where `lu` has `L` (below diagonal) and `U` (on/above),
/// `piv[i]` is the original row swapped to position `i`, and `sign` is ±1.
pub fn lu_decompose(a: &[f64], n: usize) -> NumericResult<(Vec<f64>, Vec<usize>, i32)> {
    if a.len() != n * n {
        return Err(NumericError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut lu = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign = 1_i32;

    for k in 0..n {
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1.0e-300 {
            return Err(NumericError::SingularMatrix(format!(
                "pivot at k={k} is effectively zero"
            )));
        }
        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            piv.swap(k, max_row);
            sign = -sign;
        }
        let pivot = lu[k * n + k];
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                let upd = factor * lu[k * n + j];
                lu[i * n + j] -= upd;
            }
        }
    }
    Ok((lu, piv, sign))
}

/// Solve `A x = b` given the LU decomposition.
pub fn lu_solve(lu: &[f64], piv: &[usize], n: usize, b: &[f64]) -> NumericResult<Vec<f64>> {
    if lu.len() != n * n {
        return Err(NumericError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![lu.len()],
        });
    }
    if b.len() != n {
        return Err(NumericError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        x[i] = b[piv[i]];
    }
    // forward substitution L y = P b
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu[i * n + j] * x[j];
        }
        x[i] = s;
    }
    // back substitution U x = y
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= lu[i * n + j] * x[j];
        }
        let diag = lu[i * n + i];
        if diag.abs() < 1.0e-300 {
            return Err(NumericError::SingularMatrix(format!(
                "U diagonal at i={i} is zero"
            )));
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Compute the determinant via LU.
pub fn lu_det(lu: &[f64], n: usize, sign: i32) -> f64 {
    let mut d = sign as f64;
    for i in 0..n {
        d *= lu[i * n + i];
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_two_by_two() {
        let a = vec![4.0_f64, 3.0, 6.0, 3.0];
        let (lu, piv, sign) = lu_decompose(&a, 2).expect("ok");
        let det = lu_det(&lu, 2, sign);
        assert!((det - (4.0 * 3.0 - 3.0 * 6.0)).abs() < 1.0e-10);
        let _ = piv;
    }

    #[test]
    fn lu_solve_three() {
        let a = vec![2.0_f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let (lu, piv, _sign) = lu_decompose(&a, 3).expect("ok");
        let b = vec![1.0_f64, 0.0, 1.0];
        let x = lu_solve(&lu, &piv, 3, &b).expect("ok");
        // verify A x ≈ b
        for i in 0..3 {
            let mut s = 0.0_f64;
            for j in 0..3 {
                s += a[i * 3 + j] * x[j];
            }
            assert!((s - b[i]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn lu_singular_detected() {
        // singular row-2 of zeros
        let a = vec![1.0_f64, 2.0, 2.0, 4.0];
        let res = lu_decompose(&a, 2);
        assert!(res.is_err());
    }
}
