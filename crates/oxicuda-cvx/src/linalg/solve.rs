//! Dense LU and triangular solve.

use crate::error::{CvxError, CvxResult};

/// LU decomposition with partial pivoting (Doolittle form).
///
/// Returns `(lu, piv)` where `lu` is row-major `n × n` containing L below diag (unit L)
/// and U on/above diag; `piv[i]` is the pivot row swapped with row i during factorisation.
pub fn lu_decompose(a: &[f64], n: usize) -> CvxResult<(Vec<f64>, Vec<usize>)> {
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let mut lu = a.to_vec();
    let mut piv = vec![0usize; n];
    for i in 0..n {
        // Find pivot row.
        let mut max_v = lu[i * n + i].abs();
        let mut max_r = i;
        for r in (i + 1)..n {
            let v = lu[r * n + i].abs();
            if v > max_v {
                max_v = v;
                max_r = r;
            }
        }
        if max_v < 1.0e-300 {
            return Err(CvxError::SingularMatrix(format!(
                "zero pivot at column {i}"
            )));
        }
        piv[i] = max_r;
        if max_r != i {
            for c in 0..n {
                lu.swap(i * n + c, max_r * n + c);
            }
        }
        let inv_pivot = 1.0 / lu[i * n + i];
        for r in (i + 1)..n {
            let factor = lu[r * n + i] * inv_pivot;
            lu[r * n + i] = factor;
            for c in (i + 1)..n {
                let v = lu[r * n + c] - factor * lu[i * n + c];
                lu[r * n + c] = v;
            }
        }
    }
    Ok((lu, piv))
}

/// Solve `LU x = b` using a precomputed factorisation from [`lu_decompose`].
pub fn lu_solve(lu: &[f64], piv: &[usize], n: usize, b: &[f64]) -> CvxResult<Vec<f64>> {
    if lu.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![lu.len()],
        });
    }
    if piv.len() != n {
        return Err(CvxError::DimensionMismatch { a: piv.len(), b: n });
    }
    if b.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = b.to_vec();
    // Apply pivots.
    for (i, &p) in piv.iter().enumerate().take(n) {
        if p >= n {
            return Err(CvxError::IndexOutOfBounds { index: p, len: n });
        }
        if p != i {
            x.swap(i, p);
        }
    }
    // Forward substitution (L y = Pb), L unit diag.
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu[i * n + j] * x[j];
        }
        x[i] = s;
    }
    // Back substitution (U x = y).
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= lu[i * n + j] * x[j];
        }
        let d = lu[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CvxError::SingularMatrix(format!("zero U[{i},{i}]")));
        }
        x[i] = s / d;
    }
    Ok(x)
}

/// Convenience: solve `A x = b` for a fresh dense system (allocates internally).
pub fn solve_dense(a: &[f64], n: usize, b: &[f64]) -> CvxResult<Vec<f64>> {
    let (lu, piv) = lu_decompose(a, n)?;
    lu_solve(&lu, &piv, n, b)
}

/// Solve lower-triangular `L x = b` (no unit-diagonal assumption).
pub fn solve_lower(l: &[f64], n: usize, b: &[f64]) -> CvxResult<Vec<f64>> {
    if l.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![l.len()],
        });
    }
    if b.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        let d = l[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CvxError::SingularMatrix(format!("zero L[{i},{i}]")));
        }
        x[i] = s / d;
    }
    Ok(x)
}

/// Solve upper-triangular `U x = b`.
pub fn solve_upper(u: &[f64], n: usize, b: &[f64]) -> CvxResult<Vec<f64>> {
    if u.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![u.len()],
        });
    }
    if b.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= u[i * n + j] * x[j];
        }
        let d = u[i * n + i];
        if d.abs() < 1.0e-300 {
            return Err(CvxError::SingularMatrix(format!("zero U[{i},{i}]")));
        }
        x[i] = s / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solve_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![7.0, 8.0, 9.0];
        let x = solve_dense(&a, 3, &b).expect("ok");
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1.0e-12);
        }
    }

    #[test]
    fn lu_solve_pivot() {
        // A = [[0, 1], [1, 0]], so pivoting needed.
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let b = vec![3.0, 4.0];
        let x = solve_dense(&a, 2, &b).expect("ok");
        assert!((x[0] - 4.0).abs() < 1.0e-12);
        assert!((x[1] - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn solve_upper_basic() {
        let u = vec![2.0, 1.0, 0.0, 3.0];
        let x = solve_upper(&u, 2, &[5.0, 6.0]).expect("ok");
        // 3 x1 = 6 → x1=2; 2 x0 + 1*2 = 5 → x0=1.5
        assert!((x[0] - 1.5).abs() < 1.0e-12);
        assert!((x[1] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn solve_lower_basic() {
        let l = vec![2.0, 0.0, 1.0, 3.0];
        let x = solve_lower(&l, 2, &[4.0, 11.0]).expect("ok");
        // 2 x0 = 4 → x0=2; 1*2 + 3 x1 = 11 → x1=3
        assert!((x[0] - 2.0).abs() < 1.0e-12);
        assert!((x[1] - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn lu_singular_detected() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        assert!(solve_dense(&a, 2, &[1.0, 2.0]).is_err());
    }
}
