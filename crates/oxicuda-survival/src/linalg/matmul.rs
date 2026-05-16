//! Dense matrix-matrix and matrix-vector multiply.

use crate::error::{SurvivalError, SurvivalResult};

/// Multiply `a` (m×k row-major) by `b` (k×n row-major). Returns m×n.
pub(crate) fn matmul(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
) -> SurvivalResult<Vec<f64>> {
    if a.len() != m * k {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![m * k],
            got: vec![a.len()],
        });
    }
    if b.len() != k * n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![k * n],
            got: vec![b.len()],
        });
    }
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            for j in 0..n {
                out[i * n + j] += aik * b[kk * n + j];
            }
        }
    }
    Ok(out)
}

/// Multiply `a` (m×n) by vector `x` (length n). Returns vector length m.
pub(crate) fn matvec(a: &[f64], x: &[f64], m: usize, n: usize) -> SurvivalResult<Vec<f64>> {
    if a.len() != m * n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![m * n],
            got: vec![a.len()],
        });
    }
    if x.len() != n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n],
            got: vec![x.len()],
        });
    }
    let mut y = vec![0.0_f64; m];
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2_identity() {
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let c = matmul(&i, &a, 2, 2, 2).expect("ok");
        assert_eq!(c, a);
    }

    #[test]
    fn matvec_works() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let y = matvec(&a, &x, 2, 2).expect("ok");
        assert_eq!(y, vec![17.0, 39.0]);
    }

    #[test]
    fn matmul_shape_mismatch() {
        // a has length 3 but m*k=4 expected
        let a = vec![1.0; 3];
        let b = vec![1.0; 6];
        assert!(matmul(&a, &b, 2, 2, 3).is_err());
    }
}
