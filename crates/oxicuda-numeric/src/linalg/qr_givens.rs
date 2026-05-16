//! QR decomposition via Givens rotations for general (possibly Hessenberg) matrices.
//!
//! Useful for the QR iteration on companion matrices (polynomial roots).

use crate::error::{NumericError, NumericResult};

/// Compute the QR decomposition of an `m × n` matrix `a` via Givens rotations.
/// Returns `(q, r)` where `q` is `m × m` and `r` is `m × n`.
pub fn qr_givens(a: &[f64], m: usize, n: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if a.len() != m * n {
        return Err(NumericError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if m == 0 || n == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut r = a.to_vec();
    let mut q = vec![0.0_f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    for j in 0..n.min(m) {
        for i in (j + 1)..m {
            let a_jj = r[j * n + j];
            let a_ij = r[i * n + j];
            if a_ij.abs() < 1.0e-300 {
                continue;
            }
            let h = (a_jj * a_jj + a_ij * a_ij).sqrt();
            if h < 1.0e-300 {
                continue;
            }
            let c = a_jj / h;
            let s = a_ij / h;
            for k in 0..n {
                let rjk = r[j * n + k];
                let rik = r[i * n + k];
                r[j * n + k] = c * rjk + s * rik;
                r[i * n + k] = -s * rjk + c * rik;
            }
            // accumulate Q^T from right side: Q = Q · G^T  ⇒  Q[:, j] = c·Q[:,j] + s·Q[:,i]
            for k in 0..m {
                let qkj = q[k * m + j];
                let qki = q[k * m + i];
                q[k * m + j] = c * qkj + s * qki;
                q[k * m + i] = -s * qkj + c * qki;
            }
        }
    }
    Ok((q, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
        assert_eq!(ac, br);
        let _ = (br, bc);
        let mut out = vec![0.0_f64; ar * bc];
        for i in 0..ar {
            for j in 0..bc {
                let mut s = 0.0_f64;
                for k in 0..ac {
                    s += a[i * ac + k] * b[k * bc + j];
                }
                out[i * bc + j] = s;
            }
        }
        out
    }

    #[test]
    fn qr_givens_two_by_two() {
        let a = vec![3.0_f64, 4.0, 0.0, -5.0];
        let (q, r) = qr_givens(&a, 2, 2).expect("ok");
        let qr = matmul(&q, 2, 2, &r, 2, 2);
        for i in 0..4 {
            assert!((qr[i] - a[i]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn qr_givens_three_by_three() {
        let a = vec![1.0_f64, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0];
        let (q, r) = qr_givens(&a, 3, 3).expect("ok");
        let qr = matmul(&q, 3, 3, &r, 3, 3);
        for i in 0..9 {
            assert!((qr[i] - a[i]).abs() < 1.0e-10);
        }
        // R should be upper-triangular.
        assert!(r[3].abs() < 1.0e-10);
        assert!(r[6].abs() < 1.0e-10);
        assert!(r[7].abs() < 1.0e-10);
    }
}
