//! Householder QR decomposition for general `m × n` matrices (m ≥ n).

use crate::error::{NumericError, NumericResult};

/// Householder QR. Returns `(q, r)` where `q` is `m × m` orthogonal and `r` is `m × n`
/// upper-triangular (last `m-n` rows zero).
pub fn householder_qr(a: &[f64], m: usize, n: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
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
    let kmax = n.min(m);
    for k in 0..kmax {
        // form Householder reflector for column k of R[k:, k]
        let mut norm = 0.0_f64;
        for i in k..m {
            norm += r[i * n + k].powi(2);
        }
        let norm = norm.sqrt();
        if norm < 1.0e-300 {
            continue;
        }
        let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;
        let mut v = vec![0.0_f64; m - k];
        v[0] = r[k * n + k] - alpha;
        for i in 1..(m - k) {
            v[i] = r[(k + i) * n + k];
        }
        let vnorm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if vnorm < 1.0e-300 {
            continue;
        }
        for vi in v.iter_mut() {
            *vi /= vnorm;
        }
        // R := (I - 2 v v^T) R
        for j in k..n {
            let mut dot = 0.0_f64;
            for i in 0..(m - k) {
                dot += v[i] * r[(k + i) * n + j];
            }
            let factor = 2.0 * dot;
            for i in 0..(m - k) {
                r[(k + i) * n + j] -= factor * v[i];
            }
        }
        // Q := Q (I - 2 v v^T)  → updates columns k..m of Q
        for i in 0..m {
            let mut dot = 0.0_f64;
            for jj in 0..(m - k) {
                dot += q[i * m + (k + jj)] * v[jj];
            }
            let factor = 2.0 * dot;
            for jj in 0..(m - k) {
                q[i * m + (k + jj)] -= factor * v[jj];
            }
        }
    }
    Ok((q, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], _br: usize, bc: usize) -> Vec<f64> {
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
    fn hh_qr_reconstruct() {
        let a = vec![12.0_f64, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
        let (q, r) = householder_qr(&a, 3, 3).expect("ok");
        let qr = matmul(&q, 3, 3, &r, 3, 3);
        for i in 0..9 {
            assert!((qr[i] - a[i]).abs() < 1.0e-8);
        }
        // R upper-triangular
        assert!(r[3].abs() < 1.0e-8);
        assert!(r[6].abs() < 1.0e-8);
        assert!(r[7].abs() < 1.0e-8);
    }
}
