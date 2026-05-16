//! Householder QR factorization and related helpers.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;

/// Reduced QR factorisation of a row-major `m x n` matrix (m >= n).
///
/// Returns `(Q, R)` where `Q` is `m x n` with orthonormal columns
/// and `R` is `n x n` upper-triangular.
pub fn householder_qr(a: &[f64], m: usize, n: usize) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    if m == 0 || n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if a.len() != m * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if m < n {
        return Err(ManifoldError::InvalidParameter {
            name: "shape".into(),
            reason: format!("need m >= n, got m={m}, n={n}"),
        });
    }
    let mut r = a.to_vec();
    // Q starts as identity (m x m), but we only need m x n.
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    for k in 0..n {
        // Build Householder reflector for column k below pivot row k
        let mut alpha2 = 0.0;
        for i in k..m {
            alpha2 += r[i * n + k] * r[i * n + k];
        }
        let alpha = if r[k * n + k] >= 0.0 {
            -alpha2.sqrt()
        } else {
            alpha2.sqrt()
        };
        if alpha.abs() < 1e-300 {
            continue;
        }
        let mut v = vec![0.0; m - k];
        v[0] = r[k * n + k] - alpha;
        for i in 1..(m - k) {
            v[i] = r[(k + i) * n + k];
        }
        let v_norm2: f64 = v.iter().map(|x| x * x).sum();
        if v_norm2 < 1e-300 {
            continue;
        }
        // Apply (I - 2 v v^T / v_norm2) to columns of R from row k down
        for col in k..n {
            let mut dot = 0.0;
            for i in 0..(m - k) {
                dot += v[i] * r[(k + i) * n + col];
            }
            let scale = 2.0 * dot / v_norm2;
            for i in 0..(m - k) {
                r[(k + i) * n + col] -= scale * v[i];
            }
        }
        // Apply same reflector to columns of Q (we maintain Q^T accumulation)
        for col in 0..m {
            let mut dot = 0.0;
            for i in 0..(m - k) {
                dot += v[i] * q[(k + i) * m + col];
            }
            let scale = 2.0 * dot / v_norm2;
            for i in 0..(m - k) {
                q[(k + i) * m + col] -= scale * v[i];
            }
        }
    }
    // q currently is Q^T (m x m). Transpose first n columns into reduced Q (m x n).
    let mut q_red = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            q_red[row * n + col] = q[col * m + row];
        }
    }
    let mut r_red = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            r_red[i * n + j] = r[i * n + j];
        }
    }
    Ok((q_red, r_red))
}

/// Polar decomposition orthogonalisation: returns the orthogonal factor of a square matrix.
///
/// Uses `Q = M (M^T M)^{-1/2}`.
pub fn polar_orthogonal(m: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if m.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![m.len()],
        });
    }
    // Compute M^T M
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += m[k * n + i] * m[k * n + j];
            }
            s[i * n + j] = acc;
        }
    }
    // Eigendecompose S = V diag(lam) V^T
    let (w, v) = jacobi_eigh(&s, n)?;
    // S^{-1/2} = V diag(1/sqrt(lam)) V^T
    let mut s_inv_sqrt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                if w[k].abs() < 1e-14 {
                    return Err(ManifoldError::SingularMatrix(
                        "polar_orthogonal: zero singular value".into(),
                    ));
                }
                let inv_sqrt = 1.0 / w[k].abs().sqrt();
                acc += v[i * n + k] * v[j * n + k] * inv_sqrt;
            }
            s_inv_sqrt[i * n + j] = acc;
        }
    }
    // Q = M * s_inv_sqrt
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += m[i * n + k] * s_inv_sqrt[k * n + j];
            }
            q[i * n + j] = acc;
        }
    }
    Ok(q)
}

/// Forward substitution: solve `L x = b` where `L` is `n x n` lower-triangular row-major.
pub fn solve_lower_triangular(l: &[f64], b: &[f64], n: usize) -> ManifoldResult<Vec<f64>> {
    if l.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![l.len()],
        });
    }
    if b.len() != n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n],
            got: vec![b.len()],
        });
    }
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= l[i * n + j] * x[j];
        }
        let d = l[i * n + i];
        if d.abs() < 1e-14 {
            return Err(ManifoldError::SingularMatrix(
                "solve_lower_triangular: zero pivot".into(),
            ));
        }
        x[i] = acc / d;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += a[i * k + r] * b[r * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn qr_identity() {
        let n = 3;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let (q, r) = householder_qr(&a, n, n).expect("ok");
        let rec = matmul(&q, &r, n, n, n);
        for i in 0..n * n {
            assert!((rec[i] - a[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn qr_orthonormal_columns() {
        let m = 5;
        let n = 3;
        let a: Vec<f64> = (0..m * n).map(|k| (k % 7) as f64 - 3.0).collect();
        let (q, _r) = householder_qr(&a, m, n).expect("ok");
        for a_col in 0..n {
            for b_col in 0..n {
                let mut acc = 0.0;
                for r in 0..m {
                    acc += q[r * n + a_col] * q[r * n + b_col];
                }
                let tgt = if a_col == b_col { 1.0 } else { 0.0 };
                assert!((acc - tgt).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn qr_reconstructs() {
        let m = 4;
        let n = 3;
        let a: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let (q, r) = householder_qr(&a, m, n).expect("ok");
        let rec = matmul(&q, &r, m, n, n);
        for i in 0..m * n {
            assert!(
                (rec[i] - a[i]).abs() < 1e-9,
                "i={i} got {} exp {}",
                rec[i],
                a[i]
            );
        }
    }

    #[test]
    fn polar_orthogonal_orthonormal() {
        let n = 3;
        let m = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let q = polar_orthogonal(&m, n).expect("ok");
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += q[k * n + i] * q[k * n + j];
                }
                let tgt = if i == j { 1.0 } else { 0.0 };
                assert!((acc - tgt).abs() < 1e-8);
            }
        }
    }
}
