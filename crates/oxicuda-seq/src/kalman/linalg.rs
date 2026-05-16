//! Tiny dense linear-algebra helpers for Kalman filtering.

use crate::error::{SeqError, SeqResult};

/// Matrix-vector product `y = A · x` for an n×n matrix `A`.
pub fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Matrix-vector product `y = A · x` for an arbitrary m×n matrix.
pub fn matvec_rect(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Square matrix multiply `C = A · B` (n×n).
pub fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Rectangular matrix multiply `C(m×p) = A(m×n) · B(n×p)`.
pub fn matmul_rect(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = s;
        }
    }
    c
}

/// Matrix transpose.
pub fn transpose_rect(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Square matrix transpose (n×n).
pub fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    transpose_rect(a, n, n)
}

/// Elementwise add (square).
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Elementwise subtract.
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Identity matrix of size n.
pub fn eye(n: usize) -> Vec<f64> {
    let mut e = vec![0.0; n * n];
    for i in 0..n {
        e[i * n + i] = 1.0;
    }
    e
}

/// Gauss-Jordan matrix inverse with partial pivoting for an n×n matrix.
/// Returns `Err(NumericalInstability)` if the matrix is singular.
pub fn inverse(a: &[f64], n: usize) -> SeqResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(SeqError::ShapeMismatch {
            expected: n * n,
            got: a.len(),
        });
    }
    let cols = 2 * n;
    let mut aug = vec![0.0; n * cols];
    for i in 0..n {
        for j in 0..n {
            aug[i * cols + j] = a[i * n + j];
        }
        aug[i * cols + n + i] = 1.0;
    }
    for col in 0..n {
        // pivot row
        let mut pivot = col;
        let mut max_abs = aug[col * cols + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * cols + col].abs();
            if v > max_abs {
                pivot = r;
                max_abs = v;
            }
        }
        if max_abs < 1e-300 {
            return Err(SeqError::NumericalInstability(
                "singular matrix".to_string(),
            ));
        }
        if pivot != col {
            for j in 0..cols {
                aug.swap(col * cols + j, pivot * cols + j);
            }
        }
        // normalise pivot row
        let pv = aug[col * cols + col];
        for j in 0..cols {
            aug[col * cols + j] /= pv;
        }
        // eliminate
        for r in 0..n {
            if r != col {
                let factor = aug[r * cols + col];
                if factor != 0.0 {
                    for j in 0..cols {
                        aug[r * cols + j] -= factor * aug[col * cols + j];
                    }
                }
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * cols + n + j];
        }
    }
    Ok(inv)
}

/// Cholesky factor `L` such that `A = L Lᵀ` for an SPD n×n matrix.
pub fn cholesky(a: &[f64], n: usize) -> SeqResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(SeqError::ShapeMismatch {
            expected: n * n,
            got: a.len(),
        });
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(SeqError::NumericalInstability(
                        "matrix not positive definite".to_string(),
                    ));
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Determinant of n×n matrix using Gauss elimination (sign + scaled product).
pub fn determinant(a: &[f64], n: usize) -> SeqResult<f64> {
    if a.len() != n * n {
        return Err(SeqError::ShapeMismatch {
            expected: n * n,
            got: a.len(),
        });
    }
    let mut m = a.to_vec();
    let mut sign = 1.0_f64;
    for col in 0..n {
        let mut pivot = col;
        let mut max_abs = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > max_abs {
                pivot = r;
                max_abs = v;
            }
        }
        if max_abs < 1e-300 {
            return Ok(0.0);
        }
        if pivot != col {
            for j in 0..n {
                m.swap(col * n + j, pivot * n + j);
            }
            sign = -sign;
        }
        let pv = m[col * n + col];
        for r in (col + 1)..n {
            let factor = m[r * n + col] / pv;
            for j in col..n {
                m[r * n + j] -= factor * m[col * n + j];
            }
        }
    }
    let mut det = sign;
    for i in 0..n {
        det *= m[i * n + i];
    }
    Ok(det)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_identity() {
        let i = eye(3);
        let inv = inverse(&i, 3).expect("ok");
        for d in 0..3 {
            assert!((inv[d * 3 + d] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn inverse_2x2() {
        let m = vec![4.0, 7.0, 2.0, 6.0];
        let inv = inverse(&m, 2).expect("ok");
        let prod = matmul(&m, &inv, 2);
        for d in 0..2 {
            for e in 0..2 {
                let expect = if d == e { 1.0 } else { 0.0 };
                assert!((prod[d * 2 + e] - expect).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn cholesky_spd() {
        // SPD matrix
        let m = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let l = cholesky(&m, 3).expect("ok");
        let lt = transpose(&l, 3);
        let llt = matmul(&l, &lt, 3);
        for i in 0..9 {
            assert!((llt[i] - m[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn determinant_simple() {
        let m = vec![1.0, 2.0, 3.0, 4.0];
        let d = determinant(&m, 2).expect("ok");
        assert!((d - (-2.0)).abs() < 1e-9);
    }
}
