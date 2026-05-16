//! Cyclic Jacobi eigendecomposition for symmetric matrices.
//!
//! Returns eigenvalues + eigenvectors such that `A = V diag(w) V^T`.

use crate::error::{ManifoldError, ManifoldResult};

/// Symmetric eigendecomposition via cyclic-Jacobi rotations.
///
/// Input: row-major `n x n` symmetric matrix `a`.
/// Output: `(eigenvalues, eigenvectors)` — eigenvectors stored row-major,
/// column `k` of `V` is the eigenvector for eigenvalue `w[k]`.
/// Eigenvalues are NOT sorted (see [`sort_eigen_descending`]).
pub fn jacobi_eigh(a: &[f64], n: usize) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if a.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let mut m = a.to_vec();
    // Force exact symmetry — Jacobi assumes A^T = A
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = avg;
            m[j * n + i] = avg;
        }
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_sweeps = 80;
    let tol = 1e-14;
    for _sweep in 0..max_sweeps {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off.sqrt() < tol {
            break;
        }
        for p in 0..n - 1 {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-20 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta.abs() > 1e16 {
                    0.5 / theta
                } else {
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Update the symmetric matrix using the rotation.
                m[p * n + p] = app - t * apq;
                m[q * n + q] = aqq + t * apq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = m[r * n + p];
                        let arq = m[r * n + q];
                        let new_rp = c * arp - s * arq;
                        let new_rq = s * arp + c * arq;
                        m[r * n + p] = new_rp;
                        m[p * n + r] = new_rp;
                        m[r * n + q] = new_rq;
                        m[q * n + r] = new_rq;
                    }
                }
                // Update eigenvector matrix: columns p, q
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut w = vec![0.0; n];
    for i in 0..n {
        w[i] = m[i * n + i];
    }
    Ok((w, v))
}

/// Sort an eigendecomposition by eigenvalue in descending order.
///
/// Permutes both `w` and the columns of `v` (row-major, n x n).
pub fn sort_eigen_descending(w: &mut [f64], v: &mut [f64], n: usize) {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| w[j].partial_cmp(&w[i]).unwrap_or(std::cmp::Ordering::Equal));
    let w_sorted: Vec<f64> = indices.iter().map(|&i| w[i]).collect();
    let mut v_sorted = vec![0.0; n * n];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }
    w.copy_from_slice(&w_sorted);
    v.copy_from_slice(&v_sorted);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_eigh() {
        let n = 3;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let (w, _v) = jacobi_eigh(&a, n).expect("ok");
        for wi in &w {
            assert!((wi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn diagonal_eigh() {
        let n = 3;
        let mut a = vec![0.0; n * n];
        a[0] = 3.0;
        a[n + 1] = 1.0;
        a[2 * n + 2] = 2.0;
        let (mut w, mut v) = jacobi_eigh(&a, n).expect("ok");
        sort_eigen_descending(&mut w, &mut v, n);
        assert!((w[0] - 3.0).abs() < 1e-10);
        assert!((w[1] - 2.0).abs() < 1e-10);
        assert!((w[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn reconstruct_random_symmetric() {
        let n = 5;
        // Build a random symmetric matrix
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let val = ((i * 7 + j * 11) % 13) as f64 - 6.0;
                s[i * n + j] = val;
                s[j * n + i] = val;
            }
        }
        let (w, v) = jacobi_eigh(&s, n).expect("ok");
        // Check A * v_k = w_k * v_k
        for k in 0..n {
            for i in 0..n {
                let mut av = 0.0;
                for j in 0..n {
                    av += s[i * n + j] * v[j * n + k];
                }
                assert!((av - w[k] * v[i * n + k]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn orthonormal_eigenvectors() {
        let n = 4;
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let val = ((i + 1) * (j + 2)) as f64;
                s[i * n + j] = val;
                s[j * n + i] = val;
            }
        }
        let (_w, v) = jacobi_eigh(&s, n).expect("ok");
        for a in 0..n {
            for b in 0..n {
                let mut dot = 0.0;
                for r in 0..n {
                    dot += v[r * n + a] * v[r * n + b];
                }
                let target = if a == b { 1.0 } else { 0.0 };
                assert!((dot - target).abs() < 1e-8);
            }
        }
    }
}
