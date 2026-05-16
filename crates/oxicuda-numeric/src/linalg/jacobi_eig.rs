//! Jacobi eigenvalue decomposition for symmetric matrices (cyclic sweep).
//!
//! Implements the classical cyclic Jacobi method: repeatedly zero out the largest
//! off-diagonal element using a Givens rotation `G(p, q, θ)` such that
//! `a_{pq} = 0` after `A' = G^T A G`.

use crate::error::{NumericError, NumericResult};

/// Compute eigenvalues and eigenvectors of a real symmetric `n × n` matrix `a`
/// (row-major). Returns `(eigvals[n], eigvecs[n*n])` where `eigvecs[i*n + j]` is the
/// `j`-th component of the `i`-th eigenvector.
pub fn jacobi_eig_symmetric(
    a: &[f64],
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if a.len() != n * n {
        return Err(NumericError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    let mut m = a.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q].powi(2);
            }
        }
        if off.sqrt() < tol {
            break;
        }
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let apq = m[p * n + q];
                if apq.abs() < 1.0e-300 {
                    continue;
                }
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                m[p * n + p] = app - t * apq;
                m[q * n + q] = aqq + t * apq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = m[r * n + p];
                        let arq = m[r * n + q];
                        m[r * n + p] = c * arp - s * arq;
                        m[p * n + r] = m[r * n + p];
                        m[r * n + q] = s * arp + c * arq;
                        m[q * n + r] = m[r * n + q];
                    }
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut eigvals = vec![0.0_f64; n];
    for i in 0..n {
        eigvals[i] = m[i * n + i];
    }
    // eigenvectors as rows: row i = i-th eigenvector
    let mut eigvecs = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            eigvecs[i * n + j] = v[j * n + i];
        }
    }
    Ok((eigvals, eigvecs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonal() {
        let a = vec![2.0_f64, 0.0, 0.0, 3.0];
        let (evs, _) = jacobi_eig_symmetric(&a, 2, 50, 1.0e-12).expect("ok");
        let mut s = evs.clone();
        s.sort_by(|x, y| x.partial_cmp(y).expect("ord"));
        assert!((s[0] - 2.0).abs() < 1.0e-10);
        assert!((s[1] - 3.0).abs() < 1.0e-10);
    }

    #[test]
    fn jacobi_two_by_two() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let a = vec![2.0_f64, 1.0, 1.0, 2.0];
        let (evs, _) = jacobi_eig_symmetric(&a, 2, 50, 1.0e-12).expect("ok");
        let mut s = evs.clone();
        s.sort_by(|x, y| x.partial_cmp(y).expect("ord"));
        assert!((s[0] - 1.0).abs() < 1.0e-10);
        assert!((s[1] - 3.0).abs() < 1.0e-10);
    }

    #[test]
    fn jacobi_three_by_three() {
        // diag(1,2,3) → eigenvalues 1, 2, 3
        let a = vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let (evs, _) = jacobi_eig_symmetric(&a, 3, 50, 1.0e-12).expect("ok");
        let mut s = evs.clone();
        s.sort_by(|x, y| x.partial_cmp(y).expect("ord"));
        assert!((s[0] - 1.0).abs() < 1.0e-10);
        assert!((s[1] - 2.0).abs() < 1.0e-10);
        assert!((s[2] - 3.0).abs() < 1.0e-10);
    }
}
