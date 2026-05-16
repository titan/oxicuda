//! One-sided Jacobi SVD for dense `m × n` matrices stored row-major.
//!
//! ## Algorithm
//!
//! Given `A` of shape `(m, n)`, we factor `A = U · diag(s) · V^T` with `U` of shape
//! `(m, k)`, `s` of length `k`, and `V` of shape `(n, k)` where `k = min(m, n)`.
//!
//! One-sided Jacobi works by rotating columns of `A` (and `V`) so that `A^T A` becomes
//! diagonal. After convergence, the column norms of the rotated `A` are the singular
//! values, and dividing each column by its norm yields the corresponding `U` column.
//!
//! Each sweep visits every pair `(p, q)` with `p < q`. The rotation angle is
//! `tan(2θ) = 2 a_pq / (a_pp - a_qq)` where `a_pq = A[:,p]·A[:,q]` etc. The convergence
//! criterion is that for the entire sweep no `|a_pq|/√(a_pp·a_qq)` exceeds `tol`.
//!
//! ## Properties
//!
//! - Numerical accuracy: O(ulp) for well-conditioned inputs.
//! - Sweeps: ~5–8 for matrices up to 50×50.
//! - Output convention: singular values are returned in **descending** order along with
//!   reordered `U` and `V` columns.

use crate::{TnError, TnResult};

/// SVD result: `A = U * diag(s) * V^T`.
///
/// - `u`: `(m, k)` row-major
/// - `s`: length `k`, descending
/// - `vt`: `(k, n)` row-major (i.e. `V^T`, so each row is an `n`-vector)
#[derive(Debug, Clone)]
pub struct SvdResult {
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// Compute the SVD of an `m × n` matrix using one-sided Jacobi.
///
/// `matrix` is consumed (no need to copy). Returns the *thin* SVD: `k = min(m, n)`.
///
/// # Errors
/// - [`TnError::EmptyInput`] if `m == 0` or `n == 0`.
/// - [`TnError::ShapeMismatch`] if `matrix.len() != m * n`.
/// - [`TnError::NotConverged`] if Jacobi does not converge in `max_sweeps`.
pub fn svd_jacobi(matrix: &[f64], m: usize, n: usize) -> TnResult<SvdResult> {
    svd_jacobi_full(matrix, m, n, 60, 1e-13)
}

/// Compute the SVD with explicit `max_sweeps` and `tol` parameters.
///
/// `tol` is applied to the relative off-diagonal ratio of `A^T A`.
pub fn svd_jacobi_full(
    matrix: &[f64],
    m: usize,
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> TnResult<SvdResult> {
    if m == 0 || n == 0 {
        return Err(TnError::EmptyInput);
    }
    if matrix.len() != m * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![matrix.len()],
        });
    }

    // Work in a square `(N, n)` working buffer with N >= m. We use the "fat" working
    // version: A is m×n, we rotate columns. Storage row-major.
    let mut a: Vec<f64> = matrix.to_vec();
    // V starts as identity n×n
    let mut v: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // One-sided Jacobi: rotate column pairs until off-diagonal `<a_p, a_q>` is small
    // relative to `||a_p|| * ||a_q||`. We track per-sweep rotation count: when no
    // rotations were applied, the matrix is converged.
    let mut rotations_this_sweep = usize::MAX;
    let mut sweeps_done = 0usize;
    while rotations_this_sweep > 0 && sweeps_done < max_sweeps {
        rotations_this_sweep = 0;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut app = 0.0f64;
                let mut aqq = 0.0f64;
                let mut apq = 0.0f64;
                for i in 0..m {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    app += aip * aip;
                    aqq += aiq * aiq;
                    apq += aip * aiq;
                }

                // Rotation worth doing only if |apq| > tol * sqrt(app*aqq).
                let prod = app * aqq;
                if prod < 1.0e-300 {
                    continue;
                }
                if apq.abs() < tol * prod.sqrt() {
                    continue;
                }

                let (c, s) = givens_angles(app, aqq, apq);
                // Apply rotation to columns of A
                for i in 0..m {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip + s * aiq;
                    a[i * n + q] = -s * aip + c * aiq;
                }
                // Apply rotation to columns of V
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip + s * viq;
                    v[i * n + q] = -s * vip + c * viq;
                }
                rotations_this_sweep += 1;
            }
        }
        sweeps_done += 1;
    }
    if rotations_this_sweep > 0 {
        return Err(TnError::NotConverged { iter: max_sweeps });
    }

    // After rotations, the columns of A are orthogonal: column norms = singular values.
    let k = m.min(n);
    let mut sigma = vec![0.0f64; n];
    for j in 0..n {
        let mut s2 = 0.0f64;
        for i in 0..m {
            s2 += a[i * n + j] * a[i * n + j];
        }
        sigma[j] = s2.sqrt();
    }

    // Sort indices by descending sigma
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Truncate to k = min(m, n)
    let mut s_out = vec![0.0f64; k];
    let mut u_out = vec![0.0f64; m * k];
    let mut vt_out = vec![0.0f64; k * n];

    for (new_col, &old_col) in order.iter().enumerate().take(k) {
        let sv = sigma[old_col];
        s_out[new_col] = sv;
        // u_out[:, new_col] = A[:, old_col] / sv  (or unit vector fallback)
        if sv > 1e-300 {
            for i in 0..m {
                u_out[i * k + new_col] = a[i * n + old_col] / sv;
            }
        } else {
            // Fall back: pick a canonical orthonormal vector orthogonal to existing U cols
            for i in 0..m {
                u_out[i * k + new_col] = if i == new_col { 1.0 } else { 0.0 };
            }
        }
        // vt_out[new_col, :] = V[:, old_col]
        for r in 0..n {
            vt_out[new_col * n + r] = v[r * n + old_col];
        }
    }

    // Re-orthonormalize U against rank-deficient columns via modified Gram-Schmidt.
    // (Only matters when some sigma <= 1e-300.)
    if s_out.iter().any(|&x| x <= 1e-300) {
        gram_schmidt_columns(&mut u_out, m, k);
    }

    Ok(SvdResult {
        u: u_out,
        s: s_out,
        vt: vt_out,
        m,
        n,
        k,
    })
}

/// Stable Givens rotation coefficients `(c, s)`.
///
/// With our column rotation convention
/// `[a_p_new, a_q_new] = [c*a_p + s*a_q, -s*a_p + c*a_q]`,
/// the new `<a_p_new, a_q_new>` is `sc*(aqq - app) + (c² - s²)*apq`. Setting this to
/// zero gives `cot(2θ) = (app - aqq) / (2 apq)`. Using Rutishauser's stable form for
/// `tan(θ) = t`: `t = sign(ϑ)/(|ϑ| + sqrt(1 + ϑ²))` with `ϑ = cot(2θ)`.
fn givens_angles(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    if apq.abs() < 1e-300 {
        return (1.0, 0.0);
    }
    let theta = (app - aqq) / (2.0 * apq);
    let t = if theta.abs() > 1.0e8 {
        0.5 / theta
    } else {
        theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

/// Modified Gram-Schmidt orthonormalisation of columns of an `m × k` row-major matrix.
fn gram_schmidt_columns(mat: &mut [f64], m: usize, k: usize) {
    for j in 0..k {
        // Subtract projections onto previous columns
        for i in 0..j {
            let mut dot = 0.0;
            for r in 0..m {
                dot += mat[r * k + i] * mat[r * k + j];
            }
            for r in 0..m {
                mat[r * k + j] -= dot * mat[r * k + i];
            }
        }
        // Normalise
        let mut nrm2 = 0.0;
        for r in 0..m {
            nrm2 += mat[r * k + j] * mat[r * k + j];
        }
        if nrm2 > 1e-300 {
            let nrm = nrm2.sqrt();
            for r in 0..m {
                mat[r * k + j] /= nrm;
            }
        } else {
            // Replace with canonical basis vector orthogonal to all previous
            for r in 0..m {
                mat[r * k + j] = 0.0;
            }
            for r in 0..m {
                mat[r * k + j] = if r == j { 1.0 } else { 0.0 };
            }
            // Re-orthogonalise vs previous
            for i in 0..j {
                let mut dot = 0.0;
                for r in 0..m {
                    dot += mat[r * k + i] * mat[r * k + j];
                }
                for r in 0..m {
                    mat[r * k + j] -= dot * mat[r * k + i];
                }
            }
            let mut n2 = 0.0;
            for r in 0..m {
                n2 += mat[r * k + j] * mat[r * k + j];
            }
            if n2 > 1e-300 {
                let nrm = n2.sqrt();
                for r in 0..m {
                    mat[r * k + j] /= nrm;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(svd: &SvdResult) -> Vec<f64> {
        // A_hat = U * diag(s) * V^T  (m,k)*(k,n)
        let m = svd.m;
        let n = svd.n;
        let k = svd.k;
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += svd.u[i * k + c] * svd.s[c] * svd.vt[c * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn svd_identity_3() {
        let mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = svd_jacobi(&mat, 3, 3).expect("ok");
        for &v in &r.s {
            assert!((v - 1.0).abs() < 1e-10);
        }
        let rec = reconstruct(&r);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }

    #[test]
    fn svd_diagonal_descending() {
        let mat = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let r = svd_jacobi(&mat, 3, 3).expect("ok");
        assert!((r.s[0] - 3.0).abs() < 1e-10);
        assert!((r.s[1] - 2.0).abs() < 1e-10);
        assert!((r.s[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn svd_2x3_thin() {
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = svd_jacobi(&mat, 2, 3).expect("ok");
        assert_eq!(r.k, 2);
        let rec = reconstruct(&r);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }

    #[test]
    fn svd_3x2_thin() {
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = svd_jacobi(&mat, 3, 2).expect("ok");
        assert_eq!(r.k, 2);
        let rec = reconstruct(&r);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }

    #[test]
    fn svd_random_4x4() {
        use crate::handle::LcgRng;
        let mut r = LcgRng::new(11);
        let m = 4;
        let n = 4;
        let mat: Vec<f64> = (0..m * n).map(|_| r.next_normal()).collect();
        let s = svd_jacobi(&mat, m, n).expect("ok");
        let rec = reconstruct(&s);
        assert!(fro_diff(&rec, &mat) < 1e-9);
        // singular values descending
        for i in 1..s.k {
            assert!(s.s[i - 1] + 1e-12 >= s.s[i]);
        }
    }

    #[test]
    fn svd_rank_deficient() {
        // rank-1 matrix: a*b^T
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let mut mat = vec![0.0; 6];
        for i in 0..3 {
            for j in 0..2 {
                mat[i * 2 + j] = a[i] * b[j];
            }
        }
        let s = svd_jacobi(&mat, 3, 2).expect("ok");
        assert!(s.s[1].abs() < 1e-10);
        let rec = reconstruct(&s);
        assert!(fro_diff(&rec, &mat) < 1e-10);
    }
}
