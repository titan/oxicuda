//! One-sided Jacobi SVD for dense matrices.
//!
//! Implements rotation-based singular value decomposition. Works well for thin matrices
//! (m ≥ n) where we factor `A = U Σ V^T` with U: m×n, Σ: n×n diag, V: n×n.

use crate::error::{CsError, CsResult};

/// Compute thin SVD `A = U Σ V^T` for a row-major `m × n` matrix `A` (m ≥ n).
///
/// Returns `(u, s, v)`:
/// - `u`: row-major `m × n` (left singular vectors stacked column-wise as columns of U)
/// - `s`: length `n` singular values (sorted descending)
/// - `v`: row-major `n × n` right singular vectors as **rows** of V^T (so `v[i*n+j]` is `V[i,j]`)
///
/// Algorithm: one-sided Jacobi rotations on A·V, where V starts as identity and accumulates
/// the rotations. We sweep over column pairs (i,j) and rotate so that the i-th and j-th columns
/// become orthogonal. After convergence, column norms are singular values.
pub fn jacobi_svd_thin(a: &[f64], m: usize, n: usize) -> CsResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if m < n {
        return Err(CsError::InvalidParameter(format!(
            "thin SVD requires m≥n; got m={m}, n={n}"
        )));
    }
    let mut u = a.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_sweeps = 50;
    let tol = 1.0e-14;
    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = 0.0_f64;
                for i in 0..m {
                    let ap = u[i * n + p];
                    let aq = u[i * n + q];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                off += gamma * gamma;
                if gamma.abs() < tol * (alpha.sqrt() * beta.sqrt()).max(1.0e-300) {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    1.0 / (zeta - (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..m {
                    let ap = u[i * n + p];
                    let aq = u[i * n + q];
                    u[i * n + p] = c * ap - s * aq;
                    u[i * n + q] = s * ap + c * aq;
                }
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if off.sqrt() < tol {
            break;
        }
    }
    let mut s = vec![0.0_f64; n];
    for j in 0..n {
        let mut nn = 0.0_f64;
        for i in 0..m {
            nn += u[i * n + j] * u[i * n + j];
        }
        s[j] = nn.sqrt();
    }
    // Normalise U columns by σ.
    for j in 0..n {
        if s[j] > 1.0e-300 {
            for i in 0..m {
                u[i * n + j] /= s[j];
            }
        }
    }
    // Sort descending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a_idx, &b_idx| {
        s[b_idx]
            .partial_cmp(&s[a_idx])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut u_sorted = vec![0.0_f64; m * n];
    let mut s_sorted = vec![0.0_f64; n];
    let mut v_sorted = vec![0.0_f64; n * n];
    for (new_j, &old_j) in indices.iter().enumerate() {
        s_sorted[new_j] = s[old_j];
        for i in 0..m {
            u_sorted[i * n + new_j] = u[i * n + old_j];
        }
        for i in 0..n {
            v_sorted[i * n + new_j] = v[i * n + old_j];
        }
    }
    // Return V^T as rows -> we store V already; the doc says v[i*n+j]=V[i,j].
    // Currently v_sorted[i,j] = V[i,j] which matches.
    Ok((u_sorted, s_sorted, v_sorted))
}

/// Compute full SVD `A = U Σ V^T` for a row-major `m × n` matrix.
///
/// For `m ≥ n` calls `jacobi_svd_thin`. For `m < n` factors A^T = U' Σ' V'^T and returns
/// `(V', Σ', U')` transposed appropriately.
pub fn jacobi_svd(a: &[f64], m: usize, n: usize) -> CsResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if a.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if m >= n {
        jacobi_svd_thin(a, m, n)
    } else {
        // A^T is n × m, m < n so n > m. Apply thin SVD to A^T.
        let mut at = vec![0.0_f64; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        let (uat, sat, vat) = jacobi_svd_thin(&at, n, m)?;
        // A^T = U Σ V^T, so A = V Σ U^T.
        // Return as if SVD of A: U_a (m × m), s (length m), V_a (n × m where V_a column j is U_at column j).
        // Use thin convention: U_a is m × m, s length m, V_a is n × m.
        Ok((vat, sat, uat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_svd_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (_u, s, _v) = jacobi_svd_thin(&a, 3, 3).expect("ok");
        for &si in &s {
            assert!((si - 1.0).abs() < 1.0e-9);
        }
    }

    #[test]
    fn jacobi_svd_diagonal() {
        let a = vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        let (_u, s, _v) = jacobi_svd_thin(&a, 3, 3).expect("ok");
        assert!((s[0] - 3.0).abs() < 1.0e-9);
        assert!((s[1] - 2.0).abs() < 1.0e-9);
        assert!((s[2] - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn jacobi_svd_reconstructs() {
        // A = [[1, 2], [3, 4], [5, 6]] thin SVD.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (u, s, v) = jacobi_svd_thin(&a, 3, 2).expect("ok");
        // Reconstruct: A_rec[i, j] = sum_k u[i,k] * s[k] * v[j,k]
        for i in 0..3 {
            for j in 0..2 {
                let mut x = 0.0_f64;
                for k in 0..2 {
                    x += u[i * 2 + k] * s[k] * v[j * 2 + k];
                }
                assert!((x - a[i * 2 + j]).abs() < 1.0e-6);
            }
        }
    }
}
