//! Dense symmetric eigensolver via cyclic Jacobi rotations.
//!
//! The Rayleigh-Ritz step of LOBPCG and similar block methods reduces to a
//! small dense symmetric (or symmetric-definite) eigenproblem on a subspace of
//! dimension at most `3k`. Because the crate carries no dependency on a dense
//! linear-algebra backend, this module implements the classical cyclic-Jacobi
//! algorithm: it repeatedly applies Givens rotations that annihilate the
//! largest off-diagonal entries until the matrix is diagonal to working
//! precision, accumulating the rotations into the eigenvector matrix.
//!
//! The method is unconditionally convergent for real symmetric matrices and is
//! numerically reliable for the small, possibly ill-conditioned projected
//! matrices encountered in block eigensolvers.

use crate::error::{SparseError, SparseResult};

/// Maximum number of Jacobi sweeps before declaring failure.
const MAX_SWEEPS: usize = 100;

/// Computes all eigenvalues and eigenvectors of a dense symmetric matrix.
///
/// `a` is an `n × n` symmetric matrix stored row-major; only the symmetric
/// content is used. On success the returned tuple is
/// `(eigenvalues, eigenvectors)` where:
/// - `eigenvalues` has length `n`, sorted in ascending order;
/// - `eigenvectors` is row-major `n × n`; column `j` (i.e. entries
///   `eigenvectors[i * n + j]`) is the unit eigenvector associated with
///   `eigenvalues[j]`.
///
/// The eigenvectors are orthonormal.
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if `a.len() != n * n`, or
/// [`SparseError::ConvergenceFailure`] if the sweeps do not converge (which
/// does not happen for genuine symmetric input within the internal sweep cap).
pub fn jacobi_eigh(a: &[f64], n: usize) -> SparseResult<(Vec<f64>, Vec<f64>)> {
    if a.len() != n * n {
        return Err(SparseError::InvalidArgument(format!(
            "matrix length {} does not match n*n = {}",
            a.len(),
            n * n
        )));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if n == 1 {
        return Ok((vec![a[0]], vec![1.0]));
    }

    // Work on a symmetrised copy to be robust against tiny asymmetries
    // introduced by floating-point round-off in the caller's projection.
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }

    // Eigenvector accumulator, initialised to the identity.
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..MAX_SWEEPS {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off.sqrt() <= 1e-15 {
            return Ok(finalize(m, v, n));
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];

                // Compute the Jacobi rotation (c, s) that zeroes m[p][q].
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply the rotation to rows/columns p and q of m.
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }

                // Accumulate the rotation into the eigenvector matrix.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    Err(SparseError::ConvergenceFailure(
        "cyclic Jacobi eigensolver did not converge".to_string(),
    ))
}

/// Extracts the diagonal of the converged matrix as eigenvalues and sorts the
/// eigenpairs ascending.
fn finalize(m: Vec<f64>, v: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut eigenvalues = vec![0.0f64; n];
    for (i, slot) in eigenvalues.iter_mut().enumerate() {
        *slot = m[i * n + i];
    }

    // Sort columns by ascending eigenvalue.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        eigenvalues[i]
            .partial_cmp(&eigenvalues[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_vals = vec![0.0f64; n];
    let mut sorted_vecs = vec![0.0f64; n * n];
    for (new_j, &old_j) in order.iter().enumerate() {
        sorted_vals[new_j] = eigenvalues[old_j];
        for i in 0..n {
            sorted_vecs[i * n + new_j] = v[i * n + old_j];
        }
    }
    (sorted_vals, sorted_vecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_matrix() {
        // diag(3, 1, 2) -> sorted eigenvalues 1, 2, 3
        let a = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (vals, vecs) = jacobi_eigh(&a, 3).expect("eigh");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 2.0).abs() < 1e-12);
        assert!((vals[2] - 3.0).abs() < 1e-12);
        // Eigenvectors are unit columns; check orthonormality.
        for j in 0..3 {
            let mut norm = 0.0;
            for i in 0..3 {
                norm += vecs[i * 3 + j] * vecs[i * 3 + j];
            }
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn two_by_two_known() {
        // [[2, 1], [1, 2]] -> eigenvalues 1 and 3.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, _vecs) = jacobi_eigh(&a, 2).expect("eigh");
        assert!((vals[0] - 1.0).abs() < 1e-12);
        assert!((vals[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn eigenvectors_reconstruct_matrix() {
        // Symmetric matrix; verify A = V diag(lambda) Vᵀ.
        let a = vec![4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let n = 3;
        let (vals, vecs) = jacobi_eigh(&a, n).expect("eigh");
        // Reconstruct.
        let mut recon = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += vecs[i * n + k] * vals[k] * vecs[j * n + k];
                }
                recon[i * n + j] = acc;
            }
        }
        for idx in 0..n * n {
            assert!(
                (recon[idx] - a[idx]).abs() < 1e-9,
                "mismatch at {idx}: {} vs {}",
                recon[idx],
                a[idx]
            );
        }
    }

    #[test]
    fn ascending_order() {
        let a = vec![5.0, 0.0, 0.0, 4.0];
        let (vals, _) = jacobi_eigh(&a, 2).expect("eigh");
        assert!(vals[0] <= vals[1]);
        assert!((vals[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn wrong_size_errors() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(jacobi_eigh(&a, 2).is_err());
    }

    #[test]
    fn single_element() {
        let (vals, vecs) = jacobi_eigh(&[7.0], 1).expect("eigh");
        assert_eq!(vals, vec![7.0]);
        assert_eq!(vecs, vec![1.0]);
    }
}
