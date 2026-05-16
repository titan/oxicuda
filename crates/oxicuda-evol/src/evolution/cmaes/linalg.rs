//! Jacobi eigendecomposition for symmetric matrices (classical algorithm).
//!
//! Used by CMA-ES to maintain the B (eigenvectors) and D (sqrt eigenvalues) matrices
//! needed for sampling from the current covariance distribution.

use crate::{EvolError, EvolResult};

/// Decompose an n×n symmetric matrix `A` (row-major, in-place) into its eigenvectors `B`
/// and sorted-descending eigenvalues `eigenvalues`.
///
/// Returns `(eigenvalues, B)` where `B[:,i]` is the eigenvector for `eigenvalues[i]`.
///
/// The algorithm is classical Jacobi: iteratively zero off-diagonal elements via Givens
/// rotations until convergence, or until `max_sweeps` full sweeps have been performed.
///
/// # Errors
/// Returns `EvolError::EigenFailed` if the off-diagonal residual does not drop below `tol`
/// within `max_sweeps`.
pub fn jacobi_eigen(a: &mut [f64], n: usize) -> EvolResult<(Vec<f64>, Vec<f64>)> {
    const MAX_SWEEPS: usize = 100;
    const TOL: f64 = 1e-12;

    // Build identity for eigenvectors B (n×n, row-major)
    let mut b = vec![0.0f64; n * n];
    for i in 0..n {
        b[i * n + i] = 1.0;
    }

    for sweep in 0..MAX_SWEEPS {
        // Check if off-diagonal norms are below tolerance
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let v = a[p * n + q].abs();
                if v > max_off {
                    max_off = v;
                }
            }
        }
        if max_off < TOL {
            break;
        }
        if sweep == MAX_SWEEPS - 1 {
            return Err(EvolError::EigenFailed(MAX_SWEEPS));
        }

        // One full sweep over all (p, q) pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < TOL {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                // t = sign(theta) / (|theta| + sqrt(1 + theta²)), ensures |t| ≤ 1
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal elements
                let h = t * apq;
                a[p * n + p] = app - h;
                a[q * n + q] = aqq + h;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal elements: rows/cols other than p, q
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_arp = c * arp - s * arq;
                    let new_arq = s * arp + c * arq;
                    a[r * n + p] = new_arp;
                    a[p * n + r] = new_arp;
                    a[r * n + q] = new_arq;
                    a[q * n + r] = new_arq;
                }

                // Accumulate rotations in B: B[:,p] and B[:,q]
                for r in 0..n {
                    let brp = b[r * n + p];
                    let brq = b[r * n + q];
                    b[r * n + p] = c * brp - s * brq;
                    b[r * n + q] = s * brp + c * brq;
                }
            }
        }
    }

    // Extract diagonal as eigenvalues
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    // Sort eigenvalues descending; reorder columns of B accordingly
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigs: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_b = vec![0.0f64; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..n {
            sorted_b[row * n + new_col] = b[row * n + old_col];
        }
    }

    // Copy sorted eigenvalues back
    eigenvalues.copy_from_slice(&sorted_eigs);

    Ok((sorted_eigs, sorted_b))
}

/// Compute `B^T * v` where `B` is n×n column-major (each column is an eigenvector).
/// This is the standard matrix-vector product `B^T v`.
pub fn btranspose_mv(b: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            sum += b[k * n + i] * v[k];
        }
        out[i] = sum;
    }
    out
}

/// Compute `B * v` where `B` is n×n.
pub fn b_mv(b: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            sum += b[i * n + k] * v[k];
        }
        out[i] = sum;
    }
    out
}

/// Euclidean norm of a vector.
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
