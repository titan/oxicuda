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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a guaranteed-SPD matrix `MᵀM` (row-major, n×n) from a row-major `M`.
    /// For non-singular `M` the product is symmetric positive-definite.
    fn mtm(m: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += m[k * n + i] * m[k * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Reconstruct `B · diag(d) · Bᵀ` (row-major, n×n) from eigenvectors `B`
    /// (column `i` is the eigenvector of `d[i]`) and eigenvalues `d`.
    fn reconstruct(b: &[f64], d: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += b[i * n + k] * d[k] * b[j * n + k];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Gram matrix `BᵀB` (row-major, n×n) — equals identity iff columns of `B`
    /// are orthonormal.
    fn bt_b(b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f64;
                for k in 0..n {
                    s += b[k * n + i] * b[k * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Extract column `j` (an eigenvector) from a row-major n×n matrix.
    fn column(b: &[f64], n: usize, j: usize) -> Vec<f64> {
        (0..n).map(|r| b[r * n + j]).collect()
    }

    /// Determinant of a row-major 3×3 matrix via cofactor expansion.
    fn det3(a: &[f64]) -> f64 {
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
    }

    #[test]
    fn diagonal_matrix_eigenvalues_and_axes() {
        // diag(3,1,2) → eigenvalues {1,2,3}; sorted descending → [3,2,1];
        // eigenvectors are axis-aligned standard basis vectors.
        let n = 3;
        let a_orig = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let mut a = a_orig.clone();
        let (eigs, b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen on diagonal matrix");
        assert!((eigs[0] - 3.0).abs() < 1e-12, "eigs={eigs:?}");
        assert!((eigs[1] - 2.0).abs() < 1e-12, "eigs={eigs:?}");
        assert!((eigs[2] - 1.0).abs() < 1e-12, "eigs={eigs:?}");
        for j in 0..n {
            let col = column(&b, n, j);
            let ones = col
                .iter()
                .filter(|&&x| (x.abs() - 1.0).abs() < 1e-9)
                .count();
            let zeros = col.iter().filter(|&&x| x.abs() < 1e-9).count();
            assert_eq!(ones, 1, "column {j} not axis-aligned: {col:?}");
            assert_eq!(zeros, n - 1, "column {j} not axis-aligned: {col:?}");
        }
        let recon = reconstruct(&b, &eigs, n);
        for i in 0..n * n {
            assert!(
                (recon[i] - a_orig[i]).abs() < 1e-9,
                "reconstruction mismatch at {i}"
            );
        }
    }

    #[test]
    fn two_by_two_known_spectrum() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1 with eigenvectors (1,1)/√2 and (1,-1)/√2.
        let n = 2;
        let a_orig = vec![2.0, 1.0, 1.0, 2.0];
        let mut a = a_orig.clone();
        let (eigs, b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen 2x2");
        assert!((eigs[0] - 3.0).abs() < 1e-6, "eigs={eigs:?}");
        assert!((eigs[1] - 1.0).abs() < 1e-6, "eigs={eigs:?}");
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        // λ=3 → (1,1)/√2: equal magnitude, same sign (verified up to a global sign).
        let v3 = column(&b, n, 0);
        assert!((v3[0].abs() - inv_sqrt2).abs() < 1e-6, "v3={v3:?}");
        assert!((v3[1].abs() - inv_sqrt2).abs() < 1e-6, "v3={v3:?}");
        assert!(
            v3[0] * v3[1] > 0.0,
            "λ=3 eigenvector must share sign: {v3:?}"
        );
        // λ=1 → (1,-1)/√2: equal magnitude, opposite sign.
        let v1 = column(&b, n, 1);
        assert!((v1[0].abs() - inv_sqrt2).abs() < 1e-6, "v1={v1:?}");
        assert!((v1[1].abs() - inv_sqrt2).abs() < 1e-6, "v1={v1:?}");
        assert!(
            v1[0] * v1[1] < 0.0,
            "λ=1 eigenvector must differ in sign: {v1:?}"
        );
    }

    #[test]
    fn reconstruction_matches_original() {
        // B·diag(d)·Bᵀ ≈ A for several symmetric / SPD matrices.
        let cases: Vec<(usize, Vec<f64>)> = vec![
            (2, vec![2.0, 1.0, 1.0, 2.0]),
            (3, vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0]),
            (3, mtm(&[1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], 3)),
            (
                4,
                mtm(
                    &[
                        2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0,
                        4.0,
                    ],
                    4,
                ),
            ),
        ];
        for (n, a_orig) in &cases {
            let n = *n;
            let mut a = a_orig.clone();
            let (eigs, b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen reconstruction");
            let recon = reconstruct(&b, &eigs, n);
            for i in 0..n * n {
                assert!(
                    (recon[i] - a_orig[i]).abs() < 1e-6,
                    "n={n} recon[{i}]={} vs {}",
                    recon[i],
                    a_orig[i]
                );
            }
        }
    }

    #[test]
    fn eigenvectors_orthonormal_and_unit_norm() {
        // BᵀB ≈ I and every eigenvector has unit Euclidean norm.
        let n = 3;
        let a_orig = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let mut a = a_orig.clone();
        let (_eigs, b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen orthonormal");
        let g = bt_b(&b, n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (g[i * n + j] - expected).abs() < 1e-9,
                    "BᵀB[{i}][{j}]={}",
                    g[i * n + j]
                );
            }
        }
        for j in 0..n {
            let col = column(&b, n, j);
            assert!(
                (norm(&col) - 1.0).abs() < 1e-9,
                "column {j} norm={}",
                norm(&col)
            );
        }
    }

    #[test]
    fn spd_eigenvalues_positive() {
        // SPD matrices have strictly positive eigenvalues.
        let cases: Vec<(usize, Vec<f64>)> = vec![
            (3, vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0]),
            (3, mtm(&[1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], 3)),
        ];
        for (n, a_orig) in &cases {
            let n = *n;
            let mut a = a_orig.clone();
            let (eigs, _b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen positivity");
            for &lam in &eigs {
                assert!(lam > 1e-9, "SPD eigenvalue not positive: {lam}");
            }
        }
    }

    #[test]
    fn trace_and_det_invariants() {
        // trace(A) = Σ λ_i and det(A) = Π λ_i.
        let n = 3;
        let a_orig = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let mut a = a_orig.clone();
        let (eigs, _b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen invariants");
        let trace: f64 = (0..n).map(|i| a_orig[i * n + i]).sum();
        let eig_sum: f64 = eigs.iter().sum();
        assert!(
            (trace - eig_sum).abs() < 1e-9,
            "trace {trace} vs Σλ {eig_sum}"
        );
        let det = det3(&a_orig);
        let eig_prod: f64 = eigs.iter().product();
        assert!((det - eig_prod).abs() < 1e-9, "det {det} vs Πλ {eig_prod}");
    }

    #[test]
    fn eigenvalues_sorted_descending() {
        let n = 4;
        let a_orig = mtm(
            &[
                2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 4.0,
            ],
            4,
        );
        let mut a = a_orig.clone();
        let (eigs, _b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen sort");
        for w in eigs.windows(2) {
            assert!(w[0] >= w[1] - 1e-12, "eigenvalues not descending: {eigs:?}");
        }
    }

    #[test]
    fn eigenpair_residual_a_v_equals_lambda_v() {
        // The defining spectral identity: A·vᵢ = λᵢ·vᵢ for every eigenpair.
        let n = 3;
        let a_orig = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let mut a = a_orig.clone();
        let (eigs, b) = jacobi_eigen(&mut a, n).expect("jacobi_eigen residual");
        for (j, &lam) in eigs.iter().enumerate() {
            let v = column(&b, n, j);
            let av = b_mv(&a_orig, &v, n); // A·v (row-major matrix-vector product)
            for i in 0..n {
                assert!(
                    (av[i] - lam * v[i]).abs() < 1e-9,
                    "A·v ≠ λv at component {i}: {} vs {}",
                    av[i],
                    lam * v[i]
                );
            }
        }
    }
}
