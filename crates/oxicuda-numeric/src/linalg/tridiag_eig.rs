//! Symmetric tridiagonal eigensolver: QL with implicit shifts (`tql2`).
//!
//! Given a real symmetric tridiagonal matrix specified by its diagonal `d[0..n]`
//! and off-diagonal `e[0..n]` (with `e[0]` ignored and `e[i]` the coupling
//! between rows `i-1` and `i`, i.e. the sub/super-diagonal), this computes all
//! eigenvalues and, optionally, the first component of each eigenvector.
//!
//! The algorithm is the classic implicit-shift QL iteration (Wilkinson shift,
//! Givens rotations applied bottom-up) of Bowdler, Martin, Reinsch & Wilkinson
//! (Handbook for Automatic Computation, vol. II), as popularised by EISPACK's
//! `tql2` and Numerical Recipes' `tqli`. It is the method of choice for the
//! Golub-Welsch and Laurie Jacobi-Kronrod matrices used by Gaussian quadrature,
//! because the quadrature weights are recovered from the *first component* of
//! each (normalised) eigenvector — which this routine accumulates directly,
//! avoiding the cost and rounding of forming the full eigenvector matrix.
//!
//! All inputs are validated and a [`NumericError`] is returned on failure; the
//! routine never panics.

use crate::error::{NumericError, NumericResult};

/// Outcome of a symmetric-tridiagonal eigen-decomposition.
///
/// Eigenvalues are returned in ascending order; `first_components[k]` is the
/// first entry of the unit-norm eigenvector belonging to `eigenvalues[k]`.
#[derive(Debug, Clone)]
pub struct TridiagEig {
    /// Eigenvalues, ascending.
    pub eigenvalues: Vec<f64>,
    /// First component of each (normalised) eigenvector, aligned with
    /// [`eigenvalues`](Self::eigenvalues).
    pub first_components: Vec<f64>,
}

/// `hypot` without overflow/underflow: `sqrt(a² + b²)`.
fn pythag(a: f64, b: f64) -> f64 {
    let aa = a.abs();
    let ab = b.abs();
    if aa > ab {
        let r = ab / aa;
        aa * (1.0 + r * r).sqrt()
    } else if ab == 0.0 {
        0.0
    } else {
        let r = aa / ab;
        ab * (1.0 + r * r).sqrt()
    }
}

/// Eigenvalues and first eigenvector components of a symmetric tridiagonal
/// matrix via the implicit-shift QL algorithm.
///
/// * `diag` — the `n` diagonal entries `T[i, i]`.
/// * `off` — the `n` entries where `off[i]` (for `i ≥ 1`) is the coupling
///   `T[i-1, i] = T[i, i-1]`; `off[0]` is ignored. (This matches the EISPACK
///   convention; pass `0.0` for the first slot.)
///
/// The "first component" `z_0` of an eigenvector is exactly what Gauss-type
/// quadrature needs: the weight is `μ₀ · z_0²`, where `μ₀ = ∫ dλ` is the zeroth
/// moment of the weight (for Legendre on `[-1, 1]`, `μ₀ = 2`).
///
/// # Errors
/// * [`NumericError::EmptyInput`] if `n == 0`.
/// * [`NumericError::DimensionMismatch`] if the slice lengths disagree.
/// * [`NumericError::NotConverged`] if the QL sweep fails to deflate within the
///   iteration budget (50 per eigenvalue, as in EISPACK).
pub fn tridiag_eig_ql(diag: &[f64], off: &[f64]) -> NumericResult<TridiagEig> {
    let n = diag.len();
    if n == 0 {
        return Err(NumericError::EmptyInput);
    }
    if off.len() != n {
        return Err(NumericError::DimensionMismatch {
            a: diag.len(),
            b: off.len(),
        });
    }

    let mut d = diag.to_vec();
    // Working sub-diagonal e[0..n-1]; e[n-1] is a sentinel set to 0.
    // EISPACK convention shifts `off` down by one: e[i] := off[i+1].
    let mut e = vec![0.0_f64; n];
    e[..n - 1].copy_from_slice(&off[1..n]);
    e[n - 1] = 0.0;

    // z holds the first row of the accumulated rotation matrix; initialised to
    // the identity's first row e_0 = (1, 0, …, 0). After convergence z[k] is the
    // first component of the k-th eigenvector.
    let mut z = vec![0.0_f64; n];
    z[0] = 1.0;

    if n == 1 {
        return Ok(TridiagEig {
            eigenvalues: d,
            first_components: z,
        });
    }

    for l in 0..n {
        let mut iter = 0usize;
        loop {
            // Look for a small sub-diagonal element to split the matrix.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 50 {
                return Err(NumericError::NotConverged {
                    iter,
                    residual: e[l].abs(),
                });
            }

            // Form the Wilkinson shift from the trailing 2×2 block.
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = pythag(g, 1.0);
            let denom = g + if g >= 0.0 { r.abs() } else { -r.abs() };
            g = d[m] - d[l] + e[l] / denom;

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;

            // Plane rotations sweeping from m-1 down to l (QL: bottom-up).
            let mut i = m;
            while i > l {
                i -= 1;
                let mut f = s * e[i];
                let b = c * e[i];
                r = pythag(f, g);
                e[i + 1] = r;
                if r == 0.0 {
                    // Recover from underflow.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                // Accumulate the first eigenvector component only.
                f = z[i + 1];
                z[i + 1] = s * z[i] + c * f;
                z[i] = c * z[i] - s * f;
            }

            if r == 0.0 && i >= l {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    // Sort ascending, carrying the eigenvector first-components along.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));
    let eigenvalues: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let first_components: Vec<f64> = order.iter().map(|&i| z[i]).collect();

    Ok(TridiagEig {
        eigenvalues,
        first_components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_element() {
        let r = tridiag_eig_ql(&[3.5], &[0.0]).expect("ok");
        assert_eq!(r.eigenvalues.len(), 1);
        assert!((r.eigenvalues[0] - 3.5).abs() < 1.0e-14);
        assert!((r.first_components[0].abs() - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn two_by_two_symmetric() {
        // [[2, 1], [1, 2]] → eigenvalues 1 and 3.
        let r = tridiag_eig_ql(&[2.0, 2.0], &[0.0, 1.0]).expect("ok");
        assert!((r.eigenvalues[0] - 1.0).abs() < 1.0e-12);
        assert!((r.eigenvalues[1] - 3.0).abs() < 1.0e-12);
        // For eigenvalue 1 eigenvector ∝ (1,-1)/√2, for 3 ∝ (1,1)/√2.
        assert!((r.first_components[0].abs() - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-12);
        assert!((r.first_components[1].abs() - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn matches_legendre_jacobi_three() {
        // Golub-Welsch Jacobi matrix for 3-pt Gauss-Legendre.
        // b_i = i / sqrt(4 i² - 1); nodes are ±sqrt(3/5), 0.
        let n = 3;
        let diag = vec![0.0_f64; n];
        let mut off = vec![0.0_f64; n];
        for (i, oi) in off.iter_mut().enumerate().skip(1) {
            *oi = (i as f64) / (4.0 * (i as f64).powi(2) - 1.0).sqrt();
        }
        let r = tridiag_eig_ql(&diag, &off).expect("ok");
        let want = (3.0_f64 / 5.0).sqrt();
        assert!((r.eigenvalues[0] + want).abs() < 1.0e-12);
        assert!(r.eigenvalues[1].abs() < 1.0e-12);
        assert!((r.eigenvalues[2] - want).abs() < 1.0e-12);
        // Weights = 2 z_0²: 5/9, 8/9, 5/9.
        let w: Vec<f64> = r.first_components.iter().map(|c| 2.0 * c * c).collect();
        assert!((w[0] - 5.0 / 9.0).abs() < 1.0e-12);
        assert!((w[1] - 8.0 / 9.0).abs() < 1.0e-12);
        assert!((w[2] - 5.0 / 9.0).abs() < 1.0e-12);
    }

    #[test]
    fn weights_first_component_recovers_quadrature_n5() {
        // 5-pt Gauss-Legendre weights must sum to 2 and reproduce ∫ x⁴ = 2/5.
        let n = 5;
        let diag = vec![0.0_f64; n];
        let mut off = vec![0.0_f64; n];
        for (i, oi) in off.iter_mut().enumerate().skip(1) {
            *oi = (i as f64) / (4.0 * (i as f64).powi(2) - 1.0).sqrt();
        }
        let r = tridiag_eig_ql(&diag, &off).expect("ok");
        let w: Vec<f64> = r.first_components.iter().map(|c| 2.0 * c * c).collect();
        let wsum: f64 = w.iter().sum();
        assert!((wsum - 2.0).abs() < 1.0e-12, "wsum={wsum}");
        let quad: f64 = r
            .eigenvalues
            .iter()
            .zip(w.iter())
            .map(|(&x, &wi)| wi * x.powi(4))
            .sum();
        assert!((quad - 2.0 / 5.0).abs() < 1.0e-12, "quad={quad}");
    }

    #[test]
    fn errors_on_empty_and_mismatch() {
        assert!(matches!(
            tridiag_eig_ql(&[], &[]),
            Err(NumericError::EmptyInput)
        ));
        assert!(matches!(
            tridiag_eig_ql(&[1.0, 2.0], &[0.0]),
            Err(NumericError::DimensionMismatch { .. })
        ));
    }
}
