//! Projection onto the cone of symmetric PSD matrices.
//!
//! Algorithm: symmetric eigendecomposition via classical Jacobi, then
//! reconstruct with eigenvalues clipped at 0.

use crate::error::{CvxError, CvxResult};

/// Project a symmetric `n × n` matrix (row-major) onto the PSD cone.
///
/// Returns the projected matrix (row-major) with the same shape.
pub fn project_psd_cone(a: &[f64], n: usize) -> CvxResult<Vec<f64>> {
    if n == 0 {
        return Err(CvxError::EmptyInput);
    }
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    // Symmetrise input to be safe.
    let mut s = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    let (eigvals, eigvecs) = sym_jacobi_eigen(&s, n, 200, 1.0e-14)?;
    // Clip eigenvalues to ≥ 0.
    let clipped: Vec<f64> = eigvals.into_iter().map(|v| v.max(0.0)).collect();
    // Reconstruct: M = V diag(λ) V^T.
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += eigvecs[i * n + k] * clipped[k] * eigvecs[j * n + k];
            }
            out[i * n + j] = acc;
        }
    }
    // Re-symmetrise to absorb tiny round-off.
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (out[i * n + j] + out[j * n + i]);
            out[i * n + j] = avg;
            out[j * n + i] = avg;
        }
    }
    Ok(out)
}

/// Classical Jacobi eigendecomposition for a symmetric `n × n` matrix.
///
/// Returns `(eigvals, eigvecs)` where eigvecs is row-major with eigenvector k in column k.
pub fn sym_jacobi_eigen(
    a: &[f64],
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> CvxResult<(Vec<f64>, Vec<f64>)> {
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let mut m = a.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..max_sweeps {
        // Compute off-diagonal Frobenius norm.
        let mut off_sq = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_sq += m[p * n + q] * m[p * n + q];
            }
        }
        if off_sq.sqrt() < tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1.0e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Update diagonal.
                m[p * n + p] = app - t * apq;
                m[q * n + q] = aqq + t * apq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                // Off-diagonal pq rotations.
                for r in 0..n {
                    if r != p && r != q {
                        let mrp = m[r * n + p];
                        let mrq = m[r * n + q];
                        let new_p = c * mrp - s * mrq;
                        let new_q = s * mrp + c * mrq;
                        m[r * n + p] = new_p;
                        m[p * n + r] = new_p;
                        m[r * n + q] = new_q;
                        m[q * n + r] = new_q;
                    }
                }
                // Update eigenvector matrix.
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let eigvals: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    Ok((eigvals, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psd_identity_unchanged() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p = project_psd_cone(&a, 3).expect("ok");
        for (pi, ai) in p.iter().zip(a.iter()) {
            assert!((pi - ai).abs() < 1.0e-9);
        }
    }

    #[test]
    fn psd_clips_negative_eigenvalue() {
        // diag(-1, 1): projection should be diag(0, 1).
        let a = vec![-1.0, 0.0, 0.0, 1.0];
        let p = project_psd_cone(&a, 2).expect("ok");
        assert!(p[0].abs() < 1.0e-9);
        assert!((p[3] - 1.0).abs() < 1.0e-9);
        assert!(p[1].abs() < 1.0e-9);
        assert!(p[2].abs() < 1.0e-9);
    }

    #[test]
    fn psd_two_by_two_neg_def() {
        let a = vec![-2.0, 0.0, 0.0, -1.0];
        let p = project_psd_cone(&a, 2).expect("ok");
        for &pi in &p {
            assert!(pi.abs() < 1.0e-9);
        }
    }

    #[test]
    fn jacobi_eigen_orthonormal() {
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (vals, vecs) = sym_jacobi_eigen(&a, 2, 100, 1.0e-12).expect("ok");
        // Eigenvalues should be 1, 3.
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 1.0).abs() < 1.0e-9);
        assert!((sorted[1] - 3.0).abs() < 1.0e-9);
        // V^T V = I.
        let mut vtv = vec![0.0_f64; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0_f64;
                for k in 0..2 {
                    s += vecs[k * 2 + i] * vecs[k * 2 + j];
                }
                vtv[i * 2 + j] = s;
            }
        }
        assert!((vtv[0] - 1.0).abs() < 1.0e-9);
        assert!((vtv[3] - 1.0).abs() < 1.0e-9);
        assert!(vtv[1].abs() < 1.0e-9);
        assert!(vtv[2].abs() < 1.0e-9);
    }
}
