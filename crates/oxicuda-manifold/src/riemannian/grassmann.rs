//! Grassmann manifold Gr(n, p) — p-dimensional subspaces of R^n.
//!
//! Tangent at `Y` (n x p, columns orthonormal): `T_Y Gr(n,p) = {Delta : Y^T Delta = 0}`.
//! Retraction via QR. Geodesic / distance via principal angles (SVD), here we use the
//! cheap form involving `acos` of singular values of `Y_a^T Y_b`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::householder_qr::householder_qr;
use crate::linalg::jacobi_eig::{jacobi_eigh, sort_eigen_descending};

/// Tangent projection at `Y`: `P_{T_Y}(Z) = Z - Y (Y^T Z)`.
pub fn grassmann_project_tangent(
    y: &[f64],
    z: &[f64],
    n: usize,
    p: usize,
) -> ManifoldResult<Vec<f64>> {
    if y.len() != n * p {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![y.len()],
        });
    }
    if z.len() != n * p {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![z.len()],
        });
    }
    let mut s = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for r in 0..n {
                acc += y[r * p + i] * z[r * p + j];
            }
            s[i * p + j] = acc;
        }
    }
    let mut out = z.to_vec();
    for r in 0..n {
        for c in 0..p {
            let mut acc = 0.0;
            for k in 0..p {
                acc += y[r * p + k] * s[k * p + c];
            }
            out[r * p + c] -= acc;
        }
    }
    Ok(out)
}

/// QR retraction (same as Stiefel since Grassmann = Stiefel / O(p)).
pub fn grassmann_retract(y: &[f64], delta: &[f64], n: usize, p: usize) -> ManifoldResult<Vec<f64>> {
    if y.len() != n * p || delta.len() != n * p {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![y.len()],
        });
    }
    let mut sum = vec![0.0; n * p];
    for i in 0..n * p {
        sum[i] = y[i] + delta[i];
    }
    let (q, _r) = householder_qr(&sum, n, p)?;
    Ok(q)
}

/// Grassmann distance between two subspaces represented by orthonormal bases.
///
/// `d(Y_a, Y_b) = sqrt(sum_k theta_k^2)` where `theta_k = acos(sigma_k)` and
/// `sigma_k` are the singular values of `Y_a^T Y_b`.
pub fn grassmann_distance(a: &[f64], b: &[f64], n: usize, p: usize) -> ManifoldResult<f64> {
    if a.len() != n * p || b.len() != n * p {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![a.len()],
        });
    }
    // M = A^T B
    let mut m = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for r in 0..n {
                acc += a[r * p + i] * b[r * p + j];
            }
            m[i * p + j] = acc;
        }
    }
    // Use eigendecomposition of M M^T to compute squared singular values
    let mut mmt = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0;
            for r in 0..p {
                acc += m[i * p + r] * m[j * p + r];
            }
            mmt[i * p + j] = acc;
        }
    }
    let (mut w, mut v) = jacobi_eigh(&mmt, p)?;
    sort_eigen_descending(&mut w, &mut v, p);
    let mut s = 0.0;
    for sigma2 in w {
        let sigma = sigma2.clamp(0.0, 1.0).sqrt();
        let theta = sigma.acos();
        s += theta * theta;
    }
    Ok(s.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_orthogonal_to_y() {
        let n = 4;
        let p = 2;
        let mut y = vec![0.0; n * p];
        y[0] = 1.0;
        y[p + 1] = 1.0;
        let z = vec![1.0; n * p];
        let t = grassmann_project_tangent(&y, &z, n, p).expect("ok");
        for i in 0..p {
            for j in 0..p {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += y[r * p + i] * t[r * p + j];
                }
                assert!(acc.abs() < 1e-7);
            }
        }
    }

    #[test]
    fn distance_same_subspace_zero() {
        let n = 3;
        let p = 2;
        let mut a = vec![0.0; n * p];
        a[0] = 1.0;
        a[p + 1] = 1.0;
        let d = grassmann_distance(&a, &a, n, p).expect("ok");
        assert!(d.abs() < 1e-6);
    }
}
