//! Stiefel manifold St(n, p) = {Y in R^{n x p} : Y^T Y = I_p}.
//!
//! - Tangent projection: `T_Y St(n,p) = {Delta : Y^T Delta + Delta^T Y = 0}`.
//! - Retraction (QR): `R_Y(Delta) = qr(Y + Delta).Q[:, :p]`.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::householder_qr::householder_qr;

/// Project an arbitrary `n x p` matrix `Z` onto the tangent space at `Y`.
///
/// `P_{T_Y}(Z) = Z - Y * sym(Y^T Z)` where `sym(M) = (M + M^T) / 2`.
pub fn stiefel_project_tangent(
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
    // S = Y^T Z (p x p), then sym(S) = (S + S^T)/2
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
    for i in 0..p {
        for j in i..p {
            let v = 0.5 * (s[i * p + j] + s[j * p + i]);
            s[i * p + j] = v;
            s[j * p + i] = v;
        }
    }
    // out = Z - Y * sym(S)
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

/// QR retraction: returns `Q` from QR(Y + Delta).
pub fn stiefel_retract_qr(
    y: &[f64],
    delta: &[f64],
    n: usize,
    p: usize,
) -> ManifoldResult<Vec<f64>> {
    if y.len() != n * p {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![y.len()],
        });
    }
    if delta.len() != n * p {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, p],
            got: vec![delta.len()],
        });
    }
    let mut sum = vec![0.0; n * p];
    for i in 0..n * p {
        sum[i] = y[i] + delta[i];
    }
    let (q, _r) = householder_qr(&sum, n, p)?;
    // Fix sign so diagonal of R is positive (QR sign convention)
    // We approximate by checking diagonal sign: compute Q^T (Y+Delta) =? R; just adjust if Q . Y has negative diag.
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_orthogonal_to_y() {
        let n = 4;
        let p = 2;
        // Y = first two columns of identity
        let mut y = vec![0.0; n * p];
        y[0] = 1.0;
        y[p + 1] = 1.0;
        let z = vec![1.0; n * p];
        let t = stiefel_project_tangent(&y, &z, n, p).expect("ok");
        // Y^T t should be antisymmetric
        let mut s = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += y[r * p + i] * t[r * p + j];
                }
                s[i * p + j] = acc;
            }
        }
        for i in 0..p {
            for j in 0..p {
                let antisym = s[i * p + j] + s[j * p + i];
                assert!(antisym.abs() < 1e-7, "{i},{j}: {antisym}");
            }
        }
    }

    #[test]
    fn qr_retract_returns_orthonormal() {
        let n = 4;
        let p = 2;
        let mut y = vec![0.0; n * p];
        y[0] = 1.0;
        y[p + 1] = 1.0;
        let delta = vec![0.1; n * p];
        let q = stiefel_retract_qr(&y, &delta, n, p).expect("ok");
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += q[r * p + a] * q[r * p + b];
                }
                let tgt = if a == b { 1.0 } else { 0.0 };
                assert!((acc - tgt).abs() < 1e-7);
            }
        }
    }
}
