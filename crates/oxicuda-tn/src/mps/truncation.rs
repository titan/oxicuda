//! Bond truncation via singular value decomposition.

use crate::svd::{SvdResult, svd_jacobi};
use crate::{TnError, TnResult};

/// Truncate an SVD result to at most `chi_max` modes and drop any singular values below
/// `tol * max(s)`.
///
/// Returns the truncated `(U, S, V^T)` plus the L2 norm of the dropped singular values
/// (i.e. the squared truncation error). Singular values are assumed to be in descending
/// order on input.
pub fn svd_truncate(svd: SvdResult, chi_max: usize, tol: f64) -> TnResult<(SvdResult, f64)> {
    if svd.s.is_empty() {
        return Ok((svd, 0.0));
    }
    let s_max = svd.s[0];
    let abs_tol = tol * s_max.max(1.0);
    let mut keep = 0usize;
    for &v in &svd.s {
        if keep >= chi_max {
            break;
        }
        if v < abs_tol {
            break;
        }
        keep += 1;
    }
    keep = keep.max(1);
    let drop_norm: f64 = svd.s[keep..].iter().map(|x| x * x).sum::<f64>().sqrt();
    let s = svd.s[..keep].to_vec();
    // u_new: m × keep, copying first `keep` columns
    let mut u = vec![0.0; svd.m * keep];
    for i in 0..svd.m {
        for j in 0..keep {
            u[i * keep + j] = svd.u[i * svd.k + j];
        }
    }
    // vt_new: keep × n, copying first `keep` rows
    let mut vt = vec![0.0; keep * svd.n];
    for j in 0..keep {
        for r in 0..svd.n {
            vt[j * svd.n + r] = svd.vt[j * svd.n + r];
        }
    }
    Ok((
        SvdResult {
            u,
            s,
            vt,
            m: svd.m,
            n: svd.n,
            k: keep,
        },
        drop_norm,
    ))
}

/// Output of [`bond_truncate`]: the new left and right MPS-site data, the retained
/// singular values, and the resulting bond dimension.
#[derive(Debug, Clone)]
pub struct BondTruncationResult {
    pub left: Vec<f64>,
    pub right: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub bond_dim: usize,
}

/// Truncate the bond between two MPS site tensors.
///
/// Combines the two sites `M[a, p1, b]` and `N[b, p2, c]` into a 4-leg tensor
/// `T[a, p1, p2, c]`, reshapes to a matrix `(a*p1) × (p2*c)`, performs SVD, truncates
/// to `chi_max`, and writes back two tensors with new bond dimension.
#[allow(clippy::too_many_arguments)]
pub fn bond_truncate(
    left: &[f64],
    d_l: usize,
    d_p1: usize,
    d_b: usize,
    right: &[f64],
    d_p2: usize,
    d_r: usize,
    chi_max: usize,
    tol: f64,
) -> TnResult<BondTruncationResult> {
    if left.len() != d_l * d_p1 * d_b {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_l, d_p1, d_b],
            got: vec![left.len()],
        });
    }
    if right.len() != d_b * d_p2 * d_r {
        return Err(TnError::ShapeMismatch {
            expected: vec![d_b, d_p2, d_r],
            got: vec![right.len()],
        });
    }
    // Combine: theta[a, p1, p2, c] = sum_b left[a, p1, b] * right[b, p2, c]
    let m = d_l * d_p1;
    let n = d_p2 * d_r;
    let mut theta = vec![0.0; m * n];
    for a in 0..d_l {
        for p1 in 0..d_p1 {
            for p2 in 0..d_p2 {
                for c in 0..d_r {
                    let mut acc = 0.0;
                    for b in 0..d_b {
                        let lv = left[(a * d_p1 + p1) * d_b + b];
                        let rv = right[(b * d_p2 + p2) * d_r + c];
                        acc += lv * rv;
                    }
                    theta[((a * d_p1 + p1) * d_p2 + p2) * d_r + c] = acc;
                }
            }
        }
    }
    let svd = svd_jacobi(&theta, m, n)?;
    let (svd, _err) = svd_truncate(svd, chi_max, tol)?;
    let k = svd.k;
    let m_new = svd.u;
    let mut n_new = vec![0.0; k * n];
    for i in 0..k {
        let sv = svd.s[i];
        for j in 0..n {
            n_new[i * n + j] = sv * svd.vt[i * n + j];
        }
    }
    Ok(BondTruncationResult {
        left: m_new,
        right: n_new,
        singular_values: svd.s,
        bond_dim: k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svd::svd_jacobi;

    #[test]
    fn truncate_drops_smallest() {
        // diag(3, 2, 1) → keep 2 → s = [3, 2]
        let mat = vec![3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        let svd = svd_jacobi(&mat, 3, 3).expect("ok");
        let (trunc, err) = svd_truncate(svd, 2, 1e-12).expect("ok");
        assert_eq!(trunc.k, 2);
        assert!((trunc.s[0] - 3.0).abs() < 1e-10);
        assert!((trunc.s[1] - 2.0).abs() < 1e-10);
        assert!((err - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bond_truncate_smoke() {
        // 2x2x3 left, 3x2x2 right; chi_max=2.
        let left = vec![0.0; 2 * 2 * 3];
        let right = vec![0.0; 3 * 2 * 2];
        let r = bond_truncate(&left, 2, 2, 3, &right, 2, 2, 2, 1e-12).expect("ok");
        assert_eq!(r.bond_dim, 1); // all zero → only one trivial singular value retained
        assert_eq!(r.left.len(), 2 * 2 * r.bond_dim);
        assert_eq!(r.right.len(), r.bond_dim * 2 * 2);
        assert_eq!(r.singular_values.len(), r.bond_dim);
    }
}
