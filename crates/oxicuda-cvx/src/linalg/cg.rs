//! Conjugate Gradient on dense SPD systems.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{dot, mat_vec, norm2};

/// Hestenes-Stiefel CG.  Solves `A x = b` with SPD `A` (dense row-major).
/// Returns `x` of length n.
pub fn cg_solve(
    a: &[f64],
    n: usize,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if b.len() != n || x0.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = x0.to_vec();
    let ax0 = mat_vec(a, n, n, &x)?;
    let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax0[i]).collect();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r)?;
    let b_norm = norm2(b).max(1.0);
    if rs_old.sqrt() / b_norm < tol {
        return Ok(x);
    }
    for _ in 0..max_iter {
        let ap = mat_vec(a, n, n, &p)?;
        let pap = dot(&p, &ap)?;
        if pap.abs() < 1.0e-300 {
            return Err(CvxError::NumericalInstability(
                "cg: zero p·A·p (A may not be SPD)".into(),
            ));
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r)?;
        if rs_new.sqrt() / b_norm < tol {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    Err(CvxError::NotConverged {
        iter: max_iter,
        residual: rs_old.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_identity_solve() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 4.0, 5.0];
        let x = cg_solve(&a, 3, &b, &[0.0; 3], 50, 1.0e-12).expect("ok");
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1.0e-10);
        }
    }

    #[test]
    fn cg_spd_solve() {
        // A = [[4, 1], [1, 3]]; b = [1, 2]; x ≈ [1/11, 7/11].
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = cg_solve(&a, 2, &b, &[0.0, 0.0], 50, 1.0e-12).expect("ok");
        assert!((x[0] - 1.0 / 11.0).abs() < 1.0e-9);
        assert!((x[1] - 7.0 / 11.0).abs() < 1.0e-9);
    }
}
