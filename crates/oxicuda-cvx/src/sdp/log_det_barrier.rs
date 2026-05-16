//! `−log det X` barrier and its gradient `−X⁻¹` for PD `X`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::cholesky::cholesky_factor;
use crate::linalg::solve::solve_dense;

/// Compute `log det X` for SPD `X`.  Uses Cholesky: `log det = 2 Σ log L[i,i]`.
pub fn log_det(x: &[f64], n: usize) -> CvxResult<f64> {
    if x.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![x.len()],
        });
    }
    let l = cholesky_factor(x, n)?;
    let mut acc = 0.0_f64;
    for i in 0..n {
        let d = l[i * n + i];
        if d <= 0.0 {
            return Err(CvxError::NumericalInstability(
                "log_det: non-positive diagonal in Cholesky".into(),
            ));
        }
        acc += d.ln();
    }
    Ok(2.0 * acc)
}

/// Gradient of `−log det X` is `−X⁻¹` (computed via dense solve column-by-column).
pub fn log_det_gradient(x: &[f64], n: usize) -> CvxResult<Vec<f64>> {
    if x.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![x.len()],
        });
    }
    let mut out = vec![0.0_f64; n * n];
    for k in 0..n {
        let mut e_k = vec![0.0_f64; n];
        e_k[k] = 1.0;
        let col = solve_dense(x, n, &e_k)?;
        for i in 0..n {
            out[i * n + k] = -col[i];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_det_identity() {
        let id = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = log_det(&id, 3).expect("ok");
        assert!(v.abs() < 1.0e-12);
    }

    #[test]
    fn log_det_diag() {
        let d = vec![4.0_f64, 0.0, 0.0, 9.0];
        let v = log_det(&d, 2).expect("ok");
        // log 4 + log 9 = log 36.
        assert!((v - 36.0_f64.ln()).abs() < 1.0e-10);
    }

    #[test]
    fn log_det_gradient_identity() {
        let id = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let g = log_det_gradient(&id, 3).expect("ok");
        // grad of -log det I is -I.
        for i in 0..3 {
            for j in 0..3 {
                let exp = if i == j { -1.0 } else { 0.0 };
                assert!((g[i * 3 + j] - exp).abs() < 1.0e-9);
            }
        }
    }
}
