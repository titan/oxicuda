//! Hard Thresholding Pursuit (Foucart 2011).
//!
//! IHT identifies the support, then exact least-squares solve on that support.

use crate::error::{CsError, CsResult};
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};
use crate::thresholding::ThresholdingResult;
use crate::thresholding::iht::hard_threshold_k;

/// HTP with step `mu`. Stops on support stability or `max_iter`.
pub fn htp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    k: usize,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> CsResult<ThresholdingResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if k == 0 || k > n {
        return Err(CsError::InvalidSparsity(k));
    }
    if mu <= 0.0 {
        return Err(CsError::InvalidParameter("step mu must be > 0".into()));
    }
    let mut x = vec![0.0_f64; n];
    let mut support: Vec<usize> = Vec::new();
    let mut iter = 0usize;
    for _ in 0..max_iter {
        let ax = mat_vec(phi, m, n, &x)?;
        let mut residual = vec![0.0_f64; m];
        for i in 0..m {
            residual[i] = y[i] - ax[i];
        }
        let g = mat_t_vec(phi, m, n, &residual)?;
        let mut candidate = x.clone();
        for j in 0..n {
            candidate[j] += mu * g[j];
        }
        let (_, support_new) = hard_threshold_k(&candidate, k)?;
        // Exact LS on support_new.
        let x_sub = solve_subset_ls(phi, m, n, &support_new, y)?;
        let mut x_new = vec![0.0_f64; n];
        for (i, &j) in support_new.iter().enumerate() {
            x_new[j] = x_sub[i];
        }
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        let support_changed = support != support_new;
        x = x_new;
        support = support_new;
        iter += 1;
        if !support_changed && delta.sqrt() < tol {
            break;
        }
    }
    let sub = submat_columns(phi, m, n, &support)?;
    let x_sub: Vec<f64> = support.iter().map(|&j| x[j]).collect();
    let ax = mat_vec(&sub, m, support.len(), &x_sub)?;
    let mut residual = vec![0.0_f64; m];
    for i in 0..m {
        residual[i] = y[i] - ax[i];
    }
    Ok(ThresholdingResult {
        x,
        support,
        residual_norm: norm2(&residual),
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn htp_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = htp(&phi, 4, 4, &y, 2, 0.9, 50, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
        assert!((r.x[0] - 1.0).abs() < 1.0e-6);
    }
}
