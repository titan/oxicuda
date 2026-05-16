//! Accelerated IHT with Nesterov momentum (Blumensath 2012; momentum variant).

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::ThresholdingResult;
use crate::thresholding::iht::hard_threshold_k;

/// Accelerated IHT with Nesterov-style momentum.
pub fn aiht(
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
    let mut x = vec![0.0_f64; n];
    let mut x_prev = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    let mut support: Vec<usize> = Vec::new();
    let mut t = 1.0_f64;
    let mut iter = 0usize;
    for _ in 0..max_iter {
        let az = mat_vec(phi, m, n, &z)?;
        let mut residual = vec![0.0_f64; m];
        for i in 0..m {
            residual[i] = y[i] - az[i];
        }
        let g = mat_t_vec(phi, m, n, &residual)?;
        let mut candidate = z.clone();
        for j in 0..n {
            candidate[j] += mu * g[j];
        }
        let (x_new, supp_new) = hard_threshold_k(&candidate, k)?;
        // Nesterov momentum update.
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let beta = (t - 1.0) / t_new;
        for j in 0..n {
            z[j] = x_new[j] + beta * (x_new[j] - x[j]);
        }
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        x_prev.clone_from(&x);
        x = x_new;
        support = supp_new;
        t = t_new;
        iter += 1;
        if delta.sqrt() / norm2(&x).max(1.0e-300) < tol {
            break;
        }
    }
    let ax = mat_vec(phi, m, n, &x)?;
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
    fn aiht_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = aiht(&phi, 4, 4, &y, 2, 0.9, 300, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
    }
}
