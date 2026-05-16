//! Normalised IHT: adaptive step `μ = ||g_S||² / ||Φ_S g_S||²` based on the current support.

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::ThresholdingResult;
use crate::thresholding::iht::hard_threshold_k;

/// Normalised IHT with adaptive step size.
pub fn niht(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    k: usize,
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
    let mut support: Vec<usize> = Vec::new();
    let mut iter = 0usize;
    for _ in 0..max_iter {
        let ax = mat_vec(phi, m, n, &x)?;
        let mut residual = vec![0.0_f64; m];
        for i in 0..m {
            residual[i] = y[i] - ax[i];
        }
        let g = mat_t_vec(phi, m, n, &residual)?;
        // If we have a support, restrict g to support to compute step.
        let mu = if !support.is_empty() {
            let mut g_s_full = vec![0.0_f64; n];
            for &j in &support {
                g_s_full[j] = g[j];
            }
            let phi_g = mat_vec(phi, m, n, &g_s_full)?;
            let num: f64 = support.iter().map(|&j| g[j] * g[j]).sum();
            let den: f64 = phi_g.iter().map(|v| v * v).sum::<f64>().max(1.0e-30);
            num / den
        } else {
            // Initial step: ||g||² / ||Φ g||²
            let phi_g = mat_vec(phi, m, n, &g)?;
            let num: f64 = g.iter().map(|v| v * v).sum();
            let den: f64 = phi_g.iter().map(|v| v * v).sum::<f64>().max(1.0e-30);
            num / den
        };
        let mut candidate = x.clone();
        for j in 0..n {
            candidate[j] += mu * g[j];
        }
        let (x_new, supp_new) = hard_threshold_k(&candidate, k)?;
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        x = x_new;
        support = supp_new;
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
    fn niht_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = niht(&phi, 4, 4, &y, 2, 100, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
    }
}
