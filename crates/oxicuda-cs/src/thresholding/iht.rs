//! Iterative Hard Thresholding (Blumensath-Davies 2008).
//!
//! Update: `x ← H_K(x + μ Φᵀ (y − Φ x))` with constant step `μ`.

use crate::error::{CsError, CsResult};
use crate::linalg::{mat_t_vec, mat_vec, norm2};
use crate::thresholding::ThresholdingResult;

/// Hard-threshold a vector to its top-K entries by magnitude.
///
/// Returns the K-sparse vector (length `n`) plus the indices of the kept entries (sorted asc).
pub fn hard_threshold_k(x: &[f64], k: usize) -> CsResult<(Vec<f64>, Vec<usize>)> {
    if k > x.len() {
        return Err(CsError::SupportTooLarge {
            requested: k,
            max: x.len(),
        });
    }
    if k == 0 {
        return Ok((vec![0.0; x.len()], Vec::new()));
    }
    let mut paired: Vec<(usize, f64, f64)> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v, v.abs()))
        .collect();
    paired.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0_f64; x.len()];
    let mut support = Vec::with_capacity(k);
    for it in paired.iter().take(k) {
        out[it.0] = it.1;
        support.push(it.0);
    }
    support.sort();
    Ok((out, support))
}

/// Soft-threshold helper used by AMP / FISTA / IHT-LASSO and re-exported.
pub fn soft_threshold(x: &[f64], lambda: f64) -> Vec<f64> {
    x.iter()
        .map(|v| {
            let s = v.signum();
            let mag = (v.abs() - lambda).max(0.0);
            s * mag
        })
        .collect()
}

/// IHT with constant step size `mu`. Stops when `||x_new - x|| / ||x|| < tol` or after `max_iter`.
pub fn iht(
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
        return Err(CsError::InvalidParameter(format!(
            "step mu must be > 0; got {mu}"
        )));
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
        let (x_new, supp_new) = hard_threshold_k(&candidate, k)?;
        // Convergence test.
        let mut delta = 0.0_f64;
        for j in 0..n {
            let d = x_new[j] - x[j];
            delta += d * d;
        }
        x = x_new;
        support = supp_new;
        iter += 1;
        let xnorm = norm2(&x).max(1.0e-300);
        if delta.sqrt() / xnorm < tol {
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
    fn hard_threshold_top_k() {
        let (out, supp) = hard_threshold_k(&[3.0, 1.0, 4.0, 1.0, 5.0], 2).expect("ok");
        // top-2 by |.| are indices 4 (5) and 2 (4).
        assert_eq!(supp, vec![2, 4]);
        assert_eq!(out[2], 4.0);
        assert_eq!(out[4], 5.0);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn soft_threshold_doc_example() {
        let x = [2.0, 0.5, -0.5, -2.0];
        let p = soft_threshold(&x, 1.0);
        assert!((p[0] - 1.0).abs() < 1.0e-12);
        assert!(p[1].abs() < 1.0e-12);
        assert!(p[2].abs() < 1.0e-12);
        assert!((p[3] + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn iht_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = iht(&phi, 4, 4, &y, 2, 0.9, 200, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
    }
}
