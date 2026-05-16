//! Stagewise Orthogonal Matching Pursuit (Donoho, Tsaig, Drori, Starck 2012).
//!
//! At each stage, all atoms with `|aⱼᵀ r| > t · σ̂` are added to the support, where
//! `σ̂ = median(|aⱼᵀ r|) / 0.6745`. The constant `0.6745` is the inverse-cdf of the
//! Gaussian distribution at the upper-quartile (MAD = 0.6745 σ for Gaussian noise).

use crate::error::{CsError, CsResult};
use crate::greedy::GreedyResult;
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};

/// StOMP: at each stage, pick atoms whose absolute correlation exceeds `t * σ̂`.
///
/// `threshold_factor` (the `t`) is usually in `[2, 3]` for noiseless / mild-noise problems.
pub fn stomp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    threshold_factor: f64,
    max_stages: usize,
    tol_residual: f64,
) -> CsResult<GreedyResult> {
    if phi.len() != m * n {
        return Err(CsError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![phi.len()],
        });
    }
    if y.len() != m {
        return Err(CsError::DimensionMismatch { a: y.len(), b: m });
    }
    if threshold_factor <= 0.0 {
        return Err(CsError::InvalidParameter(format!(
            "threshold_factor must be > 0; got {threshold_factor}"
        )));
    }
    if max_stages == 0 {
        return Err(CsError::InvalidParameter("max_stages = 0".into()));
    }
    let mut support: Vec<usize> = Vec::new();
    let mut residual = y.to_vec();
    let mut x_full = vec![0.0_f64; n];
    let mut iter = 0usize;
    for _ in 0..max_stages {
        let r_norm = norm2(&residual);
        if r_norm < tol_residual {
            break;
        }
        let corr = mat_t_vec(phi, m, n, &residual)?;
        let mut abs_corr: Vec<f64> = corr.iter().map(|c| c.abs()).collect();
        let mut sorted = abs_corr.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.is_empty() {
            0.0
        } else if sorted.len() % 2 == 1 {
            sorted[sorted.len() / 2]
        } else {
            0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
        };
        let sigma_hat = (median / 0.6745).max(1.0e-300);
        let thr = threshold_factor * sigma_hat;
        // Set already-in-support to 0 so we never re-select.
        for &j in &support {
            abs_corr[j] = 0.0;
        }
        let mut added_any = false;
        for (j, &c) in abs_corr.iter().enumerate() {
            if c > thr {
                support.push(j);
                added_any = true;
            }
        }
        if !added_any {
            break;
        }
        support.sort();
        support.dedup();
        if support.len() > m {
            support.truncate(m);
        }
        let x_sub = solve_subset_ls(phi, m, n, &support, y)?;
        x_full.fill(0.0);
        for (i, &j) in support.iter().enumerate() {
            x_full[j] = x_sub[i];
        }
        let sub = submat_columns(phi, m, n, &support)?;
        let ax = mat_vec(&sub, m, support.len(), &x_sub)?;
        for i in 0..m {
            residual[i] = y[i] - ax[i];
        }
        iter += 1;
    }
    Ok(GreedyResult {
        x: x_full,
        support,
        residual_norm: norm2(&residual),
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stomp_basic() {
        // Φ = I_4, y = e_1 + 0.5 e_3.
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = stomp(&phi, 4, 4, &y, 2.0, 10, 1.0e-6).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
    }
}
