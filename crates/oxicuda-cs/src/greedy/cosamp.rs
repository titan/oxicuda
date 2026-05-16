//! CoSaMP — Compressive Sampling Matching Pursuit (Needell-Tropp 2009).
//!
//! Steps each iteration:
//! 1. Compute proxy `u = Φᵀ r`.
//! 2. Identify Ω = top-2K indices of |u|.
//! 3. Merge T = Ω ∪ support.
//! 4. Solve LS on T to get b.
//! 5. Prune support to top-K of |b|.
//! 6. Update residual r = y - Φ x.

use crate::error::{CsError, CsResult};
use crate::greedy::GreedyResult;
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};

/// CoSaMP with target sparsity `k`, capped at `max_iter` and residual tolerance.
pub fn cosamp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    k: usize,
    max_iter: usize,
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
    if k == 0 || k > m.min(n) {
        return Err(CsError::InvalidSparsity(k));
    }
    if max_iter == 0 {
        return Err(CsError::InvalidParameter("max_iter = 0".into()));
    }
    let mut support: Vec<usize> = Vec::new();
    let mut residual = y.to_vec();
    let mut x_full = vec![0.0_f64; n];
    let mut iter = 0usize;
    let mut prev_r = f64::INFINITY;
    for _ in 0..max_iter {
        let r_norm = norm2(&residual);
        if r_norm < tol_residual {
            break;
        }
        if r_norm >= prev_r * (1.0 - 1.0e-10) && iter > 0 {
            break;
        }
        prev_r = r_norm;
        let proxy = mat_t_vec(phi, m, n, &residual)?;
        // top-2k indices of |proxy|.
        let take_2k = (2 * k).min(n);
        let mut abs_idx: Vec<(usize, f64)> = proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        abs_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let omega: Vec<usize> = abs_idx.into_iter().take(take_2k).map(|(i, _)| i).collect();
        // Merge with support.
        let mut t: Vec<usize> = support.clone();
        for &o in &omega {
            if !t.contains(&o) {
                t.push(o);
            }
        }
        t.sort();
        if t.len() > m {
            t.truncate(m);
        }
        // LS on T.
        let b_sub = solve_subset_ls(phi, m, n, &t, y)?;
        // Prune to top-k.
        let mut bi: Vec<(usize, f64, f64)> = b_sub
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v, v.abs()))
            .collect();
        bi.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        bi.truncate(k);
        // Build new support, x_full.
        let mut new_support: Vec<usize> = bi.iter().map(|(i, _, _)| t[*i]).collect();
        new_support.sort();
        x_full.fill(0.0);
        for &(i_sub, v, _) in &bi {
            x_full[t[i_sub]] = v;
        }
        // Update residual via Φ_T b_sub mapped to support
        let sub = submat_columns(phi, m, n, &new_support)?;
        let x_new: Vec<f64> = new_support.iter().map(|&j| x_full[j]).collect();
        let ax = mat_vec(&sub, m, new_support.len(), &x_new)?;
        for i in 0..m {
            residual[i] = y[i] - ax[i];
        }
        support = new_support;
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
    fn cosamp_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = cosamp(&phi, 4, 4, &y, 2, 20, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
        assert!((r.x[0] - 1.0).abs() < 1.0e-6);
        assert!((r.x[2] - 0.5).abs() < 1.0e-6);
    }
}
