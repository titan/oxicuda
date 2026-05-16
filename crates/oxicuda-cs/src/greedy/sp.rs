//! Subspace Pursuit (Dai-Milenkovic 2009).
//!
//! Similar to CoSaMP but identifies K (not 2K) largest entries in the proxy at each step.

use crate::error::{CsError, CsResult};
use crate::greedy::GreedyResult;
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};

/// Subspace Pursuit recovery with target sparsity `k`.
pub fn subspace_pursuit(
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
    // Initialise support: top-K of |Φᵀ y|.
    let init_corr = mat_t_vec(phi, m, n, y)?;
    let mut abs_idx: Vec<(usize, f64)> = init_corr
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    abs_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut support: Vec<usize> = abs_idx.iter().take(k).map(|(i, _)| *i).collect();
    support.sort();
    let mut x_sub = solve_subset_ls(phi, m, n, &support, y)?;
    let mut residual = y.to_vec();
    {
        let sub = submat_columns(phi, m, n, &support)?;
        let ax = mat_vec(&sub, m, support.len(), &x_sub)?;
        for i in 0..m {
            residual[i] -= ax[i];
        }
    }
    let mut prev_r = norm2(&residual);
    let mut x_full = vec![0.0_f64; n];
    let mut iter = 0usize;
    for _ in 0..max_iter {
        if norm2(&residual) < tol_residual {
            break;
        }
        // Proxy: top-K of |Φᵀ r|.
        let proxy = mat_t_vec(phi, m, n, &residual)?;
        let mut p_idx: Vec<(usize, f64)> = proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        p_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let omega: Vec<usize> = p_idx.into_iter().take(k).map(|(i, _)| i).collect();
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
        let b_sub = solve_subset_ls(phi, m, n, &t, y)?;
        // Prune to top-K of |b_sub|.
        let mut bi: Vec<(usize, f64)> = b_sub
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        bi.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        bi.truncate(k);
        let mut new_support: Vec<usize> = bi.iter().map(|(i, _)| t[*i]).collect();
        new_support.sort();
        let new_x_sub = solve_subset_ls(phi, m, n, &new_support, y)?;
        x_sub = new_x_sub;
        let sub = submat_columns(phi, m, n, &new_support)?;
        let ax = mat_vec(&sub, m, new_support.len(), &x_sub)?;
        for i in 0..m {
            residual[i] = y[i] - ax[i];
        }
        x_full.fill(0.0);
        for (i, &j) in new_support.iter().enumerate() {
            x_full[j] = x_sub[i];
        }
        support = new_support;
        let cur_r = norm2(&residual);
        if cur_r >= prev_r * (1.0 - 1.0e-12) && iter > 0 {
            break;
        }
        prev_r = cur_r;
        iter += 1;
    }
    if x_full.iter().all(|v| *v == 0.0) {
        for (i, &j) in support.iter().enumerate() {
            x_full[j] = x_sub[i];
        }
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
    fn sp_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.5, 0.0];
        let r = subspace_pursuit(&phi, 4, 4, &y, 2, 20, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
        assert!(r.support.contains(&2));
    }
}
