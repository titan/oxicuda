//! Regularised Orthogonal Matching Pursuit (Needell-Vershynin 2009).
//!
//! At each iteration:
//! 1. Compute `u = Φᵀ r`.
//! 2. Identify the top-`k` largest entries by magnitude as candidates `J`.
//! 3. Among the "regularised" subgroup `J_r ⊆ J` such that `max |u_j| ≤ 2 min |u_j|`,
//!    pick the subset with maximum total `||u_{J_r}||₂`.
//! 4. Append `J_r` to support, solve LS, update residual.

use crate::error::{CsError, CsResult};
use crate::greedy::GreedyResult;
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};

/// Regularised OMP. Stops when support reaches `k` or residual norm < `tol_residual`.
pub fn romp(
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
    while iter < max_iter && support.len() < k {
        let r_norm = norm2(&residual);
        if r_norm < tol_residual {
            break;
        }
        let mut corr = mat_t_vec(phi, m, n, &residual)?;
        for &j in &support {
            corr[j] = 0.0;
        }
        let mut abs_idx: Vec<(usize, f64)> = corr
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        abs_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f64)> = abs_idx.into_iter().take(k).collect();
        if top.is_empty() || top[0].1 < 1.0e-300 {
            break;
        }
        // Regularised subgroup: contiguous window in descending order with max/min ratio ≤ 2.
        // Pick the window with maximum sum of squares.
        let mut best_start = 0usize;
        let mut best_end = 0usize;
        let mut best_score = -1.0_f64;
        for s in 0..top.len() {
            let mut e = s;
            let mut score = 0.0_f64;
            while e < top.len() && top[s].1 <= 2.0 * top[e].1 {
                score += top[e].1 * top[e].1;
                e += 1;
            }
            if score > best_score {
                best_score = score;
                best_start = s;
                best_end = e;
            }
        }
        if best_score <= 0.0 {
            break;
        }
        for it in top.iter().take(best_end).skip(best_start) {
            if !support.contains(&it.0) {
                support.push(it.0);
            }
            if support.len() >= k {
                break;
            }
        }
        support.sort();
        support.dedup();
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
    fn romp_recovers_canonical() {
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.0, 0.0];
        let r = romp(&phi, 4, 4, &y, 1, 10, 1.0e-9).expect("ok");
        assert!(r.support.contains(&0));
    }
}
