//! Orthogonal Matching Pursuit (Pati-Rezaiifar-Krishnaprasad 1993).

use crate::error::{CsError, CsResult};
use crate::greedy::GreedyResult;
use crate::linalg::normal_equations::solve_subset_ls;
use crate::linalg::{mat_t_vec, mat_vec, norm2, submat_columns};

/// Run OMP for `m × n` sensing matrix `phi`, observation `y`, target sparsity `k`.
///
/// Returns the recovered K-sparse coefficient vector of length `n`, plus support and residual norm.
/// Stops when either the support reaches size K **or** residual norm < `tol_residual`.
pub fn omp(
    phi: &[f64],
    m: usize,
    n: usize,
    y: &[f64],
    k: usize,
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
    let mut support: Vec<usize> = Vec::with_capacity(k);
    let mut residual = y.to_vec();
    let mut x_full = vec![0.0_f64; n];
    let mut iter = 0usize;
    for _ in 0..k {
        let r_norm = norm2(&residual);
        if r_norm < tol_residual {
            break;
        }
        let corr = mat_t_vec(phi, m, n, &residual)?;
        // Pick j maximising |corr[j]| not already in support.
        let mut best_idx = usize::MAX;
        let mut best_val = -1.0_f64;
        for (j, &c) in corr.iter().enumerate() {
            if support.contains(&j) {
                continue;
            }
            let abs_c = c.abs();
            if abs_c > best_val {
                best_val = abs_c;
                best_idx = j;
            }
        }
        if best_idx == usize::MAX {
            return Err(CsError::RecoveryFailed(
                "OMP failed to find a non-included column".into(),
            ));
        }
        support.push(best_idx);
        support.sort();
        let x_sub = solve_subset_ls(phi, m, n, &support, y)?;
        // Reset x_full and write subset.
        x_full.fill(0.0);
        for (i, &j) in support.iter().enumerate() {
            x_full[j] = x_sub[i];
        }
        // Update residual.
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
    fn omp_identity_recovers_canonical() {
        // Φ = I_4, y = e_1 -> x = e_1, support={0}.
        let phi = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![1.0, 0.0, 0.0, 0.0];
        let r = omp(&phi, 4, 4, &y, 1, 1.0e-9).expect("ok");
        assert_eq!(r.support, vec![0]);
        assert!((r.x[0] - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn omp_picks_largest() {
        // Φ = [[1,2],[3,4]], y = column 2 = [2, 4]. Should pick atom 1.
        let phi = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0];
        let r = omp(&phi, 2, 2, &y, 1, 1.0e-9).expect("ok");
        assert_eq!(r.support, vec![1]);
    }
}
