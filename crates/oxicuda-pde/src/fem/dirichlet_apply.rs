//! Apply Dirichlet boundary conditions to a CSR system: row-and-column zeroing.

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::SparseCsr;

/// In-place apply Dirichlet BCs `(node_idx -> value)`:
/// 1. Modify RHS: `b[i] -= A[i, k] * val` for all `i`, then set `b[k] = val`
/// 2. Zero row `k` and column `k`; set `A[k,k] = 1`.
pub fn apply_dirichlet_csr(
    a: &mut SparseCsr,
    b: &mut [f64],
    bc_nodes: &[usize],
    bc_values: &[f64],
) -> PdeResult<()> {
    if bc_nodes.len() != bc_values.len() {
        return Err(PdeError::DimensionMismatch {
            a: bc_nodes.len(),
            b: bc_values.len(),
        });
    }
    if a.n_rows != b.len() {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: b.len(),
        });
    }
    // Build a quick lookup
    let mut is_bc = vec![false; a.n_rows];
    let mut bc_val_lookup = vec![0.0; a.n_rows];
    for (&k, &v) in bc_nodes.iter().zip(bc_values.iter()) {
        if k >= a.n_rows {
            return Err(PdeError::IndexOutOfBounds {
                index: k,
                len: a.n_rows,
            });
        }
        is_bc[k] = true;
        bc_val_lookup[k] = v;
    }
    // First update RHS: for any i with A[i, k] != 0 (and i not bc), b[i] -= A[i, k] * v_k
    for i in 0..a.n_rows {
        if is_bc[i] {
            continue;
        }
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        for k_idx in row_lo..row_hi {
            let j = a.cols[k_idx];
            if is_bc[j] {
                b[i] -= a.vals[k_idx] * bc_val_lookup[j];
            }
        }
    }
    // Zero columns belonging to BC nodes
    for i in 0..a.n_rows {
        if is_bc[i] {
            continue;
        }
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        for k_idx in row_lo..row_hi {
            let j = a.cols[k_idx];
            if is_bc[j] {
                a.vals[k_idx] = 0.0;
            }
        }
    }
    // For BC rows: zero them and set diagonal to 1, b[k] = v_k
    for &k in bc_nodes {
        let row_lo = a.row_ptr[k];
        let row_hi = a.row_ptr[k + 1];
        let mut found_diag = false;
        for k_idx in row_lo..row_hi {
            let j = a.cols[k_idx];
            if j == k {
                a.vals[k_idx] = 1.0;
                found_diag = true;
            } else {
                a.vals[k_idx] = 0.0;
            }
        }
        if !found_diag {
            return Err(PdeError::SingularMatrix(format!(
                "no diagonal entry for BC node {k}"
            )));
        }
        b[k] = bc_val_lookup[k];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_dirichlet_simple_2x2() {
        // A = [[2,-1],[-1,2]], b = [1,1]
        let mut a = SparseCsr::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, -1.0, -1.0, 2.0],
        )
        .expect("ok");
        let mut b = vec![1.0, 1.0];
        apply_dirichlet_csr(&mut a, &mut b, &[0], &[5.0]).expect("ok");
        // After: A becomes [[1,0],[0,2]], b becomes [5, 1 - (-1)*5 = 6]
        assert!((a.vals[0] - 1.0).abs() < 1.0e-12);
        // value at (1, 0) should be zero
        let row_lo = a.row_ptr[1];
        for k in row_lo..a.row_ptr[2] {
            if a.cols[k] == 0 {
                assert!(a.vals[k].abs() < 1.0e-12);
            }
        }
        assert!((b[0] - 5.0).abs() < 1.0e-12);
        assert!((b[1] - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn apply_dirichlet_dim_mismatch() {
        let mut a = SparseCsr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("ok");
        let mut b = vec![0.0, 0.0];
        let res = apply_dirichlet_csr(&mut a, &mut b, &[0], &[1.0, 2.0]);
        assert!(res.is_err());
    }
}
