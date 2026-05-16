//! ILU(0) factorisation on a CSR matrix (no extra fill-in).
//!
//! The factorisation stores L and U in-place within the original sparsity pattern of A.
//! L has implicit unit diagonal; U has explicit diagonal.

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::SparseCsr;

/// Result of ILU(0) factorisation: same sparsity pattern as A.
#[derive(Debug, Clone)]
pub struct Ilu0 {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
    pub diag_idx: Vec<usize>,
}

/// Compute ILU(0) factor of `A` (in-place sparsity pattern; no fill).
pub fn ilu0_factor(a: &SparseCsr) -> PdeResult<Ilu0> {
    let n = a.n_rows;
    if a.n_cols != n {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    let row_ptr = a.row_ptr.clone();
    let cols = a.cols.clone();
    let mut vals = a.vals.clone();
    let mut diag_idx = vec![usize::MAX; n];
    for (i, di) in diag_idx.iter_mut().enumerate().take(n) {
        let lo = row_ptr[i];
        let hi = row_ptr[i + 1];
        for (k, &c) in cols.iter().enumerate().take(hi).skip(lo) {
            if c == i {
                *di = k;
                break;
            }
        }
        if *di == usize::MAX {
            return Err(PdeError::SingularMatrix(format!(
                "ilu0: row {i} has no diagonal entry"
            )));
        }
    }
    // For each row i, for each k_idx in [lo..hi] with cols[k_idx] < i (lower):
    //   factor = vals[k_idx] / U[k, k] where k = cols[k_idx]
    //   vals[k_idx] = factor
    //   for each m_idx in [lo..hi] with cols[m_idx] > k:
    //     subtract factor * U[k, cols[m_idx]] from vals[m_idx]
    for i in 0..n {
        let lo = row_ptr[i];
        let hi = row_ptr[i + 1];
        for k_idx in lo..hi {
            let k = cols[k_idx];
            if k >= i {
                break;
            }
            let dkk = vals[diag_idx[k]];
            if dkk.abs() < 1.0e-300 {
                return Err(PdeError::SingularMatrix(format!(
                    "ilu0: zero diagonal at row {k}"
                )));
            }
            let factor = vals[k_idx] / dkk;
            vals[k_idx] = factor;
            // Update remaining row entries
            // U[k, j] is the entries of row k with col > k. We look for them in row k.
            let k_lo = row_ptr[k];
            let k_hi = row_ptr[k + 1];
            for k_row_idx in k_lo..k_hi {
                let j = cols[k_row_idx];
                if j <= k {
                    continue;
                }
                // Does row i have column j?
                for m_idx in k_idx + 1..hi {
                    if cols[m_idx] == j {
                        vals[m_idx] -= factor * vals[k_row_idx];
                        break;
                    }
                }
            }
        }
    }
    Ok(Ilu0 {
        n,
        row_ptr,
        cols,
        vals,
        diag_idx,
    })
}

/// Solve `LU z = r` using the ILU(0) factorisation.
pub fn ilu0_solve(ilu: &Ilu0, r: &[f64]) -> PdeResult<Vec<f64>> {
    let n = ilu.n;
    if r.len() != n {
        return Err(PdeError::DimensionMismatch { a: r.len(), b: n });
    }
    // Forward solve: L y = r (L has implicit unit diag)
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let lo = ilu.row_ptr[i];
        let hi = ilu.row_ptr[i + 1];
        let mut s = r[i];
        for k_idx in lo..hi {
            let j = ilu.cols[k_idx];
            if j < i {
                s -= ilu.vals[k_idx] * y[j];
            } else {
                break;
            }
        }
        y[i] = s;
    }
    // Backward solve: U z = y
    let mut z = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let lo = ilu.row_ptr[i];
        let hi = ilu.row_ptr[i + 1];
        let mut s = y[i];
        for k_idx in lo..hi {
            let j = ilu.cols[k_idx];
            if j > i {
                s -= ilu.vals[k_idx] * z[j];
            }
        }
        let dii = ilu.vals[ilu.diag_idx[i]];
        if dii.abs() < 1.0e-300 {
            return Err(PdeError::SingularMatrix("ilu0_solve: zero diag".into()));
        }
        z[i] = s / dii;
    }
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilu0_diagonal_identity() {
        let a =
            SparseCsr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).expect("ok");
        let ilu = ilu0_factor(&a).expect("ok");
        let z = ilu0_solve(&ilu, &[3.0, 5.0, 7.0]).expect("ok");
        assert!((z[0] - 3.0).abs() < 1.0e-12);
        assert!((z[1] - 5.0).abs() < 1.0e-12);
        assert!((z[2] - 7.0).abs() < 1.0e-12);
    }

    #[test]
    fn ilu0_tridiag_solves_exactly() {
        // [[2,-1,0],[-1,2,-1],[0,-1,2]] x = [1,0,1] => x = [1,1,1]
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
        .expect("ok");
        let ilu = ilu0_factor(&a).expect("ok");
        let z = ilu0_solve(&ilu, &[1.0, 0.0, 1.0]).expect("ok");
        // For a tridiag, ILU(0) is exact.
        for v in &z {
            assert!((v - 1.0).abs() < 1.0e-10);
        }
    }
}
