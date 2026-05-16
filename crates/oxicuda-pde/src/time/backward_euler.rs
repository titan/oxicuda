//! Backward (implicit) Euler for linear systems: `(I - dt*A) u^{n+1} = u^n + dt*b`.
//!
//! For a system `du/dt = A u + b` with sparse `A`, we use a CG/PCG-style iterative solve.

use crate::error::{PdeError, PdeResult};
use crate::solver::cg::cg_solve;
use crate::solver::sparse::SparseCsr;

/// One backward-Euler step for `du/dt = A u + b` with sparse `A`.
///
/// Solves `(I - dt*A) u^{n+1} = u^n + dt*b`. Returns updated `u^{n+1}`.
pub fn backward_euler_solve_linear(
    a: &SparseCsr,
    u: &[f64],
    b: &[f64],
    dt: f64,
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>> {
    let n = a.n_rows;
    if u.len() != n || b.len() != n {
        return Err(PdeError::DimensionMismatch { a: u.len(), b: n });
    }
    // Build M = I - dt*A
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        // Collect entries: -dt*A[i,j], plus +1 on the diagonal
        let mut found_diag = false;
        for k in row_lo..row_hi {
            let j = a.cols[k];
            let mut v = -dt * a.vals[k];
            if i == j {
                v += 1.0;
                found_diag = true;
            }
            cols.push(j);
            vals.push(v);
        }
        if !found_diag {
            cols.push(i);
            vals.push(1.0);
        }
        row_ptr.push(cols.len());
    }
    let m = SparseCsr::new(n, n, row_ptr, cols, vals)?;
    let rhs: Vec<f64> = u.iter().zip(b).map(|(ui, bi)| ui + dt * bi).collect();
    cg_solve(&m, &rhs, u, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_euler_linear_decay() {
        // du/dt = -u, dt=0.5, after 4 steps u = (1/1.5)^4 (sequence)
        let a = SparseCsr::new(1, 1, vec![0, 1], vec![0], vec![-1.0]).expect("ok");
        let mut u = vec![1.0];
        let dt = 0.5;
        for _ in 0..4 {
            u = backward_euler_solve_linear(&a, &u, &[0.0], dt, 100, 1.0e-12).expect("ok");
        }
        let expected = (1.0 / 1.5_f64).powi(4);
        assert!(
            (u[0] - expected).abs() < 1.0e-9,
            "u={} expected={}",
            u[0],
            expected
        );
    }
}
