//! IMEX (implicit-explicit) time stepping.
//!
//! For systems `du/dt = L u + N(u)` where `L` is stiff (linear) and `N(u)` is
//! nonstiff (nonlinear), the first-order IMEX scheme is:
//! `(I - dt*L) u^{n+1} = u^n + dt * N(u^n)`.

use crate::error::{PdeError, PdeResult};
use crate::solver::cg::cg_solve;
use crate::solver::sparse::SparseCsr;

/// One IMEX step.
pub fn imex_step<N>(
    l_matrix: &SparseCsr,
    u: &[f64],
    n_explicit: N,
    dt: f64,
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>>
where
    N: Fn(&[f64]) -> Vec<f64>,
{
    let n = l_matrix.n_rows;
    if u.len() != n {
        return Err(PdeError::DimensionMismatch { a: u.len(), b: n });
    }
    // LHS = I - dt * L
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        let row_lo = l_matrix.row_ptr[i];
        let row_hi = l_matrix.row_ptr[i + 1];
        let mut found_diag = false;
        for k in row_lo..row_hi {
            let j = l_matrix.cols[k];
            let mut v = -dt * l_matrix.vals[k];
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
    let lhs = SparseCsr::new(n, n, row_ptr, cols, vals)?;
    // RHS = u + dt * N(u)
    let n_vec = n_explicit(u);
    if n_vec.len() != n {
        return Err(PdeError::DimensionMismatch {
            a: n_vec.len(),
            b: n,
        });
    }
    let rhs: Vec<f64> = u.iter().zip(&n_vec).map(|(ui, ni)| ui + dt * ni).collect();
    cg_solve(&lhs, &rhs, u, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imex_pure_implicit_decay() {
        // du/dt = -u + 0, L = -I, N = 0
        let l = SparseCsr::new(1, 1, vec![0, 1], vec![0], vec![-1.0]).expect("ok");
        let mut u = vec![1.0];
        let dt = 0.1;
        let nsteps = 10;
        for _ in 0..nsteps {
            u = imex_step(&l, &u, |_| vec![0.0], dt, 50, 1e-12).expect("ok");
        }
        // First-order convergence: u(1) ~ (1/(1+dt))^10 = (1/1.1)^10 ≈ 0.386
        let analytic_be = (1.0_f64 / 1.1).powi(10);
        assert!((u[0] - analytic_be).abs() < 1e-9);
    }

    #[test]
    fn imex_combination_zero_implicit() {
        // du/dt = 0*u + 0.1 (constant explicit), L = 0
        let l = SparseCsr::new(1, 1, vec![0, 1], vec![0], vec![0.0]).expect("ok");
        let mut u = vec![0.0];
        let dt = 0.01;
        for _ in 0..100 {
            u = imex_step(&l, &u, |_| vec![0.1], dt, 50, 1e-12).expect("ok");
        }
        // Should be ~ 0.1 * 1.0 = 0.1
        assert!((u[0] - 0.1).abs() < 1e-9);
    }
}
