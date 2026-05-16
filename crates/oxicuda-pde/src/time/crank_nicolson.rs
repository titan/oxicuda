//! Crank-Nicolson for linear systems: `(I - dt/2 A) u^{n+1} = (I + dt/2 A) u^n + dt b`.

use crate::error::{PdeError, PdeResult};
use crate::solver::cg::cg_solve;
use crate::solver::sparse::SparseCsr;

/// One Crank-Nicolson step for `du/dt = A u + b`.
pub fn crank_nicolson_step_linear(
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
    let half_dt = 0.5 * dt;
    // LHS: I - dt/2 A
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        let mut found_diag = false;
        for k in row_lo..row_hi {
            let j = a.cols[k];
            let mut v = -half_dt * a.vals[k];
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
    // RHS: (I + dt/2 A) u + dt b
    let mut rhs = vec![0.0_f64; n];
    for i in 0..n {
        rhs[i] = u[i] + dt * b[i];
        let row_lo = a.row_ptr[i];
        let row_hi = a.row_ptr[i + 1];
        for k in row_lo..row_hi {
            let j = a.cols[k];
            rhs[i] += half_dt * a.vals[k] * u[j];
        }
    }
    cg_solve(&lhs, &rhs, u, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crank_nicolson_linear_decay() {
        // du/dt = -u, exact u(dt) ~ (1 - dt/2)/(1 + dt/2) * u^n
        let a = SparseCsr::new(1, 1, vec![0, 1], vec![0], vec![-1.0]).expect("ok");
        let mut u = vec![1.0];
        let dt = 0.1;
        let nsteps = 10;
        for _ in 0..nsteps {
            u = crank_nicolson_step_linear(&a, &u, &[0.0], dt, 50, 1.0e-12).expect("ok");
        }
        let analytic = (-1.0_f64).exp();
        // Crank-Nicolson is O(dt^2)
        assert!(
            (u[0] - analytic).abs() < 5.0e-3,
            "u={} expected~{}",
            u[0],
            analytic
        );
    }
}
