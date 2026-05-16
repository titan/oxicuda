//! BDF2 multi-step scheme: `(3 u^{n+1} - 4 u^n + u^{n-1}) / (2 dt) = F(u^{n+1})`.
//!
//! For linear F(u) = A u + b, this becomes:
//! `(3 I - 2 dt A) u^{n+1} = 4 u^n - u^{n-1} + 2 dt b`.

use crate::error::{PdeError, PdeResult};
use crate::solver::cg::cg_solve;
use crate::solver::sparse::SparseCsr;

/// One BDF2 step for linear `du/dt = A u + b`. Requires previous two states.
pub fn bdf2_step_linear(
    a: &SparseCsr,
    u_prev: &[f64],
    u_curr: &[f64],
    b: &[f64],
    dt: f64,
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>> {
    let n = a.n_rows;
    if u_prev.len() != n || u_curr.len() != n || b.len() != n {
        return Err(PdeError::DimensionMismatch {
            a: u_curr.len(),
            b: n,
        });
    }
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
            let mut v = -2.0 * dt * a.vals[k];
            if i == j {
                v += 3.0;
                found_diag = true;
            }
            cols.push(j);
            vals.push(v);
        }
        if !found_diag {
            cols.push(i);
            vals.push(3.0);
        }
        row_ptr.push(cols.len());
    }
    let lhs = SparseCsr::new(n, n, row_ptr, cols, vals)?;
    let rhs: Vec<f64> = u_curr
        .iter()
        .zip(u_prev)
        .zip(b)
        .map(|((uc, up), bi)| 4.0 * uc - up + 2.0 * dt * bi)
        .collect();
    cg_solve(&lhs, &rhs, u_curr, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf2_linear_decay() {
        // du/dt = -u; initialise with one BE step from u(0)=1.
        let a = SparseCsr::new(1, 1, vec![0, 1], vec![0], vec![-1.0]).expect("ok");
        let dt = 0.05;
        // u_prev = 1.0
        // u_curr from BE: u_curr = u_prev/(1+dt) = 1/1.05
        let u_prev = 1.0_f64;
        let u_curr = u_prev / (1.0 + dt);
        let mut up = vec![u_prev];
        let mut uc = vec![u_curr];
        let t_final = 0.5;
        let nsteps = (t_final / dt).round() as usize - 1;
        for _ in 0..nsteps {
            let u_next = bdf2_step_linear(&a, &up, &uc, &[0.0], dt, 50, 1e-12).expect("ok");
            up = uc;
            uc = u_next;
        }
        let analytic = (-t_final).exp();
        assert!(
            (uc[0] - analytic).abs() < 5.0e-3,
            "uc={} analytic={}",
            uc[0],
            analytic
        );
    }
}
