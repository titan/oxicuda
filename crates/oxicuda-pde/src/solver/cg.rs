//! Classical Hestenes-Stiefel Conjugate Gradient on SPD systems.

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::{SparseCsr, dot, norm2};

/// Solve `A x = b` using CG. `x0` is the initial guess; returns `x`.
pub fn cg_solve(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>> {
    let n = a.n_rows;
    if a.n_cols != n {
        return Err(PdeError::DimensionMismatch {
            a: a.n_rows,
            b: a.n_cols,
        });
    }
    if b.len() != n || x0.len() != n {
        return Err(PdeError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = x0.to_vec();
    let ax = a.matvec(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r)?;
    let b_norm = norm2(b).max(1.0);
    if rs_old.sqrt() / b_norm < tol {
        return Ok(x);
    }
    for _it in 0..max_iter {
        let ap = a.matvec(&p)?;
        let pap = dot(&p, &ap)?;
        if pap.abs() < 1.0e-300 {
            return Err(PdeError::NumericalInstability(
                "cg: zero p·A·p (matrix may not be SPD)".into(),
            ));
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r)?;
        if rs_new.sqrt() / b_norm < tol {
            return Ok(x);
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    Err(PdeError::NotConverged {
        iter: max_iter,
        residual: rs_old.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_identity_system() {
        let a =
            SparseCsr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).expect("ok");
        let b = vec![1.0, 2.0, 3.0];
        let x = cg_solve(&a, &b, &[0.0; 3], 10, 1e-12).expect("ok");
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - b[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn cg_tridiagonal_system() {
        // -u'' = 1 on [0,1], n=5 interior nodes (h=1/6)
        // A = (1/h^2) * tridiag(-1, 2, -1)
        let h = 1.0 / 6.0;
        let inv_h2 = 1.0 / (h * h);
        let a = SparseCsr::new(
            5,
            5,
            vec![0, 2, 5, 8, 11, 13],
            vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
            vec![
                2.0 * inv_h2,
                -inv_h2,
                -inv_h2,
                2.0 * inv_h2,
                -inv_h2,
                -inv_h2,
                2.0 * inv_h2,
                -inv_h2,
                -inv_h2,
                2.0 * inv_h2,
                -inv_h2,
                -inv_h2,
                2.0 * inv_h2,
            ],
        )
        .expect("ok");
        let b = vec![1.0; 5];
        let x = cg_solve(&a, &b, &[0.0; 5], 100, 1e-12).expect("ok");
        // exact: u(x) = x(1-x)/2; evaluate at x_i = i*h, i=1..5
        for (i, &xi_solved) in x.iter().enumerate() {
            let xi = (i + 1) as f64 * h;
            let expected = xi * (1.0 - xi) / 2.0;
            assert!((xi_solved - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn cg_zero_rhs_yields_zero() {
        let a = SparseCsr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("ok");
        let x = cg_solve(&a, &[0.0, 0.0], &[0.0; 2], 10, 1e-12).expect("ok");
        assert!(x.iter().all(|&v| v.abs() < 1e-12));
    }
}
