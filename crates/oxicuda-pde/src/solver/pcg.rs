//! Preconditioned Conjugate Gradient with Jacobi / ILU(0) / SSOR preconditioners.

use crate::error::{PdeError, PdeResult};
use crate::solver::ilu0::{ilu0_factor, ilu0_solve};
use crate::solver::sparse::{SparseCsr, dot, norm2};
use crate::solver::ssor::ssor_apply;

fn pcg_inner<P>(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    apply_precond: P,
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>>
where
    P: Fn(&[f64]) -> PdeResult<Vec<f64>>,
{
    let n = a.n_rows;
    if b.len() != n || x0.len() != n {
        return Err(PdeError::DimensionMismatch { a: b.len(), b: n });
    }
    let mut x = x0.to_vec();
    let ax = a.matvec(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let mut z = apply_precond(&r)?;
    let mut p = z.clone();
    let mut rs_old = dot(&r, &z)?;
    let b_norm = norm2(b).max(1.0);
    if norm2(&r) / b_norm < tol {
        return Ok(x);
    }
    for _it in 0..max_iter {
        let ap = a.matvec(&p)?;
        let pap = dot(&p, &ap)?;
        if pap.abs() < 1.0e-300 {
            return Err(PdeError::NumericalInstability("pcg: zero p·A·p".into()));
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if norm2(&r) / b_norm < tol {
            return Ok(x);
        }
        z = apply_precond(&r)?;
        let rs_new = dot(&r, &z)?;
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    Err(PdeError::NotConverged {
        iter: max_iter,
        residual: norm2(&r),
    })
}

/// PCG with Jacobi (diagonal) preconditioner.
pub fn pcg_jacobi(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>> {
    let diag = a.diagonal()?;
    for &d in &diag {
        if d.abs() < 1.0e-300 {
            return Err(PdeError::SingularMatrix("pcg_jacobi: zero diag".into()));
        }
    }
    pcg_inner(
        a,
        b,
        x0,
        |r| Ok(r.iter().enumerate().map(|(i, ri)| ri / diag[i]).collect()),
        max_iter,
        tol,
    )
}

/// PCG with ILU(0) preconditioner.
pub fn pcg_ilu0(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>> {
    let ilu = ilu0_factor(a)?;
    pcg_inner(a, b, x0, |r| ilu0_solve(&ilu, r), max_iter, tol)
}

/// PCG with SSOR preconditioner.
pub fn pcg_ssor(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    omega: f64,
    max_iter: usize,
    tol: f64,
) -> PdeResult<Vec<f64>> {
    pcg_inner(a, b, x0, |r| ssor_apply(a, r, omega), max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_jacobi_tridiag() {
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
        )
        .expect("ok");
        let b = vec![1.0, 2.0, 3.0];
        let x = pcg_jacobi(&a, &b, &[0.0; 3], 100, 1e-12).expect("ok");
        let ax = a.matvec(&x).expect("ok");
        let r: f64 = (0..3).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1e-10);
    }

    #[test]
    fn pcg_ilu0_tridiag() {
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
        )
        .expect("ok");
        let b = vec![1.0, 2.0, 3.0];
        let x = pcg_ilu0(&a, &b, &[0.0; 3], 50, 1e-12).expect("ok");
        let ax = a.matvec(&x).expect("ok");
        let r: f64 = (0..3).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1e-10);
    }

    #[test]
    fn pcg_ssor_tridiag() {
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
        )
        .expect("ok");
        let b = vec![1.0, 2.0, 3.0];
        let x = pcg_ssor(&a, &b, &[0.0; 3], 1.2, 100, 1e-12).expect("ok");
        let ax = a.matvec(&x).expect("ok");
        let r: f64 = (0..3).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1e-10);
    }
}
