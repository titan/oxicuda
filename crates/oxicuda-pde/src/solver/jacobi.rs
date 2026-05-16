//! Damped Jacobi smoother / solver.

use crate::error::{PdeError, PdeResult};
use crate::solver::sparse::{SparseCsr, norm2};

/// Damped Jacobi iteration: `x_{k+1} = x_k + omega * D^{-1} * (b - A x_k)`.
///
/// Returns final iterate and number of iterations performed.
pub fn jacobi_solve(
    a: &SparseCsr,
    b: &[f64],
    x0: &[f64],
    omega: f64,
    max_iter: usize,
    tol: f64,
) -> PdeResult<(Vec<f64>, usize, f64)> {
    let n = a.n_rows;
    if b.len() != n || x0.len() != n {
        return Err(PdeError::DimensionMismatch { a: b.len(), b: n });
    }
    let diag = a.diagonal()?;
    if diag.iter().any(|&d| d.abs() < 1.0e-300) {
        return Err(PdeError::SingularMatrix(
            "jacobi: zero diagonal entry".into(),
        ));
    }
    let mut x = x0.to_vec();
    let mut last_res = f64::INFINITY;
    for it in 0..max_iter {
        let ax = a.matvec(&x)?;
        let mut res: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
        let r_norm = norm2(&res);
        last_res = r_norm;
        if r_norm < tol {
            return Ok((x, it, r_norm));
        }
        for i in 0..n {
            res[i] /= diag[i];
            x[i] += omega * res[i];
        }
    }
    Ok((x, max_iter, last_res))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonal_system() {
        let a =
            SparseCsr::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 4.0]).expect("ok");
        let b = vec![4.0, 9.0, 16.0];
        let (x, _, _) = jacobi_solve(&a, &b, &[0.0; 3], 1.0, 100, 1e-12).expect("ok");
        assert!((x[0] - 2.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
        assert!((x[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn jacobi_tridiag_converges() {
        let a = SparseCsr::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
        )
        .expect("ok");
        let b = vec![1.0, 1.0, 1.0];
        let (x, iters, _res) = jacobi_solve(&a, &b, &[0.0; 3], 0.8, 5000, 1e-12).expect("ok");
        // residual A x - b should be near zero
        let ax = a.matvec(&x).expect("ok");
        let r: f64 = (0..3).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1e-10);
        assert!(iters > 0);
    }

    #[test]
    fn jacobi_singular_diag_errors() {
        let a = SparseCsr::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).expect("ok");
        let r = jacobi_solve(&a, &[0.0, 0.0], &[0.0, 0.0], 1.0, 10, 1.0e-12);
        assert!(r.is_err());
    }
}
