//! Lanczos tridiagonalisation for smallest/largest eigenpair of a symmetric matrix.

use crate::error::{ManifoldError, ManifoldResult};
use crate::linalg::jacobi_eig::jacobi_eigh;

/// Result of a Lanczos run on the smallest eigenvalue.
pub struct LanczosResult {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
}

/// Lanczos iteration that returns the smallest eigenvalue & eigenvector of a symmetric
/// linear operator given as `apply: |v| -> A v`.
pub fn lanczos_smallest_eig<F>(
    apply: F,
    n: usize,
    v0: &[f64],
    k_steps: usize,
    tol: f64,
) -> ManifoldResult<LanczosResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if v0.len() != n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n],
            got: vec![v0.len()],
        });
    }
    let k = k_steps.min(n);
    if k < 1 {
        return Err(ManifoldError::InvalidParameter {
            name: "k_steps".into(),
            reason: "must be >= 1".into(),
        });
    }
    let mut v = v0.to_vec();
    let nrm0 = norm(&v);
    if nrm0 < 1e-300 {
        return Err(ManifoldError::NumericalInstability(
            "zero start vector".into(),
        ));
    }
    for vi in &mut v {
        *vi /= nrm0;
    }
    let mut basis: Vec<Vec<f64>> = vec![v.clone()];
    let mut alpha: Vec<f64> = Vec::with_capacity(k);
    let mut beta: Vec<f64> = Vec::with_capacity(k);
    let mut v_prev = vec![0.0; n];
    let mut prev_eig = f64::INFINITY;
    for j in 0..k {
        let w = apply(&basis[j]);
        let mut w2: Vec<f64> = w
            .iter()
            .zip(&v_prev)
            .map(|(wi, pi)| wi - beta.last().copied().unwrap_or(0.0) * pi)
            .collect();
        let aj = dot(&basis[j], &w2);
        alpha.push(aj);
        for (wi, bi) in w2.iter_mut().zip(&basis[j]) {
            *wi -= aj * bi;
        }
        // Full reorthogonalisation (modified Gram-Schmidt) for stability
        for prev in &basis {
            let proj = dot(&w2, prev);
            for (wi, pi) in w2.iter_mut().zip(prev) {
                *wi -= proj * pi;
            }
        }
        let bj = norm(&w2);
        beta.push(bj);
        v_prev = basis[j].clone();
        if bj < 1e-12 || j + 1 == k {
            // Solve the small tridiagonal eigenproblem now.
            let m = alpha.len();
            let mut t = vec![0.0; m * m];
            for i in 0..m {
                t[i * m + i] = alpha[i];
                if i + 1 < m {
                    t[i * m + (i + 1)] = beta[i];
                    t[(i + 1) * m + i] = beta[i];
                }
            }
            let (mut w, v_mat) = jacobi_eigh(&t, m)?;
            // smallest eigenvalue index
            let mut idx_min = 0;
            for (i, val) in w.iter().enumerate() {
                if *val < w[idx_min] {
                    idx_min = i;
                }
            }
            let lam = w[idx_min];
            // Build eigenvector in the original space: V_basis * z
            let mut eigv = vec![0.0; n];
            for (r, b_vec) in basis.iter().enumerate() {
                let z = v_mat[r * m + idx_min];
                for i in 0..n {
                    eigv[i] += z * b_vec[i];
                }
            }
            // normalise
            let nrm = norm(&eigv);
            if nrm > 1e-300 {
                for ei in &mut eigv {
                    *ei /= nrm;
                }
            }
            // Mutate w to silence unused mut warning when m == 0 path
            w.clear();
            return Ok(LanczosResult {
                eigenvalue: lam,
                eigenvector: eigv,
            });
        }
        for wi in &mut w2 {
            *wi /= bj;
        }
        basis.push(w2);
        // Convergence check via current Ritz value (smallest)
        let m = alpha.len();
        let mut t = vec![0.0; m * m];
        for i in 0..m {
            t[i * m + i] = alpha[i];
            if i + 1 < m {
                t[i * m + (i + 1)] = beta[i];
                t[(i + 1) * m + i] = beta[i];
            }
        }
        let (w_eigs, _) = jacobi_eigh(&t, m)?;
        let cur = w_eigs.iter().copied().fold(f64::INFINITY, f64::min);
        if (cur - prev_eig).abs() < tol {
            // Re-run conversion path
            let (mut wf, v_mat) = jacobi_eigh(&t, m)?;
            let mut idx_min = 0;
            for (i, val) in wf.iter().enumerate() {
                if *val < wf[idx_min] {
                    idx_min = i;
                }
            }
            let lam = wf[idx_min];
            let mut eigv = vec![0.0; n];
            for (r, b_vec) in basis.iter().enumerate() {
                if r >= m {
                    break;
                }
                let z = v_mat[r * m + idx_min];
                for i in 0..n {
                    eigv[i] += z * b_vec[i];
                }
            }
            let nrm = norm(&eigv);
            if nrm > 1e-300 {
                for ei in &mut eigv {
                    *ei /= nrm;
                }
            }
            wf.clear();
            return Ok(LanczosResult {
                eigenvalue: lam,
                eigenvector: eigv,
            });
        }
        prev_eig = cur;
    }
    // Fallback: should never get here
    Err(ManifoldError::NotConverged { iter: k })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanczos_smallest_diag() {
        let n = 5;
        let diag: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let apply =
            |v: &[f64]| -> Vec<f64> { v.iter().enumerate().map(|(i, vi)| diag[i] * vi).collect() };
        let v0 = vec![1.0; n];
        let r = lanczos_smallest_eig(apply, n, &v0, n, 1e-12).expect("ok");
        assert!((r.eigenvalue - 1.0).abs() < 1e-7);
    }
}
