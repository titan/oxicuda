//! Power iteration for dominant eigenpair extraction.

use crate::error::{ManifoldError, ManifoldResult};

/// Plain power iteration on a row-major `n x n` matrix.
///
/// Returns `(eigenvalue, eigenvector)` for the largest-magnitude eigenvalue.
pub fn power_iteration(
    a: &[f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<(f64, Vec<f64>)> {
    if n == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if a.len() != n * n {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    let mut v = vec![0.0; n];
    // Deterministic init: standard basis e_0 + small spread to avoid degenerate cases.
    for (i, vi) in v.iter_mut().enumerate().take(n) {
        *vi = 1.0 / ((i + 1) as f64).sqrt();
    }
    // Normalize
    let nrm = norm(&v);
    if nrm < 1e-300 {
        return Err(ManifoldError::NumericalInstability("zero vector".into()));
    }
    for vi in &mut v {
        *vi /= nrm;
    }
    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let w = matvec(a, &v, n);
        let new_lambda = dot(&v, &w);
        let nrm_w = norm(&w);
        if nrm_w < 1e-300 {
            return Err(ManifoldError::NumericalInstability(
                "power iteration collapsed to zero".into(),
            ));
        }
        let new_v: Vec<f64> = w.iter().map(|wi| wi / nrm_w).collect();
        if (new_lambda - lambda).abs() < tol && lambda.abs() > 1e-300 {
            return Ok((new_lambda, new_v));
        }
        v = new_v;
        lambda = new_lambda;
    }
    Ok((lambda, v))
}

/// Deflated power iteration: extract `k` leading eigenpairs by Hotelling deflation.
pub fn power_iteration_deflated(
    a: &[f64],
    n: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<(Vec<f64>, Vec<f64>)> {
    if k > n {
        return Err(ManifoldError::InvalidParameter {
            name: "k".into(),
            reason: format!("requested k={k} eigenpairs but n={n}"),
        });
    }
    let mut work = a.to_vec();
    let mut eigvals = Vec::with_capacity(k);
    let mut eigvecs = vec![0.0; n * k];
    for j in 0..k {
        let (lam, v) = power_iteration(&work, n, max_iter, tol)?;
        eigvals.push(lam);
        for r in 0..n {
            eigvecs[r * k + j] = v[r];
        }
        // Deflate: A <- A - lam * v v^T
        for r in 0..n {
            for c in 0..n {
                work[r * n + c] -= lam * v[r] * v[c];
            }
        }
    }
    Ok((eigvals, eigvecs))
}

fn matvec(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[i * n + j] * v[j];
        }
        out[i] = acc;
    }
    out
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
    fn power_iter_diag() {
        // diag(3, 2, 1) - largest is 3
        let n = 3;
        let mut a = vec![0.0; n * n];
        a[0] = 3.0;
        a[n + 1] = 2.0;
        a[2 * n + 2] = 1.0;
        let (lam, v) = power_iteration(&a, n, 200, 1e-12).expect("ok");
        assert!((lam - 3.0).abs() < 1e-8);
        assert!((v[0].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn power_iter_deflated_diag() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        a[0] = 4.0;
        a[n + 1] = 3.0;
        a[2 * n + 2] = 2.0;
        a[3 * n + 3] = 1.0;
        let (vals, _vecs) = power_iteration_deflated(&a, n, 3, 300, 1e-12).expect("ok");
        assert!((vals[0] - 4.0).abs() < 1e-6);
        assert!((vals[1] - 3.0).abs() < 1e-6);
        assert!((vals[2] - 2.0).abs() < 1e-6);
    }
}
