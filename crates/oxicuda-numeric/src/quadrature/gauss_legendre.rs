//! Gauss-Legendre quadrature via Golub-Welsch eigenvalue computation.
//!
//! For the orthonormal Legendre polynomials, the Jacobi matrix is symmetric tridiagonal with
//! `T[i, i+1] = i / sqrt(4i² - 1)`. The nodes are the eigenvalues; the weights are
//! `w_i = 2 * v_{i,0}²` where `v` are eigenvectors.

use crate::error::{NumericError, NumericResult};
use crate::linalg::jacobi_eig::jacobi_eig_symmetric;

/// Compute Gauss-Legendre nodes and weights of order `n` on `[-1, 1]`.
pub fn gauss_legendre_nodes(n: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(NumericError::InvalidParameter("n must be ≥ 1".into()));
    }
    if n == 1 {
        return Ok((vec![0.0], vec![2.0]));
    }
    let mut t = vec![0.0_f64; n * n];
    for i in 0..(n - 1) {
        let b = (i as f64 + 1.0) / ((4.0 * (i as f64 + 1.0).powi(2) - 1.0).sqrt());
        t[i * n + (i + 1)] = b;
        t[(i + 1) * n + i] = b;
    }
    let (eigvals, eigvecs) = jacobi_eig_symmetric(&t, n, 100, 1.0e-14)?;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        eigvals[i]
            .partial_cmp(&eigvals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let nodes: Vec<f64> = idx.iter().map(|&i| eigvals[i]).collect();
    let weights: Vec<f64> = idx.iter().map(|&i| 2.0 * eigvecs[i * n].powi(2)).collect();
    Ok((nodes, weights))
}

/// Integrate `f` over `[a, b]` using `n`-point Gauss-Legendre.
pub fn gauss_legendre_integrate<F>(f: F, a: f64, b: f64, n: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    if !a.is_finite() || !b.is_finite() {
        return Err(NumericError::InvalidParameter("non-finite limits".into()));
    }
    let (nodes, weights) = gauss_legendre_nodes(n)?;
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut acc = 0.0_f64;
    for (xi, wi) in nodes.iter().zip(weights.iter()) {
        acc += wi * f(mid + half * xi)?;
    }
    Ok(half * acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_n1_constant() {
        let f = |_x: f64| -> NumericResult<f64> { Ok(1.0) };
        let r = gauss_legendre_integrate(f, -1.0, 1.0, 1).expect("ok");
        assert!((r - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn gl_n5_polynomial_degree_9_exact() {
        // n nodes integrate degree 2n-1 = 9 exactly. ∫_{-1}^1 x⁹ dx = 0
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(9)) };
        let r = gauss_legendre_integrate(f, -1.0, 1.0, 5).expect("ok");
        assert!(r.abs() < 1.0e-12);
    }

    #[test]
    fn gl_n5_polynomial_x8_exact() {
        // ∫_{-1}^1 x⁸ dx = 2/9
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(8)) };
        let r = gauss_legendre_integrate(f, -1.0, 1.0, 5).expect("ok");
        assert!((r - 2.0 / 9.0).abs() < 1.0e-12);
    }

    #[test]
    fn gl_n6_exp() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.exp()) };
        let r = gauss_legendre_integrate(f, 0.0, 1.0, 6).expect("ok");
        assert!((r - (std::f64::consts::E - 1.0)).abs() < 1.0e-10);
    }
}
