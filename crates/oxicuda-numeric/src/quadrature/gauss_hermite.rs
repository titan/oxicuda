//! Gauss-Hermite quadrature: weight `w(x) = exp(-x²)` on `(-∞, ∞)`.

use crate::error::{NumericError, NumericResult};
use crate::linalg::jacobi_eig::jacobi_eig_symmetric;

/// Compute Gauss-Hermite nodes and weights of order `n` for weight `exp(-x²)`.
pub fn gauss_hermite_nodes(n: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(NumericError::InvalidParameter("n must be ≥ 1".into()));
    }
    if n == 1 {
        return Ok((vec![0.0], vec![std::f64::consts::PI.sqrt()]));
    }
    let mut t = vec![0.0_f64; n * n];
    for i in 0..(n - 1) {
        let b = ((i as f64 + 1.0) / 2.0).sqrt();
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
    let weights: Vec<f64> = idx
        .iter()
        .map(|&i| std::f64::consts::PI.sqrt() * eigvecs[i * n].powi(2))
        .collect();
    Ok((nodes, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn gh_n5_normalization() {
        let (_nodes, weights) = gauss_hermite_nodes(5).expect("ok");
        let s: f64 = weights.iter().sum();
        assert!((s - PI.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn gh_n5_even_moment() {
        let (nodes, weights) = gauss_hermite_nodes(5).expect("ok");
        let s: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| w * x * x)
            .sum();
        assert!((s - PI.sqrt() / 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn gh_nodes_symmetric() {
        let (nodes, _) = gauss_hermite_nodes(7).expect("ok");
        let n = nodes.len();
        for i in 0..(n / 2) {
            assert!((nodes[i] + nodes[n - 1 - i]).abs() < 1.0e-10);
        }
    }
}
