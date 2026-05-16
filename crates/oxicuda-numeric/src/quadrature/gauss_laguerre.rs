//! Gauss-Laguerre quadrature: weight `w(x) = x^α exp(-x)` on `[0, ∞)`.

use crate::error::{NumericError, NumericResult};
use crate::linalg::jacobi_eig::jacobi_eig_symmetric;

/// Compute Gauss-Laguerre nodes and weights of order `n` with weight `x^α exp(-x)` on `[0,∞)`.
pub fn gauss_laguerre_nodes(n: usize, alpha: f64) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(NumericError::InvalidParameter("n must be ≥ 1".into()));
    }
    if alpha <= -1.0 {
        return Err(NumericError::OutOfDomain {
            value: alpha,
            function: "gauss_laguerre (α > -1 required)".into(),
        });
    }
    let gamma_a_plus_1 = gamma_lanczos(alpha + 1.0);
    let mut t = vec![0.0_f64; n * n];
    for i in 0..n {
        t[i * n + i] = 2.0 * (i as f64) + alpha + 1.0;
    }
    for i in 0..(n - 1) {
        let k = i as f64 + 1.0;
        let b = (k * (k + alpha)).sqrt();
        t[i * n + (i + 1)] = b;
        t[(i + 1) * n + i] = b;
    }
    let (eigvals, eigvecs) = jacobi_eig_symmetric(&t, n, 200, 1.0e-14)?;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        eigvals[i]
            .partial_cmp(&eigvals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let nodes: Vec<f64> = idx.iter().map(|&i| eigvals[i]).collect();
    let weights: Vec<f64> = idx
        .iter()
        .map(|&i| gamma_a_plus_1 * eigvecs[i * n].powi(2))
        .collect();
    Ok((nodes, weights))
}

/// Lanczos approximation to Γ.
fn gamma_lanczos(x: f64) -> f64 {
    let p = [
        676.5203681218851_f64,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_lanczos(1.0 - x))
    } else {
        let x_shift = x - 1.0;
        let mut a = 0.999_999_999_999_809_93_f64;
        for (i, &pi) in p.iter().enumerate() {
            a += pi / (x_shift + (i as f64) + 1.0);
        }
        let t = x_shift + p.len() as f64 - 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x_shift + 0.5) * (-t).exp() * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_alpha0_normalization() {
        let (_nodes, weights) = gauss_laguerre_nodes(5, 0.0).expect("ok");
        let s: f64 = weights.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn gl_alpha0_first_moment() {
        let (nodes, weights) = gauss_laguerre_nodes(6, 0.0).expect("ok");
        let s: f64 = nodes.iter().zip(weights.iter()).map(|(x, w)| w * x).sum();
        assert!((s - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn gl_alpha0_second_moment() {
        let (nodes, weights) = gauss_laguerre_nodes(8, 0.0).expect("ok");
        let s: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| w * x * x)
            .sum();
        assert!((s - 2.0).abs() < 1.0e-8);
    }
}
