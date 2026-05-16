//! Clenshaw-Curtis quadrature on `[-1, 1]` via the Chebyshev cosine transform.

use crate::error::{NumericError, NumericResult};

/// Compute Clenshaw-Curtis nodes and weights of order `n` (n+1 points) on `[-1, 1]`.
pub fn clenshaw_curtis_nodes(n: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if n < 2 {
        return Err(NumericError::InvalidParameter("n must be ≥ 2".into()));
    }
    let nn = n;
    let n_f = nn as f64;
    let mut nodes = vec![0.0_f64; nn + 1];
    for (k, node) in nodes.iter_mut().enumerate() {
        *node = (k as f64 * std::f64::consts::PI / n_f).cos();
    }
    let mut weights = vec![0.0_f64; nn + 1];
    for (k, w_slot) in weights.iter_mut().enumerate() {
        let theta = k as f64 * std::f64::consts::PI / n_f;
        let mut s = 0.0_f64;
        let jmax = nn / 2;
        for j in 1..=jmax {
            let bj = if 2 * j < nn { 2.0 } else { 1.0 };
            s += bj * (2.0 * j as f64 * theta).cos() / ((4 * j * j - 1) as f64);
        }
        let c_k = if k == 0 || k == nn { 1.0 } else { 2.0 };
        *w_slot = c_k * (1.0 - s) / n_f;
    }
    Ok((nodes, weights))
}

/// Integrate `f` over `[a, b]` via Clenshaw-Curtis with `n+1` nodes.
pub fn clenshaw_curtis<F>(f: F, a: f64, b: f64, n: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    let (nodes, weights) = clenshaw_curtis_nodes(n)?;
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut acc = 0.0_f64;
    for (x, w) in nodes.iter().zip(weights.iter()) {
        acc += w * f(mid + half * x)?;
    }
    Ok(half * acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn cc_weights_sum() {
        let (_, w) = clenshaw_curtis_nodes(8).expect("ok");
        let s: f64 = w.iter().sum();
        assert!((s - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn cc_arctan() {
        let f = |x: f64| -> NumericResult<f64> { Ok(1.0 / (1.0 + x * x)) };
        let r = clenshaw_curtis(f, 0.0, 1.0, 20).expect("ok");
        assert!((r - PI / 4.0).abs() < 1.0e-10);
    }

    #[test]
    fn cc_polynomial_quintic() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x.powi(5)) };
        let r = clenshaw_curtis(f, 0.0, 1.0, 6).expect("ok");
        assert!((r - 1.0 / 6.0).abs() < 1.0e-10);
    }
}
