//! Gauss-Chebyshev T quadrature: weight `1/sqrt(1 - x²)` on `[-1, 1]`.

use crate::error::{NumericError, NumericResult};

/// Return Gauss-Chebyshev T nodes/weights of order `n`.
pub fn gauss_chebyshev_t(n: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(NumericError::InvalidParameter("n must be ≥ 1".into()));
    }
    let mut nodes = Vec::with_capacity(n);
    let w = std::f64::consts::PI / n as f64;
    let mut weights = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = (2 * k - 1) as f64 * std::f64::consts::PI / (2.0 * n as f64);
        nodes.push(theta.cos());
        weights.push(w);
    }
    Ok((nodes, weights))
}

/// Integrate `f` on `[-1, 1]` w.r.t. weight `1/sqrt(1-x²)` via n-point Gauss-Chebyshev.
pub fn gauss_chebyshev_integrate<F>(f: F, n: usize) -> NumericResult<f64>
where
    F: Fn(f64) -> NumericResult<f64>,
{
    let (nodes, weights) = gauss_chebyshev_t(n)?;
    let mut acc = 0.0_f64;
    for (x, w) in nodes.iter().zip(weights.iter()) {
        acc += w * f(*x)?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn gc_constant_pi() {
        let f = |_x: f64| -> NumericResult<f64> { Ok(1.0) };
        let r = gauss_chebyshev_integrate(f, 5).expect("ok");
        assert!((r - PI).abs() < 1.0e-12);
    }

    #[test]
    fn gc_x_squared() {
        let f = |x: f64| -> NumericResult<f64> { Ok(x * x) };
        let r = gauss_chebyshev_integrate(f, 5).expect("ok");
        assert!((r - PI / 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn gc_nodes_symmetric() {
        let (nodes, _) = gauss_chebyshev_t(8).expect("ok");
        for i in 0..(nodes.len() / 2) {
            assert!((nodes[i] + nodes[nodes.len() - 1 - i]).abs() < 1.0e-12);
        }
    }
}
