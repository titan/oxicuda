//! Barycentric Lagrange interpolation.
//!
//! Precompute weights `w_j = 1 / Π_{i ≠ j} (x_j - x_i)`. Then
//! `p(x) = (Σ_j w_j y_j / (x - x_j)) / (Σ_j w_j / (x - x_j))`.

use crate::error::{NumericError, NumericResult};

/// Precompute barycentric weights for the nodes `xs`.
pub fn barycentric_weights(xs: &[f64]) -> NumericResult<Vec<f64>> {
    if xs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let n = xs.len();
    let mut w = vec![1.0_f64; n];
    for j in 0..n {
        for (i, &xi) in xs.iter().enumerate() {
            if i == j {
                continue;
            }
            let denom = xs[j] - xi;
            if denom.abs() < 1.0e-300 {
                return Err(NumericError::NumericalInstability(format!(
                    "duplicate node at index {j},{i}"
                )));
            }
            w[j] /= denom;
        }
    }
    Ok(w)
}

/// Evaluate the barycentric Lagrange polynomial at `x` given precomputed weights.
pub fn barycentric_eval(xs: &[f64], ys: &[f64], ws: &[f64], x: f64) -> NumericResult<f64> {
    if xs.len() != ys.len() || xs.len() != ws.len() {
        return Err(NumericError::DimensionMismatch {
            a: xs.len(),
            b: ys.len(),
        });
    }
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for ((xi, yi), wi) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        if (x - xi).abs() < 1.0e-300 {
            return Ok(*yi);
        }
        let t = wi / (x - xi);
        num += t * yi;
        den += t;
    }
    if den.abs() < 1.0e-300 {
        return Err(NumericError::NumericalInstability(
            "denominator vanishes in barycentric formula".into(),
        ));
    }
    Ok(num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bary_passes_through_nodes() {
        let xs = vec![-1.0_f64, 0.0, 1.0];
        let ys = vec![1.0_f64, 0.0, 1.0];
        let ws = barycentric_weights(&xs).expect("ok");
        for (x, y) in xs.iter().zip(ys.iter()) {
            let v = barycentric_eval(&xs, &ys, &ws, *x).expect("ok");
            assert!((v - y).abs() < 1.0e-10);
        }
    }

    #[test]
    fn bary_equals_lagrange_simple() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let ys = vec![0.0_f64, 1.0, 4.0];
        let ws = barycentric_weights(&xs).expect("ok");
        let v = barycentric_eval(&xs, &ys, &ws, 1.5).expect("ok");
        assert!((v - 2.25).abs() < 1.0e-10);
    }
}
