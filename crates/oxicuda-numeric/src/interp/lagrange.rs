//! Lagrange interpolation in classical form (O(n²) per evaluation).

use crate::error::{NumericError, NumericResult};

/// Evaluate the Lagrange interpolating polynomial through `(xs, ys)` at `x`.
pub fn lagrange_interpolate(xs: &[f64], ys: &[f64], x: f64) -> NumericResult<f64> {
    if xs.len() != ys.len() {
        return Err(NumericError::DimensionMismatch {
            a: xs.len(),
            b: ys.len(),
        });
    }
    if xs.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let n = xs.len();
    let mut sum = 0.0_f64;
    for i in 0..n {
        let mut basis = 1.0_f64;
        for j in 0..n {
            if i == j {
                continue;
            }
            let denom = xs[i] - xs[j];
            if denom.abs() < 1.0e-300 {
                return Err(NumericError::NumericalInstability(format!(
                    "duplicate node at index {i},{j}"
                )));
            }
            basis *= (x - xs[j]) / denom;
        }
        sum += ys[i] * basis;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_quadratic() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let ys = vec![0.0_f64, 1.0, 4.0]; // y = x²
        let v = lagrange_interpolate(&xs, &ys, 1.5).expect("ok");
        assert!((v - 2.25).abs() < 1.0e-12);
    }

    #[test]
    fn lagrange_passes_through_nodes() {
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
        let ys = vec![1.0_f64, 2.0, 5.0, 10.0];
        for (x, y) in xs.iter().zip(ys.iter()) {
            let v = lagrange_interpolate(&xs, &ys, *x).expect("ok");
            assert!((v - y).abs() < 1.0e-10);
        }
    }
}
