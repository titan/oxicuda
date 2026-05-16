//! Tensor-product Gauss-Legendre cubature over an axis-aligned `d`-dimensional box.
//!
//! `n^d` nodes — only suitable for low dimension.

use crate::error::{NumericError, NumericResult};
use crate::quadrature::gauss_legendre::gauss_legendre_nodes;

/// Tensor-product Gauss-Legendre quadrature with `n_per_axis` points per dimension.
pub fn tensor_product_gauss_integrate<F>(
    f: F,
    lo: &[f64],
    hi: &[f64],
    n_per_axis: usize,
) -> NumericResult<f64>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    if lo.len() != hi.len() {
        return Err(NumericError::DimensionMismatch {
            a: lo.len(),
            b: hi.len(),
        });
    }
    if lo.is_empty() {
        return Err(NumericError::EmptyInput);
    }
    let d = lo.len();
    let (nodes, weights) = gauss_legendre_nodes(n_per_axis)?;
    // index combinations in n_per_axis^d
    let total = (n_per_axis as u64).pow(d as u32);
    let mut acc = 0.0_f64;
    let mut x = vec![0.0_f64; d];
    let mut idx = vec![0_usize; d];
    for _t in 0..total {
        let mut w_prod = 1.0_f64;
        for k in 0..d {
            let mid = 0.5 * (hi[k] + lo[k]);
            let half = 0.5 * (hi[k] - lo[k]);
            x[k] = mid + half * nodes[idx[k]];
            w_prod *= half * weights[idx[k]];
        }
        acc += w_prod * f(&x)?;
        // increment counter
        let mut k = 0;
        while k < d {
            idx[k] += 1;
            if idx[k] < n_per_axis {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp_constant() {
        let f = |_x: &[f64]| -> NumericResult<f64> { Ok(1.0) };
        let v = tensor_product_gauss_integrate(f, &[0.0, 0.0], &[2.0, 3.0], 4).expect("ok");
        assert!((v - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn tp_polynomial_2d_exact() {
        // ∫_0^1 ∫_0^1 x⁵ y⁵ dx dy = 1/36 (Gauss-Legendre n=4 integrates degree 7 exact in each dim)
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0].powi(5) * x[1].powi(5)) };
        let v = tensor_product_gauss_integrate(f, &[0.0, 0.0], &[1.0, 1.0], 4).expect("ok");
        assert!((v - 1.0 / 36.0).abs() < 1.0e-10);
    }
}
