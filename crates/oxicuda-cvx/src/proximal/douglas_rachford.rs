//! Douglas-Rachford splitting for `min f(x) + g(x)`.
//!
//! Updates:
//!   y_k = prox_{γf}(x_k)
//!   z_k = prox_{γg}(2 y_k − x_k)
//!   x_{k+1} = x_k + z_k − y_k
//!
//! Converges to a point such that `y = z = argmin f + g`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Douglas-Rachford splitting.
pub fn douglas_rachford<F, G>(
    x0: &[f64],
    prox_f: F,
    prox_g: G,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>>
where
    F: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
    G: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if gamma <= 0.0 || !gamma.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "DR gamma must be > 0, got {gamma}"
        )));
    }
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let n = x0.len();
    let mut x = x0.to_vec();
    for it in 0..max_iter {
        let y = prox_f(&x, gamma)?;
        if y.len() != n {
            return Err(CvxError::DimensionMismatch { a: y.len(), b: n });
        }
        let two_y_minus_x: Vec<f64> = y
            .iter()
            .zip(x.iter())
            .map(|(yi, xi)| 2.0 * yi - xi)
            .collect();
        let z = prox_g(&two_y_minus_x, gamma)?;
        if z.len() != n {
            return Err(CvxError::DimensionMismatch { a: z.len(), b: n });
        }
        let mut x_new = vec![0.0_f64; n];
        for i in 0..n {
            x_new[i] = x[i] + z[i] - y[i];
        }
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        x = x_new;
        if d_nrm < tol {
            return prox_f(&x, gamma);
        }
        let _ = it;
    }
    prox_f(&x, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l1::prox_l1;
    use crate::prox_ops::l2::prox_l2;

    #[test]
    fn dr_simple_minimisation() {
        // min ||x - b||²/2 + (γ/2) ||x||²  - solved via DR splitting with each piece.
        let b = vec![4.0_f64, 0.0];
        let f = |v: &[f64], g: f64| -> CvxResult<Vec<f64>> {
            // prox_{g · (1/2)||·-b||²}: argmin 0.5 g ||x-b||² + 0.5 ||x-v||²
            //                          = (g b + v) / (1 + g).
            Ok(v.iter()
                .zip(b.iter())
                .map(|(vi, bi)| (g * bi + vi) / (1.0 + g))
                .collect())
        };
        // g(x) = γ_g / 2 ||x||², prox = v / (1 + γ).
        let pg = |v: &[f64], g: f64| -> CvxResult<Vec<f64>> { prox_l2(v, g) };
        let x = douglas_rachford(&[0.0, 0.0], &f, &pg, 1.0, 200, 1.0e-10).expect("ok");
        // Joint min: min 0.5 ||x-b||² + 0.5 ||x||² → x = b/2 = [2, 0].
        assert!((x[0] - 2.0).abs() < 1.0e-5);
        assert!(x[1].abs() < 1.0e-5);
    }

    #[test]
    fn dr_lasso_smoke() {
        let b = vec![3.0_f64];
        let f = |v: &[f64], g: f64| -> CvxResult<Vec<f64>> {
            Ok(v.iter()
                .zip(b.iter())
                .map(|(vi, bi)| (g * bi + vi) / (1.0 + g))
                .collect())
        };
        let pg = |v: &[f64], g: f64| -> CvxResult<Vec<f64>> { prox_l1(v, g) };
        let x = douglas_rachford(&[0.0], &f, &pg, 0.5, 500, 1.0e-10).expect("ok");
        // Joint min: 0.5 (x-3)^2 + |x|. Subgradient at 0: -3 + sign(0)·[-1,1]; so 0 ∈ [-4, -2] FALSE.
        // Actually subgradient: x - 3 + sign(x). Roots: x>0 → x-3+1=0 → x=2 (valid).
        assert!((x[0] - 2.0).abs() < 1.0e-4);
    }
}
