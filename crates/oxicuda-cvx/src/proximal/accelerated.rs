//! Accelerated proximal-gradient (Nesterov 1983 / FISTA variant).
//!
//! Thin wrapper renaming for API discoverability.

use crate::error::CvxResult;
use crate::proximal::fista::fista;

/// Accelerated proximal-gradient.  Identical to FISTA.
pub fn accelerated_prox_gradient<F, G, P>(
    x0: &[f64],
    f: F,
    grad_f: G,
    prox_g: P,
    step: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    P: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    fista(x0, f, grad_f, prox_g, step, max_iter, tol, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CvxResult;
    use crate::prox_ops::l1::prox_l1;

    #[test]
    fn accel_prox_lasso() {
        let b = vec![5.0_f64];
        let f = |x: &[f64]| -> CvxResult<f64> { Ok(0.5 * (x[0] - b[0]).powi(2)) };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![x[0] - b[0]]) };
        let p = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, s) };
        let x = accelerated_prox_gradient(&[0.0], &f, &g, &p, 1.0, 500, 1.0e-10).expect("ok");
        assert!((x[0] - 4.0).abs() < 1.0e-6);
    }
}
