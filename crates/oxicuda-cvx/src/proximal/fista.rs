//! FISTA: Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle 2009).
//!
//! Solves `min_x f(x) + g(x)` where `f` is convex smooth, `g` has a closed-form prox.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// FISTA with constant step `step` ≤ 1/L.  `prox_g(v, t)` computes `prox_{tg}(v)`.
pub fn fista<F, G, P>(
    x0: &[f64],
    f: F,
    grad_f: G,
    prox_g: P,
    mut step: f64,
    max_iter: usize,
    tol: f64,
    backtrack: bool,
) -> CvxResult<Vec<f64>>
where
    F: Fn(&[f64]) -> CvxResult<f64>,
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    P: Fn(&[f64], f64) -> CvxResult<Vec<f64>>,
{
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if step <= 0.0 || !step.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "fista step must be > 0, got {step}"
        )));
    }
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut x_prev = x0.to_vec();
    let mut y = x0.to_vec();
    let mut t = 1.0_f64;
    for it in 0..max_iter {
        let fy = f(&y)?;
        let gy = grad_f(&y)?;
        if gy.len() != n {
            return Err(CvxError::DimensionMismatch { a: gy.len(), b: n });
        }
        let mut s = step;
        let mut x_new: Vec<f64>;
        loop {
            let v: Vec<f64> = y
                .iter()
                .zip(gy.iter())
                .map(|(yi, gi)| yi - s * gi)
                .collect();
            x_new = prox_g(&v, s)?;
            if x_new.len() != n {
                return Err(CvxError::DimensionMismatch {
                    a: x_new.len(),
                    b: n,
                });
            }
            if !backtrack {
                break;
            }
            // Majorisation: f(x_new) ≤ f(y) + g·(x_new - y) + ||x_new - y||² / 2s.
            let f_new = f(&x_new)?;
            let mut dot_g = 0.0_f64;
            let mut sq = 0.0_f64;
            for i in 0..n {
                let d = x_new[i] - y[i];
                dot_g += gy[i] * d;
                sq += d * d;
            }
            let majorant = fy + dot_g + sq / (2.0 * s);
            if f_new <= majorant + 1.0e-12 {
                step = s;
                break;
            }
            s *= 0.5;
            if s < 1.0e-300 {
                return Err(CvxError::LineSearchFailed("fista: step underflowed".into()));
            }
        }
        let t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        let beta = (t - 1.0) / t_new;
        let mut y_new = vec![0.0_f64; n];
        for i in 0..n {
            y_new[i] = x_new[i] + beta * (x_new[i] - x_prev[i]);
        }
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        x_prev = x.clone();
        x = x_new;
        y = y_new;
        t = t_new;
        if d_nrm < tol {
            return Ok(x);
        }
        let _ = it;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l1::prox_l1;

    #[test]
    fn fista_lasso_simple() {
        // min 0.5 ||x - b||² + λ ||x||_1
        // Optimum: prox_{λ||·||_1}(b) = soft_threshold(b, λ).
        let b = vec![3.0_f64, -2.0, 0.5];
        let f = |x: &[f64]| -> CvxResult<f64> {
            Ok(x.iter()
                .zip(b.iter())
                .map(|(xi, bi)| 0.5 * (xi - bi).powi(2))
                .sum())
        };
        let g = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter().zip(b.iter()).map(|(xi, bi)| xi - bi).collect())
        };
        let p = |y: &[f64], s: f64| -> CvxResult<Vec<f64>> { prox_l1(y, s) };
        let x = fista(&[0.0, 0.0, 0.0], &f, &g, &p, 1.0, 1000, 1.0e-12, false).expect("ok");
        // Expected: [2, -1, 0] (soft-threshold by lambda=1 inside prox).
        assert!((x[0] - 2.0).abs() < 1.0e-5);
        assert!((x[1] + 1.0).abs() < 1.0e-5);
        assert!(x[2].abs() < 1.0e-5);
    }
}
