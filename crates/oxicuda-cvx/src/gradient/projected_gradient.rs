//! Projected gradient descent: x_{k+1} = Π_C( x_k − α ∇f(x_k) ).

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;

/// Generic projected gradient method.
///
/// `grad_f` returns the gradient at the current point; `project` projects onto the feasible set.
pub fn projected_gradient<G, P>(
    x0: &[f64],
    grad_f: G,
    project: P,
    step: f64,
    max_iter: usize,
    tol: f64,
) -> CvxResult<Vec<f64>>
where
    G: Fn(&[f64]) -> CvxResult<Vec<f64>>,
    P: Fn(&[f64]) -> CvxResult<Vec<f64>>,
{
    if step <= 0.0 || !step.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "projected gradient step must be > 0, got {step}"
        )));
    }
    if x0.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    let mut x = x0.to_vec();
    for it in 0..max_iter {
        let g = grad_f(&x)?;
        if g.len() != x.len() {
            return Err(CvxError::DimensionMismatch {
                a: g.len(),
                b: x.len(),
            });
        }
        let y: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .map(|(xi, gi)| xi - step * gi)
            .collect();
        let x_new = project(&y)?;
        if x_new.len() != x.len() {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: x.len(),
            });
        }
        let diff: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_norm = norm2(&diff);
        x = x_new;
        if d_norm < tol {
            return Ok(x);
        }
        let _ = it;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::project_box;

    #[test]
    fn projected_gd_box_quadratic() {
        // min ||x − [2, -3]||^2 s.t. x in [-1, 1]^2.  Optimum: [1, -1].
        let target = vec![2.0_f64, -3.0];
        let grad = |x: &[f64]| -> CvxResult<Vec<f64>> {
            Ok(x.iter()
                .zip(target.iter())
                .map(|(xi, ti)| 2.0 * (xi - ti))
                .collect())
        };
        let proj = |y: &[f64]| -> CvxResult<Vec<f64>> { project_box(y, -1.0, 1.0) };
        let x = projected_gradient(&[0.0, 0.0], grad, proj, 0.1, 500, 1.0e-10).expect("ok");
        assert!((x[0] - 1.0).abs() < 1.0e-6);
        assert!((x[1] + 1.0).abs() < 1.0e-6);
    }
}
